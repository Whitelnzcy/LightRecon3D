import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_stage2_region_merge_net import RegionMergeMLP, fit_plane_np, load_regions, pair_features


STAGE2_SCHEMA_VERSION = 2
PER_POINT_METADATA_KEYS = (
    "pixel_xy",
    "pixel_xy1",
    "gt_point_plane_ids",
    "point_confidence",
    "point_margin",
    "line_prob",
    "support_source_view",
)
PASSTHROUGH_METADATA_KEYS = (
    "scene_name",
    "pair_group",
    "rgb_path1",
    "rgb_path2",
    "json_path1",
    "json_path2",
    "view_id1",
    "view_id2",
    "original_hw1",
    "original_hw2",
    "stage1_input_hw1",
    "stage1_input_hw2",
    "stage1_mask_hw1",
    "stage1_mask_hw2",
    "pixel_coordinate_space",
    "pixel_coordinate_order",
    "pixel_coordinate_range",
    "pixel_coordinate_view",
    "sample_idx",
    "pixel_xy",
    "pixel_xy1",
    "pixel_xy2",
    "support_source_view",
    "gt_point_plane_ids",
    "point_confidence",
    "point_margin",
    "line_prob",
)


class UnionFind:
    def __init__(self, values):
        self.parent = {int(value): int(value) for value in values}

    def find(self, value):
        value = int(value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def load_model(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = payload.get("args", {})
    input_dim = int(payload.get("input_dim", len(payload.get("feature_names", [])) or 14))
    model = RegionMergeMLP(
        input_dim,
        hidden_dim=int(args.get("hidden_dim", 128)),
        depth=int(args.get("depth", 3)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def safe_pair_gate(region_a, region_b, args):
    normal_a = region_a["normal"]
    normal_b = region_b["normal"]
    dot = float(abs(np.dot(normal_a, normal_b)))
    dot = min(max(dot, 0.0), 1.0)
    angle = float(np.degrees(np.arccos(dot)))
    offset = abs(abs(float(region_a["offset"])) - abs(float(region_b["offset"])))
    residual_ab = abs(float(region_a["centroid"] @ normal_b + region_b["offset"]))
    residual_ba = abs(float(region_b["centroid"] @ normal_a + region_a["offset"]))
    residual = 0.5 * (residual_ab + residual_ba)
    min_area = min(region_a["area_frac"], region_b["area_frac"])
    max_area = max(region_a["area_frac"], region_b["area_frac"])
    area_ratio = min_area / max(max_area, 1e-8)
    return (
        angle <= args.max_angle_deg
        and offset <= args.max_offset
        and residual <= args.max_mutual_residual
        and area_ratio >= args.min_area_ratio
    )


@torch.no_grad()
def predict_merge_pairs(model, regions, args, device):
    pairs = []
    if len(regions) < 2:
        return pairs
    features = []
    index_pairs = []
    for i, region_a in enumerate(regions):
        for j in range(i + 1, len(regions)):
            region_b = regions[j]
            if args.use_safety_gate and not safe_pair_gate(region_a, region_b, args):
                continue
            features.append(pair_features(region_a, region_b))
            index_pairs.append((i, j))
    if not features:
        return pairs
    x = torch.from_numpy(np.stack(features, axis=0)).float().to(device)
    probs = torch.sigmoid(model(x)).detach().cpu().numpy()
    for (i, j), prob in zip(index_pairs, probs):
        pairs.append(
            {
                "plane_a": int(regions[i]["plane_id"]),
                "plane_b": int(regions[j]["plane_id"]),
                "probability": float(prob),
                "merged": bool(prob >= args.threshold),
            }
        )
    return pairs


def remap_assignment(raw_assignment, regions, merge_pairs):
    plane_ids = [int(region["plane_id"]) for region in regions]
    uf = UnionFind(plane_ids)
    for pair in merge_pairs:
        if pair["merged"]:
            uf.union(pair["plane_a"], pair["plane_b"])

    roots = {}
    for plane_id in plane_ids:
        root = uf.find(plane_id)
        roots.setdefault(root, len(roots))

    output = np.full_like(raw_assignment, -1, dtype=np.int32)
    source_groups = [[] for _ in range(len(roots))]
    for plane_id in plane_ids:
        new_id = roots[uf.find(plane_id)]
        output[raw_assignment == plane_id] = int(new_id)
        source_groups[new_id].append(int(plane_id))
    return output, source_groups


def refit_planes(points, assignment):
    normals = []
    offsets = []
    counts = []
    for plane_id in sorted(int(x) for x in np.unique(assignment) if int(x) >= 0):
        mask = assignment == plane_id
        normal, offset, _ = fit_plane_np(points[mask])
        normals.append(normal)
        offsets.append(float(offset))
        counts.append(int(mask.sum()))
    if not normals:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.stack(normals, axis=0).astype(np.float32),
        np.asarray(offsets, dtype=np.float32),
        np.asarray(counts, dtype=np.int32),
    )


def validate_metadata_lengths(raw, point_count, input_npz):
    for key in PER_POINT_METADATA_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if value.ndim > 0 and len(value) not in (0, point_count):
            raise RuntimeError(
                f"Metadata length mismatch for {input_npz}: {key} has {len(value)} rows, expected {point_count}"
            )


def export_one(input_npz, output_dir, model, args, device):
    raw = np.load(input_npz)
    points, assignment, gt_assignment, regions = load_regions(input_npz, min_points=args.min_points)
    validate_metadata_lengths(raw, len(points), input_npz)
    merge_pairs = predict_merge_pairs(model, regions, args, device)
    merged_assignment, source_groups = remap_assignment(assignment, regions, merge_pairs)
    normals, offsets, counts = refit_planes(points, merged_assignment)

    stem = input_npz.name.replace("_full_pointcloud_editable_planes_data.npz", "")
    output_npz = output_dir / f"{stem}_learned_region_merge_full_pointcloud_editable_planes_data.npz"
    output_dir.mkdir(parents=True, exist_ok=True)

    colors_key = "original_colors" if "original_colors" in raw else "colors"
    payload = {
        "schema_version": np.asarray(STAGE2_SCHEMA_VERSION, dtype=np.int32),
        "source_schema_version": raw["schema_version"] if "schema_version" in raw else np.asarray(1, dtype=np.int32),
        "points": points.astype(np.float32),
        "colors": raw[colors_key].astype(np.uint8),
        "original_colors": raw[colors_key].astype(np.uint8),
        "point_plane_ids": merged_assignment.astype(np.int32),
        "plane_ids": np.arange(len(normals), dtype=np.int32),
        "plane_normals": normals.astype(np.float32),
        "plane_offsets": offsets.astype(np.float32),
        "plane_inlier_counts": counts.astype(np.int32),
        "active_planes": np.ones((len(normals),), dtype=bool),
        "source_plane_groups": np.asarray(
            [",".join(str(x) for x in group) for group in source_groups],
            dtype=object,
        ),
        "merge_pairs_json": np.asarray(json.dumps(merge_pairs), dtype=object),
    }
    for key in PASSTHROUGH_METADATA_KEYS:
        if key in raw:
            payload[key] = raw[key]
    np.savez_compressed(output_npz, **payload)

    return {
        "input": str(input_npz),
        "output": str(output_npz),
        "regions_before": len(regions),
        "regions_after": int(len(normals)),
        "merged_pairs": int(sum(1 for pair in merge_pairs if pair["merged"])),
        "candidate_pairs": int(len(merge_pairs)),
        "source_plane_groups": source_groups,
    }


def main():
    parser = argparse.ArgumentParser("Apply learned Stage2 region merge and refit plane equations")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*_full_pointcloud_editable_planes_data.npz")
    parser.add_argument("--min_points", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_safety_gate", action="store_true")
    parser.add_argument("--max_angle_deg", type=float, default=12.0)
    parser.add_argument("--max_offset", type=float, default=0.08)
    parser.add_argument("--max_mutual_residual", type=float, default=0.08)
    parser.add_argument("--min_area_ratio", type=float, default=0.01)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No files matched {input_dir / args.pattern}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    rows = [export_one(path, output_dir, model, args, device) for path in files]
    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "threshold": float(args.threshold),
        "use_safety_gate": bool(args.use_safety_gate),
        "files": rows,
    }
    manifest_path = output_dir / "learned_region_merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "files": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()


