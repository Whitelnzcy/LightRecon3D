import argparse
import json
import math
from pathlib import Path

import numpy as np

from export_stage2_geometry_refit_editables import plane_rows, write_ascii_ply, write_html, write_params
from make_full_pointcloud_edit_comparison import PLANE_COLORS
from train_stage2_region_merge_net import fit_plane_np


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
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


def scalar_string(raw, key, default=""):
    if key not in raw:
        return default
    value = raw[key]
    if np.ndim(value) == 0:
        return str(value.item())
    if len(value) == 0:
        return default
    return str(value[0])


def group_key(raw, fallback, mode):
    scene = scalar_string(raw, "scene_name", fallback)
    pair_group = scalar_string(raw, "pair_group", "")
    rgb_path1 = scalar_string(raw, "rgb_path1", "")
    if mode == "scene":
        return scene
    if mode == "pair_group":
        return f"{scene}|{pair_group}" if pair_group else scene
    if mode == "reference_view":
        return f"{scene}|{rgb_path1}" if rgb_path1 else fallback
    return fallback


def plane_angle_deg(normal_a, normal_b):
    dot = float(abs(np.dot(normal_a, normal_b)))
    dot = min(max(dot, 0.0), 1.0)
    return float(np.degrees(np.arccos(dot)))


def mutual_residual(plane_a, plane_b):
    points_a = plane_a["points"]
    points_b = plane_b["points"]
    normal_a = plane_a["normal"]
    normal_b = plane_b["normal"]
    offset_a = plane_a["offset"]
    offset_b = plane_b["offset"]
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf")
    residual_ab = np.abs(points_a @ normal_b + offset_b)
    residual_ba = np.abs(points_b @ normal_a + offset_a)
    return float(0.5 * (np.mean(residual_ab) + np.mean(residual_ba)))


def load_scene_inputs(files, min_points):
    scene_points = []
    scene_colors = []
    scene_local_assignment = []
    planes = []
    point_offset = 0
    for file_index, path in enumerate(files):
        raw = np.load(path)
        points = raw["points"].astype(np.float32)
        colors_key = "original_colors" if "original_colors" in raw else "colors"
        colors = raw[colors_key].astype(np.uint8)
        assignment = raw["point_plane_ids"].astype(np.int32)
        normals = raw["plane_normals"].astype(np.float32)
        offsets = raw["plane_offsets"].astype(np.float32)
        scene_points.append(points)
        scene_colors.append(colors)
        local_assignment = np.full_like(assignment, -1, dtype=np.int32)
        for plane_id in sorted(int(x) for x in np.unique(assignment) if int(x) >= 0):
            mask = assignment == plane_id
            count = int(mask.sum())
            if count < min_points or plane_id >= len(normals):
                continue
            local_index = len(planes)
            local_assignment[mask] = local_index
            pts = points[mask]
            normal = normals[plane_id].astype(np.float32)
            normal /= max(float(np.linalg.norm(normal)), 1e-8)
            planes.append(
                {
                    "local_index": local_index,
                    "file_index": file_index,
                    "source_file": str(path),
                    "source_plane_id": int(plane_id),
                    "global_point_indices": np.nonzero(mask)[0].astype(np.int64) + point_offset,
                    "points": pts,
                    "normal": normal,
                    "offset": float(offsets[plane_id]),
                    "count": count,
                    "centroid": pts.mean(axis=0),
                }
            )
        scene_local_assignment.append(local_assignment)
        point_offset += len(points)
    if not scene_points:
        return None
    return {
        "points": np.concatenate(scene_points, axis=0),
        "colors": np.concatenate(scene_colors, axis=0),
        "local_assignments": scene_local_assignment,
        "planes": planes,
    }


def should_merge(plane_a, plane_b, args):
    angle = plane_angle_deg(plane_a["normal"], plane_b["normal"])
    offset = abs(abs(float(plane_a["offset"])) - abs(float(plane_b["offset"])))
    residual = mutual_residual(plane_a, plane_b)
    centroid_distance = float(np.linalg.norm(plane_a["centroid"] - plane_b["centroid"]))
    return (
        angle <= args.max_angle_deg
        and offset <= args.max_offset
        and residual <= args.max_mutual_residual
        and centroid_distance <= args.max_centroid_distance
    ), {
        "angle_deg": angle,
        "offset_abs": offset,
        "mutual_residual": residual,
        "centroid_distance": centroid_distance,
    }


def fuse_scene(scene_key, files, output_dir, args):
    loaded = load_scene_inputs(files, args.min_points)
    if loaded is None:
        return None
    points = loaded["points"]
    colors = loaded["colors"]
    planes = loaded["planes"]
    uf = UnionFind(len(planes))
    merge_records = []
    for i, plane_a in enumerate(planes):
        for j in range(i + 1, len(planes)):
            plane_b = planes[j]
            ok, stats = should_merge(plane_a, plane_b, args)
            if ok:
                uf.union(i, j)
            merge_records.append(
                {
                    "plane_a": i,
                    "plane_b": j,
                    "source_a": [plane_a["source_file"], plane_a["source_plane_id"]],
                    "source_b": [plane_b["source_file"], plane_b["source_plane_id"]],
                    "merged": bool(ok),
                    **stats,
                }
            )

    roots = {}
    for i in range(len(planes)):
        root = uf.find(i)
        roots.setdefault(root, len(roots))

    assignment = np.full((len(points),), -1, dtype=np.int32)
    source_groups = [[] for _ in range(len(roots))]
    for i, plane in enumerate(planes):
        global_id = roots[uf.find(i)]
        assignment[plane["global_point_indices"]] = global_id
        source_groups[global_id].append(
            {
                "source_file": plane["source_file"],
                "source_plane_id": int(plane["source_plane_id"]),
                "points": int(plane["count"]),
            }
        )

    normals = []
    offsets = []
    counts = []
    for global_id in range(len(source_groups)):
        mask = assignment == global_id
        normal, offset, _ = fit_plane_np(points[mask])
        normals.append(normal)
        offsets.append(float(offset))
        counts.append(int(mask.sum()))
    normals = np.stack(normals, axis=0).astype(np.float32) if normals else np.zeros((0, 3), dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int32)
    planes_out = plane_rows(normals, offsets, assignment)

    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in scene_key)[-120:]
    npz_path = output_dir / f"{safe_name}_stage3_scene_fusion_full_pointcloud_editable_planes_data.npz"
    html_path = output_dir / f"{safe_name}_stage3_scene_fusion_edit.html"
    ply_path = output_dir / f"{safe_name}_stage3_scene_fusion.ply"
    json_path = output_dir / f"{safe_name}_stage3_scene_fusion_plane_params.json"
    txt_path = output_dir / f"{safe_name}_stage3_scene_fusion_plane_params.txt"

    np.savez_compressed(
        npz_path,
        points=points.astype(np.float32),
        colors=colors.astype(np.uint8),
        original_colors=colors.astype(np.uint8),
        point_plane_ids=assignment.astype(np.int32),
        plane_ids=np.arange(len(normals), dtype=np.int32),
        plane_normals=normals,
        plane_offsets=offsets,
        plane_inlier_counts=counts,
        scene_key=np.asarray(scene_key),
        source_groups_json=np.asarray(json.dumps(source_groups), dtype=object),
        merge_records_json=np.asarray(json.dumps(merge_records), dtype=object),
    )

    display_colors = colors.copy()
    for plane_id in range(len(planes_out)):
        display_colors[assignment == plane_id] = np.asarray(
            PLANE_COLORS[plane_id % len(PLANE_COLORS)],
            dtype=np.uint8,
        )
    write_ascii_ply(ply_path, points, display_colors)
    write_html(
        html_path,
        points,
        colors,
        assignment,
        planes_out,
        args.edit_delta,
        args.edit_plane,
        args.max_display_points,
        npz_path,
    )
    write_params(json_path, txt_path, planes_out, np.arange(len(planes_out), dtype=np.int32), npz_path)
    return {
        "scene_key": scene_key,
        "input_files": [str(path) for path in files],
        "local_planes": int(len(planes)),
        "global_planes": int(len(planes_out)),
        "merged_pairs": int(sum(1 for row in merge_records if row["merged"])),
        "points": int(len(points)),
        "npz": str(npz_path),
        "html": str(html_path),
        "ply": str(ply_path),
        "json": str(json_path),
        "txt": str(txt_path),
    }


def main():
    parser = argparse.ArgumentParser("Fuse local Stage2 plane primitives into scene-level editable planes")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*_full_pointcloud_editable_planes_data.npz")
    parser.add_argument("--group_by", default="reference_view", choices=("sample", "scene", "pair_group", "reference_view"))
    parser.add_argument("--min_group_size", type=int, default=1)
    parser.add_argument("--min_points", type=int, default=64)
    parser.add_argument("--max_angle_deg", type=float, default=8.0)
    parser.add_argument("--max_offset", type=float, default=0.06)
    parser.add_argument("--max_mutual_residual", type=float, default=0.06)
    parser.add_argument("--max_centroid_distance", type=float, default=2.5)
    parser.add_argument("--edit_plane", default="largest")
    parser.add_argument("--edit_delta", type=float, default=0.25)
    parser.add_argument("--max_display_points", type=int, default=32000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No files matched {input_dir / args.pattern}")

    groups = {}
    for path in files:
        raw = np.load(path)
        key = group_key(raw, path.stem, args.group_by)
        groups.setdefault(key, []).append(path)

    rows = []
    for key, group_files in sorted(groups.items()):
        if len(group_files) < args.min_group_size:
            continue
        row = fuse_scene(key, group_files, output_dir, args)
        if row is not None:
            rows.append(row)

    manifest = output_dir / "stage3_scene_fusion_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "scenes": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
