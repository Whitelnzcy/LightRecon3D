"""Apply the learned intra-fragment partitioner to cached support records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_intra_fragment_splitter import (
    average_linkage_components,
    average_linkage_tree,
    connected_components,
    dense_pair_affinity,
)
from train_intra_fragment_splitter import (
    IntraFragmentSplitter,
    fit_fragment_plane,
    local_neighbors,
)


def inference_case(
    points: np.ndarray,
    colors: np.ndarray,
    pixels: np.ndarray,
    views: np.ndarray,
    *,
    pixel_size: float,
) -> dict[str, np.ndarray]:
    center = points.mean(axis=0)
    scale = max(
        float(np.median(np.linalg.norm(points - center, axis=1))),
        1e-6,
    )
    xyz = (points - center) / scale
    normal, offset = fit_fragment_plane(points)
    residual = (points @ normal + offset) / scale
    pixel_features = pixels / max(float(pixel_size), 1.0) * 2.0 - 1.0
    view_scale = max(float(views.max()), 1.0)
    features = np.concatenate(
        (
            xyz,
            colors,
            pixel_features,
            views.astype(np.float32)[:, None] / view_scale,
            residual[:, None],
            np.broadcast_to(normal, (len(points), 3)),
        ),
        axis=1,
    )
    return {
        "features": features.astype(np.float32),
        "xyz": xyz.astype(np.float32),
        "pixels": pixel_features.astype(np.float32),
        "views": views.astype(np.int64),
    }


def merge_small_anchor_clusters(
    labels: np.ndarray,
    affinity: np.ndarray,
    minimum_points: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32).copy()
    unique, counts = np.unique(labels, return_counts=True)
    large = unique[counts >= int(minimum_points)]
    if len(large) < 2:
        return np.zeros_like(labels)
    for cluster in unique[counts < int(minimum_points)]:
        members = labels == cluster
        scores = [
            float(affinity[np.ix_(members, labels == candidate)].mean())
            for candidate in large
        ]
        labels[members] = int(large[int(np.argmax(scores))])
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int32)


def nearest_anchor_labels(
    pixels: np.ndarray,
    anchor_pixels: np.ndarray,
    anchor_labels: np.ndarray,
    *,
    chunk_size: int = 8192,
) -> np.ndarray:
    labels, _ = nearest_anchor_assignment(
        pixels,
        anchor_pixels,
        anchor_labels,
        chunk_size=chunk_size,
    )
    return labels


def nearest_anchor_assignment(
    pixels: np.ndarray,
    anchor_pixels: np.ndarray,
    anchor_labels: np.ndarray,
    *,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.empty((len(pixels),), dtype=np.int32)
    distances = np.empty((len(pixels),), dtype=np.float32)
    for start in range(0, len(pixels), int(chunk_size)):
        chunk = pixels[start : start + int(chunk_size)]
        squared = ((chunk[:, None, :] - anchor_pixels[None, :, :]) ** 2).sum(
            axis=2
        )
        nearest = np.argmin(squared, axis=1)
        output[start : start + len(chunk)] = anchor_labels[nearest]
        distances[start : start + len(chunk)] = np.sqrt(
            squared[np.arange(len(chunk)), nearest]
        )
    return output, distances


def component_statistics(labels: np.ndarray) -> dict:
    _, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    counts = np.sort(counts.astype(np.int64))[::-1]
    point_count = int(counts.sum())
    ratios = (
        counts.astype(np.float64) / point_count
        if point_count
        else np.zeros((0,), dtype=np.float64)
    )
    return {
        "component_count": int(len(counts)),
        "component_sizes": [int(value) for value in counts],
        "largest_component_points": int(counts[0]) if len(counts) else 0,
        "largest_component_ratio": float(ratios[0]) if len(ratios) else 0.0,
        "second_largest_component_ratio": (
            float(ratios[1]) if len(ratios) > 1 else 0.0
        ),
        "smallest_component_points": int(counts[-1]) if len(counts) else 0,
        "tiny_component_count_lt16": int((counts < 16).sum()),
        "tiny_component_count_lt32": int((counts < 32).sum()),
        "tiny_component_count_lt64": int((counts < 64).sum()),
    }


def refit_planes(points: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    plane_ids = np.unique(np.asarray(labels)[np.asarray(labels) >= 0])
    normals = []
    offsets = []
    counts = []
    for plane_id in plane_ids:
        selected = points[labels == plane_id]
        normal, offset = fit_fragment_plane(selected)
        normals.append(normal)
        offsets.append(offset)
        counts.append(len(selected))
    return {
        "plane_ids": plane_ids.astype(np.int32),
        "plane_normals": np.asarray(normals, dtype=np.float32),
        "plane_offsets": np.asarray(offsets, dtype=np.float32),
        "plane_inlier_counts": np.asarray(counts, dtype=np.int32),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def split_support(
    model: IntraFragmentSplitter,
    input_path: Path,
    output_path: Path,
    *,
    device: torch.device,
    affinity_threshold: float,
    max_anchor_points: int,
    minimum_fragment_points: int,
    minimum_anchor_cluster_points: int,
    split_gate_threshold: float,
    partition_method: str,
    boundary_threshold: float,
    neighbor_count: int,
    pixel_size: float,
) -> dict:
    with np.load(input_path, allow_pickle=True) as source:
        arrays = {key: source[key] for key in source.files}
    points = arrays["points"].astype(np.float32)
    colors = arrays.get("original_colors", arrays["colors"]).astype(np.float32)
    if colors.size and float(colors.max()) > 1.5:
        colors /= 255.0
    pixels = arrays["pointmap_pixel_xy"].astype(np.float32)
    views = arrays["alignment_view_indices"].astype(np.int64)
    original_ids = arrays["point_plane_ids"].astype(np.int64)
    new_ids = np.full_like(original_ids, -1, dtype=np.int32)
    candidate_group_indices = np.full_like(original_ids, -1, dtype=np.int32)
    candidate_component_labels = np.full_like(
        original_ids, -1, dtype=np.int32
    )
    next_id = 0
    split_count = 0
    rows = []
    for fragment_id in np.unique(original_ids[original_ids >= 0]):
        fragment_indices = np.flatnonzero(original_ids == fragment_id)
        for view in np.unique(views[fragment_indices]):
            group_index = len(rows)
            indices = fragment_indices[views[fragment_indices] == view]
            selected_labels = np.zeros((len(indices),), dtype=np.int32)
            partition_labels = np.zeros((len(indices),), dtype=np.int32)
            anchor_component_count = 1
            eligible = len(indices) >= int(minimum_fragment_points)
            split_probability = None
            gate_decision = False
            anchor_distance = None
            if eligible:
                if len(indices) <= int(max_anchor_points):
                    anchor_local = np.arange(len(indices), dtype=np.int64)
                else:
                    anchor_local = np.linspace(
                        0,
                        len(indices) - 1,
                        num=int(max_anchor_points),
                    ).round().astype(np.int64)
                anchor_indices = indices[anchor_local]
                case_np = inference_case(
                    points[anchor_indices],
                    colors[anchor_indices],
                    pixels[anchor_indices],
                    views[anchor_indices],
                    pixel_size=pixel_size,
                )
                case = {
                    key: torch.from_numpy(value).to(device)
                    for key, value in case_np.items()
                }
                neighbors = local_neighbors(
                    case["xyz"],
                    neighbor_count,
                    case["pixels"],
                    case["views"],
                )
                embedding, boundary_logits = model.encode(
                    case["features"], neighbors
                )
                edge_logits = model.edge_logits(
                    embedding,
                    case["xyz"],
                    case["pixels"],
                    case["views"],
                    neighbors,
                )
                split_logit = model.fragment_split_logit(
                    embedding, boundary_logits, edge_logits
                )
                split_probability = (
                    float(torch.sigmoid(split_logit).cpu())
                    if split_logit is not None
                    else 1.0
                )
                gate_decision = split_probability >= float(
                    split_gate_threshold
                )
                if partition_method == "local_connected_components":
                    anchor_labels = connected_components(
                        len(anchor_indices),
                        neighbors.cpu().numpy(),
                        torch.sigmoid(edge_logits).cpu().numpy(),
                        torch.sigmoid(boundary_logits).cpu().numpy(),
                        affinity_threshold=affinity_threshold,
                        boundary_threshold=boundary_threshold,
                    )
                elif partition_method == "dense_average_linkage":
                    affinity = dense_pair_affinity(
                        model,
                        embedding,
                        case["xyz"],
                        case["pixels"],
                        case["views"],
                    )
                    anchor_labels = average_linkage_components(
                        average_linkage_tree(affinity),
                        len(anchor_indices),
                        affinity_threshold=affinity_threshold,
                    )
                    anchor_labels = merge_small_anchor_clusters(
                        anchor_labels,
                        affinity,
                        minimum_anchor_cluster_points,
                    )
                else:
                    raise ValueError(
                        f"Unsupported partition method: {partition_method}"
                    )
                anchor_component_count = len(np.unique(anchor_labels))
                partition_labels, anchor_distance = nearest_anchor_assignment(
                    pixels[indices],
                    pixels[anchor_indices],
                    anchor_labels,
                )
                if gate_decision and anchor_component_count > 1:
                    selected_labels = partition_labels
                    split_count += 1
            candidate_group_indices[indices] = group_index
            candidate_component_labels[indices] = partition_labels
            candidate_stats = component_statistics(partition_labels)
            selected_stats = component_statistics(selected_labels)
            if not eligible:
                skip_reason = "below_minimum_points"
            elif not gate_decision:
                skip_reason = "gate_rejected"
            elif candidate_stats["component_count"] == 1:
                skip_reason = "partition_single_component"
            else:
                skip_reason = "split_applied"
            local_ids = np.unique(selected_labels)
            for local_id in local_ids:
                new_ids[indices[selected_labels == local_id]] = next_id
                next_id += 1
            rows.append(
                {
                    "group_index": int(group_index),
                    "original_fragment_id": int(fragment_id),
                    "view_index": int(view),
                    "point_count": int(len(indices)),
                    "eligible": bool(eligible),
                    "gate_probability": split_probability,
                    "gate_decision": bool(gate_decision),
                    "skip_reason": skip_reason,
                    "anchor_point_count": (
                        int(len(anchor_indices)) if eligible else 0
                    ),
                    "anchor_component_count": int(anchor_component_count),
                    "candidate_component_count": int(
                        candidate_stats["component_count"]
                    ),
                    "candidate_component_sizes": candidate_stats[
                        "component_sizes"
                    ],
                    "candidate_largest_component_points": candidate_stats[
                        "largest_component_points"
                    ],
                    "candidate_largest_component_ratio": candidate_stats[
                        "largest_component_ratio"
                    ],
                    "candidate_second_largest_component_ratio": candidate_stats[
                        "second_largest_component_ratio"
                    ],
                    "candidate_smallest_component_points": candidate_stats[
                        "smallest_component_points"
                    ],
                    "candidate_tiny_component_count_lt16": candidate_stats[
                        "tiny_component_count_lt16"
                    ],
                    "candidate_tiny_component_count_lt32": candidate_stats[
                        "tiny_component_count_lt32"
                    ],
                    "candidate_tiny_component_count_lt64": candidate_stats[
                        "tiny_component_count_lt64"
                    ],
                    "component_count": int(selected_stats["component_count"]),
                    "component_sizes": selected_stats["component_sizes"],
                    "largest_component_points": selected_stats[
                        "largest_component_points"
                    ],
                    "largest_component_ratio": selected_stats[
                        "largest_component_ratio"
                    ],
                    "second_largest_component_ratio": selected_stats[
                        "second_largest_component_ratio"
                    ],
                    "smallest_component_points": selected_stats[
                        "smallest_component_points"
                    ],
                    "tiny_component_count_lt16": selected_stats[
                        "tiny_component_count_lt16"
                    ],
                    "tiny_component_count_lt32": selected_stats[
                        "tiny_component_count_lt32"
                    ],
                    "tiny_component_count_lt64": selected_stats[
                        "tiny_component_count_lt64"
                    ],
                    "nearest_anchor_distance_p50": (
                        float(np.quantile(anchor_distance, 0.50))
                        if anchor_distance is not None
                        else None
                    ),
                    "nearest_anchor_distance_p95": (
                        float(np.quantile(anchor_distance, 0.95))
                        if anchor_distance is not None
                        else None
                    ),
                    "nearest_anchor_distance_max": (
                        float(anchor_distance.max())
                        if anchor_distance is not None
                        else None
                    ),
                }
            )
    expected_assigned = original_ids >= 0
    if np.any(new_ids[expected_assigned] < 0):
        raise RuntimeError("Some assigned support records were not processed")
    if np.any(new_ids[~expected_assigned] >= 0):
        raise RuntimeError("Unassigned input support was converted to a plane")
    if np.any(candidate_group_indices[expected_assigned] < 0) or np.any(
        candidate_component_labels[expected_assigned] < 0
    ):
        raise RuntimeError("Some support records lack partition candidates")
    arrays["original_point_plane_ids_before_split"] = original_ids.astype(np.int32)
    arrays["point_plane_ids"] = new_ids
    arrays["intra_fragment_candidate_group_indices"] = candidate_group_indices
    arrays["intra_fragment_candidate_component_labels"] = (
        candidate_component_labels
    )
    arrays.update(refit_planes(points, new_ids))
    arrays["intra_fragment_split_metadata_json"] = np.asarray(
        json.dumps(
            {
                "schema_version": 2,
                "affinity_threshold": affinity_threshold,
                "max_anchor_points": max_anchor_points,
                "minimum_fragment_points": minimum_fragment_points,
                "minimum_anchor_cluster_points": minimum_anchor_cluster_points,
                "split_gate_threshold": split_gate_threshold,
                "partition_method": partition_method,
                "boundary_threshold": boundary_threshold,
                "groups": rows,
            }
        ),
        dtype=object,
    )
    np.savez_compressed(output_path, **arrays)
    input_label_count = int(len(np.unique(original_ids)))
    input_fragment_count = int(
        len(np.unique(original_ids[original_ids >= 0]))
    )
    group_count = int(len(rows))
    output_fragment_count = int(len(np.unique(new_ids[new_ids >= 0])))
    candidate_output_fragment_count = int(
        sum(row["candidate_component_count"] for row in rows)
    )
    return {
        "input_fragment_count": input_fragment_count,
        "input_label_count_including_negative": input_label_count,
        "negative_input_point_count": int((original_ids < 0).sum()),
        "group_count": group_count,
        "view_group_fragment_increase": group_count - input_fragment_count,
        "output_fragment_count": output_fragment_count,
        "partition_fragment_increase": output_fragment_count - group_count,
        "candidate_output_fragment_count": candidate_output_fragment_count,
        "candidate_partition_fragment_increase": (
            candidate_output_fragment_count - group_count
        ),
        "eligible_group_count": int(sum(row["eligible"] for row in rows)),
        "gate_accepted_group_count": int(
            sum(row["gate_decision"] for row in rows)
        ),
        "gate_rejected_group_count": int(
            sum(row["skip_reason"] == "gate_rejected" for row in rows)
        ),
        "split_group_count": int(split_count),
        "output_npz": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply learned support splitting and emit a derived batch."
    )
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--affinity_threshold", type=float, default=0.175)
    parser.add_argument("--max_anchor_points", type=int, default=512)
    parser.add_argument("--minimum_fragment_points", type=int, default=128)
    parser.add_argument("--minimum_anchor_cluster_points", type=int, default=16)
    parser.add_argument("--split_gate_threshold", type=float, default=0.5)
    parser.add_argument(
        "--partition_method",
        choices=("local_connected_components", "dense_average_linkage"),
        default="dense_average_linkage",
    )
    parser.add_argument("--boundary_threshold", type=float, default=1.01)
    parser.add_argument("--neighbor_count", type=int, default=8)
    parser.add_argument("--pixel_size", type=float, default=512.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    support_dir = output_dir / "support"
    support_dir.mkdir(parents=True)
    manifest = json.loads(
        Path(args.manifest_json).resolve().read_text(encoding="utf-8")
    )
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    checkpoint = torch.load(
        Path(args.checkpoint).resolve(), map_location=device, weights_only=False
    )
    model = IntraFragmentSplitter(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    selected = [
        scene for scene in manifest["scenes"] if scene["split"] == args.split
    ]
    batch_path = Path(manifest["config"]["batch_json"]).resolve()
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    derived_items = json.loads(json.dumps(batch["items"]))
    derived_item_by_id = {item["id"]: item for item in derived_items}
    scene_rows = []
    for scene in selected:
        output_path = support_dir / f"{scene['scene']}_split_support.npz"
        row = split_support(
            model,
            Path(scene["support_npz"]),
            output_path,
            device=device,
            affinity_threshold=args.affinity_threshold,
            max_anchor_points=args.max_anchor_points,
            minimum_fragment_points=args.minimum_fragment_points,
            minimum_anchor_cluster_points=args.minimum_anchor_cluster_points,
            split_gate_threshold=args.split_gate_threshold,
            partition_method=args.partition_method,
            boundary_threshold=args.boundary_threshold,
            neighbor_count=args.neighbor_count,
            pixel_size=args.pixel_size,
        )
        row["scene"] = scene["scene"]
        scene_rows.append(row)
        item = derived_item_by_id[scene["scene"]]
        item["artifacts"]["direct_support_records"] = {
            "path": str(output_path),
            "exists": True,
            "bytes": output_path.stat().st_size,
            "sha256": file_sha256(output_path),
            "expected_sha256": "",
            "checksum_matches": None,
        }
        print(json.dumps(row), flush=True)
    batch["items"] = derived_items
    batch["batch_name"] = f"{batch.get('batch_name', 'batch')}_split_{args.split}"
    batch["derived_from"] = str(batch_path)
    derived_batch = output_dir / "derived_batch_execution.json"
    derived_batch.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    summary = {
        "schema_version": 2,
        "task": "prediction_only_intra_fragment_support_split",
        "config": vars(args),
        "scenes": scene_rows,
        "outputs": {
            "derived_batch_json": str(derived_batch),
            "support_dir": str(support_dir),
        },
        "limitations": [
            "Large fragments are clustered on deterministic anchors.",
            "Non-anchor points inherit the nearest anchor label in source-pixel space.",
            "This output is an intermediate support record; stale quality JSON is preserved only for provenance.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
