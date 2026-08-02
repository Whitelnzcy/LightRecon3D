"""Oracle audit of bias-aware cross-view Stage2 fragment association.

High-purity Stage2 fragments are labeled by exact point-aligned GT joins.  For
each fragment pair, the audit compares ordinary pooled plane-fit cost with a
canonical-plane cost that marginalizes regularized per-view offsets.  ROC AUC
and average precision test whether the nuisance model actually separates
same-plane from different-plane fragment pairs, especially for near-parallel
and disjoint-view pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from evaluate_canonical_plane_bias_oracle import (
    fit_canonical_view_bias,
    weighted_fit_plane,
)


OUTPUT_SCHEMA_VERSION = 1


def pack_keys(view_indices: np.ndarray, pixel_xy: np.ndarray) -> np.ndarray:
    view_indices = np.asarray(view_indices, dtype=np.int64)
    pixel_xy = np.asarray(pixel_xy, dtype=np.int64)
    if pixel_xy.shape != (len(view_indices), 2):
        raise ValueError("pixel_xy must have shape (N,2)")
    if (view_indices < 0).any() or (pixel_xy < 0).any() or (pixel_xy >= (1 << 21)).any():
        raise ValueError("view indices and pixels must be non-negative and below 2^21")
    return (view_indices << 42) | (pixel_xy[:, 0] << 21) | pixel_xy[:, 1]


def exact_gt_join(
    record_views: np.ndarray,
    record_pixels: np.ndarray,
    gt_views: np.ndarray,
    gt_pixels: np.ndarray,
    gt_labels: np.ndarray,
) -> np.ndarray:
    record_keys = pack_keys(record_views, record_pixels)
    gt_keys = pack_keys(gt_views, gt_pixels)
    order = np.argsort(gt_keys)
    sorted_keys = gt_keys[order]
    positions = np.searchsorted(sorted_keys, record_keys)
    matched = positions < len(sorted_keys)
    matched[matched] &= sorted_keys[positions[matched]] == record_keys[matched]
    result = np.full((len(record_keys),), -1, dtype=np.int64)
    result[matched] = gt_labels[order[positions[matched]]]
    return result


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    cosine = float(
        np.clip(
            abs(first @ second)
            / max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-12),
            0.0,
            1.0,
        )
    )
    return math.degrees(math.acos(cosine))


def binary_auc_lower(labels: np.ndarray, costs: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = -np.asarray(costs, dtype=np.float64)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty((len(scores),), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = float(ranks[labels].sum())
    return (
        rank_sum - positive * (positive + 1) / 2.0
    ) / (positive * negative)


def average_precision_lower(labels: np.ndarray, costs: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    costs = np.asarray(costs, dtype=np.float64)
    positive = int(labels.sum())
    if positive == 0:
        return float("nan")
    order = np.argsort(costs, kind="mergesort")
    ordered = labels[order]
    cumulative = np.cumsum(ordered)
    ranks = np.arange(1, len(ordered) + 1)
    return float(np.sum((cumulative / ranks) * ordered) / positive)


def build_fragments(
    support_path: Path,
    gt_path: Path,
    *,
    min_fragment_points: int,
    min_gt_purity: float,
    max_points_per_fragment: int,
) -> tuple[list[dict], float]:
    with np.load(support_path, allow_pickle=True) as support:
        points = support["points"].astype(np.float64)
        fragment_ids = support["point_plane_ids"].astype(np.int64)
        views = support["alignment_view_indices"].astype(np.int64)
        pixels = support["pointmap_pixel_xy"].astype(np.int64)
    with np.load(gt_path, allow_pickle=False) as gt:
        gt_views = gt["view_indices"].astype(np.int64)
        gt_pixels = gt["pixel_xy"].astype(np.int64)
        gt_labels = gt["point_plane_ids"].astype(np.int64)
    matched_labels = exact_gt_join(
        views, pixels, gt_views, gt_pixels, gt_labels)
    center = np.median(points, axis=0)
    scene_scale = max(
        float(np.median(np.linalg.norm(points - center, axis=1))),
        1e-8,
    )
    fragments = []
    for fragment_id in np.unique(fragment_ids[fragment_ids >= 0]):
        mask = fragment_ids == int(fragment_id)
        local_gt = matched_labels[mask]
        valid_gt = local_gt >= 0
        if int(valid_gt.sum()) < int(min_fragment_points):
            continue
        values, counts = np.unique(local_gt[valid_gt], return_counts=True)
        dominant_index = int(np.argmax(counts))
        dominant_gt = int(values[dominant_index])
        purity = float(counts[dominant_index] / valid_gt.sum())
        if purity < float(min_gt_purity):
            continue
        local_points = points[mask]
        local_views = views[mask]
        if len(local_points) > int(max_points_per_fragment):
            indices = np.linspace(
                0, len(local_points) - 1, int(max_points_per_fragment), dtype=np.int64)
            local_points = local_points[indices]
            local_views = local_views[indices]
        normal, offset = weighted_fit_plane(local_points)
        fragments.append(
            {
                "fragment_id": int(fragment_id),
                "dominant_gt_plane": dominant_gt,
                "gt_purity": purity,
                "matched_gt_points": int(valid_gt.sum()),
                "points": local_points,
                "views": local_views,
                "view_set": set(int(value) for value in np.unique(local_views)),
                "normal": normal,
                "offset": float(offset),
            }
        )
    return fragments, scene_scale


def pair_rows(
    scene: str,
    fragments: list[dict],
    scene_scale: float,
    bias_ridge: float,
) -> list[dict]:
    rows = []
    for first_index, first in enumerate(fragments):
        for second in fragments[first_index + 1 :]:
            points = np.concatenate((first["points"], second["points"]))
            views = np.concatenate((first["views"], second["views"]))
            if len(np.unique(views)) < 2:
                continue
            pooled_normal, pooled_offset = weighted_fit_plane(points)
            pooled_residual = points @ pooled_normal + pooled_offset
            pooled_rmse = float(np.sqrt(np.mean(pooled_residual**2)))
            bias_model = fit_canonical_view_bias(
                points,
                views,
                np.ones((len(points),), dtype=np.float64),
                bias_ridge=bias_ridge,
                use_reliability=False,
                iterations=12,
            )
            bias_per_point = np.asarray(
                [bias_model["view_biases"][int(view)] for view in views],
                dtype=np.float64,
            )
            bias_residual = (
                points @ bias_model["normal"]
                + float(bias_model["offset"])
                - bias_per_point
            )
            bias_rmse = float(np.sqrt(np.mean(bias_residual**2)))
            bias_values = np.asarray(
                list(bias_model["view_biases"].values()), dtype=np.float64)
            rows.append(
                {
                    "scene": scene,
                    "first_fragment_id": first["fragment_id"],
                    "second_fragment_id": second["fragment_id"],
                    "same_gt_plane": bool(
                        first["dominant_gt_plane"] == second["dominant_gt_plane"]
                    ),
                    "first_gt_plane": first["dominant_gt_plane"],
                    "second_gt_plane": second["dominant_gt_plane"],
                    "first_gt_purity": first["gt_purity"],
                    "second_gt_purity": second["gt_purity"],
                    "normal_angle_deg": angle_deg(first["normal"], second["normal"]),
                    "disjoint_view_sets": bool(first["view_set"].isdisjoint(second["view_set"])),
                    "union_view_count": int(len(first["view_set"] | second["view_set"])),
                    "pooled_rmse_normalized": pooled_rmse / scene_scale,
                    "bias_corrected_rmse_normalized": bias_rmse / scene_scale,
                    "required_bias_std_normalized": float(bias_values.std()) / scene_scale,
                    "required_bias_range_normalized": float(np.ptp(bias_values)) / scene_scale,
                }
            )
    return rows


def subset_metrics(
    rows: list[dict],
    penalty_multipliers: list[float],
) -> dict:
    labels = np.asarray([row["same_gt_plane"] for row in rows], dtype=bool)
    result = {
        "pairs": int(len(rows)),
        "positive_pairs": int(labels.sum()),
        "negative_pairs": int((~labels).sum()),
    }
    costs = {
        "pooled_rmse": np.asarray(
            [row["pooled_rmse_normalized"] for row in rows], dtype=np.float64),
        "bias_corrected_rmse": np.asarray(
            [row["bias_corrected_rmse_normalized"] for row in rows], dtype=np.float64),
    }
    bias_cost = costs["bias_corrected_rmse"]
    penalty = np.asarray(
        [row["required_bias_std_normalized"] for row in rows], dtype=np.float64)
    for multiplier in penalty_multipliers:
        costs[f"bias_penalized_{multiplier:g}"] = bias_cost + float(multiplier) * penalty
    for name, values in costs.items():
        result[name] = {
            "roc_auc": binary_auc_lower(labels, values),
            "average_precision": average_precision_lower(labels, values),
        }
    return result


def summarize(rows: list[dict], near_parallel_deg: float, penalty_multipliers: list[float]) -> dict:
    subsets = {
        "all_multiview_pairs": rows,
        "near_parallel_pairs": [
            row for row in rows if row["normal_angle_deg"] <= float(near_parallel_deg)
        ],
        "near_parallel_disjoint_view_pairs": [
            row
            for row in rows
            if row["normal_angle_deg"] <= float(near_parallel_deg)
            and row["disjoint_view_sets"]
        ],
    }
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "oracle_fragment_gt_labels": True,
        "pointmaps_modified": False,
        "near_parallel_threshold_deg": float(near_parallel_deg),
        "subsets": {
            name: subset_metrics(values, penalty_multipliers)
            for name, values in subsets.items()
            if values
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "scene",
        "first_fragment_id",
        "second_fragment_id",
        "same_gt_plane",
        "first_gt_plane",
        "second_gt_plane",
        "first_gt_purity",
        "second_gt_purity",
        "normal_angle_deg",
        "disjoint_view_sets",
        "union_view_count",
        "pooled_rmse_normalized",
        "bias_corrected_rmse_normalized",
        "required_bias_std_normalized",
        "required_bias_range_normalized",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _artifact_path(root: Path, item: dict, name: str) -> Path:
    artifact = item["artifacts"][name]
    reference = artifact["path"] if isinstance(artifact, dict) else artifact
    return root / reference


def _scene_name(item: dict, path: Path) -> str:
    for key in ("item_id", "case_id", "name", "scene_key"):
        if key in item:
            return str(item[key])
    suffix = "_point_aligned_plane_gt"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test bias-aware association costs on GT-labeled Stage2 fragments."
    )
    parser.add_argument("--batch_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_fragment_points", type=int, default=64)
    parser.add_argument("--min_gt_purity", type=float, default=0.8)
    parser.add_argument("--max_points_per_fragment", type=int, default=512)
    parser.add_argument("--bias_ridge", type=float, default=0.25)
    parser.add_argument("--near_parallel_deg", type=float, default=10.0)
    parser.add_argument("--penalty_multipliers", default="0.25,0.5,1.0,2.0")
    args = parser.parse_args()

    batch_path = Path(args.batch_json).resolve()
    root = batch_path
    while root.parent != root and not (root / ".git").exists():
        root = root.parent
    if not (root / ".git").exists():
        raise ValueError("Could not infer repository root from batch_json")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    penalties = [
        float(value) for value in args.penalty_multipliers.split(",") if value.strip()
    ]

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    rows = []
    scene_fragment_counts = {}
    for item in batch["items"]:
        gt_path = _artifact_path(root, item, "point_aligned_gt")
        support_path = _artifact_path(root, item, "direct_support_records")
        scene = _scene_name(item, gt_path)
        fragments, scene_scale = build_fragments(
            support_path,
            gt_path,
            min_fragment_points=args.min_fragment_points,
            min_gt_purity=args.min_gt_purity,
            max_points_per_fragment=args.max_points_per_fragment,
        )
        scene_fragment_counts[scene] = len(fragments)
        rows.extend(
            pair_rows(scene, fragments, scene_scale, args.bias_ridge))

    result = summarize(rows, args.near_parallel_deg, penalties)
    result["config"] = {
        "batch_json": str(batch_path),
        "fragment_source": "stage2_support_direct_global_svd",
        "min_fragment_points": int(args.min_fragment_points),
        "min_gt_purity": float(args.min_gt_purity),
        "max_points_per_fragment": int(args.max_points_per_fragment),
        "bias_ridge": float(args.bias_ridge),
        "penalty_multipliers": penalties,
        "join_key": "(alignment_view_index,x,y)",
    }
    result["scene_fragment_counts"] = scene_fragment_counts
    result["limitations"] = [
        "GT labels are used only to select high-purity fragments and score pair identity.",
        "Pairwise AUC/AP does not by itself prove that global clustering will improve.",
        "The nuisance offset may merge distinct parallel planes unless bounded connectivity and appearance are added.",
    ]
    result["outputs"] = {
        "summary_json": str(output_dir / "summary.json"),
        "pair_csv": str(output_dir / "fragment_pair_metrics.csv"),
        "pair_json": str(output_dir / "fragment_pair_metrics.json"),
    }
    write_csv(output_dir / "fragment_pair_metrics.csv", rows)
    (output_dir / "fragment_pair_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
