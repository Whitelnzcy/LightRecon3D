"""Audit the frozen support-to-bounded evaluator contract.

This entry point deliberately implements E0 only.  The preregistered protocol
requires E1--E3 to stop when E0 fails, so later configurations must be added in
a separate version only after E0 passes without changing its metric contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch
from scipy.spatial import cKDTree

from evaluate_global_plane_baselines import (
    evaluate_arrays,
    iou_matrix,
    linear_sum_assignment,
)
from global_plane_baselines import load_global_cache
from lift_support_prediction_to_global_cache import pack_registry_keys


SCHEMA_VERSION = 1
PRESSURE_SCENES = (
    "final_003_scene_00184",
    "final_013_scene_00195",
    "final_014_scene_00197",
)
JOIN_FIELDS = (
    "scene",
    "join_variant",
    "gt_plane_id",
    "input_record_count",
    "unique_input_key_count",
    "duplicate_key_count",
    "duplicate_key_group_count",
    "duplicate_same_global_point_count",
    "duplicate_multiple_global_point_count",
    "ambiguous_key_count",
    "unmatched_key_count",
    "matched_unique_global_point_count",
    "match_rate",
)
RESIDUAL_FIELDS = (
    "scene",
    "gt_plane_id",
    "seed_unique_point_count",
    "raw_residual_median",
    "raw_residual_mad",
    "raw_residual_p90",
    "raw_residual_p95",
    "raw_residual_p99",
    "raw_residual_max",
    "scene_bbox_diagonal",
    "support_xyz_bbox_diagonal",
    "median_8nn_distance",
    "median_residual_over_scene_diagonal",
    "median_residual_over_support_diagonal",
    "median_residual_over_median_8nn",
    "p95_residual_over_scene_diagonal",
    "p95_residual_over_support_diagonal",
    "p95_residual_over_median_8nn",
)
EXPANSION_FIELDS = (
    "scene",
    "config",
    "gt_plane_id",
    "seed_unique_point_count",
    "tangent_bbox_area",
    "seed_convex_hull_area",
    "candidate_count_before_normal_filter",
    "candidate_count_after_normal_filter",
    "expanded_point_count",
    "expanded_unique_point_count",
    "added_point_count",
    "added_point_GT_purity",
    "added_point_dominant_GT",
    "largest_added_connected_component",
    "seed_retention_rate",
)
REFIT_FIELDS = (
    "scene",
    "config",
    "gt_plane_id",
    "normal_angle_change",
    "raw_offset_change",
    "offset_change_over_scene_diagonal",
    "offset_change_over_median_8nn",
    "seed_median_residual_before",
    "seed_median_residual_after",
    "seed_p95_residual_before",
    "seed_p95_residual_after",
    "seed_retention_before",
    "seed_retention_after",
    "support_size_before",
    "support_size_after",
    "matched_iou_before",
    "matched_iou_after",
    "added_point_purity",
    "refit_accepted",
    "refit_decision",
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    return (
        float(numerator) / float(denominator)
        if abs(float(denominator)) > 1e-12
        else float("nan")
    )


def _percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if len(values) else float("nan")


def _bbox_diagonal(points: np.ndarray) -> float:
    if not len(points):
        return float("nan")
    return float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))


def median_eight_neighbor_distance(points: np.ndarray) -> float:
    """Median per-point mean distance to up to eight non-self neighbours."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return float("nan")
    neighbor_count = min(8, len(points) - 1)
    distances, _ = cKDTree(points).query(
        points, k=neighbor_count + 1, workers=-1
    )
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim == 1:
        distances = distances[:, None]
    local = distances[:, 1 : neighbor_count + 1]
    return float(np.median(np.mean(local, axis=1)))


def _grouped_key_rows(
    record_keys: np.ndarray,
    record_labels: np.ndarray,
    cache_key_to_index: dict[int, int],
    *,
    scene: str,
    variant: str,
    plane_id: int,
) -> dict:
    order = np.argsort(record_keys, kind="stable")
    sorted_keys = record_keys[order]
    sorted_labels = record_labels[order]
    starts = (
        np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        if len(sorted_keys)
        else np.empty((0,), dtype=np.int64)
    )
    ends = np.r_[starts[1:], len(sorted_keys)]
    duplicate_groups = 0
    duplicate_records = 0
    ambiguous = 0
    unmatched = 0
    matched = 0
    same_global_duplicates = 0
    multiple_global_duplicates = 0
    for start, end in zip(starts, ends):
        key = int(sorted_keys[start])
        count = int(end - start)
        if count > 1:
            duplicate_groups += 1
            duplicate_records += count - 1
        labels = np.unique(sorted_labels[start:end])
        if len(labels) > 1:
            ambiguous += 1
            continue
        cache_index = cache_key_to_index.get(key)
        if cache_index is None:
            unmatched += 1
            continue
        matched += 1
        if count > 1:
            # The cache registry is verified one-to-one before this function.
            same_global_duplicates += count - 1
        if isinstance(cache_index, tuple):
            multiple_global_duplicates += count - 1
    unique_count = int(len(starts))
    return {
        "scene": scene,
        "join_variant": variant,
        "gt_plane_id": int(plane_id),
        "input_record_count": int(len(record_keys)),
        "unique_input_key_count": unique_count,
        "duplicate_key_count": int(duplicate_records),
        "duplicate_key_group_count": int(duplicate_groups),
        "duplicate_same_global_point_count": int(same_global_duplicates),
        "duplicate_multiple_global_point_count": int(
            multiple_global_duplicates
        ),
        "ambiguous_key_count": int(ambiguous),
        "unmatched_key_count": int(unmatched),
        "matched_unique_global_point_count": int(matched),
        "match_rate": _safe_ratio(matched, unique_count),
    }


def exact_unique_join(
    cache_views: np.ndarray,
    cache_pixels: np.ndarray,
    record_views: np.ndarray,
    record_pixels: np.ndarray,
    record_labels: np.ndarray,
    *,
    scene: str,
    variant: str,
) -> tuple[np.ndarray, list[dict]]:
    """Join unique, non-ambiguous positive records to a one-to-one cache."""

    cache_keys = pack_registry_keys(cache_views, cache_pixels, "global cache")
    unique_cache, counts = np.unique(cache_keys, return_counts=True)
    duplicate_cache = unique_cache[counts > 1]
    if len(duplicate_cache):
        raise ValueError(
            f"{scene}: cache has {len(duplicate_cache)} duplicate registry keys"
        )
    cache_map = {int(key): index for index, key in enumerate(cache_keys)}
    labels = np.asarray(record_labels, dtype=np.int64).reshape(-1)
    views = np.asarray(record_views, dtype=np.int64).reshape(-1)
    pixels = np.asarray(record_pixels, dtype=np.int64)
    positive = labels >= 0
    keys = pack_registry_keys(views[positive], pixels[positive], variant)
    positive_labels = labels[positive]
    rows = []
    for plane_id in sorted(int(value) for value in np.unique(positive_labels)):
        mask = positive_labels == plane_id
        rows.append(
            _grouped_key_rows(
                keys[mask],
                positive_labels[mask],
                cache_map,
                scene=scene,
                variant=variant,
                plane_id=plane_id,
            )
        )
    rows.append(
        _grouped_key_rows(
            keys,
            positive_labels,
            cache_map,
            scene=scene,
            variant=variant,
            plane_id=-1,
        )
    )

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_labels = positive_labels[order]
    starts = (
        np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        if len(sorted_keys)
        else np.empty((0,), dtype=np.int64)
    )
    ends = np.r_[starts[1:], len(sorted_keys)]
    assignment = np.full(len(cache_keys), -1, dtype=np.int32)
    for start, end in zip(starts, ends):
        label_set = np.unique(sorted_labels[start:end])
        if len(label_set) != 1:
            continue
        cache_index = cache_map.get(int(sorted_keys[start]))
        if cache_index is not None:
            assignment[cache_index] = int(label_set[0])
    return assignment, rows


def matching_rows(
    scene: str,
    points: np.ndarray,
    pred_labels: np.ndarray,
    pred_normals: np.ndarray,
    pred_offsets: np.ndarray,
    gt_labels: np.ndarray,
    gt_normals: np.ndarray,
) -> tuple[list[dict], dict]:
    pred_ids = np.asarray(
        sorted(int(value) for value in np.unique(pred_labels) if value >= 0)
    )
    gt_ids = np.asarray(
        sorted(int(value) for value in np.unique(gt_labels) if value >= 0)
    )
    matrix = iou_matrix(pred_labels, gt_labels, pred_ids, gt_ids)
    row_indices, column_indices = (
        linear_sum_assignment(1.0 - matrix)
        if matrix.size
        else (np.empty((0,), dtype=int), np.empty((0,), dtype=int))
    )
    pred_to_row = {
        int(pred_ids[row]): (int(gt_ids[column]), float(matrix[row, column]))
        for row, column in zip(row_indices, column_indices)
    }
    rows = []
    for pred_id in pred_ids:
        gt_id, score = pred_to_row.get(int(pred_id), (-1, 0.0))
        normal_error = float("nan")
        residual = float("nan")
        if gt_id >= 0:
            cosine = np.clip(
                abs(float(pred_normals[pred_id] @ gt_normals[gt_id])), 0.0, 1.0
            )
            normal_error = float(np.degrees(np.arccos(cosine)))
        support = pred_labels == pred_id
        matched_intersection = (
            int((support & (gt_labels == gt_id)).sum()) if gt_id >= 0 else 0
        )
        if support.any():
            residual = float(
                np.median(
                    np.abs(
                        points[support] @ pred_normals[pred_id]
                        + pred_offsets[pred_id]
                    )
                )
            )
        rows.append(
            {
                "scene": scene,
                "config": "E0",
                "pred_plane_id": int(pred_id),
                "gt_plane_id": int(gt_id),
                "matched_iou": float(score),
                "plane_tp": bool(score >= 0.5),
                "pred_point_count": int(support.sum()),
                "gt_point_count": int((gt_labels == gt_id).sum())
                if gt_id >= 0
                else 0,
                "matched_intersection_point_count": matched_intersection,
                "normal_angular_error_deg": normal_error,
                "pred_support_median_residual": residual,
            }
        )
    matched_scores = np.asarray(
        [row["matched_iou"] for row in rows], dtype=np.float64
    )
    matched_weights = np.asarray(
        [row["gt_point_count"] for row in rows], dtype=np.float64
    )
    summary = {
        "plane_weighted_matched_iou_mean": float(np.mean(matched_scores))
        if len(matched_scores)
        else 0.0,
        "plane_weighted_matched_iou_median": float(np.median(matched_scores))
        if len(matched_scores)
        else 0.0,
        "minimum_matched_iou": float(np.min(matched_scores))
        if len(matched_scores)
        else 0.0,
        "point_weighted_matched_iou": _safe_ratio(
            float(np.sum(matched_scores * matched_weights)),
            float(np.sum(matched_weights)),
        ),
        "gt_support_matched_point_count": int(
            sum(
                row["matched_intersection_point_count"]
                for row in rows
                if row["plane_tp"]
            )
        ),
    }
    return rows, summary


def residual_rows(
    scene: str,
    points: np.ndarray,
    labels: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
) -> list[dict]:
    scene_diagonal = _bbox_diagonal(points)
    rows = []
    for plane_id in range(len(normals)):
        support = points[labels == plane_id]
        residual = np.abs(support @ normals[plane_id] + offsets[plane_id])
        median = _percentile(residual, 0.5)
        p95 = _percentile(residual, 0.95)
        support_diagonal = _bbox_diagonal(support)
        local_scale = median_eight_neighbor_distance(support)
        rows.append(
            {
                "scene": scene,
                "gt_plane_id": int(plane_id),
                "seed_unique_point_count": int(len(support)),
                "raw_residual_median": median,
                "raw_residual_mad": _percentile(
                    np.abs(residual - median), 0.5
                ),
                "raw_residual_p90": _percentile(residual, 0.9),
                "raw_residual_p95": p95,
                "raw_residual_p99": _percentile(residual, 0.99),
                "raw_residual_max": float(np.max(residual)),
                "scene_bbox_diagonal": scene_diagonal,
                "support_xyz_bbox_diagonal": support_diagonal,
                "median_8nn_distance": local_scale,
                "median_residual_over_scene_diagonal": _safe_ratio(
                    median, scene_diagonal
                ),
                "median_residual_over_support_diagonal": _safe_ratio(
                    median, support_diagonal
                ),
                "median_residual_over_median_8nn": _safe_ratio(
                    median, local_scale
                ),
                "p95_residual_over_scene_diagonal": _safe_ratio(
                    p95, scene_diagonal
                ),
                "p95_residual_over_support_diagonal": _safe_ratio(
                    p95, support_diagonal
                ),
                "p95_residual_over_median_8nn": _safe_ratio(p95, local_scale),
            }
        )
    return rows


def disabled_stage_rows(
    scene: str,
    labels: np.ndarray,
    residual_by_plane: dict[int, dict],
) -> tuple[list[dict], list[dict]]:
    expansion = []
    refit = []
    for plane_id in sorted(int(value) for value in np.unique(labels) if value >= 0):
        count = int((labels == plane_id).sum())
        residual = residual_by_plane[plane_id]
        expansion.append(
            {
                "scene": scene,
                "config": "E0",
                "gt_plane_id": plane_id,
                "seed_unique_point_count": count,
                "tangent_bbox_area": float("nan"),
                "seed_convex_hull_area": float("nan"),
                "candidate_count_before_normal_filter": 0,
                "candidate_count_after_normal_filter": 0,
                "expanded_point_count": count,
                "expanded_unique_point_count": count,
                "added_point_count": 0,
                "added_point_GT_purity": float("nan"),
                "added_point_dominant_GT": -1,
                "largest_added_connected_component": 0,
                "seed_retention_rate": 1.0,
            }
        )
        refit.append(
            {
                "scene": scene,
                "config": "E0",
                "gt_plane_id": plane_id,
                "normal_angle_change": 0.0,
                "raw_offset_change": 0.0,
                "offset_change_over_scene_diagonal": 0.0,
                "offset_change_over_median_8nn": 0.0,
                "seed_median_residual_before": residual[
                    "raw_residual_median"
                ],
                "seed_median_residual_after": residual["raw_residual_median"],
                "seed_p95_residual_before": residual["raw_residual_p95"],
                "seed_p95_residual_after": residual["raw_residual_p95"],
                "seed_retention_before": 1.0,
                "seed_retention_after": 1.0,
                "support_size_before": count,
                "support_size_after": count,
                "matched_iou_before": 1.0,
                "matched_iou_after": 1.0,
                "added_point_purity": float("nan"),
                "refit_accepted": False,
                "refit_decision": "disabled_by_E0_contract",
            }
        )
    return expansion, refit


def write_table(output_dir: Path, stem: str, rows: list[dict], fields) -> None:
    with (output_dir / f"{stem}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    (output_dir / f"{stem}.json").write_text(
        json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8"
    )


def git_state(root: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, encoding="utf-8"
        ).strip()

    return {
        "sha": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "working_tree_status": run("status", "--short"),
        "diff_stat": run("diff", "--stat"),
    }


def environment_state() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
    }


def load_scene(scene_entry: dict, root: Path) -> dict:
    gt_path = Path(scene_entry["gt_npz"]).resolve()
    support_path = Path(scene_entry["support_npz"]).resolve()
    with np.load(gt_path, allow_pickle=False) as raw:
        gt = {key: raw[key] for key in raw.files}
    cache_path = Path(str(gt["source_global_cloud_npz"].item()))
    if not cache_path.is_absolute():
        cache_path = (root / cache_path).resolve()
    cache = load_global_cache(cache_path, float(gt["min_conf"].item()))
    if cache["points"].shape != gt["points"].shape or not np.allclose(
        cache["points"], gt["points"], atol=1e-5
    ):
        raise ValueError(f"{scene_entry['scene']}: cache and GT point order differ")
    return {
        "gt_path": gt_path,
        "support_path": support_path,
        "cache_path": cache_path,
        "gt": gt,
        "cache": cache,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preregistered exact-support evaluator contract E0."
    )
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--pressure_scenes", nargs="+", default=list(PRESSURE_SCENES)
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    manifest_path = Path(args.manifest_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    log_lines = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested = set(args.pressure_scenes)
    entries = [
        scene for scene in manifest["scenes"] if scene["scene"] in requested
    ]
    missing = sorted(requested - {scene["scene"] for scene in entries})
    if missing:
        raise ValueError(f"pressure scenes missing from manifest: {missing}")

    configuration = {
        "schema_version": SCHEMA_VERSION,
        "config": "E0",
        "expansion": "disabled",
        "refit": "disabled",
        "support": "exact unique point-aligned GT plane support",
        "key_policy": "unique keys; ambiguous drop; unmatched drop",
        "bounded_matcher": "full-cache IoU + Hungarian",
        "plane_tp_iou": 0.5,
        "fragmentation_iou": 0.1,
        "min_observed_plane_points": 64,
        "gt_support_assignment_rate": "|A intersection G| / |G|",
        "gt_support_matched_coverage": "|M| / |G|",
        "full_cache_label_rate": "|A| / |C|",
        "gt_plane_label_rate": "|G| / |C|",
        "nonplane_false_assignment_rate": "|A minus G| / |C minus G|",
        "legacy_assignment_rate_acceptance": "retired",
        "median_8nn_definition": (
            "median over points of mean distance to eight non-self neighbours"
        ),
        "pressure_scenes": list(args.pressure_scenes),
        "pressure_scenes_used_for_tuning": False,
        "training_calibration_performed": False,
        "checkpoint": "not_used",
        "manifest_json": str(manifest_path),
    }
    (output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    state = git_state(root)
    state["environment"] = environment_state()
    (output_dir / "git_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )

    join_rows: list[dict] = []
    residual_table: list[dict] = []
    expansion_table: list[dict] = []
    refit_table: list[dict] = []
    plane_table: list[dict] = []
    scene_table: list[dict] = []
    input_hashes = []

    for entry in entries:
        scene = entry["scene"]
        data = load_scene(entry, root)
        gt = data["gt"]
        cache = data["cache"]
        labels = gt["point_plane_ids"].astype(np.int32)
        normals = gt["plane_normals"].astype(np.float64)
        offsets = gt["plane_offsets"].astype(np.float64)
        assignment, exact_rows = exact_unique_join(
            cache["view_indices"],
            cache["pixel_xy"],
            gt["view_indices"],
            gt["pixel_xy"],
            labels,
            scene=scene,
            variant="E0_exact_point_aligned_gt",
        )
        join_rows.extend(exact_rows)

        with np.load(data["support_path"], allow_pickle=False) as sparse:
            _, sparse_rows = exact_unique_join(
                cache["view_indices"],
                cache["pixel_xy"],
                sparse["alignment_view_indices"],
                sparse["pointmap_pixel_xy"],
                sparse["point_plane_ids"],
                scene=scene,
                variant="current_sparse_support_policy_diagnostic",
            )
        # Only the aggregate sparse row is meaningful because fragment IDs are
        # not GT plane identities.
        join_rows.append(sparse_rows[-1])

        if not np.array_equal(assignment, labels):
            mismatch = int((assignment != labels).sum())
            raise ValueError(f"{scene}: exact E0 join changed {mismatch} labels")

        metrics = evaluate_arrays(
            gt["points"],
            assignment,
            normals,
            offsets,
            labels,
            gt["plane_normals"],
            0.5,
            0.1,
            64,
        )
        precision = float(metrics["plane_precision"])
        recall = float(metrics["plane_recall"])
        plane_f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        matched_rows, matched_summary = matching_rows(
            scene,
            gt["points"],
            assignment,
            normals,
            offsets,
            labels,
            gt["plane_normals"],
        )
        plane_table.extend(matched_rows)
        residual = residual_rows(
            scene, gt["points"], labels, normals, offsets
        )
        residual_table.extend(residual)
        residual_by_plane = {
            int(row["gt_plane_id"]): row for row in residual
        }
        expansion, refit = disabled_stage_rows(
            scene, labels, residual_by_plane
        )
        expansion_table.extend(expansion)
        refit_table.extend(refit)

        gt_support = labels >= 0
        assigned = assignment >= 0
        assigned_cache_point_count = int(assigned.sum())
        gt_plane_point_count = int(gt_support.sum())
        gt_support_assigned_point_count = int((assigned & gt_support).sum())
        gt_support_matched_point_count = int(
            matched_summary["gt_support_matched_point_count"]
        )
        nonplane_point_count = int((~gt_support).sum())
        nonplane_false_assigned_point_count = int(
            (assigned & ~gt_support).sum()
        )
        gt_support_assignment_rate = _safe_ratio(
            gt_support_assigned_point_count, gt_plane_point_count
        )
        gt_support_matched_coverage = _safe_ratio(
            gt_support_matched_point_count, gt_plane_point_count
        )
        full_cache_label_rate = _safe_ratio(
            assigned_cache_point_count, len(labels)
        )
        gt_plane_label_rate = _safe_ratio(gt_plane_point_count, len(labels))
        nonplane_false_assignment_rate = (
            _safe_ratio(
                nonplane_false_assigned_point_count, nonplane_point_count
            )
            if nonplane_point_count
            else None
        )
        row = {
            "scene": scene,
            "config": "E0",
            "cache_point_count": int(len(labels)),
            "assigned_cache_point_count": assigned_cache_point_count,
            "gt_plane_point_count": gt_plane_point_count,
            "gt_support_assigned_point_count": (
                gt_support_assigned_point_count
            ),
            "gt_support_matched_point_count": (
                gt_support_matched_point_count
            ),
            "nonplane_point_count": nonplane_point_count,
            "nonplane_false_assigned_point_count": (
                nonplane_false_assigned_point_count
            ),
            "pred_plane_count": int(metrics["pred_plane_count"]),
            "gt_plane_count": int(metrics["gt_plane_count"]),
            "true_positive_planes": int(metrics["true_positive_planes"]),
            "plane_precision": precision,
            "plane_recall": recall,
            "plane_f1": plane_f1,
            **matched_summary,
            "gt_support_assignment_rate": gt_support_assignment_rate,
            "gt_support_matched_coverage": gt_support_matched_coverage,
            "full_cache_label_rate": full_cache_label_rate,
            "gt_plane_label_rate": gt_plane_label_rate,
            "nonplane_false_assignment_rate": (
                nonplane_false_assignment_rate
            ),
            "legacy_support_coverage": float(metrics["support_coverage"]),
            "legacy_support_coverage_formula": "|A intersection G| / |G|",
            "fragmentation_excess": int(metrics["fragmentation_excess"]),
            "overmerge_excess": int(metrics["overmerge_excess"]),
            "e0_plane_f1_pass": bool(plane_f1 >= 0.95),
            "e0_tp_pass": bool(
                int(metrics["true_positive_planes"])
                == int(metrics["gt_plane_count"])
            ),
            "e0_gt_support_assignment_pass": bool(
                gt_support_assignment_rate >= 0.98
            ),
            "e0_gt_support_matched_coverage_pass": bool(
                gt_support_matched_coverage >= 0.98
            ),
            "e0_integer_count_pass": bool(
                assigned_cache_point_count == gt_plane_point_count
                and nonplane_false_assigned_point_count == 0
            ),
            "e0_label_rate_equality_pass": bool(
                abs(full_cache_label_rate - gt_plane_label_rate) <= 1e-8
            ),
            "e0_median_iou_pass": bool(
                matched_summary["plane_weighted_matched_iou_median"] >= 0.90
            ),
            "e0_min_iou_pass": bool(
                matched_summary["minimum_matched_iou"] >= 0.80
            ),
        }
        row["e0_all_pass"] = bool(
            row["e0_plane_f1_pass"]
            and row["e0_tp_pass"]
            and row["e0_gt_support_assignment_pass"]
            and row["e0_gt_support_matched_coverage_pass"]
            and row["e0_integer_count_pass"]
            and row["e0_label_rate_equality_pass"]
            and row["e0_median_iou_pass"]
            and row["e0_min_iou_pass"]
        )
        scene_table.append(row)
        log(
            f"{scene}: F1={plane_f1:.6f} TP="
            f"{metrics['true_positive_planes']}/{metrics['gt_plane_count']} "
            f"IoU_med={matched_summary['plane_weighted_matched_iou_median']:.6f} "
            f"gt_assignment={gt_support_assignment_rate:.6f} "
            f"matched_coverage={gt_support_matched_coverage:.6f} "
            f"full_cache_label_rate={full_cache_label_rate:.6f} "
            f"gt_plane_label_rate={gt_plane_label_rate:.6f} "
            f"nonplane_false={nonplane_false_assigned_point_count} "
            f"pass={row['e0_all_pass']}"
        )
        input_hashes.append(
            {
                "scene": scene,
                "point_aligned_gt": {
                    "path": str(data["gt_path"]),
                    "sha256": sha256_file(data["gt_path"]),
                },
                "sparse_support_diagnostic": {
                    "path": str(data["support_path"]),
                    "sha256": sha256_file(data["support_path"]),
                },
                "global_cache": {
                    "path": str(data["cache_path"]),
                    "sha256": str(
                        gt["source_global_cloud_sha256"].item()
                    ),
                    "hash_source": "recorded by point-aligned GT writer",
                },
                "checkpoint": {"used": False, "sha256": None},
            }
        )

    all_pass = all(row["e0_all_pass"] for row in scene_table)
    verdict = "E0 Passed" if all_pass else "E-Key/Data Contract"
    next_action = (
        "Keep E1-E3 and L/G/J paused until the next preregistration."
        if all_pass
        else (
            "Enter E-Key/Data Contract audit for the failed explicit E0 "
            "metric; do not run E1-E3 or resume L/G/J."
        )
    )
    log(f"contract_verdict={verdict}")
    log(f"next_mechanical_action={next_action}")

    write_table(output_dir, "join_audit", join_rows, JOIN_FIELDS)
    write_table(
        output_dir, "residual_scale", residual_table, RESIDUAL_FIELDS
    )
    write_table(
        output_dir, "expansion_audit", expansion_table, EXPANSION_FIELDS
    )
    write_table(output_dir, "refit_drift", refit_table, REFIT_FIELDS)
    plane_fields = list(plane_table[0].keys())
    scene_fields = list(scene_table[0].keys())
    write_table(output_dir, "per_plane_metrics", plane_table, plane_fields)
    write_table(output_dir, "per_scene_metrics", scene_table, scene_fields)
    (output_dir / "input_hashes.json").write_text(
        json.dumps(input_hashes, indent=2), encoding="utf-8"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config": "E0",
        "e0_passed_all_pressure_scenes": all_pass,
        "contract_verdict": verdict,
        "next_mechanical_action": next_action,
        "e1_e2_e3_executed": False,
        "e1_e2_e3_stop_reason": (
            "not authorized in the metric-only denominator correction round"
            if all_pass
            else "E0 failed the explicit denominator contract"
        ),
        "legacy_e0_verdict_status": (
            "superseded: the old fixed >=0.98 full-cache assignment gate "
            "used a contradictory denominator"
        ),
        "scenes": scene_table,
        "input_hashes_json": str(output_dir / "input_hashes.json"),
        "artifacts": {
            "config_json": str(output_dir / "config.json"),
            "git_state_json": str(output_dir / "git_state.json"),
            "join_audit_csv": str(output_dir / "join_audit.csv"),
            "residual_scale_csv": str(output_dir / "residual_scale.csv"),
            "expansion_audit_csv": str(output_dir / "expansion_audit.csv"),
            "refit_drift_csv": str(output_dir / "refit_drift.csv"),
            "per_plane_metrics_csv": str(
                output_dir / "per_plane_metrics.csv"
            ),
            "per_scene_metrics_csv": str(
                output_dir / "per_scene_metrics.csv"
            ),
            "run_log": str(output_dir / "run.log"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output_dir / "run.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
