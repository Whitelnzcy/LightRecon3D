#!/usr/bin/env python3
"""Run the preregistered E1 tangent-bbox expansion safety audit.

E1 starts from the exact point-aligned GT support that passed E0, applies the
repository's frozen tangent-box candidate rule exactly once, and deliberately
does not refit any plane parameter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from audit_support_bounded_evaluator_contract import (
    PRESSURE_SCENES,
    _bbox_diagonal,
    _safe_ratio,
    environment_state,
    exact_unique_join,
    git_state,
    load_scene,
    matching_rows,
    sha256_file,
)
from evaluate_all_fragment_common_cache import build_tangent_bounds
from evaluate_global_plane_baselines import evaluate_arrays


SCHEMA_VERSION = 1
DISTANCE_THRESHOLD = 0.03
BOUND_MARGIN = 0.02
BOUND_TRIM_QUANTILE = 0.01
MIN_POINTS_PER_PLANE = 1000
MATCH_IOU = 0.5
FRAGMENTATION_IOU = 0.1
MIN_OBSERVED_PLANE_POINTS = 64

CONFLICT_RULE = (
    "For every point, form all plane memberships passing the frozen tangent "
    "bbox (including margin) and absolute normal-distance threshold. Assign "
    "the point to the eligible plane with minimum absolute residual; an exact "
    "tie is resolved by the lowest original plane index, matching np.argmin. "
    "If that plane has fewer than 1000 assigned points, drop it without "
    "reassigning its points."
)

PER_PLANE_FIELDS = (
    "scene_id",
    "gt_plane_id",
    "matched_prediction_id",
    "E0_TP",
    "E1_TP",
    "seed_unique_point_count",
    "seed_view_count",
    "tangent_bbox_min_u",
    "tangent_bbox_max_u",
    "tangent_bbox_min_v",
    "tangent_bbox_max_v",
    "tangent_bbox_area",
    "seed_convex_hull_area",
    "convex_hull_status",
    "bbox_to_hull_area_ratio",
    "candidate_count_before_plane_distance_filter",
    "candidate_count_after_plane_distance_filter",
    "expanded_unique_point_count",
    "added_point_count",
    "added_point_fraction_vs_seed",
    "removed_seed_point_count",
    "seed_retention_rate",
    "added_same_GT_plane_count",
    "added_other_GT_plane_count",
    "added_GT_unlabeled_count",
    "added_same_GT_plane_fraction",
    "added_other_GT_plane_fraction",
    "added_unlabeled_fraction",
    "added_dominant_wrong_GT_id",
    "expanded_points_already_owned_by_other_plane",
    "points_assigned_to_multiple_expanded_planes",
    "maximum_plane_multiplicity",
    "raw_plane_distance_threshold",
    "scene_bbox_diagonal",
    "support_bbox_diagonal",
    "seed_median_8NN_distance",
    "threshold_over_scene_bbox_diagonal",
    "threshold_over_support_bbox_diagonal",
    "threshold_over_seed_median_8NN_distance",
    "added_residual_median",
    "added_residual_p90",
    "added_residual_p95",
    "added_residual_max",
    "E0_matched_IoU",
    "E1_matched_IoU",
    "matched_IoU_delta",
    "E0_plane_TP",
    "E1_plane_TP",
)

SCENE_FIELDS = (
    "scene",
    "config",
    "cache_point_count",
    "assigned_cache_point_count",
    "gt_plane_point_count",
    "gt_support_assigned_point_count",
    "gt_support_matched_point_count",
    "nonplane_point_count",
    "nonplane_false_assigned_point_count",
    "pred_plane_count",
    "gt_plane_count",
    "true_positive_planes",
    "plane_precision",
    "plane_recall",
    "plane_f1",
    "plane_weighted_matched_iou_mean",
    "plane_weighted_matched_iou_median",
    "minimum_matched_iou",
    "point_weighted_matched_iou",
    "gt_support_assignment_rate",
    "gt_support_matched_coverage",
    "full_cache_label_rate",
    "gt_plane_label_rate",
    "nonplane_false_assignment_rate",
    "fragmentation_excess",
    "overmerge_excess",
    "added_point_count",
    "added_other_GT_plane_count",
    "added_other_GT_plane_fraction",
    "multi_candidate_point_count",
    "multi_candidate_fraction_of_gt_plane_points",
    "maximum_plane_multiplicity",
    "plane_weighted_iou_delta_from_E0",
    "point_weighted_iou_delta_from_E0",
    "plane_point_weighted_direction_consistent",
    "minimum_seed_retention_rate",
    "e1_all_pass",
    "failed_conditions",
)

MULTI_FIELDS = (
    "scene",
    "candidate_membership_count",
    "unique_candidate_point_count",
    "points_assigned_to_multiple_expanded_planes",
    "multi_candidate_fraction_of_GT_plane_points",
    "maximum_plane_multiplicity",
    "conflict_resolution_rule",
)

RESIDUAL_FIELDS = (
    "scene",
    "gt_plane_id",
    "raw_plane_distance_threshold",
    "scene_bbox_diagonal",
    "support_bbox_diagonal",
    "seed_median_8NN_distance",
    "threshold_over_scene_bbox_diagonal",
    "threshold_over_support_bbox_diagonal",
    "threshold_over_seed_median_8NN_distance",
    "added_point_count",
    "added_residual_median",
    "added_residual_p90",
    "added_residual_p95",
    "added_residual_max",
)


def _write_table(
    output_dir: Path, stem: str, rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> None:
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


def _percentile(values: np.ndarray, quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if len(values) else None


def _direction(value: float, *, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _directions_consistent(first: float, second: float) -> bool:
    first_direction = _direction(first)
    second_direction = _direction(second)
    return not (
        first_direction != 0
        and second_direction != 0
        and first_direction != second_direction
    )


def _convex_hull_area(uv: np.ndarray) -> tuple[float | None, str]:
    unique = np.unique(np.asarray(uv, dtype=np.float64), axis=0)
    if len(unique) < 3:
        return None, "not_computable: fewer_than_3_unique_points"
    try:
        hull = ConvexHull(unique)
    except QhullError:
        return None, "not_computable: degenerate_or_qhull_failure"
    area = float(hull.volume)
    if not math.isfinite(area) or area <= 0.0:
        return None, "not_computable: nonpositive_or_nonfinite_area"
    return area, "computed"


def expand_tangent_bbox_once_no_refit(
    points: np.ndarray,
    seed_assignment: np.ndarray,
    seed_normals: np.ndarray,
    seed_offsets: np.ndarray,
    tangent_bounds: list[dict[str, np.ndarray] | None],
    *,
    distance_threshold: float,
    bound_margin: float,
    min_points_per_plane: int,
) -> dict[str, Any]:
    """Apply the current one-pass bbox/residual assignment without any fit."""

    points = np.asarray(points, dtype=np.float64)
    seed_assignment = np.asarray(seed_assignment, dtype=np.int32)
    normals = np.asarray(seed_normals, dtype=np.float64)
    offsets = np.asarray(seed_offsets, dtype=np.float64)
    if len(normals) != len(offsets) or len(normals) != len(tangent_bounds):
        raise ValueError("plane arrays and tangent bounds must have equal lengths")

    point_count = len(points)
    plane_count = len(normals)
    eligibility = np.zeros((point_count, plane_count), dtype=bool)
    best_distance = np.full(point_count, np.inf, dtype=np.float64)
    best_source_plane = np.full(point_count, -1, dtype=np.int32)
    candidate_before = np.zeros(plane_count, dtype=np.int64)
    candidate_after = np.zeros(plane_count, dtype=np.int64)

    for plane_id, bounds in enumerate(tangent_bounds):
        if bounds is None:
            continue
        uv = (points - bounds["origin"]) @ bounds["axes"].T
        inside = np.all(
            (uv >= bounds["lower"] - float(bound_margin))
            & (uv <= bounds["upper"] + float(bound_margin)),
            axis=1,
        )
        residual = np.abs(points @ normals[plane_id] + offsets[plane_id])
        eligible = inside & (residual <= float(distance_threshold))
        eligibility[:, plane_id] = eligible
        candidate_before[plane_id] = int(inside.sum())
        candidate_after[plane_id] = int(eligible.sum())
        # Strictly-less update preserves the lower plane ID on exact ties.
        better = eligible & (residual < best_distance)
        best_distance[better] = residual[better]
        best_source_plane[better] = plane_id

    counts_before_drop = np.asarray(
        [(best_source_plane == plane_id).sum() for plane_id in range(plane_count)],
        dtype=np.int64,
    )
    active_source_plane_ids = np.asarray(
        [
            plane_id
            for plane_id, count in enumerate(counts_before_drop)
            if int(count) >= int(min_points_per_plane)
        ],
        dtype=np.int32,
    )
    active_mask = np.zeros(plane_count, dtype=bool)
    active_mask[active_source_plane_ids] = True
    dropped_owner = (best_source_plane >= 0) & ~active_mask[
        np.maximum(best_source_plane, 0)
    ]
    best_source_plane[dropped_owner] = -1

    compact_assignment = np.full(point_count, -1, dtype=np.int32)
    for compact_id, source_id in enumerate(active_source_plane_ids):
        compact_assignment[best_source_plane == source_id] = compact_id
    compact_counts = np.asarray(
        [(compact_assignment == plane_id).sum() for plane_id in range(len(active_source_plane_ids))],
        dtype=np.int32,
    )
    multiplicity = eligibility.sum(axis=1).astype(np.int32)
    return {
        "assignment": compact_assignment,
        "assignment_source_plane_ids": best_source_plane,
        "active_source_plane_ids": active_source_plane_ids,
        "plane_normals": normals[active_source_plane_ids].astype(np.float32),
        "plane_offsets": offsets[active_source_plane_ids].astype(np.float32),
        "plane_point_counts": compact_counts,
        "eligibility": eligibility,
        "multiplicity": multiplicity,
        "candidate_count_before_distance": candidate_before,
        "candidate_count_after_distance": candidate_after,
        "refit_call_count": 0,
    }


def _load_e0_contract(
    e0_v1_dir: Path, e0_v2_dir: Path
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    expected_hash_rows = json.loads(
        (e0_v1_dir / "input_hashes.json").read_text(encoding="utf-8")
    )
    expected_hashes = {row["scene"]: row for row in expected_hash_rows}
    e0_summary = json.loads(
        (e0_v2_dir / "summary.json").read_text(encoding="utf-8")
    )
    e0_scenes = {row["scene"]: row for row in e0_summary["scenes"]}
    residual_rows = json.loads(
        (e0_v1_dir / "residual_scale.json").read_text(encoding="utf-8")
    )
    residual_by_key = {
        (row["scene"], int(row["gt_plane_id"])): row for row in residual_rows
    }
    return expected_hashes, e0_scenes, residual_by_key


def _hash_parameters() -> str:
    payload = {
        "distance_threshold": DISTANCE_THRESHOLD,
        "bound_margin": BOUND_MARGIN,
        "bound_trim_quantile": BOUND_TRIM_QUANTILE,
        "min_points_per_plane": MIN_POINTS_PER_PLANE,
        "match_iou": MATCH_IOU,
        "fragmentation_iou": FRAGMENTATION_IOU,
        "min_observed_plane_points": MIN_OBSERVED_PLANE_POINTS,
        "conflict_rule": CONFLICT_RULE,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _failure_conditions(
    row: dict[str, Any],
    e0: dict[str, Any],
    minimum_seed_retention: float,
) -> list[str]:
    checks = (
        ("plane_f1_below_0.95", row["plane_f1"] < 0.95),
        (
            "true_positive_plane_count_changed",
            row["true_positive_planes"] != e0["true_positive_planes"],
        ),
        (
            "gt_support_assignment_rate_below_0.999",
            row["gt_support_assignment_rate"] < 0.999,
        ),
        (
            "gt_support_matched_coverage_below_0.98",
            row["gt_support_matched_coverage"] < 0.98,
        ),
        (
            "matched_iou_median_below_0.90",
            row["plane_weighted_matched_iou_median"] < 0.90,
        ),
        ("minimum_matched_iou_below_0.80", row["minimum_matched_iou"] < 0.80),
        (
            "fragmentation_excess_worse_than_E0",
            row["fragmentation_excess"] > e0["fragmentation_excess"],
        ),
        (
            "overmerge_excess_worse_than_E0",
            row["overmerge_excess"] > e0["overmerge_excess"],
        ),
        ("seed_removed", minimum_seed_retention < 1.0),
        (
            "nonplane_false_assignment_rate_above_0.01",
            row["nonplane_false_assignment_rate"] is not None
            and row["nonplane_false_assignment_rate"] > 0.01,
        ),
        (
            "other_GT_added_fraction_above_0.05",
            row["added_other_GT_plane_fraction"] is not None
            and row["added_other_GT_plane_fraction"] > 0.05,
        ),
        (
            "multi_candidate_fraction_above_0.01",
            row["multi_candidate_fraction_of_gt_plane_points"] > 0.01,
        ),
        (
            "plane_point_weighted_directions_opposite",
            not row["plane_point_weighted_direction_consistent"],
        ),
    )
    return [name for name, failed in checks if failed]


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    manifest_path = Path(args.manifest_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    e0_v1_dir = Path(args.e0_v1_dir).resolve()
    e0_v2_dir = Path(args.e0_v2_dir).resolve()
    literature_path = (
        root / "docs/literature/LITERATURE_GATE_tangent_bbox_e1_20260727.md"
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir()

    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    expected_hashes, e0_scenes, residual_by_key = _load_e0_contract(
        e0_v1_dir, e0_v2_dir
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested = set(args.pressure_scenes)
    entries = [
        scene for scene in manifest["scenes"] if scene["scene"] in requested
    ]
    missing = sorted(requested - {scene["scene"] for scene in entries})
    if missing:
        raise ValueError(f"pressure scenes missing from manifest: {missing}")

    implementation_path = Path(__file__).resolve()
    source_bbox_path = root / "evaluate_all_fragment_common_cache.py"
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "config": "E1",
        "support": "exact unique point-aligned GT plane support from E0",
        "join": "same exact (view,x,y) unique/non-ambiguous join as E0",
        "expansion": "current repository tangent_bbox, one assignment pass",
        "refit": "completely disabled",
        "refit_call_count": 0,
        "weighted_fit_plane_called": False,
        "initial_plane_parameters": "point-aligned GT NPZ; no recomputation",
        "distance_threshold": DISTANCE_THRESHOLD,
        "bound_margin": BOUND_MARGIN,
        "bound_trim_quantile": BOUND_TRIM_QUANTILE,
        "min_points_per_plane": MIN_POINTS_PER_PLANE,
        "conflict_resolution_rule": CONFLICT_RULE,
        "bounded_matcher": "full-cache IoU + Hungarian",
        "plane_tp_iou": MATCH_IOU,
        "fragmentation_iou": FRAGMENTATION_IOU,
        "min_observed_plane_points": MIN_OBSERVED_PLANE_POINTS,
        "pressure_scenes": list(args.pressure_scenes),
        "pressure_scenes_used_for_tuning": False,
        "training_calibration_performed": False,
        "e2_e3_executed": False,
        "l_g_j_resumed": False,
        "parameter_contract_sha256": _hash_parameters(),
        "e1_implementation_sha256": sha256_file(implementation_path),
        "source_bbox_implementation_path": str(source_bbox_path),
        "source_bbox_implementation_sha256": sha256_file(source_bbox_path),
        "manifest_json": str(manifest_path),
        "e0_v1_dir": str(e0_v1_dir),
        "e0_v2_dir": str(e0_v2_dir),
    }
    (output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    state = git_state(root)
    state["environment"] = environment_state()
    (output_dir / "git_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    (output_dir / "literature_gate.md").write_text(
        literature_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    plane_table: list[dict[str, Any]] = []
    scene_table: list[dict[str, Any]] = []
    multi_table: list[dict[str, Any]] = []
    residual_table: list[dict[str, Any]] = []
    input_hash_table: list[dict[str, Any]] = []

    for entry in entries:
        scene = entry["scene"]
        if scene not in expected_hashes or scene not in e0_scenes:
            raise ValueError(f"{scene}: missing E0 hash or metric baseline")
        data = load_scene(entry, root)
        gt = data["gt"]
        cache = data["cache"]
        points = np.asarray(gt["points"], dtype=np.float64)
        gt_labels = np.asarray(gt["point_plane_ids"], dtype=np.int32)
        gt_normals = np.asarray(gt["plane_normals"], dtype=np.float64)
        gt_offsets = np.asarray(gt["plane_offsets"], dtype=np.float64)

        gt_hash = sha256_file(data["gt_path"])
        cache_hash = sha256_file(data["cache_path"])
        expected = expected_hashes[scene]
        gt_hash_equal = gt_hash == expected["point_aligned_gt"]["sha256"]
        cache_hash_equal = cache_hash == expected["global_cache"]["sha256"]
        if not gt_hash_equal or not cache_hash_equal:
            raise ValueError(
                f"{scene}: E0 input hash mismatch "
                f"(gt={gt_hash_equal}, cache={cache_hash_equal})"
            )

        seed_assignment, join_rows = exact_unique_join(
            cache["view_indices"],
            cache["pixel_xy"],
            gt["view_indices"],
            gt["pixel_xy"],
            gt_labels,
            scene=scene,
            variant="E1_recomputed_E0_exact_point_aligned_gt",
        )
        point_index_equal = bool(
            np.array_equal(cache["view_indices"], gt["view_indices"])
            and np.array_equal(cache["pixel_xy"], gt["pixel_xy"])
            and np.allclose(cache["points"], gt["points"], atol=1e-5)
        )
        support_equal = bool(np.array_equal(seed_assignment, gt_labels))
        if not point_index_equal or not support_equal:
            raise ValueError(
                f"{scene}: E0 point index/support could not be reproduced"
            )
        aggregate_join = join_rows[-1]

        bounds = build_tangent_bounds(
            points,
            seed_assignment,
            gt_normals,
            trim_quantile=BOUND_TRIM_QUANTILE,
        )
        expanded = expand_tangent_bbox_once_no_refit(
            points,
            seed_assignment,
            gt_normals,
            gt_offsets,
            bounds,
            distance_threshold=DISTANCE_THRESHOLD,
            bound_margin=BOUND_MARGIN,
            min_points_per_plane=MIN_POINTS_PER_PLANE,
        )
        pred_assignment = expanded["assignment"]
        assignment_source = expanded["assignment_source_plane_ids"]
        active_source_ids = expanded["active_source_plane_ids"]
        pred_normals = expanded["plane_normals"]
        pred_offsets = expanded["plane_offsets"]
        multiplicity = expanded["multiplicity"]
        eligibility = expanded["eligibility"]

        metrics = evaluate_arrays(
            points,
            pred_assignment,
            pred_normals,
            pred_offsets,
            gt_labels,
            gt_normals,
            MATCH_IOU,
            FRAGMENTATION_IOU,
            MIN_OBSERVED_PLANE_POINTS,
        )
        precision = float(metrics["plane_precision"])
        recall = float(metrics["plane_recall"])
        plane_f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        match_table, match_summary = matching_rows(
            scene,
            points,
            pred_assignment,
            pred_normals,
            pred_offsets,
            gt_labels,
            gt_normals,
        )
        match_by_gt = {
            int(row["gt_plane_id"]): row
            for row in match_table
            if int(row["gt_plane_id"]) >= 0
        }

        plane_rows_for_scene: list[dict[str, Any]] = []
        for gt_plane_id in range(len(gt_normals)):
            seed_mask = seed_assignment == gt_plane_id
            final_mask = assignment_source == gt_plane_id
            added_mask = final_mask & ~seed_mask
            removed_mask = seed_mask & ~final_mask
            bounds_row = bounds[gt_plane_id]
            if bounds_row is None:
                min_u = max_u = min_v = max_v = None
                bbox_area = None
                hull_area = None
                hull_status = "not_computable: tangent_bounds_unavailable"
            else:
                lower = np.asarray(bounds_row["lower"], dtype=np.float64)
                upper = np.asarray(bounds_row["upper"], dtype=np.float64)
                min_u, min_v = map(float, lower)
                max_u, max_v = map(float, upper)
                bbox_area = float(np.prod(np.maximum(upper - lower, 0.0)))
                seed_uv = (
                    points[seed_mask] - bounds_row["origin"]
                ) @ bounds_row["axes"].T
                hull_area, hull_status = _convex_hull_area(seed_uv)
            bbox_hull_ratio = (
                _safe_ratio(bbox_area, hull_area)
                if bbox_area is not None and hull_area is not None
                else None
            )

            seed_count = int(seed_mask.sum())
            added_count = int(added_mask.sum())
            removed_count = int(removed_mask.sum())
            added_gt = gt_labels[added_mask]
            same_count = int((added_gt == gt_plane_id).sum())
            other_mask = (added_gt >= 0) & (added_gt != gt_plane_id)
            other_count = int(other_mask.sum())
            unlabeled_count = int((added_gt < 0).sum())
            wrong_ids, wrong_counts = np.unique(
                added_gt[other_mask], return_counts=True
            )
            dominant_wrong = (
                int(wrong_ids[np.argmax(wrong_counts)]) if len(wrong_ids) else None
            )
            added_residual = np.abs(
                points[added_mask] @ gt_normals[gt_plane_id]
                + gt_offsets[gt_plane_id]
            )
            eligible_for_plane = eligibility[:, gt_plane_id]
            multi_for_plane = eligible_for_plane & (multiplicity > 1)
            e1_match = match_by_gt.get(gt_plane_id)
            e1_iou = float(e1_match["matched_iou"]) if e1_match else 0.0
            e1_tp = bool(e1_match and e1_match["plane_tp"])
            matched_prediction_id = (
                int(e1_match["pred_plane_id"]) if e1_match else None
            )
            residual_scale = residual_by_key[(scene, gt_plane_id)]
            row = {
                "scene_id": scene,
                "gt_plane_id": gt_plane_id,
                "matched_prediction_id": matched_prediction_id,
                "E0_TP": True,
                "E1_TP": e1_tp,
                "seed_unique_point_count": seed_count,
                "seed_view_count": int(
                    len(np.unique(gt["view_indices"][seed_mask]))
                ),
                "tangent_bbox_min_u": min_u,
                "tangent_bbox_max_u": max_u,
                "tangent_bbox_min_v": min_v,
                "tangent_bbox_max_v": max_v,
                "tangent_bbox_area": bbox_area,
                "seed_convex_hull_area": hull_area,
                "convex_hull_status": hull_status,
                "bbox_to_hull_area_ratio": bbox_hull_ratio,
                "candidate_count_before_plane_distance_filter": int(
                    expanded["candidate_count_before_distance"][gt_plane_id]
                ),
                "candidate_count_after_plane_distance_filter": int(
                    expanded["candidate_count_after_distance"][gt_plane_id]
                ),
                "expanded_unique_point_count": int(final_mask.sum()),
                "added_point_count": added_count,
                "added_point_fraction_vs_seed": _safe_ratio(
                    added_count, seed_count
                ),
                "removed_seed_point_count": removed_count,
                "seed_retention_rate": _safe_ratio(
                    seed_count - removed_count, seed_count
                ),
                "added_same_GT_plane_count": same_count,
                "added_other_GT_plane_count": other_count,
                "added_GT_unlabeled_count": unlabeled_count,
                "added_same_GT_plane_fraction": (
                    _safe_ratio(same_count, added_count) if added_count else None
                ),
                "added_other_GT_plane_fraction": (
                    _safe_ratio(other_count, added_count) if added_count else None
                ),
                "added_unlabeled_fraction": (
                    _safe_ratio(unlabeled_count, added_count)
                    if added_count
                    else None
                ),
                "added_dominant_wrong_GT_id": dominant_wrong,
                "expanded_points_already_owned_by_other_plane": int(
                    (
                        eligible_for_plane
                        & (gt_labels >= 0)
                        & (gt_labels != gt_plane_id)
                    ).sum()
                ),
                "points_assigned_to_multiple_expanded_planes": int(
                    multi_for_plane.sum()
                ),
                "maximum_plane_multiplicity": int(
                    multiplicity[eligible_for_plane].max()
                )
                if eligible_for_plane.any()
                else 0,
                "raw_plane_distance_threshold": DISTANCE_THRESHOLD,
                "scene_bbox_diagonal": float(
                    residual_scale["scene_bbox_diagonal"]
                ),
                "support_bbox_diagonal": float(
                    residual_scale["support_xyz_bbox_diagonal"]
                ),
                "seed_median_8NN_distance": float(
                    residual_scale["median_8nn_distance"]
                ),
                "threshold_over_scene_bbox_diagonal": _safe_ratio(
                    DISTANCE_THRESHOLD,
                    residual_scale["scene_bbox_diagonal"],
                ),
                "threshold_over_support_bbox_diagonal": _safe_ratio(
                    DISTANCE_THRESHOLD,
                    residual_scale["support_xyz_bbox_diagonal"],
                ),
                "threshold_over_seed_median_8NN_distance": _safe_ratio(
                    DISTANCE_THRESHOLD,
                    residual_scale["median_8nn_distance"],
                ),
                "added_residual_median": _percentile(added_residual, 0.5),
                "added_residual_p90": _percentile(added_residual, 0.9),
                "added_residual_p95": _percentile(added_residual, 0.95),
                "added_residual_max": (
                    float(np.max(added_residual)) if len(added_residual) else None
                ),
                "E0_matched_IoU": 1.0,
                "E1_matched_IoU": e1_iou,
                "matched_IoU_delta": e1_iou - 1.0,
                "E0_plane_TP": True,
                "E1_plane_TP": e1_tp,
            }
            plane_rows_for_scene.append(row)
            residual_row = {
                field: row.get(field) for field in RESIDUAL_FIELDS
            }
            residual_row["scene"] = row["scene_id"]
            residual_table.append(residual_row)
        plane_table.extend(plane_rows_for_scene)

        gt_support = gt_labels >= 0
        assigned = pred_assignment >= 0
        gt_point_count = int(gt_support.sum())
        assigned_count = int(assigned.sum())
        gt_assigned_count = int((assigned & gt_support).sum())
        matched_count = int(match_summary["gt_support_matched_point_count"])
        nonplane_count = int((~gt_support).sum())
        nonplane_false_count = int((assigned & ~gt_support).sum())
        added_mask_scene = assigned & (
            assignment_source != seed_assignment
        )
        added_count_scene = int(added_mask_scene.sum())
        added_other_scene = int(
            (
                added_mask_scene
                & (gt_labels >= 0)
                & (gt_labels != assignment_source)
            ).sum()
        )
        multi_count = int((multiplicity > 1).sum())
        minimum_seed_retention = float(
            min(row["seed_retention_rate"] for row in plane_rows_for_scene)
        )
        all_gt_iou = np.asarray(
            [row["E1_matched_IoU"] for row in plane_rows_for_scene],
            dtype=np.float64,
        )
        # Use all GT planes, including unmatched planes at IoU 0.
        match_summary["minimum_matched_iou"] = float(np.min(all_gt_iou))
        e0 = e0_scenes[scene]
        plane_delta = (
            float(match_summary["plane_weighted_matched_iou_mean"])
            - float(e0["plane_weighted_matched_iou_mean"])
        )
        point_delta = (
            float(match_summary["point_weighted_matched_iou"])
            - float(e0["point_weighted_matched_iou"])
        )
        scene_row = {
            "scene": scene,
            "config": "E1",
            "cache_point_count": int(len(gt_labels)),
            "assigned_cache_point_count": assigned_count,
            "gt_plane_point_count": gt_point_count,
            "gt_support_assigned_point_count": gt_assigned_count,
            "gt_support_matched_point_count": matched_count,
            "nonplane_point_count": nonplane_count,
            "nonplane_false_assigned_point_count": nonplane_false_count,
            "pred_plane_count": int(metrics["pred_plane_count"]),
            "gt_plane_count": int(metrics["gt_plane_count"]),
            "true_positive_planes": int(metrics["true_positive_planes"]),
            "plane_precision": precision,
            "plane_recall": recall,
            "plane_f1": plane_f1,
            **match_summary,
            "gt_support_assignment_rate": _safe_ratio(
                gt_assigned_count, gt_point_count
            ),
            "gt_support_matched_coverage": _safe_ratio(
                matched_count, gt_point_count
            ),
            "full_cache_label_rate": _safe_ratio(
                assigned_count, len(gt_labels)
            ),
            "gt_plane_label_rate": _safe_ratio(
                gt_point_count, len(gt_labels)
            ),
            "nonplane_false_assignment_rate": (
                _safe_ratio(nonplane_false_count, nonplane_count)
                if nonplane_count
                else None
            ),
            "fragmentation_excess": int(metrics["fragmentation_excess"]),
            "overmerge_excess": int(metrics["overmerge_excess"]),
            "added_point_count": added_count_scene,
            "added_other_GT_plane_count": added_other_scene,
            "added_other_GT_plane_fraction": (
                _safe_ratio(added_other_scene, added_count_scene)
                if added_count_scene
                else None
            ),
            "multi_candidate_point_count": multi_count,
            "multi_candidate_fraction_of_gt_plane_points": _safe_ratio(
                multi_count, gt_point_count
            ),
            "maximum_plane_multiplicity": int(multiplicity.max())
            if len(multiplicity)
            else 0,
            "plane_weighted_iou_delta_from_E0": plane_delta,
            "point_weighted_iou_delta_from_E0": point_delta,
            "plane_point_weighted_direction_consistent": _directions_consistent(
                plane_delta, point_delta
            ),
            "minimum_seed_retention_rate": minimum_seed_retention,
        }
        failures = _failure_conditions(
            scene_row, e0, minimum_seed_retention
        )
        scene_row["failed_conditions"] = ";".join(failures)
        scene_row["e1_all_pass"] = not failures
        scene_table.append(scene_row)

        multi_table.append(
            {
                "scene": scene,
                "candidate_membership_count": int(eligibility.sum()),
                "unique_candidate_point_count": int((multiplicity > 0).sum()),
                "points_assigned_to_multiple_expanded_planes": multi_count,
                "multi_candidate_fraction_of_GT_plane_points": _safe_ratio(
                    multi_count, gt_point_count
                ),
                "maximum_plane_multiplicity": int(multiplicity.max())
                if len(multiplicity)
                else 0,
                "conflict_resolution_rule": CONFLICT_RULE,
            }
        )
        input_hash_table.append(
            {
                "scene": scene,
                "point_aligned_gt": {
                    "path": str(data["gt_path"]),
                    "sha256": gt_hash,
                    "E0_sha256": expected["point_aligned_gt"]["sha256"],
                    "matches_E0": gt_hash_equal,
                },
                "global_cache": {
                    "path": str(data["cache_path"]),
                    "sha256": cache_hash,
                    "E0_sha256": expected["global_cache"]["sha256"],
                    "matches_E0": cache_hash_equal,
                },
                "point_index_equal": point_index_equal,
                "exact_support_equal": support_equal,
                "join_unique_input_key_count": int(
                    aggregate_join["unique_input_key_count"]
                ),
                "join_matched_unique_global_point_count": int(
                    aggregate_join["matched_unique_global_point_count"]
                ),
                "join_ambiguous_key_count": int(
                    aggregate_join["ambiguous_key_count"]
                ),
                "join_unmatched_key_count": int(
                    aggregate_join["unmatched_key_count"]
                ),
            }
        )
        prediction_path = prediction_dir / f"{scene}_e1_tangent_bbox_no_refit.npz"
        np.savez_compressed(
            prediction_path,
            schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
            method=np.asarray("E1_tangent_bbox_expansion_no_refit"),
            points=gt["points"].astype(np.float32),
            point_plane_ids=pred_assignment.astype(np.int32),
            plane_normals=pred_normals.astype(np.float32),
            plane_offsets=pred_offsets.astype(np.float32),
            plane_point_counts=expanded["plane_point_counts"].astype(np.int32),
            source_gt_plane_ids=active_source_ids.astype(np.int32),
            refit_call_count=np.asarray(0, dtype=np.int32),
        )
        log(
            f"{scene}: F1={plane_f1:.6f} "
            f"TP={metrics['true_positive_planes']}/{metrics['gt_plane_count']} "
            f"IoU_med={match_summary['plane_weighted_matched_iou_median']:.6f} "
            f"IoU_min={match_summary['minimum_matched_iou']:.6f} "
            f"GT_assign={scene_row['gt_support_assignment_rate']:.6f} "
            f"matched_coverage={scene_row['gt_support_matched_coverage']:.6f} "
            f"nonplane_false={scene_row['nonplane_false_assignment_rate']:.6f} "
            f"seed_retention_min={minimum_seed_retention:.6f} "
            f"multi_fraction={scene_row['multi_candidate_fraction_of_gt_plane_points']:.6f} "
            f"pass={scene_row['e1_all_pass']} "
            f"failures={scene_row['failed_conditions'] or 'none'}"
        )

    all_pass = bool(all(row["e1_all_pass"] for row in scene_table))
    verdict = "Passed" if all_pass else "Failed"
    route = "E1 contract-safe" if all_pass else "E-Bounded Expansion"
    next_action = (
        "Request a preregistration for a refit-only safety experiment on the "
        "frozen E1 support; do not run it in this round."
        if all_pass
        else (
            "Keep E0 as the trusted exact-support baseline, mark the current "
            "tangent-bbox expansion unsafe and disabled by default for formal "
            "results, and request an E2 candidate bounded-footprint "
            "preregistration. Do not implement or run E2 in this round."
        )
    )
    log(f"contract_verdict={verdict}")
    log(f"route={route}")
    log(f"refit_call_count=0")
    log(f"next_mechanical_action={next_action}")

    _write_table(
        output_dir,
        "per_plane_expansion_audit",
        plane_table,
        PER_PLANE_FIELDS,
    )
    _write_table(output_dir, "per_scene_metrics", scene_table, SCENE_FIELDS)
    _write_table(
        output_dir, "multi_assignment_audit", multi_table, MULTI_FIELDS
    )
    _write_table(
        output_dir, "residual_scale_audit", residual_table, RESIDUAL_FIELDS
    )
    (output_dir / "input_hashes.json").write_text(
        json.dumps(input_hash_table, indent=2), encoding="utf-8"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config": "E1",
        "contract_verdict": verdict,
        "route": route,
        "all_pressure_scenes_passed": all_pass,
        "refit_call_count": 0,
        "weighted_fit_plane_called": False,
        "e2_e3_executed": False,
        "l_g_j_resumed": False,
        "next_mechanical_action": next_action,
        "scenes": scene_table,
        "failed_planes": [
            {
                "scene": row["scene_id"],
                "gt_plane_id": row["gt_plane_id"],
                "seed_retention_rate": row["seed_retention_rate"],
                "E1_matched_IoU": row["E1_matched_IoU"],
                "added_other_GT_plane_fraction": row[
                    "added_other_GT_plane_fraction"
                ],
            }
            for row in plane_table
            if row["seed_retention_rate"] < 1.0
            or row["E1_matched_IoU"] < 0.80
            or (
                row["added_other_GT_plane_fraction"] is not None
                and row["added_other_GT_plane_fraction"] > 0.05
            )
        ],
        "artifacts": {
            "config_json": str(output_dir / "config.json"),
            "git_state_json": str(output_dir / "git_state.json"),
            "input_hashes_json": str(output_dir / "input_hashes.json"),
            "literature_gate_md": str(output_dir / "literature_gate.md"),
            "per_plane_expansion_audit_csv": str(
                output_dir / "per_plane_expansion_audit.csv"
            ),
            "per_scene_metrics_csv": str(output_dir / "per_scene_metrics.csv"),
            "multi_assignment_audit_csv": str(
                output_dir / "multi_assignment_audit.csv"
            ),
            "residual_scale_audit_csv": str(
                output_dir / "residual_scale_audit.csv"
            ),
            "prediction_dir": str(prediction_dir),
            "run_log": str(output_dir / "run.log"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output_dir / "run.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest_json",
        default=str(
            root
            / "local_outputs/intra_fragment_split_manifest_17scene_20260725_v2/manifest.json"
        ),
    )
    parser.add_argument(
        "--e0_v1_dir",
        default=str(root / "results/evaluator_contract_e0_20260727_v1"),
    )
    parser.add_argument(
        "--e0_v2_dir",
        default=str(
            root / "results/evaluator_contract_e0_20260727_v2_metric_contract"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(root / "results/evaluator_contract_e1_20260727_v1"),
    )
    parser.add_argument(
        "--pressure_scenes", nargs="+", default=list(PRESSURE_SCENES)
    )
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
