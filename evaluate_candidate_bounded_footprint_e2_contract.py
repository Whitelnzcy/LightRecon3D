#!/usr/bin/env python3
"""Run the preregistered E2 candidate bounded-footprint safety audit.

The prediction phase consumes only the frozen E0 full-cache points, exact seed
assignment, and plane parameters. All predictions are written and hashed
before the GT scoring phase begins. No fit, refit, RANSAC, training, or E1
prediction path is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, QhullError, cKDTree

from audit_support_bounded_evaluator_contract import (
    PRESSURE_SCENES,
    _safe_ratio,
    environment_state,
    exact_unique_join,
    git_state,
    load_scene,
    matching_rows,
    sha256_file,
)
from evaluate_global_plane_baselines import evaluate_arrays


REGISTRATION_ID = "E2_PREREG_candidate_bounded_footprint_20260727_v1"
SCHEMA_VERSION = 1
BASE_GIT_SHA = "1fa7027f16630069d3475e0d012affea25e91e81"
MINIMUM_SEED_POINTS = 1000
MAX_RAW_DISTANCE = 0.03
MATCH_IOU = 0.5
FRAGMENTATION_IOU = 0.1
MIN_OBSERVED_PLANE_POINTS = 64

PLANE_FIT_CALL_COUNT = 0
WEIGHTED_FIT_PLANE_CALL_COUNT = 0
REFIT_CALL_COUNT = 0
RANSAC_CALL_COUNT = 0

E2_MATCHER = matching_rows
E2_METRIC = evaluate_arrays

PER_PLANE_FIELDS = (
    "scene",
    "plane_id",
    "seed_count",
    "unique_projected_seed_count",
    "footprint_status",
    "local_scale_s",
    "normal_threshold_tau",
    "seed_residual_median",
    "seed_residual_mad",
    "seed_residual_q99",
    "triangle_count_total",
    "triangle_count_accepted",
    "triangle_area_sum",
    "disk_count",
    "footprint_component_count",
    "raw_candidate_count",
    "accepted_unique_candidate_count",
    "conflict_rejected_count",
    "cross_seed_intrusion_count",
    "added_count",
    "added_over_seed_count",
    "seed_retention",
    "matched_GT_plane_id",
    "E0_matched_IoU",
    "E2_matched_IoU",
    "matched_IoU_delta",
    "wrong_GT_added_count",
    "nonplane_added_count",
    "total_leakage_count",
    "wrong_GT_added_fraction",
    "nonplane_added_fraction",
    "total_leakage_fraction",
    "leakage_over_target_GT_count",
    "plane_verdict",
    "failure_reasons",
)

SCENE_FIELDS = (
    "scene",
    "scene_verdict",
    "failure_reasons",
    "cache_point_count",
    "E0_plane_count",
    "E2_plane_count",
    "plane_ids_unchanged",
    "global_seed_retention",
    "seed_deletion_count",
    "seed_reassignment_count",
    "raw_multi_candidate_points",
    "raw_multi_candidate_over_GT_plane_points",
    "raw_multi_candidate_over_full_cache",
    "maximum_multiplicity",
    "final_multi_assignment_count",
    "cross_seed_intrusion_count",
    "total_added_count",
    "total_added_over_E0_seed_count",
    "total_wrong_GT_added",
    "total_nonplane_added",
    "total_leakage",
    "total_wrong_GT_added_fraction",
    "total_leakage_fraction",
    "total_leakage_over_GT_plane_points",
    "E0_plane_f1",
    "E2_plane_f1",
    "plane_f1_delta",
    "E0_true_positive_planes",
    "E2_true_positive_planes",
    "E0_iou_median",
    "E2_iou_median",
    "iou_median_delta",
    "E0_minimum_iou",
    "E2_minimum_iou",
    "minimum_iou_delta",
    "E0_gt_support_assignment_rate",
    "E2_gt_support_assignment_rate",
    "E0_gt_support_matched_coverage",
    "E2_gt_support_matched_coverage",
    "E0_full_cache_label_rate",
    "E2_full_cache_label_rate",
    "E0_gt_plane_label_rate",
    "E2_gt_plane_label_rate",
    "E0_nonplane_false_assignment_rate",
    "E2_nonplane_false_assignment_rate",
    "E0_fragmentation_count",
    "E2_fragmentation_count",
    "E0_oversegmentation_count",
    "E2_oversegmentation_count",
)

MULTI_FIELDS = (
    "scene",
    "raw_multi_candidate_points",
    "raw_multi_candidate_over_GT_plane_points",
    "raw_multi_candidate_over_full_cache",
    "maximum_multiplicity",
    "final_multi_assignment_count",
    "multi_candidate_point_indices_count",
    "per_plane_conflict_participation_json",
    "conflict_rule",
)

SEED_FIELDS = (
    "scene",
    "plane_id",
    "seed_count",
    "retained_seed_count",
    "seed_retention",
    "seed_deletion_count",
    "seed_reassignment_count",
    "cross_seed_intrusion_count",
)

LEAKAGE_FIELDS = (
    "scene",
    "plane_id",
    "matched_GT_plane_id",
    "added_count",
    "wrong_GT_added_count",
    "nonplane_added_count",
    "total_leakage_count",
    "wrong_GT_added_fraction",
    "nonplane_added_fraction",
    "total_leakage_fraction",
    "leakage_over_target_GT_count",
)


@dataclass
class FreezeBarrier:
    required_scene_count: int
    frozen_prediction_hashes: dict[str, str]
    prediction_frozen_before_gt_scoring: bool = False

    def record_prediction(self, scene: str, sha256: str) -> None:
        if self.prediction_frozen_before_gt_scoring:
            raise RuntimeError("cannot add prediction after GT scoring barrier")
        self.frozen_prediction_hashes[scene] = sha256

    def freeze_all(self) -> None:
        if len(self.frozen_prediction_hashes) != self.required_scene_count:
            raise RuntimeError("not all predictions were written and hashed")
        self.prediction_frozen_before_gt_scoring = True

    def assert_gt_scoring_allowed(self) -> None:
        if not self.prediction_frozen_before_gt_scoring:
            raise RuntimeError("GT scoring requested before prediction freeze")


def _write_table(
    output_dir: Path,
    stem: str,
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
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


def _function_sha256(function: Any) -> str:
    return hashlib.sha256(
        inspect.getsource(function).encode("utf-8")
    ).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def deterministic_tangent_basis(
    normal: np.ndarray,
) -> tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if not np.all(np.isfinite(normal)) or not math.isfinite(norm) or norm < 1e-12:
        return "degenerate_normal", None, None, None
    n = normal / norm
    global_axes = np.eye(3, dtype=np.float64)
    reference_index = int(np.argmin(np.abs(global_axes @ n)))
    reference = global_axes[reference_index]
    u = reference - float(reference @ n) * n
    u_norm = float(np.linalg.norm(u))
    if not math.isfinite(u_norm) or u_norm < 1e-12:
        return "degenerate_normal", None, None, None
    u = u / u_norm
    v = np.cross(n, u)
    dominant_index = int(np.argmax(np.abs(u)))
    if u[dominant_index] < 0:
        u = -u
        v = -v
    return "valid", n, u, v


def unique_seed_projection(
    points: np.ndarray,
    seed_indices: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    seed_indices = np.asarray(seed_indices, dtype=np.int64)
    projected = np.column_stack(
        (points[seed_indices] @ u, points[seed_indices] @ v)
    ).astype(np.float64, copy=False)
    unique_projected, first_positions = np.unique(
        projected, axis=0, return_index=True
    )
    original_indices = seed_indices[first_positions]
    order = np.lexsort(
        (original_indices, unique_projected[:, 1], unique_projected[:, 0])
    )
    return unique_projected[order], original_indices[order]


def eighth_neighbor_scale(
    projected: np.ndarray,
) -> tuple[str, cKDTree | None, np.ndarray | None, np.ndarray | None, float | None]:
    projected = np.asarray(projected, dtype=np.float64)
    if len(projected) < 9:
        return "insufficient_unique_seeds", None, None, None, None
    tree = cKDTree(projected)
    distances, indices = tree.query(projected, k=9, workers=-1)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    d8 = distances[:, 8]
    valid = np.isfinite(d8) & (d8 > 0)
    if not valid.any():
        return "invalid_local_scale", tree, d8, indices, None
    local_scale = float(np.median(d8[valid]))
    if not math.isfinite(local_scale) or local_scale <= 0:
        return "invalid_local_scale", tree, d8, indices, None
    return "valid", tree, d8, indices, local_scale


def local_disk_radii(d8: np.ndarray, local_scale: float) -> np.ndarray:
    return np.clip(
        1.5 * np.asarray(d8, dtype=np.float64),
        0.5 * float(local_scale),
        2.0 * float(local_scale),
    )


def adaptive_normal_threshold(
    seed_points: np.ndarray,
    normal: np.ndarray,
    offset: float,
    local_scale: float,
) -> dict[str, float | None]:
    residual = np.abs(
        np.asarray(seed_points, dtype=np.float64) @ normal + float(offset)
    )
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    q99 = float(np.quantile(residual, 0.99, method="linear"))
    sigma_mad = 1.4826 * mad
    tau_noise = max(1e-6, q99 + 3.0 * sigma_mad)
    tau = min(MAX_RAW_DISTANCE, 0.5 * float(local_scale), tau_noise)
    return {
        "median": median,
        "mad": mad,
        "q99": q99,
        "sigma_mad": sigma_mad,
        "tau_noise": tau_noise,
        "tau": tau if math.isfinite(tau) and tau > 0 else None,
    }


def accepted_delaunay_triangles(
    projected: np.ndarray, local_scale: float
) -> dict[str, Any]:
    projected = np.asarray(projected, dtype=np.float64)
    try:
        triangulation = Delaunay(projected)
    except (QhullError, ValueError):
        return {
            "triangulation": None,
            "accepted": np.zeros(0, dtype=bool),
            "triangle_count_total": 0,
            "triangle_count_accepted": 0,
            "triangle_area_sum": 0.0,
            "status": "delaunay_empty:no_convex_hull_fallback",
        }
    simplices = np.asarray(triangulation.simplices, dtype=np.int64)
    p0 = projected[simplices[:, 0]]
    p1 = projected[simplices[:, 1]]
    p2 = projected[simplices[:, 2]]
    edge01 = np.linalg.norm(p0 - p1, axis=1)
    edge12 = np.linalg.norm(p1 - p2, axis=1)
    edge20 = np.linalg.norm(p2 - p0, axis=1)
    max_edge = np.maximum(np.maximum(edge01, edge12), edge20)
    double_area = np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )
    area = 0.5 * double_area
    circumradius = np.full(len(simplices), np.inf, dtype=np.float64)
    nondegenerate = area > 1e-12
    circumradius[nondegenerate] = (
        edge01[nondegenerate]
        * edge12[nondegenerate]
        * edge20[nondegenerate]
        / (4.0 * area[nondegenerate])
    )
    accepted = (
        (max_edge <= 3.0 * float(local_scale))
        & (circumradius <= 2.0 * float(local_scale))
        & nondegenerate
    )
    return {
        "triangulation": triangulation,
        "accepted": accepted,
        "triangle_count_total": int(len(simplices)),
        "triangle_count_accepted": int(accepted.sum()),
        "triangle_area_sum": float(area[accepted].sum()),
        "status": "delaunay_available",
    }


def _component_count_from_edges(
    point_count: int, rows: np.ndarray, columns: np.ndarray
) -> tuple[int, np.ndarray]:
    if point_count == 0:
        return 0, np.empty(0, dtype=np.int32)
    if len(rows) == 0:
        return point_count, np.arange(point_count, dtype=np.int32)
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(point_count, point_count),
    )
    count, labels = connected_components(
        graph, directed=False, return_labels=True
    )
    return int(count), labels.astype(np.int32)


def footprint_component_count(
    projected: np.ndarray,
    tree: cKDTree,
    disk_radii: np.ndarray,
    neighbor_indices: np.ndarray,
    triangulation: Delaunay | None,
    accepted_triangles: np.ndarray,
    local_scale: float,
) -> int:
    """Count exact union components, using a cheap connected subset first."""

    point_count = len(projected)
    rows = np.repeat(np.arange(point_count, dtype=np.int64), 8)
    columns = np.asarray(neighbor_indices[:, 1:9], dtype=np.int64).reshape(-1)
    distances = np.linalg.norm(projected[rows] - projected[columns], axis=1)
    overlap = distances <= disk_radii[rows] + disk_radii[columns]
    rows = rows[overlap]
    columns = columns[overlap]

    if triangulation is not None and accepted_triangles.any():
        simplices = triangulation.simplices[accepted_triangles]
        triangle_rows = np.concatenate(
            (simplices[:, 0], simplices[:, 1], simplices[:, 2])
        ).astype(np.int64)
        triangle_columns = np.concatenate(
            (simplices[:, 1], simplices[:, 2], simplices[:, 0])
        ).astype(np.int64)
        rows = np.concatenate((rows, triangle_rows))
        columns = np.concatenate((columns, triangle_columns))

    component_count, labels = _component_count_from_edges(
        point_count, rows, columns
    )
    if component_count <= 1:
        return component_count

    # Only needed for initially disconnected footprints. Any omitted disk
    # overlap can merge, but never split, these provisional components.
    pairs = tree.query_pairs(
        r=4.0 * float(local_scale), output_type="ndarray"
    )
    if not len(pairs):
        return component_count
    cross = labels[pairs[:, 0]] != labels[pairs[:, 1]]
    pairs = pairs[cross]
    if not len(pairs):
        return component_count
    pair_distances = np.linalg.norm(
        projected[pairs[:, 0]] - projected[pairs[:, 1]], axis=1
    )
    overlaps = pair_distances <= (
        disk_radii[pairs[:, 0]] + disk_radii[pairs[:, 1]]
    )
    pairs = pairs[overlaps]
    if not len(pairs):
        return component_count
    component_rows = labels[pairs[:, 0]].astype(np.int64)
    component_columns = labels[pairs[:, 1]].astype(np.int64)
    final_count, _ = _component_count_from_edges(
        component_count, component_rows, component_columns
    )
    return final_count


def points_in_variable_disks(
    query: np.ndarray,
    tree: cKDTree,
    disk_radii: np.ndarray,
    local_scale: float,
) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    if not len(query):
        return np.zeros(0, dtype=bool)
    nearest_distances, nearest_indices = tree.query(
        query, k=min(3, len(tree.data)), workers=-1
    )
    nearest_distances = np.asarray(nearest_distances, dtype=np.float64)
    nearest_indices = np.asarray(nearest_indices, dtype=np.int64)
    if nearest_distances.ndim == 1:
        nearest_distances = nearest_distances[:, None]
        nearest_indices = nearest_indices[:, None]
    inside = np.any(
        nearest_distances <= disk_radii[nearest_indices], axis=1
    )
    remaining = np.flatnonzero(~inside)
    if not len(remaining):
        return inside
    neighbor_lists = tree.query_ball_point(
        query[remaining],
        r=2.0 * float(local_scale),
        workers=-1,
    )
    centers = np.asarray(tree.data, dtype=np.float64)
    for output_index, neighbors in zip(remaining, neighbor_lists):
        if not neighbors:
            continue
        neighbor_array = np.asarray(neighbors, dtype=np.int64)
        distances = np.linalg.norm(
            centers[neighbor_array] - query[output_index], axis=1
        )
        inside[output_index] = bool(
            np.any(distances <= disk_radii[neighbor_array])
        )
    return inside


def points_in_accepted_triangles(
    query: np.ndarray,
    triangulation: Delaunay | None,
    accepted_triangles: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    if (
        not len(query)
        or triangulation is None
        or not accepted_triangles.any()
    ):
        return np.zeros(len(query), dtype=bool)
    simplex = triangulation.find_simplex(query, tol=0.0)
    inside = simplex >= 0
    result = np.zeros(len(query), dtype=bool)
    result[inside] = accepted_triangles[simplex[inside]]
    return result


def raw_membership_for_plane(
    points: np.ndarray,
    seed_assignment: np.ndarray,
    plane_id: int,
    normal: np.ndarray,
    offset: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(points, dtype=np.float64)
    seed_assignment = np.asarray(seed_assignment, dtype=np.int32)
    membership = np.zeros(len(points), dtype=bool)
    seed_indices = np.flatnonzero(seed_assignment == plane_id)
    audit: dict[str, Any] = {
        "plane_id": int(plane_id),
        "seed_count": int(len(seed_indices)),
        "unique_projected_seed_count": 0,
        "footprint_status": "not_started",
        "local_scale_s": None,
        "normal_threshold_tau": None,
        "seed_residual_median": None,
        "seed_residual_mad": None,
        "seed_residual_q99": None,
        "triangle_count_total": 0,
        "triangle_count_accepted": 0,
        "triangle_area_sum": 0.0,
        "disk_count": 0,
        "footprint_component_count": 0,
    }
    if len(seed_indices) < MINIMUM_SEED_POINTS:
        audit["footprint_status"] = "disabled:minimum_seed_points"
        return membership, audit

    basis_status, n, u, v = deterministic_tangent_basis(normal)
    if basis_status != "valid":
        audit["footprint_status"] = "disabled:degenerate_normal"
        return membership, audit
    projected, original_indices = unique_seed_projection(
        points, seed_indices, u, v
    )
    audit["unique_projected_seed_count"] = int(len(projected))
    scale_status, tree, d8, neighbor_indices, local_scale = (
        eighth_neighbor_scale(projected)
    )
    if scale_status != "valid":
        audit["footprint_status"] = f"disabled:{scale_status}"
        return membership, audit
    assert tree is not None
    assert d8 is not None
    assert neighbor_indices is not None
    assert local_scale is not None
    audit["local_scale_s"] = local_scale

    threshold = adaptive_normal_threshold(
        points[seed_indices], n, float(offset), local_scale
    )
    audit["seed_residual_median"] = threshold["median"]
    audit["seed_residual_mad"] = threshold["mad"]
    audit["seed_residual_q99"] = threshold["q99"]
    audit["normal_threshold_tau"] = threshold["tau"]
    if threshold["tau"] is None:
        audit["footprint_status"] = "disabled:invalid_normal_threshold"
        return membership, audit
    tau = float(threshold["tau"])

    radii = local_disk_radii(d8, local_scale)
    delaunay = accepted_delaunay_triangles(projected, local_scale)
    triangulation = delaunay["triangulation"]
    accepted_triangles = delaunay["accepted"]
    audit["triangle_count_total"] = delaunay["triangle_count_total"]
    audit["triangle_count_accepted"] = delaunay["triangle_count_accepted"]
    audit["triangle_area_sum"] = delaunay["triangle_area_sum"]
    audit["disk_count"] = int(len(projected))
    audit["footprint_component_count"] = footprint_component_count(
        projected,
        tree,
        radii,
        neighbor_indices,
        triangulation,
        accepted_triangles,
        local_scale,
    )
    audit["footprint_status"] = (
        "enabled"
        if triangulation is not None
        else "enabled:disks_only_no_convex_hull_fallback"
    )
    audit["tangent_normal"] = n.tolist()
    audit["tangent_u"] = u.tolist()
    audit["tangent_v"] = v.tolist()
    audit["unique_projection_original_point_indices_sha256"] = _array_sha256(
        original_indices
    )
    audit["disk_radii_sha256"] = _array_sha256(radii)

    query_indices = np.flatnonzero(seed_assignment != plane_id)
    residual = np.abs(points[query_indices] @ n + float(offset))
    residual_pass = residual <= tau
    residual_indices = query_indices[residual_pass]
    if not len(residual_indices):
        return membership, audit
    query_projection = np.column_stack(
        (
            points[residual_indices] @ u,
            points[residual_indices] @ v,
        )
    )
    third_distance, _ = tree.query(
        query_projection, k=3, workers=-1
    )
    third_distance = np.asarray(third_distance, dtype=np.float64)
    local_pass = third_distance[:, 2] <= 2.0 * local_scale
    local_indices = residual_indices[local_pass]
    local_projection = query_projection[local_pass]
    if not len(local_indices):
        return membership, audit
    in_disks = points_in_variable_disks(
        local_projection, tree, radii, local_scale
    )
    in_triangles = points_in_accepted_triangles(
        local_projection, triangulation, accepted_triangles
    )
    membership[local_indices[in_disks | in_triangles]] = True
    return membership, audit


def resolve_e2_assignment(
    seed_assignment: np.ndarray, raw_membership: np.ndarray
) -> dict[str, np.ndarray | int | float]:
    seed_assignment = np.asarray(seed_assignment, dtype=np.int32)
    raw_membership = np.asarray(raw_membership, dtype=bool)
    if raw_membership.shape[0] != len(seed_assignment):
        raise ValueError("membership and seed assignment length mismatch")
    nonseed = seed_assignment < 0
    multiplicity = raw_membership.sum(axis=1).astype(np.int16)
    unique_candidate = nonseed & (multiplicity == 1)
    conflict_rejected = nonseed & (multiplicity > 1)
    accepted_candidate_assignment = np.full(
        len(seed_assignment), -1, dtype=np.int32
    )
    if unique_candidate.any():
        accepted_candidate_assignment[unique_candidate] = np.argmax(
            raw_membership[unique_candidate], axis=1
        ).astype(np.int32)
    final_assignment = seed_assignment.copy()
    final_assignment[unique_candidate] = accepted_candidate_assignment[
        unique_candidate
    ]
    seed_mask = seed_assignment >= 0
    seed_retention_mask = ~seed_mask | (
        final_assignment == seed_assignment
    )
    seed_deletion_count = int(
        (seed_mask & (final_assignment < 0)).sum()
    )
    seed_reassignment_count = int(
        (
            seed_mask
            & (final_assignment >= 0)
            & (final_assignment != seed_assignment)
        ).sum()
    )
    return {
        "raw_candidate_multiplicity": multiplicity,
        "conflict_rejected_mask": conflict_rejected,
        "accepted_candidate_assignment": accepted_candidate_assignment,
        "final_assignment": final_assignment,
        "seed_retention_mask": seed_retention_mask,
        "seed_deletion_count": seed_deletion_count,
        "seed_reassignment_count": seed_reassignment_count,
        "final_multi_assignment_count": 0,
    }


def _plane_f1(metrics: dict[str, Any]) -> float:
    precision = float(metrics["plane_precision"])
    recall = float(metrics["plane_recall"])
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _explicit_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    match_summary: dict[str, Any],
) -> dict[str, Any]:
    gt_support = labels >= 0
    assigned = prediction >= 0
    gt_count = int(gt_support.sum())
    nonplane_count = int((~gt_support).sum())
    assigned_count = int(assigned.sum())
    gt_assigned = int((assigned & gt_support).sum())
    matched_count = int(match_summary["gt_support_matched_point_count"])
    nonplane_false = int((assigned & ~gt_support).sum())
    return {
        "assigned_cache_point_count": assigned_count,
        "gt_plane_point_count": gt_count,
        "gt_support_assigned_point_count": gt_assigned,
        "gt_support_matched_point_count": matched_count,
        "nonplane_point_count": nonplane_count,
        "nonplane_false_assigned_point_count": nonplane_false,
        "gt_support_assignment_rate": _safe_ratio(gt_assigned, gt_count),
        "gt_support_matched_coverage": _safe_ratio(
            matched_count, gt_count
        ),
        "full_cache_label_rate": _safe_ratio(
            assigned_count, len(labels)
        ),
        "gt_plane_label_rate": _safe_ratio(gt_count, len(labels)),
        "nonplane_false_assignment_rate": (
            _safe_ratio(nonplane_false, nonplane_count)
            if nonplane_count
            else None
        ),
    }


def _compare_e0_reproduction(
    baseline: dict[str, Any],
    metrics: dict[str, Any],
    match_summary: dict[str, Any],
    explicit: dict[str, Any],
) -> list[str]:
    expected = {
        "true_positive_planes": int(baseline["true_positive_planes"]),
        "plane_f1": float(baseline["plane_f1"]),
        "plane_weighted_matched_iou_median": float(
            baseline["plane_weighted_matched_iou_median"]
        ),
        "minimum_matched_iou": float(baseline["minimum_matched_iou"]),
        "gt_support_assignment_rate": float(
            baseline["gt_support_assignment_rate"]
        ),
        "gt_support_matched_coverage": float(
            baseline["gt_support_matched_coverage"]
        ),
        "full_cache_label_rate": float(
            baseline["full_cache_label_rate"]
        ),
        "gt_plane_label_rate": float(baseline["gt_plane_label_rate"]),
        "nonplane_false_assignment_rate": float(
            baseline["nonplane_false_assignment_rate"]
        ),
        "fragmentation_excess": int(baseline["fragmentation_excess"]),
        "overmerge_excess": int(baseline["overmerge_excess"]),
    }
    actual = {
        "true_positive_planes": int(metrics["true_positive_planes"]),
        "plane_f1": _plane_f1(metrics),
        "plane_weighted_matched_iou_median": float(
            match_summary["plane_weighted_matched_iou_median"]
        ),
        "minimum_matched_iou": float(
            match_summary["minimum_matched_iou"]
        ),
        "gt_support_assignment_rate": float(
            explicit["gt_support_assignment_rate"]
        ),
        "gt_support_matched_coverage": float(
            explicit["gt_support_matched_coverage"]
        ),
        "full_cache_label_rate": float(
            explicit["full_cache_label_rate"]
        ),
        "gt_plane_label_rate": float(explicit["gt_plane_label_rate"]),
        "nonplane_false_assignment_rate": float(
            explicit["nonplane_false_assignment_rate"]
        ),
        "fragmentation_excess": int(metrics["fragmentation_excess"]),
        "overmerge_excess": int(metrics["overmerge_excess"]),
    }
    failures = []
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, int):
            equal = actual_value == expected_value
        else:
            equal = abs(actual_value - expected_value) <= 1e-12
        if not equal:
            failures.append(
                f"E0_reproduction_mismatch:{key}:"
                f"{actual_value}!={expected_value}"
            )
    return failures


def _prediction_filename(scene: str) -> str:
    return f"{scene}_e2_candidate_bounded_footprint_no_refit.npz"


def _load_e0_records(
    e0_v1_dir: Path, e0_v2_dir: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    hashes = json.loads(
        (e0_v1_dir / "input_hashes.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (e0_v2_dir / "summary.json").read_text(encoding="utf-8")
    )
    return (
        {row["scene"]: row for row in hashes},
        {row["scene"]: row for row in summary["scenes"]},
    )


def _preflight_scene(
    entry: dict[str, Any],
    root: Path,
    e0_hash: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    scene = entry["scene"]
    data = load_scene(entry, root)
    gt = data["gt"]
    cache = data["cache"]
    current_gt_hash = sha256_file(data["gt_path"])
    current_cache_hash = sha256_file(data["cache_path"])
    failures = []
    if current_gt_hash != e0_hash["point_aligned_gt"]["sha256"]:
        failures.append("frozen_input_hash_mismatch:point_aligned_gt")
    if current_cache_hash != e0_hash["global_cache"]["sha256"]:
        failures.append("frozen_input_hash_mismatch:global_cache")
    view_equal = np.array_equal(cache["view_indices"], gt["view_indices"])
    pixel_equal = np.array_equal(cache["pixel_xy"], gt["pixel_xy"])
    point_equal = np.array_equal(cache["points"], gt["points"])
    if not view_equal:
        failures.append("frozen_join_contract_violation:view_indices")
    if not pixel_equal:
        failures.append("frozen_join_contract_violation:pixel_xy")
    if not point_equal:
        failures.append("frozen_join_contract_violation:point_order")
    assignment, join_rows = exact_unique_join(
        cache["view_indices"],
        cache["pixel_xy"],
        gt["view_indices"],
        gt["pixel_xy"],
        gt["point_plane_ids"],
        scene=scene,
        variant="E2_frozen_E0_exact_unique_join",
    )
    aggregate = join_rows[-1]
    assignment_equal = np.array_equal(
        assignment, gt["point_plane_ids"]
    )
    if not assignment_equal:
        failures.append("frozen_join_contract_violation:assignment")
    if int(aggregate["ambiguous_key_count"]) != 0:
        failures.append("frozen_join_contract_violation:ambiguous_key")
    if int(aggregate["unmatched_key_count"]) != 0:
        failures.append("frozen_join_contract_violation:unmatched_key")
    row = {
        "scene": scene,
        "point_aligned_gt": {
            "path": str(data["gt_path"]),
            "sha256": current_gt_hash,
            "E0_sha256": e0_hash["point_aligned_gt"]["sha256"],
            "matches_E0": current_gt_hash
            == e0_hash["point_aligned_gt"]["sha256"],
        },
        "global_cache": {
            "path": str(data["cache_path"]),
            "sha256": current_cache_hash,
            "E0_sha256": e0_hash["global_cache"]["sha256"],
            "matches_E0": current_cache_hash
            == e0_hash["global_cache"]["sha256"],
        },
        "view_indices_exact_equal": view_equal,
        "pixel_xy_exact_equal": pixel_equal,
        "point_order_exact_equal": point_equal,
        "join_assignment_exact_equal": assignment_equal,
        "join_assignment_dtype": str(assignment.dtype),
        "join_assignment_shape": list(assignment.shape),
        "ambiguous_key_count": int(aggregate["ambiguous_key_count"]),
        "unmatched_key_count": int(aggregate["unmatched_key_count"]),
        "failures": failures,
    }
    return row, failures


def _build_scene_prediction(
    entry: dict[str, Any],
    root: Path,
    prediction_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = load_scene(entry, root)
    gt = data["gt"]
    points = np.asarray(gt["points"], dtype=np.float64)
    seed_assignment = np.asarray(gt["point_plane_ids"], dtype=np.int32)
    normals = np.asarray(gt["plane_normals"])
    offsets = np.asarray(gt["plane_offsets"])
    plane_ids = np.arange(len(normals), dtype=np.int32)
    raw_membership = np.zeros(
        (len(points), len(plane_ids)), dtype=bool
    )
    plane_audits: list[dict[str, Any]] = []
    for plane_id in plane_ids:
        membership, audit = raw_membership_for_plane(
            points,
            seed_assignment,
            int(plane_id),
            np.asarray(normals[plane_id], dtype=np.float64),
            float(offsets[plane_id]),
        )
        raw_membership[:, plane_id] = membership
        audit["scene"] = entry["scene"]
        plane_audits.append(audit)

    resolution = resolve_e2_assignment(
        seed_assignment, raw_membership
    )
    multiplicity = np.asarray(
        resolution["raw_candidate_multiplicity"], dtype=np.int16
    )
    conflict_rejected = np.asarray(
        resolution["conflict_rejected_mask"], dtype=bool
    )
    accepted_assignment = np.asarray(
        resolution["accepted_candidate_assignment"], dtype=np.int32
    )
    final_assignment = np.asarray(
        resolution["final_assignment"], dtype=np.int32
    )
    seed_retention = np.asarray(
        resolution["seed_retention_mask"], dtype=bool
    )
    nonseed = seed_assignment < 0
    per_plane_conflicts = np.asarray(
        [
            int((raw_membership[:, plane_id] & conflict_rejected).sum())
            for plane_id in plane_ids
        ],
        dtype=np.int64,
    )
    cross_seed_by_plane = np.asarray(
        [
            int(
                (
                    raw_membership[:, plane_id]
                    & (seed_assignment >= 0)
                    & (seed_assignment != plane_id)
                ).sum()
            )
            for plane_id in plane_ids
        ],
        dtype=np.int64,
    )
    for plane_id, audit in enumerate(plane_audits):
        audit["raw_candidate_count"] = int(
            (raw_membership[:, plane_id] & nonseed).sum()
        )
        audit["accepted_unique_candidate_count"] = int(
            (accepted_assignment == plane_id).sum()
        )
        audit["conflict_rejected_count"] = int(
            per_plane_conflicts[plane_id]
        )
        audit["cross_seed_intrusion_count"] = int(
            cross_seed_by_plane[plane_id]
        )

    footprint_parameters = [
        {
            key: audit.get(key)
            for key in (
                "plane_id",
                "footprint_status",
                "local_scale_s",
                "normal_threshold_tau",
                "triangle_count_total",
                "triangle_count_accepted",
                "triangle_area_sum",
                "disk_count",
                "footprint_component_count",
                "tangent_normal",
                "tangent_u",
                "tangent_v",
                "unique_projection_original_point_indices_sha256",
                "disk_radii_sha256",
            )
        }
        for audit in plane_audits
    ]
    content_hashes = {
        "points": _array_sha256(gt["points"]),
        "E0_seed_assignment": _array_sha256(seed_assignment),
        "raw_candidate_membership": _array_sha256(raw_membership),
        "raw_candidate_multiplicity": _array_sha256(multiplicity),
        "accepted_candidate_assignment": _array_sha256(
            accepted_assignment
        ),
        "final_E2_assignment": _array_sha256(final_assignment),
        "frozen_plane_normals": _array_sha256(normals),
        "frozen_plane_offsets": _array_sha256(offsets),
    }
    metadata = {
        "registration_id": REGISTRATION_ID,
        "prediction_file_sha256": (
            "recorded_externally_after_npz_freeze_to_avoid_self_hash_cycle"
        ),
        "content_sha256": content_hashes,
        "prediction_phase_GT_access": (
            "only preregistered frozen E0 seed assignment and plane "
            "parameters; no GT scoring, matching, leakage labels, or tuning"
        ),
    }
    np.savez_compressed(
        prediction_path,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        registration_id=np.asarray(REGISTRATION_ID),
        scene=np.asarray(entry["scene"]),
        points=np.asarray(gt["points"]),
        plane_ids=plane_ids,
        frozen_plane_normals=np.asarray(normals),
        frozen_plane_offsets=np.asarray(offsets),
        E0_seed_assignment=seed_assignment,
        raw_candidate_membership=raw_membership,
        raw_candidate_membership_plane_ids=plane_ids,
        raw_candidate_multiplicity=multiplicity,
        conflict_rejected_mask=conflict_rejected,
        multi_candidate_point_indices=np.flatnonzero(
            nonseed & (multiplicity > 1)
        ).astype(np.int64),
        per_plane_conflict_participation=per_plane_conflicts,
        accepted_candidate_assignment=accepted_assignment,
        final_E2_assignment=final_assignment,
        point_plane_ids=final_assignment,
        seed_retention_mask=seed_retention,
        footprint_parameters_json=np.asarray(
            json.dumps(footprint_parameters, allow_nan=True)
        ),
        plane_fit_call_count=np.asarray(
            PLANE_FIT_CALL_COUNT, dtype=np.int32
        ),
        weighted_fit_plane_call_count=np.asarray(
            WEIGHTED_FIT_PLANE_CALL_COUNT, dtype=np.int32
        ),
        refit_call_count=np.asarray(REFIT_CALL_COUNT, dtype=np.int32),
        ransac_call_count=np.asarray(RANSAC_CALL_COUNT, dtype=np.int32),
        prediction_sha256_metadata=np.asarray(
            json.dumps(metadata, sort_keys=True)
        ),
    )
    scene_prediction_audit = {
        "scene": entry["scene"],
        "raw_multi_candidate_points": int(
            (nonseed & (multiplicity > 1)).sum()
        ),
        "maximum_multiplicity": int(
            multiplicity[nonseed].max()
        )
        if nonseed.any()
        else 0,
        "final_multi_assignment_count": int(
            resolution["final_multi_assignment_count"]
        ),
        "seed_deletion_count": int(
            resolution["seed_deletion_count"]
        ),
        "seed_reassignment_count": int(
            resolution["seed_reassignment_count"]
        ),
        "cross_seed_intrusion_count": int(cross_seed_by_plane.sum()),
        "per_plane_conflict_participation": per_plane_conflicts.tolist(),
        "prediction_path": str(prediction_path),
    }
    return plane_audits, scene_prediction_audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest_json).resolve()
    e0_v1_dir = Path(args.e0_v1_dir).resolve()
    e0_v2_dir = Path(args.e0_v2_dir).resolve()
    preregistration_path = Path(args.preregistration_md).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir()

    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    e0_hashes, e0_scenes = _load_e0_records(e0_v1_dir, e0_v2_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry_by_scene = {
        row["scene"]: row
        for row in manifest["scenes"]
        if row["scene"] in set(args.pressure_scenes)
    }
    entries = [entry_by_scene[scene] for scene in args.pressure_scenes]
    if len(entries) != len(args.pressure_scenes):
        raise ValueError("one or more frozen scenes are missing")
    (output_dir / "preregistration.md").write_text(
        preregistration_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    source_hashes = {
        "E2_evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "E0_contract_audit": {
            "path": str(root / "audit_support_bounded_evaluator_contract.py"),
            "sha256": sha256_file(
                root / "audit_support_bounded_evaluator_contract.py"
            ),
        },
        "matcher_function_sha256": _function_sha256(E2_MATCHER),
        "metric_function_sha256": _function_sha256(E2_METRIC),
        "matcher_and_metric_module": {
            "path": str(root / "evaluate_global_plane_baselines.py"),
            "sha256": sha256_file(
                root / "evaluate_global_plane_baselines.py"
            ),
        },
        "exact_join_function_sha256": _function_sha256(exact_unique_join),
    }
    (output_dir / "source_hashes.json").write_text(
        json.dumps(source_hashes, indent=2), encoding="utf-8"
    )
    state = git_state(root)
    state["environment"] = environment_state()
    state["base_audit_git_sha_expected"] = BASE_GIT_SHA
    state["base_audit_git_sha_matches"] = state["sha"] == BASE_GIT_SHA
    (output_dir / "git_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "registration_id": REGISTRATION_ID,
        "experiment": "evaluator-only safety audit",
        "input_baseline": "E0 exact support",
        "E1_prediction_used_as_input": False,
        "minimum_seed_points": MINIMUM_SEED_POINTS,
        "tangent_basis": "deterministic frozen-normal global-axis rule",
        "local_scale": "median projected 8th-neighbour distance",
        "footprint": (
            "accepted Delaunay triangles union clipped seed-local disks"
        ),
        "convex_hull_fallback": False,
        "bbox_fallback": False,
        "normal_threshold": (
            "min(0.03, 0.5*s_p, max(1e-6, q99+3*1.4826*MAD))"
        ),
        "candidate_d3_limit": "2.0*s_p",
        "multi_candidate_rule": "reject all; no reassignment",
        "seed_rule": "immutable",
        "plane_fit_call_count": PLANE_FIT_CALL_COUNT,
        "weighted_fit_plane_call_count": WEIGHTED_FIT_PLANE_CALL_COUNT,
        "refit_call_count": REFIT_CALL_COUNT,
        "ransac_call_count": RANSAC_CALL_COUNT,
        "training_executed": False,
        "checkpoint_used_or_modified": False,
        "L_G_J_executed": False,
        "E3_executed": False,
        "pressure_scenes": list(args.pressure_scenes),
        "pressure_scenes_used_for_tuning": False,
        "formal_run_count": 1,
        "matcher_source_hash": source_hashes[
            "matcher_function_sha256"
        ],
        "metric_source_hash": source_hashes["metric_function_sha256"],
        "matcher_config": {
            "objective": "full-cache IoU",
            "solver": "Hungarian linear_sum_assignment(1-IoU)",
            "one_to_one": True,
            "plane_order": "sorted nonnegative IDs",
            "plane_tp_iou": MATCH_IOU,
        },
        "metric_config": {
            "match_iou": MATCH_IOU,
            "fragmentation_iou": FRAGMENTATION_IOU,
            "min_observed_plane_points": MIN_OBSERVED_PLANE_POINTS,
        },
        "manifest_json": str(manifest_path),
        "E0_input_hashes_json": str(e0_v1_dir / "input_hashes.json"),
        "E0_metric_summary_json": str(e0_v2_dir / "summary.json"),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    input_rows = []
    validity_failures: list[str] = []
    for entry in entries:
        row, failures = _preflight_scene(
            entry, root, e0_hashes[entry["scene"]]
        )
        input_rows.append(row)
        validity_failures.extend(
            f"{entry['scene']}:{failure}" for failure in failures
        )
        log(
            f"preflight {entry['scene']}: "
            f"valid={not failures} failures={failures or ['none']}"
        )
    (output_dir / "input_hashes.json").write_text(
        json.dumps(input_rows, indent=2), encoding="utf-8"
    )
    if any(
        count != 0
        for count in (
            PLANE_FIT_CALL_COUNT,
            WEIGHTED_FIT_PLANE_CALL_COUNT,
            REFIT_CALL_COUNT,
            RANSAC_CALL_COUNT,
        )
    ):
        validity_failures.append("fit_refit_or_ransac_call_count_nonzero")
    if state["sha"] != BASE_GIT_SHA:
        validity_failures.append("base_audit_git_sha_mismatch")

    contract_validation: dict[str, Any] = {
        "registration_id": REGISTRATION_ID,
        "preflight_valid": not validity_failures,
        "validity_failures": validity_failures,
        "input_hashes_match": all(
            row["point_aligned_gt"]["matches_E0"]
            and row["global_cache"]["matches_E0"]
            for row in input_rows
        ),
        "join_contract_exact": all(
            row["view_indices_exact_equal"]
            and row["pixel_xy_exact_equal"]
            and row["point_order_exact_equal"]
            and row["join_assignment_exact_equal"]
            and row["ambiguous_key_count"] == 0
            and row["unmatched_key_count"] == 0
            for row in input_rows
        ),
        "E1_prediction_used_as_input": False,
        "training_executed": False,
        "L_G_J_executed": False,
        "E3_executed": False,
        "plane_fit_call_count": PLANE_FIT_CALL_COUNT,
        "weighted_fit_plane_call_count": WEIGHTED_FIT_PLANE_CALL_COUNT,
        "refit_call_count": REFIT_CALL_COUNT,
        "ransac_call_count": RANSAC_CALL_COUNT,
        "prediction_frozen_before_GT_scoring": False,
        "matcher_exact_reuse": E2_MATCHER is matching_rows,
        "metric_exact_reuse": E2_METRIC is evaluate_arrays,
    }
    if validity_failures:
        contract_validation["overall_contract_valid"] = False
        (output_dir / "contract_validation.json").write_text(
            json.dumps(contract_validation, indent=2), encoding="utf-8"
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "registration_id": REGISTRATION_ID,
            "contract_verdict": "Invalid",
            "invalid_reasons": validity_failures,
            "formal_metrics_generated": False,
            "next_action": (
                "Stop. Do not modify thresholds or run E3/L/G/J; await a "
                "new mechanical instruction."
            ),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (output_dir / "run.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8"
        )
        return summary

    # Prediction phase. No matcher, metric, or post-freeze leakage audit is
    # called before every prediction has been written and hashed.
    barrier = FreezeBarrier(len(entries), {})
    pre_plane_audits: dict[str, list[dict[str, Any]]] = {}
    pre_scene_audits: dict[str, dict[str, Any]] = {}
    prediction_hash_rows = []
    for entry in entries:
        scene = entry["scene"]
        prediction_path = prediction_dir / _prediction_filename(scene)
        log(f"prediction_phase_start {scene}")
        plane_audits, scene_audit = _build_scene_prediction(
            entry, root, prediction_path
        )
        prediction_hash = sha256_file(prediction_path)
        barrier.record_prediction(scene, prediction_hash)
        pre_plane_audits[scene] = plane_audits
        pre_scene_audits[scene] = scene_audit
        prediction_hash_rows.append(
            {
                "scene": scene,
                "path": str(prediction_path),
                "sha256": prediction_hash,
            }
        )
        log(
            f"prediction_phase_frozen {scene}: sha256={prediction_hash} "
            f"raw_multi={scene_audit['raw_multi_candidate_points']} "
            f"cross_seed={scene_audit['cross_seed_intrusion_count']}"
        )
    barrier.freeze_all()
    contract_validation["prediction_frozen_before_GT_scoring"] = True
    contract_validation["prediction_hashes"] = prediction_hash_rows
    log("prediction_frozen_before_GT_scoring=true")

    plane_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    multi_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    matcher_outputs: list[dict[str, Any]] = []

    # GT scoring phase starts only after the barrier above.
    barrier.assert_gt_scoring_allowed()
    for entry in entries:
        scene = entry["scene"]
        prediction_path = prediction_dir / _prediction_filename(scene)
        if sha256_file(prediction_path) != barrier.frozen_prediction_hashes[scene]:
            validity_failures.append(
                f"{scene}:prediction_changed_after_freeze"
            )
            break
        data = load_scene(entry, root)
        gt = data["gt"]
        points = np.asarray(gt["points"], dtype=np.float64)
        gt_labels = np.asarray(gt["point_plane_ids"], dtype=np.int32)
        gt_normals = np.asarray(gt["plane_normals"])
        with np.load(prediction_path, allow_pickle=False) as pred:
            seed_assignment = pred["E0_seed_assignment"].astype(np.int32)
            final_assignment = pred["final_E2_assignment"].astype(np.int32)
            raw_membership = pred["raw_candidate_membership"].astype(bool)
            multiplicity = pred["raw_candidate_multiplicity"].astype(np.int16)
            conflict_rejected = pred["conflict_rejected_mask"].astype(bool)
            accepted_assignment = pred[
                "accepted_candidate_assignment"
            ].astype(np.int32)
            frozen_normals = pred["frozen_plane_normals"]
            frozen_offsets = pred["frozen_plane_offsets"]
            plane_ids = pred["plane_ids"].astype(np.int32)
            call_counts = (
                int(pred["plane_fit_call_count"]),
                int(pred["weighted_fit_plane_call_count"]),
                int(pred["refit_call_count"]),
                int(pred["ransac_call_count"]),
            )
        if call_counts != (0, 0, 0, 0):
            validity_failures.append(f"{scene}:nonzero_fit_call_count")
            break
        if not np.array_equal(seed_assignment, gt_labels):
            validity_failures.append(
                f"{scene}:frozen_seed_assignment_modified"
            )
            break
        if not np.array_equal(frozen_normals, gt["plane_normals"]):
            validity_failures.append(
                f"{scene}:frozen_plane_normals_modified"
            )
            break
        if not np.array_equal(frozen_offsets, gt["plane_offsets"]):
            validity_failures.append(
                f"{scene}:frozen_plane_offsets_modified"
            )
            break

        e0_metrics = E2_METRIC(
            points,
            seed_assignment,
            frozen_normals,
            frozen_offsets,
            gt_labels,
            gt["plane_normals"],
            MATCH_IOU,
            FRAGMENTATION_IOU,
            MIN_OBSERVED_PLANE_POINTS,
        )
        e0_match_rows, e0_match_summary = E2_MATCHER(
            scene,
            points,
            seed_assignment,
            frozen_normals,
            frozen_offsets,
            gt_labels,
            gt["plane_normals"],
        )
        e0_explicit = _explicit_metrics(
            gt_labels, seed_assignment, e0_match_summary
        )
        e0_reproduction_failures = _compare_e0_reproduction(
            e0_scenes[scene],
            e0_metrics,
            e0_match_summary,
            e0_explicit,
        )
        if e0_reproduction_failures:
            validity_failures.extend(
                f"{scene}:{failure}"
                for failure in e0_reproduction_failures
            )
            break

        e2_metrics = E2_METRIC(
            points,
            final_assignment,
            frozen_normals,
            frozen_offsets,
            gt_labels,
            gt["plane_normals"],
            MATCH_IOU,
            FRAGMENTATION_IOU,
            MIN_OBSERVED_PLANE_POINTS,
        )
        e2_match_rows, e2_match_summary = E2_MATCHER(
            scene,
            points,
            final_assignment,
            frozen_normals,
            frozen_offsets,
            gt_labels,
            gt["plane_normals"],
        )
        e2_explicit = _explicit_metrics(
            gt_labels, final_assignment, e2_match_summary
        )
        matcher_outputs.append(
            {
                "scene": scene,
                "matcher_source_hash": source_hashes[
                    "matcher_function_sha256"
                ],
                "matcher_config": config["matcher_config"],
                "matcher_plane_order": {
                    "prediction": plane_ids.tolist(),
                    "GT": sorted(
                        int(value)
                        for value in np.unique(gt_labels)
                        if value >= 0
                    ),
                },
                "E0_output": e0_match_rows,
                "E2_output": e2_match_rows,
            }
        )
        e0_by_pred = {
            int(row["pred_plane_id"]): row for row in e0_match_rows
        }
        e2_by_pred = {
            int(row["pred_plane_id"]): row for row in e2_match_rows
        }

        scene_plane_rows = []
        for plane_id, preaudit in enumerate(pre_plane_audits[scene]):
            seed_mask = seed_assignment == plane_id
            retained = final_assignment[seed_mask] == plane_id
            seed_count = int(seed_mask.sum())
            retained_count = int(retained.sum())
            deletion_count = int(
                (seed_mask & (final_assignment < 0)).sum()
            )
            reassignment_count = int(
                (
                    seed_mask
                    & (final_assignment >= 0)
                    & (final_assignment != plane_id)
                ).sum()
            )
            added_mask = (accepted_assignment == plane_id) & (
                seed_assignment < 0
            )
            added_count = int(added_mask.sum())
            e0_match = e0_by_pred.get(plane_id)
            e2_match = e2_by_pred.get(plane_id)
            matched_gt = (
                int(e2_match["gt_plane_id"]) if e2_match else -1
            )
            if matched_gt >= 0:
                wrong_mask = (
                    added_mask
                    & (gt_labels >= 0)
                    & (gt_labels != matched_gt)
                )
                nonplane_mask = added_mask & (gt_labels < 0)
            else:
                wrong_mask = np.zeros(len(gt_labels), dtype=bool)
                nonplane_mask = added_mask.copy()
            wrong_count = int(wrong_mask.sum())
            nonplane_added = int(nonplane_mask.sum())
            total_leakage = (
                added_count
                if matched_gt < 0
                else int((wrong_mask | nonplane_mask).sum())
            )
            target_gt_count = (
                int((gt_labels == matched_gt).sum())
                if matched_gt >= 0
                else 0
            )
            e0_iou = (
                float(e0_match["matched_iou"]) if e0_match else 0.0
            )
            e2_iou = (
                float(e2_match["matched_iou"]) if e2_match else 0.0
            )
            plane_failures = []
            if retained_count != seed_count:
                plane_failures.append("seed_retention_not_exact")
            if deletion_count:
                plane_failures.append("seed_deletion")
            if reassignment_count:
                plane_failures.append("seed_reassignment")
            if int(preaudit["cross_seed_intrusion_count"]) != 0:
                plane_failures.append("cross_seed_intrusion")
            if _safe_ratio(added_count, seed_count) > 0.25:
                plane_failures.append("added_over_seed_above_0.25")
            if added_count:
                if _safe_ratio(wrong_count, added_count) > 0.01:
                    plane_failures.append(
                        "wrong_GT_added_fraction_above_0.01"
                    )
                if _safe_ratio(nonplane_added, added_count) > 0.05:
                    plane_failures.append(
                        "nonplane_added_fraction_above_0.05"
                    )
                if _safe_ratio(total_leakage, added_count) > 0.05:
                    plane_failures.append(
                        "total_leakage_fraction_above_0.05"
                    )
            if matched_gt >= 0:
                if (
                    _safe_ratio(total_leakage, target_gt_count)
                    > 0.01
                ):
                    plane_failures.append(
                        "leakage_over_target_GT_above_0.01"
                    )
            elif added_count:
                plane_failures.append(
                    "unmatched_prediction_has_added_points"
                )
            if e2_iou < e0_iou - 0.01:
                plane_failures.append("matched_IoU_regression")

            row = {
                **{
                    field: preaudit.get(field)
                    for field in PER_PLANE_FIELDS
                },
                "scene": scene,
                "plane_id": plane_id,
                "added_count": added_count,
                "added_over_seed_count": _safe_ratio(
                    added_count, seed_count
                ),
                "seed_retention": _safe_ratio(
                    retained_count, seed_count
                ),
                "matched_GT_plane_id": matched_gt,
                "E0_matched_IoU": e0_iou,
                "E2_matched_IoU": e2_iou,
                "matched_IoU_delta": e2_iou - e0_iou,
                "wrong_GT_added_count": wrong_count,
                "nonplane_added_count": nonplane_added,
                "total_leakage_count": total_leakage,
                "wrong_GT_added_fraction": _safe_ratio(
                    wrong_count, max(added_count, 1)
                ),
                "nonplane_added_fraction": _safe_ratio(
                    nonplane_added, max(added_count, 1)
                ),
                "total_leakage_fraction": _safe_ratio(
                    total_leakage, max(added_count, 1)
                ),
                "leakage_over_target_GT_count": _safe_ratio(
                    total_leakage, max(target_gt_count, 1)
                ),
                "plane_verdict": (
                    "Passed" if not plane_failures else "Failed"
                ),
                "failure_reasons": ";".join(plane_failures),
            }
            scene_plane_rows.append(row)
            seed_rows.append(
                {
                    "scene": scene,
                    "plane_id": plane_id,
                    "seed_count": seed_count,
                    "retained_seed_count": retained_count,
                    "seed_retention": row["seed_retention"],
                    "seed_deletion_count": deletion_count,
                    "seed_reassignment_count": reassignment_count,
                    "cross_seed_intrusion_count": int(
                        preaudit["cross_seed_intrusion_count"]
                    ),
                }
            )
            leakage_rows.append(
                {field: row.get(field) for field in LEAKAGE_FIELDS}
            )
        plane_rows.extend(scene_plane_rows)

        seed_mask = seed_assignment >= 0
        nonseed_mask = ~seed_mask
        total_seed = int(seed_mask.sum())
        seed_deletion = int(
            (seed_mask & (final_assignment < 0)).sum()
        )
        seed_reassignment = int(
            (
                seed_mask
                & (final_assignment >= 0)
                & (final_assignment != seed_assignment)
            ).sum()
        )
        total_added = int(
            (nonseed_mask & (final_assignment >= 0)).sum()
        )
        total_wrong = int(
            sum(row["wrong_GT_added_count"] for row in scene_plane_rows)
        )
        total_nonplane = int(
            sum(row["nonplane_added_count"] for row in scene_plane_rows)
        )
        total_leakage = int(
            sum(row["total_leakage_count"] for row in scene_plane_rows)
        )
        raw_multi = int(
            (nonseed_mask & (multiplicity > 1)).sum()
        )
        max_multiplicity = (
            int(multiplicity[nonseed_mask].max())
            if nonseed_mask.any()
            else 0
        )
        final_multi_count = 0
        cross_seed = int(
            sum(
                row["cross_seed_intrusion_count"]
                for row in scene_plane_rows
            )
        )
        e0_f1 = _plane_f1(e0_metrics)
        e2_f1 = _plane_f1(e2_metrics)
        scene_failures = []
        if seed_deletion:
            scene_failures.append("seed_deletion")
        if seed_reassignment:
            scene_failures.append("seed_reassignment")
        if cross_seed:
            scene_failures.append("cross_seed_intrusion")
        if not np.array_equal(
            np.unique(final_assignment[final_assignment >= 0]),
            plane_ids,
        ):
            scene_failures.append("plane_IDs_or_count_changed")
        if _safe_ratio(raw_multi, total_seed) > 0.01:
            scene_failures.append(
                "raw_multi_over_GT_plane_points_above_0.01"
            )
        if max_multiplicity > 2:
            scene_failures.append("maximum_multiplicity_above_2")
        if final_multi_count:
            scene_failures.append("final_multi_assignment_nonzero")
        if _safe_ratio(total_added, total_seed) > 0.25:
            scene_failures.append(
                "scene_added_over_seed_above_0.25"
            )
        if _safe_ratio(total_wrong, max(total_added, 1)) > 0.01:
            scene_failures.append(
                "scene_wrong_GT_fraction_above_0.01"
            )
        if _safe_ratio(total_leakage, max(total_added, 1)) > 0.05:
            scene_failures.append(
                "scene_total_leakage_fraction_above_0.05"
            )
        if _safe_ratio(total_leakage, total_seed) > 0.005:
            scene_failures.append(
                "scene_total_leakage_over_GT_above_0.005"
            )
        if (
            float(e2_match_summary["plane_weighted_matched_iou_median"])
            < float(e0_match_summary["plane_weighted_matched_iou_median"])
            - 0.005
        ):
            scene_failures.append("scene_IoU_median_regression")
        if (
            float(e2_match_summary["minimum_matched_iou"])
            < float(e0_match_summary["minimum_matched_iou"]) - 0.01
        ):
            scene_failures.append("scene_minimum_IoU_regression")
        if e2_f1 < e0_f1:
            scene_failures.append("scene_Plane_F1_regression")
        if int(e2_metrics["true_positive_planes"]) < int(
            e0_metrics["true_positive_planes"]
        ):
            scene_failures.append("scene_Plane_TP_regression")
        if int(e2_metrics["fragmentation_excess"]) > int(
            e0_metrics["fragmentation_excess"]
        ):
            scene_failures.append("fragmentation_count_regression")
        if int(e2_metrics["overmerge_excess"]) > int(
            e0_metrics["overmerge_excess"]
        ):
            scene_failures.append("oversegmentation_count_regression")
        if any(row["plane_verdict"] == "Failed" for row in scene_plane_rows):
            scene_failures.append("one_or_more_plane_gates_failed")

        scene_row = {
            "scene": scene,
            "scene_verdict": (
                "Passed" if not scene_failures else "Failed"
            ),
            "failure_reasons": ";".join(scene_failures),
            "cache_point_count": int(len(gt_labels)),
            "E0_plane_count": int(len(plane_ids)),
            "E2_plane_count": int(
                len(np.unique(final_assignment[final_assignment >= 0]))
            ),
            "plane_ids_unchanged": bool(
                np.array_equal(
                    np.unique(final_assignment[final_assignment >= 0]),
                    plane_ids,
                )
            ),
            "global_seed_retention": _safe_ratio(
                total_seed - seed_deletion - seed_reassignment, total_seed
            ),
            "seed_deletion_count": seed_deletion,
            "seed_reassignment_count": seed_reassignment,
            "raw_multi_candidate_points": raw_multi,
            "raw_multi_candidate_over_GT_plane_points": _safe_ratio(
                raw_multi, total_seed
            ),
            "raw_multi_candidate_over_full_cache": _safe_ratio(
                raw_multi, len(gt_labels)
            ),
            "maximum_multiplicity": max_multiplicity,
            "final_multi_assignment_count": final_multi_count,
            "cross_seed_intrusion_count": cross_seed,
            "total_added_count": total_added,
            "total_added_over_E0_seed_count": _safe_ratio(
                total_added, total_seed
            ),
            "total_wrong_GT_added": total_wrong,
            "total_nonplane_added": total_nonplane,
            "total_leakage": total_leakage,
            "total_wrong_GT_added_fraction": _safe_ratio(
                total_wrong, max(total_added, 1)
            ),
            "total_leakage_fraction": _safe_ratio(
                total_leakage, max(total_added, 1)
            ),
            "total_leakage_over_GT_plane_points": _safe_ratio(
                total_leakage, total_seed
            ),
            "E0_plane_f1": e0_f1,
            "E2_plane_f1": e2_f1,
            "plane_f1_delta": e2_f1 - e0_f1,
            "E0_true_positive_planes": int(
                e0_metrics["true_positive_planes"]
            ),
            "E2_true_positive_planes": int(
                e2_metrics["true_positive_planes"]
            ),
            "E0_iou_median": float(
                e0_match_summary["plane_weighted_matched_iou_median"]
            ),
            "E2_iou_median": float(
                e2_match_summary["plane_weighted_matched_iou_median"]
            ),
            "iou_median_delta": float(
                e2_match_summary["plane_weighted_matched_iou_median"]
                - e0_match_summary["plane_weighted_matched_iou_median"]
            ),
            "E0_minimum_iou": float(
                e0_match_summary["minimum_matched_iou"]
            ),
            "E2_minimum_iou": float(
                e2_match_summary["minimum_matched_iou"]
            ),
            "minimum_iou_delta": float(
                e2_match_summary["minimum_matched_iou"]
                - e0_match_summary["minimum_matched_iou"]
            ),
            "E0_gt_support_assignment_rate": float(
                e0_explicit["gt_support_assignment_rate"]
            ),
            "E2_gt_support_assignment_rate": float(
                e2_explicit["gt_support_assignment_rate"]
            ),
            "E0_gt_support_matched_coverage": float(
                e0_explicit["gt_support_matched_coverage"]
            ),
            "E2_gt_support_matched_coverage": float(
                e2_explicit["gt_support_matched_coverage"]
            ),
            "E0_full_cache_label_rate": float(
                e0_explicit["full_cache_label_rate"]
            ),
            "E2_full_cache_label_rate": float(
                e2_explicit["full_cache_label_rate"]
            ),
            "E0_gt_plane_label_rate": float(
                e0_explicit["gt_plane_label_rate"]
            ),
            "E2_gt_plane_label_rate": float(
                e2_explicit["gt_plane_label_rate"]
            ),
            "E0_nonplane_false_assignment_rate": float(
                e0_explicit["nonplane_false_assignment_rate"]
            ),
            "E2_nonplane_false_assignment_rate": float(
                e2_explicit["nonplane_false_assignment_rate"]
            ),
            "E0_fragmentation_count": int(
                e0_metrics["fragmentation_excess"]
            ),
            "E2_fragmentation_count": int(
                e2_metrics["fragmentation_excess"]
            ),
            "E0_oversegmentation_count": int(
                e0_metrics["overmerge_excess"]
            ),
            "E2_oversegmentation_count": int(
                e2_metrics["overmerge_excess"]
            ),
        }
        scene_rows.append(scene_row)
        multi_rows.append(
            {
                "scene": scene,
                "raw_multi_candidate_points": raw_multi,
                "raw_multi_candidate_over_GT_plane_points": _safe_ratio(
                    raw_multi, total_seed
                ),
                "raw_multi_candidate_over_full_cache": _safe_ratio(
                    raw_multi, len(gt_labels)
                ),
                "maximum_multiplicity": max_multiplicity,
                "final_multi_assignment_count": final_multi_count,
                "multi_candidate_point_indices_count": int(
                    conflict_rejected.sum()
                ),
                "per_plane_conflict_participation_json": json.dumps(
                    [
                        int(
                            (
                                raw_membership[:, plane_id]
                                & conflict_rejected
                            ).sum()
                        )
                        for plane_id in plane_ids
                    ]
                ),
                "conflict_rule": (
                    "all nonseed points with multiplicity > 1 are rejected; "
                    "no residual tie-break and no second reassignment"
                ),
            }
        )
        log(
            f"scoring {scene}: verdict={scene_row['scene_verdict']} "
            f"added={total_added} leakage={total_leakage} "
            f"raw_multi={raw_multi} cross_seed={cross_seed} "
            f"F1={e2_f1:.6f} "
            f"IoU_med={scene_row['E2_iou_median']:.6f} "
            f"failures={scene_failures or ['none']}"
        )

    if validity_failures:
        contract_validation["overall_contract_valid"] = False
        contract_validation["validity_failures"] = validity_failures
        verdict = "Invalid"
    else:
        contract_validation["overall_contract_valid"] = True
        verdict = (
            "Passed"
            if all(row["scene_verdict"] == "Passed" for row in scene_rows)
            else "Failed"
        )
    (output_dir / "contract_validation.json").write_text(
        json.dumps(contract_validation, indent=2), encoding="utf-8"
    )
    (output_dir / "prediction_hashes.json").write_text(
        json.dumps(prediction_hash_rows, indent=2), encoding="utf-8"
    )
    (output_dir / "matcher_output.json").write_text(
        json.dumps(matcher_outputs, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    _write_table(
        output_dir,
        "per_plane_footprint_audit",
        plane_rows,
        PER_PLANE_FIELDS,
    )
    _write_table(output_dir, "per_scene_metrics", scene_rows, SCENE_FIELDS)
    _write_table(
        output_dir, "multi_assignment_audit", multi_rows, MULTI_FIELDS
    )
    _write_table(
        output_dir, "seed_retention_audit", seed_rows, SEED_FIELDS
    )
    _write_table(output_dir, "leakage_audit", leakage_rows, LEAKAGE_FIELDS)

    next_action = (
        "Freeze E2 as contract-safe only and await a new mechanical "
        "instruction; do not automatically enter another experiment."
        if verdict == "Passed"
        else (
            "Stop and preserve the complete E2 audit. Do not tune constants, "
            "run E3, restore L/G/J, refit, or train; await a new mechanical "
            "instruction."
        )
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "registration_id": REGISTRATION_ID,
        "contract_verdict": verdict,
        "contract_valid": not validity_failures,
        "invalid_reasons": validity_failures,
        "all_three_scenes_passed": bool(
            scene_rows
            and all(row["scene_verdict"] == "Passed" for row in scene_rows)
        ),
        "prediction_frozen_before_GT_scoring": (
            barrier.prediction_frozen_before_gt_scoring
        ),
        "plane_fit_call_count": PLANE_FIT_CALL_COUNT,
        "weighted_fit_plane_call_count": WEIGHTED_FIT_PLANE_CALL_COUNT,
        "refit_call_count": REFIT_CALL_COUNT,
        "ransac_call_count": RANSAC_CALL_COUNT,
        "E1_prediction_used_as_input": False,
        "training_executed": False,
        "L_G_J_executed": False,
        "E3_executed": False,
        "scenes": scene_rows,
        "failed_planes": [
            {
                "scene": row["scene"],
                "plane_id": row["plane_id"],
                "failure_reasons": row["failure_reasons"],
            }
            for row in plane_rows
            if row["plane_verdict"] == "Failed"
        ],
        "next_action": next_action,
        "artifacts": {
            "preregistration_md": str(
                output_dir / "preregistration.md"
            ),
            "config_json": str(output_dir / "config.json"),
            "git_state_json": str(output_dir / "git_state.json"),
            "input_hashes_json": str(output_dir / "input_hashes.json"),
            "source_hashes_json": str(output_dir / "source_hashes.json"),
            "contract_validation_json": str(
                output_dir / "contract_validation.json"
            ),
            "prediction_hashes_json": str(
                output_dir / "prediction_hashes.json"
            ),
            "matcher_output_json": str(
                output_dir / "matcher_output.json"
            ),
            "per_plane_footprint_audit_csv": str(
                output_dir / "per_plane_footprint_audit.csv"
            ),
            "per_scene_metrics_csv": str(
                output_dir / "per_scene_metrics.csv"
            ),
            "multi_assignment_audit_csv": str(
                output_dir / "multi_assignment_audit.csv"
            ),
            "seed_retention_audit_csv": str(
                output_dir / "seed_retention_audit.csv"
            ),
            "leakage_audit_csv": str(
                output_dir / "leakage_audit.csv"
            ),
            "prediction_dir": str(prediction_dir),
            "run_log": str(output_dir / "run.log"),
        },
    }
    log(f"contract_verdict={verdict}")
    log(f"next_action={next_action}")
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
        "--preregistration_md",
        default=(
            "C:\\Users\\admin\\.codex\\attachments\\"
            "12cb7b9f-dd1f-47ff-af5c-a11607873daf\\pasted-text.txt"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(
            root
            / "results/evaluator_contract_e2_candidate_bounded_footprint_20260727_v1"
        ),
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
