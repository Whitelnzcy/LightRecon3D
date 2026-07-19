"""Learning-support-guided plane RANSAC on a frozen DUSt3R global cache.

Stage1/Stage2 plane supports generate hypotheses.  Every hypothesis is lifted
with the exact ``(alignment_view_index, x, y)`` registry, verified against the
full global cloud, confidence-weighted refit, and split into bounded connected
components.  A reduced random RANSAC pass may cover geometry that the learned
supports missed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from global_plane_baselines import (
    euclidean_components,
    load_global_cache,
    save_result,
    sequential_plane_ransac,
    voxel_components,
)
from lift_support_prediction_to_global_cache import (
    file_sha256,
    load_support_prediction,
    pack_registry_keys,
)


METHOD = "learning_guided_ransac_cc"
SCHEMA_VERSION = 2

MECHANISM_MODES = {
    "none": {
        "proposal_guidance": False,
        "consensus_guidance": False,
        "refit_guidance": False,
    },
    "proposal_only": {
        "proposal_guidance": True,
        "consensus_guidance": False,
        "refit_guidance": False,
    },
    "consensus_only": {
        "proposal_guidance": False,
        "consensus_guidance": True,
        "refit_guidance": False,
    },
    "refit_only": {
        "proposal_guidance": False,
        "consensus_guidance": False,
        "refit_guidance": True,
    },
    "proposal_consensus": {
        "proposal_guidance": True,
        "consensus_guidance": True,
        "refit_guidance": False,
    },
    "all": {
        "proposal_guidance": True,
        "consensus_guidance": True,
        "refit_guidance": True,
    },
}

METHOD_BY_MODE = {
    "none": "support_mechanism_none_cc",
    "proposal_only": "support_mechanism_proposal_only_cc",
    "consensus_only": "support_mechanism_consensus_only_cc",
    "refit_only": "support_mechanism_refit_only_cc",
    # This is the historical B4 implementation. Keep its method identifier so
    # a default invocation remains backward compatible with archived outputs.
    "proposal_consensus": METHOD,
    "all": "support_mechanism_proposal_consensus_refit_cc",
}


def canonicalize_plane(normal: np.ndarray, offset: float) -> tuple[np.ndarray, float]:
    normal = np.asarray(normal, dtype=np.float32).reshape(3)
    pivot = int(np.argmax(np.abs(normal)))
    if normal[pivot] < 0:
        normal = -normal
        offset = -float(offset)
    return normal, float(offset)


def fit_weighted_plane(
    points: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, float, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("plane fitting requires at least three 3D points")
    if weights is None:
        weights = np.ones(len(points), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(weights) != len(points):
        raise ValueError("weights and points must have equal lengths")
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if float(weights.sum()) <= 0:
        weights = np.ones(len(points), dtype=np.float64)
    weights /= float(weights.sum())
    center = np.sum(points.astype(np.float64) * weights[:, None], axis=0)
    centered = points.astype(np.float64) - center
    covariance = np.sum(
        centered[:, :, None]
        * centered[:, None, :]
        * weights[:, None, None],
        axis=0,
    )
    ok, _, eigenvectors = cv2.eigen(covariance)
    if not ok:
        raise RuntimeError("OpenCV failed to solve the plane covariance eigensystem")
    normal = eigenvectors[-1].astype(np.float32)
    normal /= max(float(np.sqrt(np.sum(normal * normal))), 1e-12)
    offset = -float(np.sum(normal * center))
    normal, offset = canonicalize_plane(normal, offset)
    residual = np.abs(np.sum(points * normal[None], axis=1) + offset).astype(np.float32)
    return normal, offset, residual


def map_support_records_to_cache(
    cache_view_indices: np.ndarray,
    cache_pixel_xy: np.ndarray,
    support_view_indices: np.ndarray,
    support_pixel_xy: np.ndarray,
    support_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map every positive record exactly, preserving duplicate/conflict rows."""

    cache_keys = pack_registry_keys(
        cache_view_indices, cache_pixel_xy, "global cache"
    )
    _, cache_counts = np.unique(cache_keys, return_counts=True)
    duplicate_cache_keys = int((cache_counts > 1).sum())
    if duplicate_cache_keys:
        raise ValueError(
            "global cache registry is not one-to-one: "
            f"{duplicate_cache_keys} duplicate (view,x,y) keys"
        )

    labels = np.asarray(support_labels, dtype=np.int32).reshape(-1)
    views = np.asarray(support_view_indices, dtype=np.int64).reshape(-1)
    pixels = np.asarray(support_pixel_xy, dtype=np.int64)
    if len(labels) != len(views) or pixels.shape != (len(labels), 2):
        raise ValueError("support labels/views/pixels must have (N,)/(N,)/(N,2) shapes")
    positive = labels >= 0
    positive_keys = pack_registry_keys(views[positive], pixels[positive], "positive support")
    positive_labels = labels[positive]

    cache_order = np.argsort(cache_keys, kind="stable")
    sorted_cache_keys = cache_keys[cache_order]
    positions = np.searchsorted(sorted_cache_keys, positive_keys)
    matched = positions < len(sorted_cache_keys)
    if matched.any():
        matched_positions = positions[matched]
        matched[matched] = sorted_cache_keys[matched_positions] == positive_keys[matched]
    cache_indices = cache_order[positions[matched]]
    matched_labels = positive_labels[matched]

    unique_keys, inverse, key_counts = np.unique(
        positive_keys, return_inverse=True, return_counts=True
    )
    conflicts = 0
    conflicting_records = 0
    for key_index in range(len(unique_keys)):
        key_labels = positive_labels[inverse == key_index]
        if len(np.unique(key_labels)) > 1:
            conflicts += 1
            conflicting_records += int(len(key_labels))
    diagnostics = {
        "coordinate_order": "xy",
        "coordinate_space": "dust3r_aligned_pointmap",
        "join_key": "(alignment_view_index,x,y)",
        "duplicate_conflicts_preserved_as_competing_hypotheses": True,
        "cache_points": int(len(cache_keys)),
        "support_records": int(len(labels)),
        "positive_support_records": int(positive.sum()),
        "unique_positive_support_keys": int(len(unique_keys)),
        "duplicate_positive_support_records": int((key_counts - 1).sum()),
        "conflicting_support_keys": int(conflicts),
        "conflicting_support_records": int(conflicting_records),
        "matched_positive_support_records": int(matched.sum()),
        "unmatched_positive_support_records": int((~matched).sum()),
        "matched_unique_cache_points": int(len(np.unique(cache_indices))),
    }
    return cache_indices.astype(np.int64), matched_labels.astype(np.int32), diagnostics


def build_support_groups(
    cache: dict[str, np.ndarray], support: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic label-to-cache memberships for mechanism controls.

    Duplicate records with the same ``(label, cache index)`` are collapsed so
    repeated observations cannot inflate a score. A conflicting cache point is
    retained once in every competing label group, matching the historical B4
    competing-hypothesis contract.
    """

    mapped_indices, mapped_labels, mapping = map_support_records_to_cache(
        cache["view_indices"],
        cache["pixel_xy"],
        support["view_indices"],
        support["pixel_xy"],
        support["labels"],
    )
    groups: list[dict[str, Any]] = []
    for source_plane_id in sorted(int(value) for value in np.unique(mapped_labels)):
        groups.append(
            {
                "source_plane_id": source_plane_id,
                "support_indices": np.unique(
                    mapped_indices[mapped_labels == source_plane_id]
                ).astype(np.int64),
            }
        )
    diagnostics = {
        "mapping": mapping,
        "source_plane_labels": int(len(groups)),
        "unique_label_cache_memberships": int(
            sum(len(group["support_indices"]) for group in groups)
        ),
    }
    return groups, diagnostics


def pack_support_groups(
    groups: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack group memberships for vectorized consensus scoring."""

    if not groups:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    cache_indices = np.concatenate(
        [np.asarray(group["support_indices"], dtype=np.int64) for group in groups]
    )
    group_indices = np.concatenate(
        [
            np.full(len(group["support_indices"]), index, dtype=np.int32)
            for index, group in enumerate(groups)
        ]
    )
    source_plane_ids = np.asarray(
        [int(group["source_plane_id"]) for group in groups], dtype=np.int32
    )
    return cache_indices, group_indices, source_plane_ids


def best_support_group_for_plane(
    points: np.ndarray,
    membership_cache_indices: np.ndarray,
    membership_group_indices: np.ndarray,
    source_plane_ids: np.ndarray,
    remaining_mask: np.ndarray,
    normal: np.ndarray,
    offset: float,
    distance_threshold: float,
) -> tuple[int, int, int]:
    """Return ``(group index, source plane id, active inlier memberships)``."""

    if not len(source_plane_ids) or not len(membership_cache_indices):
        return -1, -1, 0
    active = remaining_mask[membership_cache_indices]
    if not active.any():
        return -1, -1, 0
    active_cache_indices = membership_cache_indices[active]
    residual = np.abs(
        np.sum(points[active_cache_indices] * normal[None], axis=1) + offset
    )
    inlier_groups = membership_group_indices[active][
        residual <= float(distance_threshold)
    ]
    if not len(inlier_groups):
        return -1, -1, 0
    counts = np.bincount(inlier_groups, minlength=len(source_plane_ids))
    best_count = int(counts.max())
    # np.argmax gives the lowest deterministic group index on a tie. Groups
    # are ordered by source_plane_id, so this matches the historical tie rule.
    best_group_index = int(np.argmax(counts))
    return (
        best_group_index,
        int(source_plane_ids[best_group_index]),
        best_count,
    )


def refit_global_inliers(
    cache: dict[str, np.ndarray],
    global_indices: np.ndarray,
    *,
    support_indices: np.ndarray | None = None,
    support_refit_weight: float = 0.0,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Confidence-refit a plane, optionally boosting one coherent support.

    DUSt3R confidence weighting is shared by every mechanism row and is not a
    learned-support intervention. With a positive ``support_refit_weight``, a
    point in the selected predicted-support group receives multiplier
    ``1 + support_refit_weight``. All global inliers remain in the fit.
    """

    global_indices = np.asarray(global_indices, dtype=np.int64).reshape(-1)
    weights = np.asarray(cache["confidence"][global_indices], dtype=np.float64)
    supported = np.zeros(len(global_indices), dtype=bool)
    if support_indices is not None and float(support_refit_weight) > 0:
        support_indices = np.asarray(support_indices, dtype=np.int64).reshape(-1)
        # A direct cache-index mask is substantially faster than np.isin for
        # 700k-point caches on Windows and has identical set-membership
        # semantics. Allocate it only for the few selected refit hypotheses.
        support_mask = np.zeros(len(cache["points"]), dtype=bool)
        support_mask[support_indices] = True
        supported = support_mask[global_indices]
        weights *= 1.0 + float(support_refit_weight) * supported.astype(np.float64)
    normal, offset, _ = fit_weighted_plane(cache["points"][global_indices], weights)
    diagnostics = {
        "global_inliers": int(len(global_indices)),
        "support_guided_inliers": int(supported.sum()),
        "support_refit_weight": float(support_refit_weight),
        "support_weight_multiplier": float(1.0 + support_refit_weight),
    }
    return normal, offset, diagnostics


def robust_support_plane(
    points: np.ndarray,
    confidence: np.ndarray,
    *,
    distance_threshold: float,
    iterations: int,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit one proposal robustly; the learned mask replaces global sampling."""

    points = np.asarray(points, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if len(points) > int(max_points):
        sample_indices = rng.choice(len(points), int(max_points), replace=False)
        score_points = points[sample_indices]
        score_confidence = confidence[sample_indices]
    else:
        score_points = points
        score_confidence = confidence

    normal, offset, residual = fit_weighted_plane(score_points, score_confidence)
    best = (int((residual <= distance_threshold).sum()), -float(np.median(residual)), normal, offset)
    for _ in range(max(0, int(iterations))):
        sample = score_points[rng.choice(len(score_points), 3, replace=False)]
        candidate_normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.sqrt(np.sum(candidate_normal * candidate_normal)))
        if norm < 1e-8:
            continue
        candidate_normal = (candidate_normal / norm).astype(np.float32)
        candidate_offset = -float(np.sum(candidate_normal * sample[0]))
        candidate_normal, candidate_offset = canonicalize_plane(
            candidate_normal, candidate_offset
        )
        candidate_residual = np.abs(
            np.sum(score_points * candidate_normal[None], axis=1) + candidate_offset
        )
        score = (
            int((candidate_residual <= distance_threshold).sum()),
            -float(np.median(candidate_residual)),
            candidate_normal,
            candidate_offset,
        )
        if score[:2] > best[:2]:
            best = score

    normal, offset = best[2], best[3]
    all_residual = np.abs(np.sum(points * normal[None], axis=1) + offset)
    inliers = all_residual <= distance_threshold
    if int(inliers.sum()) >= 3:
        normal, offset, _ = fit_weighted_plane(points[inliers], confidence[inliers])
        all_residual = np.abs(np.sum(points * normal[None], axis=1) + offset)
        inliers = all_residual <= distance_threshold
    return normal, offset, inliers


def build_guided_hypotheses(
    cache: dict[str, np.ndarray],
    support: dict[str, np.ndarray],
    *,
    distance_threshold: float,
    proposal_iterations: int,
    proposal_min_points: int,
    proposal_min_inlier_ratio: float,
    proposal_max_points: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups, group_diagnostics = build_support_groups(cache, support)
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    rejected_small = 0
    rejected_nonplanar = 0
    for group in groups:
        source_plane_id = int(group["source_plane_id"])
        support_indices = group["support_indices"]
        if len(support_indices) < int(proposal_min_points):
            rejected_small += 1
            continue
        normal, offset, proposal_inliers = robust_support_plane(
            cache["points"][support_indices],
            cache["confidence"][support_indices],
            distance_threshold=distance_threshold,
            iterations=proposal_iterations,
            max_points=proposal_max_points,
            rng=rng,
        )
        inlier_indices = support_indices[proposal_inliers]
        inlier_ratio = float(len(inlier_indices) / len(support_indices))
        if (
            len(inlier_indices) < int(proposal_min_points)
            or inlier_ratio < float(proposal_min_inlier_ratio)
        ):
            rejected_nonplanar += 1
            continue
        candidates.append(
            {
                "source_plane_id": source_plane_id,
                "normal": normal,
                "offset": float(offset),
                "support_indices": inlier_indices,
                "support_points": int(len(support_indices)),
                "support_inlier_points": int(len(inlier_indices)),
                "support_inlier_ratio": inlier_ratio,
                "source_views": int(len(np.unique(cache["view_indices"][support_indices]))),
            }
        )
    diagnostics = {
        **group_diagnostics,
        "accepted_hypotheses": int(len(candidates)),
        "rejected_small_hypotheses": int(rejected_small),
        "rejected_nonplanar_hypotheses": int(rejected_nonplanar),
    }
    return candidates, diagnostics


def split_bounded_components(
    points: np.ndarray,
    indices: np.ndarray,
    *,
    cluster_radius: float,
    min_component_points: int,
    component_exact_max_points: int,
) -> list[np.ndarray]:
    if len(indices) <= int(component_exact_max_points):
        components = euclidean_components(
            points[indices], cluster_radius, min_component_points
        )
    else:
        components = voxel_components(
            points[indices], cluster_radius, min_component_points
        )
    return [indices[component] for component in components]


def guided_sequential_plane_ransac(
    cache: dict[str, np.ndarray],
    candidates: list[dict[str, Any]],
    *,
    distance_threshold: float,
    min_inliers: int,
    cluster_radius: float,
    min_component_points: int,
    max_planes: int,
    seed: int,
    hypothesis_max_points: int,
    component_exact_max_points: int,
    support_score_weight: float,
    fallback_iterations: int,
    support_refit_weight: float = 0.0,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    points = cache["points"]
    confidence = cache["confidence"]
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(points), dtype=np.int64)
    remaining_mask = np.ones(len(points), dtype=bool)
    unused = set(range(len(candidates)))
    supports: list[np.ndarray] = []
    selections: list[dict[str, Any]] = []

    while unused and len(remaining) >= int(min_inliers) and len(supports) < int(max_planes):
        xyz = points[remaining]
        if len(xyz) > int(hypothesis_max_points):
            local = rng.choice(len(xyz), int(hypothesis_max_points), replace=False)
            score_xyz = xyz[local]
        else:
            score_xyz = xyz
        best_index = None
        best_key = None
        best_counts = None
        for candidate_index in sorted(unused):
            candidate = candidates[candidate_index]
            normal, offset = candidate["normal"], candidate["offset"]
            global_sample_count = int(
                (
                    np.abs(np.sum(score_xyz * normal[None], axis=1) + offset)
                    <= distance_threshold
                ).sum()
            )
            estimated_global_count = float(
                global_sample_count * len(xyz) / max(1, len(score_xyz))
            )
            active_support = candidate["support_indices"]
            active_support = active_support[remaining_mask[active_support]]
            active_support_count = int(
                (
                    np.abs(
                        np.sum(points[active_support] * normal[None], axis=1)
                        + offset
                    )
                    <= distance_threshold
                ).sum()
            )
            combined_score = (
                estimated_global_count
                + float(support_score_weight) * active_support_count
            )
            key = (
                combined_score,
                estimated_global_count,
                active_support_count,
                -int(candidate["source_plane_id"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = candidate_index
                best_counts = (estimated_global_count, active_support_count)
        assert best_index is not None and best_counts is not None
        unused.remove(best_index)
        candidate = candidates[best_index]
        normal, offset = candidate["normal"], candidate["offset"]
        inliers = (
            np.abs(np.sum(points[remaining] * normal[None], axis=1) + offset)
            <= distance_threshold
        )
        if int(inliers.sum()) < int(min_inliers):
            continue
        global_indices = remaining[inliers]
        normal, offset, refit_diagnostics = refit_global_inliers(
            cache,
            global_indices,
            support_indices=candidate["support_indices"],
            support_refit_weight=support_refit_weight,
        )
        refined = (
            np.abs(np.sum(points[remaining] * normal[None], axis=1) + offset)
            <= distance_threshold
        )
        global_indices = remaining[refined]
        if len(global_indices) < int(min_inliers):
            continue
        components = split_bounded_components(
            points,
            global_indices,
            cluster_radius=cluster_radius,
            min_component_points=min_component_points,
            component_exact_max_points=component_exact_max_points,
        )
        accepted = components[: max(0, int(max_planes) - len(supports))]
        supports.extend(accepted)
        selections.append(
            {
                "source_plane_id": int(candidate["source_plane_id"]),
                "estimated_global_inliers": float(best_counts[0]),
                "active_learned_support_inliers": int(best_counts[1]),
                "full_global_inliers": int(len(global_indices)),
                "accepted_components": int(len(accepted)),
                "accepted_component_points": [int(len(value)) for value in accepted],
                "refit": refit_diagnostics,
            }
        )
        # Match the baseline contract: remove all infinite-plane inliers so a
        # dominant plane cannot be rediscovered under another local proposal.
        removed = remaining[refined]
        remaining_mask[removed] = False
        remaining = remaining[~refined]

    guided_components = len(supports)
    if (
        int(fallback_iterations) > 0
        and len(remaining) >= int(min_inliers)
        and len(supports) < int(max_planes)
    ):
        fallback = sequential_plane_ransac(
            points[remaining],
            distance_threshold=distance_threshold,
            iterations=fallback_iterations,
            min_inliers=min_inliers,
            cluster_radius=cluster_radius,
            min_component_points=min_component_points,
            max_planes=max_planes - len(supports),
            seed=seed + 1,
            hypothesis_max_points=hypothesis_max_points,
            component_exact_max_points=component_exact_max_points,
        )
        supports.extend(remaining[indices] for indices in fallback)
    diagnostics = {
        "guided_hypotheses_selected": int(len(selections)),
        "guided_components": int(guided_components),
        "fallback_components": int(len(supports) - guided_components),
        "output_components": int(len(supports)),
        "remaining_points_before_fallback": int(len(remaining)),
        "support_score_weight": float(support_score_weight),
        "support_refit_weight": float(support_refit_weight),
        "selections": selections,
    }
    return supports, diagnostics


def global_mechanism_plane_ransac(
    cache: dict[str, np.ndarray],
    support_groups: list[dict[str, Any]],
    *,
    distance_threshold: float,
    global_proposal_iterations: int,
    min_inliers: int,
    cluster_radius: float,
    min_component_points: int,
    max_planes: int,
    seed: int,
    hypothesis_max_points: int,
    component_exact_max_points: int,
    consensus_guidance: bool,
    refit_guidance: bool,
    support_score_weight: float,
    support_refit_weight: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Global-proposal matched control with optional support interventions.

    Random proposal generation and DUSt3R-confidence refit are shared across
    ``none``, ``consensus_only``, and ``refit_only``. Consensus guidance adds
    the largest coherent predicted-label inlier count to candidate selection.
    Refit guidance boosts only that same coherent label during the otherwise
    global confidence-weighted fit.
    """

    points = cache["points"]
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(points), dtype=np.int64)
    remaining_mask = np.ones(len(points), dtype=bool)
    supports: list[np.ndarray] = []
    selections: list[dict[str, Any]] = []
    membership_cache_indices, membership_group_indices, source_plane_ids = (
        pack_support_groups(support_groups)
    )
    evaluate_support = bool(consensus_guidance or refit_guidance)

    while len(remaining) >= int(min_inliers) and len(supports) < int(max_planes):
        xyz = points[remaining]
        if len(xyz) > int(hypothesis_max_points):
            score_local_indices = rng.choice(
                len(xyz), int(hypothesis_max_points), replace=False
            )
            score_xyz = xyz[score_local_indices]
        else:
            score_xyz = xyz

        best_model: tuple[np.ndarray, float] | None = None
        best_key: tuple[float, float, int, int, int] | None = None
        best_group_index = -1
        best_source_plane_id = -1
        best_support_count = 0
        for hypothesis_index in range(int(global_proposal_iterations)):
            sample = score_xyz[rng.choice(len(score_xyz), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = float(np.sqrt(np.sum(normal * normal)))
            if norm < 1e-8:
                continue
            normal = (normal / norm).astype(np.float32)
            offset = -float(np.sum(normal * sample[0]))
            normal, offset = canonicalize_plane(normal, offset)
            global_sample_count = int(
                (
                    np.abs(np.sum(score_xyz * normal[None], axis=1) + offset)
                    <= distance_threshold
                ).sum()
            )
            estimated_global_count = float(
                global_sample_count * len(xyz) / max(1, len(score_xyz))
            )
            group_index, source_plane_id, support_count = -1, -1, 0
            if evaluate_support:
                group_index, source_plane_id, support_count = (
                    best_support_group_for_plane(
                        points,
                        membership_cache_indices,
                        membership_group_indices,
                        source_plane_ids,
                        remaining_mask,
                        normal,
                        offset,
                        distance_threshold,
                    )
                )
            combined_score = estimated_global_count
            if consensus_guidance:
                combined_score += float(support_score_weight) * support_count
            key = (
                float(combined_score),
                estimated_global_count,
                int(support_count if consensus_guidance else 0),
                -int(source_plane_id),
                -int(hypothesis_index),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_model = (normal, offset)
                best_group_index = int(group_index)
                best_source_plane_id = int(source_plane_id)
                best_support_count = int(support_count)

        if best_model is None or best_key is None:
            break
        normal, offset = best_model
        inliers = (
            np.abs(np.sum(points[remaining] * normal[None], axis=1) + offset)
            <= distance_threshold
        )
        if int(inliers.sum()) < int(min_inliers):
            break
        global_indices = remaining[inliers]
        selected_support_indices = None
        active_refit_weight = 0.0
        if refit_guidance and best_group_index >= 0:
            selected_support_indices = support_groups[best_group_index][
                "support_indices"
            ]
            active_refit_weight = float(support_refit_weight)
        normal, offset, refit_diagnostics = refit_global_inliers(
            cache,
            global_indices,
            support_indices=selected_support_indices,
            support_refit_weight=active_refit_weight,
        )
        refined = (
            np.abs(np.sum(points[remaining] * normal[None], axis=1) + offset)
            <= distance_threshold
        )
        global_indices = remaining[refined]
        if len(global_indices) < int(min_inliers):
            break
        components = split_bounded_components(
            points,
            global_indices,
            cluster_radius=cluster_radius,
            min_component_points=min_component_points,
            component_exact_max_points=component_exact_max_points,
        )
        accepted = components[: max(0, int(max_planes) - len(supports))]
        supports.extend(accepted)
        selections.append(
            {
                "proposal_source": "global_random",
                "source_plane_id": int(best_source_plane_id),
                "estimated_global_inliers": float(best_key[1]),
                "active_learned_support_inliers": int(best_support_count),
                "combined_score": float(best_key[0]),
                "full_global_inliers": int(len(global_indices)),
                "accepted_components": int(len(accepted)),
                "accepted_component_points": [int(len(value)) for value in accepted],
                "refit": refit_diagnostics,
            }
        )
        removed = remaining[refined]
        remaining_mask[removed] = False
        remaining = remaining[~refined]

    diagnostics = {
        "proposal_source": "global_random",
        "global_proposal_iterations": int(global_proposal_iterations),
        "consensus_guidance": bool(consensus_guidance),
        "refit_guidance": bool(refit_guidance),
        "support_score_weight": float(
            support_score_weight if consensus_guidance else 0.0
        ),
        "support_refit_weight": float(
            support_refit_weight if refit_guidance else 0.0
        ),
        "support_groups": int(len(support_groups)),
        "support_label_cache_memberships": int(len(membership_cache_indices)),
        "output_components": int(len(supports)),
        "remaining_points": int(len(remaining)),
        "selections": selections,
    }
    return supports, diagnostics


def serializable_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"normal", "support_indices"}
        }
        | {"normal": np.asarray(candidate["normal"]).tolist()}
        for candidate in candidates
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        "Learning-support-guided sequential RANSAC on a shared global cache"
    )
    parser.add_argument("--global_cloud_npz", required=True)
    parser.add_argument("--support_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_key", default="scene")
    parser.add_argument("--min_conf", type=float, default=1.0)
    parser.add_argument("--distance_threshold", type=float, default=0.03)
    parser.add_argument("--proposal_iterations", type=int, default=64)
    parser.add_argument("--proposal_min_points", type=int, default=64)
    parser.add_argument("--proposal_min_inlier_ratio", type=float, default=0.60)
    parser.add_argument("--proposal_max_points", type=int, default=4000)
    parser.add_argument(
        "--mechanism_mode",
        choices=sorted(MECHANISM_MODES),
        default="proposal_consensus",
        help=(
            "Frozen support intervention. The default reproduces historical B4: "
            "support proposals plus support-aware candidate scoring, followed by "
            "a global DUSt3R-confidence refit."
        ),
    )
    parser.add_argument("--global_proposal_iterations", type=int, default=300)
    parser.add_argument("--support_score_weight", type=float, default=1.0)
    parser.add_argument("--support_refit_weight", type=float, default=1.0)
    parser.add_argument("--fallback_iterations", type=int, default=96)
    parser.add_argument("--min_inliers", type=int, default=2000)
    parser.add_argument("--cluster_radius", type=float, default=0.08)
    parser.add_argument("--min_component_points", type=int, default=1000)
    parser.add_argument("--max_planes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hypothesis_max_points", type=int, default=50000)
    parser.add_argument("--component_exact_max_points", type=int, default=20000)
    args = parser.parse_args()
    if not 0.0 <= args.proposal_min_inlier_ratio <= 1.0:
        parser.error("--proposal_min_inlier_ratio must be in [0,1]")
    if args.global_proposal_iterations <= 0:
        parser.error("--global_proposal_iterations must be positive")
    if args.support_score_weight < 0:
        parser.error("--support_score_weight must be non-negative")
    if args.support_refit_weight < 0:
        parser.error("--support_refit_weight must be non-negative")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        parser.error(f"refusing to overwrite existing output: {output_dir}")

    cache = load_global_cache(args.global_cloud_npz, args.min_conf)
    support = load_support_prediction(args.support_npz)
    started = time.perf_counter()
    mechanism = MECHANISM_MODES[args.mechanism_mode]
    if mechanism["proposal_guidance"]:
        candidates, proposal_diagnostics = build_guided_hypotheses(
            cache,
            support,
            distance_threshold=args.distance_threshold,
            proposal_iterations=args.proposal_iterations,
            proposal_min_points=args.proposal_min_points,
            proposal_min_inlier_ratio=args.proposal_min_inlier_ratio,
            proposal_max_points=args.proposal_max_points,
            seed=args.seed,
        )
        supports, extraction_diagnostics = guided_sequential_plane_ransac(
            cache,
            candidates,
            distance_threshold=args.distance_threshold,
            min_inliers=args.min_inliers,
            cluster_radius=args.cluster_radius,
            min_component_points=args.min_component_points,
            max_planes=args.max_planes,
            seed=args.seed,
            hypothesis_max_points=args.hypothesis_max_points,
            component_exact_max_points=args.component_exact_max_points,
            support_score_weight=(
                args.support_score_weight
                if mechanism["consensus_guidance"]
                else 0.0
            ),
            fallback_iterations=args.fallback_iterations,
            support_refit_weight=(
                args.support_refit_weight if mechanism["refit_guidance"] else 0.0
            ),
        )
    else:
        support_groups, group_diagnostics = build_support_groups(cache, support)
        candidates = []
        proposal_diagnostics = {
            **group_diagnostics,
            "proposal_source": "global_random",
            "accepted_hypotheses": None,
        }
        supports, extraction_diagnostics = global_mechanism_plane_ransac(
            cache,
            support_groups,
            distance_threshold=args.distance_threshold,
            global_proposal_iterations=args.global_proposal_iterations,
            min_inliers=args.min_inliers,
            cluster_radius=args.cluster_radius,
            min_component_points=args.min_component_points,
            max_planes=args.max_planes,
            seed=args.seed,
            hypothesis_max_points=args.hypothesis_max_points,
            component_exact_max_points=args.component_exact_max_points,
            consensus_guidance=mechanism["consensus_guidance"],
            refit_guidance=mechanism["refit_guidance"],
            support_score_weight=args.support_score_weight,
            support_refit_weight=args.support_refit_weight,
        )
    runtime = time.perf_counter() - started
    diagnostics = {
        "mechanism": {
            "schema_version": 1,
            "mode": args.mechanism_mode,
            **mechanism,
            "shared_refit_weighting": "dust3r_confidence",
            "support_refit_rule": (
                "selected_label_inliers_receive_multiplier_1_plus_weight"
                if mechanism["refit_guidance"]
                else "disabled"
            ),
            "historical_b4_equivalent": args.mechanism_mode
            == "proposal_consensus",
        },
        "proposal": proposal_diagnostics,
        "extraction": extraction_diagnostics,
        "candidates": serializable_candidates(candidates),
    }
    config = vars(args).copy()
    config["diagnostics"] = diagnostics
    method = METHOD_BY_MODE[args.mechanism_mode]
    row = save_result(
        output_dir,
        args.scene_key,
        cache,
        supports,
        method,
        runtime,
        config,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **row,
        "global_cloud_npz": str(Path(args.global_cloud_npz)),
        "global_cloud_sha256": file_sha256(args.global_cloud_npz),
        "support_npz": str(Path(args.support_npz)),
        "support_npz_sha256": file_sha256(args.support_npz),
        "mechanism": diagnostics["mechanism"],
        "diagnostics": diagnostics,
    }
    manifest_path = output_dir / "guided_plane_ransac_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({**row, "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
