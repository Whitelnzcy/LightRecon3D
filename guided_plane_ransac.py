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
SCHEMA_VERSION = 1


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
    covariance = (centered * weights[:, None]).T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0].astype(np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    offset = -float(normal @ center)
    normal, offset = canonicalize_plane(normal, offset)
    residual = np.abs(points @ normal + offset).astype(np.float32)
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
        norm = float(np.linalg.norm(candidate_normal))
        if norm < 1e-8:
            continue
        candidate_normal = (candidate_normal / norm).astype(np.float32)
        candidate_offset = -float(candidate_normal @ sample[0])
        candidate_normal, candidate_offset = canonicalize_plane(
            candidate_normal, candidate_offset
        )
        candidate_residual = np.abs(
            score_points @ candidate_normal + candidate_offset
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
    all_residual = np.abs(points @ normal + offset)
    inliers = all_residual <= distance_threshold
    if int(inliers.sum()) >= 3:
        normal, offset, _ = fit_weighted_plane(points[inliers], confidence[inliers])
        all_residual = np.abs(points @ normal + offset)
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
    mapped_indices, mapped_labels, mapping = map_support_records_to_cache(
        cache["view_indices"],
        cache["pixel_xy"],
        support["view_indices"],
        support["pixel_xy"],
        support["labels"],
    )
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    rejected_small = 0
    rejected_nonplanar = 0
    for source_plane_id in sorted(int(value) for value in np.unique(mapped_labels)):
        support_indices = np.unique(mapped_indices[mapped_labels == source_plane_id])
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
        "mapping": mapping,
        "source_plane_labels": int(len(np.unique(mapped_labels))),
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
                (np.abs(score_xyz @ normal + offset) <= distance_threshold).sum()
            )
            estimated_global_count = float(
                global_sample_count * len(xyz) / max(1, len(score_xyz))
            )
            active_support = candidate["support_indices"]
            active_support = active_support[remaining_mask[active_support]]
            active_support_count = int(
                (
                    np.abs(points[active_support] @ normal + offset)
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
        inliers = np.abs(points[remaining] @ normal + offset) <= distance_threshold
        if int(inliers.sum()) < int(min_inliers):
            continue
        global_indices = remaining[inliers]
        normal, offset, _ = fit_weighted_plane(
            points[global_indices], confidence[global_indices]
        )
        refined = np.abs(points[remaining] @ normal + offset) <= distance_threshold
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
    parser.add_argument("--support_score_weight", type=float, default=1.0)
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
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        parser.error(f"refusing to overwrite existing output: {output_dir}")

    cache = load_global_cache(args.global_cloud_npz, args.min_conf)
    support = load_support_prediction(args.support_npz)
    started = time.perf_counter()
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
        support_score_weight=args.support_score_weight,
        fallback_iterations=args.fallback_iterations,
    )
    runtime = time.perf_counter() - started
    diagnostics = {
        "proposal": proposal_diagnostics,
        "extraction": extraction_diagnostics,
        "candidates": serializable_candidates(candidates),
    }
    config = vars(args).copy()
    config["diagnostics"] = diagnostics
    row = save_result(
        output_dir,
        args.scene_key,
        cache,
        supports,
        METHOD,
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
