"""Lift learned all-fragment support predictions to the common 80k cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_bias_aware_fragment_association import _artifact_path, _scene_name
from evaluate_canonical_plane_bias_oracle import weighted_fit_plane
from evaluate_global_plane_baselines import evaluate_arrays
from global_plane_baselines import load_global_cache, supports_to_primitives
from lift_support_prediction_to_global_cache import (
    load_support_prediction,
    resolve_support_assignments,
)


OUTPUT_SCHEMA_VERSION = 1


def build_tangent_bounds(
    points: np.ndarray,
    seed_assignment: np.ndarray,
    seed_normals: np.ndarray,
    *,
    trim_quantile: float,
) -> list[dict[str, np.ndarray] | None]:
    """Estimate robust 2D support boxes in each seed plane's tangent frame."""

    points = np.asarray(points, dtype=np.float64)
    seed_assignment = np.asarray(seed_assignment, dtype=np.int32)
    seed_normals = np.asarray(seed_normals, dtype=np.float64)
    if not 0.0 <= float(trim_quantile) < 0.5:
        raise ValueError("trim_quantile must be in [0, 0.5)")
    bounds = []
    for plane_id, normal in enumerate(seed_normals):
        support = points[seed_assignment == plane_id]
        if len(support) < 3:
            bounds.append(None)
            continue
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= 1e-12:
            bounds.append(None)
            continue
        normal = normal / normal_norm
        origin = np.median(support, axis=0)
        centered = support - origin
        tangent_points = centered - np.outer(centered @ normal, normal)
        _, singular_values, vh = np.linalg.svd(tangent_points, full_matrices=False)
        if len(singular_values) < 2 or singular_values[1] <= 1e-10:
            bounds.append(None)
            continue
        axes = vh[:2]
        uv = centered @ axes.T
        lower = np.quantile(uv, float(trim_quantile), axis=0)
        upper = np.quantile(uv, 1.0 - float(trim_quantile), axis=0)
        bounds.append(
            {
                "origin": origin,
                "axes": axes,
                "lower": lower,
                "upper": upper,
            }
        )
    return bounds


def assignment_to_primitives(
    points: np.ndarray,
    source_assignment: np.ndarray,
    min_points_per_plane: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    supports = []
    source_ids = []
    for source_id in sorted(
        int(value) for value in np.unique(source_assignment) if value >= 0
    ):
        indices = np.flatnonzero(source_assignment == source_id)
        if len(indices) < int(min_points_per_plane):
            continue
        supports.append(indices)
        source_ids.append(source_id)
    assignment, normals, offsets, counts = supports_to_primitives(points, supports)
    return assignment, normals, offsets, counts, source_ids


def expand_plane_seeds(
    points: np.ndarray,
    seed_normals: np.ndarray,
    seed_offsets: np.ndarray,
    *,
    distance_threshold: float,
    refit_iterations: int,
    min_points_per_plane: int,
    tangent_bounds: list[dict[str, np.ndarray] | None] | None = None,
    bound_margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(seed_normals, dtype=np.float64).copy()
    offsets = np.asarray(seed_offsets, dtype=np.float64).copy()
    active_bounds = list(tangent_bounds) if tangent_bounds is not None else None
    if active_bounds is not None and len(active_bounds) != len(normals):
        raise ValueError("tangent_bounds and seed_normals have different lengths")
    if len(normals) == 0:
        return (
            np.full((len(points),), -1, dtype=np.int32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    assignment = np.full((len(points),), -1, dtype=np.int32)
    for _ in range(max(int(refit_iterations), 0) + 1):
        distances = np.abs(points @ normals.T + offsets[None, :])
        if active_bounds is not None:
            for plane_id, bounds in enumerate(active_bounds):
                if bounds is None:
                    distances[:, plane_id] = np.inf
                    continue
                uv = (points - bounds["origin"]) @ bounds["axes"].T
                inside = np.all(
                    (uv >= bounds["lower"] - float(bound_margin))
                    & (uv <= bounds["upper"] + float(bound_margin)),
                    axis=1,
                )
                distances[~inside, plane_id] = np.inf
        nearest = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(len(points)), nearest]
        assignment.fill(-1)
        assignment[nearest_distance <= float(distance_threshold)] = nearest[
            nearest_distance <= float(distance_threshold)
        ]
        kept_ids = [
            plane_id
            for plane_id in range(len(normals))
            if int((assignment == plane_id).sum()) >= int(min_points_per_plane)
        ]
        if not kept_ids:
            return (
                np.full((len(points),), -1, dtype=np.int32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        remap = {old: new for new, old in enumerate(kept_ids)}
        compact = np.full_like(assignment, -1)
        for old, new in remap.items():
            compact[assignment == old] = new
        assignment = compact
        fitted_normals = []
        fitted_offsets = []
        for plane_id in range(len(kept_ids)):
            normal, offset = weighted_fit_plane(points[assignment == plane_id])
            fitted_normals.append(normal)
            fitted_offsets.append(offset)
        normals = np.asarray(fitted_normals, dtype=np.float64)
        offsets = np.asarray(fitted_offsets, dtype=np.float64)
        if active_bounds is not None:
            active_bounds = [active_bounds[old] for old in kept_ids]
    counts = np.asarray(
        [(assignment == plane_id).sum() for plane_id in range(len(normals))],
        dtype=np.int32,
    )
    return (
        assignment.astype(np.int32),
        normals.astype(np.float32),
        offsets.astype(np.float32),
        counts,
    )


def save_common_prediction(
    path: Path,
    cache: dict,
    assignment: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    counts: np.ndarray,
    scene: str,
    diagnostics: dict,
) -> None:
    colors = cache["colors"].astype(np.uint8)
    np.savez_compressed(
        path,
        schema_version=np.asarray(OUTPUT_SCHEMA_VERSION, dtype=np.int32),
        points=cache["points"].astype(np.float32),
        colors=colors,
        original_colors=colors,
        point_plane_ids=assignment.astype(np.int32),
        plane_ids=np.arange(len(normals), dtype=np.int32),
        plane_normals=normals.astype(np.float32),
        plane_offsets=offsets.astype(np.float32),
        plane_inlier_counts=counts.astype(np.int32),
        source_views=cache["view_indices"].astype(np.int32),
        pixel_xy=cache["pixel_xy"].astype(np.int32),
        pixel_coordinate_order=np.asarray("xy"),
        pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
        method=np.asarray("learned_all_fragment_identity_common_cache"),
        scene_key=np.asarray(scene),
        join_diagnostics_json=np.asarray(json.dumps(diagnostics, sort_keys=True)),
    )


def scalar_metrics(metrics: dict) -> dict:
    result = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    precision = float(metrics["plane_precision"])
    recall = float(metrics["plane_recall"])
    result["plane_f1"] = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {key: float(value) for key, value in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate learned all-fragment predictions on the common cache."
    )
    parser.add_argument("--batch_json", required=True)
    parser.add_argument("--support_prediction_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_conf", type=float, default=1.0)
    parser.add_argument("--min_points_per_plane", type=int, default=3)
    parser.add_argument("--expansion_distance", type=float, default=0.03)
    parser.add_argument("--refit_iterations", type=int, default=2)
    parser.add_argument("--expanded_min_points_per_plane", type=int, default=1000)
    parser.add_argument(
        "--bound_mode", choices=("none", "tangent_bbox"), default="none"
    )
    parser.add_argument("--bound_margin", type=float, default=0.05)
    parser.add_argument("--bound_trim_quantile", type=float, default=0.01)
    parser.add_argument("--match_iou", type=float, default=0.5)
    parser.add_argument("--fragmentation_iou", type=float, default=0.1)
    parser.add_argument("--min_observed_plane_points", type=int, default=64)
    args = parser.parse_args()

    batch_path = Path(args.batch_json).resolve()
    prediction_dir = Path(args.support_prediction_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    common_dir = output_dir / "predictions"
    common_dir.mkdir()
    root = batch_path
    while root.parent != root and not (root / ".git").exists():
        root = root.parent
    if not (root / ".git").exists():
        raise ValueError("Could not infer repository root from batch_json")
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    scene_rows = []
    for item in batch["items"]:
        gt_path = _artifact_path(root, item, "point_aligned_gt")
        scene = _scene_name(item, gt_path)
        cache_path = _artifact_path(root, item, "global_cloud_cache")
        support_path = prediction_dir / f"{scene}_all_fragment_identity.npz"
        cache = load_global_cache(cache_path, args.min_conf)
        support = load_support_prediction(support_path)
        with np.load(support_path, allow_pickle=False) as raw:
            seed_normals = raw["plane_normals"].astype(np.float64)
            seed_offsets = raw["plane_offsets"].astype(np.float64)
        source_assignment, diagnostics = resolve_support_assignments(
            cache["view_indices"],
            cache["pixel_xy"],
            support["view_indices"],
            support["pixel_xy"],
            support["labels"],
            "drop",
        )
        _, _, _, _, source_ids = assignment_to_primitives(
            cache["points"],
            source_assignment,
            args.min_points_per_plane,
        )
        if args.expansion_distance > 0:
            source_ids = list(range(len(seed_normals)))
            tangent_bounds = None
            if args.bound_mode == "tangent_bbox":
                tangent_bounds = build_tangent_bounds(
                    cache["points"],
                    source_assignment,
                    seed_normals,
                    trim_quantile=args.bound_trim_quantile,
                )
            assignment, normals, offsets, counts = expand_plane_seeds(
                cache["points"],
                seed_normals,
                seed_offsets,
                distance_threshold=args.expansion_distance,
                refit_iterations=args.refit_iterations,
                min_points_per_plane=args.expanded_min_points_per_plane,
                tangent_bounds=tangent_bounds,
                bound_margin=args.bound_margin,
            )
        else:
            assignment, normals, offsets, counts, source_ids = (
                assignment_to_primitives(
                    cache["points"],
                    source_assignment,
                    args.min_points_per_plane,
                )
            )
        common_path = common_dir / f"{scene}_learned_common_cache.npz"
        diagnostics["source_plane_ids_by_output_id"] = source_ids
        save_common_prediction(
            common_path,
            cache,
            assignment,
            normals,
            offsets,
            counts,
            scene,
            diagnostics,
        )
        with np.load(gt_path, allow_pickle=False) as gt:
            if not np.allclose(cache["points"], gt["points"], atol=1e-5):
                raise ValueError(f"{scene}: cache and GT points differ")
            metrics = evaluate_arrays(
                gt["points"],
                assignment,
                normals,
                offsets,
                gt["point_plane_ids"],
                gt["plane_normals"],
                args.match_iou,
                args.fragmentation_iou,
                args.min_observed_plane_points,
            )
        row = {
            "scene": scene,
            **scalar_metrics(metrics),
            "assignment_rate": float((assignment >= 0).mean()),
            "conflicting_support_keys": diagnostics["conflicting_support_keys"],
            "prediction_npz": str(common_path),
        }
        scene_rows.append(row)
        print(
            json.dumps(
                {
                    "scene": scene,
                    "plane_f1": row["plane_f1"],
                    "pairwise_f1": row["support_partition_pairwise_f1"],
                    "assignment_rate": row["assignment_rate"],
                    "conflicts": row["conflicting_support_keys"],
                }
            ),
            flush=True,
        )
    numeric_keys = [
        key
        for key, value in scene_rows[0].items()
        if isinstance(value, (int, float))
    ]
    mean = {}
    for key in numeric_keys:
        values = np.asarray([row[key] for row in scene_rows], dtype=np.float64)
        mean[key] = (
            float(np.nanmean(values))
            if np.isfinite(values).any()
            else float("nan")
        )
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "scene_count": len(scene_rows),
        "method": "learned_all_fragment_identity_common_cache",
        "mean": mean,
        "scenes": scene_rows,
        "config": {
            "batch_json": str(batch_path),
            "support_prediction_dir": str(prediction_dir),
            "min_conf": float(args.min_conf),
            "conflict_policy": "drop",
            "min_points_per_plane": int(args.min_points_per_plane),
            "expansion_distance": float(args.expansion_distance),
            "refit_iterations": int(args.refit_iterations),
            "expanded_min_points_per_plane": int(
                args.expanded_min_points_per_plane
            ),
            "bound_mode": args.bound_mode,
            "bound_margin": float(args.bound_margin),
            "bound_trim_quantile": float(args.bound_trim_quantile),
        },
        "outputs": {
            "summary_json": str(output_dir / "summary.json"),
            "prediction_dir": str(common_dir),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "mean": mean,
                "summary_json": result["outputs"]["summary_json"],
            },
            indent=2,
            allow_nan=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
