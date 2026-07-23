"""PlaneGraph-BA v0: structural feedback for frozen global pointmaps.

This module does not fine-tune DUSt3R.  It consumes one method-independent
DUSt3R global-cloud cache plus plane supports already mapped by Stage3, then
alternates between global plane refits and small per-view Sim(3) corrections.
The reference view is fixed, and quadratic priors keep every correction close
to the frozen foundation-model alignment.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from global_plane_baselines import PLANE_COLORS


OUTPUT_SCHEMA_VERSION = 1


def rodrigues(rotvec):
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        x, y, z = rotvec
        skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + skew
    axis = rotvec / theta
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def apply_sim3(points, parameters):
    """Apply x' = exp(log_scale) R(rotvec) x + translation."""
    points = np.asarray(points, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    rotation = rodrigues(parameters[:3])
    scale = math.exp(float(parameters[6]))
    return (scale * (points @ rotation.T) + parameters[3:6]).astype(np.float64)


def weighted_fit_plane(points, weights=None):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("A plane requires at least three points")
    if weights is None:
        weights = np.ones((len(points),), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).clip(1e-8)
    weights /= weights.sum()
    center = (points * weights[:, None]).sum(axis=0)
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    offset = -float(normal @ center)
    return normal.astype(np.float64), offset


def _pack_keys(view_indices, pixel_xy):
    view_indices = np.asarray(view_indices, dtype=np.int64)
    pixel_xy = np.asarray(pixel_xy, dtype=np.int64)
    if (view_indices < 0).any() or (pixel_xy < 0).any() or (pixel_xy >= (1 << 21)).any():
        raise ValueError("View indices and pointmap pixels must be non-negative and smaller than 2^21")
    return (view_indices << 42) | (pixel_xy[:, 0] << 21) | pixel_xy[:, 1]


def load_inputs(global_cloud_npz, support_npz, min_conf=0.0, min_plane_points=64):
    cache_raw = np.load(global_cloud_npz, allow_pickle=False)
    cache_required = {"points", "colors", "confidence", "view_indices", "pixel_xy"}
    missing = sorted(cache_required - set(cache_raw.files))
    if missing:
        raise ValueError(f"global cloud cache missing fields: {missing}")
    points = cache_raw["points"].astype(np.float64)
    confidence = cache_raw["confidence"].astype(np.float64).reshape(-1)
    valid_cache = np.isfinite(points).all(axis=1) & (confidence >= float(min_conf))
    cache = {
        "points": points[valid_cache],
        "colors": cache_raw["colors"].astype(np.uint8)[valid_cache],
        "confidence": confidence[valid_cache],
        "view_indices": cache_raw["view_indices"].astype(np.int32)[valid_cache],
        "pixel_xy": cache_raw["pixel_xy"].astype(np.int32)[valid_cache],
    }

    support_raw = np.load(support_npz, allow_pickle=True)
    required = {"point_plane_ids"}
    missing = sorted(required - set(support_raw.files))
    if missing:
        raise ValueError(f"support NPZ missing fields: {missing}")
    if "alignment_view_indices" in support_raw and "pointmap_pixel_xy" in support_raw:
        support_views = support_raw["alignment_view_indices"].astype(np.int32)
        support_pixels = support_raw["pointmap_pixel_xy"].astype(np.int32)
    elif "source_views" in support_raw and "pixel_xy" in support_raw:
        # The global RANSAC baseline already uses the global-cache convention.
        support_views = support_raw["source_views"].astype(np.int32)
        support_pixels = support_raw["pixel_xy"].astype(np.int32)
    elif "view_indices" in support_raw and "pixel_xy" in support_raw:
        # Point-aligned Structured3D GT uses the cache's canonical provenance.
        support_views = support_raw["view_indices"].astype(np.int32)
        support_pixels = support_raw["pixel_xy"].astype(np.int32)
    else:
        raise ValueError(
            "support NPZ needs one of alignment_view_indices + pointmap_pixel_xy, "
            "source_views + pixel_xy, or view_indices + pixel_xy"
        )
    support_labels = support_raw["point_plane_ids"].astype(np.int64).reshape(-1)
    if len(support_views) != len(support_labels) or support_pixels.shape != (len(support_labels), 2):
        raise ValueError("support provenance length/shape does not match point_plane_ids")

    positive = support_labels >= 0
    support_views = support_views[positive]
    support_pixels = support_pixels[positive]
    support_labels = support_labels[positive]
    cache_keys = _pack_keys(cache["view_indices"], cache["pixel_xy"])
    support_keys = _pack_keys(support_views, support_pixels)
    order = np.argsort(cache_keys)
    sorted_keys = cache_keys[order]
    positions = np.searchsorted(sorted_keys, support_keys)
    mapped = positions < len(sorted_keys)
    mapped[mapped] &= sorted_keys[positions[mapped]] == support_keys[mapped]
    cache_indices = order[positions[mapped]]
    source_labels = support_labels[mapped]

    # Overlapping Stage2 records may repeat an observation.  Preserve a point
    # at a true plane intersection, but remove exact (point, plane) duplicates.
    pairs = np.column_stack((cache_indices, source_labels))
    pairs = np.unique(pairs, axis=0)
    cache_indices, source_labels = pairs[:, 0], pairs[:, 1]
    kept_source_ids = []
    for plane_id in np.unique(source_labels):
        if int((source_labels == plane_id).sum()) >= int(min_plane_points):
            kept_source_ids.append(int(plane_id))
    if not kept_source_ids:
        raise ValueError("No mapped plane support satisfies min_plane_points")
    id_map = {source_id: new_id for new_id, source_id in enumerate(kept_source_ids)}
    keep = np.isin(source_labels, kept_source_ids)
    cache_indices = cache_indices[keep].astype(np.int64)
    plane_ids = np.asarray([id_map[int(value)] for value in source_labels[keep]], dtype=np.int32)
    return cache, {
        "cache_indices": cache_indices,
        "plane_ids": plane_ids,
        "source_plane_ids": np.asarray(kept_source_ids, dtype=np.int64),
        "requested_observations": int(positive.sum()),
        "mapped_observations": int(mapped.sum()),
        "unique_observations": int(len(cache_indices)),
    }


def _confidence_weights(confidence):
    confidence = np.asarray(confidence, dtype=np.float64)
    positive = confidence[confidence > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    return np.clip(confidence / max(scale, 1e-8), 0.25, 4.0)


def _huber_weights(residual, delta):
    absolute = np.abs(residual)
    return np.where(absolute <= delta, 1.0, delta / np.maximum(absolute, 1e-12))


def _huber_loss(residual, delta):
    absolute = np.abs(residual)
    return np.where(absolute <= delta, 0.5 * residual * residual, delta * (absolute - 0.5 * delta))


def transform_by_views(points, view_indices, view_values, parameters):
    result = np.empty_like(np.asarray(points, dtype=np.float64))
    lookup = {int(value): index for index, value in enumerate(view_values)}
    for view_value in view_values:
        mask = np.asarray(view_indices) == view_value
        result[mask] = apply_sim3(np.asarray(points)[mask], parameters[lookup[int(view_value)]])
    return result


def fit_planes(observation_points, plane_ids, observation_weights, plane_count):
    normals = np.zeros((plane_count, 3), dtype=np.float64)
    offsets = np.zeros((plane_count,), dtype=np.float64)
    for plane_id in range(plane_count):
        mask = plane_ids == plane_id
        normals[plane_id], offsets[plane_id] = weighted_fit_plane(
            observation_points[mask], observation_weights[mask])
    return normals, offsets


def residuals_for(observation_points, plane_ids, normals, offsets):
    return np.einsum("ij,ij->i", observation_points, normals[plane_ids]) + offsets[plane_ids]


def _clip_parameters(parameters, max_rotation, max_translation, max_log_scale):
    parameters = np.asarray(parameters, dtype=np.float64).copy()
    rotation_norm = float(np.linalg.norm(parameters[:3]))
    if rotation_norm > max_rotation:
        parameters[:3] *= max_rotation / rotation_norm
    translation_norm = float(np.linalg.norm(parameters[3:6]))
    if translation_norm > max_translation:
        parameters[3:6] *= max_translation / translation_norm
    parameters[6] = np.clip(parameters[6], -max_log_scale, max_log_scale)
    return parameters


def optimize_planegraph_ba(
    cache,
    support,
    iterations=10,
    huber_delta=0.03,
    min_plane_views=2,
    min_view_observations=32,
    rotation_anchor=10.0,
    translation_anchor=10.0,
    scale_anchor=20.0,
    damping=1e-5,
    max_rotation_deg=5.0,
    max_translation_fraction=0.05,
    max_scale_fraction=0.05,
    reference_view=None,
    tolerance=1e-5,
    max_points_per_view=20000,
):
    points = cache["points"]
    view_indices = cache["view_indices"]
    obs_indices = support["cache_indices"]
    plane_ids = support["plane_ids"]
    obs_points0 = points[obs_indices]
    obs_views = view_indices[obs_indices]
    obs_weights = _confidence_weights(cache["confidence"][obs_indices])
    plane_count = int(plane_ids.max()) + 1
    view_values = np.unique(view_indices).astype(np.int32)
    view_lookup = {int(value): index for index, value in enumerate(view_values)}
    parameters = np.zeros((len(view_values), 7), dtype=np.float64)

    support_counts = np.asarray([(obs_views == value).sum() for value in view_values])
    if reference_view is None:
        reference_view = int(view_values[int(np.argmax(support_counts))])
    if int(reference_view) not in view_lookup:
        raise ValueError(f"reference view {reference_view} is not present in global cache")

    plane_view_counts = np.asarray([
        len(np.unique(obs_views[plane_ids == plane_id])) for plane_id in range(plane_count)
    ], dtype=np.int32)
    active_planes = plane_view_counts >= int(min_plane_views)
    if not active_planes.any():
        raise ValueError("No plane is supported by enough distinct views for structural feedback")

    center = np.median(points, axis=0)
    scene_scale = float(np.median(np.linalg.norm(points - center, axis=1)))
    scene_scale = max(scene_scale, 1e-6)
    max_rotation = math.radians(float(max_rotation_deg))
    max_translation = float(max_translation_fraction) * scene_scale
    max_log_scale = abs(math.log1p(float(max_scale_fraction)))
    anchor_diagonal = np.asarray([
        rotation_anchor, rotation_anchor, rotation_anchor,
        translation_anchor / (scene_scale * scene_scale),
        translation_anchor / (scene_scale * scene_scale),
        translation_anchor / (scene_scale * scene_scale),
        scale_anchor,
    ], dtype=np.float64)
    finite_difference = np.asarray([1e-5, 1e-5, 1e-5,
                                    scene_scale * 1e-5, scene_scale * 1e-5, scene_scale * 1e-5,
                                    1e-5], dtype=np.float64)

    initial_planes = fit_planes(obs_points0, plane_ids, obs_weights, plane_count)
    initial_residual = residuals_for(obs_points0, plane_ids, *initial_planes)
    history = []
    for iteration in range(int(iterations)):
        transformed_obs = np.empty_like(obs_points0)
        for view_value in np.unique(obs_views):
            mask = obs_views == view_value
            transformed_obs[mask] = apply_sim3(obs_points0[mask], parameters[view_lookup[int(view_value)]])
        normals, offsets = fit_planes(transformed_obs, plane_ids, obs_weights, plane_count)
        largest_step = 0.0

        for view_value in view_values:
            if int(view_value) == int(reference_view):
                continue
            mask = (obs_views == view_value) & active_planes[plane_ids]
            selected = np.nonzero(mask)[0]
            if len(selected) < int(min_view_observations):
                continue
            if len(selected) > int(max_points_per_view):
                chosen = np.linspace(0, len(selected) - 1, int(max_points_per_view), dtype=np.int64)
                selected = selected[chosen]
            local_points = obs_points0[selected]
            local_plane_ids = plane_ids[selected]
            local_weights = obs_weights[selected]
            parameter_index = view_lookup[int(view_value)]
            current = parameters[parameter_index]
            transformed = apply_sim3(local_points, current)
            residual = residuals_for(transformed, local_plane_ids, normals, offsets)
            robust_weights = local_weights * _huber_weights(residual, float(huber_delta))
            jacobian = np.empty((len(selected), 7), dtype=np.float64)
            for column in range(7):
                perturbed = current.copy()
                perturbed[column] += finite_difference[column]
                moved = apply_sim3(local_points, perturbed)
                moved_residual = residuals_for(moved, local_plane_ids, normals, offsets)
                jacobian[:, column] = (moved_residual - residual) / finite_difference[column]
            weighted_jacobian = jacobian * robust_weights[:, None]
            hessian = jacobian.T @ weighted_jacobian + np.diag(anchor_diagonal + float(damping))
            gradient = jacobian.T @ (robust_weights * residual) + anchor_diagonal * current
            try:
                step = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]

            current_cost = float((local_weights * _huber_loss(residual, float(huber_delta))).sum()
                                 + 0.5 * (anchor_diagonal * current * current).sum())
            accepted = current
            accepted_step = np.zeros_like(step)
            for fraction in (1.0, 0.5, 0.25, 0.125):
                candidate = _clip_parameters(
                    current + fraction * step, max_rotation, max_translation, max_log_scale)
                candidate_residual = residuals_for(
                    apply_sim3(local_points, candidate), local_plane_ids, normals, offsets)
                candidate_cost = float(
                    (local_weights * _huber_loss(candidate_residual, float(huber_delta))).sum()
                    + 0.5 * (anchor_diagonal * candidate * candidate).sum()
                )
                if candidate_cost < current_cost:
                    accepted = candidate
                    accepted_step = candidate - current
                    break
            parameters[parameter_index] = accepted
            largest_step = max(largest_step, float(np.linalg.norm(accepted_step)))

        transformed_obs = transform_by_views(obs_points0, obs_views, view_values, parameters)
        normals, offsets = fit_planes(transformed_obs, plane_ids, obs_weights, plane_count)
        residual = residuals_for(transformed_obs, plane_ids, normals, offsets)
        active_residual = residual[active_planes[plane_ids]]
        history.append({
            "iteration": iteration + 1,
            "mean_abs_active_residual": float(np.abs(active_residual).mean()),
            "p95_abs_active_residual": float(np.percentile(np.abs(active_residual), 95)),
            "largest_parameter_step": largest_step,
        })
        if largest_step < float(tolerance):
            break

    corrected_points = transform_by_views(points, view_indices, view_values, parameters)
    corrected_obs = corrected_points[obs_indices]
    normals, offsets = fit_planes(corrected_obs, plane_ids, obs_weights, plane_count)
    final_residual = residuals_for(corrected_obs, plane_ids, normals, offsets)
    return {
        "points": corrected_points.astype(np.float32),
        "plane_normals": normals.astype(np.float32),
        "plane_offsets": offsets.astype(np.float32),
        "parameters": parameters.astype(np.float64),
        "view_values": view_values,
        "reference_view": int(reference_view),
        "active_planes": active_planes,
        "plane_view_counts": plane_view_counts,
        "initial_residual": initial_residual.astype(np.float64),
        "final_residual": final_residual.astype(np.float64),
        "history": history,
        "scene_scale": scene_scale,
    }


def point_assignment(point_count, cache_indices, plane_ids, plane_count):
    assignment = np.full((point_count,), -1, dtype=np.int32)
    counts = np.asarray([(plane_ids == plane_id).sum() for plane_id in range(plane_count)])
    for plane_id in np.argsort(-counts):
        indices = cache_indices[plane_ids == plane_id]
        indices = indices[assignment[indices] < 0]
        assignment[indices] = int(plane_id)
    return assignment, counts.astype(np.int32)


def write_ascii_ply(path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(handle, np.column_stack((points, colors)), fmt="%.7g %.7g %.7g %d %d %d")


def residual_summary(residual):
    absolute = np.abs(np.asarray(residual, dtype=np.float64))
    return {
        "mean": float(absolute.mean()),
        "median": float(np.median(absolute)),
        "p95": float(np.percentile(absolute, 95)),
    }


def save_outputs(output_dir, scene_key, cache, support, result, runtime, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method = "planegraph_ba_v0"
    plane_count = len(result["plane_normals"])
    assignment, counts = point_assignment(
        len(cache["points"]), support["cache_indices"], support["plane_ids"], plane_count)
    display = cache["colors"].copy()
    for plane_id in range(plane_count):
        display[assignment == plane_id] = PLANE_COLORS[plane_id % len(PLANE_COLORS)]
    stem = f"{scene_key}_{method}"
    npz_path = output_dir / f"{stem}_full_pointcloud_editable_planes_data.npz"
    before_ply = output_dir / f"{stem}_before.ply"
    after_ply = output_dir / f"{stem}_after.ply"
    json_path = output_dir / f"{stem}_summary.json"
    np.savez_compressed(
        npz_path,
        schema_version=np.asarray(OUTPUT_SCHEMA_VERSION, dtype=np.int32),
        points=result["points"], original_points=cache["points"].astype(np.float32),
        colors=cache["colors"], original_colors=cache["colors"],
        confidence=cache["confidence"].astype(np.float32),
        point_plane_ids=assignment, plane_ids=np.arange(plane_count, dtype=np.int32),
        source_plane_ids=support["source_plane_ids"],
        plane_normals=result["plane_normals"], plane_offsets=result["plane_offsets"],
        plane_inlier_counts=counts, plane_view_counts=result["plane_view_counts"],
        source_views=cache["view_indices"], pixel_xy=cache["pixel_xy"],
        sim3_view_indices=result["view_values"], sim3_parameters=result["parameters"],
        reference_view=np.asarray(result["reference_view"], dtype=np.int32),
        method=np.asarray(method), runtime_seconds=np.asarray(runtime, dtype=np.float64),
        config_json=np.asarray(json.dumps(config, sort_keys=True)), scene_key=np.asarray(scene_key),
    )
    write_ascii_ply(before_ply, cache["points"], display)
    write_ascii_ply(after_ply, result["points"], display)
    transforms = []
    for view_value, parameters in zip(result["view_values"], result["parameters"]):
        transforms.append({
            "view_index": int(view_value),
            "rotation_vector_rad": parameters[:3].tolist(),
            "translation": parameters[3:6].tolist(),
            "scale": float(math.exp(float(parameters[6]))),
            "fixed_reference": int(view_value) == int(result["reference_view"]),
        })
    summary = {
        "method": method,
        "scene_key": scene_key,
        "planes": plane_count,
        "active_multiview_planes": int(result["active_planes"].sum()),
        "reference_view": int(result["reference_view"]),
        "support_mapping": {key: value for key, value in support.items() if isinstance(value, int)},
        "residual_before": residual_summary(result["initial_residual"]),
        "residual_after": residual_summary(result["final_residual"]),
        "iterations": result["history"],
        "view_corrections": transforms,
        "runtime_seconds": float(runtime),
        "outputs": {"npz": str(npz_path), "before_ply": str(before_ply), "after_ply": str(after_ply)},
    }
    summary["outputs"]["json"] = str(json_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(
        "PlaneGraph-BA v0: refine frozen DUSt3R global alignment with mapped plane supports")
    parser.add_argument("--global_cloud_npz", required=True)
    parser.add_argument("--support_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_key", default="scene")
    parser.add_argument("--min_conf", type=float, default=1.0)
    parser.add_argument("--min_plane_points", type=int, default=64)
    parser.add_argument("--min_plane_views", type=int, default=2)
    parser.add_argument("--min_view_observations", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--huber_delta", type=float, default=0.03)
    parser.add_argument("--rotation_anchor", type=float, default=10.0)
    parser.add_argument("--translation_anchor", type=float, default=10.0)
    parser.add_argument("--scale_anchor", type=float, default=20.0)
    parser.add_argument("--damping", type=float, default=1e-5)
    parser.add_argument("--max_rotation_deg", type=float, default=5.0)
    parser.add_argument("--max_translation_fraction", type=float, default=0.05)
    parser.add_argument("--max_scale_fraction", type=float, default=0.05)
    parser.add_argument("--reference_view", type=int)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--max_points_per_view", type=int, default=20000)
    args = parser.parse_args()
    config = vars(args).copy()
    cache, support = load_inputs(
        args.global_cloud_npz, args.support_npz, args.min_conf, args.min_plane_points)
    started = time.perf_counter()
    result = optimize_planegraph_ba(
        cache, support, iterations=args.iterations, huber_delta=args.huber_delta,
        min_plane_views=args.min_plane_views,
        min_view_observations=args.min_view_observations,
        rotation_anchor=args.rotation_anchor, translation_anchor=args.translation_anchor,
        scale_anchor=args.scale_anchor, damping=args.damping,
        max_rotation_deg=args.max_rotation_deg,
        max_translation_fraction=args.max_translation_fraction,
        max_scale_fraction=args.max_scale_fraction, reference_view=args.reference_view,
        tolerance=args.tolerance, max_points_per_view=args.max_points_per_view)
    summary = save_outputs(
        args.output_dir, args.scene_key, cache, support, result,
        time.perf_counter() - started, config)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
