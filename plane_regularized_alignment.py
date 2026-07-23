"""Differentiable plane feedback for DUSt3R global alignment.

This module keeps DUSt3R frozen, but continues optimizing the global aligner's
depth, pose, focal, and pairwise parameters with a robust multi-view plane
incidence term.  Plane support/identity is supplied by the learned Stage1/Stage2
pipeline; single-view planes are excluded from structural feedback.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F


def _fit_plane(points):
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0:
        normal = -normal
    return normal, -float(normal @ center)


def _residual_summary(values):
    values = np.abs(np.asarray(values, dtype=np.float64))
    if len(values) == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _prepare_observations(scene, view_indices, pixel_xy, plane_ids, min_plane_views, min_plane_points):
    views = np.asarray(view_indices, dtype=np.int64).reshape(-1)
    pixels = np.asarray(pixel_xy, dtype=np.int64)
    labels = np.asarray(plane_ids, dtype=np.int64).reshape(-1)
    if pixels.shape != (len(views), 2) or len(labels) != len(views):
        raise ValueError("view_indices, pixel_xy, and plane_ids must have matching lengths")

    valid = labels >= 0
    for index, (height, width) in enumerate(scene.imshapes):
        mask = views == index
        valid[mask] &= (
            (pixels[mask, 0] >= 0)
            & (pixels[mask, 0] < int(width))
            & (pixels[mask, 1] >= 0)
            & (pixels[mask, 1] < int(height))
        )
    valid &= (views >= 0) & (views < int(scene.n_imgs))
    views, pixels, labels = views[valid], pixels[valid], labels[valid]
    if len(labels) == 0:
        raise ValueError("No valid plane observations remain after coordinate validation")

    # Repeated pair records can map the same plane observation more than once.
    packed = np.column_stack((labels, views, pixels))
    _, unique_indices = np.unique(packed, axis=0, return_index=True)
    unique_indices.sort()
    views, pixels, labels = views[unique_indices], pixels[unique_indices], labels[unique_indices]

    active_source_ids = []
    plane_view_counts = {}
    plane_point_counts = {}
    for source_id in np.unique(labels):
        mask = labels == source_id
        view_count = int(len(np.unique(views[mask])))
        point_count = int(mask.sum())
        plane_view_counts[int(source_id)] = view_count
        plane_point_counts[int(source_id)] = point_count
        if view_count >= int(min_plane_views) and point_count >= int(min_plane_points):
            active_source_ids.append(int(source_id))
    if not active_source_ids:
        raise ValueError("No plane has enough points and distinct views for structural feedback")

    keep = np.isin(labels, np.asarray(active_source_ids, dtype=np.int64))
    views, pixels, labels = views[keep], pixels[keep], labels[keep]
    id_map = {source_id: index for index, source_id in enumerate(active_source_ids)}
    labels = np.asarray([id_map[int(value)] for value in labels], dtype=np.int64)
    linear_indices = np.asarray(
        [int(y) * int(scene.imshapes[int(view)][1]) + int(x) for view, (x, y) in zip(views, pixels)],
        dtype=np.int64,
    )
    return {
        "view_indices": views,
        "pixel_xy": pixels,
        "linear_indices": linear_indices,
        "plane_ids": labels,
        "source_plane_ids": np.asarray(active_source_ids, dtype=np.int64),
        "plane_view_counts": plane_view_counts,
        "plane_point_counts": plane_point_counts,
    }


def _gather_points(scene, view_indices, linear_indices):
    raw_points = scene.get_pts3d(raw=True)
    view_tensor = torch.as_tensor(view_indices, dtype=torch.long, device=raw_points.device)
    linear_tensor = torch.as_tensor(linear_indices, dtype=torch.long, device=raw_points.device)
    return raw_points[view_tensor, linear_tensor]


def _gather_confidence(scene, view_indices, pixel_xy):
    confidence_maps = scene.get_conf()
    values = []
    for view, (x, y) in zip(view_indices, pixel_xy):
        values.append(confidence_maps[int(view)][int(y), int(x)])
    confidence = torch.stack(values).detach().float()
    median = confidence.median().clamp_min(1e-6)
    return (confidence / median).clamp(0.1, 10.0)


def _snapshot_pointmaps(scene):
    """Copy the current globally aligned pointmaps before any rollback."""

    snapshots = []
    for pointmap in scene.get_pts3d():
        snapshots.append(
            pointmap.detach().cpu().numpy().astype(np.float32, copy=True)
        )
    return snapshots


def _balanced_weights(confidence, plane_ids, plane_count):
    weights = confidence.clone()
    for plane_id in range(plane_count):
        mask = plane_ids == plane_id
        weights[mask] /= weights[mask].sum().clamp_min(1e-6)
    return weights * (len(weights) / max(plane_count, 1))


def _robust_incidence(residual, delta):
    absolute = residual.abs()
    delta = float(delta)
    return torch.where(
        absolute <= delta,
        0.5 * absolute.square() / delta,
        absolute - 0.5 * delta,
    )


def _world_planes(normal_raw, offsets, center, scale):
    normals = F.normalize(normal_raw, dim=-1)
    world_offsets = offsets * scale - (normals * center[None]).sum(dim=-1)
    return normals, world_offsets


def optimize_scene_with_plane_feedback(
    scene,
    view_indices,
    pixel_xy,
    plane_ids,
    *,
    niter=100,
    lr=0.002,
    plane_weight=0.2,
    huber_delta=0.01,
    min_plane_views=2,
    min_plane_points=64,
    max_base_loss_increase=0.03,
    min_relative_plane_improvement=1e-4,
    log_every=20,
):
    """Continue DUSt3R alignment with multi-view plane incidence feedback.

    The proposed update is rolled back unless plane residual improves and the
    original DUSt3R objective stays within ``max_base_loss_increase``.
    """

    started = time.perf_counter()
    observations = _prepare_observations(
        scene,
        view_indices,
        pixel_xy,
        plane_ids,
        min_plane_views,
        min_plane_points,
    )
    device = scene.device
    obs_views = observations["view_indices"]
    obs_linear = observations["linear_indices"]
    labels = torch.as_tensor(observations["plane_ids"], dtype=torch.long, device=device)
    plane_count = len(observations["source_plane_ids"])

    with torch.no_grad():
        initial_points = _gather_points(scene, obs_views, obs_linear).detach().clone()
        finite = torch.isfinite(initial_points).all(dim=1)
    if not bool(finite.all()):
        keep = finite.detach().cpu().numpy()
        for key in ("view_indices", "pixel_xy", "linear_indices", "plane_ids"):
            observations[key] = observations[key][keep]
        obs_views = observations["view_indices"]
        obs_linear = observations["linear_indices"]
        labels = torch.as_tensor(observations["plane_ids"], dtype=torch.long, device=device)
        initial_points = initial_points[finite]
    if len(initial_points) < 3 * plane_count:
        raise ValueError("Too few finite support points for plane feedback")

    center = initial_points.median(dim=0).values
    distances = torch.linalg.vector_norm(initial_points - center, dim=-1)
    scale = distances.median().clamp_min(1e-6)
    normalized_initial = ((initial_points - center) / scale).detach().cpu().numpy()

    initial_normals = []
    initial_offsets = []
    labels_np = labels.detach().cpu().numpy()
    for plane_id in range(plane_count):
        normal, offset = _fit_plane(normalized_initial[labels_np == plane_id])
        initial_normals.append(normal)
        initial_offsets.append(offset)
    normal_raw = torch.nn.Parameter(
        torch.as_tensor(np.asarray(initial_normals), dtype=torch.float32, device=device)
    )
    offsets = torch.nn.Parameter(
        torch.as_tensor(np.asarray(initial_offsets), dtype=torch.float32, device=device)
    )
    with torch.no_grad():
        initial_normals_world, initial_offsets_world = _world_planes(
            normal_raw, offsets, center, scale)
        initial_normals_world = initial_normals_world.detach().clone()
        initial_offsets_world = initial_offsets_world.detach().clone()

    confidence = _gather_confidence(scene, obs_views, observations["pixel_xy"])
    weights = _balanced_weights(confidence, labels, plane_count)
    scene_parameters = [parameter for parameter in scene.parameters() if parameter.requires_grad]
    snapshots = [parameter.detach().clone() for parameter in scene_parameters]
    optimizer = torch.optim.Adam(scene_parameters + [normal_raw, offsets], lr=float(lr), betas=(0.9, 0.9))

    def measure():
        points = _gather_points(scene, obs_views, obs_linear)
        normalized = (points - center) / scale
        normals = F.normalize(normal_raw, dim=-1)
        residual = (normalized * normals[labels]).sum(dim=-1) + offsets[labels]
        plane_loss = (_robust_incidence(residual, huber_delta) * weights).sum() / weights.sum().clamp_min(1e-6)
        return points, residual, plane_loss

    with torch.no_grad():
        base_before = float(scene().detach())
        _, initial_residual_tensor, initial_plane_loss_tensor = measure()
        initial_residual = initial_residual_tensor.detach().cpu().numpy() * float(scale)
        initial_plane_loss = float(initial_plane_loss_tensor)

    history = []
    for iteration in range(1, int(niter) + 1):
        progress = (iteration - 1) / max(int(niter), 1)
        current_lr = float(lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = max(current_lr, float(lr) * 1e-3)
        optimizer.zero_grad(set_to_none=True)
        base_loss = scene()
        _, _, plane_loss = measure()
        total_loss = base_loss + float(plane_weight) * plane_loss
        total_loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % max(int(log_every), 1) == 0 or iteration == int(niter):
            row = {
                "iteration": iteration,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "base_loss": float(base_loss.detach()),
                "plane_loss": float(plane_loss.detach()),
                "total_loss": float(total_loss.detach()),
            }
            history.append(row)
            print(
                "[PlaneFeedback] "
                f"{iteration}/{int(niter)} base={row['base_loss']:.6g} "
                f"plane={row['plane_loss']:.6g} total={row['total_loss']:.6g}",
                flush=True,
            )

    with torch.no_grad():
        final_points, final_residual_tensor, final_plane_loss_tensor = measure()
        base_after = float(scene().detach())
        final_residual = final_residual_tensor.detach().cpu().numpy() * float(scale)
        final_plane_loss = float(final_plane_loss_tensor)
        support_displacement = torch.linalg.vector_norm(final_points - initial_points, dim=-1)
        displacement_summary = _residual_summary(support_displacement.detach().cpu().numpy())
        normals_world, offsets_world = _world_planes(normal_raw, offsets, center, scale)
        proposed_pointmaps = _snapshot_pointmaps(scene)

    residual_before = _residual_summary(initial_residual)
    residual_after = _residual_summary(final_residual)
    relative_improvement = (
        (residual_before["mean"] - residual_after["mean"])
        / max(residual_before["mean"], 1e-12)
    )
    base_limit = base_before * (1.0 + float(max_base_loss_increase)) + 1e-8
    accepted = bool(
        np.isfinite(base_after)
        and np.isfinite(residual_after["mean"])
        and base_after <= base_limit
        and relative_improvement >= float(min_relative_plane_improvement)
    )
    proposed = {
        "base_loss": base_after,
        "plane_loss": final_plane_loss,
        "residual": residual_after,
        "support_displacement": displacement_summary,
        "relative_plane_improvement": float(relative_improvement),
    }
    if not accepted:
        with torch.no_grad():
            for parameter, snapshot in zip(scene_parameters, snapshots):
                parameter.copy_(snapshot)
        base_after = base_before
        residual_after = residual_before
        displacement_summary = _residual_summary(np.zeros((len(initial_points),), dtype=np.float32))
        normals_world = initial_normals_world
        offsets_world = initial_offsets_world

    return {
        "accepted": accepted,
        "observations": int(len(initial_points)),
        "active_planes": int(plane_count),
        "source_plane_ids": observations["source_plane_ids"],
        "base_loss_before": base_before,
        "base_loss_after": base_after,
        "base_loss_limit": float(base_limit),
        "plane_loss_before": initial_plane_loss,
        "plane_loss_after": final_plane_loss if accepted else initial_plane_loss,
        "residual_before": residual_before,
        "residual_after": residual_after,
        "support_displacement": displacement_summary,
        "proposed": proposed,
        "proposed_pointmaps": proposed_pointmaps,
        "plane_normals": normals_world.detach().cpu().numpy().astype(np.float32),
        "plane_offsets": offsets_world.detach().cpu().numpy().astype(np.float32),
        "history": history,
        "runtime_seconds": float(time.perf_counter() - started),
        "config": {
            "niter": int(niter),
            "lr": float(lr),
            "plane_weight": float(plane_weight),
            "huber_delta_normalized": float(huber_delta),
            "min_plane_views": int(min_plane_views),
            "min_plane_points": int(min_plane_points),
            "max_base_loss_increase": float(max_base_loss_increase),
        },
    }
