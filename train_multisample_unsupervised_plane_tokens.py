from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PLANE_COLORS = np.asarray(
    [
        [229, 57, 53],
        [30, 136, 229],
        [67, 160, 71],
        [251, 140, 0],
        [142, 36, 170],
        [0, 137, 123],
        [109, 76, 65],
        [57, 73, 171],
        [198, 40, 40],
        [0, 121, 107],
    ],
    dtype=np.uint8,
)


def build_smooth_pairs(
    points_norm,
    colors_float,
    pair_count,
    candidate_count,
    xyz_sigma,
    rgb_sigma,
    seed,
):
    if pair_count <= 0 or len(points_norm) <= 1:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    rng = np.random.default_rng(seed)
    n_points = len(points_norm)
    anchors = rng.integers(0, n_points, size=pair_count, endpoint=False)
    candidate_count = max(2, min(int(candidate_count), n_points))
    candidates = rng.integers(0, n_points, size=(pair_count, candidate_count), endpoint=False)
    same = candidates == anchors[:, None]
    if same.any():
        candidates[same] = (candidates[same] + 1) % n_points

    anchor_xyz = points_norm[anchors, None, :]
    candidate_xyz = points_norm[candidates]
    xyz_dist2 = np.sum((candidate_xyz - anchor_xyz) ** 2, axis=-1)
    nearest = np.argmin(xyz_dist2, axis=1)
    paired = candidates[np.arange(pair_count), nearest]
    xyz_dist2 = xyz_dist2[np.arange(pair_count), nearest]

    rgb_dist2 = np.sum((colors_float[paired] - colors_float[anchors]) ** 2, axis=-1)
    xyz_sigma2 = max(float(xyz_sigma) ** 2, 1e-8)
    rgb_sigma2 = max(float(rgb_sigma) ** 2, 1e-8)
    weights = np.exp(-xyz_dist2 / xyz_sigma2 - rgb_dist2 / rgb_sigma2).astype(np.float32)
    keep = weights > 1e-4
    return anchors[keep].astype(np.int64), paired[keep].astype(np.int64), weights[keep]


def sample_case(
    path,
    max_points,
    seed,
    smooth_pairs_per_sample=0,
    smooth_candidates=24,
    smooth_xyz_sigma=0.06,
    smooth_rgb_sigma=0.25,
):
    raw = np.load(path)
    points = raw["points"].astype(np.float32)
    colors = raw["colors"].astype(np.uint8)
    rng = np.random.default_rng(seed)
    if max_points > 0 and len(points) > max_points:
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    center = points.mean(axis=0, keepdims=True).astype(np.float32)
    scale = max(float(np.linalg.norm(points - center, axis=1).max()), 1e-6)
    points_norm = ((points - center) / scale).astype(np.float32)
    colors_float = colors.astype(np.float32) / 255.0
    radius = np.linalg.norm(points_norm, axis=1, keepdims=True).astype(np.float32)
    features = np.concatenate([points_norm, colors_float, radius], axis=1).astype(np.float32)
    smooth_i, smooth_j, smooth_w = build_smooth_pairs(
        points_norm,
        colors_float,
        smooth_pairs_per_sample,
        smooth_candidates,
        smooth_xyz_sigma,
        smooth_rgb_sigma,
        seed + 13,
    )
    return {
        "path": str(path),
        "stem": Path(path).name.replace("_full_pointcloud_editable_planes_data.npz", ""),
        "points": points,
        "colors": colors,
        "points_norm": points_norm,
        "features": features,
        "smooth_i": smooth_i,
        "smooth_j": smooth_j,
        "smooth_w": smooth_w,
        "center": center[0],
        "scale": scale,
    }


def entropy(assign, eps=1e-8):
    return -(assign * (assign + eps).log()).sum(dim=-1)


def diversity_loss(normals, offsets, margin_normal=0.18, margin_offset=0.04):
    count = normals.shape[0]
    if count <= 1:
        return normals.sum() * 0.0
    similarity = torch.abs(normals @ normals.t())
    offset_distance = torch.abs(offsets[:, None] - offsets[None, :])
    diagonal = torch.eye(count, device=normals.device, dtype=torch.bool)
    similarity = similarity[~diagonal]
    offset_distance = offset_distance[~diagonal]
    return F.relu(similarity - (1.0 - margin_normal)).mean() + F.relu(
        margin_offset - offset_distance
    ).mean()


def coverage_loss(assign, min_coverage):
    coverage = assign.mean(dim=0)
    return F.relu(min_coverage - coverage).mean(), coverage


def dead_token_loss(assign, min_coverage):
    coverage = assign.mean(dim=0)
    return F.relu(min_coverage - coverage).pow(2).mean(), coverage


def trimmed_mean(values, keep_ratio):
    if keep_ratio >= 1.0:
        return values.mean()
    keep = max(1, int(round(values.numel() * keep_ratio)))
    return torch.topk(values.reshape(-1), k=keep, largest=False).values.mean()


def confidence_separation_loss(assign, margin):
    if assign.shape[1] <= 1:
        return assign.sum() * 0.0
    top_two = torch.topk(assign, k=2, dim=-1).values
    return F.relu(margin - (top_two[:, 0] - top_two[:, 1])).mean()


def local_smoothness_loss(assign, smooth_i, smooth_j, smooth_w):
    if smooth_i.numel() == 0:
        return assign.sum() * 0.0
    difference = (assign[smooth_i] - assign[smooth_j]).pow(2).sum(dim=-1)
    return (difference * smooth_w).sum() / smooth_w.sum().clamp_min(1e-6)
