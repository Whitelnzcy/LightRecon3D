import argparse
import functools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_amortized_plane_tokens import (
    PLANE_COLORS,
    entropy,
    sample_case_with_aux_planes,
)


def fit_patch_geometry(points):
    centroid = points.mean(axis=0).astype(np.float32)
    centered = points - centroid[None, :]
    if len(points) < 3:
        return centroid, np.asarray([0.0, 0.0, 1.0], dtype=np.float32), 0.0, np.zeros(2, dtype=np.float32)
    _, s, vh = np.linalg.svd(centered.astype(np.float32), full_matrices=False)
    normal = vh[-1].astype(np.float32)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-6)
    denom = max(float(s.sum()), 1e-6)
    planarity = float((s[1] - s[2]) / denom) if len(s) >= 3 else 0.0
    extent = np.asarray([float(s[0] / max(len(points), 1)), float(s[1] / max(len(points), 1))], dtype=np.float32)
    return centroid, normal, planarity, extent


def majority_label(labels):
    valid = labels[labels >= 0]
    if len(valid) == 0:
        return -1, 0.0
    counts = np.bincount(valid)
    label = int(np.argmax(counts))
    return label, float(counts[label] / max(len(labels), 1))


def plane_basis_np(normal):
    n = normal.astype(np.float32)
    n = n / max(float(np.linalg.norm(n)), 1e-6)
    ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(n, ref).astype(np.float32)
    u = u / max(float(np.linalg.norm(u)), 1e-6)
    v = np.cross(n, u).astype(np.float32)
    v = v / max(float(np.linalg.norm(v)), 1e-6)
    return u, v


def build_surface_patches(case, args):
    points = case["points_norm"].astype(np.float32)
    features = case["features"].astype(np.float32)
    labels = case.get("aux_point_labels", np.full(len(points), -1, dtype=np.int64)).astype(np.int64)
    point_line_prob = case.get("point_line_prob", np.zeros(len(points), dtype=np.float32)).astype(np.float32)
    if features.shape[1] >= 9:
        pixel_xy = features[:, -2:].astype(np.float32)
        coords = np.floor((pixel_xy + 1.0) / max(args.patch_pixel_size, 1e-6)).astype(np.int64)
    else:
        coords = np.floor(points / max(args.patch_voxel_size, 1e-6)).astype(np.int64)
    buckets = {}
    for idx, c in enumerate(coords):
        key = tuple(int(x) for x in c)
        buckets.setdefault(key, []).append(idx)

    patch_indices = []
    patch_grid_keys = []
    for key, ids in buckets.items():
        if len(ids) >= args.min_patch_points:
            patch_indices.append(np.asarray(ids, dtype=np.int64))
            patch_grid_keys.append(key)
    if not patch_indices:
        patch_indices = [np.arange(len(points), dtype=np.int64)]
        patch_grid_keys = [(0, 0)]

    patch_features = []
    patch_centroids = []
    patch_normals = []
    patch_labels = []
    patch_label_conf = []
    patch_counts = []
    point_to_patch = np.full(len(points), -1, dtype=np.int64)
    for patch_id, ids in enumerate(patch_indices):
        point_to_patch[ids] = patch_id
        centroid, normal, planarity, extent = fit_patch_geometry(points[ids])
        label, conf = majority_label(labels[ids])
        mean_feat = features[ids].mean(axis=0)
        line_mean = float(point_line_prob[ids].mean()) if len(ids) else 0.0
        line_max = float(point_line_prob[ids].max()) if len(ids) else 0.0
        log_count = np.asarray([np.log1p(len(ids)) / 10.0], dtype=np.float32)
        geom_feat = np.concatenate(
            [
                centroid,
                normal,
                np.asarray([planarity], dtype=np.float32),
                extent,
                log_count,
                np.asarray([0.0 if args.hide_teacher_feature_conf else conf], dtype=np.float32),
                np.asarray([line_mean, line_max], dtype=np.float32),
            ]
        )
        patch_features.append(np.concatenate([mean_feat, geom_feat]).astype(np.float32))
        patch_centroids.append(centroid)
        patch_normals.append(normal)
        patch_labels.append(label)
        patch_label_conf.append(conf)
        patch_counts.append(len(ids))

    patch_features = np.stack(patch_features).astype(np.float32)
    patch_centroids = np.stack(patch_centroids).astype(np.float32)
    patch_normals = np.stack(patch_normals).astype(np.float32)
    patch_labels = np.asarray(patch_labels, dtype=np.int64)
    patch_label_conf = np.asarray(patch_label_conf, dtype=np.float32)
    patch_counts = np.asarray(patch_counts, dtype=np.float32)

    key_to_patch = {key: i for i, key in enumerate(patch_grid_keys)}
    edge_i = []
    edge_j = []
    edge_line_prob = []
    edge_boundary_conf = []
    boundary_neighbor = np.zeros((len(patch_features), args.num_planes), dtype=np.float32)

    def append_patch_edge(a, b):
        edge_i.append(a)
        edge_j.append(b)
        line_prob = max(float(patch_features[a, -1]), float(patch_features[b, -1]))
        normal_gap = 1.0 - abs(float(np.dot(patch_normals[a], patch_normals[b])))
        geom_conf = min(1.0, max(0.0, normal_gap / max(float(args.edge_boundary_normal_gap), 1e-6)))
        edge_line_prob.append(line_prob)
        edge_boundary_conf.append(line_prob * geom_conf)

    for key, patch_id in key_to_patch.items():
        if len(key) == 2:
            offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
            for dx, dy in offsets:
                other = (key[0] + dx, key[1] + dy)
                if other in key_to_patch and patch_id < key_to_patch[other]:
                    other_id = key_to_patch[other]
                    append_patch_edge(patch_id, other_id)
        else:
            for axis in range(3):
                for delta in (-1, 1):
                    other = list(key)
                    other[axis] += delta
                    other = tuple(other)
                    if other in key_to_patch and patch_id < key_to_patch[other]:
                        other_id = key_to_patch[other]
                        append_patch_edge(patch_id, other_id)

    for a, b in zip(edge_i, edge_j):
        la = int(patch_labels[a])
        lb = int(patch_labels[b])
        if 0 <= la < args.num_planes and 0 <= lb < args.num_planes and la != lb:
            boundary_neighbor[a, lb] = 1.0
            boundary_neighbor[b, la] = 1.0

    support_prior = np.full((len(patch_features), args.num_planes), -float(args.outside_support_penalty), dtype=np.float32)
    support_inside = np.zeros((len(patch_features), args.num_planes), dtype=np.float32)
    support_distance_cells = np.full((len(patch_features), args.num_planes), float(args.support_max_distance_cells), dtype=np.float32)
    teacher_label_prior = np.zeros((len(patch_features), args.num_planes), dtype=np.float32)
    for patch_id, label in enumerate(patch_labels):
        if 0 <= int(label) < args.num_planes:
            teacher_label_prior[patch_id, int(label)] = float(args.teacher_label_logit_weight) * float(patch_label_conf[patch_id])
    aux_normals = case.get("aux_plane_normals", np.zeros((0, 3), dtype=np.float32)).astype(np.float32)
    max_planes = min(args.num_planes, len(aux_normals))
    for plane_id in range(max_planes):
        own = (patch_labels == plane_id) & (patch_label_conf >= args.support_min_label_conf)
        if not np.any(own):
            own = patch_labels == plane_id
        if not np.any(own):
            continue
        u, v = plane_basis_np(aux_normals[plane_id])
        uv = np.stack([patch_centroids @ u, patch_centroids @ v], axis=1)
        grid = np.floor(uv / max(args.support_grid_size, 1e-6)).astype(np.int64)
        occupied = grid[own]
        occupied_set = {tuple(c) for c in occupied}
        for patch_id, cell in enumerate(grid):
            best = args.support_max_distance_cells
            cx, cy = int(cell[0]), int(cell[1])
            for ox, oy in occupied_set:
                dist = max(abs(cx - ox), abs(cy - oy))
                if dist < best:
                    best = dist
                    if best <= args.support_dilate_cells:
                        break
            support_distance_cells[patch_id, plane_id] = float(best)
            if best <= args.support_dilate_cells:
                support_inside[patch_id, plane_id] = 1.0
                support_prior[patch_id, plane_id] = float(args.support_logit_weight)
            elif boundary_neighbor[patch_id, plane_id] > 0:
                support_prior[patch_id, plane_id] = float(args.boundary_support_logit_weight)

    case["patch_features"] = patch_features
    case["patch_centroids"] = patch_centroids
    case["patch_normals"] = patch_normals
    case["patch_labels"] = patch_labels
    case["patch_label_conf"] = patch_label_conf
    case["patch_counts"] = patch_counts
    case["point_to_patch"] = point_to_patch
    case["patch_edge_i"] = np.asarray(edge_i, dtype=np.int64)
    case["patch_edge_j"] = np.asarray(edge_j, dtype=np.int64)
    case["patch_edge_line_prob"] = np.asarray(edge_line_prob, dtype=np.float32)
    case["patch_edge_boundary_conf"] = np.asarray(edge_boundary_conf, dtype=np.float32)
    case["patch_support_prior"] = support_prior.astype(np.float32)
    case["patch_teacher_label_prior"] = teacher_label_prior.astype(np.float32)
    case["patch_support_inside"] = support_inside.astype(np.float32)
    case["patch_support_distance_cells"] = support_distance_cells.astype(np.float32)
    case["patch_boundary_neighbor"] = boundary_neighbor.astype(np.float32)
    return case


class PatchPlaneTokenHead(nn.Module):
    def __init__(self, num_planes, patch_feature_dim, hidden_dim=192, context_dim=256):
        super().__init__()
        self.num_planes = num_planes
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
            nn.GELU(),
        )
        self.token_head = nn.Sequential(
            nn.Linear(context_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_planes * 5 + 1),
        )
        pair_dim = patch_feature_dim + 3 + 1 + 1 + 1
        self.assignment_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def predict_tokens(self, patch_features):
        encoded = self.patch_encoder(patch_features)
        pooled = torch.cat([encoded.mean(dim=0), encoded.max(dim=0).values], dim=-1)
        raw = self.token_head(pooled)
        background_logit = raw[-1]
        raw = raw[:-1].view(self.num_planes, 5)
        normals = F.normalize(raw[:, :3], dim=-1)
        offsets = raw[:, 3]
        token_logits = raw[:, 4]
        return normals, offsets, token_logits, background_logit

    def forward_one(
        self,
        patch_centroids,
        patch_normals,
        patch_features,
        temperature,
        distance_logit_weight,
        normal_logit_weight,
        teacher_normals=None,
        teacher_offsets=None,
        teacher_param_blend=0.0,
        support_prior=None,
        teacher_label_prior=None,
    ):
        normals, offsets, token_logits, background_logit = self.predict_tokens(patch_features)
        if teacher_param_blend > 0 and teacher_normals is not None and teacher_normals.numel() > 0:
            m = min(normals.shape[0], teacher_normals.shape[0])
            blend = float(teacher_param_blend)
            normals = normals.clone()
            offsets = offsets.clone()
            normals[:m] = F.normalize((1.0 - blend) * normals[:m] + blend * teacher_normals[:m], dim=-1)
            offsets[:m] = (1.0 - blend) * offsets[:m] + blend * teacher_offsets[:m]
        dists = torch.abs(patch_centroids @ normals.t() + offsets.view(1, -1))
        normal_align = torch.abs(patch_normals @ normals.t()).clamp(0.0, 1.0)
        n_patches = patch_centroids.shape[0]
        n_planes = normals.shape[0]
        patch_feat = patch_features[:, None, :].expand(n_patches, n_planes, -1)
        normal_feat = normals[None, :, :].expand(n_patches, n_planes, -1)
        offset_feat = offsets.view(1, n_planes, 1).expand(n_patches, n_planes, 1)
        pair_feat = torch.cat([patch_feat, normal_feat, offset_feat, dists.unsqueeze(-1), normal_align.unsqueeze(-1)], dim=-1)
        logits = self.assignment_head(pair_feat).squeeze(-1)
        logits = logits - distance_logit_weight * dists / temperature
        logits = logits + normal_logit_weight * normal_align / max(float(temperature), 1e-6)
        logits = logits + token_logits.view(1, -1)
        if support_prior is not None:
            logits = logits + support_prior
        if teacher_label_prior is not None:
            logits = logits + teacher_label_prior
        logits = torch.cat([logits, background_logit.expand(n_patches, 1)], dim=1)
        assign = F.softmax(logits, dim=-1)
        return normals, offsets, dists, normal_align, assign


def class_balanced_patch_ce(assign, labels, num_planes, label_conf):
    valid = (labels >= 0) & (labels < num_planes)
    if not torch.any(valid):
        return assign.sum() * 0.0
    labels_v = labels[valid]
    log_probs = torch.log(assign[valid].clamp_min(1e-8))
    counts = torch.bincount(labels_v, minlength=num_planes).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = 1.0 / counts[present].clamp_min(1.0)
    weights = weights * (present.float().sum() / weights[present].sum().clamp_min(1e-6))
    point_weights = weights[labels_v] * label_conf[valid].clamp_min(0.2)
    per_patch = F.nll_loss(log_probs, labels_v, reduction="none")
    return (per_patch * point_weights).sum() / point_weights.sum().clamp_min(1e-6)


def permutation_invariant_labels(
    assign,
    labels,
    label_conf,
    pred_normals,
    pred_offsets,
    teacher_normals,
    teacher_offsets,
    classification_weight,
    normal_weight,
    offset_weight,
):
    active_count = min(int(teacher_normals.shape[0]), pred_normals.shape[0])
    if active_count <= 0:
        return labels
    query_count = pred_normals.shape[0]
    with torch.no_grad():
        costs = torch.empty(
            (active_count, query_count),
            device=assign.device,
            dtype=assign.dtype,
        )
        for teacher_id in range(active_count):
            own = labels == teacher_id
            weights = label_conf[own].clamp_min(0.2)
            if torch.any(own):
                class_cost = -(
                    torch.log(assign[own, :query_count].clamp_min(1e-8)) * weights[:, None]
                ).sum(dim=0) / weights.sum().clamp_min(1e-6)
            else:
                class_cost = torch.full(
                    (query_count,),
                    4.0,
                    device=assign.device,
                    dtype=assign.dtype,
                )
            normal_cost = 1.0 - torch.abs(pred_normals @ teacher_normals[teacher_id])
            offset_cost = torch.abs(pred_offsets - teacher_offsets[teacher_id])
            costs[teacher_id] = (
                float(classification_weight) * class_cost
                + float(normal_weight) * normal_cost
                + float(offset_weight) * offset_cost
            )

        costs_cpu = costs.cpu().numpy()

        @functools.lru_cache(maxsize=None)
        def solve(teacher_id, used_mask):
            if teacher_id == active_count:
                return 0.0, ()
            best_cost = float("inf")
            best_queries = ()
            for query_id in range(query_count):
                if used_mask & (1 << query_id):
                    continue
                tail_cost, tail_queries = solve(teacher_id + 1, used_mask | (1 << query_id))
                total = float(costs_cpu[teacher_id, query_id]) + tail_cost
                if total < best_cost:
                    best_cost = total
                    best_queries = (query_id,) + tail_queries
            return best_cost, best_queries

        _, matched_queries = solve(0, 0)
        remapped = labels.clone()
        for teacher_id, query_id in enumerate(matched_queries):
            remapped[labels == teacher_id] = int(query_id)
        return remapped


def background_patch_loss(assign, labels, num_planes):
    bg = labels < 0
    if not torch.any(bg):
        return assign.sum() * 0.0
    targets = torch.full((int(bg.sum().item()),), num_planes, dtype=torch.long, device=assign.device)
    return F.nll_loss(torch.log(assign[bg].clamp_min(1e-8)), targets)


def patch_smoothness_loss(
    assign,
    edge_i,
    edge_j,
    labels,
    edge_line_prob=None,
    edge_boundary_conf=None,
    margin=0.20,
    line_smooth_suppress=0.0,
    line_boundary_boost=0.0,
):
    plane_assign = assign[:, :-1]
    if edge_i.numel() == 0:
        return assign.sum() * 0.0, assign.sum() * 0.0
    if edge_line_prob is None:
        edge_line_prob = torch.zeros_like(edge_i, dtype=plane_assign.dtype)
    edge_line_prob = edge_line_prob.to(dtype=plane_assign.dtype, device=plane_assign.device).clamp(0.0, 1.0)
    if edge_boundary_conf is None:
        edge_boundary_conf = edge_line_prob
    edge_boundary_conf = edge_boundary_conf.to(dtype=plane_assign.dtype, device=plane_assign.device).clamp(0.0, 1.0)
    same_teacher = (labels[edge_i] >= 0) & (labels[edge_i] == labels[edge_j])
    diff_teacher = (labels[edge_i] >= 0) & (labels[edge_j] >= 0) & (labels[edge_i] != labels[edge_j])
    same_loss = assign.sum() * 0.0
    boundary_loss = assign.sum() * 0.0
    if torch.any(same_teacher):
        same_raw = (plane_assign[edge_i[same_teacher]] - plane_assign[edge_j[same_teacher]]).abs().mean(dim=1)
        same_w = (1.0 - float(line_smooth_suppress) * edge_boundary_conf[same_teacher]).clamp_min(0.05)
        same_loss = (same_raw * same_w).sum() / same_w.sum().clamp_min(1e-6)
    if torch.any(diff_teacher):
        same_prob = (plane_assign[edge_i[diff_teacher]] * plane_assign[edge_j[diff_teacher]]).sum(dim=1)
        boundary_raw = F.relu(same_prob - margin)
        boundary_w = 1.0 + float(line_boundary_boost) * edge_boundary_conf[diff_teacher]
        boundary_loss = (boundary_raw * boundary_w).sum() / boundary_w.sum().clamp_min(1e-6)
    return same_loss, boundary_loss


def boundary_side_consistency_loss(assign, labels, dists, boundary_neighbor, num_planes, distance_margin=0.05, distance_scale=0.03):
    plane_assign = assign[:, :-1]
    valid = (labels >= 0) & (labels < num_planes)
    if not torch.any(valid):
        return assign.sum() * 0.0
    rows = torch.nonzero(valid, as_tuple=False).flatten()
    own_labels = labels[rows]
    own_dist = dists[rows, own_labels].unsqueeze(1)
    plane_ids = torch.arange(num_planes, device=assign.device).view(1, -1)
    other = plane_ids != own_labels.view(-1, 1)
    neighbor = boundary_neighbor[rows, :num_planes] > 0.5
    candidate = other & neighbor
    if not torch.any(candidate):
        return assign.sum() * 0.0
    # This is a side-of-boundary correction, not a generic nearest-plane rule:
    # it only fires for adjacent teacher plane pairs and stays weak when the
    # plane distances are genuinely ambiguous.
    advantage = dists[rows, :num_planes] - own_dist
    side_weight = torch.sigmoid((advantage - float(distance_margin)) / max(float(distance_scale), 1e-6))
    loss = plane_assign[rows, :num_planes] * side_weight
    return loss[candidate].mean()


def patch_compactness_loss(centroids, assign):
    plane_assign = assign[:, :-1]
    weights = plane_assign / plane_assign.sum(dim=0, keepdim=True).clamp_min(1e-6)
    centers = weights.t() @ centroids
    sq = (centroids[:, None, :] - centers[None, :, :]).pow(2).sum(dim=-1)
    per_plane = (plane_assign * sq).sum(dim=0) / plane_assign.sum(dim=0).clamp_min(1e-6)
    active = plane_assign.mean(dim=0) > 0.01
    if not torch.any(active):
        return assign.sum() * 0.0
    return per_plane[active].mean()


def refit_plane_from_points(points, reference_normal=None):
    centroid = points.mean(axis=0).astype(np.float32)
    centered = points - centroid[None, :]
    _, _, vh = np.linalg.svd(centered.astype(np.float32), full_matrices=False)
    normal = vh[-1].astype(np.float32)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-6)
    if reference_normal is not None and float(np.dot(normal, reference_normal)) < 0.0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    return normal.astype(np.float32), offset


def teacher_plane_coverage_losses(assign, labels, num_planes, min_recall=0.82, max_leakage=0.18, label_conf=None):
    plane_assign = assign[:, :-1]
    if label_conf is None:
        label_conf = torch.ones_like(labels, dtype=plane_assign.dtype)
    recall_losses = []
    leakage_losses = []
    for plane_id in range(num_planes):
        own = labels == plane_id
        if torch.any(own):
            own_w = label_conf[own].clamp_min(0.2)
            recall = (plane_assign[own, plane_id] * own_w).sum() / own_w.sum().clamp_min(1e-6)
            recall_losses.append(F.relu(float(min_recall) - recall))
        other = (labels >= 0) & (labels != plane_id)
        if torch.any(other):
            other_w = label_conf[other].clamp_min(0.2)
            leakage = (plane_assign[other, plane_id] * other_w).sum() / other_w.sum().clamp_min(1e-6)
            leakage_losses.append(F.relu(leakage - float(max_leakage)))
    zero = assign.sum() * 0.0
    recall_loss = torch.stack(recall_losses).mean() if recall_losses else zero
    leakage_loss = torch.stack(leakage_losses).mean() if leakage_losses else zero
    return recall_loss, leakage_loss


def teacher_pairwise_leakage_loss(assign, labels, num_planes, max_pair_leakage=0.06, label_conf=None):
    plane_assign = assign[:, :-1]
    if label_conf is None:
        label_conf = torch.ones_like(labels, dtype=plane_assign.dtype)
    losses = []
    for teacher_id in range(num_planes):
        own = labels == teacher_id
        if not torch.any(own):
            continue
        own_w = label_conf[own].clamp_min(0.2)
        for pred_id in range(num_planes):
            if pred_id == teacher_id:
                continue
            leakage = (plane_assign[own, pred_id] * own_w).sum() / own_w.sum().clamp_min(1e-6)
            losses.append(F.relu(leakage - float(max_pair_leakage)))
    if not losses:
        return assign.sum() * 0.0
    return torch.stack(losses).mean()


def support_violation_loss(assign, support_inside, boundary_neighbor, labels, num_planes):
    plane_assign = assign[:, :-1]
    valid_planes = plane_assign[:, :num_planes]
    outside = support_inside[:, :num_planes] < 0.5
    boundary = boundary_neighbor[:, :num_planes] > 0.5
    own = F.one_hot(labels.clamp_min(0).clamp_max(num_planes - 1), num_classes=num_planes).bool()
    valid_label = (labels >= 0).view(-1, 1)
    # Do not punish the teacher-owned support or immediate line/boundary neighbors as harshly.
    violation = outside & (~boundary) & (~(own & valid_label))
    if not torch.any(violation):
        return assign.sum() * 0.0
    return valid_planes[violation].mean()


def compute_loss(model, tensors, args, temperature):
    (
        patch_centroids,
        patch_normals,
        patch_features,
        patch_labels,
        patch_label_conf,
        patch_counts,
        edge_i,
        edge_j,
        edge_line_prob,
        edge_boundary_conf,
        aux_normals,
        aux_offsets,
        support_prior,
        teacher_label_prior,
        support_inside,
        boundary_neighbor,
    ) = tensors
    normals, offsets, dists, normal_align, assign = model.forward_one(
        patch_centroids,
        patch_normals,
        patch_features,
        temperature,
        args.distance_logit_weight,
        args.normal_logit_weight,
        teacher_normals=aux_normals,
        teacher_offsets=aux_offsets,
        teacher_param_blend=args.teacher_param_blend,
        support_prior=support_prior,
        teacher_label_prior=teacher_label_prior,
    )
    loss_labels = patch_labels
    if args.permutation_invariant_matching:
        loss_labels = permutation_invariant_labels(
            assign,
            patch_labels,
            patch_label_conf,
            normals,
            offsets,
            aux_normals,
            aux_offsets,
            args.match_classification_weight,
            args.match_normal_weight,
            args.match_offset_weight,
        )
    plane_assign = assign[:, :-1]
    teacher_loss = class_balanced_patch_ce(assign, loss_labels, args.num_planes, patch_label_conf)
    bg_loss = background_patch_loss(assign, patch_labels, args.num_planes)
    foreground = plane_assign.sum(dim=-1).clamp_min(1e-6)
    residual = (plane_assign * dists).sum(dim=-1) / foreground
    residual_loss = (residual * patch_counts).sum() / patch_counts.sum().clamp_min(1e-6)
    normal_loss = ((plane_assign * (1.0 - normal_align)).sum(dim=-1) / foreground * patch_counts).sum() / patch_counts.sum().clamp_min(1e-6)
    same_loss, boundary_loss = patch_smoothness_loss(
        assign,
        edge_i,
        edge_j,
        loss_labels,
        edge_line_prob=edge_line_prob,
        edge_boundary_conf=edge_boundary_conf,
        margin=args.patch_boundary_margin,
        line_smooth_suppress=args.line_smooth_suppress,
        line_boundary_boost=args.line_boundary_boost,
    )
    side_loss = boundary_side_consistency_loss(
        assign,
        loss_labels,
        dists,
        boundary_neighbor,
        args.num_planes,
        distance_margin=args.boundary_side_distance_margin,
        distance_scale=args.boundary_side_distance_scale,
    )
    compact_loss = patch_compactness_loss(patch_centroids, assign)
    coverage_loss, overcoverage_loss = teacher_plane_coverage_losses(
        assign,
        loss_labels,
        args.num_planes,
        min_recall=args.teacher_min_plane_recall,
        max_leakage=args.teacher_max_plane_leakage,
        label_conf=patch_label_conf,
    )
    pairwise_leakage_loss = teacher_pairwise_leakage_loss(
        assign,
        loss_labels,
        args.num_planes,
        max_pair_leakage=args.teacher_max_pair_leakage,
        label_conf=patch_label_conf,
    )
    support_loss = support_violation_loss(assign, support_inside, boundary_neighbor, loss_labels, args.num_planes)
    ent_loss = entropy(assign).mean()
    present_queries = torch.zeros(args.num_planes, dtype=torch.bool, device=assign.device)
    valid_loss_labels = loss_labels[(loss_labels >= 0) & (loss_labels < args.num_planes)]
    if valid_loss_labels.numel():
        present_queries[torch.unique(valid_loss_labels)] = True
    inactive_loss = (
        plane_assign[:, ~present_queries].mean()
        if torch.any(~present_queries)
        else assign.sum() * 0.0
    )
    loss = (
        args.teacher_assignment_weight * teacher_loss
        + args.background_weight * bg_loss
        + args.patch_residual_weight * residual_loss
        + args.patch_normal_weight * normal_loss
        + args.patch_smooth_weight * same_loss
        + args.patch_boundary_weight * boundary_loss
        + args.boundary_side_weight * side_loss
        + args.patch_compactness_weight * compact_loss
        + args.teacher_coverage_weight * coverage_loss
        + args.teacher_overcoverage_weight * overcoverage_loss
        + args.teacher_pairwise_leakage_weight * pairwise_leakage_loss
        + args.support_violation_weight * support_loss
        + args.entropy_weight * ent_loss
        + args.inactive_query_weight * inactive_loss
    )
    stats = {
        "loss": loss.detach(),
        "teacher_assignment_loss": teacher_loss.detach(),
        "background_loss": bg_loss.detach(),
        "patch_residual_loss": residual_loss.detach(),
        "patch_normal_loss": normal_loss.detach(),
        "patch_smooth_loss": same_loss.detach(),
        "patch_boundary_loss": boundary_loss.detach(),
        "boundary_side_loss": side_loss.detach(),
        "patch_compactness_loss": compact_loss.detach(),
        "teacher_coverage_loss": coverage_loss.detach(),
        "teacher_overcoverage_loss": overcoverage_loss.detach(),
        "teacher_pairwise_leakage_loss": pairwise_leakage_loss.detach(),
        "support_violation_loss": support_loss.detach(),
        "mean_edge_line_prob": edge_line_prob.mean().detach() if edge_line_prob.numel() else assign.sum().detach() * 0.0,
        "entropy": ent_loss.detach(),
        "inactive_query_loss": inactive_loss.detach(),
        "background_ratio": assign[:, -1].mean().detach(),
    }
    return loss, stats


def export_case(model, case, output_dir, args, device):
    patch_centroids = torch.from_numpy(case["patch_centroids"]).to(device)
    patch_normals = torch.from_numpy(case["patch_normals"]).to(device)
    patch_features = torch.from_numpy(case["patch_features"]).to(device)
    aux_normals = torch.from_numpy(case.get("aux_plane_normals", np.zeros((0, 3), dtype=np.float32))).to(device)
    aux_offsets = torch.from_numpy(case.get("aux_plane_offsets_norm", np.zeros((0,), dtype=np.float32))).to(device)
    support_prior = torch.from_numpy(case["patch_support_prior"]).to(device)
    teacher_label_prior = torch.from_numpy(case["patch_teacher_label_prior"]).to(device)
    with torch.no_grad():
        normals, offsets_norm, dists, normal_align, assign = model.forward_one(
            patch_centroids,
            patch_normals,
            patch_features,
            args.min_temperature,
            args.distance_logit_weight,
            args.normal_logit_weight,
            teacher_normals=aux_normals,
            teacher_offsets=aux_offsets,
            teacher_param_blend=args.teacher_param_blend,
            support_prior=support_prior,
            teacher_label_prior=teacher_label_prior,
        )
    normals_np = normals.detach().cpu().numpy()
    offsets_norm_np = offsets_norm.detach().cpu().numpy()
    assign_np = assign.detach().cpu().numpy()
    patch_assign = assign.argmax(dim=-1).detach().cpu().numpy().astype(np.int32)
    patch_assign[patch_assign >= args.num_planes] = -1
    dists_np = dists.detach().cpu().numpy()
    teacher_delta_flips = 0
    teacher_delta_locked = 0
    if args.teacher_delta_export:
        labels_np = case["patch_labels"].astype(np.int32)
        label_conf_np = case["patch_label_conf"].astype(np.float32)
        boundary_neighbor_np = case["patch_boundary_neighbor"].astype(np.float32)
        refined = patch_assign.copy()
        plane_probs = assign_np[:, : args.num_planes]
        for patch_id, teacher_id in enumerate(labels_np):
            if teacher_id < 0 or teacher_id >= args.num_planes:
                continue
            candidates = boundary_neighbor_np[patch_id, : args.num_planes] > 0.5
            candidates[teacher_id] = True
            if not np.any(candidates):
                candidates[teacher_id] = True
            candidate_ids = np.flatnonzero(candidates)
            best_id = int(candidate_ids[np.argmax(plane_probs[patch_id, candidate_ids])])
            teacher_prob = float(plane_probs[patch_id, teacher_id])
            best_prob = float(plane_probs[patch_id, best_id])
            teacher_dist = float(dists_np[patch_id, teacher_id])
            best_dist = float(dists_np[patch_id, best_id])
            confident_teacher = float(label_conf_np[patch_id]) >= args.teacher_delta_lock_conf
            can_flip = (
                best_id != int(teacher_id)
                and best_prob >= teacher_prob + args.teacher_delta_flip_prob_margin
                and best_dist + args.teacher_delta_flip_residual_margin < teacher_dist
            )
            if confident_teacher:
                refined[patch_id] = best_id if (can_flip and args.teacher_delta_allow_confident_flip) else int(teacher_id)
                teacher_delta_locked += 0 if refined[patch_id] != int(teacher_id) else 1
            else:
                refined[patch_id] = best_id if can_flip or args.teacher_delta_low_conf_model else int(teacher_id)
            if refined[patch_id] != patch_assign[patch_id]:
                teacher_delta_flips += 1
        patch_assign = refined
    if args.patch_residual_reassign:
        active_candidate_ids = np.arange(args.num_planes, dtype=np.int64)
        for patch_id, plane_id in enumerate(patch_assign.copy()):
            if plane_id < 0:
                continue
            current_residual = float(dists_np[patch_id, plane_id])
            best_plane = int(active_candidate_ids[np.argmin(dists_np[patch_id, active_candidate_ids])])
            best_residual = float(dists_np[patch_id, best_plane])
            if (
                current_residual > args.patch_reassign_max_residual
                and best_plane != int(plane_id)
                and best_residual + args.patch_reassign_margin < current_residual
            ):
                patch_assign[patch_id] = best_plane
    point_assignment = np.full(len(case["points"]), -1, dtype=np.int32)
    for patch_id, plane_id in enumerate(patch_assign):
        point_assignment[case["point_to_patch"] == patch_id] = int(plane_id)

    refit_planes = 0
    if args.refit_plane_params_from_support:
        points_norm = case["points_norm"].astype(np.float32)
        for plane_id in range(args.num_planes):
            mask = point_assignment == plane_id
            if int(mask.sum()) < args.refit_min_points:
                continue
            normal, offset = refit_plane_from_points(points_norm[mask], normals_np[plane_id])
            normals_np[plane_id] = normal
            offsets_norm_np[plane_id] = float(offset)
            refit_planes += 1
        dists_np = np.abs(case["patch_centroids"].astype(np.float32) @ normals_np.T + offsets_norm_np.reshape(1, -1))

    offsets_world = [float(d * case["scale"] - float(np.dot(n, case["center"]))) for n, d in zip(normals_np, offsets_norm_np)]
    active = []
    params = []
    for i in range(args.num_planes):
        mask = point_assignment == i
        patch_mask = patch_assign == i
        mean_residual = float(dists_np[patch_mask, i].mean()) if patch_mask.any() else None
        assigned_ratio = float(mask.mean())
        is_active = bool(assigned_ratio >= args.export_min_coverage and mean_residual is not None and mean_residual <= args.export_max_mean_residual)
        active.append(is_active)
        params.append(
            {
                "id": int(i),
                "normal": [float(x) for x in normals_np[i]],
                "offset": offsets_world[i],
                "offset_normalized": float(offsets_norm_np[i]),
                "assigned_point_count": int(mask.sum()),
                "assigned_patch_count": int(patch_mask.sum()),
                "assigned_ratio": assigned_ratio,
                "mean_abs_distance_normalized": mean_residual,
                "active": is_active,
            }
        )
    inactive = np.asarray([not x for x in active], dtype=bool)
    if inactive.any():
        inactive_ids = np.flatnonzero(inactive)
        patch_assign[np.isin(patch_assign, inactive_ids)] = -1
        point_assignment[np.isin(point_assignment, inactive_ids)] = -1

    colors = np.zeros_like(case["colors"], dtype=np.uint8)
    colors[point_assignment < 0] = np.asarray([160, 160, 160], dtype=np.uint8)
    valid = point_assignment >= 0
    colors[valid] = PLANE_COLORS[point_assignment[valid] % len(PLANE_COLORS)]
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{case['stem']}_patch_plane_tokens.json"
    npz_path = output_dir / f"{case['stem']}_patch_plane_tokens_assignment.npz"
    summary = {
        "input_npz": case["path"],
        "method": "patch_plane_token_head",
        "num_points_used": int(len(case["points"])),
        "num_patches": int(len(case["patch_features"])),
        "num_planes": int(args.num_planes),
        "active_planes": int(sum(active)),
        "background_point_count": int(np.sum(point_assignment < 0)),
        "background_ratio": float(np.mean(point_assignment < 0)),
        "teacher_delta_export": bool(args.teacher_delta_export),
        "teacher_delta_flips": int(teacher_delta_flips),
        "teacher_delta_locked": int(teacher_delta_locked),
        "refit_plane_params_from_support": bool(args.refit_plane_params_from_support),
        "refit_planes": int(refit_planes),
        "center": [float(x) for x in case["center"]],
        "scale": float(case["scale"]),
        "planes": params,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        points=case["points"].astype(np.float32),
        colors=colors.astype(np.uint8),
        original_colors=case["colors"].astype(np.uint8),
        assignment=point_assignment.astype(np.int32),
        patch_assignment=patch_assign.astype(np.int32),
        point_to_patch=case["point_to_patch"].astype(np.int32),
        patch_centroids=case["patch_centroids"].astype(np.float32),
        patch_normals=case["patch_normals"].astype(np.float32),
        patch_labels=case["patch_labels"].astype(np.int32),
        patch_support_prior=case["patch_support_prior"].astype(np.float32),
        patch_teacher_label_prior=case["patch_teacher_label_prior"].astype(np.float32),
        patch_support_inside=case["patch_support_inside"].astype(np.float32),
        patch_support_distance_cells=case["patch_support_distance_cells"].astype(np.float32),
        patch_boundary_neighbor=case["patch_boundary_neighbor"].astype(np.float32),
        patch_edge_line_prob=case["patch_edge_line_prob"].astype(np.float32),
        patch_edge_boundary_conf=case["patch_edge_boundary_conf"].astype(np.float32),
        plane_normals=normals_np.astype(np.float32),
        plane_offsets=np.asarray(offsets_world, dtype=np.float32),
        plane_offsets_normalized=offsets_norm_np.astype(np.float32),
        active_planes=np.asarray(active, dtype=np.bool_),
    )
    return json_path, npz_path, summary


def main():
    parser = argparse.ArgumentParser("Patch-level Stage2 plane-token assignment")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_checkpoint", default=None)
    parser.add_argument("--load_checkpoint", default=None)
    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--pattern", default="*_stage1_teacher_full_pointcloud_editable_planes_data.npz")
    parser.add_argument("--num_planes", type=int, default=8)
    parser.add_argument("--max_points_per_sample", type=int, default=18000)
    parser.add_argument("--smooth_pairs_per_sample", type=int, default=40000)
    parser.add_argument("--smooth_candidates", type=int, default=32)
    parser.add_argument("--smooth_xyz_sigma", type=float, default=0.045)
    parser.add_argument("--smooth_rgb_sigma", type=float, default=0.18)
    parser.add_argument("--patch_pixel_size", type=float, default=0.08)
    parser.add_argument("--patch_voxel_size", type=float, default=0.045)
    parser.add_argument("--min_patch_points", type=int, default=12)
    parser.add_argument(
        "--hide_teacher_feature_conf",
        action="store_true",
        help="Do not expose GT-derived patch label confidence to the network input.",
    )
    parser.add_argument("--permutation_invariant_matching", action="store_true")
    parser.add_argument("--match_classification_weight", type=float, default=1.0)
    parser.add_argument("--match_normal_weight", type=float, default=0.25)
    parser.add_argument("--match_offset_weight", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=2600)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=160)
    parser.add_argument("--context_dim", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--temperature_decay", type=float, default=0.9992)
    parser.add_argument("--min_temperature", type=float, default=0.025)
    parser.add_argument("--distance_logit_weight", type=float, default=0.30)
    parser.add_argument("--normal_logit_weight", type=float, default=0.06)
    parser.add_argument("--teacher_param_blend", type=float, default=1.0)
    parser.add_argument("--teacher_assignment_weight", type=float, default=1.2)
    parser.add_argument("--teacher_label_logit_weight", type=float, default=0.0)
    parser.add_argument("--background_weight", type=float, default=0.004)
    parser.add_argument("--patch_residual_weight", type=float, default=0.18)
    parser.add_argument("--patch_normal_weight", type=float, default=0.08)
    parser.add_argument("--patch_smooth_weight", type=float, default=0.10)
    parser.add_argument("--patch_boundary_weight", type=float, default=0.18)
    parser.add_argument("--patch_boundary_margin", type=float, default=0.12)
    parser.add_argument("--line_smooth_suppress", type=float, default=0.0)
    parser.add_argument("--line_boundary_boost", type=float, default=0.0)
    parser.add_argument("--edge_boundary_normal_gap", type=float, default=0.12)
    parser.add_argument("--boundary_side_weight", type=float, default=0.0)
    parser.add_argument("--boundary_side_distance_margin", type=float, default=0.05)
    parser.add_argument("--boundary_side_distance_scale", type=float, default=0.03)
    parser.add_argument("--patch_compactness_weight", type=float, default=0.004)
    parser.add_argument("--teacher_coverage_weight", type=float, default=0.0)
    parser.add_argument("--teacher_overcoverage_weight", type=float, default=0.0)
    parser.add_argument("--teacher_pairwise_leakage_weight", type=float, default=0.0)
    parser.add_argument("--teacher_min_plane_recall", type=float, default=0.82)
    parser.add_argument("--teacher_max_plane_leakage", type=float, default=0.18)
    parser.add_argument("--teacher_max_pair_leakage", type=float, default=0.06)
    parser.add_argument("--support_logit_weight", type=float, default=0.0)
    parser.add_argument("--boundary_support_logit_weight", type=float, default=0.0)
    parser.add_argument("--outside_support_penalty", type=float, default=0.0)
    parser.add_argument("--support_violation_weight", type=float, default=0.0)
    parser.add_argument("--support_grid_size", type=float, default=0.055)
    parser.add_argument("--support_dilate_cells", type=int, default=2)
    parser.add_argument("--support_max_distance_cells", type=int, default=10)
    parser.add_argument("--support_min_label_conf", type=float, default=0.60)
    parser.add_argument("--entropy_weight", type=float, default=0.001)
    parser.add_argument("--inactive_query_weight", type=float, default=0.06)
    parser.add_argument("--export_min_coverage", type=float, default=0.02)
    parser.add_argument("--export_max_mean_residual", type=float, default=0.15)
    parser.add_argument("--patch_residual_reassign", action="store_true")
    parser.add_argument("--patch_reassign_max_residual", type=float, default=0.20)
    parser.add_argument("--patch_reassign_margin", type=float, default=0.03)
    parser.add_argument("--teacher_delta_export", action="store_true")
    parser.add_argument("--teacher_delta_lock_conf", type=float, default=0.65)
    parser.add_argument("--teacher_delta_flip_prob_margin", type=float, default=0.15)
    parser.add_argument("--teacher_delta_flip_residual_margin", type=float, default=0.03)
    parser.add_argument("--teacher_delta_low_conf_model", action="store_true")
    parser.add_argument("--teacher_delta_allow_confident_flip", action="store_true")
    parser.add_argument("--refit_plane_params_from_support", action="store_true")
    parser.add_argument("--refit_min_points", type=int, default=300)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    paths = sorted(Path(args.input_dir).glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No npz files matched under {args.input_dir}")
    cases = [build_surface_patches(sample_case_with_aux_planes(p, args, args.seed + i * 97), args) for i, p in enumerate(paths)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = []
    for case in cases:
        tensors.append(
            (
                torch.from_numpy(case["patch_centroids"]).to(device),
                torch.from_numpy(case["patch_normals"]).to(device),
                torch.from_numpy(case["patch_features"]).to(device),
                torch.from_numpy(case["patch_labels"]).to(device),
                torch.from_numpy(case["patch_label_conf"]).to(device),
                torch.from_numpy(case["patch_counts"]).to(device),
                torch.from_numpy(case["patch_edge_i"]).to(device),
                torch.from_numpy(case["patch_edge_j"]).to(device),
                torch.from_numpy(case["patch_edge_line_prob"]).to(device),
                torch.from_numpy(case["patch_edge_boundary_conf"]).to(device),
                torch.from_numpy(case.get("aux_plane_normals", np.zeros((0, 3), dtype=np.float32))).to(device),
                torch.from_numpy(case.get("aux_plane_offsets_norm", np.zeros((0,), dtype=np.float32))).to(device),
                torch.from_numpy(case["patch_support_prior"]).to(device),
                torch.from_numpy(case["patch_teacher_label_prior"]).to(device),
                torch.from_numpy(case["patch_support_inside"]).to(device),
                torch.from_numpy(case["patch_boundary_neighbor"]).to(device),
            )
        )
    model = PatchPlaneTokenHead(
        args.num_planes,
        patch_feature_dim=tensors[0][2].shape[1],
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
    ).to(device)
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
        print(f"loaded_checkpoint={args.load_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    history = []
    if args.export_only:
        args.steps = 0
    for step in range(1, args.steps + 1):
        temperature = max(args.min_temperature, args.temperature * (args.temperature_decay ** step))
        sample_ids = rng.choice(len(cases), size=min(args.sample_batch_size, len(cases)), replace=False)
        losses = []
        stat_rows = []
        for sid in sample_ids:
            loss, stats = compute_loss(model, tensors[int(sid)], args, temperature)
            losses.append(loss)
            stat_rows.append(stats)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {"step": int(step), "loss": float(loss.detach().cpu()), "temperature": float(temperature)}
        for key in stat_rows[0]:
            row[key] = float(torch.stack([s[key] for s in stat_rows]).mean().cpu())
        history.append(row)
        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:04d} loss={row['loss']:.5f} teacher={row['teacher_assignment_loss']:.4f} "
                f"res={row['patch_residual_loss']:.4f} normal={row['patch_normal_loss']:.4f} "
                f"smooth={row['patch_smooth_loss']:.4f} boundary={row['patch_boundary_loss']:.4f} "
                f"side={row['boundary_side_loss']:.4f} "
                f"cover={row['teacher_coverage_loss']:.4f}/{row['teacher_overcoverage_loss']:.4f} "
                f"pair={row['teacher_pairwise_leakage_loss']:.4f} support={row['support_violation_loss']:.4f} "
                f"line={row['mean_edge_line_prob']:.3f} bg={row['background_ratio']:.3f}"
            )

    output_dir = Path(args.output_dir)
    exported = []
    for case in cases:
        json_path, npz_path, summary = export_case(model, case, output_dir, args, device)
        exported.append({"json": str(json_path), "npz": str(npz_path), "summary": summary})
    overview = {
        "input_dir": args.input_dir,
        "num_samples": len(cases),
        "num_planes": args.num_planes,
        "method": "patch_plane_token_head",
        "patch_pixel_size": args.patch_pixel_size,
        "min_patch_points": args.min_patch_points,
        "teacher_param_blend": args.teacher_param_blend,
        "teacher_label_logit_weight": args.teacher_label_logit_weight,
        "teacher_coverage_weight": args.teacher_coverage_weight,
        "teacher_overcoverage_weight": args.teacher_overcoverage_weight,
        "teacher_pairwise_leakage_weight": args.teacher_pairwise_leakage_weight,
        "teacher_min_plane_recall": args.teacher_min_plane_recall,
        "teacher_max_plane_leakage": args.teacher_max_plane_leakage,
        "teacher_max_pair_leakage": args.teacher_max_pair_leakage,
        "line_smooth_suppress": args.line_smooth_suppress,
        "line_boundary_boost": args.line_boundary_boost,
        "edge_boundary_normal_gap": args.edge_boundary_normal_gap,
        "boundary_side_weight": args.boundary_side_weight,
        "boundary_side_distance_margin": args.boundary_side_distance_margin,
        "boundary_side_distance_scale": args.boundary_side_distance_scale,
        "support_logit_weight": args.support_logit_weight,
        "boundary_support_logit_weight": args.boundary_support_logit_weight,
        "outside_support_penalty": args.outside_support_penalty,
        "support_violation_weight": args.support_violation_weight,
        "support_grid_size": args.support_grid_size,
        "support_dilate_cells": args.support_dilate_cells,
        "support_max_distance_cells": args.support_max_distance_cells,
        "history": history,
        "exported": exported,
    }
    overview_path = output_dir / "patch_plane_tokens_summary.json"
    overview_path.write_text(json.dumps(overview, indent=2), encoding="utf-8")
    if args.save_checkpoint:
        checkpoint_path = Path(args.save_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args), "history": history, "overview": overview}, checkpoint_path)
        print(checkpoint_path)
    print(overview_path)
    print(f"samples={len(cases)}")


if __name__ == "__main__":
    main()
