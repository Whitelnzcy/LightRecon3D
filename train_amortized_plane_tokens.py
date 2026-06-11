import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_multisample_unsupervised_plane_tokens import (
    PLANE_COLORS,
    confidence_separation_loss,
    coverage_loss,
    dead_token_loss,
    diversity_loss,
    entropy,
    local_smoothness_loss,
    sample_case,
    trimmed_mean,
)


def sample_case_with_aux_planes(path, args, seed):
    case = sample_case(
        path,
        args.max_points_per_sample,
        seed,
        args.smooth_pairs_per_sample,
        args.smooth_candidates,
        args.smooth_xyz_sigma,
        args.smooth_rgb_sigma,
    )
    raw = np.load(path)
    if "pixel_xy" in raw:
        pixel_xy = raw["pixel_xy"].astype(np.float32)
        rng = np.random.default_rng(seed)
        if args.max_points_per_sample > 0 and len(pixel_xy) > args.max_points_per_sample:
            idx = rng.choice(len(pixel_xy), size=args.max_points_per_sample, replace=False)
            pixel_xy = pixel_xy[idx]
        if len(pixel_xy) == len(case["features"]):
            case["features"] = np.concatenate([case["features"], pixel_xy], axis=1).astype(np.float32)
    if "line_prob" in raw:
        line_prob = raw["line_prob"].astype(np.float32).reshape(-1)
        rng = np.random.default_rng(seed)
        if args.max_points_per_sample > 0 and len(line_prob) > args.max_points_per_sample:
            idx = rng.choice(len(line_prob), size=args.max_points_per_sample, replace=False)
            line_prob = line_prob[idx]
        if len(line_prob) == len(case["features"]):
            case["point_line_prob"] = line_prob.astype(np.float32)
    if "plane_normals" not in raw or "plane_offsets" not in raw:
        return case

    normals = raw["plane_normals"].astype(np.float32)
    offsets = raw["plane_offsets"].astype(np.float32)
    counts = raw["plane_inlier_counts"].astype(np.float32) if "plane_inlier_counts" in raw else np.ones(len(normals))
    plane_ids = raw["plane_ids"].astype(np.int32) if "plane_ids" in raw else np.arange(len(normals), dtype=np.int32)
    order = np.argsort(-counts)[: args.num_planes]
    normals = normals[order]
    offsets = offsets[order]
    plane_ids = plane_ids[order]
    counts = counts[order]

    offsets_norm = []
    for n, d_world in zip(normals, offsets):
        offsets_norm.append((float(d_world) + float(np.dot(n, case["center"]))) / case["scale"])
    case["aux_plane_normals"] = normals.astype(np.float32)
    case["aux_plane_offsets_norm"] = np.asarray(offsets_norm, dtype=np.float32)
    case["aux_plane_ids"] = plane_ids.astype(np.int32)
    case["aux_plane_counts"] = counts.astype(np.float32)

    if "point_plane_ids" in raw:
        # Recreate the same subsampling deterministically so pseudo labels align with sampled points.
        point_ids = raw["point_plane_ids"].astype(np.int32)
        rng = np.random.default_rng(seed)
        if args.max_points_per_sample > 0 and len(point_ids) > args.max_points_per_sample:
            idx = rng.choice(len(point_ids), size=args.max_points_per_sample, replace=False)
            point_ids = point_ids[idx]
        remap = {int(pid): i for i, pid in enumerate(plane_ids)}
        labels = np.full(len(point_ids), -1, dtype=np.int64)
        for pid, target_idx in remap.items():
            labels[point_ids == pid] = int(target_idx)
        case["aux_point_labels"] = labels
    return case


class AmortizedPlaneTokenHead(nn.Module):
    """Predict per-sample plane tokens from the input point cloud instead of optimizing them per sample."""

    def __init__(self, num_planes, point_feature_dim=7, hidden_dim=192, context_dim=256):
        super().__init__()
        self.num_planes = num_planes
        self.point_encoder = nn.Sequential(
            nn.Linear(point_feature_dim, hidden_dim),
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
        assign_input_dim = point_feature_dim + 3 + 1 + 1
        self.assignment_head = nn.Sequential(
            nn.Linear(assign_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def predict_tokens(self, features):
        encoded = self.point_encoder(features)
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
        points,
        features,
        temperature,
        distance_logit_weight,
        teacher_normals=None,
        teacher_offsets=None,
        teacher_param_blend=0.0,
    ):
        normals, offsets, token_logits, background_logit = self.predict_tokens(features)
        if (
            teacher_param_blend > 0
            and teacher_normals is not None
            and teacher_offsets is not None
            and teacher_normals.numel() > 0
        ):
            m = min(normals.shape[0], teacher_normals.shape[0])
            blend = float(teacher_param_blend)
            blended_normals = normals.clone()
            blended_offsets = offsets.clone()
            blended_normals[:m] = F.normalize(
                (1.0 - blend) * normals[:m] + blend * teacher_normals[:m],
                dim=-1,
            )
            blended_offsets[:m] = (1.0 - blend) * offsets[:m] + blend * teacher_offsets[:m]
            normals = blended_normals
            offsets = blended_offsets
        dists = torch.abs(points @ normals.t() + offsets.view(1, -1))
        n_points = points.shape[0]
        n_planes = normals.shape[0]
        point_feat = features[:, None, :].expand(n_points, n_planes, -1)
        normal_feat = normals[None, :, :].expand(n_points, n_planes, -1)
        offset_feat = offsets.view(1, n_planes, 1).expand(n_points, n_planes, 1)
        pair_feat = torch.cat([point_feat, normal_feat, offset_feat, dists.unsqueeze(-1)], dim=-1)
        logits = self.assignment_head(pair_feat).squeeze(-1)
        logits = logits - distance_logit_weight * dists / temperature + token_logits.view(1, -1)
        background_logits = background_logit.expand(n_points, 1)
        logits = torch.cat([logits, background_logits], dim=1)
        assign = F.softmax(logits, dim=-1)
        return normals, offsets, dists, assign


def match_predicted_to_aux_planes(pred_normals, pred_offsets, aux_normals, aux_offsets):
    if aux_normals.numel() == 0:
        zero = pred_normals.sum() * 0.0
        return zero, []
    k = pred_normals.shape[0]
    m = min(k, aux_normals.shape[0])
    aux_normals = aux_normals[:m]
    aux_offsets = aux_offsets[:m]
    dot = pred_normals @ aux_normals.t()
    same_cost = (1.0 - dot.clamp(-1, 1)) + torch.abs(pred_offsets[:, None] - aux_offsets[None, :])
    flip_cost = (1.0 + dot.clamp(-1, 1)) + torch.abs(pred_offsets[:, None] + aux_offsets[None, :])
    cost = torch.minimum(same_cost, flip_cost)
    available_pred = set(range(k))
    available_aux = set(range(m))
    matches = []
    cost_np = cost.detach().cpu().numpy()
    while available_pred and available_aux:
        best = min(
            ((cost_np[pred_id, aux_id], pred_id, aux_id) for pred_id in available_pred for aux_id in available_aux),
            key=lambda x: x[0],
        )
        _, pred_id, aux_id = best
        matches.append((int(pred_id), int(aux_id)))
        available_pred.remove(pred_id)
        available_aux.remove(aux_id)
    if not matches:
        return cost.sum() * 0.0, []
    pred_ids = torch.tensor([p for p, _ in matches], dtype=torch.long, device=cost.device)
    aux_ids = torch.tensor([a for _, a in matches], dtype=torch.long, device=cost.device)
    return cost[pred_ids, aux_ids].mean(), matches


def auxiliary_assignment_loss(assign, aux_point_labels, matches):
    if aux_point_labels.numel() == 0 or not matches:
        return assign.sum() * 0.0
    target_to_pred = {target_id: pred_id for pred_id, target_id in matches}
    labels = torch.full_like(aux_point_labels, -1)
    for target_id, pred_id in target_to_pred.items():
        labels[aux_point_labels == target_id] = pred_id
    valid = labels >= 0
    if not torch.any(valid):
        return assign.sum() * 0.0
    return F.nll_loss(torch.log(assign[valid, :-1].clamp_min(1e-8)), labels[valid])


def auxiliary_background_loss(assign, aux_point_labels, num_planes):
    if aux_point_labels.numel() == 0:
        return assign.sum() * 0.0
    background = aux_point_labels < 0
    if not torch.any(background):
        return assign.sum() * 0.0
    targets = torch.full(
        (int(background.sum().item()),),
        int(num_planes),
        dtype=torch.long,
        device=assign.device,
    )
    return F.nll_loss(torch.log(assign[background].clamp_min(1e-8)), targets)


def ordered_teacher_param_loss(pred_normals, pred_offsets, aux_normals, aux_offsets):
    if aux_normals.numel() == 0:
        return pred_normals.sum() * 0.0
    m = min(pred_normals.shape[0], aux_normals.shape[0])
    pred_n = pred_normals[:m]
    pred_d = pred_offsets[:m]
    aux_n = aux_normals[:m]
    aux_d = aux_offsets[:m]
    dot = (pred_n * aux_n).sum(dim=1).clamp(-1.0, 1.0)
    same = (1.0 - dot) + F.smooth_l1_loss(pred_d, aux_d, reduction="none")
    flipped = (1.0 + dot) + F.smooth_l1_loss(pred_d, -aux_d, reduction="none")
    return torch.minimum(same, flipped).mean()


def ordered_teacher_assignment_loss(assign, aux_point_labels, num_planes, class_balanced=False):
    if aux_point_labels.numel() == 0:
        return assign.sum() * 0.0
    valid = (aux_point_labels >= 0) & (aux_point_labels < num_planes)
    if not torch.any(valid):
        return assign.sum() * 0.0
    log_probs = torch.log(assign[valid].clamp_min(1e-8))
    labels = aux_point_labels[valid]
    if not class_balanced:
        return F.nll_loss(log_probs, labels)
    counts = torch.bincount(labels, minlength=num_planes).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = 1.0 / counts[present].clamp_min(1.0)
    weights = weights * (present.float().sum() / weights[present].sum().clamp_min(1e-6))
    point_weights = weights[labels]
    per_point = F.nll_loss(log_probs, labels, reduction="none")
    return (per_point * point_weights).sum() / point_weights.sum().clamp_min(1e-6)


def inactive_query_loss(plane_assign, active_count):
    if active_count >= plane_assign.shape[1]:
        return plane_assign.sum() * 0.0
    inactive = plane_assign[:, active_count:]
    return inactive.mean()


def teacher_boundary_contrast_loss(plane_assign, aux_point_labels, smooth_i, smooth_j, max_pairs=20000, margin=0.20):
    if aux_point_labels.numel() == 0 or smooth_i.numel() == 0:
        return plane_assign.sum() * 0.0
    valid = (
        (smooth_i >= 0)
        & (smooth_j >= 0)
        & (smooth_i < aux_point_labels.shape[0])
        & (smooth_j < aux_point_labels.shape[0])
    )
    if not torch.any(valid):
        return plane_assign.sum() * 0.0
    i = smooth_i[valid]
    j = smooth_j[valid]
    li = aux_point_labels[i]
    lj = aux_point_labels[j]
    boundary = (li >= 0) & (lj >= 0) & (li != lj)
    if not torch.any(boundary):
        return plane_assign.sum() * 0.0
    i = i[boundary]
    j = j[boundary]
    if max_pairs > 0 and i.numel() > max_pairs:
        keep = torch.randperm(i.numel(), device=i.device)[:max_pairs]
        i = i[keep]
        j = j[keep]
    same_token_prob = (plane_assign[i] * plane_assign[j]).sum(dim=1)
    return F.relu(same_token_prob - margin).mean()


def region_compactness_loss(points, plane_assign, eps=1e-6):
    weights = plane_assign / plane_assign.sum(dim=0, keepdim=True).clamp_min(eps)
    centers = weights.t() @ points
    sq_dist = (points[:, None, :] - centers[None, :, :]).pow(2).sum(dim=-1)
    per_plane = (plane_assign * sq_dist).sum(dim=0) / plane_assign.sum(dim=0).clamp_min(eps)
    active = plane_assign.mean(dim=0) > 0.01
    if not torch.any(active):
        return plane_assign.sum() * 0.0
    return per_plane[active].mean()


def _connected_components_from_edges(nodes, edge_i, edge_j):
    node_set = set(int(x) for x in nodes)
    if not node_set:
        return []
    adjacency = {int(x): [] for x in nodes}
    for a, b in zip(edge_i, edge_j):
        a = int(a)
        b = int(b)
        if a in node_set and b in node_set:
            adjacency[a].append(b)
            adjacency[b].append(a)
    seen = set()
    components = []
    for start in nodes:
        start = int(start)
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adjacency.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(np.asarray(comp, dtype=np.int64))
    components.sort(key=len, reverse=True)
    return components


def _plane_basis(normal):
    normal = normal.astype(np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
    ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, normal))) > 0.9:
        ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(normal, ref)
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    v = np.cross(normal, u)
    v = v / max(float(np.linalg.norm(v)), 1e-9)
    return u.astype(np.float32), v.astype(np.float32)


def _connected_components_from_grid(points_uv, nodes, cell_size):
    if len(nodes) == 0:
        return []
    uv = points_uv[nodes]
    cells = np.floor(uv / max(float(cell_size), 1e-6)).astype(np.int64)
    cell_to_nodes = {}
    for local_idx, cell in enumerate(cells):
        key = (int(cell[0]), int(cell[1]))
        cell_to_nodes.setdefault(key, []).append(int(nodes[local_idx]))
    adjacency = {key: [] for key in cell_to_nodes}
    for key in list(cell_to_nodes):
        x, y = key
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                other = (x + dx, y + dy)
                if other in cell_to_nodes:
                    adjacency[key].append(other)
    seen = set()
    components = []
    for start in cell_to_nodes:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp_nodes = []
        while stack:
            cur = stack.pop()
            comp_nodes.extend(cell_to_nodes[cur])
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(np.asarray(comp_nodes, dtype=np.int64))
    components.sort(key=len, reverse=True)
    return components


def refine_assignments_with_surface_components(hard_np, dists_np, normals_np, case, active, args):
    if not args.component_refine:
        return hard_np
    smooth_i = case.get("smooth_i", np.zeros((0,), dtype=np.int64)).astype(np.int64)
    smooth_j = case.get("smooth_j", np.zeros((0,), dtype=np.int64)).astype(np.int64)
    points_norm = case["points_norm"].astype(np.float32)
    refined = hard_np.copy()
    active_ids = [i for i, is_active in enumerate(active) if is_active]
    for plane_id in active_ids:
        nodes = np.flatnonzero(refined == plane_id)
        if len(nodes) < args.component_min_points:
            refined[nodes] = -1
            continue
        if args.component_method == "grid":
            u, v = _plane_basis(normals_np[plane_id])
            points_uv = np.stack([points_norm @ u, points_norm @ v], axis=1)
            comps = _connected_components_from_grid(points_uv, nodes, args.component_grid_size)
        else:
            if len(smooth_i) == 0:
                continue
            comps = _connected_components_from_edges(nodes, smooth_i, smooth_j)
        if not comps:
            continue
        largest = len(comps[0])
        keep = np.zeros(len(refined), dtype=bool)
        for comp in comps:
            large_enough = len(comp) >= args.component_min_points
            relative_enough = len(comp) >= args.component_keep_ratio * largest
            if large_enough and relative_enough:
                keep[comp] = True
        drop = nodes[~keep[nodes]]
        refined[drop] = -1

    if args.component_reassign and np.any(refined < 0) and active_ids:
        active_ids_np = np.asarray(active_ids, dtype=np.int64)
        dropped = np.flatnonzero(refined < 0)
        if len(dropped) > 0:
            candidate_dists = dists_np[dropped][:, active_ids_np]
            best_local = candidate_dists.argmin(axis=1)
            best_planes = active_ids_np[best_local]
            best_dists = candidate_dists[np.arange(len(dropped)), best_local]
            good = best_dists <= args.component_reassign_max_residual
            refined[dropped[good]] = best_planes[good]
    return refined


def fill_background_with_surface_support(hard_np, dists_np, case, active, args):
    if not args.surface_fill:
        return hard_np
    smooth_i = case.get("smooth_i", np.zeros((0,), dtype=np.int64)).astype(np.int64)
    smooth_j = case.get("smooth_j", np.zeros((0,), dtype=np.int64)).astype(np.int64)
    if len(smooth_i) == 0:
        return hard_np
    active_ids = [i for i, is_active in enumerate(active) if is_active]
    if not active_ids:
        return hard_np
    neighbors = [[] for _ in range(len(hard_np))]
    for a, b in zip(smooth_i, smooth_j):
        a = int(a)
        b = int(b)
        if 0 <= a < len(hard_np) and 0 <= b < len(hard_np):
            neighbors[a].append(b)
            neighbors[b].append(a)
    filled = hard_np.copy()
    active_ids_np = np.asarray(active_ids, dtype=np.int64)
    for _ in range(max(1, int(args.surface_fill_iters))):
        updates = []
        background_nodes = np.flatnonzero(filled < 0)
        for idx in background_nodes:
            nbr = neighbors[int(idx)]
            if len(nbr) < args.surface_fill_min_neighbors:
                continue
            nbr_labels = filled[np.asarray(nbr, dtype=np.int64)]
            nbr_labels = nbr_labels[nbr_labels >= 0]
            if len(nbr_labels) < args.surface_fill_min_neighbors:
                continue
            counts = np.bincount(nbr_labels, minlength=args.num_planes)
            plane_id = int(np.argmax(counts))
            support = int(counts[plane_id])
            if plane_id not in active_ids:
                continue
            if support / max(len(nbr), 1) < args.surface_fill_neighbor_ratio:
                continue
            if float(dists_np[idx, plane_id]) > args.surface_fill_max_residual:
                continue
            best_plane = int(active_ids_np[np.argmin(dists_np[idx, active_ids_np])])
            if best_plane != plane_id and float(dists_np[idx, best_plane]) + args.surface_fill_residual_margin < float(
                dists_np[idx, plane_id]
            ):
                continue
            updates.append((int(idx), plane_id))
        if not updates:
            break
        for idx, plane_id in updates:
            filled[idx] = plane_id
    return filled


def compute_one_loss(
    model,
    points,
    features,
    smooth_i,
    smooth_j,
    smooth_w,
    aux_normals,
    aux_offsets,
    aux_point_labels,
    args,
    temperature,
):
    n_points_before_sample = points.shape[0]
    if args.train_points_per_sample > 0 and n_points_before_sample > args.train_points_per_sample:
        sample_idx = torch.randperm(points.shape[0], device=points.device)[: args.train_points_per_sample]
        remap = torch.full((n_points_before_sample,), -1, dtype=torch.long, device=points.device)
        remap[sample_idx] = torch.arange(sample_idx.numel(), device=points.device)
        if smooth_i.numel() > 0:
            keep_pairs = (remap[smooth_i] >= 0) & (remap[smooth_j] >= 0)
            smooth_i = remap[smooth_i[keep_pairs]]
            smooth_j = remap[smooth_j[keep_pairs]]
            smooth_w = smooth_w[keep_pairs]
        points = points[sample_idx]
        features = features[sample_idx]
        if aux_point_labels.numel() == n_points_before_sample:
            aux_point_labels = aux_point_labels[sample_idx]
    normals, offsets, dists, assign = model.forward_one(
        points,
        features,
        temperature,
        args.distance_logit_weight,
        teacher_normals=aux_normals,
        teacher_offsets=aux_offsets,
        teacher_param_blend=args.teacher_param_blend,
    )
    plane_assign = assign[:, :-1]
    foreground = plane_assign.sum(dim=-1).clamp_min(1e-6)
    soft_residual = (plane_assign * dists).sum(dim=-1) / foreground
    soft_fit = trimmed_mean(soft_residual, args.trimmed_fit_ratio)
    hard_fit = dists.min(dim=-1).values.mean()
    ent = entropy(assign).mean()
    div = diversity_loss(normals, offsets, args.diversity_normal_margin, args.diversity_offset_margin)
    cov_loss, coverage = coverage_loss(plane_assign, args.min_coverage)
    dead_loss, _ = dead_token_loss(plane_assign, args.dead_token_min_coverage)
    sep_loss = confidence_separation_loss(plane_assign, args.assignment_margin)
    smooth_loss = local_smoothness_loss(plane_assign, smooth_i, smooth_j, smooth_w)
    compact_loss = region_compactness_loss(points, plane_assign)
    aux_param_loss, aux_matches = match_predicted_to_aux_planes(normals, offsets, aux_normals, aux_offsets)
    aux_assign_loss = auxiliary_assignment_loss(assign, aux_point_labels, aux_matches)
    aux_bg_loss = auxiliary_background_loss(assign, aux_point_labels, args.num_planes)
    teacher_active_count = min(int(aux_normals.shape[0]), args.num_planes)
    teacher_param_loss = ordered_teacher_param_loss(normals, offsets, aux_normals, aux_offsets)
    teacher_assign_loss = ordered_teacher_assignment_loss(
        assign,
        aux_point_labels,
        args.num_planes,
        class_balanced=args.class_balanced_teacher,
    )
    teacher_boundary_loss = teacher_boundary_contrast_loss(
        plane_assign,
        aux_point_labels,
        smooth_i,
        smooth_j,
        max_pairs=args.teacher_boundary_pairs,
        margin=args.teacher_boundary_margin,
    )
    inactive_loss = inactive_query_loss(plane_assign, teacher_active_count)
    confidence = 1.0 - entropy(assign) / np.log(args.num_planes + 1)
    confident_fit = (confidence.detach() * dists.min(dim=-1).values).mean()
    background_ratio = assign[:, -1].mean()
    foreground_loss = F.relu(args.min_foreground_coverage - plane_assign.sum(dim=1).mean())
    loss = (
        args.fit_weight * soft_fit
        + args.hard_fit_weight * hard_fit
        + args.entropy_weight * ent
        + args.diversity_weight * div
        + args.coverage_weight * cov_loss
        + args.dead_token_weight * dead_loss
        + args.assignment_margin_weight * sep_loss
        + args.smooth_weight * smooth_loss
        + args.compactness_weight * compact_loss
        + args.aux_plane_weight * aux_param_loss
        + args.aux_assignment_weight * aux_assign_loss
        + args.aux_background_weight * aux_bg_loss
        + args.teacher_param_weight * teacher_param_loss
        + args.teacher_assignment_weight * teacher_assign_loss
        + args.teacher_boundary_weight * teacher_boundary_loss
        + args.inactive_query_weight * inactive_loss
        + args.confident_fit_weight * confident_fit
        + args.foreground_weight * foreground_loss
    )
    stats = {
        "loss": loss,
        "soft_fit": soft_fit.detach(),
        "hard_fit": hard_fit.detach(),
        "entropy": ent.detach(),
        "diversity": div.detach(),
        "coverage_loss": cov_loss.detach(),
        "dead_token_loss": dead_loss.detach(),
        "smooth_loss": smooth_loss.detach(),
        "compactness_loss": compact_loss.detach(),
        "aux_param_loss": aux_param_loss.detach(),
        "aux_assignment_loss": aux_assign_loss.detach(),
        "aux_background_loss": aux_bg_loss.detach(),
        "teacher_param_loss": teacher_param_loss.detach(),
        "teacher_assignment_loss": teacher_assign_loss.detach(),
        "teacher_boundary_loss": teacher_boundary_loss.detach(),
        "inactive_query_loss": inactive_loss.detach(),
        "confidence": confidence.mean().detach(),
        "background_ratio": background_ratio.detach(),
        "foreground_loss": foreground_loss.detach(),
        "coverage": coverage.detach(),
    }
    return loss, stats


def export_case(model, case, output_dir, args, device):
    points = torch.from_numpy(case["points_norm"]).to(device)
    features = torch.from_numpy(case["features"]).to(device)
    aux_normals = torch.from_numpy(case.get("aux_plane_normals", np.zeros((0, 3), dtype=np.float32))).to(device)
    aux_offsets = torch.from_numpy(case.get("aux_plane_offsets_norm", np.zeros((0,), dtype=np.float32))).to(device)
    with torch.no_grad():
        normals, offsets_norm, dists, assign = model.forward_one(
            points,
            features,
            args.min_temperature,
            args.distance_logit_weight,
            teacher_normals=aux_normals,
            teacher_offsets=aux_offsets,
            teacher_param_blend=args.teacher_param_blend,
        )
        plane_assign = assign[:, :-1]
        hard = assign.argmax(dim=-1)
    normals_np = normals.detach().cpu().numpy()
    offsets_norm_np = offsets_norm.detach().cpu().numpy()
    hard_np = hard.detach().cpu().numpy().astype(np.int32)
    hard_np[hard_np >= args.num_planes] = -1
    plane_assign_np = plane_assign.detach().cpu().numpy()
    offsets_world = []
    for n, d_norm in zip(normals_np, offsets_norm_np):
        offsets_world.append(float(d_norm * case["scale"] - float(np.dot(n, case["center"]))))

    dists_np = dists.detach().cpu().numpy()
    params = []
    active = []
    for i in range(args.num_planes):
        mask = hard_np == i
        mean_residual = float(dists_np[mask, i].mean()) if mask.any() else None
        assigned_ratio = float(mask.mean())
        is_active = bool(
            assigned_ratio >= args.export_min_coverage
            and mean_residual is not None
            and mean_residual <= args.export_max_mean_residual
        )
        active.append(is_active)
        params.append(
            {
                "id": int(i),
                "normal": [float(x) for x in normals_np[i]],
                "offset": offsets_world[i],
                "offset_normalized": float(offsets_norm_np[i]),
                "assigned_point_count": int(mask.sum()),
                "assigned_ratio": assigned_ratio,
                "mean_abs_distance_normalized": mean_residual,
                "active": is_active,
            }
        )
    inactive = np.asarray([not x for x in active], dtype=bool)
    if inactive.any():
        inactive_ids = np.flatnonzero(inactive)
        hard_np[np.isin(hard_np, inactive_ids)] = -1
    before_refine_background = int(np.sum(hard_np < 0))
    hard_np = refine_assignments_with_surface_components(hard_np, dists_np, normals_np, case, active, args)
    after_refine_background = int(np.sum(hard_np < 0))
    hard_np = fill_background_with_surface_support(hard_np, dists_np, case, active, args)
    after_surface_fill_background = int(np.sum(hard_np < 0))
    learned_colors = np.zeros_like(case["colors"], dtype=np.uint8)
    learned_colors[hard_np < 0] = np.asarray([160, 160, 160], dtype=np.uint8)
    valid_hard = hard_np >= 0
    learned_colors[valid_hard] = PLANE_COLORS[hard_np[valid_hard] % len(PLANE_COLORS)]
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{case['stem']}_amortized_plane_tokens.json"
    npz_path = output_dir / f"{case['stem']}_amortized_plane_tokens_assignment.npz"
    summary = {
        "input_npz": case["path"],
        "num_points_used": int(len(case["points"])),
        "num_planes": int(args.num_planes),
        "active_planes": int(sum(active)),
        "background_point_count": int(np.sum(hard_np < 0)),
        "background_ratio": float(np.mean(hard_np < 0)),
        "component_refine": bool(args.component_refine),
        "background_before_component_refine": before_refine_background,
        "background_after_component_refine": after_refine_background,
        "surface_fill": bool(args.surface_fill),
        "background_after_surface_fill": after_surface_fill_background,
        "method": "amortized_plane_token_head",
        "center": [float(x) for x in case["center"]],
        "scale": float(case["scale"]),
        "planes": params,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        points=case["points"].astype(np.float32),
        colors=learned_colors.astype(np.uint8),
        original_colors=case["colors"].astype(np.uint8),
        assignment=hard_np,
        soft_assignment=plane_assign_np.astype(np.float32),
        plane_normals=normals_np.astype(np.float32),
        plane_offsets=np.asarray(offsets_world, dtype=np.float32),
        plane_offsets_normalized=offsets_norm_np.astype(np.float32),
        active_planes=np.asarray(active, dtype=np.bool_),
    )
    return json_path, npz_path, summary


def main():
    parser = argparse.ArgumentParser("Amortized unsupervised plane-token prediction")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_checkpoint", default=None)
    parser.add_argument("--load_checkpoint", default=None)
    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--pattern", default="*_full_pointcloud_editable_planes_data.npz")
    parser.add_argument("--sample_glob", default=None)
    parser.add_argument("--num_planes", type=int, default=4)
    parser.add_argument("--max_points_per_sample", type=int, default=30000)
    parser.add_argument(
        "--train_points_per_sample",
        type=int,
        default=6000,
        help="Random points used in each training step; export still uses max_points_per_sample.",
    )
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--sample_batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--context_dim", type=int, default=256)
    parser.add_argument("--distance_logit_weight", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--temperature_decay", type=float, default=0.999)
    parser.add_argument("--min_temperature", type=float, default=0.012)
    parser.add_argument("--fit_weight", type=float, default=1.0)
    parser.add_argument("--trimmed_fit_ratio", type=float, default=0.8)
    parser.add_argument("--hard_fit_weight", type=float, default=0.2)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--diversity_weight", type=float, default=0.04)
    parser.add_argument("--coverage_weight", type=float, default=0.05)
    parser.add_argument("--dead_token_weight", type=float, default=0.8)
    parser.add_argument("--dead_token_min_coverage", type=float, default=0.02)
    parser.add_argument("--assignment_margin_weight", type=float, default=0.03)
    parser.add_argument("--confident_fit_weight", type=float, default=0.05)
    parser.add_argument("--foreground_weight", type=float, default=0.05)
    parser.add_argument("--min_foreground_coverage", type=float, default=0.25)
    parser.add_argument(
        "--aux_plane_weight",
        type=float,
        default=0.0,
        help="Weakly match predicted planes to high-confidence candidate plane equations.",
    )
    parser.add_argument(
        "--aux_assignment_weight",
        type=float,
        default=0.0,
        help="Weakly align point assignments to candidate plane labels after plane matching.",
    )
    parser.add_argument(
        "--aux_background_weight",
        type=float,
        default=0.0,
        help="Weakly send points without candidate-plane labels to the background/no-plane class.",
    )
    parser.add_argument(
        "--teacher_param_weight",
        type=float,
        default=0.0,
        help="Directly supervise the first teacher planes with ordered Stage1 teacher parameters.",
    )
    parser.add_argument(
        "--teacher_param_blend",
        type=float,
        default=0.0,
        help="Blend Stage1 teacher plane params into assignment/export; 1.0 uses teacher params directly.",
    )
    parser.add_argument(
        "--teacher_assignment_weight",
        type=float,
        default=0.0,
        help="Directly supervise point labels to ordered Stage1 teacher query ids.",
    )
    parser.add_argument(
        "--class_balanced_teacher",
        action="store_true",
        help="Weight teacher assignment CE by inverse plane size so small planes are not ignored.",
    )
    parser.add_argument(
        "--teacher_boundary_weight",
        type=float,
        default=0.0,
        help="Discourage nearby points with different teacher plane ids from sharing one token.",
    )
    parser.add_argument("--teacher_boundary_pairs", type=int, default=20000)
    parser.add_argument("--teacher_boundary_margin", type=float, default=0.20)
    parser.add_argument(
        "--inactive_query_weight",
        type=float,
        default=0.0,
        help="Suppress query assignment probability beyond the number of Stage1 teacher planes.",
    )
    parser.add_argument("--smooth_weight", type=float, default=0.0)
    parser.add_argument(
        "--compactness_weight",
        type=float,
        default=0.0,
        help="Penalize one token covering spatially separated point islands.",
    )
    parser.add_argument("--smooth_pairs_per_sample", type=int, default=0)
    parser.add_argument("--smooth_candidates", type=int, default=24)
    parser.add_argument("--smooth_xyz_sigma", type=float, default=0.06)
    parser.add_argument("--smooth_rgb_sigma", type=float, default=0.25)
    parser.add_argument("--min_coverage", type=float, default=0.02)
    parser.add_argument("--assignment_margin", type=float, default=0.12)
    parser.add_argument("--diversity_normal_margin", type=float, default=0.18)
    parser.add_argument("--diversity_offset_margin", type=float, default=0.04)
    parser.add_argument("--export_min_coverage", type=float, default=0.04)
    parser.add_argument("--export_max_mean_residual", type=float, default=0.08)
    parser.add_argument(
        "--component_refine",
        action="store_true",
        help="At export time, keep connected surface components for each plane and suppress small islands.",
    )
    parser.add_argument("--component_min_points", type=int, default=160)
    parser.add_argument("--component_keep_ratio", type=float, default=0.08)
    parser.add_argument("--component_method", choices=["graph", "grid"], default="graph")
    parser.add_argument("--component_grid_size", type=float, default=0.035)
    parser.add_argument("--component_reassign", action="store_true")
    parser.add_argument("--component_reassign_max_residual", type=float, default=0.06)
    parser.add_argument(
        "--surface_fill",
        action="store_true",
        help="Fill background holes when local neighbors agree on a plane and the point is close to that plane.",
    )
    parser.add_argument("--surface_fill_iters", type=int, default=2)
    parser.add_argument("--surface_fill_min_neighbors", type=int, default=4)
    parser.add_argument("--surface_fill_neighbor_ratio", type=float, default=0.35)
    parser.add_argument("--surface_fill_max_residual", type=float, default=0.045)
    parser.add_argument("--surface_fill_residual_margin", type=float, default=0.01)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    paths = sorted(Path(args.input_dir).glob(args.pattern))
    if args.sample_glob:
        paths = [p for p in paths if args.sample_glob in p.name]
    if not paths:
        raise FileNotFoundError(f"No npz files matched under {args.input_dir}")
    cases = [sample_case_with_aux_planes(p, args, args.seed + i * 97) for i, p in enumerate(paths)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = [
        (
            torch.from_numpy(case["points_norm"]).to(device),
            torch.from_numpy(case["features"]).to(device),
            torch.from_numpy(case["smooth_i"]).to(device),
            torch.from_numpy(case["smooth_j"]).to(device),
            torch.from_numpy(case["smooth_w"]).to(device),
            torch.from_numpy(case.get("aux_plane_normals", np.zeros((0, 3), dtype=np.float32))).to(device),
            torch.from_numpy(case.get("aux_plane_offsets_norm", np.zeros((0,), dtype=np.float32))).to(device),
            torch.from_numpy(case.get("aux_point_labels", np.zeros((0,), dtype=np.int64))).to(device),
        )
        for case in cases
    ]
    model = AmortizedPlaneTokenHead(
        args.num_planes,
        point_feature_dim=tensors[0][1].shape[1],
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
    ).to(device)
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state, strict=True)
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
            points, features, smooth_i, smooth_j, smooth_w, aux_normals, aux_offsets, aux_point_labels = tensors[int(sid)]
            loss, stats = compute_one_loss(
                model,
                points,
                features,
                smooth_i,
                smooth_j,
                smooth_w,
                aux_normals,
                aux_offsets,
                aux_point_labels,
                args,
                temperature,
            )
            losses.append(loss)
            stat_rows.append(stats)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "soft_fit": float(torch.stack([s["soft_fit"] for s in stat_rows]).mean().cpu()),
            "hard_fit": float(torch.stack([s["hard_fit"] for s in stat_rows]).mean().cpu()),
            "entropy": float(torch.stack([s["entropy"] for s in stat_rows]).mean().cpu()),
            "confidence": float(torch.stack([s["confidence"] for s in stat_rows]).mean().cpu()),
            "dead_token_loss": float(torch.stack([s["dead_token_loss"] for s in stat_rows]).mean().cpu()),
            "smooth_loss": float(torch.stack([s["smooth_loss"] for s in stat_rows]).mean().cpu()),
            "compactness_loss": float(torch.stack([s["compactness_loss"] for s in stat_rows]).mean().cpu()),
            "aux_param_loss": float(torch.stack([s["aux_param_loss"] for s in stat_rows]).mean().cpu()),
            "aux_assignment_loss": float(torch.stack([s["aux_assignment_loss"] for s in stat_rows]).mean().cpu()),
            "aux_background_loss": float(torch.stack([s["aux_background_loss"] for s in stat_rows]).mean().cpu()),
            "teacher_param_loss": float(torch.stack([s["teacher_param_loss"] for s in stat_rows]).mean().cpu()),
            "teacher_assignment_loss": float(torch.stack([s["teacher_assignment_loss"] for s in stat_rows]).mean().cpu()),
            "teacher_boundary_loss": float(torch.stack([s["teacher_boundary_loss"] for s in stat_rows]).mean().cpu()),
            "inactive_query_loss": float(torch.stack([s["inactive_query_loss"] for s in stat_rows]).mean().cpu()),
            "background_ratio": float(torch.stack([s["background_ratio"] for s in stat_rows]).mean().cpu()),
            "foreground_loss": float(torch.stack([s["foreground_loss"] for s in stat_rows]).mean().cpu()),
            "temperature": float(temperature),
        }
        history.append(row)
        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:04d} loss={row['loss']:.5f} fit={row['soft_fit']:.5f} "
                f"hard={row['hard_fit']:.5f} ent={row['entropy']:.4f} "
                f"dead={row['dead_token_loss']:.4f} smooth={row['smooth_loss']:.4f} "
                f"compact={row['compactness_loss']:.4f} "
                f"aux={row['aux_param_loss']:.4f}/{row['aux_assignment_loss']:.4f}/{row['aux_background_loss']:.4f} "
                f"teacher={row['teacher_param_loss']:.4f}/{row['teacher_assignment_loss']:.4f} "
                f"boundary={row['teacher_boundary_loss']:.4f} "
                f"inactive={row['inactive_query_loss']:.4f} "
                f"bg={row['background_ratio']:.3f} conf={row['confidence']:.3f}"
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
        "max_points_per_sample": args.max_points_per_sample,
        "train_points_per_sample": args.train_points_per_sample,
        "method": "amortized_plane_token_head",
        "aux_plane_weight": args.aux_plane_weight,
        "aux_assignment_weight": args.aux_assignment_weight,
        "aux_background_weight": args.aux_background_weight,
        "teacher_param_weight": args.teacher_param_weight,
        "teacher_param_blend": args.teacher_param_blend,
        "teacher_assignment_weight": args.teacher_assignment_weight,
        "class_balanced_teacher": args.class_balanced_teacher,
        "teacher_boundary_weight": args.teacher_boundary_weight,
        "teacher_boundary_margin": args.teacher_boundary_margin,
        "inactive_query_weight": args.inactive_query_weight,
        "history": history,
        "exported": exported,
    }
    overview_path = output_dir / "amortized_plane_tokens_summary.json"
    overview_path.write_text(json.dumps(overview, indent=2), encoding="utf-8")
    if args.save_checkpoint:
        checkpoint_path = Path(args.save_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "history": history,
                "overview": overview,
                "num_samples": len(cases),
                "point_feature_dim": int(tensors[0][1].shape[1]),
            },
            checkpoint_path,
        )
        print(checkpoint_path)
    print(overview_path)
    print(f"samples={len(cases)}")


if __name__ == "__main__":
    main()
