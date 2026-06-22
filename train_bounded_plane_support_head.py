import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_amortized_plane_tokens import PLANE_COLORS, sample_case_with_aux_planes
from train_patch_plane_tokens import build_surface_patches, refit_plane_from_points


class BoundedPlaneSupportHead(nn.Module):
    def __init__(self, patch_feature_dim, hidden_dim=192):
        super().__init__()
        pair_dim = patch_feature_dim + 3 + 1 + 1 + 1
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.fit_confidence_net = nn.Sequential(
            nn.Linear(patch_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.background_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, patch_features, patch_centroids, patch_normals, plane_normals, plane_offsets, support_prior=None):
        n_patches = patch_features.shape[0]
        n_planes = plane_normals.shape[0]
        dists = torch.abs(patch_centroids @ plane_normals.t() + plane_offsets.view(1, -1))
        normal_align = torch.abs(patch_normals @ plane_normals.t()).clamp(0.0, 1.0)
        patch_feat = patch_features[:, None, :].expand(n_patches, n_planes, -1)
        normal_feat = plane_normals[None, :, :].expand(n_patches, n_planes, -1)
        offset_feat = plane_offsets.view(1, n_planes, 1).expand(n_patches, n_planes, 1)
        pair_feat = torch.cat([patch_feat, normal_feat, offset_feat, dists.unsqueeze(-1), normal_align.unsqueeze(-1)], dim=-1)
        logits = self.net(pair_feat).squeeze(-1)
        if support_prior is not None:
            logits = logits + support_prior[:, :n_planes]
        patch_fit_confidence = torch.sigmoid(
            self.fit_confidence_net(patch_features).squeeze(-1)
        )
        fit_confidence = patch_fit_confidence[:, None].expand(n_patches, n_planes)
        logits = torch.cat([logits, self.background_logit.expand(n_patches, 1)], dim=1)
        return F.softmax(logits, dim=-1), dists, fit_confidence


def differentiable_weighted_plane_fit(
    points,
    weights,
    patch_counts=None,
    eps=1e-6,
    cov_jitter=1e-6,
):
    """Fit K planes from soft patch support using differentiable weighted PCA."""
    weights = weights.clamp_min(0.0)
    if patch_counts is not None:
        weights = weights * patch_counts[:, None].to(dtype=weights.dtype)

    mass = weights.sum(dim=0).clamp_min(eps)
    centers = (weights.transpose(0, 1) @ points) / mass[:, None]
    centered = points[:, None, :] - centers[None, :, :]
    covariance = torch.einsum(
        "nk,nki,nkj->kij",
        weights,
        centered,
        centered,
    ) / mass[:, None, None]
    eye = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0)
    covariance = covariance + float(cov_jitter) * eye
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    normals = F.normalize(eigvecs[:, :, 0], dim=-1)
    offsets = -(normals * centers).sum(dim=-1)
    return normals, offsets, centers, eigvals, mass


def edge_smoothness_loss(assign, edge_i, edge_j, labels, edge_boundary_conf, line_smooth_suppress=0.6, margin=0.1):
    plane_assign = assign[:, :-1]
    if edge_i.numel() == 0:
        return assign.sum() * 0.0, assign.sum() * 0.0
    same = (labels[edge_i] >= 0) & (labels[edge_i] == labels[edge_j])
    diff = (labels[edge_i] >= 0) & (labels[edge_j] >= 0) & (labels[edge_i] != labels[edge_j])
    same_loss = assign.sum() * 0.0
    boundary_loss = assign.sum() * 0.0
    conf = edge_boundary_conf.to(dtype=plane_assign.dtype, device=plane_assign.device).clamp(0.0, 1.0)
    if torch.any(same):
        raw = (plane_assign[edge_i[same]] - plane_assign[edge_j[same]]).abs().mean(dim=1)
        w = (1.0 - float(line_smooth_suppress) * conf[same]).clamp_min(0.05)
        same_loss = (raw * w).sum() / w.sum().clamp_min(1e-6)
    if torch.any(diff):
        same_prob = (plane_assign[edge_i[diff]] * plane_assign[edge_j[diff]]).sum(dim=1)
        raw = F.relu(same_prob - float(margin))
        w = 1.0 + conf[diff]
        boundary_loss = (raw * w).sum() / w.sum().clamp_min(1e-6)
    return same_loss, boundary_loss


def hard_boundary_pair_loss(assign, edge_i, edge_j, labels, edge_boundary_conf, min_edge_conf=0.25):
    plane_assign = assign[:, :-1]
    if edge_i.numel() == 0:
        return assign.sum() * 0.0
    conf = edge_boundary_conf.to(dtype=plane_assign.dtype, device=plane_assign.device).clamp(0.0, 1.0)
    diff = (
        (labels[edge_i] >= 0)
        & (labels[edge_j] >= 0)
        & (labels[edge_i] != labels[edge_j])
        & (conf >= float(min_edge_conf))
    )
    if not torch.any(diff):
        return assign.sum() * 0.0
    same_plane_prob = (plane_assign[edge_i[diff]] * plane_assign[edge_j[diff]]).sum(dim=1).clamp(0.0, 1.0 - 1e-6)
    barrier = -torch.log1p(-same_plane_prob)
    weights = conf[diff].clamp_min(float(min_edge_conf))
    return (barrier * weights).sum() / weights.sum().clamp_min(1e-6)


def add_boundary_patch_weights(case, args):
    labels = case["patch_labels"].astype(np.int64)
    weights = np.ones(len(labels), dtype=np.float32)
    edge_i = case["patch_edge_i"].astype(np.int64)
    edge_j = case["patch_edge_j"].astype(np.int64)
    edge_conf = case["patch_edge_boundary_conf"].astype(np.float32)
    boundary_neighbor = case["patch_boundary_neighbor"].astype(np.float32)
    near_boundary = boundary_neighbor.max(axis=1) > 0.5
    weights[near_boundary] += float(args.boundary_error_weight) * 0.5
    for a, b, conf in zip(edge_i, edge_j, edge_conf):
        la = int(labels[a])
        lb = int(labels[b])
        if la < 0 or lb < 0 or la == lb:
            continue
        strength = max(float(conf), float(args.boundary_error_min_edge_conf))
        weights[a] += float(args.boundary_error_weight) * strength
        weights[b] += float(args.boundary_error_weight) * strength
    case["patch_boundary_ce_weight"] = weights.astype(np.float32)
    return case


def boundary_weighted_patch_ce(assign, labels, num_planes, label_conf, patch_weights):
    valid = (labels >= 0) & (labels < num_planes)
    if not torch.any(valid):
        return assign.sum() * 0.0
    labels_v = labels[valid]
    log_probs = torch.log(assign[valid].clamp_min(1e-8))
    counts = torch.bincount(labels_v, minlength=num_planes).float()
    class_weights = torch.zeros_like(counts)
    present = counts > 0
    class_weights[present] = 1.0 / counts[present].clamp_min(1.0)
    class_weights = class_weights * (present.float().sum() / class_weights[present].sum().clamp_min(1e-6))
    weights = class_weights[labels_v] * label_conf[valid].clamp_min(0.2) * patch_weights[valid].clamp_min(1.0)
    per_patch = F.nll_loss(log_probs, labels_v, reduction="none")
    return (per_patch * weights).sum() / weights.sum().clamp_min(1e-6)


def support_metrics(assign, labels, num_planes):
    pred = assign.argmax(dim=-1)
    pred = torch.where(pred >= num_planes, torch.full_like(pred, -1), pred)
    valid = labels >= 0
    if not torch.any(valid):
        return {"patch_acc": 0.0, "fg_acc": 0.0}
    patch_acc = (pred[valid] == labels[valid]).float().mean()
    return {"patch_acc": float(patch_acc.detach().cpu()), "fg_acc": float(patch_acc.detach().cpu())}


def compute_loss(model, tensors, args):
    (
        patch_centroids,
        patch_normals,
        patch_features,
        patch_labels,
        patch_label_conf,
        patch_boundary_ce_weight,
        edge_i,
        edge_j,
        edge_boundary_conf,
        aux_normals,
        aux_offsets,
        support_prior,
        patch_counts,
    ) = tensors
    assign, dists, fit_confidence = model(
        patch_features,
        patch_centroids,
        patch_normals,
        aux_normals,
        aux_offsets,
        support_prior,
    )
    teacher_loss = boundary_weighted_patch_ce(
        assign,
        patch_labels,
        aux_normals.shape[0],
        patch_label_conf,
        patch_boundary_ce_weight,
    )
    same_loss, boundary_loss = edge_smoothness_loss(
        assign,
        edge_i,
        edge_j,
        patch_labels,
        edge_boundary_conf,
        line_smooth_suppress=args.line_smooth_suppress,
        margin=args.boundary_margin,
    )
    hard_boundary_loss = hard_boundary_pair_loss(
        assign,
        edge_i,
        edge_j,
        patch_labels,
        edge_boundary_conf,
        min_edge_conf=args.hard_boundary_min_edge_conf,
    )
    plane_assign = assign[:, :-1]
    fg = plane_assign.sum(dim=-1).clamp_min(1e-6)
    residual_loss = ((plane_assign * dists).sum(dim=-1) / fg).mean()
    valid_label = (patch_labels >= 0) & (patch_labels < aux_normals.shape[0])
    if torch.any(valid_label):
        valid_ids = torch.nonzero(valid_label, as_tuple=False).flatten()
        own_ids = patch_labels[valid_label]
        own_distance = dists[valid_ids, own_ids]
        own_alignment = torch.abs(
            (patch_normals[valid_label] * aux_normals[own_ids]).sum(dim=-1)
        ).clamp(0.0, 1.0)
        confidence_target = torch.exp(
            -torch.square(own_distance / max(float(args.fit_confidence_distance_scale), 1e-6))
        ) * own_alignment.pow(float(args.fit_confidence_normal_power))
        confidence_prediction = fit_confidence[valid_ids, own_ids]
        fit_confidence_loss = F.binary_cross_entropy(
            confidence_prediction.clamp(1e-6, 1.0 - 1e-6),
            confidence_target.detach(),
            weight=patch_label_conf[valid_label].clamp_min(0.2),
        )
        mean_fit_confidence_target = confidence_target.mean()
    else:
        fit_confidence_loss = assign.sum() * 0.0
        mean_fit_confidence_target = assign.sum().detach() * 0.0
    fit_weights = plane_assign * fit_confidence
    (
        normals_fit,
        offsets_fit,
        _,
        eigvals,
        plane_mass,
    ) = differentiable_weighted_plane_fit(
        patch_centroids,
        fit_weights,
        patch_counts=patch_counts if args.use_patch_count_weight else None,
        cov_jitter=args.cov_jitter,
    )
    fitted_residual = torch.abs(patch_centroids @ normals_fit.t() + offsets_fit[None, :])
    robust_residual = torch.sqrt(
        fitted_residual.square() + float(args.fit_charbonnier_eps) ** 2
    )
    valid_plane = plane_mass > float(args.min_plane_mass)
    if torch.any(valid_plane):
        valid_weights = fit_weights[:, valid_plane]
        diff_fit_loss = (
            valid_weights * robust_residual[:, valid_plane]
        ).sum() / valid_weights.sum().clamp_min(1e-6)
        mean_eig_min = eigvals[valid_plane, 0].mean()
        mean_eig_gap = (eigvals[valid_plane, 1] - eigvals[valid_plane, 0]).mean()
    else:
        diff_fit_loss = assign.sum() * 0.0
        mean_eig_min = assign.sum().detach() * 0.0
        mean_eig_gap = assign.sum().detach() * 0.0
    coverage_loss = F.relu(float(args.min_plane_mass) - plane_mass).mean() / max(
        float(args.min_plane_mass),
        1e-6,
    )
    loss = (
        args.teacher_weight * teacher_loss
        + args.smooth_weight * same_loss
        + args.boundary_weight * boundary_loss
        + args.hard_boundary_weight * hard_boundary_loss
        + args.residual_weight * residual_loss
        + args.fit_confidence_weight * fit_confidence_loss
        + args.diff_plane_fit_weight * diff_fit_loss
        + args.coverage_weight * coverage_loss
    )
    stats = {
        "loss": loss.detach(),
        "teacher_loss": teacher_loss.detach(),
        "smooth_loss": same_loss.detach(),
        "boundary_loss": boundary_loss.detach(),
        "hard_boundary_loss": hard_boundary_loss.detach(),
        "residual_loss": residual_loss.detach(),
        "fit_confidence_loss": fit_confidence_loss.detach(),
        "mean_fit_confidence": fit_confidence.mean().detach(),
        "mean_fit_confidence_target": mean_fit_confidence_target.detach(),
        "diff_fit_loss": diff_fit_loss.detach(),
        "coverage_loss": coverage_loss.detach(),
        "mean_plane_mass": plane_mass.mean().detach(),
        "mean_eig_min": mean_eig_min.detach(),
        "mean_eig_gap": mean_eig_gap.detach(),
        "background_ratio": assign[:, -1].mean().detach(),
    }
    return loss, stats


def export_case(model, case, output_dir, args, device):
    patch_centroids = torch.from_numpy(case["patch_centroids"]).to(device)
    patch_normals = torch.from_numpy(case["patch_normals"]).to(device)
    patch_features = torch.from_numpy(case["patch_features"]).to(device)
    aux_normals_np = case.get("aux_plane_normals", np.zeros((0, 3), dtype=np.float32)).astype(np.float32)
    aux_offsets_np = case.get("aux_plane_offsets_norm", np.zeros((0,), dtype=np.float32)).astype(np.float32)
    aux_normals = torch.from_numpy(aux_normals_np).to(device)
    aux_offsets = torch.from_numpy(aux_offsets_np).to(device)
    support_prior = torch.from_numpy(case["patch_support_prior"]).to(device)
    with torch.no_grad():
        assign, _, fit_confidence = model(
            patch_features,
            patch_centroids,
            patch_normals,
            aux_normals,
            aux_offsets,
            support_prior,
        )
    patch_assign = assign.argmax(dim=-1).detach().cpu().numpy().astype(np.int32)
    patch_assign[patch_assign >= len(aux_normals_np)] = -1
    teacher_patch_labels = case["patch_labels"].astype(np.int32)
    valid_teacher = teacher_patch_labels >= 0
    teacher_patch_acc = (
        float(np.mean(patch_assign[valid_teacher] == teacher_patch_labels[valid_teacher]))
        if np.any(valid_teacher)
        else 1.0
    )
    if args.boundary_strict_decode:
        patch_assign = boundary_strict_decode(
            patch_assign,
            teacher_patch_labels,
            case["patch_label_conf"].astype(np.float32),
            case["patch_edge_i"].astype(np.int64),
            case["patch_edge_j"].astype(np.int64),
            case["patch_edge_boundary_conf"].astype(np.float32),
            min_conf=args.boundary_strict_label_conf,
            min_edge_conf=args.boundary_strict_edge_conf,
        )
    used_teacher_fallback = False
    if (
        args.teacher_fallback_min_patch_acc > 0.0
        and teacher_patch_acc < args.teacher_fallback_min_patch_acc
    ):
        patch_assign = teacher_patch_labels.copy()
        used_teacher_fallback = True
    point_assignment = np.full(len(case["points"]), -1, dtype=np.int32)
    for patch_id, plane_id in enumerate(patch_assign):
        point_assignment[case["point_to_patch"] == patch_id] = int(plane_id)

    hard_fit_weights = torch.zeros_like(fit_confidence)
    valid_patch_ids = np.flatnonzero(patch_assign >= 0)
    if len(valid_patch_ids):
        valid_patch_ids_t = torch.from_numpy(valid_patch_ids).to(device=device, dtype=torch.long)
        valid_plane_ids_t = torch.from_numpy(patch_assign[valid_patch_ids]).to(
            device=device,
            dtype=torch.long,
        )
        hard_fit_weights[valid_patch_ids_t, valid_plane_ids_t] = fit_confidence[
            valid_patch_ids_t,
            valid_plane_ids_t,
        ]
    with torch.no_grad():
        fitted_normals_t, fitted_offsets_t, _, _, fitted_mass_t = (
            differentiable_weighted_plane_fit(
                patch_centroids,
                hard_fit_weights,
                patch_counts=torch.from_numpy(case["patch_counts"]).to(device),
                cov_jitter=args.cov_jitter,
            )
        )
    fitted_normals_np = fitted_normals_t.cpu().numpy().astype(np.float32)
    fitted_offsets_np = fitted_offsets_t.cpu().numpy().astype(np.float32)
    fitted_mass_np = fitted_mass_t.cpu().numpy().astype(np.float32)
    fit_confidence_np = fit_confidence.cpu().numpy().astype(np.float32)

    normals = []
    offsets_norm = []
    planes = []
    for plane_id in range(len(aux_normals_np)):
        mask = point_assignment == plane_id
        enough_support = int(mask.sum()) >= args.refit_min_points
        enough_fit_mass = fitted_mass_np[plane_id] >= args.refit_min_effective_mass
        if not (enough_support and enough_fit_mass):
            normals.append(aux_normals_np[plane_id])
            offsets_norm.append(aux_offsets_np[plane_id])
            active = False
            mean_res = None
        else:
            normal = fitted_normals_np[plane_id]
            offset = float(fitted_offsets_np[plane_id])
            if float(np.dot(normal, aux_normals_np[plane_id])) < 0.0:
                normal = -normal
                offset = -offset
            normals.append(normal)
            offsets_norm.append(float(offset))
            selected_points = case["points_norm"][mask].astype(np.float32)
            dist = np.abs((selected_points * normal[None, :]).sum(axis=1) + float(offset))
            mean_res = float(dist.mean())
            fit_conf_mean = (
                float(fit_confidence_np[patch_assign == plane_id, plane_id].mean())
                if np.any(patch_assign == plane_id)
                else None
            )
            active = True
            if args.max_active_mean_residual > 0.0 and mean_res > args.max_active_mean_residual:
                active = False
            if (
                args.min_active_fit_confidence > 0.0
                and fit_conf_mean is not None
                and fit_conf_mean < args.min_active_fit_confidence
            ):
                active = False
        planes.append(
            {
                "id": int(plane_id),
                "normal": [float(x) for x in normals[-1]],
                "offset_normalized": float(offsets_norm[-1]),
                "assigned_point_count": int(mask.sum()),
                "assigned_patch_count": int((patch_assign == plane_id).sum()),
                "assigned_ratio": float(mask.mean()),
                "mean_abs_distance_normalized": mean_res,
                "fit_confidence_mean": fit_conf_mean,
                "fit_effective_mass": float(fitted_mass_np[plane_id]),
                "active": bool(active),
            }
        )
    normals = np.stack(normals).astype(np.float32) if normals else np.zeros((0, 3), dtype=np.float32)
    offsets_norm = np.asarray(offsets_norm, dtype=np.float32)
    inactive_ids = np.asarray([p["id"] for p in planes if not p["active"]], dtype=np.int32)
    if len(inactive_ids):
        point_assignment[np.isin(point_assignment, inactive_ids)] = -1
        patch_assign[np.isin(patch_assign, inactive_ids)] = -1
    offsets_world = np.asarray(
        [float(d * case["scale"] - float(np.dot(n, case["center"]))) for n, d in zip(normals, offsets_norm)],
        dtype=np.float32,
    )
    colors = np.tile(np.asarray([160, 160, 160], dtype=np.uint8)[None, :], (len(case["points"]), 1))
    valid = point_assignment >= 0
    colors[valid] = PLANE_COLORS[point_assignment[valid] % len(PLANE_COLORS)]
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{case['stem']}_bounded_support_head.json"
    npz_path = output_dir / f"{case['stem']}_bounded_support_head_assignment.npz"
    summary = {
        "input_npz": case["path"],
        "method": "bounded_plane_support_head_refit",
        "num_points_used": int(len(case["points"])),
        "num_patches": int(len(case["patch_features"])),
        "num_planes": int(len(aux_normals_np)),
        "active_planes": int(sum(p["active"] for p in planes)),
        "teacher_patch_acc_before_fallback": float(teacher_patch_acc),
        "used_teacher_fallback": bool(used_teacher_fallback),
        "background_point_count": int((point_assignment < 0).sum()),
        "background_ratio": float((point_assignment < 0).mean()),
        "planes": planes,
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
        patch_labels=case["patch_labels"].astype(np.int32),
        patch_centroids=case["patch_centroids"].astype(np.float32),
        patch_normals=case["patch_normals"].astype(np.float32),
        patch_label_conf=case["patch_label_conf"].astype(np.float32),
        patch_boundary_ce_weight=case["patch_boundary_ce_weight"].astype(np.float32),
        patch_edge_i=case["patch_edge_i"].astype(np.int32),
        patch_edge_j=case["patch_edge_j"].astype(np.int32),
        patch_edge_boundary_conf=case["patch_edge_boundary_conf"].astype(np.float32),
        patch_boundary_neighbor=case["patch_boundary_neighbor"].astype(np.float32),
        patch_support_prior=case["patch_support_prior"].astype(np.float32),
        patch_fit_confidence=fit_confidence_np.astype(np.float32),
        patch_fit_weight=hard_fit_weights.cpu().numpy().astype(np.float32),
        plane_normals=normals.astype(np.float32),
        plane_offsets=offsets_world.astype(np.float32),
        plane_offsets_normalized=offsets_norm.astype(np.float32),
        active_planes=np.asarray([p["active"] for p in planes], dtype=np.bool_),
        gt_point_plane_ids=case.get(
            "gt_point_plane_ids",
            np.full(len(case["points"]), -1, dtype=np.int32),
        ).astype(np.int32),
    )
    return summary


def boundary_strict_decode(patch_assign, labels, label_conf, edge_i, edge_j, edge_boundary_conf, min_conf=0.65, min_edge_conf=0.25):
    decoded = patch_assign.copy()
    locked = (labels >= 0) & (label_conf >= float(min_conf))
    decoded[locked] = labels[locked]
    # If a high-confidence patch is separated from a neighbor by a confident
    # boundary edge, prevent the neighbor from taking that high-confidence side.
    for a, b, conf in zip(edge_i, edge_j, edge_boundary_conf):
        if float(conf) < float(min_edge_conf):
            continue
        la = int(labels[a])
        lb = int(labels[b])
        if la < 0 or lb < 0 or la == lb:
            continue
        if label_conf[a] >= min_conf and decoded[b] == la:
            decoded[b] = lb
        if label_conf[b] >= min_conf and decoded[a] == lb:
            decoded[a] = la
    return decoded.astype(np.int32)


def main():
    parser = argparse.ArgumentParser("Bounded plane support head")
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
    parser.add_argument("--edge_boundary_normal_gap", type=float, default=0.10)
    parser.add_argument("--support_logit_weight", type=float, default=0.15)
    parser.add_argument("--boundary_support_logit_weight", type=float, default=0.05)
    parser.add_argument("--outside_support_penalty", type=float, default=0.40)
    parser.add_argument("--support_grid_size", type=float, default=0.055)
    parser.add_argument("--support_dilate_cells", type=int, default=2)
    parser.add_argument("--support_max_distance_cells", type=int, default=10)
    parser.add_argument("--support_min_label_conf", type=float, default=0.60)
    parser.add_argument("--teacher_label_logit_weight", type=float, default=0.0)
    parser.add_argument(
        "--disable_support_prior",
        action="store_true",
        help="Zero the teacher-derived bounded-support prior before training and export.",
    )
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--teacher_weight", type=float, default=1.0)
    parser.add_argument("--smooth_weight", type=float, default=0.06)
    parser.add_argument("--boundary_weight", type=float, default=0.18)
    parser.add_argument("--residual_weight", type=float, default=0.04)
    parser.add_argument("--fit_confidence_weight", type=float, default=0.20)
    parser.add_argument("--fit_confidence_distance_scale", type=float, default=0.03)
    parser.add_argument("--fit_confidence_normal_power", type=float, default=2.0)
    parser.add_argument("--line_smooth_suppress", type=float, default=0.60)
    parser.add_argument("--boundary_margin", type=float, default=0.10)
    parser.add_argument("--boundary_error_weight", type=float, default=3.0)
    parser.add_argument("--boundary_error_min_edge_conf", type=float, default=0.25)
    parser.add_argument("--hard_boundary_weight", type=float, default=0.0)
    parser.add_argument("--hard_boundary_min_edge_conf", type=float, default=0.25)
    parser.add_argument("--diff_plane_fit_weight", type=float, default=0.0)
    parser.add_argument("--coverage_weight", type=float, default=0.0)
    parser.add_argument("--min_plane_mass", type=float, default=30.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-6)
    parser.add_argument("--fit_charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--use_patch_count_weight", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--refit_min_points", type=int, default=300)
    parser.add_argument("--refit_min_effective_mass", type=float, default=50.0)
    parser.add_argument("--max_active_mean_residual", type=float, default=0.0)
    parser.add_argument("--min_active_fit_confidence", type=float, default=0.0)
    parser.add_argument("--teacher_fallback_min_patch_acc", type=float, default=0.0)
    parser.add_argument("--boundary_strict_decode", action="store_true")
    parser.add_argument("--boundary_strict_label_conf", type=float, default=0.65)
    parser.add_argument("--boundary_strict_edge_conf", type=float, default=0.25)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    paths = sorted(Path(args.input_dir).glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No npz files matched under {args.input_dir}")
    cases = [
        add_boundary_patch_weights(
            build_surface_patches(sample_case_with_aux_planes(p, args, args.seed + i * 97), args),
            args,
        )
        for i, p in enumerate(paths)
    ]
    if args.disable_support_prior:
        for case in cases:
            case["patch_support_prior"].fill(0.0)
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
                torch.from_numpy(case["patch_boundary_ce_weight"]).to(device),
                torch.from_numpy(case["patch_edge_i"]).to(device),
                torch.from_numpy(case["patch_edge_j"]).to(device),
                torch.from_numpy(case["patch_edge_boundary_conf"]).to(device),
                torch.from_numpy(case.get("aux_plane_normals", np.zeros((0, 3), dtype=np.float32))).to(device),
                torch.from_numpy(case.get("aux_plane_offsets_norm", np.zeros((0,), dtype=np.float32))).to(device),
                torch.from_numpy(case["patch_support_prior"]).to(device),
                torch.from_numpy(case["patch_counts"]).to(device),
            )
        )
    model = BoundedPlaneSupportHead(tensors[0][2].shape[1], hidden_dim=args.hidden_dim).to(device)
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
        sample_ids = rng.choice(len(cases), size=min(args.sample_batch_size, len(cases)), replace=False)
        losses = []
        stat_rows = []
        metric_rows = []
        for sid in sample_ids:
            loss, stats = compute_loss(model, tensors[int(sid)], args)
            losses.append(loss)
            stat_rows.append(stats)
            with torch.no_grad():
                assign, _, _ = model(
                    tensors[int(sid)][2],
                    tensors[int(sid)][0],
                    tensors[int(sid)][1],
                    tensors[int(sid)][9],
                    tensors[int(sid)][10],
                    tensors[int(sid)][11],
                )
                metric_rows.append(support_metrics(assign, tensors[int(sid)][3], tensors[int(sid)][9].shape[0]))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        row = {"step": int(step), "loss": float(loss.detach().cpu())}
        for key in stat_rows[0]:
            row[key] = float(torch.stack([s[key] for s in stat_rows]).mean().cpu())
        row["patch_acc"] = float(np.mean([m["patch_acc"] for m in metric_rows]))
        history.append(row)
        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:04d} loss={row['loss']:.5f} teacher={row['teacher_loss']:.4f} "
                f"smooth={row['smooth_loss']:.4f} boundary={row['boundary_loss']:.4f} "
                f"hard={row['hard_boundary_loss']:.4f} "
                f"res={row['residual_loss']:.4f} conf={row['fit_confidence_loss']:.4f} "
                f"conf_mean={row['mean_fit_confidence']:.3f}/{row['mean_fit_confidence_target']:.3f} "
                f"diff={row['diff_fit_loss']:.4f} "
                f"cover={row['coverage_loss']:.4f} mass={row['mean_plane_mass']:.2f} "
                f"eig={row['mean_eig_min']:.6f}/{row['mean_eig_gap']:.6f} "
                f"bg={row['background_ratio']:.3f} acc={row['patch_acc']:.3f}"
            )

    output_dir = Path(args.output_dir)
    exported = [export_case(model, case, output_dir, args, device) for case in cases]
    overview = {"input_dir": args.input_dir, "num_samples": len(cases), "history": history, "exported": exported}
    (output_dir / "bounded_support_head_summary.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    if args.save_checkpoint:
        ckpt = Path(args.save_checkpoint)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
        print(ckpt)
    print(output_dir / "bounded_support_head_summary.json")
    print(f"samples={len(cases)}")


if __name__ == "__main__":
    main()
