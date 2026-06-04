import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_points(points):
    center = points.mean(axis=0, keepdims=True).astype(np.float32)
    scale = np.linalg.norm(points - center, axis=1).max()
    scale = max(float(scale), 1e-6)
    return ((points - center) / scale).astype(np.float32), center[0], scale


def world_to_normalized_offset(normal, offset_world, center, scale):
    return float((offset_world + float(np.dot(normal, center))) / scale)


def normalized_to_world_offset(normal, offset_norm, center, scale):
    return float(offset_norm * scale - float(np.dot(normal, center)))


def fit_svd_plane(points):
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1].astype(np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1e-8)
    offset = -float(np.dot(normal, centroid))
    return normal, offset


def robust_refine_plane(points_norm, normal, offset, trim_ratio, min_points):
    residual = np.abs(points_norm @ normal + offset)
    if len(residual) < min_points:
        return normal, offset, False
    keep = max(min_points, int(round(len(residual) * trim_ratio)))
    keep = min(keep, len(residual))
    idx = np.argpartition(residual, keep - 1)[:keep]
    refined_normal, refined_offset = fit_svd_plane(points_norm[idx])
    if float(np.dot(refined_normal, normal)) < 0:
        refined_normal = -refined_normal
        refined_offset = -refined_offset
    return refined_normal.astype(np.float32), float(refined_offset), True


def plane_features(points_norm, colors, point_ids, plane_id, normal, offset, min_points):
    mask = point_ids == plane_id
    if int(mask.sum()) < min_points:
        return None
    pts = points_norm[mask]
    rgb = colors[mask].astype(np.float32) / 255.0
    residual = np.abs(pts @ normal + offset)
    stats = np.concatenate(
        [
            normal.astype(np.float32),
            np.asarray([offset, float(mask.mean())], dtype=np.float32),
            pts.mean(axis=0).astype(np.float32),
            pts.std(axis=0).astype(np.float32),
            rgb.mean(axis=0).astype(np.float32),
            rgb.std(axis=0).astype(np.float32),
            np.asarray(
                [
                    float(residual.mean()),
                    float(np.median(residual)),
                    float(np.percentile(residual, 90)),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )
    return stats


def sampled_point_features(points_norm, colors, point_ids, plane_id, normal, offset, target_normal, target_offset, args, seed):
    inlier_idx = np.flatnonzero(point_ids == plane_id)
    if len(inlier_idx) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    pos_count = args.point_samples_per_plane // 2
    neg_count = args.point_samples_per_plane - pos_count
    pos_idx = rng.choice(inlier_idx, size=pos_count, replace=len(inlier_idx) < pos_count)
    random_pool_size = min(len(points_norm), max(args.point_samples_per_plane * 8, neg_count))
    random_pool = rng.choice(len(points_norm), size=random_pool_size, replace=False)
    pool = np.unique(np.concatenate([random_pool, inlier_idx]))
    pool_residual = np.abs(points_norm[pool] @ normal + offset)
    near_pool = pool[pool_residual <= args.keep_candidate_threshold]
    neg_pool = np.setdiff1d(near_pool if len(near_pool) >= neg_count else pool, inlier_idx, assume_unique=False)
    if len(neg_pool) == 0:
        neg_pool = pool
    neg_idx = rng.choice(neg_pool, size=neg_count, replace=len(neg_pool) < neg_count)
    idx = np.concatenate([pos_idx, neg_idx])
    pts = points_norm[idx]
    rgb = colors[idx].astype(np.float32) / 255.0
    base_r = np.abs(pts @ normal + offset).reshape(-1, 1)
    target_r = np.abs(pts @ target_normal + target_offset)
    original_inlier = (point_ids[idx] == plane_id).astype(np.float32).reshape(-1, 1)
    plane_feat = np.concatenate(
        [
            np.repeat(normal.reshape(1, 3), len(idx), axis=0),
            np.full((len(idx), 1), offset, dtype=np.float32),
        ],
        axis=1,
    )
    features = np.concatenate([pts, rgb, base_r, original_inlier, plane_feat], axis=1).astype(np.float32)
    if args.use_soft_keep_labels:
        tau = max(float(args.soft_keep_temperature), 1e-6)
        labels = np.exp(-target_r / tau).astype(np.float32)
        if args.soft_keep_inlier_floor > 0:
            labels = np.maximum(labels, original_inlier[:, 0] * float(args.soft_keep_inlier_floor))
        labels = np.clip(labels, 0.0, 1.0).astype(np.float32)
    else:
        labels = (target_r <= args.keep_target_threshold).astype(np.float32)
    weights = np.ones_like(labels, dtype=np.float32)
    weights[original_inlier[:, 0] > 0.5] *= float(args.keep_inlier_weight)
    near_target = target_r <= args.keep_target_threshold
    weights[near_target] *= float(args.keep_positive_weight)
    return features, labels, weights


def build_records(path, args):
    raw = np.load(path)
    points = raw["points"].astype(np.float32)
    colors = raw["colors"].astype(np.uint8)
    point_ids = raw["point_plane_ids"].astype(np.int32)
    plane_ids = raw["plane_ids"].astype(np.int32)
    normals = raw["plane_normals"].astype(np.float32)
    offsets_world = raw["plane_offsets"].astype(np.float32)
    counts = raw["plane_inlier_counts"].astype(np.int32)
    points_norm, center, scale = normalize_points(points)
    records = []
    for plane_id, normal, offset_world, count in zip(plane_ids, normals, offsets_world, counts):
        offset_norm = world_to_normalized_offset(normal, float(offset_world), center, scale)
        feat = plane_features(points_norm, colors, point_ids, int(plane_id), normal, offset_norm, args.min_points)
        if feat is None:
            continue
        mask = point_ids == int(plane_id)
        target_normal, target_offset, ok = robust_refine_plane(
            points_norm[mask],
            normal,
            offset_norm,
            args.target_trim_ratio,
            args.min_points,
        )
        if not ok:
            continue
        point_features, point_keep_labels, point_keep_weights = sampled_point_features(
            points_norm,
            colors,
            point_ids,
            int(plane_id),
            normal,
            offset_norm,
            target_normal,
            target_offset,
            args,
            args.seed + int(plane_id) * 1009 + len(records),
        )
        if point_features is None:
            continue
        records.append(
            {
                "sample": Path(path).name.replace("_full_pointcloud_editable_planes_data.npz", ""),
                "path": str(path),
                "plane_id": int(plane_id),
                "feature": feat.astype(np.float32),
                "normal": normal.astype(np.float32),
                "offset_norm": np.float32(offset_norm),
                "target_normal": target_normal.astype(np.float32),
                "target_offset_norm": np.float32(target_offset),
                "center": center.astype(np.float32),
                "scale": np.float32(scale),
                "count": int(count),
                "point_features": point_features,
                "point_keep_labels": point_keep_labels,
                "point_keep_weights": point_keep_weights,
            }
        )
    return records


class PlaneProposalRefiner(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.keep_head = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feature, normal, offset):
        raw = self.net(feature)
        delta_normal = raw[:, :3]
        delta_offset = raw[:, 3]
        refined_normal = F.normalize(normal + delta_normal, dim=-1)
        refined_offset = offset + delta_offset
        return refined_normal, refined_offset

    def confidence(self, feature, base_residual, refined_residual):
        x = torch.cat([feature, base_residual[:, None], refined_residual[:, None]], dim=-1)
        return self.confidence_head(x).squeeze(-1)

    def keep_logits(self, point_feature):
        return self.keep_head(point_feature).squeeze(-1)


def plane_distance_loss(points, labels, plane_ids, normals, offsets):
    losses = []
    for i, plane_id in enumerate(plane_ids):
        mask = labels == int(plane_id)
        if np.any(mask):
            pts = torch.from_numpy(points[mask]).to(normals.device)
            losses.append(torch.abs(pts @ normals[i] + offsets[i]).mean())
    if not losses:
        return normals.sum() * 0.0
    return torch.stack(losses).mean()


def soft_dice_loss(logits, labels, weights, eps=1e-6):
    prob = torch.sigmoid(logits)
    intersection = torch.sum(weights * prob * labels)
    denominator = torch.sum(weights * prob) + torch.sum(weights * labels)
    return 1.0 - (2.0 * intersection + eps) / (denominator + eps)


def evaluate_records(model, records, device):
    feature = torch.from_numpy(np.stack([r["feature"] for r in records])).to(device)
    normal = torch.from_numpy(np.stack([r["normal"] for r in records])).to(device)
    offset = torch.from_numpy(np.asarray([r["offset_norm"] for r in records], dtype=np.float32)).to(device)
    target_normal = torch.from_numpy(np.stack([r["target_normal"] for r in records])).to(device)
    target_offset = torch.from_numpy(np.asarray([r["target_offset_norm"] for r in records], dtype=np.float32)).to(device)
    with torch.no_grad():
        refined_normal, refined_offset = model(feature, normal, offset)
        base_delta = torch.abs(offset - target_offset) + (1.0 - torch.sum(normal * target_normal, dim=-1).clamp(-1, 1))
        refined_delta = torch.abs(refined_offset - target_offset) + (
            1.0 - torch.sum(refined_normal * target_normal, dim=-1).clamp(-1, 1)
        )
        conf_logit = model.confidence(feature, base_delta, refined_delta)
        conf_prob = torch.sigmoid(conf_logit)
    return refined_normal.cpu().numpy(), refined_offset.cpu().numpy(), conf_prob.cpu().numpy()


def predict_keep_mask(model, points_norm, colors, point_ids, plane_id, normal, offset_norm, device, args):
    base_residual = np.abs(points_norm @ normal + offset_norm)
    candidate = base_residual <= args.keep_candidate_threshold
    if not np.any(candidate):
        return point_ids == plane_id
    idx = np.flatnonzero(candidate)
    pts = points_norm[idx]
    rgb = colors[idx].astype(np.float32) / 255.0
    base_r = base_residual[idx].reshape(-1, 1)
    original_inlier = (point_ids[idx] == plane_id).astype(np.float32).reshape(-1, 1)
    plane_feat = np.concatenate(
        [
            np.repeat(normal.reshape(1, 3), len(idx), axis=0),
            np.full((len(idx), 1), offset_norm, dtype=np.float32),
        ],
        axis=1,
    )
    feat = np.concatenate([pts, rgb, base_r, original_inlier, plane_feat], axis=1).astype(np.float32)
    logits = []
    with torch.no_grad():
        for start in range(0, len(feat), args.keep_eval_batch_size):
            batch = torch.from_numpy(feat[start : start + args.keep_eval_batch_size]).to(device)
            logits.append(model.keep_logits(batch).detach().cpu().numpy())
    prob = 1.0 / (1.0 + np.exp(-np.concatenate(logits)))
    mask = np.zeros(len(points_norm), dtype=bool)
    mask[idx[prob >= args.keep_prob_threshold]] = True
    return mask


def export_refined_npzs(model, paths, output_dir, args, device):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        raw = np.load(path)
        points = raw["points"].astype(np.float32)
        colors = raw["colors"].astype(np.uint8)
        point_ids = raw["point_plane_ids"].astype(np.int32)
        plane_ids = raw["plane_ids"].astype(np.int32)
        base_normals = raw["plane_normals"].astype(np.float32)
        base_offsets_world = raw["plane_offsets"].astype(np.float32)
        counts = raw["plane_inlier_counts"].astype(np.int32)
        points_norm, center, scale = normalize_points(points)

        records = build_records(path, args)
        refined_by_id = {}
        if records:
            refined_normals, refined_offsets_norm, confidence = evaluate_records(model, records, device)
            for rec, n, d_norm, conf in zip(records, refined_normals, refined_offsets_norm, confidence):
                refined_by_id[rec["plane_id"]] = (n.astype(np.float32), float(d_norm), float(conf))

        out_normals = []
        out_offsets_world = []
        refined_point_ids = np.full_like(point_ids, -1)
        for plane_id, base_n, base_d_world in zip(plane_ids, base_normals, base_offsets_world):
            conf = 0.0
            if int(plane_id) in refined_by_id:
                n, d_norm, conf = refined_by_id[int(plane_id)]
                d_world = normalized_to_world_offset(n, d_norm, center, scale)
            else:
                n = base_n
                d_norm = world_to_normalized_offset(base_n, float(base_d_world), center, scale)
                d_world = float(base_d_world)
            mask = point_ids == int(plane_id)
            base_res = np.abs(points[mask] @ base_n + float(base_d_world)).mean() if np.any(mask) else np.nan
            ref_res = np.abs(points[mask] @ n + float(d_world)).mean() if np.any(mask) else np.nan
            accepted = bool(np.isfinite(base_res) and np.isfinite(ref_res) and ref_res <= base_res)
            accepted = accepted and (conf >= args.confidence_threshold or not args.use_learned_confidence_gate)
            if args.accept_only_if_improves and not accepted:
                n = base_n
                d_norm = world_to_normalized_offset(base_n, float(base_d_world), center, scale)
                d_world = float(base_d_world)
                ref_res = base_res
                accepted = False
            if args.refine_point_bindings and accepted:
                keep_mask = predict_keep_mask(model, points_norm, colors, point_ids, int(plane_id), n, d_norm, device, args)
                if args.refit_with_learned_bindings and int(keep_mask.sum()) >= args.min_points:
                    refit_n, refit_d_norm = fit_svd_plane(points_norm[keep_mask])
                    if float(np.dot(refit_n, n)) < 0:
                        refit_n = -refit_n
                        refit_d_norm = -refit_d_norm
                    refit_d_world = normalized_to_world_offset(refit_n, refit_d_norm, center, scale)
                    refit_res = (
                        np.abs(points[mask] @ refit_n + float(refit_d_world)).mean() if np.any(mask) else np.nan
                    )
                    can_use_refit = bool(np.isfinite(refit_res) and refit_res <= ref_res)
                    if can_use_refit or args.allow_refit_without_original_improvement:
                        n = refit_n.astype(np.float32)
                        d_norm = float(refit_d_norm)
                        d_world = float(refit_d_world)
                        ref_res = float(refit_res)
            else:
                keep_mask = mask
            refined_point_ids[keep_mask] = int(plane_id)
            out_normals.append(n)
            out_offsets_world.append(d_world)
            rows.append(
                {
                    "sample": Path(path).name.replace("_full_pointcloud_editable_planes_data.npz", ""),
                    "plane_id": int(plane_id),
                    "points": int(mask.sum()),
                    "base_mean_abs_distance": float(base_res),
                    "refined_mean_abs_distance": float(ref_res),
                    "delta": float(base_res - ref_res) if np.isfinite(base_res) and np.isfinite(ref_res) else np.nan,
                    "accepted": accepted,
                    "confidence": float(conf),
                    "base_bound_points": int(mask.sum()),
                    "refined_bound_points": int(keep_mask.sum()),
                }
            )

        stem = Path(path).name.replace("_full_pointcloud_editable_planes_data.npz", "")
        npz_path = output_dir / f"{stem}_refined_plane_proposals_data.npz"
        np.savez_compressed(
            npz_path,
            points=points,
            colors=colors,
            point_plane_ids=refined_point_ids if args.refine_point_bindings else point_ids,
            plane_ids=plane_ids,
            plane_normals=np.asarray(out_normals, dtype=np.float32),
            plane_offsets=np.asarray(out_offsets_world, dtype=np.float32),
            plane_inlier_counts=counts,
        )
        json_path = output_dir / f"{stem}_refined_plane_proposals_report.json"
        json_path.write_text(
            json.dumps(
                {
                    "input_npz": str(path),
                    "output_npz": str(npz_path),
                    "planes": [r for r in rows if r["sample"] == stem],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    csv_path = output_dir / "plane_proposal_refinement_eval.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "plane_id",
                "points",
                "base_mean_abs_distance",
                "refined_mean_abs_distance",
                "delta",
                "accepted",
                "confidence",
                "base_bound_points",
                "refined_bound_points",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "planes": len(rows),
        "mean_base_abs_distance": float(np.nanmean([r["base_mean_abs_distance"] for r in rows])),
        "mean_refined_abs_distance": float(np.nanmean([r["refined_mean_abs_distance"] for r in rows])),
        "mean_delta": float(np.nanmean([r["delta"] for r in rows])),
        "accepted_planes": int(sum(1 for r in rows if r["accepted"])),
        "mean_confidence": float(np.nanmean([r["confidence"] for r in rows])),
        "mean_base_bound_points": float(np.nanmean([r["base_bound_points"] for r in rows])),
        "mean_refined_bound_points": float(np.nanmean([r["refined_bound_points"] for r in rows])),
    }
    (output_dir / "plane_proposal_refinement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows, summary


def main():
    parser = argparse.ArgumentParser("Train a learnable refiner for RANSAC/GT plane proposals")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--eval_input_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*_full_pointcloud_editable_planes_data.npz")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--target_trim_ratio", type=float, default=0.8)
    parser.add_argument("--min_points", type=int, default=500)
    parser.add_argument("--accept_only_if_improves", action="store_true")
    parser.add_argument("--use_learned_confidence_gate", action="store_true")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--refine_point_bindings", action="store_true")
    parser.add_argument("--point_samples_per_plane", type=int, default=2048)
    parser.add_argument("--keep_candidate_threshold", type=float, default=0.08)
    parser.add_argument("--keep_target_threshold", type=float, default=0.035)
    parser.add_argument("--keep_prob_threshold", type=float, default=0.5)
    parser.add_argument("--keep_eval_batch_size", type=int, default=65536)
    parser.add_argument("--use_soft_keep_labels", action="store_true")
    parser.add_argument("--soft_keep_temperature", type=float, default=0.035)
    parser.add_argument("--soft_keep_inlier_floor", type=float, default=0.25)
    parser.add_argument("--keep_positive_weight", type=float, default=1.0)
    parser.add_argument("--keep_inlier_weight", type=float, default=1.0)
    parser.add_argument("--keep_dice_weight", type=float, default=0.0)
    parser.add_argument("--refit_with_learned_bindings", action="store_true")
    parser.add_argument("--allow_refit_without_original_improvement", action="store_true")
    parser.add_argument("--normal_weight", type=float, default=1.0)
    parser.add_argument("--offset_weight", type=float, default=1.0)
    parser.add_argument("--confidence_weight", type=float, default=0.2)
    parser.add_argument("--keep_weight", type=float, default=0.2)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    paths = sorted(Path(args.input_dir).glob(args.pattern))
    eval_paths = sorted(Path(args.eval_input_dir).glob(args.pattern)) if args.eval_input_dir else paths
    records = []
    for path in paths:
        records.extend(build_records(path, args))
    if not records:
        raise FileNotFoundError(f"No usable plane proposals found under {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature = torch.from_numpy(np.stack([r["feature"] for r in records])).to(device)
    normal = torch.from_numpy(np.stack([r["normal"] for r in records])).to(device)
    offset = torch.from_numpy(np.asarray([r["offset_norm"] for r in records], dtype=np.float32)).to(device)
    target_normal = torch.from_numpy(np.stack([r["target_normal"] for r in records])).to(device)
    target_offset = torch.from_numpy(np.asarray([r["target_offset_norm"] for r in records], dtype=np.float32)).to(device)
    point_record_count = int(sum(len(r["point_keep_labels"]) for r in records))

    model = PlaneProposalRefiner(feature.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    history = []
    for step in range(1, args.steps + 1):
        idx = rng.choice(len(records), size=min(args.batch_size, len(records)), replace=False)
        idx_t = torch.from_numpy(idx).to(device)
        pred_normal, pred_offset = model(feature[idx_t], normal[idx_t], offset[idx_t])
        n_loss = (1.0 - torch.sum(pred_normal * target_normal[idx_t], dim=-1).clamp(-1, 1)).mean()
        o_loss = torch.abs(pred_offset - target_offset[idx_t]).mean()
        base_delta = torch.abs(offset[idx_t] - target_offset[idx_t]) + (
            1.0 - torch.sum(normal[idx_t] * target_normal[idx_t], dim=-1).clamp(-1, 1)
        )
        refined_delta = torch.abs(pred_offset - target_offset[idx_t]) + (
            1.0 - torch.sum(pred_normal * target_normal[idx_t], dim=-1).clamp(-1, 1)
        )
        conf_target = (refined_delta.detach() < base_delta.detach()).float()
        conf_logit = model.confidence(feature[idx_t], base_delta.detach(), refined_delta.detach())
        conf_loss = F.binary_cross_entropy_with_logits(conf_logit, conf_target)

        batch_point_features = np.concatenate([records[int(i)]["point_features"] for i in idx], axis=0)
        batch_point_labels = np.concatenate([records[int(i)]["point_keep_labels"] for i in idx], axis=0)
        batch_point_weights = np.concatenate([records[int(i)]["point_keep_weights"] for i in idx], axis=0)
        point_feature_t = torch.from_numpy(batch_point_features).to(device)
        point_label_t = torch.from_numpy(batch_point_labels).to(device)
        point_weight_t = torch.from_numpy(batch_point_weights).to(device)
        keep_logit = model.keep_logits(point_feature_t)
        keep_bce = F.binary_cross_entropy_with_logits(keep_logit, point_label_t, reduction="none")
        keep_loss = torch.sum(keep_bce * point_weight_t) / torch.clamp(point_weight_t.sum(), min=1.0)
        keep_dice = soft_dice_loss(keep_logit, point_label_t, point_weight_t)

        loss = (
            args.normal_weight * n_loss
            + args.offset_weight * o_loss
            + args.confidence_weight * conf_loss
            + args.keep_weight * keep_loss
            + args.keep_dice_weight * keep_dice
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "normal_loss": float(n_loss.detach().cpu()),
            "offset_loss": float(o_loss.detach().cpu()),
            "confidence_loss": float(conf_loss.detach().cpu()),
            "keep_loss": float(keep_loss.detach().cpu()),
            "keep_dice_loss": float(keep_dice.detach().cpu()),
        }
        history.append(row)
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step:04d} loss={row['loss']:.6f} normal={row['normal_loss']:.6f} "
                f"offset={row['offset_loss']:.6f} conf={row['confidence_loss']:.6f} "
                f"keep={row['keep_loss']:.6f} dice={row['keep_dice_loss']:.6f}"
            )

    output_dir = Path(args.output_dir)
    rows, summary = export_refined_npzs(model, eval_paths, output_dir, args, device)
    checkpoint = {
        "model": model.state_dict(),
        "args": vars(args),
        "feature_dim": int(feature.shape[1]),
        "history": history,
        "summary": summary,
        "train_records": len(records),
        "eval_files": len(eval_paths),
        "point_records": point_record_count,
    }
    torch.save(checkpoint, output_dir / "plane_proposal_refiner.pt")
    print(output_dir / "plane_proposal_refinement_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
