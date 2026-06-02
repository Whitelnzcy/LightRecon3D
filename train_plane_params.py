import argparse
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")

if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

if DUST3R_REPO_ROOT in sys.path:
    sys.path.remove(DUST3R_REPO_ROOT)
sys.path.insert(1, DUST3R_REPO_ROOT)

from dataloaders.s3d_dataset import Structured3DDataset
from loss.line_loss import bce_dice_line_loss
from loss.plane_param_loss import plane_parameter_loss
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel
from train import build_views_from_batch


def parse_args():
    parser = argparse.ArgumentParser("Train parameterized structure plane head")
    parser.add_argument("--root_dir", type=str, default="/data/zhucy23u/datasets/Structured3D")
    parser.add_argument("--weights_path", type=str, default="/data/zhucy23u/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    parser.add_argument("--save_dir", type=str, default="/data/zhucy23u/checkpoints/lightrecon_param/param_head_debug")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--plane_offset_scale", type=float, default=1000.0)
    parser.add_argument("--small_train_size", type=int, default=32)
    parser.add_argument("--small_val_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--run_val", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)
    parser.add_argument("--train_line", action="store_true")
    parser.add_argument(
        "--param_head_type",
        type=str,
        default="token",
        choices=[
            "token",
            "pixel",
            "geom_token",
            "geom_token_centered",
            "geom_token_conf",
            "geom_token_point_anchor",
            "geom_token_point_anchor_conf",
        ],
        help="token pools each plane region before parameter regression; pixel keeps the old dense map baseline.",
    )
    parser.add_argument("--line_weight", type=float, default=0.1)
    parser.add_argument("--param_weight", type=float, default=1.0)
    parser.add_argument("--param_normal_weight", type=float, default=1.0)
    parser.add_argument("--param_offset_weight", type=float, default=0.25)
    parser.add_argument("--param_consistency_weight", type=float, default=0.1)
    parser.add_argument("--point_plane_weight", type=float, default=0.0)
    parser.add_argument("--point_plane_max_points", type=int, default=512)
    parser.add_argument("--point_plane_clip", type=float, default=0.25)
    parser.add_argument("--confidence_weight", type=float, default=0.0)
    parser.add_argument("--confidence_angle_scale", type=float, default=30.0)
    parser.add_argument("--confidence_offset_scale", type=float, default=1.5)
    parser.add_argument("--param_min_pixels", type=int, default=64)
    parser.add_argument("--param_max_pixels_per_plane", type=int, default=2048)
    parser.add_argument("--param_max_planes_per_image", type=int, default=8)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def move_batch_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()


def make_loader(args, split):
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        plane_offset_scale=args.plane_offset_scale,
    )
    limit = args.small_train_size if split == "train" else args.small_val_size
    if limit is not None and limit > 0:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=True,
    )


def compute_param_losses(res, batch, args):
    if args.param_head_type in (
        "token",
        "geom_token",
        "geom_token_centered",
        "geom_token_conf",
        "geom_token_point_anchor",
        "geom_token_point_anchor_conf",
    ):
        return compute_token_param_losses(res, batch, args)

    pred_params = res.get("pred_plane_params_lowres", res["pred_plane_params"])
    loss_param, stats = plane_parameter_loss(
        pred_params=pred_params,
        gt_normal=batch["gt_plane_normal"],
        gt_offset=batch["gt_plane_offset"],
        gt_plane=batch["gt_plane"],
        valid_mask=batch["gt_plane_param_valid"],
        min_pixels=args.param_min_pixels,
        max_pixels_per_plane=args.param_max_pixels_per_plane,
        max_planes_per_image=args.param_max_planes_per_image,
        normal_weight=args.param_normal_weight,
        offset_weight=args.param_offset_weight,
        consistency_weight=args.param_consistency_weight,
    )

    if args.train_line:
        loss_line, line_stats = bce_dice_line_loss(res["pred_line"], batch["gt_line"])
    else:
        loss_line = loss_param * 0.0
        line_stats = {
            "loss_line_bce": loss_line,
            "loss_line_dice": loss_line,
        }

    loss_total = args.param_weight * loss_param + args.line_weight * loss_line

    log = {
        "loss_total": to_float(loss_total),
        "loss_param": to_float(loss_param),
        "loss_param_normal": to_float(stats["loss_param_normal"]),
        "loss_param_offset": to_float(stats["loss_param_offset"]),
        "loss_param_consistency": to_float(stats["loss_param_consistency"]),
        "num_param_planes": to_float(stats["num_param_planes"]),
        "loss_line": to_float(loss_line),
        "loss_line_bce": to_float(line_stats["loss_line_bce"]),
        "loss_line_dice": to_float(line_stats["loss_line_dice"]),
    }
    return loss_total, log


def resize_label_like(gt_plane, target_hw):
    if gt_plane.shape[-2:] == target_hw:
        return gt_plane.long()
    return torch.nn.functional.interpolate(
        gt_plane.unsqueeze(1).float(),
        size=target_hw,
        mode="nearest",
    )[:, 0].long()


def resize_map_like(x, target_hw):
    if x.shape[-2:] == target_hw:
        return x
    return torch.nn.functional.interpolate(x, size=target_hw, mode="nearest")


def safe_normalize(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def get_pts3d_from_res(res):
    for key in ["pts3d", "pts3d_in_other_view", "pointmap", "pred_pts3d"]:
        if key in res:
            pts = res[key]
            break
    else:
        raise KeyError(f"Cannot find pointmap keys in {list(res.keys())}")
    if pts.ndim != 4:
        raise ValueError(f"Expected 4D pointmap, got {pts.shape}")
    if pts.shape[-1] == 3:
        return pts
    if pts.shape[1] == 3:
        return pts.permute(0, 2, 3, 1).contiguous()
    raise ValueError(f"Cannot interpret pointmap shape {pts.shape}")


def pointmap_geometry_descriptor(points, eps=1e-6):
    """
    Return a 10D no-grad geometry descriptor from a plane-region point cloud:
      normal(3), plane offset(1), centroid(3), sqrt eigenvalues(3)
    """
    with torch.no_grad():
        finite = torch.isfinite(points).all(dim=1)
        points = points[finite]
        coord_ok = points.abs().amax(dim=1) < 1e4
        points = points[coord_ok]

        if points.shape[0] < 3:
            return torch.zeros(10, device=points.device, dtype=points.dtype)

        centroid = points.mean(dim=0)
        x = points - centroid
        cov = x.transpose(0, 1) @ x / max(1, points.shape[0] - 1)
        cov = 0.5 * (cov + cov.transpose(0, 1))
        cov = cov + eps * torch.eye(3, device=points.device, dtype=points.dtype)
        eigvals, eigvecs = torch.linalg.eigh(cov.float())
        eigvals = eigvals.to(points.dtype).clamp_min(0.0)
        eigvecs = eigvecs.to(points.dtype)
        normal = safe_normalize(eigvecs[:, 0], dim=0)
        d = -torch.dot(normal, centroid)

        # Canonicalize sign for descriptor stability.
        idx = torch.argmax(normal.abs())
        sign = torch.where(normal[idx] < 0, -1.0, 1.0)
        normal = normal * sign
        d = d * sign

        scales = torch.sqrt(torch.flip(eigvals, dims=[0]).clamp_min(0.0))
        desc = torch.cat([normal, d.view(1), centroid, scales], dim=0)
        desc = torch.nan_to_num(desc, nan=0.0, posinf=1e4, neginf=-1e4)
        return desc


def clean_pointmap_points(points):
    finite = torch.isfinite(points).all(dim=1)
    points = points[finite]
    coord_ok = points.abs().amax(dim=1) < 1e4
    return points[coord_ok]


def compute_token_param_losses(res, batch, args):
    feature_map = res["dec_feature_map"]
    use_geom = args.param_head_type in (
        "geom_token",
        "geom_token_centered",
        "geom_token_conf",
        "geom_token_point_anchor",
        "geom_token_point_anchor_conf",
    )
    use_centered_offset = args.param_head_type in (
        "geom_token_centered",
        "geom_token_point_anchor",
        "geom_token_point_anchor_conf",
    )
    use_point_anchor_target = args.param_head_type in (
        "geom_token_point_anchor",
        "geom_token_point_anchor_conf",
    )
    use_conf = args.param_head_type in ("geom_token_conf", "geom_token_point_anchor_conf")
    if use_conf:
        head = res["_geom_plane_token_conf_head"]
    else:
        head = res["_geom_plane_token_param_head"] if use_geom else res["_plane_token_param_head"]
    bsz, channels, h, w = feature_map.shape

    gt_plane = resize_label_like(batch["gt_plane"], (h, w))
    gt_normal = resize_map_like(batch["gt_plane_normal"], (h, w))
    gt_offset = resize_map_like(batch["gt_plane_offset"], (h, w))
    valid_mask = resize_map_like(batch["gt_plane_param_valid"], (h, w)) > 0.5

    token_feats = []
    target_normals = []
    target_offsets = []
    plane_centroids = []
    point_sets = []
    plane_counts = []

    feat_hw = feature_map.permute(0, 2, 3, 1).contiguous()
    normal_hw = gt_normal.permute(0, 2, 3, 1).contiguous()
    pts3d = get_pts3d_from_res(res) if use_geom else None
    gt_plane_pts = resize_label_like(batch["gt_plane"], pts3d.shape[1:3]) if use_geom else None

    for b in range(bsz):
        count_b = 0
        plane_ids = torch.unique(gt_plane[b])
        candidates = []
        for pid in plane_ids:
            pid_int = int(pid.item())
            if pid_int in (-1, 0, 255):
                continue
            mask = (gt_plane[b] == pid) & valid_mask[b, 0]
            count = int(mask.sum().item())
            if count >= args.param_min_pixels:
                candidates.append((pid, count))
        candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[: args.param_max_planes_per_image]

        for pid, _ in candidates:
            mask = (gt_plane[b] == pid) & valid_mask[b, 0]
            feat = feat_hw[b][mask].mean(dim=0)
            if use_geom:
                mask_pts = gt_plane_pts[b] == pid
                pts_plane = clean_pointmap_points(pts3d[b][mask_pts])
                geom_desc = pointmap_geometry_descriptor(pts_plane)
                feat = torch.cat([feat, geom_desc.to(dtype=feat.dtype)], dim=0)
                plane_centroids.append(geom_desc[4:7].to(dtype=feat.dtype))
                point_sets.append(pts_plane)
            target_n = safe_normalize(normal_hw[b][mask].mean(dim=0), dim=0)
            if use_point_anchor_target and use_geom:
                target_d = -torch.dot(target_n.to(dtype=geom_desc.dtype), geom_desc[4:7]).to(dtype=feat.dtype)
            else:
                target_d = gt_offset[b, 0][mask].mean()

            token_feats.append(feat)
            target_normals.append(target_n)
            target_offsets.append(target_d)
            if not use_geom:
                plane_centroids.append(torch.zeros(3, device=feat.device, dtype=feat.dtype))
                point_sets.append(None)
            count_b += 1
        plane_counts.append(count_b)

    if len(token_feats) == 0:
        zero = feature_map.sum() * 0.0
        log = {
            "loss_total": 0.0,
            "loss_param": 0.0,
            "loss_param_normal": 0.0,
            "loss_param_offset": 0.0,
            "loss_param_consistency": 0.0,
            "loss_param_confidence": 0.0,
            "mean_param_confidence": 0.0,
            "num_param_planes": 0.0,
            "loss_line": 0.0,
            "loss_line_bce": 0.0,
            "loss_line_dice": 0.0,
        }
        return zero, log

    token_feats = torch.stack(token_feats, dim=0)
    target_normals = torch.stack(target_normals, dim=0)
    target_offsets = torch.stack(target_offsets, dim=0)
    plane_centroids = torch.stack(plane_centroids, dim=0)

    pred = head(token_feats)
    pred_normals = safe_normalize(pred[:, :3], dim=1)
    if use_centered_offset:
        pred_offsets = -(pred_normals * plane_centroids).sum(dim=1) + pred[:, 3]
    else:
        pred_offsets = pred[:, 3]

    signs = torch.where((pred_normals * target_normals).sum(dim=1, keepdim=True) < 0, -1.0, 1.0)
    pred_normals = pred_normals * signs
    pred_offsets = pred_offsets * signs[:, 0]

    cos = (pred_normals * target_normals).sum(dim=1).clamp(-1.0, 1.0)
    loss_normal = (1.0 - cos).mean()
    loss_offset = torch.nn.functional.smooth_l1_loss(pred_offsets, target_offsets)
    angle_deg = torch.rad2deg(torch.acos(cos.detach().clamp(-1.0, 1.0)))
    offset_err = torch.abs((pred_offsets - target_offsets).detach())
    if use_conf:
        conf_logits = pred[:, 4]
        target_conf = torch.exp(
            -angle_deg / args.confidence_angle_scale
            -offset_err / args.confidence_offset_scale
        ).clamp(0.0, 1.0)
        loss_confidence = torch.nn.functional.binary_cross_entropy_with_logits(conf_logits, target_conf)
        mean_confidence = torch.sigmoid(conf_logits.detach()).mean()
    else:
        loss_confidence = loss_normal * 0.0
        mean_confidence = loss_normal.detach() * 0.0
    point_losses = []
    if args.point_plane_weight > 0 and use_geom:
        for idx, points in enumerate(point_sets):
            if points is None or points.shape[0] < 3:
                continue
            if points.shape[0] > args.point_plane_max_points:
                select = torch.linspace(
                    0,
                    points.shape[0] - 1,
                    steps=args.point_plane_max_points,
                    device=points.device,
                ).long()
                points = points[select]
            dist = torch.abs(points @ pred_normals[idx] + pred_offsets[idx])
            point_losses.append(torch.clamp(dist, max=args.point_plane_clip).mean())
    if point_losses:
        loss_consistency = torch.stack(point_losses).mean()
    else:
        loss_consistency = loss_normal * 0.0

    loss_param = (
        args.param_normal_weight * loss_normal
        + args.param_offset_weight * loss_offset
        + args.param_consistency_weight * loss_consistency
        + args.point_plane_weight * loss_consistency
        + args.confidence_weight * loss_confidence
    )

    if args.train_line:
        loss_line, line_stats = bce_dice_line_loss(res["pred_line"], batch["gt_line"])
    else:
        loss_line = loss_param * 0.0
        line_stats = {"loss_line_bce": loss_line, "loss_line_dice": loss_line}

    loss_total = args.param_weight * loss_param + args.line_weight * loss_line
    log = {
        "loss_total": to_float(loss_total),
        "loss_param": to_float(loss_param),
        "loss_param_normal": to_float(loss_normal),
        "loss_param_offset": to_float(loss_offset),
        "loss_param_consistency": to_float(loss_consistency),
        "loss_param_confidence": to_float(loss_confidence),
        "mean_param_confidence": to_float(mean_confidence),
        "num_param_planes": float(sum(plane_counts) / max(1, len(plane_counts))),
        "loss_line": to_float(loss_line),
        "loss_line_bce": to_float(line_stats["loss_line_bce"]),
        "loss_line_dice": to_float(line_stats["loss_line_dice"]),
    }
    return loss_total, log


def run_epoch(model, loader, optimizer, device, args, train):
    model.train(train)
    model.backbone.eval()
    stats_sum = defaultdict(float)
    steps = 0

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        view1, view2 = build_views_from_batch(batch, prefix="param")

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            res1, _ = model(view1, view2)
            res1["_plane_token_param_head"] = model.plane_token_param_head
            res1["_geom_plane_token_param_head"] = model.geom_plane_token_param_head
            res1["_geom_plane_token_conf_head"] = model.geom_plane_token_conf_head
            loss, stats = compute_param_losses(res1, batch, args)

            if train:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.grad_clip,
                    )
                optimizer.step()

        for key, value in stats.items():
            stats_sum[key] += value
        steps += 1

        if step == 1 or step % args.log_every == 0 or step == len(loader):
            tag = "Train" if train else "Val"
            print(
                f"[{tag}] {step}/{len(loader)} "
                f"total={stats['loss_total']:.4f} "
                f"param={stats['loss_param']:.4f} "
                f"normal={stats['loss_param_normal']:.4f} "
                f"offset={stats['loss_param_offset']:.4f} "
                f"cons={stats['loss_param_consistency']:.4f} "
                f"conf={stats.get('mean_param_confidence', 0.0):.3f} "
                f"planes={stats['num_param_planes']:.1f}"
            )

    return {key: value / max(1, steps) for key, value in stats_sum.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Parameterized Plane Head Training")
    print("=" * 80)
    print(f"device: {device}")
    print(f"save_dir: {args.save_dir}")
    print("This script ignores old stable/anchor checkpoints and trains a fresh parameter head.")

    train_loader = make_loader(args, "train")
    val_loader = make_loader(args, "val") if args.run_val else None

    backbone = build_dust3r_backbone(args.weights_path, device=device)
    model = LightReconModel(
        dust3r_backbone=backbone,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    ).to(device)
    freeze_backbone(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        print("=" * 80)
        print(f"Epoch {epoch}/{args.num_epochs}")
        train_stats = run_epoch(model, train_loader, optimizer, device, args, train=True)
        print("Train | " + ", ".join(f"{k}: {v:.6f}" for k, v in train_stats.items()))

        val_stats = None
        if val_loader is not None:
            val_stats = run_epoch(model, val_loader, optimizer, device, args, train=False)
            print("Val   | " + ", ".join(f"{k}: {v:.6f}" for k, v in val_stats.items()))

        metric = val_stats["loss_param"] if val_stats is not None else train_stats["loss_param"]
        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "train_stats": train_stats,
            "val_stats": val_stats,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "latest.pth"))
        if metric < best_val:
            best_val = metric
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))

    print("=" * 80)
    print("Training finished.")
    print(f"latest: {os.path.join(args.save_dir, 'latest.pth')}")
    print(f"best  : {os.path.join(args.save_dir, 'best.pth')}")


if __name__ == "__main__":
    main()
