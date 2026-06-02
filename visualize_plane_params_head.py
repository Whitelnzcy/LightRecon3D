import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")

if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

if DUST3R_REPO_ROOT in sys.path:
    sys.path.remove(DUST3R_REPO_ROOT)
sys.path.insert(1, DUST3R_REPO_ROOT)

from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel
from train import build_views_from_batch
from visualize_pointmap_compare import colorize_label_map, tensor_img_to_uint8


def safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def resize_label(gt_plane, target_hw):
    if gt_plane.shape[-2:] == target_hw:
        return gt_plane.long()
    return F.interpolate(gt_plane.unsqueeze(1).float(), size=target_hw, mode="nearest")[:, 0].long()


def resize_map(x, target_hw):
    if x.shape[-2:] == target_hw:
        return x
    return F.interpolate(x, size=target_hw, mode="nearest")


def normal_to_rgb(normal_hw3):
    rgb = (normal_hw3 + 1.0) * 0.5
    return np.clip(rgb, 0.0, 1.0)


def angle_deg(pred_n, gt_n):
    v = torch.abs((pred_n * gt_n).sum()).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(v))


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
        normal = normalize(eigvecs[:, 0], dim=0)
        d = -torch.dot(normal, centroid)

        idx = torch.argmax(normal.abs())
        sign = torch.where(normal[idx] < 0, -1.0, 1.0)
        normal = normal * sign
        d = d * sign

        scales = torch.sqrt(torch.flip(eigvals, dims=[0]).clamp_min(0.0))
        desc = torch.cat([normal, d.view(1), centroid, scales], dim=0)
        return torch.nan_to_num(desc, nan=0.0, posinf=1e4, neginf=-1e4)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--param_head_type", type=str, default="token", choices=["token", "geom_token", "geom_token_centered", "geom_token_conf"])
    parser.add_argument("--sample_idx", type=int, action="append", required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--plane_offset_scale", type=float, default=1000.0)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)
    parser.add_argument("--min_pixels", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        plane_offset_scale=args.plane_offset_scale,
    )

    backbone = build_dust3r_backbone(args.weights_path, device=device)
    model = LightReconModel(
        dust3r_backbone=backbone,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    ).to(device)
    ckpt = safe_load(args.ckpt_path, device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()

    for sample_idx in args.sample_idx:
        sample = dataset[sample_idx]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}
        view1, view2 = build_views_from_batch(batch, prefix=f"{args.split}_{sample_idx}")
        res1, _ = model(view1, view2)

        feat = res1["dec_feature_map"][0]
        h, w = feat.shape[-2:]
        feat_hw = feat.permute(1, 2, 0).contiguous()
        use_geom = args.param_head_type in ("geom_token", "geom_token_centered", "geom_token_conf")
        pts3d = get_pts3d_from_res(res1)[0] if use_geom else None
        gt_plane_pts = resize_label(batch["gt_plane"], pts3d.shape[:2])[0] if use_geom else None

        gt_plane = resize_label(batch["gt_plane"], (h, w))[0]
        gt_normal = resize_map(batch["gt_plane_normal"], (h, w))[0].permute(1, 2, 0)
        gt_offset = resize_map(batch["gt_plane_offset"], (h, w))[0, 0]
        valid = resize_map(batch["gt_plane_param_valid"], (h, w))[0, 0] > 0.5

        pred_normal = torch.zeros_like(gt_normal)
        pred_offset = torch.zeros_like(gt_offset)
        angle_map = torch.zeros_like(gt_offset)
        rows = []

        for pid in torch.unique(gt_plane):
            pid_int = int(pid.item())
            if pid_int in (-1, 0, 255):
                continue
            mask = (gt_plane == pid) & valid
            if int(mask.sum().item()) < args.min_pixels:
                continue
            token = feat_hw[mask].mean(dim=0, keepdim=True)
            if use_geom:
                mask_pts = gt_plane_pts == pid
                geom_desc = pointmap_geometry_descriptor(pts3d[mask_pts]).to(dtype=token.dtype)
                if args.param_head_type == "geom_token_conf":
                    pred = model.geom_plane_token_conf_head(torch.cat([token[0], geom_desc], dim=0).unsqueeze(0))[0]
                    pred_conf = float(torch.sigmoid(pred[4]).cpu())
                else:
                    pred = model.geom_plane_token_param_head(torch.cat([token[0], geom_desc], dim=0).unsqueeze(0))[0]
                    pred_conf = None
            else:
                pred = model.plane_token_param_head(token)[0]
                pred_conf = None
            pred_n = normalize(pred[:3], dim=0)
            if args.param_head_type == "geom_token_centered":
                pred_d = -torch.dot(pred_n, geom_desc[4:7]) + pred[3]
            else:
                pred_d = pred[3]
            gt_n = normalize(gt_normal[mask].mean(dim=0), dim=0)
            gt_d = gt_offset[mask].mean()
            sign = torch.where((pred_n * gt_n).sum() < 0, -1.0, 1.0)
            pred_n = pred_n * sign
            pred_d = pred_d * sign
            a = angle_deg(pred_n, gt_n)

            pred_normal[mask] = pred_n
            pred_offset[mask] = pred_d
            angle_map[mask] = a
            rows.append((pid_int, int(mask.sum().item()), float(a.cpu()), float(torch.abs(pred_d - gt_d).cpu()), pred_conf))

        rgb = tensor_img_to_uint8(batch["img"])
        plane_rgb = colorize_label_map(gt_plane.detach().cpu().numpy().astype(np.int32))
        gt_norm_rgb = normal_to_rgb(gt_normal.detach().cpu().numpy())
        pred_norm_rgb = normal_to_rgb(pred_normal.detach().cpu().numpy())
        angle_np = angle_map.detach().cpu().numpy()

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.ravel()
        panels = [
            ("RGB", rgb),
            ("GT plane mask", plane_rgb),
            ("GT normal", gt_norm_rgb),
            ("Pred normal", pred_norm_rgb),
        ]
        for ax, (title, img) in zip(axes[:4], panels):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        im = axes[4].imshow(angle_np, cmap="magma", vmin=0, vmax=90)
        axes[4].set_title("Angle error map (deg)")
        axes[4].axis("off")
        fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

        axes[5].axis("off")
        has_conf = any(row[4] is not None for row in rows)
        lines = ["plane | pixels | angle | offset err" + (" | conf" if has_conf else "")]
        for pid, pixels, angle, offerr, conf in rows[:10]:
            line = f"{pid:>5} | {pixels:>6} | {angle:>5.1f} | {offerr:>8.3f}"
            if has_conf:
                line += f" | {conf:>4.2f}"
            lines.append(line)
        axes[5].text(0.0, 1.0, "\n".join(lines), family="monospace", va="top", fontsize=10)
        axes[5].set_title("Plane parameter errors")

        fig.suptitle(f"{args.split} sample {sample_idx} - {args.param_head_type} plane parameters", fontsize=14)
        fig.tight_layout()
        out_path = out_dir / f"{args.split}_{sample_idx:06d}_plane_params.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(out_path)


if __name__ == "__main__":
    main()
