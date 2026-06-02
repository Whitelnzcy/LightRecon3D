import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset


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


def safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def move_batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


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
    if pts.shape[-1] == 3:
        return pts
    if pts.shape[1] == 3:
        return pts.permute(0, 2, 3, 1).contiguous()
    raise ValueError(f"Cannot interpret pointmap shape {pts.shape}")


def geometry_descriptor(points, eps=1e-6):
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
    return torch.nan_to_num(torch.cat([normal, d.view(1), centroid, scales], dim=0), nan=0.0, posinf=1e4, neginf=-1e4)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--plane_offset_scale", type=float, default=1000.0)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)
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
    )
    parser.add_argument("--min_pixels", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        plane_offset_scale=args.plane_offset_scale,
    )
    if args.num_samples > 0:
        dataset = Subset(dataset, list(range(min(args.num_samples, len(dataset)))))

    backbone = build_dust3r_backbone(args.weights_path, device=device)
    model = LightReconModel(
        dust3r_backbone=backbone,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    ).to(device)
    ckpt = safe_load(args.ckpt_path, device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    rows = []
    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        batch = {
            key: value.unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in sample.items()
        }
        batch = move_batch_to_device(batch, device)
        view1, view2 = build_views_from_batch(batch, prefix=f"{args.split}_{sample_idx}")
        res1, _ = model(view1, view2)
        if args.param_head_type == "pixel":
            pred = res1["pred_plane_params"][0]  # [4,H,W]
            target_hw = pred.shape[-2:]
            pred_hw = pred.permute(1, 2, 0)
            feature_hw = None
        else:
            feature_map = res1["dec_feature_map"][0]
            target_hw = feature_map.shape[-2:]
            pred_hw = None
            feature_hw = feature_map.permute(1, 2, 0).contiguous()
            use_geom = args.param_head_type in (
                "geom_token",
                "geom_token_centered",
                "geom_token_conf",
                "geom_token_point_anchor",
                "geom_token_point_anchor_conf",
            )
            pts3d = get_pts3d_from_res(res1)[0] if use_geom else None
            gt_plane_pts = resize_label(batch["gt_plane"], pts3d.shape[:2])[0] if use_geom else None

        gt_plane = resize_label(batch["gt_plane"], target_hw)[0]
        gt_normal = resize_map(batch["gt_plane_normal"], target_hw)[0].permute(1, 2, 0)
        gt_offset = resize_map(batch["gt_plane_offset"], target_hw)[0, 0]
        valid = resize_map(batch["gt_plane_param_valid"], target_hw)[0, 0] > 0.5

        for pid in torch.unique(gt_plane):
            pid_int = int(pid.item())
            if pid_int in (-1, 0, 255):
                continue
            mask = (gt_plane == pid) & valid
            if int(mask.sum().item()) < args.min_pixels:
                continue

            if args.param_head_type == "pixel":
                pred_plane = pred_hw[mask]
                pred_n_px = normalize(pred_plane[:, :3], dim=1)
                pred_n = normalize(pred_n_px.mean(dim=0), dim=0)
                pred_d = pred_plane[:, 3].mean()
            else:
                token = feature_hw[mask].mean(dim=0, keepdim=True)
                pred_conf = None
                if args.param_head_type in (
                    "geom_token",
                    "geom_token_centered",
                    "geom_token_conf",
                    "geom_token_point_anchor",
                    "geom_token_point_anchor_conf",
                ):
                    geom = geometry_descriptor(pts3d[gt_plane_pts == pid]).to(dtype=token.dtype).view(1, -1)
                    if args.param_head_type in ("geom_token_conf", "geom_token_point_anchor_conf"):
                        pred = model.geom_plane_token_conf_head(torch.cat([token, geom], dim=1))[0]
                        pred_conf = torch.sigmoid(pred[4])
                    else:
                        pred = model.geom_plane_token_param_head(torch.cat([token, geom], dim=1))[0]
                else:
                    pred = model.plane_token_param_head(token)[0]
                pred_n = normalize(pred[:3], dim=0)
                if args.param_head_type in ("geom_token_centered", "geom_token_point_anchor", "geom_token_point_anchor_conf"):
                    centroid = geom.view(-1)[4:7]
                    pred_d = -torch.dot(pred_n, centroid) + pred[3]
                else:
                    pred_d = pred[3]
            gt_n_plane = gt_normal[mask]
            gt_d_plane = gt_offset[mask]

            gt_n = normalize(gt_n_plane.mean(dim=0), dim=0)
            if args.param_head_type in ("geom_token_point_anchor", "geom_token_point_anchor_conf"):
                centroid = geom.view(-1)[4:7]
                gt_d = -torch.dot(gt_n.to(dtype=centroid.dtype), centroid)
            else:
                gt_d = gt_d_plane.mean()

            sign = torch.where((pred_n * gt_n).sum() < 0, -1.0, 1.0)
            pred_n = pred_n * sign
            pred_d = pred_d * sign

            rows.append(
                {
                    "sample_idx": sample_idx,
                    "plane_id": pid_int,
                    "pixels": int(mask.sum().item()),
                    "angle_deg": float(angle_deg(pred_n, gt_n).cpu().item()),
                    "offset_abs_error": float(torch.abs(pred_d - gt_d).cpu().item()),
                    "pred_nx": float(pred_n[0].cpu().item()),
                    "pred_ny": float(pred_n[1].cpu().item()),
                    "pred_nz": float(pred_n[2].cpu().item()),
                    "pred_offset": float(pred_d.cpu().item()),
                    "gt_nx": float(gt_n[0].cpu().item()),
                    "gt_ny": float(gt_n[1].cpu().item()),
                    "gt_nz": float(gt_n[2].cpu().item()),
                    "gt_offset": float(gt_d.cpu().item()),
                    "pred_confidence": float(pred_conf.cpu().item()) if pred_conf is not None else "",
                }
            )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        mean_angle = sum(r["angle_deg"] for r in rows) / len(rows)
        mean_offset = sum(r["offset_abs_error"] for r in rows) / len(rows)
        print(f"rows={len(rows)} mean_angle_deg={mean_angle:.4f} mean_offset_abs_error={mean_offset:.4f}")
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
