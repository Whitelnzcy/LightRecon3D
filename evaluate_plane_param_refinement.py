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


def clean_points(points):
    finite = torch.isfinite(points).all(dim=1)
    points = points[finite]
    coord_ok = points.abs().amax(dim=1) < 1e4
    return points[coord_ok]


def fit_plane_svd(points, eps=1e-6):
    points = clean_points(points)
    if points.shape[0] < 3:
        return None
    centroid = points.mean(dim=0)
    x = points - centroid
    cov = x.transpose(0, 1) @ x / max(1, points.shape[0] - 1)
    cov = 0.5 * (cov + cov.transpose(0, 1))
    cov = cov + eps * torch.eye(3, device=points.device, dtype=points.dtype)
    eigvals, eigvecs = torch.linalg.eigh(cov.float())
    normal = normalize(eigvecs[:, 0].to(points.dtype), dim=0)
    offset = -torch.dot(normal, centroid)
    return normal, offset


def align_to_ref(normal, offset, ref_normal):
    sign = torch.where((normal * ref_normal).sum() < 0, -1.0, 1.0)
    return normal * sign, offset * sign


def geometry_descriptor(points):
    fit = fit_plane_svd(points)
    if fit is None:
        return torch.zeros(10, device=points.device, dtype=points.dtype)
    normal, offset = fit
    points = clean_points(points)
    centroid = points.mean(dim=0)
    x = points - centroid
    cov = x.transpose(0, 1) @ x / max(1, points.shape[0] - 1)
    cov = 0.5 * (cov + cov.transpose(0, 1))
    cov = cov + 1e-6 * torch.eye(3, device=points.device, dtype=points.dtype)
    eigvals, _ = torch.linalg.eigh(cov.float())
    eigvals = eigvals.to(points.dtype).clamp_min(0.0)

    idx = torch.argmax(normal.abs())
    sign = torch.where(normal[idx] < 0, -1.0, 1.0)
    normal = normal * sign
    offset = offset * sign

    scales = torch.sqrt(torch.flip(eigvals, dims=[0]).clamp_min(0.0))
    desc = torch.cat([normal, offset.view(1), centroid, scales], dim=0)
    return torch.nan_to_num(desc, nan=0.0, posinf=1e4, neginf=-1e4)


def trimmed_refine(points, init_normal, init_offset, keep_ratio=0.7, steps=3):
    points = clean_points(points)
    if points.shape[0] < 8:
        return init_normal, init_offset, points.shape[0]
    normal = init_normal
    offset = init_offset
    keep_n = max(8, int(points.shape[0] * keep_ratio))
    keep_n = min(keep_n, points.shape[0])
    last_count = points.shape[0]
    for _ in range(steps):
        dist = torch.abs(points @ normal + offset)
        order = torch.argsort(dist)
        inliers = points[order[:keep_n]]
        fit = fit_plane_svd(inliers)
        if fit is None:
            break
        normal, offset = align_to_ref(fit[0], fit[1], normal)
        last_count = int(inliers.shape[0])
    return normal, offset, last_count


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--plane_offset_scale", type=float, default=1000.0)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)
    parser.add_argument("--param_head_type", type=str, default="geom_token", choices=["token", "geom_token"])
    parser.add_argument("--min_pixels", type=int, default=64)
    parser.add_argument("--keep_ratio", type=float, default=0.7)
    parser.add_argument("--refine_steps", type=int, default=3)
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
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}
        view1, view2 = build_views_from_batch(batch, prefix=f"{args.split}_{sample_idx}")
        res1, _ = model(view1, view2)

        feature_map = res1["dec_feature_map"][0]
        target_hw = feature_map.shape[-2:]
        feature_hw = feature_map.permute(1, 2, 0).contiguous()
        pts3d = get_pts3d_from_res(res1)[0]
        gt_plane_pts = resize_label(batch["gt_plane"], pts3d.shape[:2])[0]

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

            token = feature_hw[mask].mean(dim=0, keepdim=True)
            plane_points = pts3d[gt_plane_pts == pid]
            if args.param_head_type == "geom_token":
                geom = geometry_descriptor(plane_points).to(dtype=token.dtype).view(1, -1)
                pred = model.geom_plane_token_param_head(torch.cat([token, geom], dim=1))[0]
            else:
                pred = model.plane_token_param_head(token)[0]

            pred_n = normalize(pred[:3], dim=0)
            pred_d = pred[3]
            gt_n = normalize(gt_normal[mask].mean(dim=0), dim=0)
            gt_d = gt_offset[mask].mean()
            pred_n, pred_d = align_to_ref(pred_n, pred_d, gt_n)

            svd_fit = fit_plane_svd(plane_points)
            if svd_fit is None:
                svd_n, svd_d = pred_n, pred_d
            else:
                svd_n, svd_d = align_to_ref(svd_fit[0], svd_fit[1], gt_n)

            ref_n, ref_d, inliers = trimmed_refine(
                plane_points,
                pred_n,
                pred_d,
                keep_ratio=args.keep_ratio,
                steps=args.refine_steps,
            )
            ref_n, ref_d = align_to_ref(ref_n, ref_d, gt_n)

            candidates = {
                "pred": (pred_n, pred_d),
                "svd_all": (svd_n, svd_d),
                "trimmed_refine": (ref_n, ref_d),
            }
            for method, (n, d) in candidates.items():
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "plane_id": pid_int,
                        "method": method,
                        "pixels": int(mask.sum().item()),
                        "inlier_points": int(inliers if method == "trimmed_refine" else clean_points(plane_points).shape[0]),
                        "angle_deg": float(angle_deg(n, gt_n).cpu().item()),
                        "offset_abs_error": float(torch.abs(d - gt_d).cpu().item()),
                        "pred_nx": float(n[0].cpu().item()),
                        "pred_ny": float(n[1].cpu().item()),
                        "pred_nz": float(n[2].cpu().item()),
                        "pred_offset": float(d.cpu().item()),
                        "gt_nx": float(gt_n[0].cpu().item()),
                        "gt_ny": float(gt_n[1].cpu().item()),
                        "gt_nz": float(gt_n[2].cpu().item()),
                        "gt_offset": float(gt_d.cpu().item()),
                    }
                )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    for method in ["pred", "svd_all", "trimmed_refine"]:
        vals = [r for r in rows if r["method"] == method]
        if not vals:
            continue
        mean_angle = sum(r["angle_deg"] for r in vals) / len(vals)
        mean_offset = sum(r["offset_abs_error"] for r in vals) / len(vals)
        print(f"{method}: rows={len(vals)} mean_angle_deg={mean_angle:.4f} mean_offset_abs_error={mean_offset:.4f}")
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
