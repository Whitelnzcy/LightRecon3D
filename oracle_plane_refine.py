import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


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


def build_raw_model(weights_path, device, hidden_dim=768, plane_embed_dim=16):
    backbone = build_dust3r_backbone(weights_path, device=device)
    model = LightReconModel(
        dust3r_backbone=backbone,
        hidden_dim=hidden_dim,
        plane_embed_dim=plane_embed_dim,
    ).to(device)
    model.eval()
    return model


def get_pts3d(res):
    for key in ["pts3d", "pts3d_in_other_view", "pointmap", "pred_pts3d"]:
        if key in res:
            pts = res[key]
            break
    else:
        raise KeyError(f"No pointmap found. keys={list(res.keys())}")

    if pts.ndim != 4:
        raise ValueError(f"Bad pts shape: {tuple(pts.shape)}")

    if pts.shape[1] == 3:
        pts = pts.permute(0, 2, 3, 1).contiguous()

    if pts.shape[-1] != 3:
        raise ValueError(f"Bad pts shape: {tuple(pts.shape)}")

    return pts


def get_conf(res):
    for key in ["conf", "confidence", "pred_conf", "conf_self"]:
        if key not in res:
            continue
        conf = res[key]
        if not torch.is_tensor(conf):
            continue

        if conf.ndim == 4 and conf.shape[1] == 1:
            conf = conf[:, 0]
        elif conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        elif conf.ndim == 3:
            pass
        else:
            continue

        return conf, key

    return None, None


def resize_label_nearest(x, size):
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.shape[-2:] == size:
        return x.long()
    return F.interpolate(x.unsqueeze(1).float(), size=size, mode="nearest")[:, 0].long()


def resize_scalar(x, size):
    if x is None:
        return None
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.shape[-2:] == size:
        return x.float()
    return F.interpolate(x.unsqueeze(1).float(), size=size, mode="bilinear", align_corners=False)[:, 0]


def img_to_np(img_chw):
    img = img_chw.detach().float().cpu().permute(1, 2, 0).numpy()
    lo = np.percentile(img, 1)
    hi = np.percentile(img, 99)
    img = (img - lo) / max(hi - lo, 1e-8)
    return np.clip(img, 0, 1)


def resize_rgb_np(rgb, size):
    x = torch.from_numpy(rgb).float().permute(2, 0, 1)[None]
    y = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return np.clip(y[0].permute(1, 2, 0).numpy(), 0, 1)


def colorize_label(label, ignore_ids=(-1, 255)):
    label = label.astype(np.int64)
    out = np.zeros((*label.shape, 3), dtype=np.float32)

    valid = np.ones(label.shape, dtype=bool)
    for ig in ignore_ids:
        valid &= label != ig

    ids = np.unique(label[valid])
    rng = np.random.default_rng(12345)
    colors = rng.uniform(0.15, 0.95, size=(max(len(ids), 1), 3)).astype(np.float32)

    for i, pid in enumerate(ids):
        out[label == pid] = colors[i % len(colors)]

    return out


def plane_boundary_mask(plane_hw_np):
    p = plane_hw_np.astype(np.int64)
    valid = (p >= 0) & (p != 255)

    b = np.zeros_like(valid, dtype=bool)

    diff_x = (p[:, 1:] != p[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    b[:, 1:] |= diff_x
    b[:, :-1] |= diff_x

    diff_y = (p[1:, :] != p[:-1, :]) & valid[1:, :] & valid[:-1, :]
    b[1:, :] |= diff_y
    b[:-1, :] |= diff_y

    return b


def dilate_mask_np(mask, radius):
    if radius <= 0:
        return mask.astype(bool)
    x = torch.from_numpy(mask.astype(np.float32))[None, None]
    y = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y[0, 0].numpy() > 0.5


def erode_mask_np(mask, radius):
    if radius <= 0:
        return mask.astype(bool)
    x = torch.from_numpy(mask.astype(np.float32))[None, None]
    y = -F.max_pool2d(-x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y[0, 0].numpy() > 0.5


def depth_jump_mask(pts_hw3, quantile=0.95):
    pts = pts_hw3.detach().float()
    h, w, _ = pts.shape
    finite = torch.isfinite(pts).all(dim=-1)

    jump = torch.zeros((h, w), device=pts.device, dtype=torch.float32)

    dx = torch.linalg.norm(pts[:, 1:] - pts[:, :-1], dim=-1)
    vx = finite[:, 1:] & finite[:, :-1]
    dx = torch.where(vx, dx, torch.zeros_like(dx))
    jump[:, 1:] = torch.maximum(jump[:, 1:], dx)
    jump[:, :-1] = torch.maximum(jump[:, :-1], dx)

    dy = torch.linalg.norm(pts[1:, :] - pts[:-1, :], dim=-1)
    vy = finite[1:, :] & finite[:-1, :]
    dy = torch.where(vy, dy, torch.zeros_like(dy))
    jump[1:, :] = torch.maximum(jump[1:, :], dy)
    jump[:-1, :] = torch.maximum(jump[:-1, :], dy)

    vals = jump[finite]
    vals = vals[torch.isfinite(vals)]

    if vals.numel() == 0:
        thr = 0.0
    else:
        thr = torch.quantile(vals, quantile).item()

    mask = (jump >= thr) & finite

    return jump.detach().cpu().numpy(), mask.detach().cpu().numpy().astype(bool), float(thr)


def deterministic_sample(points, max_points):
    n = points.shape[0]
    if max_points <= 0 or n <= max_points:
        return points
    idx = torch.linspace(0, n - 1, steps=max_points, device=points.device).long()
    return points[idx]


def fit_plane_pca(points, max_points=50000, max_abs_coord=1e4, eps=1e-8):
    if points.ndim != 2 or points.shape[-1] != 3:
        return None

    finite = torch.isfinite(points).all(dim=1)
    finite &= torch.amax(torch.abs(points), dim=1) < max_abs_coord
    points = points[finite]

    if points.shape[0] < 20:
        return None

    pts = deterministic_sample(points, max_points)
    center = pts.mean(dim=0, keepdim=True)
    x = pts - center

    cov = x.t().matmul(x) / max(pts.shape[0], 1)
    cov = 0.5 * (cov + cov.t())

    try:
        eigvals, eigvecs = torch.linalg.eigh(cov.float())
    except RuntimeError:
        return None

    eigvals = eigvals.clamp_min(0)
    eig_sum = eigvals.sum().clamp_min(eps)

    normal = eigvecs[:, 0]
    normal = normal / normal.norm().clamp_min(eps)

    dist = (x.float() @ normal).abs()

    return {
        "center": center[0].float(),
        "normal": normal.float(),
        "eigvals": eigvals.float(),
        "flatness": float((eigvals[0] / eig_sum).item()),
        "median_abs_dist": float(dist.median().item()),
        "mean_abs_dist": float(dist.mean().item()),
        "p90_abs_dist": float(torch.quantile(dist, 0.90).item()),
        "p95_abs_dist": float(torch.quantile(dist, 0.95).item()),
        "num_points": int(pts.shape[0]),
    }


def robust_fit_plane(points, max_points=50000, trim_quantile=0.90):
    first = fit_plane_pca(points, max_points=max_points)
    if first is None:
        return None, None

    c = first["center"]
    n = first["normal"]

    dist = ((points.float() - c[None, :]) @ n).abs()
    finite = torch.isfinite(dist)

    if finite.sum() < 20:
        return first, finite

    thr = torch.quantile(dist[finite], trim_quantile)
    inlier = finite & (dist <= thr)

    if inlier.sum() < 20:
        return first, finite

    second = fit_plane_pca(points[inlier], max_points=max_points)
    if second is None:
        return first, inlier

    return second, inlier


def plane_metrics(points):
    fit = fit_plane_pca(points, max_points=50000)
    if fit is None:
        return {
            "flatness": float("nan"),
            "median_abs_dist": float("nan"),
            "mean_abs_dist": float("nan"),
            "p90_abs_dist": float("nan"),
            "p95_abs_dist": float("nan"),
            "num_points": 0,
        }
    return {
        "flatness": fit["flatness"],
        "median_abs_dist": fit["median_abs_dist"],
        "mean_abs_dist": fit["mean_abs_dist"],
        "p90_abs_dist": fit["p90_abs_dist"],
        "p95_abs_dist": fit["p95_abs_dist"],
        "num_points": fit["num_points"],
    }


def project_points_to_plane(points, center, normal, alpha=1.0):
    signed = (points.float() - center[None, :]) @ normal
    projected = points.float() - signed[:, None] * normal[None, :]
    return (1.0 - alpha) * points.float() + alpha * projected


def write_ply(path, pts_hw3, rgb_hwc, mask=None, max_points=250000):
    pts = pts_hw3.detach().float().cpu().numpy()
    h, w, _ = pts.shape

    if rgb_hwc.shape[:2] != (h, w):
        rgb_hwc = resize_rgb_np(rgb_hwc, (h, w))

    rgb_uint8 = (np.clip(rgb_hwc, 0, 1) * 255).astype(np.uint8)

    pts_flat = pts.reshape(-1, 3)
    rgb_flat = rgb_uint8.reshape(-1, 3)

    finite = np.isfinite(pts_flat).all(axis=1)
    finite &= np.max(np.abs(pts_flat), axis=1) < 1e4

    if mask is not None:
        finite &= mask.reshape(-1).astype(bool)

    pts_flat = pts_flat[finite]
    rgb_flat = rgb_flat[finite]

    if max_points > 0 and pts_flat.shape[0] > max_points:
        idx = np.linspace(0, pts_flat.shape[0] - 1, max_points).astype(np.int64)
        pts_flat = pts_flat[idx]
        rgb_flat = rgb_flat[idx]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts_flat.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(pts_flat, rgb_flat):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def save_panel(path, title, panels):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, img, cmap) in zip(axes, panels):
        ax.set_title(name)
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def refine_one_view(res, img_chw, gt_plane_bhw, sample_idx, view_idx, args, output_dir):
    pts = get_pts3d(res)[0]
    h, w = pts.shape[:2]

    pts_refined = pts.clone()

    gt_plane = resize_label_nearest(gt_plane_bhw, (h, w))[0]
    gt_np = gt_plane.detach().cpu().numpy().astype(np.int64)

    conf, conf_key = get_conf(res)
    conf_map = resize_scalar(conf, (h, w))
    if conf_map is None:
        conf_np = np.ones((h, w), dtype=np.float32) * 999.0
    else:
        conf_np = conf_map[0].detach().cpu().numpy().astype(np.float32)

    rgb = img_to_np(img_chw)
    rgb = resize_rgb_np(rgb, (h, w))

    plane_boundary = plane_boundary_mask(gt_np)
    boundary_band = dilate_mask_np(plane_boundary, args.boundary_dilate)

    jump_map, jump_mask, jump_thr = depth_jump_mask(pts, quantile=args.depth_jump_quantile)
    jump_band = dilate_mask_np(jump_mask, args.depth_dilate)

    finite_np = torch.isfinite(pts).all(dim=-1).detach().cpu().numpy()
    coord_ok_np = (torch.amax(torch.abs(pts), dim=-1) < args.max_abs_coord).detach().cpu().numpy()

    valid_3d = finite_np & coord_ok_np
    high_conf = conf_np >= args.conf_thr

    ids, counts = np.unique(gt_np, return_counts=True)

    refined_mask_total = np.zeros((h, w), dtype=bool)
    core_mask_total = np.zeros((h, w), dtype=bool)
    plane_core_color = np.zeros((h, w), dtype=np.float32)
    rows = []

    for pid, count in zip(ids.tolist(), counts.tolist()):
        pid = int(pid)
        count = int(count)

        if pid < 0 or pid == 255:
            continue
        if count < args.min_plane_pixels:
            continue

        plane_mask = gt_np == pid

        # core mask: conservative reliable region
        core = plane_mask.copy()
        core &= valid_3d
        core &= high_conf
        core &= ~boundary_band
        core &= ~jump_band

        # erode plane region to avoid thin strips
        if args.plane_erode > 0:
            core &= erode_mask_np(plane_mask, args.plane_erode)

        if core.sum() < args.min_core_pixels:
            continue

        core_t = torch.from_numpy(core).to(pts.device, dtype=torch.bool)
        plane_t = torch.from_numpy(plane_mask & valid_3d).to(pts.device, dtype=torch.bool)

        core_points = pts[core_t]
        plane_points_before = pts[plane_t]
        core_points_before = pts[core_t]

        before_core = plane_metrics(core_points_before)
        before_full = plane_metrics(plane_points_before)

        fit, inlier = robust_fit_plane(
            core_points,
            max_points=args.max_points_per_plane,
            trim_quantile=args.trim_quantile,
        )

        if fit is None:
            continue

        # only refine core region, not boundary/window/jump areas
        projected = project_points_to_plane(
            core_points,
            center=fit["center"],
            normal=fit["normal"],
            alpha=args.alpha,
        )

        pts_refined[core_t] = projected.to(pts_refined.dtype)

        plane_points_after = pts_refined[plane_t]
        core_points_after = pts_refined[core_t]

        after_core = plane_metrics(core_points_after)
        after_full = plane_metrics(plane_points_after)

        refined_mask_total |= core
        core_mask_total |= core
        plane_core_color[core] = float(pid + 1)

        rows.append({
            "sample_idx": sample_idx,
            "view_idx": view_idx,
            "plane_id": pid,
            "plane_pixels": int(plane_mask.sum()),
            "core_pixels": int(core.sum()),
            "core_ratio": float(core.sum() / max(plane_mask.sum(), 1)),
            "conf_mean_core": float(conf_np[core].mean()) if core.sum() > 0 else float("nan"),
            "depth_jump_thr": float(jump_thr),
            "before_core_flatness": before_core["flatness"],
            "after_core_flatness": after_core["flatness"],
            "delta_core_flatness": after_core["flatness"] - before_core["flatness"],
            "before_core_p90": before_core["p90_abs_dist"],
            "after_core_p90": after_core["p90_abs_dist"],
            "delta_core_p90": after_core["p90_abs_dist"] - before_core["p90_abs_dist"],
            "before_full_flatness": before_full["flatness"],
            "after_full_flatness": after_full["flatness"],
            "delta_full_flatness": after_full["flatness"] - before_full["flatness"],
            "before_full_p90": before_full["p90_abs_dist"],
            "after_full_p90": after_full["p90_abs_dist"],
            "delta_full_p90": after_full["p90_abs_dist"] - before_full["p90_abs_dist"],
        })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prefix = f"sample{sample_idx:04d}_view{view_idx}"

    save_panel(
        out / f"{prefix}_oracle_refine_panel.png",
        title=prefix,
        panels=[
            ("RGB", rgb, None),
            ("GT Plane", colorize_label(gt_np), None),
            ("Plane Boundary Band", boundary_band.astype(float), "gray"),
            ("Depth Jump Band", jump_band.astype(float), "gray"),
            ("Confidence", conf_np, "gray"),
            ("Oracle Core Mask", core_mask_total.astype(float), "gray"),
        ],
    )

    if args.export_ply:
        write_ply(out / f"{prefix}_before_raw.ply", pts, rgb, max_points=args.max_export_points)
        write_ply(out / f"{prefix}_after_oracle_refine.ply", pts_refined, rgb, max_points=args.max_export_points)
        write_ply(out / f"{prefix}_core_before.ply", pts, rgb, mask=core_mask_total, max_points=args.max_export_points)
        write_ply(out / f"{prefix}_core_after.ply", pts_refined, rgb, mask=core_mask_total, max_points=args.max_export_points)

    return rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Oracle plane-core refinement for raw DUSt3R")

    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)

    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--sample_indices", type=int, nargs="+", required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)

    parser.add_argument("--input_mode", default="pair", choices=["single", "pair"])
    parser.add_argument("--pair_strategy", default="adjacent")
    parser.add_argument("--eval_views", default="both", choices=["view1", "view2", "both"])

    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)

    parser.add_argument("--min_plane_pixels", type=int, default=500)
    parser.add_argument("--min_core_pixels", type=int, default=1000)
    parser.add_argument("--max_points_per_plane", type=int, default=50000)
    parser.add_argument("--max_abs_coord", type=float, default=1e4)

    parser.add_argument("--boundary_dilate", type=int, default=8)
    parser.add_argument("--depth_dilate", type=int, default=8)
    parser.add_argument("--plane_erode", type=int, default=4)
    parser.add_argument("--depth_jump_quantile", type=float, default=0.95)
    parser.add_argument("--conf_thr", type=float, default=1.5)

    parser.add_argument("--trim_quantile", type=float, default=0.90)
    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--export_ply", action="store_true")
    parser.add_argument("--max_export_points", type=int, default=250000)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Oracle Plane-Core Refinement")
    print("=" * 80)
    print(f"device         : {device}")
    print(f"split          : {args.split}")
    print(f"sample_indices : {args.sample_indices}")
    print(f"eval_views     : {args.eval_views}")
    print(f"output_dir     : {args.output_dir}")
    print(f"output_csv     : {args.output_csv}")
    print("=" * 80)

    ds = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        input_mode=args.input_mode,
        pair_strategy=args.pair_strategy,
    )

    model = build_raw_model(
        weights_path=args.weights_path,
        device=device,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    )

    all_rows = []

    for sample_idx in args.sample_indices:
        sample = ds[sample_idx]

        batch = {}
        for k, v in sample.items():
            if torch.is_tensor(v):
                batch[k] = v.unsqueeze(0).to(device)

        if "img" not in batch:
            batch["img"] = batch["img1"]
        if "gt_line" not in batch:
            batch["gt_line"] = batch["gt_line1"]
        if "gt_plane" not in batch:
            batch["gt_plane"] = batch["gt_plane1"]

        view1, view2 = build_views_from_batch(batch, prefix=f"oracle_refine_{sample_idx}")
        res1, res2 = model(view1, view2)

        if args.eval_views in ["view1", "both"]:
            rows1 = refine_one_view(
                res=res1,
                img_chw=batch["img1"][0] if "img1" in batch else batch["img"][0],
                gt_plane_bhw=batch["gt_plane1"] if "gt_plane1" in batch else batch["gt_plane"],
                sample_idx=sample_idx,
                view_idx=1,
                args=args,
                output_dir=args.output_dir,
            )
            all_rows.extend(rows1)

        if args.eval_views in ["view2", "both"]:
            rows2 = refine_one_view(
                res=res2,
                img_chw=batch["img2"][0] if "img2" in batch else batch["img"][0],
                gt_plane_bhw=batch["gt_plane2"] if "gt_plane2" in batch else batch["gt_plane"],
                sample_idx=sample_idx,
                view_idx=2,
                args=args,
                output_dir=args.output_dir,
            )
            all_rows.extend(rows2)

        print(f"[Done] sample_idx={sample_idx}, rows={len(all_rows)}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_idx",
        "view_idx",
        "plane_id",
        "plane_pixels",
        "core_pixels",
        "core_ratio",
        "conf_mean_core",
        "depth_jump_thr",
        "before_core_flatness",
        "after_core_flatness",
        "delta_core_flatness",
        "before_core_p90",
        "after_core_p90",
        "delta_core_p90",
        "before_full_flatness",
        "after_full_flatness",
        "delta_full_flatness",
        "before_full_p90",
        "after_full_p90",
        "delta_full_p90",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"rows saved : {len(all_rows)}")
    print(f"csv        : {output_csv}")

    if all_rows:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        print("mean before_core_flatness:", df["before_core_flatness"].mean())
        print("mean after_core_flatness :", df["after_core_flatness"].mean())
        print("mean delta_core_flatness :", df["delta_core_flatness"].mean())
        print("mean before_core_p90     :", df["before_core_p90"].mean())
        print("mean after_core_p90      :", df["after_core_p90"].mean())
        print("mean delta_core_p90      :", df["delta_core_p90"].mean())
        print("mean core_ratio          :", df["core_ratio"].mean())

    print("Done.")


if __name__ == "__main__":
    main()