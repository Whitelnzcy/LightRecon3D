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


def resize_rgb_np(rgb, size):
    x = torch.from_numpy(rgb).float().permute(2, 0, 1)[None]
    y = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return np.clip(y[0].permute(1, 2, 0).numpy(), 0, 1)


def img_to_np(img_chw):
    img = img_chw.detach().float().cpu().permute(1, 2, 0).numpy()
    lo = np.percentile(img, 1)
    hi = np.percentile(img, 99)
    img = (img - lo) / max(hi - lo, 1e-8)
    return np.clip(img, 0, 1)


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


def deterministic_sample(x, max_points):
    n = x.shape[0]
    if max_points <= 0 or n <= max_points:
        return x
    idx = torch.linspace(0, n - 1, steps=max_points, device=x.device).long()
    return x[idx]


def plane_pca_metrics(points, max_points=50000, max_abs_coord=1e4, eps=1e-8):
    if points.ndim != 2 or points.shape[-1] != 3:
        return None

    finite = torch.isfinite(points).all(dim=1)
    finite &= torch.amax(torch.abs(points), dim=1) < max_abs_coord
    points = points[finite]

    if points.shape[0] < 20:
        return None

    points_s = deterministic_sample(points, max_points)

    center = points_s.mean(dim=0, keepdim=True)
    x = points_s - center

    cov = x.t().matmul(x) / max(points_s.shape[0], 1)
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
    point_norm = torch.linalg.norm(points_s.float(), dim=1)

    return {
        "num_valid_points": int(points_s.shape[0]),
        "center": center[0],
        "normal": normal,
        "flatness": float((eigvals[0] / eig_sum).item()),
        "extent": float((eigvals[1] + eigvals[2]).item()),
        "eig0": float(eigvals[0].item()),
        "eig1": float(eigvals[1].item()),
        "eig2": float(eigvals[2].item()),
        "median_abs_dist": float(dist.median().item()),
        "mean_abs_dist": float(dist.mean().item()),
        "p90_abs_dist": float(torch.quantile(dist, 0.90).item()),
        "p95_abs_dist": float(torch.quantile(dist, 0.95).item()),
        "point_norm_mean": float(point_norm.mean().item()),
        "point_norm_median": float(point_norm.median().item()),
    }


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


def depth_jump(pts_hw3, quantile=0.95):
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


def scalar_stats(arr, mask, low_thr=None, high_thr=None):
    if arr is None:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "low_ratio": float("nan"),
            "high_ratio": float("nan"),
        }

    vals = arr[mask]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "low_ratio": float("nan"),
            "high_ratio": float("nan"),
        }

    if low_thr is None:
        low_ratio = float("nan")
    else:
        low_ratio = float((vals < low_thr).mean())

    if high_thr is None:
        high_ratio = float("nan")
    else:
        high_ratio = float((vals > high_thr).mean())

    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "low_ratio": low_ratio,
        "high_ratio": high_ratio,
    }


def save_gray(path, arr):
    arr = np.asarray(arr)
    if arr.dtype == bool:
        arr = arr.astype(np.float32)

    finite = np.isfinite(arr)
    if finite.any():
        lo = np.percentile(arr[finite], 1)
        hi = np.percentile(arr[finite], 99)
        arr = (arr - lo) / max(hi - lo, 1e-8)
    arr = np.clip(arr, 0, 1)

    plt.imsave(path, arr, cmap="gray")


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


def write_ply(path, pts_hw3, rgb_hwc, max_points=250000):
    pts = pts_hw3.detach().float().cpu().numpy()
    h, w, _ = pts.shape

    if rgb_hwc.shape[:2] != (h, w):
        rgb_hwc = resize_rgb_np(rgb_hwc, (h, w))

    rgb_uint8 = (np.clip(rgb_hwc, 0, 1) * 255).astype(np.uint8)

    pts_flat = pts.reshape(-1, 3)
    rgb_flat = rgb_uint8.reshape(-1, 3)

    finite = np.isfinite(pts_flat).all(axis=1)
    finite &= np.max(np.abs(pts_flat), axis=1) < 1e4

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


@torch.no_grad()
def audit_one_view(
    res,
    img_chw,
    gt_plane_bhw,
    sample_idx,
    view_idx,
    args,
    output_dir=None,
):
    pts = get_pts3d(res)[0]
    h, w = pts.shape[:2]

    gt_plane = resize_label_nearest(gt_plane_bhw, (h, w))[0]
    gt_np = gt_plane.detach().cpu().numpy().astype(np.int64)

    conf, conf_key = get_conf(res)
    conf_low = resize_scalar(conf, (h, w))
    conf_np = None if conf_low is None else conf_low[0].detach().cpu().numpy()

    rgb = img_to_np(img_chw)
    rgb = resize_rgb_np(rgb, (h, w))

    boundary = plane_boundary_mask(gt_np)
    jump_map, jump_mask, jump_thr = depth_jump(pts, quantile=args.depth_jump_quantile)

    ids, counts = np.unique(gt_np, return_counts=True)

    rows = []
    flatness_map = np.zeros((h, w), dtype=np.float32)

    for pid, count in zip(ids.tolist(), counts.tolist()):
        pid = int(pid)
        count = int(count)

        if pid < 0 or pid == 255:
            continue
        if count < args.min_plane_pixels:
            continue

        mask = gt_np == pid

        metrics = plane_pca_metrics(
            pts[torch.from_numpy(mask).to(pts.device, dtype=torch.bool)],
            max_points=args.max_points_per_plane,
        )

        if metrics is None:
            continue

        conf_stats = scalar_stats(conf_np, mask, low_thr=args.conf_low_thr)
        boundary_ratio = float(boundary[mask].mean()) if mask.sum() > 0 else float("nan")
        depth_jump_ratio = float(jump_mask[mask].mean()) if mask.sum() > 0 else float("nan")

        flatness_map[mask] = metrics["flatness"]

        row = {
            "sample_idx": sample_idx,
            "view_idx": view_idx,
            "plane_id": pid,
            "num_pixels": count,
            "num_valid_points": metrics["num_valid_points"],
            "flatness": metrics["flatness"],
            "extent": metrics["extent"],
            "eig0": metrics["eig0"],
            "eig1": metrics["eig1"],
            "eig2": metrics["eig2"],
            "median_abs_dist": metrics["median_abs_dist"],
            "mean_abs_dist": metrics["mean_abs_dist"],
            "p90_abs_dist": metrics["p90_abs_dist"],
            "p95_abs_dist": metrics["p95_abs_dist"],
            "point_norm_mean": metrics["point_norm_mean"],
            "point_norm_median": metrics["point_norm_median"],
            "conf_mean": conf_stats["mean"],
            "conf_median": conf_stats["median"],
            "conf_min": conf_stats["min"],
            "conf_max": conf_stats["max"],
            "conf_low_ratio": conf_stats["low_ratio"],
            "plane_boundary_ratio": boundary_ratio,
            "depth_jump_ratio": depth_jump_ratio,
            "depth_jump_thr": jump_thr,
            "conf_key": conf_key if conf_key is not None else "",
        }

        rows.append(row)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = f"sample{sample_idx:04d}_view{view_idx}"

        gt_color = colorize_label(gt_np)

        save_panel(
            out / f"{prefix}_audit_panel.png",
            title=prefix,
            panels=[
                ("RGB", rgb, None),
                ("GT Plane", gt_color, None),
                ("Plane Boundary", boundary.astype(float), "gray"),
                ("Depth Jump", jump_mask.astype(float), "gray"),
                ("Confidence", conf_np if conf_np is not None else np.zeros((h, w)), "gray"),
                ("Flatness per GT plane", flatness_map, "magma"),
            ],
        )

        save_gray(out / f"{prefix}_flatness_map.png", flatness_map)
        save_gray(out / f"{prefix}_depth_jump.png", jump_map)
        if conf_np is not None:
            save_gray(out / f"{prefix}_confidence.png", conf_np)

        if args.export_ply:
            write_ply(out / f"{prefix}_raw_dust3r.ply", pts, rgb, max_points=args.max_export_points)

    return rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Audit raw DUSt3R baseline failure modes")

    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)

    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)

    parser.add_argument("--input_mode", default="pair", choices=["single", "pair"])
    parser.add_argument("--pair_strategy", default="adjacent")
    parser.add_argument("--eval_views", default="both", choices=["view1", "view2", "both"])

    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)

    parser.add_argument("--min_plane_pixels", type=int, default=500)
    parser.add_argument("--max_points_per_plane", type=int, default=50000)
    parser.add_argument("--depth_jump_quantile", type=float, default=0.95)
    parser.add_argument("--conf_low_thr", type=float, default=1.5)

    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--save_worst_dir", default=None)
    parser.add_argument("--save_worst_topk", type=int, default=20)
    parser.add_argument("--export_ply", action="store_true")
    parser.add_argument("--max_export_points", type=int, default=250000)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Raw DUSt3R Baseline Audit")
    print("=" * 80)
    print(f"device      : {device}")
    print(f"split       : {args.split}")
    print(f"num_samples : {args.num_samples}")
    print(f"input_mode  : {args.input_mode}")
    print(f"eval_views  : {args.eval_views}")
    print(f"output_csv  : {args.output_csv}")
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

    n = min(args.num_samples, len(ds))
    all_rows = []

    for sample_idx in range(n):
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

        view1, view2 = build_views_from_batch(batch, prefix=f"raw_audit_{args.split}_{sample_idx}")
        res1, res2 = model(view1, view2)

        if args.eval_views in ["view1", "both"]:
            rows1 = audit_one_view(
                res=res1,
                img_chw=batch["img1"][0] if "img1" in batch else batch["img"][0],
                gt_plane_bhw=batch["gt_plane1"] if "gt_plane1" in batch else batch["gt_plane"],
                sample_idx=sample_idx,
                view_idx=1,
                args=args,
                output_dir=None,
            )
            all_rows.extend(rows1)

        if args.eval_views in ["view2", "both"]:
            rows2 = audit_one_view(
                res=res2,
                img_chw=batch["img2"][0] if "img2" in batch else batch["img"][0],
                gt_plane_bhw=batch["gt_plane2"] if "gt_plane2" in batch else batch["gt_plane"],
                sample_idx=sample_idx,
                view_idx=2,
                args=args,
                output_dir=None,
            )
            all_rows.extend(rows2)

        if (sample_idx + 1) % 10 == 0 or sample_idx == 0:
            print(f"[{sample_idx+1}/{n}] rows={len(all_rows)}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_idx",
        "view_idx",
        "plane_id",
        "num_pixels",
        "num_valid_points",
        "flatness",
        "extent",
        "eig0",
        "eig1",
        "eig2",
        "median_abs_dist",
        "mean_abs_dist",
        "p90_abs_dist",
        "p95_abs_dist",
        "point_norm_mean",
        "point_norm_median",
        "conf_mean",
        "conf_median",
        "conf_min",
        "conf_max",
        "conf_low_ratio",
        "plane_boundary_ratio",
        "depth_jump_ratio",
        "depth_jump_thr",
        "conf_key",
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
        flats = np.array([r["flatness"] for r in all_rows], dtype=np.float64)
        p90s = np.array([r["p90_abs_dist"] for r in all_rows], dtype=np.float64)
        confs = np.array([r["conf_mean"] for r in all_rows], dtype=np.float64)

        print(f"flatness mean   : {np.nanmean(flats):.8f}")
        print(f"flatness median : {np.nanmedian(flats):.8f}")
        print(f"flatness max    : {np.nanmax(flats):.8f}")
        print(f"p90 dist mean   : {np.nanmean(p90s):.8f}")
        print(f"p90 dist median : {np.nanmedian(p90s):.8f}")
        print(f"conf mean       : {np.nanmean(confs):.8f}")

    # Save worst visualizations after CSV is written.
    if args.save_worst_dir is not None and all_rows:
        ranked = sorted(
            all_rows,
            key=lambda r: (
                float(r["flatness"]),
                float(r["p90_abs_dist"]),
                int(r["num_pixels"]),
            ),
            reverse=True,
        )

        targets = []
        seen = set()
        for r in ranked:
            key = (int(r["sample_idx"]), int(r["view_idx"]))
            if key in seen:
                continue
            seen.add(key)
            targets.append(key)
            if len(targets) >= args.save_worst_topk:
                break

        print("=" * 80)
        print("Saving worst visualizations")
        print("=" * 80)
        print("targets:", targets)

        for sample_idx, view_idx in targets:
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

            view1, view2 = build_views_from_batch(batch, prefix=f"raw_audit_worst_{sample_idx}")
            res1, res2 = model(view1, view2)

            if view_idx == 1:
                audit_one_view(
                    res=res1,
                    img_chw=batch["img1"][0] if "img1" in batch else batch["img"][0],
                    gt_plane_bhw=batch["gt_plane1"] if "gt_plane1" in batch else batch["gt_plane"],
                    sample_idx=sample_idx,
                    view_idx=1,
                    args=args,
                    output_dir=args.save_worst_dir,
                )
            else:
                audit_one_view(
                    res=res2,
                    img_chw=batch["img2"][0] if "img2" in batch else batch["img"][0],
                    gt_plane_bhw=batch["gt_plane2"] if "gt_plane2" in batch else batch["gt_plane"],
                    sample_idx=sample_idx,
                    view_idx=2,
                    args=args,
                    output_dir=args.save_worst_dir,
                )

    print("Done.")


if __name__ == "__main__":
    main()