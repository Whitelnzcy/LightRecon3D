import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# Path setup
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")

if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

if DUST3R_REPO_ROOT in sys.path:
    sys.path.remove(DUST3R_REPO_ROOT)
sys.path.insert(1, DUST3R_REPO_ROOT)

from dataloaders.s3d_dataset import Structured3DDataset

from dust3r.model import load_model
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images


# ============================================================
# Basic utils
# ============================================================

def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def robust_diag(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]

    if len(points) < 10:
        return 1.0

    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)
    diag = float(np.linalg.norm(hi - lo))

    return max(diag, 1e-6)


def fit_plane_pca(points: np.ndarray):
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]

    if len(points) < 20:
        return None, None

    center = points.mean(axis=0)
    x = points - center

    try:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None

    normal = vh[-1].astype(np.float32)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    return center.astype(np.float32), normal


def point_to_plane_dist(points: np.ndarray, center: np.ndarray, normal: np.ndarray):
    return np.abs((points - center[None, :]) @ normal)


def img_tensor_to_rgb(img_chw):
    img = to_numpy(img_chw)

    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    img = img.astype(np.float32)
    img = np.clip(img, 0, 1)

    return img


def group_key_from_sample(sample_info):
    """
    Structured3D path:
    scene_xxxxx/2D_rendering/<room_id>/perspective/empty/<view_id>/layout.json

    We want to pair views from the same scene + same room/rendering group.
    """
    json_path = Path(sample_info["json_path"])
    parts = json_path.parts

    scene_name = sample_info.get("scene_name", "unknown_scene")

    if "2D_rendering" in parts:
        k = parts.index("2D_rendering")
        if k + 1 < len(parts):
            room_id = parts[k + 1]
            return f"{scene_name}/{room_id}"

    return scene_name


def build_pair_index(dataset):
    groups = defaultdict(list)

    for idx, info in enumerate(dataset.samples):
        groups[group_key_from_sample(info)].append(idx)

    pair_idx = {}

    for _, indices in groups.items():
        indices = sorted(indices)

        if len(indices) < 2:
            continue

        for j, idx in enumerate(indices):
            # 用同一个 room 里的下一个 view 当 pair
            pair_idx[idx] = indices[(j + 1) % len(indices)]

    return pair_idx


# ============================================================
# DUSt3R inference
# ============================================================

@torch.no_grad()
def run_pair_dust3r(model, rgb1_path, rgb2_path, image_size, device, batch_size=1):
    imgs = load_images([rgb1_path, rgb2_path], size=image_size, verbose=False)

    pairs = make_pairs(
        imgs,
        scene_graph="complete",
        prefilter=None,
        symmetrize=False,
    )

    output = inference(
        pairs,
        model,
        device,
        batch_size=batch_size,
        verbose=False,
    )

    # We put target image as image 0, so use pred1.
    pts = output["pred1"]["pts3d"][0]       # [H, W, 3]
    conf = output["pred1"]["conf"][0]       # [H, W]

    return pts.detach().cpu().numpy(), conf.detach().cpu().numpy()


# ============================================================
# Candidate metrics
# ============================================================

def analyze_one_plane(
    pts_hw3,
    conf_hw,
    gt_plane_hw,
    plane_id,
    conf_thr=1.5,
    min_plane_pixels=5000,
    planar_dist_ratio=0.006,
):
    mask = gt_plane_hw == plane_id

    plane_pixels = int(mask.sum())
    if plane_pixels < min_plane_pixels:
        return None

    pts = pts_hw3[mask]
    conf = conf_hw[mask]

    valid = np.isfinite(pts).all(axis=1)
    valid &= np.isfinite(conf)
    valid &= np.max(np.abs(pts), axis=1) < 1e5

    if int(valid.sum()) < min_plane_pixels:
        return None

    pts_valid = pts[valid]
    conf_valid = conf[valid]

    center, normal = fit_plane_pca(pts_valid)
    if center is None:
        return None

    diag = robust_diag(pts_valid)
    dist_thr = max(diag * planar_dist_ratio, 1e-4)

    dist = point_to_plane_dist(pts_valid, center, normal)

    flatness_mean = float(dist.mean())
    flatness_median = float(np.median(dist))
    flatness_p90 = float(np.percentile(dist, 90))
    flatness_p95 = float(np.percentile(dist, 95))

    flatness_mean_norm = flatness_mean / diag
    flatness_p90_norm = flatness_p90 / diag

    planar_core = dist <= dist_thr

    high_conf = conf_valid >= conf_thr
    low_conf = conf_valid < conf_thr

    valid_pixels = int(valid.sum())
    core_pixels = int(planar_core.sum())

    high_conf_ratio = float(high_conf.mean())
    low_conf_ratio = float(low_conf.mean())
    planar_core_ratio = float(planar_core.mean())

    if core_pixels > 0:
        low_conf_but_planar_ratio = float((planar_core & low_conf).sum() / core_pixels)
        high_conf_in_core_ratio = float((planar_core & high_conf).sum() / core_pixels)
    else:
        low_conf_but_planar_ratio = 0.0
        high_conf_in_core_ratio = 0.0

    conf_mean = float(conf_valid.mean())
    conf_std = float(conf_valid.std())
    conf_cv = conf_std / (conf_mean + 1e-6)

    # 这个 score 专门找：
    # 大平面 + 几何上接近平面 + 但是存在不少低置信点
    size_score = np.log1p(valid_pixels) / 12.0
    geometry_score = min(planar_core_ratio / 0.85, 1.0)
    low_conf_score = low_conf_but_planar_ratio
    threshold_gap_score = 1.0 - high_conf_ratio

    score = float(
        size_score
        * geometry_score
        * (0.70 * low_conf_score + 0.30 * threshold_gap_score)
    )

    return {
        "plane_id": int(plane_id),
        "plane_pixels": plane_pixels,
        "valid_pixels": valid_pixels,
        "diag": float(diag),
        "dist_thr": float(dist_thr),

        "flatness_mean": flatness_mean,
        "flatness_median": flatness_median,
        "flatness_p90": flatness_p90,
        "flatness_p95": flatness_p95,
        "flatness_mean_norm": float(flatness_mean_norm),
        "flatness_p90_norm": float(flatness_p90_norm),

        "conf_mean": conf_mean,
        "conf_std": conf_std,
        "conf_cv": float(conf_cv),

        "high_conf_ratio": high_conf_ratio,
        "low_conf_ratio": low_conf_ratio,
        "planar_core_ratio": planar_core_ratio,
        "low_conf_but_planar_ratio": low_conf_but_planar_ratio,
        "high_conf_in_core_ratio": high_conf_in_core_ratio,

        "score": score,
    }


def analyze_sample(
    sample_idx,
    dataset,
    pair_idx,
    model,
    device,
    image_size=512,
    batch_size=1,
    conf_thr=1.5,
    min_plane_pixels=5000,
    planar_dist_ratio=0.006,
):
    if sample_idx not in pair_idx:
        return [], None

    pair_sample_idx = pair_idx[sample_idx]

    sample = dataset[sample_idx]
    sample_info = dataset.samples[sample_idx]
    pair_info = dataset.samples[pair_sample_idx]

    rgb1_path = sample_info["rgb_path"]
    rgb2_path = pair_info["rgb_path"]

    gt_plane = to_numpy(sample["gt_plane"]).astype(np.int32)
    gt_line = to_numpy(sample["gt_line"])[0].astype(np.float32)
    rgb = img_tensor_to_rgb(sample["img"])

    pts, conf = run_pair_dust3r(
        model=model,
        rgb1_path=rgb1_path,
        rgb2_path=rgb2_path,
        image_size=image_size,
        device=device,
        batch_size=batch_size,
    )

    # 保证尺寸对齐
    h, w = gt_plane.shape
    if pts.shape[:2] != (h, w):
        pts_resized = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            pts_resized[..., c] = cv2.resize(
                pts[..., c],
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )
        pts = pts_resized

    if conf.shape[:2] != (h, w):
        conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)

    plane_ids = sorted([int(x) for x in np.unique(gt_plane) if int(x) > 0])

    rows = []

    for pid in plane_ids:
        stat = analyze_one_plane(
            pts_hw3=pts,
            conf_hw=conf,
            gt_plane_hw=gt_plane,
            plane_id=pid,
            conf_thr=conf_thr,
            min_plane_pixels=min_plane_pixels,
            planar_dist_ratio=planar_dist_ratio,
        )

        if stat is None:
            continue

        stat.update({
            "sample_idx": int(sample_idx),
            "pair_sample_idx": int(pair_sample_idx),
            "scene_name": sample_info.get("scene_name", ""),
            "group_key": group_key_from_sample(sample_info),
            "rgb_path": rgb1_path,
            "pair_rgb_path": rgb2_path,
            "json_path": sample_info["json_path"],
        })

        rows.append(stat)

    cache = {
        "rgb": rgb,
        "gt_plane": gt_plane,
        "gt_line": gt_line,
        "pts": pts,
        "conf": conf,
        "sample_info": sample_info,
        "pair_info": pair_info,
    }

    return rows, cache


# ============================================================
# Visualization
# ============================================================

def make_panel_for_candidate(row, cache, output_path, conf_thr=1.5, planar_dist_ratio=0.006):
    rgb = cache["rgb"]
    gt_plane = cache["gt_plane"]
    gt_line = cache["gt_line"]
    pts = cache["pts"]
    conf = cache["conf"]

    pid = int(row["plane_id"])
    mask = gt_plane == pid

    pts_plane = pts[mask]
    conf_plane = conf[mask]

    valid = np.isfinite(pts_plane).all(axis=1)
    valid &= np.isfinite(conf_plane)
    valid &= np.max(np.abs(pts_plane), axis=1) < 1e5

    pts_valid = pts_plane[valid]
    conf_valid = conf_plane[valid]

    center, normal = fit_plane_pca(pts_valid)
    diag = robust_diag(pts_valid)
    dist_thr = max(diag * planar_dist_ratio, 1e-4)

    full_dist = np.zeros(gt_plane.shape, dtype=np.float32)
    full_dist[:] = np.nan

    if center is not None:
        pts_flat = pts.reshape(-1, 3)
        valid_flat = np.isfinite(pts_flat).all(axis=1)
        valid_flat &= np.max(np.abs(pts_flat), axis=1) < 1e5

        dist_flat = np.full((pts_flat.shape[0],), np.nan, dtype=np.float32)
        dist_flat[valid_flat] = point_to_plane_dist(pts_flat[valid_flat], center, normal)

        full_dist = dist_flat.reshape(gt_plane.shape)

    plane_mask = mask.astype(np.float32)

    high_conf_mask = (conf >= conf_thr) & mask
    planar_core_mask = (full_dist <= dist_thr) & mask
    low_conf_but_planar = planar_core_mask & (conf < conf_thr)

    # 只显示当前平面的误差，其他区域置 nan
    dist_show = full_dist.copy()
    dist_show[~mask] = np.nan

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.reshape(-1)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")

    axes[1].imshow(plane_mask, cmap="gray")
    axes[1].contour(gt_line > 0.5, colors="yellow", linewidths=0.5)
    axes[1].set_title(f"GT Plane id={pid}")

    im2 = axes[2].imshow(conf, cmap="gray")
    axes[2].set_title("DUSt3R Confidence")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    axes[3].imshow(high_conf_mask, cmap="gray")
    axes[3].set_title(f"High Conf Mask >= {conf_thr}")

    axes[4].imshow(low_conf_but_planar, cmap="gray")
    axes[4].set_title("Low Conf but Planar")

    im5 = axes[5].imshow(dist_show, cmap="magma")
    axes[5].set_title("Point-to-Plane Dist")
    plt.colorbar(im5, ax=axes[5], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    title = (
        f"sample={row['sample_idx']} plane={pid} score={row['score']:.4f} | "
        f"low_conf_but_planar={row['low_conf_but_planar_ratio']:.3f}, "
        f"high_conf={row['high_conf_ratio']:.3f}, "
        f"p90_norm={row['flatness_p90_norm']:.5f}"
    )
    fig.suptitle(title, fontsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Mine DUSt3R baseline failure cases on Structured3D")

    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--train_ratio", type=float, default=0.9)

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--start_idx", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--conf_thr", type=float, default=1.5)
    parser.add_argument("--min_plane_pixels", type=int, default=5000)
    parser.add_argument("--planar_dist_ratio", type=float, default=0.006)

    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--save_top_panels", type=int, default=20)

    parser.add_argument("--only_good_geometry", action="store_true")
    parser.add_argument("--max_p90_norm", type=float, default=0.015)
    parser.add_argument("--min_low_conf_planar", type=float, default=0.15)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DUSt3R Baseline Failure Mining")
    print("=" * 80)
    print("device          :", device)
    print("root_dir        :", args.root_dir)
    print("weights_path    :", args.weights_path)
    print("split           :", args.split)
    print("num_samples     :", args.num_samples)
    print("output_dir      :", out_dir)
    print("conf_thr        :", args.conf_thr)
    print("min_plane_pixels:", args.min_plane_pixels)
    print("=" * 80)

    try:
        dataset = Structured3DDataset(
            root_dir=args.root_dir,
            split=args.split,
            train_ratio=args.train_ratio,
            image_size=(args.image_size, args.image_size),
            input_mode="single",
        )
    except TypeError:
        # 兼容旧版 Structured3DDataset
        dataset = Structured3DDataset(
            root_dir=args.root_dir,
            split=args.split,
            train_ratio=args.train_ratio,
            image_size=(args.image_size, args.image_size),
        )

    pair_idx = build_pair_index(dataset)

    print("dataset size:", len(dataset))
    print("pairable samples:", len(pair_idx))

    model = load_model(args.weights_path, device=device)
    model.eval()

    all_rows = []
    cache_for_top = {}

    end_idx = min(len(dataset), args.start_idx + args.num_samples)

    for idx in range(args.start_idx, end_idx):
        if idx not in pair_idx:
            continue

        try:
            rows, cache = analyze_sample(
                sample_idx=idx,
                dataset=dataset,
                pair_idx=pair_idx,
                model=model,
                device=device,
                image_size=args.image_size,
                batch_size=args.batch_size,
                conf_thr=args.conf_thr,
                min_plane_pixels=args.min_plane_pixels,
                planar_dist_ratio=args.planar_dist_ratio,
            )
        except Exception as e:
            print(f"[Skip] idx={idx}, error={repr(e)}")
            continue

        if len(rows) == 0:
            continue

        all_rows.extend(rows)

        # 暂时存一下 cache，后面 top 里用得到再重算也行；
        # 这里为了省事，先按 sample_idx 存。
        cache_for_top[idx] = cache

        if (idx - args.start_idx + 1) % 10 == 0:
            print(f"[{idx + 1}/{end_idx}] rows={len(all_rows)}")

    if len(all_rows) == 0:
        raise RuntimeError("No candidate rows found. Try lowering min_plane_pixels or increasing num_samples.")

    # 可选过滤：只保留几何还不错但 confidence 有问题的平面
    rows_for_rank = all_rows

    if args.only_good_geometry:
        rows_for_rank = [
            r for r in all_rows
            if r["flatness_p90_norm"] <= args.max_p90_norm
            and r["low_conf_but_planar_ratio"] >= args.min_low_conf_planar
        ]

        if len(rows_for_rank) == 0:
            print("[Warn] filter removed all rows. Fall back to all rows.")
            rows_for_rank = all_rows

    rows_for_rank = sorted(rows_for_rank, key=lambda r: r["score"], reverse=True)
    all_rows_sorted = sorted(all_rows, key=lambda r: r["score"], reverse=True)

    csv_path = out_dir / "candidates.csv"

    fieldnames = [
        "score",
        "sample_idx",
        "pair_sample_idx",
        "scene_name",
        "group_key",
        "plane_id",
        "plane_pixels",
        "valid_pixels",
        "diag",
        "dist_thr",
        "flatness_mean",
        "flatness_median",
        "flatness_p90",
        "flatness_p95",
        "flatness_mean_norm",
        "flatness_p90_norm",
        "conf_mean",
        "conf_std",
        "conf_cv",
        "high_conf_ratio",
        "low_conf_ratio",
        "planar_core_ratio",
        "low_conf_but_planar_ratio",
        "high_conf_in_core_ratio",
        "rgb_path",
        "pair_rgb_path",
        "json_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in all_rows_sorted:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print("=" * 80)
    print("Saved candidates")
    print("=" * 80)
    print("csv:", csv_path)
    print("rows:", len(all_rows_sorted))

    print("=" * 80)
    print(f"Top {min(args.top_k, len(rows_for_rank))} candidates")
    print("=" * 80)

    for i, r in enumerate(rows_for_rank[:args.top_k]):
        print(
            f"[{i+1:03d}] "
            f"score={r['score']:.4f} "
            f"idx={r['sample_idx']} "
            f"plane={r['plane_id']} "
            f"pixels={r['plane_pixels']} "
            f"low_planar={r['low_conf_but_planar_ratio']:.3f} "
            f"high_conf={r['high_conf_ratio']:.3f} "
            f"p90_norm={r['flatness_p90_norm']:.5f} "
            f"path={r['rgb_path']}"
        )

    panel_dir = out_dir / "top_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # 保存 top panel
    for rank, r in enumerate(rows_for_rank[:args.save_top_panels], start=1):
        sample_idx = int(r["sample_idx"])

        if sample_idx in cache_for_top:
            cache = cache_for_top[sample_idx]
        else:
            # 保险：如果 cache 没有，就重新算这个样本
            _, cache = analyze_sample(
                sample_idx=sample_idx,
                dataset=dataset,
                pair_idx=pair_idx,
                model=model,
                device=device,
                image_size=args.image_size,
                batch_size=args.batch_size,
                conf_thr=args.conf_thr,
                min_plane_pixels=args.min_plane_pixels,
                planar_dist_ratio=args.planar_dist_ratio,
            )

        name = (
            f"top_{rank:03d}"
            f"_idx{r['sample_idx']}"
            f"_plane{r['plane_id']}"
            f"_score{r['score']:.3f}.png"
        )
        name = safe_name(name)

        make_panel_for_candidate(
            row=r,
            cache=cache,
            output_path=panel_dir / name,
            conf_thr=args.conf_thr,
            planar_dist_ratio=args.planar_dist_ratio,
        )

    print("=" * 80)
    print("Saved panels")
    print("=" * 80)
    print("panel_dir:", panel_dir)


if __name__ == "__main__":
    main()