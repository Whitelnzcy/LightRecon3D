import argparse
import csv
import os
import re
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


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
    s = re.sub(r"[^\w\-.]+", "_", str(s))
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def read_txt_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def robust_diag(points):
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]

    if len(points) < 20:
        return 1.0

    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    return max(diag, 1e-6)


def fit_plane_pca(points, max_points=20000):
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]

    if len(points) < 30:
        return None, None

    if len(points) > max_points:
        rng = np.random.default_rng(123)
        idx = rng.choice(len(points), size=max_points, replace=False)
        fit_points = points[idx]
    else:
        fit_points = points

    center = fit_points.mean(axis=0)
    x = fit_points - center

    try:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None

    normal = vh[-1].astype(np.float32)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    return center.astype(np.float32), normal


def point_to_plane_dist(points, center, normal):
    return np.abs((points - center[None, :]) @ normal)


def img_tensor_to_rgb(img_chw):
    img = to_numpy(img_chw)

    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    img = img.astype(np.float32)
    img = np.clip(img, 0, 1)
    return img


def resize_pred_to_gt(pts, conf, h, w):
    if pts.shape[:2] != (h, w):
        out = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            out[..., c] = cv2.resize(
                pts[..., c],
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )
        pts = out

    if conf.shape[:2] != (h, w):
        conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)

    return pts, conf


def find_latest_case01():
    roots = sorted(
        Path("/gemini/data-1/important_results").glob("badcase_archive_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not roots:
        raise FileNotFoundError("Cannot find /gemini/data-1/important_results/badcase_archive_*")

    archive_root = roots[0]
    cases = sorted((archive_root / "cases").glob("case_01_*"))

    if not cases:
        raise FileNotFoundError(f"Cannot find case_01_* under {archive_root / 'cases'}")

    return archive_root, cases[0]


def build_dataset_index(root_dir, split, train_ratio, image_size):
    try:
        dataset = Structured3DDataset(
            root_dir=root_dir,
            split=split,
            train_ratio=train_ratio,
            image_size=(image_size, image_size),
            input_mode="single",
        )
    except TypeError:
        dataset = Structured3DDataset(
            root_dir=root_dir,
            split=split,
            train_ratio=train_ratio,
            image_size=(image_size, image_size),
        )

    rgb_to_idx = {}

    for idx, info in enumerate(dataset.samples):
        rgb_path = info.get("rgb_path", None)
        if rgb_path is None:
            raise KeyError(
                f"dataset.samples[{idx}] has no rgb_path. keys={list(info.keys())}. "
                f"Make sure Structured3DDataset is in input_mode='single'."
            )

        rgb_to_idx[str(Path(rgb_path).resolve())] = idx
        rgb_to_idx[str(Path(rgb_path))] = idx

    return dataset, rgb_to_idx


def get_sample_by_rgb_path(dataset, rgb_to_idx, rgb_path):
    p1 = str(Path(rgb_path).resolve())
    p2 = str(Path(rgb_path))

    if p1 in rgb_to_idx:
        return dataset[rgb_to_idx[p1]], dataset.samples[rgb_to_idx[p1]], rgb_to_idx[p1]

    if p2 in rgb_to_idx:
        return dataset[rgb_to_idx[p2]], dataset.samples[rgb_to_idx[p2]], rgb_to_idx[p2]

    # fallback: suffix match
    for k, idx in rgb_to_idx.items():
        if k.endswith(str(Path(rgb_path))):
            return dataset[idx], dataset.samples[idx], idx

    raise FileNotFoundError(f"Cannot map rgb path to dataset sample: {rgb_path}")


def extract_pred(pred_dict):
    # DUSt3R 不同版本可能 key 不一样，这里做兼容
    pts = None

    for key in ["pts3d", "pts3d_in_other_view"]:
        if key in pred_dict:
            pts = pred_dict[key]
            break

    if pts is None:
        raise KeyError(f"Cannot find pts3d key in pred_dict. keys={list(pred_dict.keys())}")

    if "conf" not in pred_dict:
        raise KeyError(f"Cannot find conf in pred_dict. keys={list(pred_dict.keys())}")

    conf = pred_dict["conf"]

    if torch.is_tensor(pts):
        pts = pts.detach().cpu().numpy()
    if torch.is_tensor(conf):
        conf = conf.detach().cpu().numpy()

    if pts.ndim == 4:
        pts = pts[0]
    if conf.ndim == 3:
        conf = conf[0]

    return pts.astype(np.float32), conf.astype(np.float32)


@torch.no_grad()
def run_dust3r_pair(model, img_i, img_j, image_size, device, batch_size=1):
    imgs = load_images([img_i, img_j], size=image_size, verbose=False)

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

    pts1, conf1 = extract_pred(output["pred1"])
    pts2, conf2 = extract_pred(output["pred2"])

    return pts1, conf1, pts2, conf2


# ============================================================
# Structure analysis
# ============================================================

def make_boundary_band(gt_line, radius):
    line = gt_line.astype(np.float32)

    if line.ndim == 3:
        line = line[0]

    line = line > 0.5

    if radius <= 0:
        return line

    k = 2 * radius + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    band = cv2.dilate(line.astype(np.uint8), kernel, iterations=1) > 0
    return band


def analyze_view_structure(
    pts,
    conf,
    sample,
    sample_info,
    view_name,
    pair_name,
    conf_thr=1.5,
    min_plane_pixels=5000,
    boundary_radius=3,
    plane_dist_ratio=0.006,
):
    gt_plane = to_numpy(sample["gt_plane"]).astype(np.int32)
    gt_line = to_numpy(sample["gt_line"]).astype(np.float32)

    if gt_line.ndim == 3:
        gt_line_2d = gt_line[0]
    else:
        gt_line_2d = gt_line

    h, w = gt_plane.shape
    pts, conf = resize_pred_to_gt(pts, conf, h, w)

    boundary_band = make_boundary_band(gt_line_2d, boundary_radius)

    rows = []
    maps = {
        "gt_plane": gt_plane,
        "gt_line": gt_line_2d,
        "boundary_band": boundary_band,
        "conf": conf,
        "under_conf_good": np.zeros((h, w), dtype=np.float32),
        "over_conf_bad": np.zeros((h, w), dtype=np.float32),
        "bad_geometry": np.zeros((h, w), dtype=np.float32),
    }

    plane_ids = sorted([int(x) for x in np.unique(gt_plane) if int(x) > 0])

    for plane_id in plane_ids:
        raw_mask = gt_plane == plane_id
        core_mask = raw_mask & (~boundary_band)

        plane_pixels = int(raw_mask.sum())
        core_pixels = int(core_mask.sum())

        if core_pixels < min_plane_pixels:
            continue

        pts_plane = pts[core_mask]
        conf_plane = conf[core_mask]

        valid = np.isfinite(pts_plane).all(axis=1)
        valid &= np.isfinite(conf_plane)
        valid &= np.max(np.abs(pts_plane), axis=1) < 1e5

        valid_pixels = int(valid.sum())

        if valid_pixels < min_plane_pixels:
            continue

        pts_valid = pts_plane[valid]
        conf_valid = conf_plane[valid]

        center, normal = fit_plane_pca(pts_valid)
        if center is None:
            continue

        diag = robust_diag(pts_valid)
        dist_thr = max(diag * plane_dist_ratio, 1e-4)

        dist = point_to_plane_dist(pts_valid, center, normal)

        flatness_mean = float(dist.mean())
        flatness_p90 = float(np.percentile(dist, 90))
        flatness_p95 = float(np.percentile(dist, 95))
        flatness_mean_norm = flatness_mean / diag
        flatness_p90_norm = flatness_p90 / diag
        flatness_p95_norm = flatness_p95 / diag

        planar_core = dist <= dist_thr
        high_conf = conf_valid >= conf_thr
        low_conf = conf_valid < conf_thr

        high_conf_ratio = float(high_conf.mean())
        low_conf_ratio = float(low_conf.mean())
        planar_core_ratio = float(planar_core.mean())

        if int(planar_core.sum()) > 0:
            low_conf_but_planar_ratio = float((planar_core & low_conf).sum() / planar_core.sum())
            high_conf_in_planar_ratio = float((planar_core & high_conf).sum() / planar_core.sum())
        else:
            low_conf_but_planar_ratio = 0.0
            high_conf_in_planar_ratio = 0.0

        conf_mean = float(conf_valid.mean())
        conf_std = float(conf_valid.std())

        good_geometry = (
            flatness_p90_norm <= 0.010
            and planar_core_ratio >= 0.65
        )

        under_conf_good = (
            good_geometry
            and low_conf_but_planar_ratio >= 0.30
        )

        bad_geometry = (
            flatness_p90_norm >= 0.020
            or planar_core_ratio <= 0.35
        )

        over_conf_bad = (
            bad_geometry
            and high_conf_ratio >= 0.50
        )

        # 映射回图像，用于可视化
        valid_full = np.zeros(core_pixels, dtype=bool)
        valid_full[np.where(valid)[0]] = True

        # 这里简单用整块 mask 着色，不做逐点回填，便于观察区域级别
        if under_conf_good:
            maps["under_conf_good"][raw_mask] = 1.0
        if bad_geometry:
            maps["bad_geometry"][raw_mask] = 1.0
        if over_conf_bad:
            maps["over_conf_bad"][raw_mask] = 1.0

        rows.append({
            "pair": pair_name,
            "view": view_name,
            "scene_name": sample_info.get("scene_name", ""),
            "rgb_path": sample_info.get("rgb_path", ""),
            "plane_id": plane_id,
            "plane_pixels": plane_pixels,
            "core_pixels": core_pixels,
            "valid_pixels": valid_pixels,
            "diag": diag,
            "dist_thr": dist_thr,
            "flatness_mean": flatness_mean,
            "flatness_p90": flatness_p90,
            "flatness_p95": flatness_p95,
            "flatness_mean_norm": flatness_mean_norm,
            "flatness_p90_norm": flatness_p90_norm,
            "flatness_p95_norm": flatness_p95_norm,
            "conf_mean": conf_mean,
            "conf_std": conf_std,
            "high_conf_ratio": high_conf_ratio,
            "low_conf_ratio": low_conf_ratio,
            "planar_core_ratio": planar_core_ratio,
            "low_conf_but_planar_ratio": low_conf_but_planar_ratio,
            "high_conf_in_planar_ratio": high_conf_in_planar_ratio,
            "good_geometry": int(good_geometry),
            "under_conf_good": int(under_conf_good),
            "bad_geometry": int(bad_geometry),
            "over_conf_bad": int(over_conf_bad),
        })

    return rows, maps


def weighted_mean(rows, key, weight_key="core_pixels"):
    num = 0.0
    den = 0.0

    for r in rows:
        w = float(r.get(weight_key, 0.0))
        v = float(r.get(key, 0.0))
        num += w * v
        den += w

    if den <= 0:
        return 0.0

    return num / den


def summarize_pair(pair_name, rows):
    if len(rows) == 0:
        return {
            "pair": pair_name,
            "num_planes": 0,
            "total_core_pixels": 0,
            "mean_flatness_p90_norm": 0.0,
            "mean_high_conf_ratio": 0.0,
            "mean_low_conf_but_planar_ratio": 0.0,
            "mean_planar_core_ratio": 0.0,
            "under_conf_good_planes": 0,
            "bad_geometry_planes": 0,
            "over_conf_bad_planes": 0,
            "under_conf_good_area_ratio": 0.0,
            "bad_geometry_area_ratio": 0.0,
            "over_conf_bad_area_ratio": 0.0,
            "pair_unreliable_score": 0.0,
        }

    total_core = sum(int(r["core_pixels"]) for r in rows)

    under_pixels = sum(int(r["core_pixels"]) for r in rows if int(r["under_conf_good"]) == 1)
    bad_pixels = sum(int(r["core_pixels"]) for r in rows if int(r["bad_geometry"]) == 1)
    over_pixels = sum(int(r["core_pixels"]) for r in rows if int(r["over_conf_bad"]) == 1)

    under_area = under_pixels / max(total_core, 1)
    bad_area = bad_pixels / max(total_core, 1)
    over_area = over_pixels / max(total_core, 1)

    # 这个不是最终方法，只是诊断排序分数：
    # under_conf_good 说明“可能被低估的可靠平面”
    # over_conf_bad / bad_geometry 说明“可能拖坏全局的结构异常”
    unreliable_score = (
        0.40 * bad_area
        + 0.35 * over_area
        + 0.25 * abs(weighted_mean(rows, "high_conf_ratio") - weighted_mean(rows, "planar_core_ratio"))
    )

    return {
        "pair": pair_name,
        "num_planes": len(rows),
        "total_core_pixels": total_core,
        "mean_flatness_p90_norm": weighted_mean(rows, "flatness_p90_norm"),
        "mean_high_conf_ratio": weighted_mean(rows, "high_conf_ratio"),
        "mean_low_conf_but_planar_ratio": weighted_mean(rows, "low_conf_but_planar_ratio"),
        "mean_planar_core_ratio": weighted_mean(rows, "planar_core_ratio"),
        "under_conf_good_planes": sum(int(r["under_conf_good"]) for r in rows),
        "bad_geometry_planes": sum(int(r["bad_geometry"]) for r in rows),
        "over_conf_bad_planes": sum(int(r["over_conf_bad"]) for r in rows),
        "under_conf_good_area_ratio": under_area,
        "bad_geometry_area_ratio": bad_area,
        "over_conf_bad_area_ratio": over_area,
        "pair_unreliable_score": unreliable_score,
    }


def save_pair_panel(pair_name, sample_i, sample_j, maps_i, maps_j, out_path):
    rgb_i = img_tensor_to_rgb(sample_i["img"])
    rgb_j = img_tensor_to_rgb(sample_j["img"])

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    data = [
        (rgb_i, "view_i RGB", None),
        (maps_i["gt_plane"], "view_i GT Plane", "tab20"),
        (maps_i["conf"], "view_i Confidence", "gray"),
        (maps_i["under_conf_good"], "view_i UnderConf GoodPlane", "gray"),
        (maps_i["over_conf_bad"], "view_i OverConf BadPlane", "gray"),

        (rgb_j, "view_j RGB", None),
        (maps_j["gt_plane"], "view_j GT Plane", "tab20"),
        (maps_j["conf"], "view_j Confidence", "gray"),
        (maps_j["under_conf_good"], "view_j UnderConf GoodPlane", "gray"),
        (maps_j["over_conf_bad"], "view_j OverConf BadPlane", "gray"),
    ]

    for ax, (img, title, cmap) in zip(axes.reshape(-1), data):
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(pair_name, fontsize=14)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Oracle pair structure audit for badcase case01")

    parser.add_argument("--root_dir", default="/gemini/data-1/Structured3D")
    parser.add_argument("--weights_path", default="/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    parser.add_argument("--archive_root", default="")
    parser.add_argument("--case_dir", default="")

    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--train_ratio", type=float, default=0.9)

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--conf_thr", type=float, default=1.5)
    parser.add_argument("--min_plane_pixels", type=int, default=5000)
    parser.add_argument("--boundary_radius", type=int, default=3)
    parser.add_argument("--plane_dist_ratio", type=float, default=0.006)

    args = parser.parse_args()

    if args.archive_root and args.case_dir:
        archive_root = Path(args.archive_root)
        case_dir = Path(args.case_dir)
    else:
        archive_root, case_dir = find_latest_case01()

    out_dir = case_dir / "oracle_pair_structure"
    panel_dir = out_dir / "pair_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    used_images_path = case_dir / "used_images.txt"
    if not used_images_path.exists():
        raise FileNotFoundError(f"Cannot find {used_images_path}")

    image_paths = read_txt_lines(used_images_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Oracle Pair Structure Audit")
    print("=" * 80)
    print("device       :", device)
    print("archive_root :", archive_root)
    print("case_dir     :", case_dir)
    print("out_dir      :", out_dir)
    print("num images   :", len(image_paths))
    print("conf_thr     :", args.conf_thr)
    print("=" * 80)

    print("[1] Build dataset index")
    dataset, rgb_to_idx = build_dataset_index(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=args.image_size,
    )

    sample_cache = {}

    for p in image_paths:
        sample, info, idx = get_sample_by_rgb_path(dataset, rgb_to_idx, p)
        sample_cache[p] = {
            "sample": sample,
            "info": info,
            "idx": idx,
        }
        print(f"mapped: idx={idx} path={p}")

    print("[2] Load DUSt3R")
    model = load_model(args.weights_path, device=device)
    model.eval()

    plane_rows = []
    pair_rows = []

    print("[3] Run pair oracle analysis")

    n = len(image_paths)

    for i in range(n):
        for j in range(i + 1, n):
            img_i = image_paths[i]
            img_j = image_paths[j]

            pair_name = f"pair_v{i}_v{j}"

            print("-" * 80)
            print(pair_name)
            print("img_i:", img_i)
            print("img_j:", img_j)

            pts_i, conf_i, pts_j, conf_j = run_dust3r_pair(
                model=model,
                img_i=img_i,
                img_j=img_j,
                image_size=args.image_size,
                device=device,
                batch_size=args.batch_size,
            )

            sample_i = sample_cache[img_i]["sample"]
            info_i = sample_cache[img_i]["info"]

            sample_j = sample_cache[img_j]["sample"]
            info_j = sample_cache[img_j]["info"]

            rows_i, maps_i = analyze_view_structure(
                pts=pts_i,
                conf=conf_i,
                sample=sample_i,
                sample_info=info_i,
                view_name=f"v{i}",
                pair_name=pair_name,
                conf_thr=args.conf_thr,
                min_plane_pixels=args.min_plane_pixels,
                boundary_radius=args.boundary_radius,
                plane_dist_ratio=args.plane_dist_ratio,
            )

            rows_j, maps_j = analyze_view_structure(
                pts=pts_j,
                conf=conf_j,
                sample=sample_j,
                sample_info=info_j,
                view_name=f"v{j}",
                pair_name=pair_name,
                conf_thr=args.conf_thr,
                min_plane_pixels=args.min_plane_pixels,
                boundary_radius=args.boundary_radius,
                plane_dist_ratio=args.plane_dist_ratio,
            )

            pair_plane_rows = rows_i + rows_j
            plane_rows.extend(pair_plane_rows)

            summary = summarize_pair(pair_name, pair_plane_rows)
            pair_rows.append(summary)

            print(
                f"{pair_name}: "
                f"planes={summary['num_planes']}, "
                f"p90_norm={summary['mean_flatness_p90_norm']:.5f}, "
                f"high_conf={summary['mean_high_conf_ratio']:.3f}, "
                f"low_conf_planar={summary['mean_low_conf_but_planar_ratio']:.3f}, "
                f"under_area={summary['under_conf_good_area_ratio']:.3f}, "
                f"bad_area={summary['bad_geometry_area_ratio']:.3f}, "
                f"over_area={summary['over_conf_bad_area_ratio']:.3f}, "
                f"score={summary['pair_unreliable_score']:.3f}"
            )

            save_pair_panel(
                pair_name=pair_name,
                sample_i=sample_i,
                sample_j=sample_j,
                maps_i=maps_i,
                maps_j=maps_j,
                out_path=panel_dir / f"{pair_name}_oracle_panel.png",
            )

    plane_csv = out_dir / "oracle_plane_rows.csv"
    pair_csv = out_dir / "oracle_pair_summary.csv"

    print("[4] Save CSV")

    if plane_rows:
        fieldnames = list(plane_rows[0].keys())
        with open(plane_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(plane_rows)

    if pair_rows:
        fieldnames = list(pair_rows[0].keys())
        with open(pair_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pair_rows)

    print("=" * 80)
    print("Done")
    print("=" * 80)
    print("out_dir:", out_dir)
    print("plane_csv:", plane_csv)
    print("pair_csv:", pair_csv)
    print("panel_dir:", panel_dir)

    print("=" * 80)
    print("Pair summary sorted by unreliable score")
    print("=" * 80)

    pair_rows_sorted = sorted(
        pair_rows,
        key=lambda x: float(x["pair_unreliable_score"]),
        reverse=True,
    )

    for r in pair_rows_sorted:
        print(
            f"{r['pair']:10s} "
            f"score={r['pair_unreliable_score']:.4f} "
            f"p90_norm={r['mean_flatness_p90_norm']:.5f} "
            f"high_conf={r['mean_high_conf_ratio']:.3f} "
            f"low_planar={r['mean_low_conf_but_planar_ratio']:.3f} "
            f"under_area={r['under_conf_good_area_ratio']:.3f} "
            f"bad_area={r['bad_geometry_area_ratio']:.3f} "
            f"over_area={r['over_conf_bad_area_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()