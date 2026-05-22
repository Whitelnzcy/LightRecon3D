import argparse
import csv
import os
import sys
from pathlib import Path

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
from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import load_model
from dust3r.utils.image import load_images
from run_raw_dust3r_multiview import (
    filter_pairs_by_args,
    reweight_output_conf_by_pair,
)


METHODS = [
    {"method": "all_pairs", "drop_pairs": None, "pair_weights": None},
    {"method": "drop_bad3", "drop_pairs": ["0-1", "0-2", "2-3"], "pair_weights": None},
    {"method": "bad3_w0p5", "drop_pairs": None, "pair_weights": ["0-1:0.5", "0-2:0.5", "2-3:0.5"]},
    {"method": "bad3_w0p6", "drop_pairs": None, "pair_weights": ["0-1:0.6", "0-2:0.6", "2-3:0.6"]},
    {"method": "bad3_w0p7", "drop_pairs": None, "pair_weights": ["0-1:0.7", "0-2:0.7", "2-3:0.7"]},
    {"method": "bad3_w0p8", "drop_pairs": None, "pair_weights": ["0-1:0.8", "0-2:0.8", "2-3:0.8"]},
    {"method": "only_01_w0p5", "drop_pairs": None, "pair_weights": ["0-1:0.5"]},
    {"method": "only_02_w0p5", "drop_pairs": None, "pair_weights": ["0-2:0.5"]},
    {"method": "only_23_w0p5", "drop_pairs": None, "pair_weights": ["2-3:0.5"]},
]


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def read_used_images(case_dir):
    case_dir = Path(case_dir)
    used_path = case_dir / "used_images.txt"

    if not used_path.exists():
        matches = sorted(case_dir.glob("*used_images*.txt"))
        if not matches:
            raise FileNotFoundError(f"Cannot find used_images.txt under {case_dir}")
        used_path = matches[0]

    image_paths = []
    with open(used_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                image_paths.append(line)

    if len(image_paths) < 2:
        raise RuntimeError(f"Need at least two used images, got {len(image_paths)}")

    return image_paths, used_path


def find_latest_case01(archive_root):
    if archive_root:
        roots = [Path(archive_root)]
    else:
        roots = sorted(
            Path("/gemini/data-1/important_results").glob("badcase_archive_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    if not roots:
        raise FileNotFoundError("Cannot find latest badcase_archive_*")

    for root in roots:
        case_dirs = sorted((root / "cases").glob("case_01_*"))
        if case_dirs:
            return root, case_dirs[0]

    raise FileNotFoundError(f"Cannot find case_01_* under {roots[0] / 'cases'}")


def path_keys(path):
    p = Path(str(path))
    keys = {str(p)}

    try:
        keys.add(str(p.resolve()))
    except OSError:
        pass

    parts = p.parts
    if "Structured3D" in parts:
        idx = parts.index("Structured3D")
        keys.add("/".join(parts[idx:]))
        keys.add(os.path.join(*parts[idx:]))

    return keys


def build_rgb_index(dataset):
    rgb_to_idx = {}

    for idx, info in enumerate(dataset.samples):
        rgb_path = info.get("rgb_path")
        if rgb_path is None:
            continue

        for key in path_keys(rgb_path):
            rgb_to_idx[key] = idx

    return rgb_to_idx


def lookup_dataset_sample(dataset, rgb_to_idx, rgb_path):
    for key in path_keys(rgb_path):
        if key in rgb_to_idx:
            idx = rgb_to_idx[key]
            return dataset[idx], dataset.samples[idx], idx

    wanted = str(Path(rgb_path)).replace("\\", "/")
    for key, idx in rgb_to_idx.items():
        if wanted.endswith(str(Path(key)).replace("\\", "/")) or str(Path(key)).replace("\\", "/").endswith(wanted):
            return dataset[idx], dataset.samples[idx], idx

    raise FileNotFoundError(f"Cannot map used image to Structured3D sample: {rgb_path}")


def resize_label_nearest(label_hw, size):
    if torch.is_tensor(label_hw):
        x = label_hw.detach().cpu()
    else:
        x = torch.from_numpy(np.asarray(label_hw))

    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]

    if tuple(x.shape[-2:]) == tuple(size):
        return x.long().numpy()

    y = F.interpolate(
        x[None, None].float(),
        size=size,
        mode="nearest",
    )[0, 0]
    return y.long().numpy()


def resize_line_mask(line, size):
    if torch.is_tensor(line):
        x = line.detach().cpu().float()
    else:
        x = torch.from_numpy(np.asarray(line)).float()

    if x.ndim == 2:
        x = x[None]

    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]

    if tuple(x.shape[-2:]) != tuple(size):
        x = F.interpolate(
            x[None, None].float(),
            size=size,
            mode="nearest",
        )[0, 0]

    return x.numpy() > 0.5


def dilate_binary_mask(mask, radius):
    if radius <= 0:
        return mask.astype(bool)

    x = torch.from_numpy(mask.astype(np.float32))[None, None]
    y = F.max_pool2d(
        x,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return y[0, 0].numpy() > 0.5


def fit_plane_metrics(points, conf, plane_pixels, max_abs_coord, outlier_dist_ratio):
    valid = np.isfinite(points).all(axis=1)
    valid &= np.isfinite(conf)
    valid &= np.max(np.abs(points), axis=1) < max_abs_coord

    valid_pixels = int(valid.sum())
    valid_ratio = float(valid_pixels / max(plane_pixels, 1))

    row = {
        "plane_pixels": int(plane_pixels),
        "valid_pixels": valid_pixels,
        "valid_ratio": valid_ratio,
        "conf1_ratio": float("nan"),
        "conf15_ratio": float("nan"),
        "flatness_mean": float("nan"),
        "flatness_median": float("nan"),
        "flatness_p90": float("nan"),
        "flatness_p95": float("nan"),
        "flatness_mean_norm": float("nan"),
        "flatness_p90_norm": float("nan"),
        "outlier_ratio": float("nan"),
    }

    if valid_pixels < 20:
        return row

    points_valid = points[valid].astype(np.float32)
    conf_valid = conf[valid].astype(np.float32)

    row["conf1_ratio"] = float(np.mean(conf_valid >= 1.0))
    row["conf15_ratio"] = float(np.mean(conf_valid >= 1.5))

    lo = np.percentile(points_valid, 1, axis=0)
    hi = np.percentile(points_valid, 99, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    diag = max(diag, 1e-6)

    center = points_valid.mean(axis=0)
    x = points_valid - center[None, :]

    try:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return row

    normal = vh[-1].astype(np.float32)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    dist = np.abs((points_valid - center[None, :]) @ normal)

    flatness_mean = float(dist.mean())
    flatness_p90 = float(np.percentile(dist, 90))

    row.update({
        "flatness_mean": flatness_mean,
        "flatness_median": float(np.median(dist)),
        "flatness_p90": flatness_p90,
        "flatness_p95": float(np.percentile(dist, 95)),
        "flatness_mean_norm": float(flatness_mean / diag),
        "flatness_p90_norm": float(flatness_p90 / diag),
        "outlier_ratio": float(np.mean(dist > diag * outlier_dist_ratio)),
    })

    return row


def evaluate_view_planes(
    method,
    final_loss,
    view_idx,
    pts_hw3,
    conf_hw,
    gt_plane,
    gt_line,
    args,
):
    pts = to_numpy(pts_hw3).astype(np.float32)
    conf = to_numpy(conf_hw).astype(np.float32)

    if conf.ndim == 3 and conf.shape[-1] == 1:
        conf = conf[..., 0]

    h, w = pts.shape[:2]
    plane = resize_label_nearest(gt_plane, (h, w))
    line = resize_line_mask(gt_line, (h, w))
    line_band = dilate_binary_mask(line, args.line_dilate)

    rows = []

    for pid in sorted(np.unique(plane).tolist()):
        pid = int(pid)
        if pid <= 0 or pid == 255:
            continue

        mask = (plane == pid) & (~line_band)
        plane_pixels = int(mask.sum())
        if plane_pixels < args.min_plane_pixels:
            continue

        metrics = fit_plane_metrics(
            points=pts[mask],
            conf=conf[mask],
            plane_pixels=plane_pixels,
            max_abs_coord=args.max_abs_coord,
            outlier_dist_ratio=args.outlier_dist_ratio,
        )

        row = {
            "method": method,
            "final_loss": final_loss,
            "view_idx": view_idx,
            "plane_id": pid,
        }
        row.update(metrics)
        rows.append(row)

    return rows


def is_finite_number(x):
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def weighted_average(rows, key, weight_key="valid_pixels"):
    values = []
    weights = []

    for row in rows:
        value = row.get(key)
        weight = row.get(weight_key, 0)

        if is_finite_number(value) and is_finite_number(weight) and float(weight) > 0:
            values.append(float(value))
            weights.append(float(weight))

    if not values:
        return float("nan")

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(values * weights) / max(np.sum(weights), 1e-12))


def summarize_method(method, final_loss, num_views, plane_rows):
    total_plane_pixels = sum(int(r["plane_pixels"]) for r in plane_rows)
    total_valid_pixels = sum(int(r["valid_pixels"]) for r in plane_rows)

    return {
        "method": method,
        "final_loss": final_loss,
        "num_views": num_views,
        "total_valid_pixels": total_valid_pixels,
        "mean_valid_ratio": float(total_valid_pixels / max(total_plane_pixels, 1)),
        "mean_conf1_ratio": weighted_average(plane_rows, "conf1_ratio"),
        "mean_conf15_ratio": weighted_average(plane_rows, "conf15_ratio"),
        "weighted_flatness_p90_norm": weighted_average(plane_rows, "flatness_p90_norm"),
        "weighted_outlier_ratio": weighted_average(plane_rows, "outlier_ratio"),
        "weighted_flatness_mean_norm": weighted_average(plane_rows, "flatness_mean_norm"),
    }


def run_one_method(method_cfg, base_pairs, model, image_paths, gt_samples, args, device):
    method = method_cfg["method"]

    print("=" * 80)
    print(f"Method: {method}")
    print("=" * 80)

    pairs = filter_pairs_by_args(
        base_pairs,
        keep_pairs=None,
        drop_pairs=method_cfg.get("drop_pairs"),
    )

    output = inference(
        pairs,
        model,
        device,
        batch_size=args.batch_size,
        verbose=True,
    )

    output = reweight_output_conf_by_pair(
        output,
        pairs,
        pair_weights=method_cfg.get("pair_weights"),
    )

    scene = global_aligner(
        output,
        device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
    )

    loss = scene.compute_global_alignment(
        init="mst",
        niter=args.niter,
        schedule=args.schedule,
        lr=args.lr,
    )
    final_loss = float(loss.detach().cpu().item() if torch.is_tensor(loss) else loss)

    pts3d = scene.get_pts3d()
    confs = scene.get_conf()

    if len(pts3d) != len(gt_samples):
        raise RuntimeError(
            f"Method {method}: scene returned {len(pts3d)} views, "
            f"but case has {len(gt_samples)} GT samples."
        )

    plane_rows = []

    for view_idx, (pts, conf, gt_sample) in enumerate(zip(pts3d, confs, gt_samples)):
        plane_rows.extend(evaluate_view_planes(
            method=method,
            final_loss=final_loss,
            view_idx=view_idx,
            pts_hw3=pts,
            conf_hw=conf,
            gt_plane=gt_sample["gt_plane"],
            gt_line=gt_sample["gt_line"],
            args=args,
        ))

    summary = summarize_method(
        method=method,
        final_loss=final_loss,
        num_views=len(image_paths),
        plane_rows=plane_rows,
    )

    return summary, plane_rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser("Evaluate case01 global structure metrics")

    parser.add_argument("--archive_root", default="")
    parser.add_argument("--root_dir", default="/gemini/data-1/Structured3D")
    parser.add_argument("--weights_path", default="/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--image_size", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--schedule", default="cosine")
    parser.add_argument("--scene_graph", default="complete")

    parser.add_argument("--line_dilate", type=int, default=3)
    parser.add_argument("--min_plane_pixels", type=int, default=500)
    parser.add_argument("--max_abs_coord", type=float, default=1e5)
    parser.add_argument("--outlier_dist_ratio", type=float, default=0.006)

    parser.add_argument(
        "--summary_csv",
        default="/gemini/data-1/logs/case01_global_structure_metrics.csv",
    )
    parser.add_argument(
        "--plane_csv",
        default="/gemini/data-1/logs/case01_global_structure_plane_rows.csv",
    )

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    archive_root, case_dir = find_latest_case01(args.archive_root)
    image_paths, used_images_path = read_used_images(case_dir)

    print("=" * 80)
    print("Case01 Global Structure Evaluation")
    print("=" * 80)
    print("device          :", device)
    print("archive_root    :", archive_root)
    print("case_dir        :", case_dir)
    print("used_images     :", used_images_path)
    print("root_dir        :", args.root_dir)
    print("weights_path    :", args.weights_path)
    print("num_images      :", len(image_paths))
    print("summary_csv     :", args.summary_csv)
    print("plane_csv       :", args.plane_csv)
    print("=" * 80)

    for idx, path in enumerate(image_paths):
        print(f"[{idx}] {path}")

    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        input_mode="single",
    )
    rgb_to_idx = build_rgb_index(dataset)

    gt_samples = []
    for rgb_path in image_paths:
        sample, info, idx = lookup_dataset_sample(dataset, rgb_to_idx, rgb_path)
        gt_samples.append(sample)
        print(f"[GT] image={rgb_path} -> dataset_idx={idx} json={info['json_path']}")

    print("=" * 80)
    print("Loading DUSt3R")
    print("=" * 80)
    model = load_model(args.weights_path, device=device)
    model.eval()

    print("=" * 80)
    print("Loading images and pairs")
    print("=" * 80)
    imgs = load_images(image_paths, size=args.image_size, verbose=True)
    base_pairs = make_pairs(
        imgs,
        scene_graph=args.scene_graph,
        prefilter=None,
        symmetrize=True,
    )
    print("base pair count:", len(base_pairs))

    summary_rows = []
    plane_rows = []

    for method_cfg in METHODS:
        summary, rows = run_one_method(
            method_cfg=method_cfg,
            base_pairs=base_pairs,
            model=model,
            image_paths=image_paths,
            gt_samples=gt_samples,
            args=args,
            device=device,
        )
        summary_rows.append(summary)
        plane_rows.extend(rows)

    summary_fields = [
        "method",
        "final_loss",
        "num_views",
        "total_valid_pixels",
        "mean_valid_ratio",
        "mean_conf1_ratio",
        "mean_conf15_ratio",
        "weighted_flatness_p90_norm",
        "weighted_outlier_ratio",
        "weighted_flatness_mean_norm",
    ]
    plane_fields = [
        "method",
        "final_loss",
        "view_idx",
        "plane_id",
        "plane_pixels",
        "valid_pixels",
        "valid_ratio",
        "conf1_ratio",
        "conf15_ratio",
        "flatness_mean",
        "flatness_median",
        "flatness_p90",
        "flatness_p95",
        "flatness_mean_norm",
        "flatness_p90_norm",
        "outlier_ratio",
    ]

    write_csv(args.summary_csv, summary_rows, summary_fields)
    write_csv(args.plane_csv, plane_rows, plane_fields)

    print("=" * 80)
    print("Saved")
    print("=" * 80)
    print("summary_csv:", args.summary_csv)
    print("plane_csv  :", args.plane_csv)


if __name__ == "__main__":
    main()
