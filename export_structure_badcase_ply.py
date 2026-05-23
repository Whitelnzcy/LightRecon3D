import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
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
from structure_badcase_mining import (
    build_pair_index,
    dilate,
    fit_plane_pca,
    img_tensor_to_rgb,
    load_gt_masks,
    mask_boundary,
    point_to_plane_dist,
    resize_hw,
    run_pair_dust3r,
    valid_points,
)


def read_candidate(candidate_csv, sample_idx, plane_id):
    candidate_csv = Path(candidate_csv)
    if not candidate_csv.exists():
        raise FileNotFoundError(f"candidate_csv not found: {candidate_csv}")

    with open(candidate_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row_sample_idx = int(row.get("sample_idx", ""))
                row_plane_id = int(row.get("plane_id", ""))
            except ValueError:
                continue
            if row_sample_idx == sample_idx and row_plane_id == plane_id:
                return row

    raise RuntimeError(
        f"No row with sample_idx={sample_idx} and plane_id={plane_id} in {candidate_csv}"
    )


def build_masks(plane_mask, line_mask, plane_id, line_band_radius, boundary_band_radius):
    target_plane = plane_mask == plane_id
    plane_edge = mask_boundary(target_plane)

    line_band_full = dilate(line_mask, line_band_radius)
    line_band = line_band_full & dilate(target_plane, max(line_band_radius, 1))

    boundary_band = target_plane & dilate(plane_edge, boundary_band_radius)
    plane_core = target_plane & ~(boundary_band | line_band)

    return {
        "target_plane": target_plane,
        "plane_core": plane_core,
        "boundary_band": boundary_band,
        "line_band": line_band,
        "boundary_line": boundary_band & line_band,
    }


def rgb_to_uint8(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def flatten_valid(pts_hw3, rgb_hwc, mask):
    h, w, _ = pts_hw3.shape
    if rgb_hwc.shape[:2] != (h, w):
        rgb_hwc = resize_hw(rgb_hwc, (h, w), cv2.INTER_LINEAR)

    pts = pts_hw3.reshape(-1, 3)
    rgb = rgb_to_uint8(rgb_hwc).reshape(-1, 3)
    mask_flat = mask.reshape(-1).astype(bool)
    valid = mask_flat & valid_points(pts)
    return pts[valid], rgb[valid], valid


def write_ply(path, points, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)

    if points.shape[0] != colors.shape[0]:
        raise ValueError(f"points/colors length mismatch: {points.shape[0]} vs {colors.shape[0]}")

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(
                f"{float(p[0]):.8f} {float(p[1]):.8f} {float(p[2]):.8f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def structure_colors(shape, masks, base_mask=None):
    if base_mask is None:
        base_mask = np.ones(shape, dtype=bool)

    colors = np.zeros((*shape, 3), dtype=np.uint8)
    colors[base_mask] = np.array([150, 150, 150], dtype=np.uint8)

    colors[masks["plane_core"]] = np.array([0, 210, 80], dtype=np.uint8)
    colors[masks["boundary_band"]] = np.array([235, 45, 45], dtype=np.uint8)
    colors[masks["line_band"]] = np.array([40, 110, 245], dtype=np.uint8)
    colors[masks["boundary_line"]] = np.array([255, 220, 20], dtype=np.uint8)
    return colors


def heat_colormap(values):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    out = np.zeros((values.shape[0], 3), dtype=np.uint8)
    if not finite.any():
        out[:] = np.array([120, 120, 120], dtype=np.uint8)
        return out

    lo = np.percentile(values[finite], 5)
    hi = np.percentile(values[finite], 98)
    t = (values - lo) / max(hi - lo, 1e-8)
    t = np.clip(t, 0.0, 1.0)

    # Blue/cyan for small distance, yellow/red for large distance.
    r = np.clip(3.0 * t - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(3.0 * t - 1.5), 0.0, 1.0)
    b = np.clip(1.2 - 2.5 * t, 0.0, 1.0)
    out[:, 0] = (255 * r).astype(np.uint8)
    out[:, 1] = (255 * g).astype(np.uint8)
    out[:, 2] = (255 * b).astype(np.uint8)
    out[~finite] = np.array([120, 120, 120], dtype=np.uint8)
    return out


def build_dataset(args):
    return Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        input_mode="single",
    )


def export_case(args):
    candidate = read_candidate(args.candidate_csv, args.sample_idx, args.plane_id)

    dataset = build_dataset(args)
    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        raise IndexError(f"sample_idx={args.sample_idx} outside dataset size={len(dataset)}")

    pair_idx = build_pair_index(dataset)
    if args.sample_idx not in pair_idx:
        raise RuntimeError(f"sample_idx={args.sample_idx} has no pair in split={args.split}")

    sample = dataset[args.sample_idx]
    sample_info = dataset.samples[args.sample_idx]
    pair_info = dataset.samples[pair_idx[args.sample_idx]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from dust3r.model import load_model

    model = load_model(args.weights_path, device=device)
    model.eval()

    pts, conf = run_pair_dust3r(
        model=model,
        rgb1_path=sample_info["rgb_path"],
        rgb2_path=pair_info["rgb_path"],
        image_size=args.image_size,
        device=device,
        batch_size=args.batch_size,
    )

    h, w = pts.shape[:2]
    rgb = img_tensor_to_rgb(sample["img"])
    if rgb.shape[:2] != (h, w):
        rgb = resize_hw(rgb, (h, w), cv2.INTER_LINEAR)
    if conf.shape[:2] != (h, w):
        conf = resize_hw(conf, (h, w), cv2.INTER_LINEAR)

    plane_mask, line_mask = load_gt_masks(dataset, sample_info, (h, w))
    masks = build_masks(
        plane_mask=plane_mask,
        line_mask=line_mask,
        plane_id=args.plane_id,
        line_band_radius=args.line_band_radius,
        boundary_band_radius=args.boundary_band_radius,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_hw = valid_points(pts)
    conf_mask = valid_hw & (conf >= args.conf_thr)
    full_pts, full_rgb, _ = flatten_valid(pts, rgb, conf_mask)
    write_ply(out_dir / "full_conf1p5.ply", full_pts, full_rgb)

    all_valid_pts = pts.reshape(-1, 3)
    all_valid_mask = valid_points(all_valid_pts)
    struct_color_map = structure_colors((h, w), masks)
    struct_colors = struct_color_map.reshape(-1, 3)
    write_ply(
        out_dir / "target_structure_colored.ply",
        all_valid_pts[all_valid_mask],
        struct_colors[all_valid_mask],
    )

    target_export_mask = valid_hw & (
        masks["plane_core"] | masks["boundary_band"] | masks["line_band"]
    )
    target_pts = pts.reshape(-1, 3)
    target_colors = struct_color_map.reshape(-1, 3)
    target_flat = target_export_mask.reshape(-1)
    write_ply(
        out_dir / "target_plane_only_colored.ply",
        target_pts[target_flat],
        target_colors[target_flat],
    )

    core_pts = pts[masks["plane_core"]]
    core_pts = core_pts[valid_points(core_pts)]
    plane = fit_plane_pca(core_pts, min_points=args.min_fit_points)
    if plane is None:
        raise RuntimeError(
            f"Cannot fit core plane: core valid points={len(core_pts)}, "
            f"min_fit_points={args.min_fit_points}"
        )

    heat_mask = valid_hw & (
        masks["target_plane"] | masks["line_band"] | masks["boundary_band"]
    )
    heat_pts = pts.reshape(-1, 3)
    heat_flat = heat_mask.reshape(-1)
    heat_dist = np.full((h * w,), np.nan, dtype=np.float32)
    heat_dist[heat_flat] = point_to_plane_dist(heat_pts[heat_flat], plane[0], plane[1])
    heat_colors = heat_colormap(heat_dist[heat_flat])
    write_ply(
        out_dir / "distance_heat_colored.ply",
        heat_pts[heat_flat],
        heat_colors,
    )

    meta_path = out_dir / "export_info.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"sample_idx: {args.sample_idx}\n")
        f.write(f"plane_id: {args.plane_id}\n")
        f.write(f"candidate_csv: {args.candidate_csv}\n")
        f.write(f"candidate_score: {candidate.get('score', '')}\n")
        f.write(f"scene: {sample_info.get('scene_name', '')}\n")
        f.write(f"rgb_path: {sample_info['rgb_path']}\n")
        f.write(f"pair_rgb_path: {pair_info['rgb_path']}\n")
        f.write(f"json_path: {sample_info['json_path']}\n")
        f.write(f"image_size: {args.image_size}\n")
        f.write(f"conf_thr: {args.conf_thr}\n")
        f.write(f"line_band_radius: {args.line_band_radius}\n")
        f.write(f"boundary_band_radius: {args.boundary_band_radius}\n")
        f.write(f"full_conf_points: {len(full_pts)}\n")
        f.write(f"target_plane_points: {int(target_flat.sum())}\n")

    print("Saved:")
    print(out_dir / "full_conf1p5.ply")
    print(out_dir / "target_structure_colored.ply")
    print(out_dir / "target_plane_only_colored.ply")
    print(out_dir / "distance_heat_colored.ply")
    print(meta_path)


def main():
    parser = argparse.ArgumentParser("Export PLY visualizations for a mined structure badcase")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--sample_idx", type=int, required=True)
    parser.add_argument("--plane_id", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--conf_thr", type=float, default=1.5)
    parser.add_argument("--line_band_radius", type=int, default=4)
    parser.add_argument("--boundary_band_radius", type=int, default=4)

    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--min_fit_points", type=int, default=50)

    args = parser.parse_args()
    export_case(args)


if __name__ == "__main__":
    main()
