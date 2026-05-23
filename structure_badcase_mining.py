import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
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


def safe_name(text):
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "case"


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def parse_room_view(sample_info):
    json_path = Path(sample_info["json_path"])
    parts = json_path.parts
    room = ""
    view = str(sample_info.get("view_id", ""))

    if "2D_rendering" in parts:
        idx = parts.index("2D_rendering")
        if idx + 1 < len(parts):
            room = parts[idx + 1]
        if idx + 5 < len(parts):
            view = parts[idx + 5]

    return room, view


def group_key_from_sample(sample_info):
    room, _ = parse_room_view(sample_info)
    scene = sample_info.get("scene_name", "unknown_scene")
    return f"{scene}/{room}" if room else scene


def build_pair_index(dataset):
    groups = defaultdict(list)
    for idx, info in enumerate(dataset.samples):
        groups[group_key_from_sample(info)].append(idx)

    pair_idx = {}
    for _, indices in groups.items():
        indices = sorted(indices)
        if len(indices) < 2:
            continue
        for pos, idx in enumerate(indices):
            pair_idx[idx] = indices[(pos + 1) % len(indices)]
    return pair_idx


def img_tensor_to_rgb(img_chw):
    img = to_numpy(img_chw)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def valid_points(points, max_abs_coord=1e5):
    points = np.asarray(points)
    mask = np.isfinite(points).all(axis=-1)
    mask &= np.max(np.abs(points), axis=-1) < max_abs_coord
    return mask


def robust_diag(points):
    points = np.asarray(points, dtype=np.float32)
    points = points[valid_points(points)]
    if len(points) < 10:
        return 1.0
    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)
    return max(float(np.linalg.norm(hi - lo)), 1e-6)


def fit_plane_pca(points, min_points=20):
    points = np.asarray(points, dtype=np.float32)
    points = points[valid_points(points)]
    if len(points) < min_points:
        return None

    center = points.mean(axis=0)
    x = points - center
    try:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    normal = vh[-1].astype(np.float32)
    normal /= np.linalg.norm(normal) + 1e-8
    return center.astype(np.float32), normal


def point_to_plane_dist(points, center, normal):
    return np.abs((points - center[None, :]) @ normal)


def fit_line_pca(points, min_points=20):
    points = np.asarray(points, dtype=np.float32)
    points = points[valid_points(points)]
    if len(points) < min_points:
        return None

    center = points.mean(axis=0)
    x = points - center
    try:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    direction = vh[0].astype(np.float32)
    direction /= np.linalg.norm(direction) + 1e-8
    return center.astype(np.float32), direction


def point_to_line_dist(points, center, direction):
    vec = points - center[None, :]
    proj = (vec @ direction)[:, None] * direction[None, :]
    return np.linalg.norm(vec - proj, axis=1)


def intersect_planes(plane_a, plane_b):
    if plane_a is None or plane_b is None:
        return None

    center_a, normal_a = plane_a
    center_b, normal_b = plane_b
    direction = np.cross(normal_a, normal_b).astype(np.float32)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None
    direction /= norm

    # Solve n1.x=d1, n2.x=d2, direction.x=0 for the point nearest origin.
    mat = np.stack([normal_a, normal_b, direction], axis=0).astype(np.float32)
    rhs = np.array([normal_a @ center_a, normal_b @ center_b, 0.0], dtype=np.float32)
    try:
        point = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return None
    return point.astype(np.float32), direction


def mask_boundary(mask):
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1).astype(bool)
    return mask & ~eroded


def dilate(mask, radius):
    if radius <= 0:
        return mask.astype(bool)
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def resize_hw(arr, size, interpolation):
    h, w = size
    return cv2.resize(arr, (w, h), interpolation=interpolation)


def colorize_label(label):
    label = label.astype(np.int64)
    out = np.zeros((*label.shape, 3), dtype=np.float32)
    ids = [int(x) for x in np.unique(label) if int(x) > 0]
    rng = np.random.default_rng(20260523)
    colors = rng.uniform(0.18, 0.95, size=(max(len(ids), 1), 3)).astype(np.float32)
    for i, pid in enumerate(ids):
        out[label == pid] = colors[i % len(colors)]
    return out


def normalize_for_show(arr):
    arr = np.asarray(arr, dtype=np.float32)
    out = arr.copy()
    finite = np.isfinite(out)
    if finite.any():
        lo = np.percentile(out[finite], 2)
        hi = np.percentile(out[finite], 98)
        out = (out - lo) / max(hi - lo, 1e-8)
    out[~np.isfinite(out)] = 0
    return np.clip(out, 0, 1)


@torch.no_grad()
def run_pair_dust3r(model, rgb1_path, rgb2_path, image_size, device, batch_size=1):
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    imgs = load_images([rgb1_path, rgb2_path], size=image_size, verbose=False)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=False)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=False)
    pts = output["pred1"]["pts3d"][0]
    conf = output["pred1"]["conf"][0]
    return pts.detach().cpu().numpy(), conf.detach().cpu().numpy()


def load_gt_masks(dataset, sample_info, image_shape):
    with open(sample_info["json_path"], "r", encoding="utf-8") as f:
        layout = json.load(f)
    plane_mask, line_mask = dataset.generate_masks(layout)
    h, w = image_shape
    plane_mask = resize_hw(plane_mask, (h, w), cv2.INTER_NEAREST).astype(np.int32)
    line_mask = resize_hw(line_mask, (h, w), cv2.INTER_NEAREST).astype(np.uint8)
    return plane_mask, line_mask > 0


def plane_regions(plane_mask, line_mask, plane_id, line_radius, boundary_radius):
    mask = plane_mask == plane_id
    edge = mask_boundary(mask)
    line_band = dilate(line_mask, line_radius)
    boundary_band = dilate(edge, boundary_radius)
    boundary = mask & (line_band | boundary_band)
    core = mask & ~dilate(line_mask | edge, max(line_radius, boundary_radius))
    return core, boundary, line_band


def p90_norm(dist, diag):
    dist = np.asarray(dist, dtype=np.float32)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return float("nan")
    return float(np.percentile(dist, 90) / max(diag, 1e-6))


def line_band_for_plane(plane_mask, line_band, plane_id, radius):
    near_plane = dilate(plane_mask == plane_id, radius)
    return line_band & near_plane


def adjacent_plane_ids(plane_mask, line_near, plane_id, radius):
    around = dilate(line_near, radius)
    ids = sorted(int(x) for x in np.unique(plane_mask[around]) if int(x) > 0 and int(x) != plane_id)
    return ids


def score_candidate(row):
    def finite_or(value, default):
        return float(value) if np.isfinite(value) else default

    core = finite_or(row["core_flatness_p90_norm"], 1.0)
    ratio = finite_or(row["boundary_to_core_ratio"], 0.0)
    line = finite_or(row["line_dist_p90_norm"], 0.0)
    inter = finite_or(row["intersection_dist_p90_norm"], 0.0)

    core_score = 1.0 / (1.0 + max(core, 0.0) / 0.006)
    ratio_score = np.log1p(max(ratio, 0.0)) / np.log(11.0)
    line_score = min(max(line, 0.0) / 0.025, 2.0)
    inter_score = min(max(inter, 0.0) / 0.025, 2.0)
    return float(core_score * (0.50 * ratio_score + 0.35 * line_score + 0.15 * inter_score))


def analyze_plane(pts, conf, plane_mask, line_mask, plane_id, args, fitted_planes):
    core, boundary, line_band = plane_regions(
        plane_mask,
        line_mask,
        plane_id,
        line_radius=args.line_band_radius,
        boundary_radius=args.boundary_band_radius,
    )

    if int((plane_mask == plane_id).sum()) < args.min_plane_pixels:
        return None
    if int(core.sum()) < args.min_core_pixels or int(boundary.sum()) < args.min_boundary_pixels:
        return None

    core_pts = pts[core]
    boundary_pts = pts[boundary]
    core_valid = valid_points(core_pts)
    boundary_valid = valid_points(boundary_pts)
    if int(core_valid.sum()) < args.min_core_pixels or int(boundary_valid.sum()) < args.min_boundary_pixels:
        return None

    core_pts = core_pts[core_valid]
    boundary_pts = boundary_pts[boundary_valid]
    plane = fit_plane_pca(core_pts, min_points=args.min_fit_points)
    if plane is None:
        return None
    fitted_planes[plane_id] = plane

    center, normal = plane
    diag = robust_diag(np.concatenate([core_pts, boundary_pts], axis=0))
    core_dist = point_to_plane_dist(core_pts, center, normal)
    boundary_dist = point_to_plane_dist(boundary_pts, center, normal)
    core_flatness = p90_norm(core_dist, diag)
    boundary_flatness = p90_norm(boundary_dist, diag)
    boundary_to_core_ratio = float(boundary_flatness / max(core_flatness, 1e-6))

    line_near = line_band_for_plane(
        plane_mask,
        line_band,
        plane_id,
        radius=args.line_context_radius,
    )
    line_pts = pts[line_near]
    line_pts = line_pts[valid_points(line_pts)]
    line_model = fit_line_pca(line_pts, min_points=args.min_line_points)
    if line_model is None:
        line_dist_p90_norm = float("nan")
        line_dist_map = np.full(plane_mask.shape, np.nan, dtype=np.float32)
    else:
        line_dist = point_to_line_dist(line_pts, line_model[0], line_model[1])
        line_dist_p90_norm = p90_norm(line_dist, diag)
        line_dist_map = np.full(plane_mask.shape, np.nan, dtype=np.float32)
        valid_line_flat = valid_points(pts.reshape(-1, 3))
        all_dist = np.full((pts.shape[0] * pts.shape[1],), np.nan, dtype=np.float32)
        all_dist[valid_line_flat] = point_to_line_dist(
            pts.reshape(-1, 3)[valid_line_flat],
            line_model[0],
            line_model[1],
        )
        line_dist_map = all_dist.reshape(plane_mask.shape)
        line_dist_map[~line_near] = np.nan

    intersection_dist_p90_norm = float("nan")
    best_intersection_map = np.full(plane_mask.shape, np.nan, dtype=np.float32)
    for other_id in adjacent_plane_ids(plane_mask, line_near, plane_id, radius=args.intersection_context_radius):
        other_core, _, _ = plane_regions(
            plane_mask,
            line_mask,
            other_id,
            line_radius=args.line_band_radius,
            boundary_radius=args.boundary_band_radius,
        )
        if int(other_core.sum()) < args.min_core_pixels:
            continue
        if other_id not in fitted_planes:
            other_pts = pts[other_core]
            other_pts = other_pts[valid_points(other_pts)]
            fitted_planes[other_id] = fit_plane_pca(other_pts, min_points=args.min_fit_points)

        intersection = intersect_planes(plane, fitted_planes.get(other_id))
        if intersection is None:
            continue

        pair_line = line_near & dilate(plane_mask == other_id, args.line_context_radius)
        pair_pts = pts[pair_line]
        pair_pts = pair_pts[valid_points(pair_pts)]
        if len(pair_pts) < args.min_line_points:
            continue

        dist = point_to_line_dist(pair_pts, intersection[0], intersection[1])
        cur = p90_norm(dist, diag)
        if not np.isfinite(intersection_dist_p90_norm) or cur > intersection_dist_p90_norm:
            intersection_dist_p90_norm = cur
            all_dist = np.full((pts.shape[0] * pts.shape[1],), np.nan, dtype=np.float32)
            valid_flat = valid_points(pts.reshape(-1, 3))
            all_dist[valid_flat] = point_to_line_dist(
                pts.reshape(-1, 3)[valid_flat],
                intersection[0],
                intersection[1],
            )
            best_intersection_map = all_dist.reshape(plane_mask.shape)
            best_intersection_map[~pair_line] = np.nan

    dist_map = np.full(plane_mask.shape, np.nan, dtype=np.float32)
    valid_flat = valid_points(pts.reshape(-1, 3))
    flat_dist = np.full((pts.shape[0] * pts.shape[1],), np.nan, dtype=np.float32)
    flat_dist[valid_flat] = point_to_plane_dist(pts.reshape(-1, 3)[valid_flat], center, normal)
    dist_map = flat_dist.reshape(plane_mask.shape)
    dist_map[~(core | boundary)] = np.nan

    row = {
        "plane_id": int(plane_id),
        "core_flatness_p90_norm": core_flatness,
        "boundary_flatness_p90_norm": boundary_flatness,
        "boundary_to_core_ratio": boundary_to_core_ratio,
        "line_dist_p90_norm": line_dist_p90_norm,
        "intersection_dist_p90_norm": intersection_dist_p90_norm,
        "conf_mean_core": float(np.nanmean(conf[core])),
        "conf_mean_boundary": float(np.nanmean(conf[boundary])),
    }
    row["score"] = score_candidate(row)

    panel_cache = {
        "core": core,
        "boundary": boundary,
        "line_near": line_near,
        "plane_dist": dist_map,
        "line_dist": line_dist_map,
        "intersection_dist": best_intersection_map,
    }
    return row, panel_cache


def analyze_sample(sample_idx, dataset, pair_idx, model, device, args):
    if sample_idx not in pair_idx:
        return [], None

    sample = dataset[sample_idx]
    sample_info = dataset.samples[sample_idx]
    pair_info = dataset.samples[pair_idx[sample_idx]]

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

    plane_mask, line_mask = load_gt_masks(dataset, sample_info, (h, w))
    if conf.shape[:2] != (h, w):
        conf = resize_hw(conf, (h, w), cv2.INTER_LINEAR)

    scene = sample_info.get("scene_name", "")
    room, view = parse_room_view(sample_info)
    fitted_planes = {}
    rows = []
    panels = {}

    plane_ids = [int(x) for x in np.unique(plane_mask) if int(x) > 0]
    for plane_id in plane_ids:
        result = analyze_plane(pts, conf, plane_mask, line_mask, plane_id, args, fitted_planes)
        if result is None:
            continue
        row, panel_cache = result
        row.update({
            "sample_idx": int(sample_idx),
            "scene": scene,
            "room": room,
            "view": view,
            "rgb_path": sample_info["rgb_path"],
            "pair_rgb_path": pair_info["rgb_path"],
            "json_path": sample_info["json_path"],
        })
        rows.append(row)
        panels[int(plane_id)] = panel_cache

    cache = {
        "rgb": rgb,
        "plane_mask": plane_mask,
        "line_mask": line_mask,
        "conf": conf,
        "panels": panels,
    }
    return rows, cache


def make_panel(row, cache, output_path):
    import matplotlib.pyplot as plt

    plane_id = int(row["plane_id"])
    panel = cache["panels"][plane_id]
    plane_mask = cache["plane_mask"]
    line_mask = cache["line_mask"]

    imgs = [
        ("RGB", cache["rgb"], None),
        ("Plane mask", colorize_label(plane_mask), None),
        ("Line mask", line_mask.astype(np.float32), "gray"),
        ("Confidence", normalize_for_show(cache["conf"]), "gray"),
        ("Core mask", panel["core"].astype(np.float32), "gray"),
        ("Boundary mask", panel["boundary"].astype(np.float32), "gray"),
        ("Point-to-plane dist", normalize_for_show(panel["plane_dist"]), "magma"),
        ("Point-to-line dist", normalize_for_show(panel["line_dist"]), "magma"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.reshape(-1)
    for ax, (title, img, cmap) in zip(axes, imgs):
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    title = (
        f"sample={row['sample_idx']} scene={row['scene']} room={row['room']} "
        f"view={row['view']} plane={plane_id} score={row['score']:.4f}\n"
        f"core={row['core_flatness_p90_norm']:.5f} "
        f"boundary/core={row['boundary_to_core_ratio']:.2f} "
        f"line={row['line_dist_p90_norm']:.5f} "
        f"intersection={row['intersection_dist_p90_norm']:.5f}"
    )
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_candidates(rows, csv_path):
    fieldnames = [
        "sample_idx",
        "scene",
        "room",
        "view",
        "plane_id",
        "core_flatness_p90_norm",
        "boundary_flatness_p90_norm",
        "boundary_to_core_ratio",
        "line_dist_p90_norm",
        "intersection_dist_p90_norm",
        "conf_mean_core",
        "conf_mean_boundary",
        "score",
        "rgb_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_dataset(args):
    return Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        input_mode="single",
    )


def main():
    parser = argparse.ArgumentParser(
        "Mine Structured3D structure-prior badcases for DUSt3R baseline"
    )
    parser.add_argument("--root_dir", "--structured3d_root", required=True)
    parser.add_argument("--weights_path", "--dust3r_checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--save_top_panels", type=int, default=20)

    parser.add_argument("--min_plane_pixels", type=int, default=5000)
    parser.add_argument("--min_core_pixels", type=int, default=800)
    parser.add_argument("--min_boundary_pixels", type=int, default=200)
    parser.add_argument("--min_fit_points", type=int, default=50)
    parser.add_argument("--min_line_points", type=int, default=50)
    parser.add_argument("--line_band_radius", type=int, default=4)
    parser.add_argument("--boundary_band_radius", type=int, default=4)
    parser.add_argument("--line_context_radius", type=int, default=8)
    parser.add_argument("--intersection_context_radius", type=int, default=12)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Structured3D Structure Badcase Mining")
    print("=" * 80)
    print("device      :", device)
    print("root_dir    :", args.root_dir)
    print("weights_path:", args.weights_path)
    print("split       :", args.split)
    print("num_samples :", args.num_samples)
    print("image_size  :", args.image_size)
    print("output_dir  :", out_dir)

    dataset = build_dataset(args)
    pair_idx = build_pair_index(dataset)
    print("dataset size    :", len(dataset))
    print("pairable samples:", len(pair_idx))

    from dust3r.model import load_model

    model = load_model(args.weights_path, device=device)
    model.eval()

    all_rows = []
    cache_by_sample = {}
    end_idx = min(len(dataset), args.start_idx + args.num_samples)

    for idx in range(args.start_idx, end_idx):
        if idx not in pair_idx:
            continue
        try:
            rows, cache = analyze_sample(idx, dataset, pair_idx, model, device, args)
        except Exception as exc:
            print(f"[Skip] idx={idx}, error={exc!r}")
            continue
        if not rows:
            continue
        all_rows.extend(rows)
        cache_by_sample[idx] = cache
        if (idx - args.start_idx + 1) % 10 == 0:
            print(f"[{idx + 1}/{end_idx}] candidate rows={len(all_rows)}")

    if not all_rows:
        raise RuntimeError(
            "No candidates found. Try increasing num_samples or lowering min_* thresholds."
        )

    rows_sorted = sorted(all_rows, key=lambda x: x["score"], reverse=True)
    csv_path = out_dir / "candidates.csv"
    write_candidates(rows_sorted, csv_path)

    print("=" * 80)
    print("Saved candidates:", csv_path)
    print("rows:", len(rows_sorted))
    print("=" * 80)
    for rank, row in enumerate(rows_sorted[: args.top_k], start=1):
        print(
            f"[{rank:03d}] score={row['score']:.4f} idx={row['sample_idx']} "
            f"scene={row['scene']} room={row['room']} view={row['view']} "
            f"plane={row['plane_id']} core={row['core_flatness_p90_norm']:.5f} "
            f"ratio={row['boundary_to_core_ratio']:.2f} "
            f"line={row['line_dist_p90_norm']:.5f} "
            f"intersection={row['intersection_dist_p90_norm']:.5f}"
        )

    panel_dir = out_dir / "top_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(rows_sorted[: args.save_top_panels], start=1):
        sample_idx = int(row["sample_idx"])
        cache = cache_by_sample.get(sample_idx)
        if cache is None:
            _, cache = analyze_sample(sample_idx, dataset, pair_idx, model, device, args)

        name = safe_name(
            f"top_{rank:03d}_idx{sample_idx}_scene{row['scene']}_room{row['room']}"
            f"_view{row['view']}_plane{row['plane_id']}_score{row['score']:.3f}.png"
        )
        make_panel(row, cache, panel_dir / name)

    print("Saved panels:", panel_dir)


if __name__ == "__main__":
    main()
