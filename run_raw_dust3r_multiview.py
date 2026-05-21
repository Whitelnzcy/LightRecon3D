import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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

from dust3r.model import load_model
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


# ============================================================
# Basic utilities
# ============================================================

def safe_tag(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "run"


def collect_images(image_dir=None, image_paths=None, max_images=8):
    exts = {".jpg", ".jpeg", ".png"}
    paths = []

    if image_paths:
        paths.extend([Path(p) for p in image_paths])

    if image_dir:
        root = Path(image_dir)
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in exts:
                paths.append(p)

    seen = set()
    uniq = []
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)

    if max_images > 0:
        uniq = uniq[:max_images]

    if len(uniq) < 2:
        raise RuntimeError(f"Need at least 2 images, got {len(uniq)}")

    return [str(p) for p in uniq]


def _parse_pair_set(pair_list):
    if pair_list is None:
        return None

    pair_set = set()

    for item in pair_list:
        text = str(item).strip()
        if not text:
            continue

        parts = re.split(r"[-,]", text)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid pair format: {item!r}. Expected format like 0-1."
            )

        try:
            i = int(parts[0])
            j = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid pair format: {item!r}. Expected integer indices."
            ) from exc

        if i == j:
            raise ValueError(f"Invalid self pair: {item!r}")

        pair_set.add((min(i, j), max(i, j)))

    return pair_set


def _get_view_idx(view):
    def _to_int(value):
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Cannot convert tensor idx with shape {tuple(value.shape)}")
            return int(value.detach().cpu().item())

        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(f"Cannot convert ndarray idx with shape {value.shape}")
            return int(value.item())

        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Cannot convert sequence idx with length {len(value)}")
            return _to_int(value[0])

        if isinstance(value, (int, np.integer)):
            return int(value)

        if isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"[+-]?\d+", text):
                return int(text)
            match = re.search(r"(?:^|[_\-/])(\d+)(?:$|[_\-.])", text)
            if match:
                return int(match.group(1))

        raise ValueError(f"Cannot convert view index from value={value!r}")

    if "idx" in view:
        return _to_int(view["idx"])

    if "instance" in view:
        return _to_int(view["instance"])

    print("Cannot find view index. view.keys():", list(view.keys()))
    raise KeyError("DUSt3R view dict has neither 'idx' nor 'instance'")


def filter_pairs_by_args(pairs, keep_pairs=None, drop_pairs=None):
    keep_set = _parse_pair_set(keep_pairs)
    drop_set = _parse_pair_set(drop_pairs)

    original_count = len(pairs)
    filtered = []

    for pair in pairs:
        view1, view2 = pair
        i = _get_view_idx(view1)
        j = _get_view_idx(view2)
        key = (min(i, j), max(i, j))

        if keep_set is not None and key not in keep_set:
            continue
        if drop_set is not None and key in drop_set:
            continue

        filtered.append(pair)

    print("original pair count:", original_count)
    print("keep_pairs:", sorted(keep_set) if keep_set is not None else None)
    print("drop_pairs:", sorted(drop_set) if drop_set is not None else None)
    print("filtered pair count:", len(filtered))

    if len(filtered) == 0:
        raise RuntimeError(
            "Pair filtering removed all pairs. "
            f"keep_pairs={keep_pairs}, drop_pairs={drop_pairs}"
        )

    return filtered


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_rgb(img):
    img = to_numpy(img)

    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    img = img.astype(np.float32)

    if img.max() > 2.0:
        img = img / 255.0

    return np.clip(img, 0.0, 1.0)


def write_ply(path, points, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)

    if len(points) == 0:
        raise RuntimeError(f"No points to write: {path}")

    finite = np.isfinite(points).all(axis=1)
    finite &= np.max(np.abs(points), axis=1) < 1e5

    points = points[finite]
    colors = colors[finite]

    if len(points) == 0:
        raise RuntimeError(f"No finite points after filtering: {path}")

    if colors.max() <= 1.0:
        colors = colors * 255.0

    colors = np.clip(colors, 0, 255).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, colors):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def save_gray_png(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - np.nanmin(arr)
    arr = arr / (np.nanmax(arr) + 1e-8)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)

    Image.fromarray(arr).save(path)


def save_mask_png(path, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(mask).astype(np.uint8) * 255
    Image.fromarray(mask).save(path)


# ============================================================
# Plane geometry utilities
# ============================================================

def fit_plane_pca(points):
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]

    if len(points) < 20:
        return None

    center = points.mean(axis=0)
    x = points - center

    try:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    normal = vh[-1].astype(np.float32)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    return center.astype(np.float32), normal


def plane_from_3pts(a, b, c):
    normal = np.cross(b - a, c - a)
    norm = np.linalg.norm(normal)

    if norm < 1e-8:
        return None

    normal = normal / norm
    return a.astype(np.float32), normal.astype(np.float32)


def point_to_plane_dist(points, center, normal):
    return np.abs((points - center[None, :]) @ normal)


def extract_planes_ransac(
    seed_points,
    max_planes=5,
    max_seed_points=20000,
    dist_ratio=0.006,
    num_iters=120,
    min_inliers=1200,
    min_inlier_ratio=0.03,
):
    seed_points = np.asarray(seed_points, dtype=np.float32)
    seed_points = seed_points[np.isfinite(seed_points).all(axis=1)]

    if len(seed_points) < min_inliers:
        return [], None

    if len(seed_points) > max_seed_points:
        idx = np.linspace(0, len(seed_points) - 1, max_seed_points).astype(np.int64)
        seed_points = seed_points[idx]

    lo = np.percentile(seed_points, 1, axis=0)
    hi = np.percentile(seed_points, 99, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    dist_thr = max(diag * dist_ratio, 1e-4)

    rng = np.random.default_rng(12345)
    remaining = seed_points.copy()
    planes = []

    for _ in range(max_planes):
        n = len(remaining)

        if n < min_inliers:
            break

        best_inliers = None
        best_count = 0

        for _ in range(num_iters):
            ids = rng.choice(n, size=3, replace=False)
            plane = plane_from_3pts(remaining[ids[0]], remaining[ids[1]], remaining[ids[2]])

            if plane is None:
                continue

            center, normal = plane
            dist = point_to_plane_dist(remaining, center, normal)
            inliers = dist < dist_thr
            count = int(inliers.sum())

            if count > best_count:
                best_count = count
                best_inliers = inliers

        min_need = max(min_inliers, int(n * min_inlier_ratio))

        if best_inliers is None or best_count < min_need:
            break

        refined = fit_plane_pca(remaining[best_inliers])

        if refined is None:
            break

        planes.append(refined)
        remaining = remaining[~best_inliers]

    return planes, dist_thr


def depth_jump_risk_np(pts_hw3, quantile=0.985):
    pts = np.asarray(pts_hw3, dtype=np.float32)
    h, w, _ = pts.shape

    finite = np.isfinite(pts).all(axis=-1)
    jump = np.zeros((h, w), dtype=np.float32)

    dx = np.linalg.norm(pts[:, 1:] - pts[:, :-1], axis=-1)
    valid_x = finite[:, 1:] & finite[:, :-1]
    dx = np.where(valid_x, dx, 0.0)
    jump[:, 1:] = np.maximum(jump[:, 1:], dx)
    jump[:, :-1] = np.maximum(jump[:, :-1], dx)

    dy = np.linalg.norm(pts[1:, :] - pts[:-1, :], axis=-1)
    valid_y = finite[1:, :] & finite[:-1, :]
    dy = np.where(valid_y, dy, 0.0)
    jump[1:, :] = np.maximum(jump[1:, :], dy)
    jump[:-1, :] = np.maximum(jump[:-1, :], dy)

    vals = jump[finite]
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]

    if len(vals) == 0:
        return np.zeros((h, w), dtype=bool), 0.0

    thr = float(np.quantile(vals, quantile))
    return jump >= thr, thr


def dilate_bool_mask_np(mask, radius=1):
    if radius <= 0:
        return mask.astype(bool)

    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    k = 2 * radius + 1
    out = F.max_pool2d(t, kernel_size=k, stride=1, padding=radius)
    return (out[0, 0].numpy() > 0.5)


# ============================================================
# Plane confidence calibration
# ============================================================

def calibrate_conf_single_map_by_plane_core(
    pts_np,
    conf_np,
    seed_quantile=0.70,
    dist_ratio=0.006,
    near_scale=1.5,
    plane_floor=0.80,
    blend_alpha=0.50,
    max_planes=5,
    num_iters=120,
    max_seed_points=20000,
    min_inliers=1200,
    exclude_jump_quantile=0.985,
    exclude_jump_dilate=1,
):
    """
    输入：
        pts_np  : [H, W, 3]
        conf_np : [H, W]

    输出：
        new_conf        : 校准后的 confidence
        boost_mask      : 哪些像素被提高了 confidence
        stat            : 统计信息

    逻辑：
        只提高几何上一致的平面区域 confidence。
        不降低任何区域 confidence。
    """
    pts_np = np.asarray(pts_np, dtype=np.float32)
    conf_np = np.asarray(conf_np, dtype=np.float32)

    h, w, _ = pts_np.shape

    new_conf = conf_np.copy()
    boost_mask = np.zeros((h, w), dtype=bool)

    valid = np.isfinite(pts_np).all(axis=-1)
    valid &= np.isfinite(conf_np)
    valid &= np.max(np.abs(pts_np), axis=-1) < 1e5

    valid_conf = conf_np[valid]
    valid_conf = valid_conf[np.isfinite(valid_conf)]

    if len(valid_conf) < min_inliers:
        return new_conf, boost_mask, {
            "planes": 0,
            "valid": int(valid.sum()),
            "boosted": 0,
            "seed_thr": 0.0,
            "dist_thr": 0.0,
        }

    seed_thr = float(np.quantile(valid_conf, seed_quantile))
    seed_mask = valid & (conf_np >= seed_thr)
    seed_points = pts_np[seed_mask].reshape(-1, 3)

    planes, dist_thr = extract_planes_ransac(
        seed_points,
        max_planes=max_planes,
        max_seed_points=max_seed_points,
        dist_ratio=dist_ratio,
        num_iters=num_iters,
        min_inliers=min_inliers,
        min_inlier_ratio=0.03,
    )

    if dist_thr is None or len(planes) == 0:
        return new_conf, boost_mask, {
            "planes": 0,
            "valid": int(valid.sum()),
            "boosted": 0,
            "seed_thr": seed_thr,
            "dist_thr": 0.0,
        }

    # 这里只是避免在强遮挡/跳变边界上升权，不是降权
    jump_risk, _ = depth_jump_risk_np(pts_np, quantile=exclude_jump_quantile)
    jump_risk = dilate_bool_mask_np(jump_risk, radius=exclude_jump_dilate)

    pts_flat = pts_np.reshape(-1, 3)
    conf_flat = conf_np.reshape(-1)
    new_flat = new_conf.reshape(-1)
    valid_flat = valid.reshape(-1)
    jump_flat = jump_risk.reshape(-1)
    boost_flat = boost_mask.reshape(-1)

    used_planes = 0
    boosted_total = 0

    for center, normal in planes:
        dist = point_to_plane_dist(pts_flat, center, normal)

        close = valid_flat & (dist <= dist_thr * near_scale) & (~jump_flat)

        if int(close.sum()) < min_inliers:
            continue

        core = close & (conf_flat >= seed_thr)

        if int(core.sum()) < min_inliers:
            continue

        plane_conf = float(np.median(conf_flat[core]))
        target_conf = plane_floor * plane_conf

        if not np.isfinite(target_conf) or target_conf <= 0:
            continue

        to_boost = close & (new_flat < target_conf)

        if int(to_boost.sum()) == 0:
            continue

        # 只往 target_conf 拉近，不暴力设成 target_conf
        target_vals = np.maximum(new_flat[to_boost], target_conf)
        new_flat[to_boost] = (
            (1.0 - blend_alpha) * new_flat[to_boost]
            + blend_alpha * target_vals
        )

        boost_flat[to_boost] = True
        used_planes += 1
        boosted_total += int(to_boost.sum())

    return new_flat.reshape(h, w), boost_mask, {
        "planes": used_planes,
        "valid": int(valid.sum()),
        "boosted": boosted_total,
        "seed_thr": seed_thr,
        "dist_thr": float(dist_thr),
    }


def calibrate_output_conf_by_plane_core(
    output,
    output_dir,
    run_tag,
    seed_quantile=0.70,
    dist_ratio=0.006,
    near_scale=1.5,
    plane_floor=0.80,
    blend_alpha=0.50,
    max_planes=5,
    num_iters=120,
    max_seed_points=20000,
    min_inliers=1200,
    exclude_jump_quantile=0.985,
    exclude_jump_dilate=1,
    save_debug=False,
    debug_items=6,
):
    output_dir = Path(output_dir)
    debug_dir = output_dir / f"{run_tag}__plane_conf_calib_debug"

    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    total_valid = 0
    total_boosted = 0
    total_planes = 0

    with torch.no_grad():
        branches = [
            ("pred1", "pts3d", "conf"),
            ("pred2", "pts3d_in_other_view", "conf"),
        ]

        for branch_name, pts_key, conf_key in branches:
            pts = output[branch_name][pts_key]
            conf = output[branch_name][conf_key]

            device = conf.device
            dtype = conf.dtype

            pts_cpu = pts.detach().cpu().numpy()
            conf_cpu = conf.detach().cpu().numpy()
            new_conf_cpu = conf_cpu.copy()

            batch_size = conf_cpu.shape[0]

            for b in range(batch_size):
                new_b, boost_mask, stat = calibrate_conf_single_map_by_plane_core(
                    pts_np=pts_cpu[b],
                    conf_np=conf_cpu[b],
                    seed_quantile=seed_quantile,
                    dist_ratio=dist_ratio,
                    near_scale=near_scale,
                    plane_floor=plane_floor,
                    blend_alpha=blend_alpha,
                    max_planes=max_planes,
                    num_iters=num_iters,
                    max_seed_points=max_seed_points,
                    min_inliers=min_inliers,
                    exclude_jump_quantile=exclude_jump_quantile,
                    exclude_jump_dilate=exclude_jump_dilate,
                )

                new_conf_cpu[b] = new_b

                total_valid += stat["valid"]
                total_boosted += stat["boosted"]
                total_planes += stat["planes"]

                if save_debug and b < debug_items:
                    prefix = debug_dir / f"{branch_name}_pair{b:03d}"
                    save_gray_png(f"{prefix}_old_conf.png", conf_cpu[b])
                    save_gray_png(f"{prefix}_new_conf.png", new_b)
                    save_gray_png(f"{prefix}_delta_conf.png", new_b - conf_cpu[b])
                    save_mask_png(f"{prefix}_boost_mask.png", boost_mask)

            output[branch_name][conf_key] = torch.from_numpy(new_conf_cpu).to(
                device=device,
                dtype=dtype,
            )

            old_mean = float(conf.mean().detach().cpu())
            new_mean = float(output[branch_name][conf_key].mean().detach().cpu())

            print(
                f"[PlaneConfCalib] {branch_name}: "
                f"conf mean {old_mean:.6f} -> {new_mean:.6f}"
            )

    boost_ratio = total_boosted / max(total_valid, 1)

    print("=" * 80)
    print("Plane confidence calibration before global alignment")
    print("=" * 80)
    print(f"seed_quantile       : {seed_quantile}")
    print(f"dist_ratio          : {dist_ratio}")
    print(f"near_scale          : {near_scale}")
    print(f"plane_floor         : {plane_floor}")
    print(f"blend_alpha         : {blend_alpha}")
    print(f"max_planes          : {max_planes}")
    print(f"num_iters           : {num_iters}")
    print(f"total used planes   : {total_planes}")
    print(f"total valid pixels  : {total_valid}")
    print(f"total boosted pixels: {total_boosted}")
    print(f"boost ratio         : {boost_ratio:.6f}")

    if save_debug:
        print(f"[PlaneConfCalib] debug saved to: {debug_dir}")

    return output


# ============================================================
# PLY export
# ============================================================

def export_scene_ply(scene, output_path, min_conf=1.5, max_points=1000000):
    pts3d = scene.get_pts3d()
    confs = scene.get_conf()
    imgs = scene.imgs

    all_points = []
    all_colors = []

    for i, (pts, conf, img) in enumerate(zip(pts3d, confs, imgs)):
        pts_np = to_numpy(pts).astype(np.float32)
        conf_np = to_numpy(conf).astype(np.float32)
        img_np = normalize_rgb(img)

        if pts_np.ndim != 3 or pts_np.shape[-1] != 3:
            print(f"[Warn] skip view {i}, bad pts shape: {pts_np.shape}")
            continue

        h, w = pts_np.shape[:2]

        if img_np.shape[:2] != (h, w):
            img_np = img_np[:h, :w]
            pts_np = pts_np[: img_np.shape[0], : img_np.shape[1]]
            conf_np = conf_np[: img_np.shape[0], : img_np.shape[1]]

        mask = np.isfinite(pts_np).all(axis=-1)
        mask &= np.max(np.abs(pts_np), axis=-1) < 1e5
        mask &= conf_np >= min_conf

        p = pts_np[mask].reshape(-1, 3)
        c = img_np[mask].reshape(-1, 3)

        if len(p) == 0:
            print(f"[Export] view {i}: kept 0 points, skip")
            continue

        all_points.append(p)
        all_colors.append(c)

        print(f"[Export] view {i}: kept {len(p)} points, conf >= {min_conf}")

    if len(all_points) == 0:
        raise RuntimeError(f"No points to export with conf >= {min_conf}")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    if max_points > 0 and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[idx]
        colors = colors[idx]

    write_ply(output_path, points, colors)

    print(f"[Saved] {output_path}")
    print(f"[Saved] points: {len(points)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Run raw DUSt3R multi-view reconstruction")

    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--image_paths", nargs="*", default=None)

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_tag", type=str, default="")

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_images", type=int, default=8)

    parser.add_argument("--scene_graph", default="complete")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--keep_pairs",
        nargs="*",
        default=None,
        help="Only keep selected image pairs, format: 0-1 1-2",
    )
    parser.add_argument(
        "--drop_pairs",
        nargs="*",
        default=None,
        help="Drop selected image pairs, format: 0-1 1-2",
    )

    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--schedule", default="cosine")

    parser.add_argument("--max_points", type=int, default=1000000)

    parser.add_argument(
        "--conf_reweight",
        type=str,
        default="none",
        choices=["none", "plane_conf_calib"],
    )

    parser.add_argument(
        "--export_thresholds",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 1.5],
    )

    # Plane confidence calibration params
    parser.add_argument("--plane_seed_quantile", type=float, default=0.70)
    parser.add_argument("--plane_dist_ratio", type=float, default=0.006)
    parser.add_argument("--plane_near_scale", type=float, default=1.5)
    parser.add_argument("--plane_floor", type=float, default=0.80)
    parser.add_argument("--plane_blend_alpha", type=float, default=0.50)

    parser.add_argument("--plane_max_planes", type=int, default=5)
    parser.add_argument("--plane_num_iters", type=int, default=120)
    parser.add_argument("--plane_max_seed_points", type=int, default=20000)
    parser.add_argument("--plane_min_inliers", type=int, default=1200)

    parser.add_argument("--plane_exclude_jump_quantile", type=float, default=0.985)
    parser.add_argument("--plane_exclude_jump_dilate", type=int, default=1)

    parser.add_argument("--save_conf_calib_debug", action="store_true")
    parser.add_argument("--conf_calib_debug_items", type=int, default=6)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = safe_tag(args.run_tag if args.run_tag else out_dir.name)

    image_paths = collect_images(
        image_dir=args.image_dir,
        image_paths=args.image_paths,
        max_images=args.max_images,
    )

    print("=" * 80)
    print("Raw DUSt3R Multi-view Reconstruction")
    print("=" * 80)
    print("device        :", device)
    print("weights       :", args.weights_path)
    print("image_count   :", len(image_paths))
    print("output_dir    :", out_dir)
    print("run_tag       :", run_tag)
    print("conf_reweight :", args.conf_reweight)
    print("export_thr    :", args.export_thresholds)
    print("=" * 80)

    for i, p in enumerate(image_paths):
        print(f"[{i}] {p}")

    with open(out_dir / f"{run_tag}__used_images.txt", "w", encoding="utf-8") as f:
        for p in image_paths:
            f.write(p + "\n")

    print("=" * 80)
    print("Loading model")
    print("=" * 80)

    model = load_model(args.weights_path, device=device)
    model.eval()

    print("=" * 80)
    print("Loading images")
    print("=" * 80)

    imgs = load_images(image_paths, size=args.image_size, verbose=True)

    print("=" * 80)
    print("Making pairs")
    print("=" * 80)

    pairs = make_pairs(
        imgs,
        scene_graph=args.scene_graph,
        prefilter=None,
        symmetrize=True,
    )

    pairs = filter_pairs_by_args(
        pairs,
        keep_pairs=args.keep_pairs,
        drop_pairs=args.drop_pairs,
    )

    print("=" * 80)
    print("Running pair inference")
    print("=" * 80)

    output = inference(
        pairs,
        model,
        device,
        batch_size=args.batch_size,
        verbose=True,
    )

    if args.conf_reweight == "plane_conf_calib":
        output = calibrate_output_conf_by_plane_core(
            output,
            output_dir=out_dir,
            run_tag=run_tag,
            seed_quantile=args.plane_seed_quantile,
            dist_ratio=args.plane_dist_ratio,
            near_scale=args.plane_near_scale,
            plane_floor=args.plane_floor,
            blend_alpha=args.plane_blend_alpha,
            max_planes=args.plane_max_planes,
            num_iters=args.plane_num_iters,
            max_seed_points=args.plane_max_seed_points,
            min_inliers=args.plane_min_inliers,
            exclude_jump_quantile=args.plane_exclude_jump_quantile,
            exclude_jump_dilate=args.plane_exclude_jump_dilate,
            save_debug=args.save_conf_calib_debug,
            debug_items=args.conf_calib_debug_items,
        )

    print("=" * 80)
    print("Global alignment")
    print("=" * 80)

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

    print("final loss:", loss)

    print("=" * 80)
    print("Export PLY")
    print("=" * 80)

    for thr in args.export_thresholds:
        try:
            export_scene_ply(
                scene,
                output_path=out_dir / f"{run_tag}__{args.conf_reweight}__conf{thr:.1f}.ply",
                min_conf=thr,
                max_points=args.max_points,
            )
        except RuntimeError as e:
            print(f"[Skip export] conf>={thr:.1f}: {e}")


if __name__ == "__main__":
    main()
