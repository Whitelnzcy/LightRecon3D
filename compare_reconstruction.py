import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np


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


METHODS = [
    ("baseline", "Baseline"),
    ("geo_only", "Geo-only"),
    ("geo_anchor", "Geo+Anchor"),
]


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_args():
    parser = argparse.ArgumentParser(
        "Compare baseline / geo-only / geo+anchor pointmap reconstructions"
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default="/data/zhucy23u/datasets/Structured3D",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="/data/zhucy23u/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    )

    parser.add_argument(
        "--baseline_ckpt",
        type=str,
        default="/data/zhucy23u/checkpoints/lightrecon/plane_embedding_512_stable/latest.pth",
    )
    parser.add_argument(
        "--geo_only_ckpt",
        type=str,
        default="/data/zhucy23u/checkpoints/lightrecon/plane_embedding_512_geo0005_from_stable_robust/latest.pth",
    )
    parser.add_argument(
        "--geo_anchor_ckpt",
        type=str,
        default="/data/zhucy23u/checkpoints/lightrecon/plane_embedding_512_geo0005_anchor01/latest.pth",
    )

    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument(
        "--sample_idx",
        type=int,
        nargs="+",
        default=[0],
        help="One or more dataset indices to visualize.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--input_mode", type=str, default="pair", choices=["pair", "single"])
    parser.add_argument("--pair_strategy", type=str, default="adjacent", choices=["adjacent", "all"])

    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/zhucy23u/logs/recon_compare",
    )
    parser.add_argument(
        "--max_abs_coord",
        type=float,
        default=1e4,
        help="Filter absurd pointmap coordinates before writing PLY.",
    )
    parser.add_argument(
        "--top_planes",
        type=int,
        default=3,
        help="Export PLY files for the largest N gt_plane regions. Set 0 to disable.",
    )
    parser.add_argument(
        "--min_plane_pixels",
        type=int,
        default=512,
        help="Ignore plane regions smaller than this after resizing to pointmap size.",
    )
    parser.add_argument(
        "--ply_stride",
        type=int,
        default=1,
        help="Subsample points before writing PLY. 1 keeps all points.",
    )
    parser.add_argument(
        "--allow_partial_load",
        action="store_true",
        help="Load only shape-compatible checkpoint tensors.",
    )

    return parser.parse_args()


def safe_torch_load(path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_state_dict(model, ckpt, allow_partial=False):
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    if not allow_partial:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        return missing, unexpected, []

    model_dict = model.state_dict()
    filtered = {}
    skipped = []

    for key, value in state_dict.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return missing, unexpected, skipped


def load_model_from_ckpt(args, ckpt_path: str, device: torch.device):
    backbone = build_dust3r_backbone(args.weights_path, device=device)

    model = LightReconModel(
        dust3r_backbone=backbone,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    ).to(device)

    ckpt = safe_torch_load(ckpt_path, device)
    missing, unexpected, skipped = load_state_dict(
        model,
        ckpt,
        allow_partial=args.allow_partial_load,
    )

    print(f"[Load] ckpt: {ckpt_path}")
    print(f"[Load] missing keys   : {len(missing)}")
    print(f"[Load] unexpected keys: {len(unexpected)}")
    print(f"[Load] skipped keys   : {len(skipped)}")
    if missing:
        print("[Load] first missing keys:", missing[:10])
    if unexpected:
        print("[Load] first unexpected keys:", unexpected[:10])
    if skipped:
        print("[Load] first skipped keys:", skipped[:10])

    model.eval()
    return model


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def tensor_image_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img, 0.0, 1.0)


def get_pts3d_from_res(res: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Return pointmap as [B, H, W, 3].
    """
    candidate_keys = [
        "pts3d",
        "pts3d_in_other_view",
        "pointmap",
        "pred_pts3d",
    ]

    pts = None
    used_key = None

    for key in candidate_keys:
        if key in res:
            pts = res[key]
            used_key = key
            break

    if pts is None:
        raise KeyError(
            f"Cannot find pointmap in model output. "
            f"Available keys: {list(res.keys())}"
        )

    if pts.ndim != 4:
        raise ValueError(
            f"Expected 4D pointmap, got shape={tuple(pts.shape)} from key={used_key}"
        )

    if pts.shape[1] == 3:
        pts = pts.permute(0, 2, 3, 1).contiguous()

    if pts.shape[-1] != 3:
        raise ValueError(
            f"Expected pointmap with last dim 3, got shape={tuple(pts.shape)} "
            f"from key={used_key}"
        )

    return pts


def resize_rgb_to_points(rgb: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """
    rgb: [3, H, W], range [0,1]
    pts: [H2, W2, 3]
    return [H2, W2, 3], uint8-ish float
    """
    target_h, target_w = pts.shape[:2]
    rgb_bchw = rgb.unsqueeze(0).float()
    if rgb_bchw.shape[-2:] != (target_h, target_w):
        rgb_bchw = F.interpolate(
            rgb_bchw,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
    return rgb_bchw[0].permute(1, 2, 0).contiguous()


def resize_plane_to_points(gt_plane: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """
    gt_plane: [H, W]
    pts: [H2, W2, 3]
    return [H2, W2]
    """
    target_h, target_w = pts.shape[:2]
    plane = gt_plane.unsqueeze(0).unsqueeze(0).float()
    if plane.shape[-2:] != (target_h, target_w):
        plane = F.interpolate(
            plane,
            size=(target_h, target_w),
            mode="nearest",
        )
    return plane[0, 0].long()


def valid_point_mask(pts: torch.Tensor, max_abs_coord: float) -> torch.Tensor:
    return (
        torch.isfinite(pts).all(dim=-1)
        & (pts.abs().amax(dim=-1) < max_abs_coord)
    )


def robust_minmax(x: np.ndarray, mask: np.ndarray, lo_q=2.0, hi_q=98.0):
    valid = x[mask]
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(valid, [lo_q, hi_q])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(valid))
        hi = float(np.nanmax(valid))
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def save_scalar_map(
    values: np.ndarray,
    mask: np.ndarray,
    path: str,
    title: str,
    cmap: str = "viridis",
    value_range=None,
):
    plt = get_pyplot()
    if value_range is None:
        lo, hi = robust_minmax(values, mask)
    else:
        lo, hi = value_range
        if hi <= lo:
            hi = lo + 1e-6
    vis = np.clip((values - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    vis[~mask] = 0.0

    plt.figure(figsize=(6, 6))
    plt.imshow(vis, cmap=cmap)
    plt.title(f"{title}\nrange [{lo:.4g}, {hi:.4g}]")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_rgb(path: str, rgb: np.ndarray, title: str = None):
    plt = get_pyplot()
    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(rgb, 0.0, 1.0))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plane_palette(num_colors=4096) -> np.ndarray:
    rng = np.random.default_rng(12345)
    colors = rng.uniform(0.15, 1.0, size=(num_colors, 3))
    colors[0] = np.array([0.0, 0.0, 0.0])
    return colors


def save_gt_plane_overlay(path: str, rgb: np.ndarray, gt_plane: np.ndarray, alpha=0.45):
    colors = plane_palette()
    plane_ids = np.mod(gt_plane.astype(np.int64), colors.shape[0])
    color_map = colors[plane_ids]
    mask = gt_plane > 0
    overlay = rgb.copy()
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_map[mask]
    save_rgb(path, overlay, title="GT plane overlay")


def write_ply(path: str, points: np.ndarray, colors: np.ndarray):
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same length")

    colors = np.clip(colors, 0, 255).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
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

        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def export_point_cloud(
    path: str,
    pts: torch.Tensor,
    rgb: torch.Tensor,
    mask: torch.Tensor,
    stride: int = 1,
):
    if stride > 1:
        pts = pts[::stride, ::stride]
        rgb = rgb[::stride, ::stride]
        mask = mask[::stride, ::stride]

    points = pts[mask].detach().cpu().float().numpy()
    colors = (rgb[mask].detach().cpu().float().numpy() * 255.0)

    if points.shape[0] == 0:
        print(f"[Warn] no valid points for {path}")
        return 0

    write_ply(path, points, colors)
    return points.shape[0]


def top_plane_ids(
    gt_plane_resized: torch.Tensor,
    valid_mask: torch.Tensor,
    top_k: int,
    min_pixels: int,
) -> List[Tuple[int, int]]:
    if top_k <= 0:
        return []

    plane = gt_plane_resized[valid_mask]
    plane = plane[plane > 0]

    if plane.numel() == 0:
        return []

    ids, counts = torch.unique(plane, return_counts=True)
    keep = counts >= min_pixels
    ids = ids[keep]
    counts = counts[keep]

    if ids.numel() == 0:
        return []

    order = torch.argsort(counts, descending=True)
    out = []
    for idx in order[:top_k]:
        out.append((int(ids[idx].item()), int(counts[idx].item())))
    return out


def plane_fit_metrics(points: torch.Tensor, eps=1e-6) -> Dict[str, float]:
    invalid = {
        "flatness": float("nan"),
        "extent": float("nan"),
        "point_norm": float("nan"),
        "residual_mean": float("nan"),
        "residual_p95": float("nan"),
        "valid": 0,
    }

    if points.ndim != 2 or points.shape[-1] != 3 or points.shape[0] < 3:
        return invalid

    points = points[torch.isfinite(points).all(dim=1)]
    if points.shape[0] < 3:
        return invalid

    center = points.mean(dim=0, keepdim=True)
    x = points - center
    cov = x.transpose(0, 1) @ x / (points.shape[0] + eps)
    cov = 0.5 * (cov + cov.transpose(0, 1))

    try:
        eigvals, eigvecs = torch.linalg.eigh(cov.float())
    except RuntimeError:
        return invalid

    eigvals = torch.clamp(eigvals, min=0.0)
    normal = eigvecs[:, 0].to(points.device, points.dtype)
    residual = torch.abs(x @ normal)
    eig_sum = eigvals.sum()

    return {
        "flatness": float((eigvals[0] / (eig_sum + eps)).detach().cpu().item()),
        "extent": float((eigvals[1] + eigvals[2]).detach().cpu().item()),
        "point_norm": float(points.norm(dim=1).mean().detach().cpu().item()),
        "residual_mean": float(residual.mean().detach().cpu().item()),
        "residual_p95": float(torch.quantile(residual.float(), 0.95).detach().cpu().item()),
        "valid": 1,
    }


def plane_residual_map(
    pts: torch.Tensor,
    gt_plane_resized: torch.Tensor,
    valid_mask: torch.Tensor,
    plane_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    residual_map = torch.full(
        gt_plane_resized.shape,
        float("nan"),
        dtype=pts.dtype,
        device=pts.device,
    )
    output_mask = torch.zeros_like(gt_plane_resized, dtype=torch.bool)
    metrics = {}

    for plane_id in plane_ids:
        mask = valid_mask & (gt_plane_resized == plane_id)
        points = pts[mask]
        metrics[str(plane_id)] = plane_fit_metrics(points)

        if points.shape[0] < 3 or metrics[str(plane_id)]["valid"] == 0:
            continue

        center = points.mean(dim=0, keepdim=True)
        x = points - center
        cov = x.transpose(0, 1) @ x / (points.shape[0] + 1e-6)
        cov = 0.5 * (cov + cov.transpose(0, 1))
        eigvals, eigvecs = torch.linalg.eigh(cov.float())
        normal = eigvecs[:, 0].to(pts.device, pts.dtype)

        residual_map[mask] = torch.abs((pts[mask] - center) @ normal)
        output_mask[mask] = True

    return (
        residual_map.detach().cpu().numpy(),
        output_mask.detach().cpu().numpy(),
        metrics,
    )


@torch.no_grad()
def run_model_on_sample(args, ckpt_path: str, batch, device: torch.device):
    model = load_model_from_ckpt(args, ckpt_path, device)
    view1, view2 = build_views_from_batch(batch, prefix="compare")
    res1, _ = model(view1, view2)
    pts = get_pts3d_from_res(res1)[0].detach()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pts


def process_sample(args, dataset, sample_idx: int, device: torch.device):
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"sample_idx={sample_idx} out of range, dataset size={len(dataset)}")

    sample = dataset[sample_idx]

    batch = {
        "img": sample["img"].unsqueeze(0),
        "gt_line": sample["gt_line"].unsqueeze(0),
        "gt_plane": sample["gt_plane"].unsqueeze(0),
    }
    if "img2" in sample:
        batch.update({
            "img1": sample["img1"].unsqueeze(0),
            "img2": sample["img2"].unsqueeze(0),
            "gt_line1": sample["gt_line1"].unsqueeze(0),
            "gt_line2": sample["gt_line2"].unsqueeze(0),
            "gt_plane1": sample["gt_plane1"].unsqueeze(0),
            "gt_plane2": sample["gt_plane2"].unsqueeze(0),
        })
    batch = move_batch_to_device(batch, device)

    sample_dir = os.path.join(args.output_dir, f"{args.split}_{sample_idx:06d}")
    os.makedirs(sample_dir, exist_ok=True)

    rgb_cpu = batch["img"][0].detach().cpu()
    gt_plane_cpu = batch["gt_plane"][0].detach().cpu().long()

    rgb_np = tensor_image_to_numpy(rgb_cpu)
    gt_plane_np = gt_plane_cpu.numpy()

    save_rgb(os.path.join(sample_dir, "input_rgb.png"), rgb_np, title="Input RGB")
    save_gt_plane_overlay(
        os.path.join(sample_dir, "gt_plane_overlay.png"),
        rgb_np,
        gt_plane_np,
    )

    method_ckpts = {
        "baseline": args.baseline_ckpt,
        "geo_only": args.geo_only_ckpt,
        "geo_anchor": args.geo_anchor_ckpt,
    }

    manifest = {
        "sample_idx": sample_idx,
        "split": args.split,
        "outputs": {},
    }
    method_maps = {}
    selected_plane_ids = None

    for method_key, method_label in METHODS:
        print(f"[Sample {sample_idx}] running {method_label}")
        pts = run_model_on_sample(args, method_ckpts[method_key], batch, device)
        pts_cpu = pts.detach().cpu()
        rgb_pts = resize_rgb_to_points(rgb_cpu, pts_cpu)
        plane_pts = resize_plane_to_points(gt_plane_cpu, pts_cpu)
        valid = valid_point_mask(pts_cpu, args.max_abs_coord)

        z = pts_cpu[..., 2].numpy()
        norm = pts_cpu.norm(dim=-1).numpy()
        valid_np = valid.numpy()

        if selected_plane_ids is None:
            selected_plane_ids = [
                plane_id
                for plane_id, _ in top_plane_ids(
                    plane_pts,
                    valid,
                    args.top_planes,
                    args.min_plane_pixels,
                )
            ]

        residual, residual_mask, plane_metrics = plane_residual_map(
            pts_cpu,
            plane_pts,
            valid,
            selected_plane_ids,
        )

        z_path = os.path.join(sample_dir, f"{method_key}_z.png")
        norm_path = os.path.join(sample_dir, f"{method_key}_norm.png")
        residual_path = os.path.join(sample_dir, f"{method_key}_plane_residual.png")
        ply_path = os.path.join(sample_dir, f"{method_key}_colored.ply")

        save_scalar_map(z, valid_np, z_path, f"{method_label} z map")
        save_scalar_map(norm, valid_np, norm_path, f"{method_label} point norm")
        save_scalar_map(
            residual,
            residual_mask,
            residual_path,
            f"{method_label} plane residual",
            cmap="magma",
        )

        method_maps[method_key] = {
            "label": method_label,
            "z": z,
            "norm": norm,
            "valid": valid_np,
            "residual": residual,
            "residual_mask": residual_mask,
        }

        n_points = export_point_cloud(
            ply_path,
            pts_cpu,
            rgb_pts,
            valid,
            stride=max(1, args.ply_stride),
        )

        method_info = {
            "z_png": os.path.basename(z_path),
            "norm_png": os.path.basename(norm_path),
            "plane_residual_png": os.path.basename(residual_path),
            "colored_ply": os.path.basename(ply_path),
            "num_points": n_points,
            "plane_metrics": plane_metrics,
            "plane_ply": [],
        }

        for plane_id in selected_plane_ids:
            plane_mask = valid & (plane_pts == plane_id)
            count = int(plane_mask.sum().item())
            plane_path = os.path.join(
                sample_dir,
                f"{method_key}_plane_{plane_id}_colored.ply",
            )
            n_plane = export_point_cloud(
                plane_path,
                pts_cpu,
                rgb_pts,
                plane_mask,
                stride=max(1, args.ply_stride),
            )
            method_info["plane_ply"].append(
                {
                    "plane_id": plane_id,
                    "num_pixels_before_stride": count,
                    "num_points_written": n_plane,
                    "ply": os.path.basename(plane_path),
                }
            )

        manifest["outputs"][method_key] = method_info

    if method_maps:
        shared_ranges = {}
        for map_key, mask_key in [
            ("z", "valid"),
            ("norm", "valid"),
            ("residual", "residual_mask"),
        ]:
            values = []
            masks = []
            for info in method_maps.values():
                values.append(info[map_key])
                masks.append(info[mask_key])
            all_values = np.concatenate([v[m] for v, m in zip(values, masks) if m.any()])
            all_mask = np.isfinite(all_values)
            if all_values.size > 0 and all_mask.any():
                lo, hi = np.percentile(all_values[all_mask], [2.0, 98.0])
            else:
                lo, hi = 0.0, 1.0
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0
            shared_ranges[map_key] = (float(lo), float(hi))

        for method_key, method_label in METHODS:
            info = method_maps[method_key]
            z_shared_path = os.path.join(sample_dir, f"{method_key}_z_shared.png")
            norm_shared_path = os.path.join(sample_dir, f"{method_key}_norm_shared.png")
            residual_shared_path = os.path.join(
                sample_dir,
                f"{method_key}_plane_residual_shared.png",
            )
            save_scalar_map(
                info["z"],
                info["valid"],
                z_shared_path,
                f"{method_label} z map shared scale",
                value_range=shared_ranges["z"],
            )
            save_scalar_map(
                info["norm"],
                info["valid"],
                norm_shared_path,
                f"{method_label} point norm shared scale",
                value_range=shared_ranges["norm"],
            )
            save_scalar_map(
                info["residual"],
                info["residual_mask"],
                residual_shared_path,
                f"{method_label} plane residual shared scale",
                cmap="magma",
                value_range=shared_ranges["residual"],
            )
            manifest["outputs"][method_key]["z_shared_png"] = os.path.basename(
                z_shared_path
            )
            manifest["outputs"][method_key]["norm_shared_png"] = os.path.basename(
                norm_shared_path
            )
            manifest["outputs"][method_key]["plane_residual_shared_png"] = os.path.basename(
                residual_shared_path
            )
        manifest["shared_ranges"] = shared_ranges
        manifest["selected_plane_ids"] = selected_plane_ids or []

    manifest_path = os.path.join(sample_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[Done] sample {sample_idx}: {sample_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("LightRecon3D reconstruction comparison")
    print("=" * 80)
    print(f"device         : {device}")
    print(f"root_dir       : {args.root_dir}")
    print(f"split          : {args.split}")
    print(f"sample_idx     : {args.sample_idx}")
    print(f"output_dir     : {args.output_dir}")
    print(f"baseline_ckpt  : {args.baseline_ckpt}")
    print(f"geo_only_ckpt  : {args.geo_only_ckpt}")
    print(f"geo_anchor_ckpt: {args.geo_anchor_ckpt}")
    print("=" * 80)

    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        input_mode=args.input_mode,
        pair_strategy=args.pair_strategy,
    )

    print(f"Dataset size ({args.split}): {len(dataset)}")

    for sample_idx in args.sample_idx:
        process_sample(args, dataset, sample_idx, device)


if __name__ == "__main__":
    main()
