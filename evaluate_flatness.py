import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

# ============================================================
# Path setup, consistent with train.py
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")

# Keep project root before dust3r repo root.
# Otherwise `from train import ...` may accidentally import dust3r/train.py.
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


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model_from_ckpt(
    weights_path: str,
    ckpt_path: str,
    device: torch.device,
    hidden_dim: int = 768,
    plane_embed_dim: int = 16,
) -> LightReconModel:
    backbone = build_dust3r_backbone(weights_path, device=device)

    model = LightReconModel(
        dust3r_backbone=backbone,
        hidden_dim=hidden_dim,
        plane_embed_dim=plane_embed_dim,
    ).to(device)

    ckpt = safe_torch_load(ckpt_path, device)
    state_dict = ckpt.get("model", ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[Load] ckpt: {ckpt_path}")
    print(f"[Load] missing keys: {len(missing)}")
    print(f"[Load] unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[Load] first missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("[Load] first unexpected keys:", unexpected[:10])

    model.eval()
    return model


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


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

    # [B, 3, H, W] -> [B, H, W, 3]
    if pts.shape[1] == 3:
        pts = pts.permute(0, 2, 3, 1).contiguous()

    if pts.shape[-1] != 3:
        raise ValueError(
            f"Expected pointmap with last dim 3, got shape={tuple(pts.shape)} "
            f"from key={used_key}"
        )

    return pts


def resize_plane_to_pts(gt_plane: torch.Tensor, pts3d: torch.Tensor) -> torch.Tensor:
    """
    gt_plane: [B, H, W]
    pts3d:    [B, Hp, Wp, 3]

    return:
        gt_plane resized to [B, Hp, Wp]
    """
    target_h, target_w = pts3d.shape[1], pts3d.shape[2]

    if gt_plane.shape[-2:] == (target_h, target_w):
        return gt_plane.long()

    plane = gt_plane.unsqueeze(1).float()
    plane = F.interpolate(
        plane,
        size=(target_h, target_w),
        mode="nearest",
    )
    return plane[:, 0].long()


@torch.no_grad()
def plane_geometry_metrics_for_points(
    points: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute geometry metrics for one gt_plane region.

    Args:
        points:
            [N, 3] pointmap points selected from one gt plane region.

        normalize:
            If True:
                flatness = lambda_min / (lambda_1 + lambda_2 + lambda_3)
            If False:
                flatness = lambda_min

    Returns:
        flatness:
            Normalized smallest eigenvalue. Smaller means more planar.

        plane_extent:
            lambda_2 + lambda_3.
            Measures in-plane spread. If this collapses sharply, the point cloud
            may become flatter simply because it shrinks.

        eig_sum:
            lambda_1 + lambda_2 + lambda_3.
            Overall point cloud variance.

        point_norm:
            mean(||p||).
            Used to check whether pointmap scale becomes abnormal.

        valid:
            1 if metrics are valid, 0 otherwise.
    """
    invalid = {
        "flatness": float("nan"),
        "plane_extent": float("nan"),
        "eig_sum": float("nan"),
        "point_norm": float("nan"),
        "valid": 0,
    }

    if points.ndim != 2 or points.shape[-1] != 3:
        return invalid

    finite_mask = torch.isfinite(points).all(dim=1)
    points = points[finite_mask]

    # Remove rare absurd coordinates to avoid extreme pointmap values dominating eval.
    coord_ok = points.abs().amax(dim=1) < 1e4
    points = points[coord_ok]

    if points.shape[0] < 3:
        return invalid

    point_norm = points.norm(dim=1).mean()

    center = points.mean(dim=0, keepdim=True)
    x = points - center

    if not torch.isfinite(x).all():
        return invalid

    cov = x.transpose(0, 1) @ x / (points.shape[0] + eps)
    cov = 0.5 * (cov + cov.transpose(0, 1))

    eye = torch.eye(3, device=points.device, dtype=points.dtype)
    cov = cov + eps * eye
    cov = torch.nan_to_num(cov, nan=0.0, posinf=1e4, neginf=-1e4)

    try:
        eigvals = torch.linalg.eigvalsh(cov.float())
    except RuntimeError:
        return invalid

    eigvals = torch.clamp(eigvals, min=0.0)

    eig_sum = eigvals.sum()
    flatness_raw = eigvals[0]
    plane_extent = eigvals[1] + eigvals[2]

    if normalize:
        flatness = flatness_raw / (eig_sum + eps)
    else:
        flatness = flatness_raw

    if not torch.isfinite(flatness):
        return invalid

    if not torch.isfinite(plane_extent):
        return invalid

    if not torch.isfinite(eig_sum):
        return invalid

    if not torch.isfinite(point_norm):
        return invalid

    return {
        "flatness": float(flatness.item()),
        "plane_extent": float(plane_extent.item()),
        "eig_sum": float(eig_sum.item()),
        "point_norm": float(point_norm.item()),
        "valid": 1,
    }


def deterministic_sample_indices(num_points: int, max_points: int, device: torch.device):
    """
    Deterministic sampling for reproducible baseline-vs-geo comparison.
    """
    if num_points <= max_points:
        return torch.arange(num_points, device=device)

    return torch.linspace(
        0,
        num_points - 1,
        steps=max_points,
        device=device,
    ).long()


@torch.no_grad()
def evaluate_one_output(
    res: Dict[str, torch.Tensor],
    gt_plane: torch.Tensor,
    min_points: int = 128,
    max_points_per_plane: int = 2048,
    max_planes_per_image: int = 8,
    ignore_ids: Tuple[int, ...] = (-1, 255),
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Evaluate pointmap geometry for one batch.

    Current script usually uses B=1, but this supports B>=1.
    """
    pts3d = get_pts3d_from_res(res)
    gt_plane = resize_plane_to_pts(gt_plane, pts3d)

    all_flatness: List[float] = []
    all_plane_extent: List[float] = []
    all_eig_sum: List[float] = []
    all_point_norm: List[float] = []
    all_plane_sizes: List[int] = []

    B = pts3d.shape[0]

    for b in range(B):
        pts_b = pts3d[b]        # [H, W, 3]
        plane_b = gt_plane[b]   # [H, W]

        plane_ids = torch.unique(plane_b)

        valid_planes = []
        for pid in plane_ids:
            pid_int = int(pid.item())

            if pid_int in ignore_ids:
                continue

            mask = plane_b == pid
            count = int(mask.sum().item())

            if count < min_points:
                continue

            valid_planes.append((pid_int, count))

        # Prioritize large structural planes.
        valid_planes = sorted(valid_planes, key=lambda x: x[1], reverse=True)
        valid_planes = valid_planes[:max_planes_per_image]

        for pid_int, count in valid_planes:
            mask = plane_b == pid_int
            points = pts_b[mask]  # [N, 3]

            if points.shape[0] < min_points:
                continue

            idx = deterministic_sample_indices(
                points.shape[0],
                max_points_per_plane,
                device=points.device,
            )
            points = points[idx]

            metrics = plane_geometry_metrics_for_points(
                points,
                normalize=normalize,
            )

            if metrics["valid"] == 1:
                all_flatness.append(metrics["flatness"])
                all_plane_extent.append(metrics["plane_extent"])
                all_eig_sum.append(metrics["eig_sum"])
                all_point_norm.append(metrics["point_norm"])
                all_plane_sizes.append(int(count))

    if len(all_flatness) == 0:
        return {
            "mean_flatness": float("nan"),
            "median_flatness": float("nan"),
            "mean_plane_extent": float("nan"),
            "median_plane_extent": float("nan"),
            "mean_eig_sum": float("nan"),
            "median_eig_sum": float("nan"),
            "mean_point_norm": float("nan"),
            "median_point_norm": float("nan"),
            "num_valid_planes": 0,
            "mean_plane_pixels": float("nan"),
        }

    flatness_tensor = torch.tensor(all_flatness)
    extent_tensor = torch.tensor(all_plane_extent)
    eig_sum_tensor = torch.tensor(all_eig_sum)
    point_norm_tensor = torch.tensor(all_point_norm)

    return {
        "mean_flatness": float(flatness_tensor.mean().item()),
        "median_flatness": float(flatness_tensor.median().item()),
        "mean_plane_extent": float(extent_tensor.mean().item()),
        "median_plane_extent": float(extent_tensor.median().item()),
        "mean_eig_sum": float(eig_sum_tensor.mean().item()),
        "median_eig_sum": float(eig_sum_tensor.median().item()),
        "mean_point_norm": float(point_norm_tensor.mean().item()),
        "median_point_norm": float(point_norm_tensor.median().item()),
        "num_valid_planes": len(all_flatness),
        "mean_plane_pixels": float(sum(all_plane_sizes) / len(all_plane_sizes)),
    }


def make_dataset(args):
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        input_mode=args.input_mode,
        pair_strategy=args.pair_strategy,
        pair_max_view_id_gap=args.pair_max_view_id_gap,
    )

    if args.num_samples is not None and args.num_samples > 0:
        indices = list(range(min(args.num_samples, len(dataset))))
        dataset = Subset(dataset, indices)

    return dataset


def is_valid_number(x: float) -> bool:
    return x == x


def average_eval_stats(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    valid_stats = [
        stats for stats in stats_list
        if stats.get("num_valid_planes", 0) > 0
    ]

    if not valid_stats:
        return stats_list[0] if stats_list else {}

    averaged = {}
    keys = valid_stats[0].keys()

    for key in keys:
        values = [
            stats[key] for stats in valid_stats
            if key in stats and is_valid_number(float(stats[key]))
        ]

        if not values:
            averaged[key] = float("nan")
        elif key == "num_valid_planes":
            averaged[key] = int(sum(values))
        else:
            averaged[key] = float(sum(values) / len(values))

    return averaged


@torch.no_grad()
def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print("=" * 80)
    print("LightRecon3D Pointmap Flatness Evaluation")
    print("=" * 80)
    print(f"device       : {device}")
    print(f"split        : {args.split}")
    print(f"num_samples  : {args.num_samples}")
    print(f"baseline ckpt: {args.baseline_ckpt}")
    print(f"geo ckpt     : {args.geo_ckpt}")
    print(f"normalize    : {args.normalize}")
    print(f"eval_views   : {args.eval_views}")
    print("=" * 80)

    dataset = make_dataset(args)

    baseline_model = load_model_from_ckpt(
        weights_path=args.weights_path,
        ckpt_path=args.baseline_ckpt,
        device=device,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    )

    geo_model = load_model_from_ckpt(
        weights_path=args.weights_path,
        ckpt_path=args.geo_ckpt,
        device=device,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    )

    rows = []

    baseline_values = []
    geo_values = []
    deltas = []

    baseline_extents = []
    geo_extents = []
    extent_deltas = []

    baseline_eig_sums = []
    geo_eig_sums = []
    eig_sum_deltas = []

    baseline_point_norms = []
    geo_point_norms = []
    point_norm_deltas = []

    valid_sample_count = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]

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

        view1, view2 = build_views_from_batch(batch, prefix=f"{args.split}_{idx}")

        res_base_1, res_base_2 = baseline_model(view1, view2)
        res_geo_1, res_geo_2 = geo_model(view1, view2)

        base_stats_list = []
        geo_stats_list = []

        if args.eval_views in ["view1", "both"]:
            base_stats_list.append(evaluate_one_output(
                res=res_base_1,
                gt_plane=batch.get("gt_plane1", batch["gt_plane"]),
                min_points=args.min_points,
                max_points_per_plane=args.max_points_per_plane,
                max_planes_per_image=args.max_planes_per_image,
                normalize=args.normalize,
            ))
            geo_stats_list.append(evaluate_one_output(
                res=res_geo_1,
                gt_plane=batch.get("gt_plane1", batch["gt_plane"]),
                min_points=args.min_points,
                max_points_per_plane=args.max_points_per_plane,
                max_planes_per_image=args.max_planes_per_image,
                normalize=args.normalize,
            ))

        if args.eval_views in ["view2", "both"]:
            if "gt_plane2" not in batch:
                raise KeyError("--eval_views includes view2, but batch does not contain gt_plane2")

            base_stats_list.append(evaluate_one_output(
                res=res_base_2,
                gt_plane=batch["gt_plane2"],
                min_points=args.min_points,
                max_points_per_plane=args.max_points_per_plane,
                max_planes_per_image=args.max_planes_per_image,
                normalize=args.normalize,
            ))
            geo_stats_list.append(evaluate_one_output(
                res=res_geo_2,
                gt_plane=batch["gt_plane2"],
                min_points=args.min_points,
                max_points_per_plane=args.max_points_per_plane,
                max_planes_per_image=args.max_planes_per_image,
                normalize=args.normalize,
            ))

        base_stats = average_eval_stats(base_stats_list)
        geo_stats = average_eval_stats(geo_stats_list)

        base_mean = base_stats["mean_flatness"]
        geo_mean = geo_stats["mean_flatness"]

        if (
            is_valid_number(base_mean)
            and is_valid_number(geo_mean)
            and base_stats["num_valid_planes"] > 0
            and geo_stats["num_valid_planes"] > 0
        ):
            delta = geo_mean - base_mean
            rel = delta / (base_mean + 1e-12)

            baseline_values.append(base_mean)
            geo_values.append(geo_mean)
            deltas.append(delta)

            base_extent = base_stats["mean_plane_extent"]
            geo_extent = geo_stats["mean_plane_extent"]
            base_eig_sum = base_stats["mean_eig_sum"]
            geo_eig_sum = geo_stats["mean_eig_sum"]
            base_norm = base_stats["mean_point_norm"]
            geo_norm = geo_stats["mean_point_norm"]

            if is_valid_number(base_extent) and is_valid_number(geo_extent):
                baseline_extents.append(base_extent)
                geo_extents.append(geo_extent)
                extent_deltas.append(geo_extent - base_extent)

            if is_valid_number(base_eig_sum) and is_valid_number(geo_eig_sum):
                baseline_eig_sums.append(base_eig_sum)
                geo_eig_sums.append(geo_eig_sum)
                eig_sum_deltas.append(geo_eig_sum - base_eig_sum)

            if is_valid_number(base_norm) and is_valid_number(geo_norm):
                baseline_point_norms.append(base_norm)
                geo_point_norms.append(geo_norm)
                point_norm_deltas.append(geo_norm - base_norm)

            valid_sample_count += 1
        else:
            delta = float("nan")
            rel = float("nan")

        row = {
            "sample_idx": idx,
            "eval_views": args.eval_views,

            "baseline_mean_flatness": base_mean,
            "geo_mean_flatness": geo_mean,
            "delta_geo_minus_baseline": delta,
            "relative_delta": rel,

            "baseline_median_flatness": base_stats["median_flatness"],
            "geo_median_flatness": geo_stats["median_flatness"],

            "baseline_mean_plane_extent": base_stats["mean_plane_extent"],
            "geo_mean_plane_extent": geo_stats["mean_plane_extent"],
            "delta_extent_geo_minus_baseline": (
                geo_stats["mean_plane_extent"] - base_stats["mean_plane_extent"]
                if is_valid_number(base_stats["mean_plane_extent"])
                and is_valid_number(geo_stats["mean_plane_extent"])
                else float("nan")
            ),

            "baseline_median_plane_extent": base_stats["median_plane_extent"],
            "geo_median_plane_extent": geo_stats["median_plane_extent"],

            "baseline_mean_eig_sum": base_stats["mean_eig_sum"],
            "geo_mean_eig_sum": geo_stats["mean_eig_sum"],
            "delta_eig_sum_geo_minus_baseline": (
                geo_stats["mean_eig_sum"] - base_stats["mean_eig_sum"]
                if is_valid_number(base_stats["mean_eig_sum"])
                and is_valid_number(geo_stats["mean_eig_sum"])
                else float("nan")
            ),

            "baseline_median_eig_sum": base_stats["median_eig_sum"],
            "geo_median_eig_sum": geo_stats["median_eig_sum"],

            "baseline_mean_point_norm": base_stats["mean_point_norm"],
            "geo_mean_point_norm": geo_stats["mean_point_norm"],
            "delta_point_norm_geo_minus_baseline": (
                geo_stats["mean_point_norm"] - base_stats["mean_point_norm"]
                if is_valid_number(base_stats["mean_point_norm"])
                and is_valid_number(geo_stats["mean_point_norm"])
                else float("nan")
            ),

            "baseline_median_point_norm": base_stats["median_point_norm"],
            "geo_median_point_norm": geo_stats["median_point_norm"],

            "baseline_num_planes": base_stats["num_valid_planes"],
            "geo_num_planes": geo_stats["num_valid_planes"],
            "baseline_mean_plane_pixels": base_stats["mean_plane_pixels"],
            "geo_mean_plane_pixels": geo_stats["mean_plane_pixels"],
        }
        rows.append(row)

        if (idx + 1) % args.log_every == 0 or idx == 0 or idx + 1 == len(dataset):
            print(
                f"[{idx + 1}/{len(dataset)}] "
                f"base_flat={base_mean:.8f}, geo_flat={geo_mean:.8f}, "
                f"delta={delta:.8f}, "
                f"base_extent={base_stats['mean_plane_extent']:.6f}, "
                f"geo_extent={geo_stats['mean_plane_extent']:.6f}, "
                f"base_norm={base_stats['mean_point_norm']:.6f}, "
                f"geo_norm={geo_stats['mean_point_norm']:.6f}, "
                f"base_planes={base_stats['num_valid_planes']}, "
                f"geo_planes={geo_stats['num_valid_planes']}"
            )

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if len(rows) == 0:
        print("No rows generated.")
        return

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("Summary")
    print("=" * 80)

    if valid_sample_count == 0:
        print("No valid samples for comparison.")
        print(f"output csv          : {args.output_csv}")
        return

    baseline_tensor = torch.tensor(baseline_values)
    geo_tensor = torch.tensor(geo_values)
    delta_tensor = torch.tensor(deltas)

    mean_base = float(baseline_tensor.mean().item())
    mean_geo = float(geo_tensor.mean().item())
    mean_delta = float(delta_tensor.mean().item())
    median_delta = float(delta_tensor.median().item())

    improved = int((delta_tensor < 0).sum().item())
    worsened = int((delta_tensor > 0).sum().item())

    print(f"valid samples       : {valid_sample_count}")
    print(f"baseline mean       : {mean_base:.10f}")
    print(f"geo mean            : {mean_geo:.10f}")
    print(f"mean delta geo-base : {mean_delta:.10f}")
    print(f"median delta        : {median_delta:.10f}")
    print(f"improved samples    : {improved}")
    print(f"worsened samples    : {worsened}")

    if len(extent_deltas) > 0:
        base_extent_tensor = torch.tensor(baseline_extents)
        geo_extent_tensor = torch.tensor(geo_extents)
        extent_delta_tensor = torch.tensor(extent_deltas)

        print("-" * 80)
        print("Plane extent check")
        print("-" * 80)
        print(f"baseline extent mean : {float(base_extent_tensor.mean().item()):.10f}")
        print(f"geo extent mean      : {float(geo_extent_tensor.mean().item()):.10f}")
        print(f"extent delta geo-base: {float(extent_delta_tensor.mean().item()):.10f}")

        if float(base_extent_tensor.mean().item()) > 0:
            extent_ratio = float(geo_extent_tensor.mean().item()) / float(base_extent_tensor.mean().item())
            print(f"geo/base extent ratio: {extent_ratio:.10f}")

    if len(eig_sum_deltas) > 0:
        base_eig_tensor = torch.tensor(baseline_eig_sums)
        geo_eig_tensor = torch.tensor(geo_eig_sums)
        eig_delta_tensor = torch.tensor(eig_sum_deltas)

        print("-" * 80)
        print("Eigenvalue sum check")
        print("-" * 80)
        print(f"baseline eig_sum mean : {float(base_eig_tensor.mean().item()):.10f}")
        print(f"geo eig_sum mean      : {float(geo_eig_tensor.mean().item()):.10f}")
        print(f"eig_sum delta geo-base: {float(eig_delta_tensor.mean().item()):.10f}")

        if float(base_eig_tensor.mean().item()) > 0:
            eig_sum_ratio = float(geo_eig_tensor.mean().item()) / float(base_eig_tensor.mean().item())
            print(f"geo/base eig_sum ratio: {eig_sum_ratio:.10f}")

    if len(point_norm_deltas) > 0:
        base_norm_tensor = torch.tensor(baseline_point_norms)
        geo_norm_tensor = torch.tensor(geo_point_norms)
        norm_delta_tensor = torch.tensor(point_norm_deltas)

        print("-" * 80)
        print("Point norm check")
        print("-" * 80)
        print(f"baseline point norm : {float(base_norm_tensor.mean().item()):.10f}")
        print(f"geo point norm      : {float(geo_norm_tensor.mean().item()):.10f}")
        print(f"point norm delta    : {float(norm_delta_tensor.mean().item()):.10f}")

        if float(base_norm_tensor.mean().item()) > 0:
            point_norm_ratio = float(geo_norm_tensor.mean().item()) / float(base_norm_tensor.mean().item())
            print(f"geo/base norm ratio : {point_norm_ratio:.10f}")

    print("-" * 80)
    print(f"output csv          : {args.output_csv}")

    if mean_delta < 0:
        print("Result: geo checkpoint is flatter on average.")
    elif mean_delta > 0:
        print("Result: geo checkpoint is less flat on average.")
    else:
        print("Result: no average flatness difference.")

    print("-" * 80)
    print("Interpretation reminder:")
    print("  flatness lower is better.")
    print("  extent should not collapse too much.")
    print("  point norm should not shrink abnormally.")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)

    parser.add_argument("--baseline_ckpt", type=str, required=True)
    parser.add_argument("--geo_ckpt", type=str, required=True)

    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--input_mode", type=str, default="pair", choices=["pair", "single"])
    parser.add_argument("--pair_strategy", type=str, default="adjacent", choices=["adjacent", "all"])
    parser.add_argument("--pair_max_view_id_gap", type=int, default=0)

    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)

    parser.add_argument("--min_points", type=int, default=128)
    parser.add_argument("--max_points_per_plane", type=int, default=2048)
    parser.add_argument("--max_planes_per_image", type=int, default=8)

    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--eval_views",
        type=str,
        default="view1",
        choices=["view1", "view2", "both"],
        help=(
            "Which DUSt3R output to evaluate. view1 uses res1+gt_plane1; "
            "view2 uses res2+gt_plane2; both averages valid per-view stats."
        ),
    )
    parser.add_argument("--output_csv", type=str, default="/data/zhucy23u/logs/flatness_eval.csv")

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
