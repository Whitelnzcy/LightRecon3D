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

if DUST3R_REPO_ROOT not in sys.path:
    sys.path.insert(0, DUST3R_REPO_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
        raise ValueError(f"Expected 4D pointmap, got shape={tuple(pts.shape)} from key={used_key}")

    # [B, 3, H, W] -> [B, H, W, 3]
    if pts.shape[1] == 3:
        pts = pts.permute(0, 2, 3, 1).contiguous()

    if pts.shape[-1] != 3:
        raise ValueError(f"Expected pointmap with last dim 3, got shape={tuple(pts.shape)}")

    return pts


def resize_plane_to_pts(gt_plane: torch.Tensor, pts3d: torch.Tensor) -> torch.Tensor:
    """
    gt_plane: [B, H, W]
    pts3d:    [B, Hp, Wp, 3]
    return:   [B, Hp, Wp]
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
def flatness_metric_for_points(
    points: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    points: [N, 3]

    Returns a scalar flatness metric.
    Smaller is better.

    Normalized version:
        lambda_min / (lambda_1 + lambda_2 + lambda_3)

    This is scale-invariant and matches the idea of coplanarity loss.
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        return torch.tensor(float("nan"), device=points.device)

    finite_mask = torch.isfinite(points).all(dim=1)
    points = points[finite_mask]

    # Remove absurd coordinates to avoid rare pointmap explosions dominating eval.
    coord_ok = points.abs().amax(dim=1) < 1e4
    points = points[coord_ok]

    if points.shape[0] < 3:
        return torch.tensor(float("nan"), device=points.device)

    center = points.mean(dim=0, keepdim=True)
    x = points - center

    if not torch.isfinite(x).all():
        return torch.tensor(float("nan"), device=points.device)

    cov = x.transpose(0, 1) @ x / (points.shape[0] + eps)
    cov = 0.5 * (cov + cov.transpose(0, 1))

    eye = torch.eye(3, device=points.device, dtype=points.dtype)
    cov = cov + eps * eye
    cov = torch.nan_to_num(cov, nan=0.0, posinf=1e4, neginf=-1e4)

    try:
        eigvals = torch.linalg.eigvalsh(cov.float())
    except RuntimeError:
        return torch.tensor(float("nan"), device=points.device)

    eigvals = torch.clamp(eigvals, min=0.0)
    smallest = eigvals[0]

    if normalize:
        return smallest / (eigvals.sum() + eps)

    return smallest


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
    Evaluate pointmap flatness for one batch.
    Current script uses B=1, but this supports B>=1.
    """
    pts3d = get_pts3d_from_res(res)
    gt_plane = resize_plane_to_pts(gt_plane, pts3d)

    all_metrics: List[float] = []
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

            metric = flatness_metric_for_points(
                points,
                normalize=normalize,
            )

            if torch.isfinite(metric):
                all_metrics.append(float(metric.item()))
                all_plane_sizes.append(int(count))

    if len(all_metrics) == 0:
        return {
            "mean_flatness": float("nan"),
            "median_flatness": float("nan"),
            "num_valid_planes": 0,
            "mean_plane_pixels": float("nan"),
        }

    metrics_tensor = torch.tensor(all_metrics)

    return {
        "mean_flatness": float(metrics_tensor.mean().item()),
        "median_flatness": float(metrics_tensor.median().item()),
        "num_valid_planes": len(all_metrics),
        "mean_plane_pixels": float(sum(all_plane_sizes) / len(all_plane_sizes)),
    }


def make_dataset(args):
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
    )

    if args.num_samples is not None and args.num_samples > 0:
        indices = list(range(min(args.num_samples, len(dataset))))
        dataset = Subset(dataset, indices)

    return dataset


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
    valid_sample_count = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]

        batch = {
            "img": sample["img"].unsqueeze(0),
            "gt_line": sample["gt_line"].unsqueeze(0),
            "gt_plane": sample["gt_plane"].unsqueeze(0),
        }
        batch = move_batch_to_device(batch, device)

        view1, view2 = build_views_from_batch(batch, prefix=f"{args.split}_{idx}")

        res_base_1, _ = baseline_model(view1, view2)
        res_geo_1, _ = geo_model(view1, view2)

        base_stats = evaluate_one_output(
            res=res_base_1,
            gt_plane=batch["gt_plane"],
            min_points=args.min_points,
            max_points_per_plane=args.max_points_per_plane,
            max_planes_per_image=args.max_planes_per_image,
            normalize=args.normalize,
        )

        geo_stats = evaluate_one_output(
            res=res_geo_1,
            gt_plane=batch["gt_plane"],
            min_points=args.min_points,
            max_points_per_plane=args.max_points_per_plane,
            max_planes_per_image=args.max_planes_per_image,
            normalize=args.normalize,
        )

        base_mean = base_stats["mean_flatness"]
        geo_mean = geo_stats["mean_flatness"]

        if (
            base_mean == base_mean
            and geo_mean == geo_mean
            and base_stats["num_valid_planes"] > 0
            and geo_stats["num_valid_planes"] > 0
        ):
            delta = geo_mean - base_mean
            rel = delta / (base_mean + 1e-12)

            baseline_values.append(base_mean)
            geo_values.append(geo_mean)
            deltas.append(delta)
            valid_sample_count += 1
        else:
            delta = float("nan")
            rel = float("nan")

        row = {
            "sample_idx": idx,
            "baseline_mean_flatness": base_mean,
            "geo_mean_flatness": geo_mean,
            "delta_geo_minus_baseline": delta,
            "relative_delta": rel,
            "baseline_num_planes": base_stats["num_valid_planes"],
            "geo_num_planes": geo_stats["num_valid_planes"],
            "baseline_median_flatness": base_stats["median_flatness"],
            "geo_median_flatness": geo_stats["median_flatness"],
            "baseline_mean_plane_pixels": base_stats["mean_plane_pixels"],
            "geo_mean_plane_pixels": geo_stats["mean_plane_pixels"],
        }
        rows.append(row)

        if (idx + 1) % args.log_every == 0 or idx == 0 or idx + 1 == len(dataset):
            print(
                f"[{idx + 1}/{len(dataset)}] "
                f"base={base_mean:.8f}, geo={geo_mean:.8f}, "
                f"delta={delta:.8f}, "
                f"base_planes={base_stats['num_valid_planes']}, "
                f"geo_planes={geo_stats['num_valid_planes']}"
            )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("Summary")
    print("=" * 80)

    if valid_sample_count == 0:
        print("No valid samples for comparison.")
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
    print(f"output csv          : {args.output_csv}")

    if mean_delta < 0:
        print("Result: geo checkpoint is flatter on average.")
    elif mean_delta > 0:
        print("Result: geo checkpoint is less flat on average.")
    else:
        print("Result: no average difference.")


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

    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)

    parser.add_argument("--min_points", type=int, default=128)
    parser.add_argument("--max_points_per_plane", type=int, default=2048)
    parser.add_argument("--max_planes_per_image", type=int, default=8)

    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--output_csv", type=str, default="/data/zhucy23u/logs/flatness_eval.csv")

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)