import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


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
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel
from train import build_views_from_batch


# ============================================================
# Utilities
# ============================================================

def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def load_model(
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
    state = ckpt.get("model", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[Load] ckpt: {ckpt_path}")
    print(f"[Load] missing keys   : {len(missing)}")
    print(f"[Load] unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[Load] first missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("[Load] first unexpected keys:", unexpected[:10])

    model.eval()
    return model


def resize_label_nearest(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    x: [B, H, W]
    return [B, size[0], size[1]]
    """
    if x.shape[-2:] == size:
        return x.long()

    y = x.unsqueeze(1).float()
    y = F.interpolate(y, size=size, mode="nearest")
    return y[:, 0].long()


def resize_binary_nearest(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    x: [B, H, W] or [B, 1, H, W]
    return [B, size[0], size[1]]
    """
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]

    if x.shape[-2:] == size:
        return x.float()

    y = x.unsqueeze(1).float()
    y = F.interpolate(y, size=size, mode="nearest")
    return y[:, 0].float()


def get_pred_line_logits(res: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "pred_line" not in res:
        raise KeyError(f"pred_line not found. keys={list(res.keys())}")

    line = res["pred_line"]

    if line.ndim == 4 and line.shape[1] == 1:
        return line[:, 0]

    if line.ndim == 3:
        return line

    raise ValueError(f"Unsupported pred_line shape: {tuple(line.shape)}")


def get_pred_plane_embedding(res: Dict[str, torch.Tensor], prefer_lowres: bool = True) -> torch.Tensor:
    """
    Return [B, C, H, W].

    Prefer pred_plane_lowres if available because it avoids evaluating
    interpolated full-resolution boundary mixtures.
    """
    if prefer_lowres and "pred_plane_lowres" in res:
        emb = res["pred_plane_lowres"]
    elif "pred_plane" in res:
        emb = res["pred_plane"]
    elif "pred_plane_embedding" in res:
        emb = res["pred_plane_embedding"]
    else:
        raise KeyError(f"plane embedding not found. keys={list(res.keys())}")

    if emb.ndim != 4:
        raise ValueError(f"Unsupported plane embedding shape: {tuple(emb.shape)}")

    return emb


def dilate_binary(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    mask: [B, H, W], bool or float
    return bool [B, H, W]
    """
    if radius <= 0:
        return mask.bool()

    x = mask.float().unsqueeze(1)
    k = 2 * radius + 1
    y = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
    return y[:, 0] > 0.5


def update_binary_counts(
    counts: Dict[str, float],
    pred: torch.Tensor,
    gt: torch.Tensor,
):
    """
    pred, gt: bool [B, H, W]
    """
    pred = pred.bool()
    gt = gt.bool()

    tp = (pred & gt).sum().item()
    fp = (pred & (~gt)).sum().item()
    fn = ((~pred) & gt).sum().item()
    tn = ((~pred) & (~gt)).sum().item()

    counts["tp"] += tp
    counts["fp"] += fp
    counts["fn"] += fn
    counts["tn"] += tn


def counts_to_metrics(counts: Dict[str, float], eps: float = 1e-8) -> Dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "acc": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def update_tolerant_counts(
    counts: Dict[str, float],
    pred: torch.Tensor,
    gt: torch.Tensor,
    radius: int,
):
    """
    Tolerant line metric.

    precision:
        predicted pixels are correct if they fall within dilated GT.

    recall:
        GT pixels are recovered if they fall within dilated prediction.
    """
    pred = pred.bool()
    gt = gt.bool()

    gt_dil = dilate_binary(gt, radius)
    pred_dil = dilate_binary(pred, radius)

    tp_prec = (pred & gt_dil).sum().item()
    pred_pos = pred.sum().item()

    tp_rec = (gt & pred_dil).sum().item()
    gt_pos = gt.sum().item()

    counts["tp_prec"] += tp_prec
    counts["pred_pos"] += pred_pos
    counts["tp_rec"] += tp_rec
    counts["gt_pos"] += gt_pos


def tolerant_counts_to_metrics(counts: Dict[str, float], eps: float = 1e-8) -> Dict[str, float]:
    precision = counts["tp_prec"] / (counts["pred_pos"] + eps)
    recall = counts["tp_rec"] / (counts["gt_pos"] + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "tolerant_precision": precision,
        "tolerant_recall": recall,
        "tolerant_f1": f1,
    }


def deterministic_sample(x: torch.Tensor, max_points: int) -> torch.Tensor:
    n = x.shape[0]
    if max_points <= 0 or n <= max_points:
        return x

    idx = torch.linspace(0, n - 1, steps=max_points, device=x.device).long()
    return x[idx]


@torch.no_grad()
def compute_plane_embedding_metrics(
    emb: torch.Tensor,
    gt_plane: torch.Tensor,
    min_pixels: int,
    max_planes: int,
    max_pixels_per_plane: int,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    emb: [B, C, H, W]
    gt_plane: [B, H, W] already resized to emb size

    Returns oracle embedding separability metrics.
    """
    b, c, h, w = emb.shape

    intra_values = []
    inter_values = []
    ratios = []
    num_planes_values = []

    for bi in range(b):
        emb_i = emb[bi].permute(1, 2, 0).contiguous()  # [H, W, C]
        plane_i = gt_plane[bi]

        ids, counts = torch.unique(plane_i, return_counts=True)

        candidates = []
        for pid, count in zip(ids.tolist(), counts.tolist()):
            pid = int(pid)
            count = int(count)

            if pid < 0 or pid == 255:
                continue

            if count < min_pixels:
                continue

            candidates.append((pid, count))

        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        candidates = candidates[:max_planes]

        centers = []
        intra_for_img = []

        for pid, count in candidates:
            mask = plane_i == pid
            vals = emb_i[mask]  # [N, C]
            vals = vals[torch.isfinite(vals).all(dim=1)]

            if vals.shape[0] < min_pixels:
                continue

            vals = deterministic_sample(vals, max_pixels_per_plane)

            center = vals.mean(dim=0)
            centers.append(center)

            # Mean L2 distance to its own center.
            d = torch.linalg.norm(vals - center.unsqueeze(0), dim=1)
            intra_for_img.append(d.mean())

        if len(centers) == 0:
            continue

        intra_img = torch.stack(intra_for_img).mean()
        intra_values.append(float(intra_img.item()))
        num_planes_values.append(len(centers))

        if len(centers) >= 2:
            centers_t = torch.stack(centers, dim=0)  # [K, C]
            dist = torch.cdist(centers_t, centers_t, p=2)

            k = centers_t.shape[0]
            tri = torch.triu(torch.ones(k, k, device=dist.device, dtype=torch.bool), diagonal=1)
            inter_img = dist[tri].mean()

            inter_values.append(float(inter_img.item()))
            ratios.append(float((inter_img / intra_img.clamp_min(eps)).item()))

    if len(intra_values) == 0:
        return {
            "plane_intra": float("nan"),
            "plane_inter": float("nan"),
            "plane_separation_ratio": float("nan"),
            "plane_num_images": 0,
            "plane_num_avg": float("nan"),
        }

    return {
        "plane_intra": float(np.mean(intra_values)),
        "plane_inter": float(np.mean(inter_values)) if inter_values else float("nan"),
        "plane_separation_ratio": float(np.mean(ratios)) if ratios else float("nan"),
        "plane_num_images": len(intra_values),
        "plane_num_avg": float(np.mean(num_planes_values)),
    }


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Evaluate LightRecon3D line / plane heads")

    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)

    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)

    parser.add_argument("--line_thresholds", type=str, default="0.1,0.2,0.3,0.5")
    parser.add_argument("--dilate_radius", type=int, default=3)

    parser.add_argument("--plane_min_pixels", type=int, default=16)
    parser.add_argument("--plane_max_planes", type=int, default=20)
    parser.add_argument("--plane_max_pixels_per_plane", type=int, default=5000)
    parser.add_argument("--prefer_lowres_plane", action="store_true")

    parser.add_argument("--output_csv", type=str, default="")

    args = parser.parse_args()

    thresholds = [float(x) for x in args.line_thresholds.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("LightRecon3D Heads Evaluation")
    print("=" * 80)
    print(f"device       : {device}")
    print(f"root_dir     : {args.root_dir}")
    print(f"weights_path : {args.weights_path}")
    print(f"ckpt_path    : {args.ckpt_path}")
    print(f"split        : {args.split}")
    print(f"num_samples  : {args.num_samples}")
    print(f"thresholds   : {thresholds}")
    print(f"dilate_radius: {args.dilate_radius}")
    print("=" * 80)

    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
    )

    n = min(args.num_samples, len(dataset))

    model = load_model(
        weights_path=args.weights_path,
        ckpt_path=args.ckpt_path,
        device=device,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    )

    # Global line counts for each threshold.
    line_counts = {
        thr: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
        for thr in thresholds
    }

    tolerant_counts = {
        thr: {"tp_prec": 0.0, "pred_pos": 0.0, "tp_rec": 0.0, "gt_pos": 0.0}
        for thr in thresholds
    }

    plane_metric_rows = []
    per_sample_rows = []

    for sample_idx in range(n):
        sample = dataset[sample_idx]

        batch = {
            "img": sample["img"].unsqueeze(0),
            "gt_line": sample["gt_line"].unsqueeze(0),
            "gt_plane": sample["gt_plane"].unsqueeze(0),
        }

        batch = move_batch_to_device(batch, device)

        view1, view2 = build_views_from_batch(batch, prefix=f"eval_{args.split}_{sample_idx}")

        res1, _ = model(view1, view2)

        # ------------------------------
        # Line metrics
        # ------------------------------
        line_logits = get_pred_line_logits(res1)  # [B, H, W]
        line_prob = torch.sigmoid(line_logits)

        gt_line = batch["gt_line"]
        if gt_line.ndim == 4 and gt_line.shape[1] == 1:
            gt_line = gt_line[:, 0]
        gt_line = resize_binary_nearest(gt_line, size=line_prob.shape[-2:])
        gt_line_bool = gt_line > 0.5

        sample_line_metrics = {}

        for thr in thresholds:
            pred_bool = line_prob >= thr

            local_counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
            update_binary_counts(local_counts, pred_bool, gt_line_bool)
            update_binary_counts(line_counts[thr], pred_bool, gt_line_bool)

            local_tol = {"tp_prec": 0.0, "pred_pos": 0.0, "tp_rec": 0.0, "gt_pos": 0.0}
            update_tolerant_counts(local_tol, pred_bool, gt_line_bool, args.dilate_radius)
            update_tolerant_counts(tolerant_counts[thr], pred_bool, gt_line_bool, args.dilate_radius)

            m = counts_to_metrics(local_counts)
            tm = tolerant_counts_to_metrics(local_tol)

            sample_line_metrics[f"line_f1_thr{thr}"] = m["f1"]
            sample_line_metrics[f"line_iou_thr{thr}"] = m["iou"]
            sample_line_metrics[f"line_tol_f1_thr{thr}"] = tm["tolerant_f1"]

        # ------------------------------
        # Plane embedding oracle metrics
        # ------------------------------
        emb = get_pred_plane_embedding(res1, prefer_lowres=args.prefer_lowres_plane)

        gt_plane = batch["gt_plane"]
        if gt_plane.ndim == 4 and gt_plane.shape[1] == 1:
            gt_plane = gt_plane[:, 0]

        gt_plane_low = resize_label_nearest(gt_plane, size=emb.shape[-2:])

        plane_metrics = compute_plane_embedding_metrics(
            emb=emb,
            gt_plane=gt_plane_low,
            min_pixels=args.plane_min_pixels,
            max_planes=args.plane_max_planes,
            max_pixels_per_plane=args.plane_max_pixels_per_plane,
        )

        plane_metric_rows.append(plane_metrics)

        row = {
            "sample_idx": sample_idx,
            **sample_line_metrics,
            **plane_metrics,
        }
        per_sample_rows.append(row)

        print(
            f"[{sample_idx + 1}/{n}] "
            f"plane_intra={plane_metrics['plane_intra']:.6f}, "
            f"plane_inter={plane_metrics['plane_inter']:.6f}, "
            f"sep={plane_metrics['plane_separation_ratio']:.3f}"
        )

    # ========================================================
    # Global summary
    # ========================================================

    print("=" * 80)
    print("Line metrics")
    print("=" * 80)

    global_summary = {}

    for thr in thresholds:
        m = counts_to_metrics(line_counts[thr])
        tm = tolerant_counts_to_metrics(tolerant_counts[thr])

        print(
            f"thr={thr:.2f} | "
            f"P={m['precision']:.4f}, R={m['recall']:.4f}, "
            f"F1={m['f1']:.4f}, IoU={m['iou']:.4f}, "
            f"tolP={tm['tolerant_precision']:.4f}, "
            f"tolR={tm['tolerant_recall']:.4f}, "
            f"tolF1={tm['tolerant_f1']:.4f}"
        )

        global_summary[f"line_precision_thr{thr}"] = m["precision"]
        global_summary[f"line_recall_thr{thr}"] = m["recall"]
        global_summary[f"line_f1_thr{thr}"] = m["f1"]
        global_summary[f"line_iou_thr{thr}"] = m["iou"]
        global_summary[f"line_tolerant_precision_thr{thr}"] = tm["tolerant_precision"]
        global_summary[f"line_tolerant_recall_thr{thr}"] = tm["tolerant_recall"]
        global_summary[f"line_tolerant_f1_thr{thr}"] = tm["tolerant_f1"]

    print("=" * 80)
    print("Plane embedding oracle metrics")
    print("=" * 80)

    def mean_of(key: str):
        vals = [r[key] for r in plane_metric_rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    plane_intra = mean_of("plane_intra")
    plane_inter = mean_of("plane_inter")
    plane_sep = mean_of("plane_separation_ratio")
    plane_num_avg = mean_of("plane_num_avg")

    print(f"plane_intra_mean          : {plane_intra:.6f}")
    print(f"plane_inter_mean          : {plane_inter:.6f}")
    print(f"plane_separation_ratio    : {plane_sep:.6f}")
    print(f"plane_num_avg             : {plane_num_avg:.3f}")

    global_summary["plane_intra_mean"] = plane_intra
    global_summary["plane_inter_mean"] = plane_inter
    global_summary["plane_separation_ratio"] = plane_sep
    global_summary["plane_num_avg"] = plane_num_avg

    print("=" * 80)

    # ========================================================
    # Save per-sample CSV
    # ========================================================

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = sorted(per_sample_rows[0].keys()) if per_sample_rows else ["sample_idx"]

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_sample_rows:
                writer.writerow(row)

        print(f"per-sample csv saved to: {output_csv}")

        summary_path = output_csv.with_suffix(".summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            for k, v in global_summary.items():
                f.write(f"{k}: {v}\n")

        print(f"summary saved to       : {summary_path}")


if __name__ == "__main__":
    main()