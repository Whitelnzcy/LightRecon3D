import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


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


def safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(weights_path, ckpt_path, device, hidden_dim=768, plane_embed_dim=16):
    backbone = build_dust3r_backbone(weights_path, device=device)

    model = LightReconModel(
        dust3r_backbone=backbone,
        hidden_dim=hidden_dim,
        plane_embed_dim=plane_embed_dim,
    ).to(device)

    ckpt = safe_load(ckpt_path, device)
    state = ckpt.get("model", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Load] ckpt: {ckpt_path}")
    print(f"[Load] missing keys   : {len(missing)}")
    print(f"[Load] unexpected keys: {len(unexpected)}")

    model.eval()
    return model


def resize_gt_plane(gt_plane, size):
    if gt_plane.ndim == 4 and gt_plane.shape[1] == 1:
        gt_plane = gt_plane[:, 0]

    if gt_plane.shape[-2:] == size:
        return gt_plane.long()

    x = gt_plane.unsqueeze(1).float()
    x = F.interpolate(x, size=size, mode="nearest")
    return x[:, 0].long()


def get_plane_embedding(res, prefer_lowres=True):
    if prefer_lowres and "pred_plane_lowres" in res:
        emb = res["pred_plane_lowres"]
    elif "pred_plane" in res:
        emb = res["pred_plane"]
    elif "pred_plane_embedding" in res:
        emb = res["pred_plane_embedding"]
    else:
        raise KeyError(f"No plane embedding found. keys={list(res.keys())}")

    if emb.ndim != 4:
        raise ValueError(f"Bad embedding shape: {tuple(emb.shape)}")

    return emb


def compute_iou_matrix(pred_labels, gt_labels, pred_ids, gt_ids):
    iou = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float64)

    for i, pid in enumerate(pred_ids):
        p_mask = pred_labels == pid
        for j, gid in enumerate(gt_ids):
            g_mask = gt_labels == gid
            inter = np.logical_and(p_mask, g_mask).sum()
            union = np.logical_or(p_mask, g_mask).sum()
            iou[i, j] = inter / max(union, 1)

    return iou


def evaluate_one_image(
    emb_chw,
    gt_hw,
    min_plane_pixels=4,
    max_pixels=4096,
    normalize_emb=True,
    random_state=0,
):
    """
    emb_chw: torch [C, H, W]
    gt_hw:   torch [H, W]
    """
    emb = emb_chw.permute(1, 2, 0).detach().float().cpu().numpy()
    gt = gt_hw.detach().cpu().numpy().astype(np.int64)

    h, w, c = emb.shape

    valid = (gt >= 0) & (gt != 255)

    gt_ids = []
    for gid in np.unique(gt[valid]):
        count = int((gt == gid).sum())
        if count >= min_plane_pixels:
            gt_ids.append(int(gid))

    if len(gt_ids) < 2:
        return None

    valid = np.zeros_like(gt, dtype=bool)
    for gid in gt_ids:
        valid |= gt == gid

    y = gt[valid]
    x = emb[valid]

    if x.shape[0] > max_pixels:
        idx = np.linspace(0, x.shape[0] - 1, max_pixels).astype(np.int64)
        x = x[idx]
        y = y[idx]

    if normalize_emb:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norm, 1e-8)

    k = len(gt_ids)

    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    pred = km.fit_predict(x)

    pred_ids = list(range(k))

    iou = compute_iou_matrix(pred, y, pred_ids, gt_ids)

    # Hungarian matching maximizes total IoU.
    row_ind, col_ind = linear_sum_assignment(-iou)
    matched_ious = iou[row_ind, col_ind]

    mean_matched_iou = float(matched_ious.mean())
    median_matched_iou = float(np.median(matched_ious))

    # For every GT plane, best predicted cluster IoU.
    best_gt_iou = iou.max(axis=0)
    mean_best_gt_iou = float(best_gt_iou.mean())
    min_best_gt_iou = float(best_gt_iou.min())

    ari = float(adjusted_rand_score(y, pred))
    nmi = float(normalized_mutual_info_score(y, pred))

    return {
        "num_gt_planes": k,
        "num_pixels_used": int(x.shape[0]),
        "mean_matched_iou": mean_matched_iou,
        "median_matched_iou": median_matched_iou,
        "mean_best_gt_iou": mean_best_gt_iou,
        "min_best_gt_iou": min_best_gt_iou,
        "ari": ari,
        "nmi": nmi,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Evaluate plane embedding clustering")

    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)

    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)

    parser.add_argument("--prefer_lowres_plane", action="store_true")
    parser.add_argument("--min_plane_pixels", type=int, default=4)
    parser.add_argument("--max_pixels", type=int, default=4096)
    parser.add_argument("--no_normalize_emb", action="store_true")

    parser.add_argument("--output_csv", required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Plane Embedding Oracle-K KMeans Evaluation")
    print("=" * 80)
    print(f"device      : {device}")
    print(f"ckpt_path   : {args.ckpt_path}")
    print(f"split       : {args.split}")
    print(f"num_samples : {args.num_samples}")
    print(f"output_csv  : {args.output_csv}")
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

    rows = []

    for sample_idx in range(n):
        sample = dataset[sample_idx]

        batch = {
            "img": sample["img"].unsqueeze(0).to(device),
            "gt_line": sample["gt_line"].unsqueeze(0).to(device),
            "gt_plane": sample["gt_plane"].unsqueeze(0).to(device),
        }

        view1, view2 = build_views_from_batch(batch, prefix=f"cluster_{args.split}_{sample_idx}")

        res1, _ = model(view1, view2)

        emb = get_plane_embedding(res1, prefer_lowres=args.prefer_lowres_plane)
        gt_plane = resize_gt_plane(batch["gt_plane"], size=emb.shape[-2:])

        metrics = evaluate_one_image(
            emb_chw=emb[0],
            gt_hw=gt_plane[0],
            min_plane_pixels=args.min_plane_pixels,
            max_pixels=args.max_pixels,
            normalize_emb=not args.no_normalize_emb,
            random_state=sample_idx,
        )

        if metrics is None:
            print(f"[{sample_idx+1}/{n}] skipped")
            continue

        row = {
            "sample_idx": sample_idx,
            **metrics,
        }
        rows.append(row)

        print(
            f"[{sample_idx+1}/{n}] "
            f"K={metrics['num_gt_planes']}, "
            f"mIoU={metrics['mean_matched_iou']:.4f}, "
            f"bestGT={metrics['mean_best_gt_iou']:.4f}, "
            f"ARI={metrics['ari']:.4f}, "
            f"NMI={metrics['nmi']:.4f}"
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_idx",
        "num_gt_planes",
        "num_pixels_used",
        "mean_matched_iou",
        "median_matched_iou",
        "mean_best_gt_iou",
        "min_best_gt_iou",
        "ari",
        "nmi",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"valid images: {len(rows)}")

    if rows:
        for key in [
            "mean_matched_iou",
            "median_matched_iou",
            "mean_best_gt_iou",
            "min_best_gt_iou",
            "ari",
            "nmi",
        ]:
            vals = np.array([r[key] for r in rows], dtype=np.float64)
            print(f"{key:22s}: mean={vals.mean():.6f}, median={np.median(vals):.6f}")

    print(f"output csv: {output_csv}")


if __name__ == "__main__":
    main()