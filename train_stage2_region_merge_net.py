import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fit_plane_np(points):
    if len(points) < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0, np.zeros((0,), dtype=np.float32)
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered.astype(np.float32), full_matrices=False)
    normal = vh[-1].astype(np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1e-8)
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    residual = np.abs(points @ normal + offset).astype(np.float32)
    return normal, offset, residual


def majority_label(values):
    values = values[values >= 0]
    if len(values) == 0:
        return -1, 0.0
    labels, counts = np.unique(values, return_counts=True)
    index = int(np.argmax(counts))
    return int(labels[index]), float(counts[index] / max(len(values), 1))


def bbox_from_xy(xy):
    if len(xy) == 0:
        return np.zeros((4,), dtype=np.float32)
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    return np.array([mins[0], mins[1], maxs[0], maxs[1]], dtype=np.float32)


def bbox_gap(a, b):
    dx = max(0.0, max(float(a[0] - b[2]), float(b[0] - a[2])))
    dy = max(0.0, max(float(a[1] - b[3]), float(b[1] - a[3])))
    return math.sqrt(dx * dx + dy * dy)


def bbox_iou(a, b):
    ix0 = max(float(a[0]), float(b[0]))
    iy0 = max(float(a[1]), float(b[1]))
    ix1 = min(float(a[2]), float(b[2]))
    iy1 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / max(union, 1e-8)


def load_regions(npz_path, min_points=64):
    raw = np.load(npz_path)
    points = raw["points"].astype(np.float32)
    assignment = raw["point_plane_ids"].astype(np.int32)
    gt = raw.get("gt_point_plane_ids")
    gt = gt.astype(np.int32) if gt is not None else np.full_like(assignment, -1)
    xy = raw.get("pixel_xy")
    xy = xy.astype(np.float32) if xy is not None else np.zeros((len(points), 2), dtype=np.float32)
    confidence = raw.get("point_confidence")
    confidence = confidence.astype(np.float32) if confidence is not None else np.ones((len(points),), dtype=np.float32)
    margin = raw.get("point_margin")
    margin = margin.astype(np.float32) if margin is not None else np.ones((len(points),), dtype=np.float32)
    line_prob = raw.get("line_prob")
    line_prob = line_prob.astype(np.float32) if line_prob is not None else np.zeros((len(points),), dtype=np.float32)

    regions = []
    for plane_id in sorted(int(x) for x in np.unique(assignment) if int(x) >= 0):
        mask = assignment == plane_id
        count = int(mask.sum())
        if count < min_points:
            continue
        pts = points[mask]
        normal, offset, residual = fit_plane_np(pts)
        label, purity = majority_label(gt[mask])
        centroid = pts.mean(axis=0).astype(np.float32)
        bbox = bbox_from_xy(xy[mask])
        regions.append(
            {
                "plane_id": int(plane_id),
                "count": count,
                "area_frac": float(count / max(len(points), 1)),
                "normal": normal,
                "offset": float(offset),
                "centroid": centroid,
                "residual_mean": float(residual.mean()) if len(residual) else 0.0,
                "residual_p95": float(np.quantile(residual, 0.95)) if len(residual) else 0.0,
                "confidence_mean": float(confidence[mask].mean()),
                "margin_mean": float(margin[mask].mean()),
                "line_mean": float(line_prob[mask].mean()),
                "bbox": bbox,
                "source_gt": int(label),
                "source_gt_purity": float(purity),
            }
        )
    return points, assignment, gt, regions


def pair_features(region_a, region_b):
    normal_a = region_a["normal"]
    normal_b = region_b["normal"]
    abs_dot = float(abs(np.dot(normal_a, normal_b)))
    abs_dot = min(max(abs_dot, 0.0), 1.0)
    angle = math.degrees(math.acos(abs_dot)) / 90.0
    offset = abs(abs(float(region_a["offset"])) - abs(float(region_b["offset"])))
    centroid_dist = float(np.linalg.norm(region_a["centroid"] - region_b["centroid"]))
    residual_ab = abs(float(region_a["centroid"] @ normal_b + region_b["offset"]))
    residual_ba = abs(float(region_b["centroid"] @ normal_a + region_a["offset"]))
    mutual_residual = 0.5 * (residual_ab + residual_ba)
    min_area = min(region_a["area_frac"], region_b["area_frac"])
    max_area = max(region_a["area_frac"], region_b["area_frac"])
    area_ratio = min_area / max(max_area, 1e-8)
    return np.asarray(
        [
            angle,
            offset,
            centroid_dist,
            mutual_residual,
            area_ratio,
            min_area,
            max_area,
            abs(region_a["residual_mean"] - region_b["residual_mean"]),
            max(region_a["residual_mean"], region_b["residual_mean"]),
            0.5 * (region_a["confidence_mean"] + region_b["confidence_mean"]),
            0.5 * (region_a["margin_mean"] + region_b["margin_mean"]),
            0.5 * (region_a["line_mean"] + region_b["line_mean"]),
            bbox_gap(region_a["bbox"], region_b["bbox"]),
            bbox_iou(region_a["bbox"], region_b["bbox"]),
        ],
        dtype=np.float32,
    )


def collect_pairs(input_dir, pattern, min_points, require_pure_gt):
    rows = []
    for path in sorted(Path(input_dir).glob(pattern)):
        _, _, _, regions = load_regions(path, min_points=min_points)
        for i, region_a in enumerate(regions):
            for j in range(i + 1, len(regions)):
                region_b = regions[j]
                valid_label = region_a["source_gt"] >= 0 and region_b["source_gt"] >= 0
                if require_pure_gt and (
                    region_a["source_gt_purity"] < 0.5
                    or region_b["source_gt_purity"] < 0.5
                    or not valid_label
                ):
                    continue
                label = 1.0 if valid_label and region_a["source_gt"] == region_b["source_gt"] else 0.0
                rows.append(
                    {
                        "path": str(path),
                        "plane_a": region_a["plane_id"],
                        "plane_b": region_b["plane_id"],
                        "features": pair_features(region_a, region_b),
                        "label": label,
                    }
                )
    return rows


class RegionMergeMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, depth=3):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True), nn.LayerNorm(hidden_dim)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features).squeeze(-1)


def metrics_from_logits(logits, labels, threshold):
    probs = torch.sigmoid(logits)
    pred = probs >= threshold
    gt = labels >= 0.5
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_rate": float(gt.float().mean().item()) if labels.numel() else 0.0,
    }


def tensorize(rows, device):
    if not rows:
        return (
            torch.zeros((0, 14), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
        )
    x = torch.from_numpy(np.stack([row["features"] for row in rows], axis=0)).to(device)
    y = torch.tensor([row["label"] for row in rows], dtype=torch.float32, device=device)
    return x, y


def main():
    parser = argparse.ArgumentParser("Train Stage2 region-pair merge classifier")
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*_full_pointcloud_editable_planes_data.npz")
    parser.add_argument("--min_points", type=int, default=64)
    parser.add_argument("--require_pure_gt", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = collect_pairs(args.train_dir, args.pattern, args.min_points, args.require_pure_gt)
    val_rows = collect_pairs(args.val_dir, args.pattern, args.min_points, args.require_pure_gt)
    if not train_rows:
        raise RuntimeError(f"No train pairs found in {args.train_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, train_y = tensorize(train_rows, device)
    val_x, val_y = tensorize(val_rows, device)
    model = RegionMergeMLP(train_x.shape[1], args.hidden_dim, args.depth).to(device)
    pos = float(train_y.sum().item())
    neg = float(len(train_y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_f1 = -1.0
    best_payload = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_metrics = metrics_from_logits(model(train_x), train_y, args.threshold)
            val_logits = model(val_x) if len(val_y) else torch.zeros((0,), device=device)
            val_metrics = metrics_from_logits(val_logits, val_y, args.threshold)
        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_payload = {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "input_dim": int(train_x.shape[1]),
                "feature_names": [
                    "normal_angle_norm",
                    "offset_abs_diff",
                    "centroid_distance",
                    "mutual_centroid_residual",
                    "area_ratio",
                    "min_area_frac",
                    "max_area_frac",
                    "residual_mean_abs_diff",
                    "max_residual_mean",
                    "mean_confidence",
                    "mean_margin",
                    "mean_line_prob",
                    "bbox_gap",
                    "bbox_iou",
                ],
                "val": val_metrics,
            }
            torch.save(best_payload, output_dir / "best.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    torch.save(best_payload, output_dir / "latest_best_copy.pt")

    fields = ["path", "plane_a", "plane_b", "label"]
    with (output_dir / "val_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in val_rows:
            writer.writerow({key: row[key] for key in fields})
    summary = {
        "train_pairs": len(train_rows),
        "train_positive_pairs": int(sum(row["label"] for row in train_rows)),
        "val_pairs": len(val_rows),
        "val_positive_pairs": int(sum(row["label"] for row in val_rows)),
        "best_f1": best_f1,
        "best_checkpoint": str(output_dir / "best.pt"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
