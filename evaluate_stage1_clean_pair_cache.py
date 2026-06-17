import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from models.clean_plane_mask_head import CleanPlaneMaskHead
from train_stage1_clean_baseline import class_target_from_matches
from train_stage1_plane_masks import masks_for_plane_ids, match_queries, select_plane_ids


def parse_args():
    parser = argparse.ArgumentParser(
        "Evaluate a clean pair checkpoint on cached pair features with raw and gated assignment"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_cache_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument(
        "--thresholds",
        default="0.3,0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated existence thresholds for gated evaluation",
    )
    return parser.parse_args()


def parse_thresholds(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            value = float(part)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid threshold {value}; expected [0,1]")
            values.append(value)
    if not values:
        raise ValueError("At least one threshold is required")
    return values


def build_head(checkpoint, device):
    config = checkpoint["args"]
    head = CleanPlaneMaskHead(
        feature_dim=int(config.get("feature_dim", 768)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_queries=int(config.get("num_queries", 8)),
        num_decoder_layers=int(config.get("decoder_layers", 3)),
        num_heads=int(config.get("decoder_heads", 8)),
    ).to(device)
    head.load_state_dict(checkpoint["head"], strict=True)
    head.eval()
    return head, config


def output_slice(output, index):
    return {key: value[index] for key, value in output.items()}


def assignment(mask_logits, background_logits, existence_logits, threshold=None):
    class_logits = torch.cat((mask_logits, background_logits), dim=0).clone()
    if threshold is not None:
        active = existence_logits.sigmoid() > threshold
        class_logits[:-1][~active] = -1e4
    return class_logits.argmax(dim=0)


def mean_or_empty(values):
    return float(np.mean(values)) if values else ""


def sample_metrics(output, labels, config, threshold=None):
    num_queries = int(config.get("num_queries", 8))
    min_plane_pixels = int(config.get("min_plane_pixels", 4))
    args_like = argparse.Namespace(
        num_queries=num_queries,
        min_plane_pixels=min_plane_pixels,
        match_bce_weight=float(config.get("match_bce_weight", 1.0)),
        match_dice_weight=float(config.get("match_dice_weight", 2.0)),
    )

    _, plane_ids = select_plane_ids(
        labels[None],
        output["mask_logits"].shape[-2:],
        num_queries,
        min_plane_pixels,
    )
    labels32, masks = masks_for_plane_ids(
        labels[None],
        output["mask_logits"].shape[-2:],
        plane_ids,
    )
    targets = (
        torch.stack(masks[0])
        if masks[0]
        else output["mask_logits"].new_zeros((0, *output["mask_logits"].shape[-2:]))
    )
    query_ids, target_ids = match_queries(output["mask_logits"], targets, args_like)
    predicted = assignment(
        output["mask_logits"],
        output["background_logits"],
        output["existence_logits"],
        threshold=threshold,
    )
    gt_query = class_target_from_matches(
        labels32[0],
        targets,
        query_ids,
        target_ids,
        num_queries,
    )

    ious = []
    size_bins = {"small": [], "medium": [], "large": []}
    for query_id, target_id in zip(query_ids, target_ids):
        gt = targets[target_id] > 0.5
        pred = predicted == int(query_id)
        union = (pred | gt).sum().clamp_min(1)
        iou = float(((pred & gt).sum() / union).detach().cpu())
        ious.append(iou)
        ratio = float(gt.sum().detach().cpu()) / max(float(gt.numel()), 1.0)
        bucket = "small" if ratio < 0.02 else "medium" if ratio < 0.10 else "large"
        size_bins[bucket].append(iou)

    valid = gt_query < num_queries
    background = gt_query == num_queries
    wrong_plane = valid & (predicted < num_queries) & (predicted != gt_query)
    plane_miss = valid & (predicted == num_queries)
    background_error = background & (predicted < num_queries)

    existence_prob = output["existence_logits"].sigmoid()
    existence_threshold = 0.5 if threshold is None else float(threshold)
    active_set = set(
        torch.nonzero(existence_prob > existence_threshold, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )
    matched_set = {int(value) for value in query_ids.tolist()}
    true_positive = len(active_set & matched_set)

    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "small_plane_iou": mean_or_empty(size_bins["small"]),
        "medium_plane_iou": mean_or_empty(size_bins["medium"]),
        "large_plane_iou": mean_or_empty(size_bins["large"]),
        "leakage_rate": float(wrong_plane.sum() / valid.sum().clamp_min(1)),
        "plane_miss_rate": float(plane_miss.sum() / valid.sum().clamp_min(1)),
        "background_error_rate": float(
            background_error.sum() / background.sum().clamp_min(1)
        ),
        "gt_planes": len(target_ids),
        "pred_planes": len(active_set),
        "plane_count_abs_error": abs(len(active_set) - len(target_ids)),
        "existence_precision_true": true_positive / max(len(active_set), 1),
        "existence_recall_true": true_positive / max(len(matched_set), 1),
    }


def aggregate(rows):
    totals = defaultdict(float)
    counts = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals[key] += float(value)
                counts[key] += 1
    return {
        key: totals[key] / max(counts[key], 1)
        for key in sorted(totals)
    }


def evaluate_mode(head, samples, device, config, threshold, mode_name):
    overall_rows = []
    view1_rows = []
    view2_rows = []

    for sample in samples:
        feature1 = sample["feature1"].to(device=device, dtype=torch.float32)
        feature2 = sample["feature2"].to(device=device, dtype=torch.float32)
        gt1 = sample["gt_plane1"].to(device)
        gt2 = sample["gt_plane2"].to(device)
        with torch.no_grad():
            output = head(torch.cat((feature1, feature2), dim=0))

        batch_size = feature1.shape[0]
        for index in range(batch_size):
            metrics1 = sample_metrics(
                output_slice({key: value[:batch_size] for key, value in output.items()}, index),
                gt1[index],
                config,
                threshold=threshold,
            )
            metrics2 = sample_metrics(
                output_slice({key: value[batch_size:] for key, value in output.items()}, index),
                gt2[index],
                config,
                threshold=threshold,
            )
            view1_rows.append(metrics1)
            view2_rows.append(metrics2)
            combined = {}
            for key in metrics1:
                value1 = metrics1[key]
                value2 = metrics2[key]
                if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    combined[key] = 0.5 * (float(value1) + float(value2))
            overall_rows.append(combined)

    summary = {
        "mode": mode_name,
        "threshold": "" if threshold is None else float(threshold),
        **aggregate(overall_rows),
    }
    for prefix, rows in (("view1", view1_rows), ("view2", view2_rows)):
        for key, value in aggregate(rows).items():
            summary[f"{prefix}_{key}"] = value
    summary["view_gap"] = abs(
        summary.get("view1_mean_iou", 0.0) - summary.get("view2_mean_iou", 0.0)
    )
    return summary


def main():
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head, config = build_head(checkpoint, device)
    cache = torch.load(args.feature_cache_path, map_location="cpu", weights_only=False)
    samples = cache[args.split]

    summaries = [evaluate_mode(head, samples, device, config, None, "raw")]
    for threshold in thresholds:
        summaries.append(
            evaluate_mode(
                head,
                samples,
                device,
                config,
                threshold,
                f"gated_{threshold:.2f}",
            )
        )

    (output_dir / "threshold_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "threshold_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print("=" * 104)
    print(
        f"{'mode':<12} {'thr':>5} {'mIoU':>8} {'v1':>8} {'v2':>8} "
        f"{'gap':>8} {'leak':>8} {'miss':>8} {'bgerr':>8} "
        f"{'cnt':>7} {'exP':>7} {'exR':>7}"
    )
    for row in summaries:
        print(
            f"{row['mode']:<12} {str(row['threshold']):>5} "
            f"{row.get('mean_iou', 0):8.4f} "
            f"{row.get('view1_mean_iou', 0):8.4f} "
            f"{row.get('view2_mean_iou', 0):8.4f} "
            f"{row.get('view_gap', 0):8.4f} "
            f"{row.get('leakage_rate', 0):8.4f} "
            f"{row.get('plane_miss_rate', 0):8.4f} "
            f"{row.get('background_error_rate', 0):8.4f} "
            f"{row.get('plane_count_abs_error', 0):7.3f} "
            f"{row.get('existence_precision_true', 0):7.3f} "
            f"{row.get('existence_recall_true', 0):7.3f}"
        )
    print("=" * 104)
    print(f"Saved summaries to: {output_dir}")


if __name__ == "__main__":
    main()
