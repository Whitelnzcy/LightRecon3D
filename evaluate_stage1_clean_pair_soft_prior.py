import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.clean_plane_mask_head import CleanPlaneMaskHead
from train_stage1_clean_baseline import class_target_from_matches
from train_stage1_plane_masks import masks_for_plane_ids, match_queries, select_plane_ids


def parse_args():
    parser = argparse.ArgumentParser(
        "Evaluate soft existence priors on clean pair cached features"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_cache_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument(
        "--alphas",
        default="0,0.1,0.25,0.5,1.0,2.0",
        help="Comma-separated multipliers for log-sigmoid existence prior",
    )
    return parser.parse_args()


def parse_alphas(text):
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one alpha is required")
    if any(value < 0 for value in values):
        raise ValueError("All alphas must be non-negative")
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


def slice_output(output, index):
    return {key: value[index] for key, value in output.items()}


def soft_assignment(mask_logits, background_logits, existence_logits, alpha):
    prior = float(alpha) * F.logsigmoid(existence_logits)[:, None, None]
    query_logits = mask_logits + prior
    class_logits = torch.cat((query_logits, background_logits), dim=0)
    return class_logits.argmax(dim=0)


def mean_or_empty(values):
    return float(np.mean(values)) if values else ""


def sample_metrics(output, labels, config, alpha):
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
    predicted = soft_assignment(
        output["mask_logits"],
        output["background_logits"],
        output["existence_logits"],
        alpha,
    )
    gt_query = class_target_from_matches(
        labels32[0],
        targets,
        query_ids,
        target_ids,
        num_queries,
    )

    ious = []
    bins = {"small": [], "medium": [], "large": []}
    for query_id, target_id in zip(query_ids, target_ids):
        gt = targets[target_id] > 0.5
        pred = predicted == int(query_id)
        union = (pred | gt).sum().clamp_min(1)
        iou = float(((pred & gt).sum() / union).detach().cpu())
        ious.append(iou)
        ratio = float(gt.sum().detach().cpu()) / max(float(gt.numel()), 1.0)
        bucket = "small" if ratio < 0.02 else "medium" if ratio < 0.10 else "large"
        bins[bucket].append(iou)

    valid = gt_query < num_queries
    background = gt_query == num_queries
    wrong_plane = valid & (predicted < num_queries) & (predicted != gt_query)
    plane_miss = valid & (predicted == num_queries)
    background_error = background & (predicted < num_queries)

    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "small_plane_iou": mean_or_empty(bins["small"]),
        "medium_plane_iou": mean_or_empty(bins["medium"]),
        "large_plane_iou": mean_or_empty(bins["large"]),
        "leakage_rate": float(wrong_plane.sum() / valid.sum().clamp_min(1)),
        "plane_miss_rate": float(plane_miss.sum() / valid.sum().clamp_min(1)),
        "background_error_rate": float(
            background_error.sum() / background.sum().clamp_min(1)
        ),
    }


def aggregate(rows):
    totals = defaultdict(float)
    counts = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals[key] += float(value)
                counts[key] += 1
    return {key: totals[key] / max(counts[key], 1) for key in sorted(totals)}


def evaluate_alpha(head, samples, device, config, alpha):
    overall = []
    view1 = []
    view2 = []

    for sample in samples:
        feature1 = sample["feature1"].to(device=device, dtype=torch.float32)
        feature2 = sample["feature2"].to(device=device, dtype=torch.float32)
        gt1 = sample["gt_plane1"].to(device)
        gt2 = sample["gt_plane2"].to(device)

        with torch.no_grad():
            output = head(torch.cat((feature1, feature2), dim=0))

        batch_size = feature1.shape[0]
        output1 = {key: value[:batch_size] for key, value in output.items()}
        output2 = {key: value[batch_size:] for key, value in output.items()}

        for index in range(batch_size):
            row1 = sample_metrics(slice_output(output1, index), gt1[index], config, alpha)
            row2 = sample_metrics(slice_output(output2, index), gt2[index], config, alpha)
            view1.append(row1)
            view2.append(row2)
            overall.append(
                {
                    key: 0.5 * (float(row1[key]) + float(row2[key]))
                    for key in row1
                    if isinstance(row1[key], (int, float))
                    and isinstance(row2[key], (int, float))
                }
            )

    summary = {"alpha": float(alpha), **aggregate(overall)}
    for prefix, rows in (("view1", view1), ("view2", view2)):
        for key, value in aggregate(rows).items():
            summary[f"{prefix}_{key}"] = value
    summary["view_gap"] = abs(
        summary.get("view1_mean_iou", 0.0) - summary.get("view2_mean_iou", 0.0)
    )
    return summary


def main():
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head, config = build_head(checkpoint, device)
    cache = torch.load(args.feature_cache_path, map_location="cpu", weights_only=False)
    samples = cache[args.split]

    rows = [evaluate_alpha(head, samples, device, config, alpha) for alpha in alphas]

    (output_dir / "soft_prior_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "soft_prior_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 92)
    print(
        f"{'alpha':>7} {'mIoU':>8} {'v1':>8} {'v2':>8} {'gap':>8} "
        f"{'leak':>8} {'miss':>8} {'bgerr':>8}"
    )
    for row in rows:
        print(
            f"{row['alpha']:7.2f} "
            f"{row.get('mean_iou', 0):8.4f} "
            f"{row.get('view1_mean_iou', 0):8.4f} "
            f"{row.get('view2_mean_iou', 0):8.4f} "
            f"{row.get('view_gap', 0):8.4f} "
            f"{row.get('leakage_rate', 0):8.4f} "
            f"{row.get('plane_miss_rate', 0):8.4f} "
            f"{row.get('background_error_rate', 0):8.4f}"
        )
    print("=" * 92)
    print(f"Saved summaries to: {output_dir}")


if __name__ == "__main__":
    main()
