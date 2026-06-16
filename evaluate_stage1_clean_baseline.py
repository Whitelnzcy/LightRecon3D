import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone
from models.clean_plane_mask_head import CleanPlaneMaskHead
from train_stage1_clean_baseline import class_target_from_matches, move_batch
from train_stage1_plane_masks import (
    build_views,
    feature_maps_from_result,
    masks_for_plane_ids,
    match_queries,
    select_plane_ids,
    typed_boundary_maps,
)


COLORS = np.asarray(
    [[230, 57, 53], [33, 150, 243], [67, 160, 71], [255, 143, 0], [142, 36, 170], [0, 137, 123], [244, 81, 30], [117, 117, 117]],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate clean Stage1 plane mask baseline")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample_indices", default="1,7,10")
    parser.add_argument("--all_first_n", type=int, default=32)
    parser.add_argument("--existence_threshold", type=float, default=0.5)
    return parser.parse_args()


def colorize(ids):
    output = np.full((*ids.shape, 3), 245, dtype=np.uint8)
    for item in np.unique(ids):
        if item >= 0:
            output[ids == item] = COLORS[int(item) % len(COLORS)]
    return output


def image_uint8(image):
    image = image.detach().cpu()
    if float(image.min()) < -0.05:
        image = (image + 1.0) * 0.5
    return (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def boundary_f1(predicted, target, tolerance):
    pred = torch.zeros_like(predicted, dtype=torch.bool)
    gt = torch.zeros_like(target, dtype=torch.bool)
    pred[:, 1:] |= predicted[:, 1:] != predicted[:, :-1]
    pred[:, :-1] |= predicted[:, 1:] != predicted[:, :-1]
    pred[1:, :] |= predicted[1:, :] != predicted[:-1, :]
    pred[:-1, :] |= predicted[1:, :] != predicted[:-1, :]
    gt[:, 1:] |= target[:, 1:] != target[:, :-1]
    gt[:, :-1] |= target[:, 1:] != target[:, :-1]
    gt[1:, :] |= target[1:, :] != target[:-1, :]
    gt[:-1, :] |= target[1:, :] != target[:-1, :]
    kernel = tolerance * 2 + 1
    pred_d = F.max_pool2d(pred[None, None].float(), kernel, 1, tolerance)[0, 0] > 0
    gt_d = F.max_pool2d(gt[None, None].float(), kernel, 1, tolerance)[0, 0] > 0
    precision = (pred & gt_d).sum().float() / pred.sum().clamp_min(1)
    recall = (gt & pred_d).sum().float() / gt.sum().clamp_min(1)
    return float((2 * precision * recall / (precision + recall).clamp_min(1e-6)).cpu())


def evaluate_one(output, labels, args_like):
    _, plane_ids = select_plane_ids(labels[None], output["mask_logits"].shape[-2:], args_like.num_queries, args_like.min_plane_pixels)
    labels32, masks = masks_for_plane_ids(labels[None], output["mask_logits"].shape[-2:], plane_ids)
    targets = torch.stack(masks[0]) if masks[0] else output["mask_logits"].new_zeros((0, *output["mask_logits"].shape[-2:]))
    query_ids, target_ids = match_queries(output["mask_logits"], targets, args_like)
    class_logits = torch.cat((output["mask_logits"], output["background_logits"]), dim=0)
    predicted = class_logits.argmax(dim=0)
    gt_query = class_target_from_matches(labels32[0], targets, query_ids, target_ids, args_like.num_queries)
    ious = []
    size_bins = {"small": [], "medium": [], "large": []}
    confusion = np.zeros((len(target_ids), args_like.num_queries), dtype=np.int64)
    for local_id, (query_id, target_id) in enumerate(zip(query_ids, target_ids)):
        gt = targets[target_id] > 0.5
        pred = predicted == int(query_id)
        iou = float(((pred & gt).sum() / (pred | gt).sum().clamp_min(1)).cpu())
        ious.append(iou)
        ratio = float(gt.sum()) / float(gt.numel())
        size_bins["small" if ratio < 0.02 else "medium" if ratio < 0.10 else "large"].append(iou)
        for q in range(args_like.num_queries):
            confusion[local_id, q] = int(((predicted == q) & gt).sum())
    valid = gt_query < args_like.num_queries
    leakage = float(((predicted != gt_query) & valid & (predicted < args_like.num_queries)).sum() / valid.sum().clamp_min(1))
    background = gt_query == args_like.num_queries
    background_error = float(((predicted < args_like.num_queries) & background).sum() / background.sum().clamp_min(1))
    normalized_rows = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    split_rate = float(((normalized_rows > 0.10).sum(axis=1) > 1).mean()) if len(normalized_rows) else 0.0
    normalized_cols = confusion / np.maximum(confusion.sum(axis=0, keepdims=True), 1)
    merge_rate = float(((normalized_cols > 0.10).sum(axis=0) > 1).mean()) if normalized_cols.shape[1] else 0.0
    active = int((output["existence_logits"].sigmoid() > args_like.existence_threshold).sum())
    metrics = {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "small_plane_iou": float(np.mean(size_bins["small"])) if size_bins["small"] else "",
        "medium_plane_iou": float(np.mean(size_bins["medium"])) if size_bins["medium"] else "",
        "large_plane_iou": float(np.mean(size_bins["large"])) if size_bins["large"] else "",
        "plane_count_mae": abs(active - len(target_ids)),
        "existence_precision": min(active, len(target_ids)) / max(active, 1),
        "existence_recall": min(active, len(target_ids)) / max(len(target_ids), 1),
        "merge_rate": merge_rate,
        "split_rate": split_rate,
        "leakage_rate": leakage,
        "background_error_rate": background_error,
        "boundary_f1_t2": boundary_f1(predicted, gt_query, 2),
        "boundary_f1_t4": boundary_f1(predicted, gt_query, 4),
    }
    return metrics, predicted, gt_query, confusion, query_ids, target_ids


def classify_failure(metrics):
    failures = []
    if metrics["mean_iou"] < 0.35:
        failures.append("mask_bad")
    if metrics["plane_count_mae"] >= 2:
        failures.append("count_bad")
    if metrics["merge_rate"] >= 0.25:
        failures.append("merge")
    if metrics["split_rate"] >= 0.35:
        failures.append("split")
    if metrics["leakage_rate"] >= 0.25:
        failures.append("leakage")
    if metrics["background_error_rate"] >= 0.20:
        failures.append("background_error")
    if metrics["boundary_f1_t2"] < 0.65:
        failures.append("boundary_bad")
    if not failures:
        failures.append("ok")
    return failures


def save_figure(path, rgb, gt_query, predicted, error, existence, confusion):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        (rgb, "Input RGB"),
        (colorize(gt_query.cpu().numpy()), "GT matched query"),
        (colorize(predicted.cpu().numpy()), "Predicted query"),
        (error, "Error map"),
        (np.tile((existence[None, :, None] * 255).astype(np.uint8), (64, 1, 3)), "Existence"),
        (confusion, "Confusion"),
    ]
    for axis, (image, title) in zip(axes.flat, panels):
        axis.imshow(image, cmap=None if image.ndim == 3 else "magma")
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    args_like = argparse.Namespace(**config)
    dataset = Structured3DDataset(args.root_dir, split=args.split, image_size=(512, 512), input_mode=config.get("input_mode", "pair"))
    explicit = [int(x) for x in args.sample_indices.split(",") if x.strip()]
    indices = sorted(set(explicit + list(range(min(args.all_first_n, len(dataset))))))
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)
    backbone = build_dust3r_backbone(args.weights_path, device=device)
    backbone.eval()
    head = CleanPlaneMaskHead(
        feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"],
        num_queries=config["num_queries"],
        num_decoder_layers=config["decoder_layers"],
        num_heads=config["decoder_heads"],
    ).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    rows = []
    failure_cases = []
    for local_idx, batch in zip(indices, loader):
        batch = move_batch(batch, device)
        view1, view2 = build_views(batch, "clean_eval")
        result1, _ = backbone(view1, view2)
        features = feature_maps_from_result(result1, view1["img"], (0, 6, 9, config.get("feature_index", 12)))
        output = head(features["deep"])
        metrics, predicted, gt_query, confusion, _, _ = evaluate_one(
            {key: value[0] if value.ndim > 1 else value for key, value in output.items()},
            batch.get("gt_plane1", batch["gt_plane"])[0],
            args_like,
        )
        metrics["sample_idx"] = local_idx
        failures = classify_failure(metrics)
        metrics["failure_types"] = "|".join(failures)
        rows.append(metrics)
        failure_cases.append(
            {
                "sample_idx": local_idx,
                "failure_types": failures,
                "metrics": metrics,
            }
        )
        error_rgb = colorize(predicted.cpu().numpy())
        error = (gt_query < config["num_queries"]) & (predicted != gt_query)
        error_rgb[error.cpu().numpy()] = np.asarray([255, 0, 255], dtype=np.uint8)
        save_figure(
            output_dir / f"{args.split}_{local_idx:06d}_clean_diagnostic.png",
            image_uint8(view1["img"][0]),
            gt_query,
            predicted,
            error_rgb,
            output["existence_logits"][0].sigmoid().detach().cpu().numpy(),
            confusion,
        )
    fieldnames = list(rows[0].keys()) if rows else []
    with (output_dir / "diagnostics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "diagnostics_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "failure_cases.json").write_text(json.dumps(failure_cases, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
