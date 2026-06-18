import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_stage1_clean_baseline import class_target_from_matches
from train_stage1_plane_masks import masks_for_plane_ids, match_queries, select_plane_ids
from visualize_predictions import (
    QUERY_COLORS,
    STAGE_NAMES,
    build_multiscale_head,
    colorize_class_map,
    load_cache_metadata,
    output_slice,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Ablate dynamic-K query suppression and merge for Stage1 plane masks"
    )
    parser.add_argument(
        "--checkpoint",
        default="local_outputs/stage1_multiscale_symref_balanced512_v1/best.pt",
    )
    parser.add_argument(
        "--feature_cache_path",
        default="local_outputs/feature_cache/multiscale_symref_balanced_train512_val128.pt",
    )
    parser.add_argument(
        "--output_dir",
        default="local_outputs/dynamic_query_postprocess_v1",
    )
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--min_query_pixels", type=int, default=16)
    parser.add_argument("--min_query_purity", type=float, default=0.25)
    parser.add_argument("--inactive_area_threshold", type=int, default=16)
    parser.add_argument("--focus_indices", default="6,10,44,72,74")
    parser.add_argument("--top_k_visuals", type=int, default=16)
    return parser.parse_args()


def safe_load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_image_to_numpy(image_tensor):
    image = image_tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def compute_raw_iou(predicted, gt_compact, query_ids, target_count):
    ious = []
    for gt_id in range(target_count):
        gt = gt_compact == gt_id
        if not gt.any():
            continue
        pred = predicted == query_ids[gt_id]
        union = (gt | pred).sum().clamp_min(1)
        ious.append(float(((gt & pred).sum() / union).detach().cpu()))
    return float(np.mean(ious)) if ious else 0.0


def compute_compact_iou(predicted_compact, gt_compact, target_count):
    ious = []
    for gt_id in range(target_count):
        gt = gt_compact == gt_id
        pred = predicted_compact == gt_id
        union = (gt | pred).sum().clamp_min(1)
        ious.append(float(((gt & pred).sum() / union).detach().cpu()))
    return float(np.mean(ious)) if ious else 0.0


def summarize_assignment(predicted, gt_compact, num_queries, target_count, min_pixels, purity):
    background_id = target_count
    confusion = torch.zeros(
        (num_queries + 1, target_count + 1), dtype=torch.long, device=predicted.device
    )
    pred_compact = predicted.clamp(max=num_queries)
    for pred_id in range(num_queries + 1):
        pred_mask = pred_compact == pred_id
        if not pred_mask.any():
            continue
        for gt_id in range(target_count + 1):
            confusion[pred_id, gt_id] = (pred_mask & (gt_compact == gt_id)).sum()

    query_main_gt = {}
    gt_to_queries = {gt_id: [] for gt_id in range(target_count)}
    active_queries = []
    for query_id in range(num_queries):
        pixels = int(confusion[query_id].sum().item())
        if pixels < min_pixels:
            continue
        main_gt = int(confusion[query_id].argmax().item())
        main_pixels = int(confusion[query_id, main_gt].item())
        query_purity = main_pixels / max(pixels, 1)
        query_main_gt[query_id] = {
            "main_gt": main_gt,
            "pixels": pixels,
            "main_pixels": main_pixels,
            "purity": query_purity,
        }
        active_queries.append(query_id)
        if main_gt < background_id and query_purity >= purity:
            gt_to_queries[main_gt].append(query_id)

    split_gt_count = 0
    redundant_query_count = 0
    max_split_parts = 0
    for query_list in gt_to_queries.values():
        split_parts = len(query_list)
        if split_parts > 1:
            split_gt_count += 1
            redundant_query_count += split_parts - 1
        max_split_parts = max(max_split_parts, split_parts)

    return {
        "confusion": confusion,
        "query_main_gt": query_main_gt,
        "gt_to_queries": gt_to_queries,
        "active_queries": active_queries,
        "split_gt_count": split_gt_count,
        "redundant_query_count": redundant_query_count,
        "max_split_parts": max_split_parts,
    }


def oracle_merge_prediction(predicted, assignment, target_count, background_query_id):
    """Merge queries that mostly explain the same GT plane.

    This is deliberately an oracle ablation because it uses GT-derived query
    attribution. It measures whether dynamic-K query merging could fix the
    current failure mode before implementing a no-GT merge rule.
    """
    merged = torch.full_like(predicted, target_count)
    suppressed = torch.zeros_like(predicted, dtype=torch.bool)
    merge_groups = {}
    for gt_id, query_list in assignment["gt_to_queries"].items():
        if not query_list:
            continue
        merge_groups[gt_id] = list(query_list)
        group_mask = torch.zeros_like(predicted, dtype=torch.bool)
        for query_id in query_list:
            group_mask |= predicted == int(query_id)
        merged[group_mask] = int(gt_id)

    for query_id, info in assignment["query_main_gt"].items():
        if int(info["main_gt"]) >= target_count:
            suppressed |= predicted == int(query_id)
            continue
        if int(info["main_gt"]) not in merge_groups:
            merged[predicted == int(query_id)] = int(info["main_gt"])
            merge_groups[int(info["main_gt"])] = [int(query_id)]

    merged[predicted == background_query_id] = target_count
    return merged, merge_groups, suppressed


def analyze_view(output, labels, config, args):
    num_queries = int(config.get("num_queries", 8))
    min_plane_pixels = int(config.get("min_plane_pixels", 4))
    match_args = argparse.Namespace(
        match_bce_weight=float(config.get("match_bce_weight", 1.0)),
        match_dice_weight=float(config.get("match_dice_weight", 2.0)),
    )
    target_hw = output["mask_logits"].shape[-2:]
    _, plane_ids = select_plane_ids(
        labels[None], target_hw, num_queries, min_plane_pixels
    )
    resized_labels, masks = masks_for_plane_ids(labels[None], target_hw, plane_ids)
    targets = (
        torch.stack(masks[0])
        if masks[0]
        else output["mask_logits"].new_zeros((0, *target_hw))
    )
    query_ids, target_ids = match_queries(output["mask_logits"], targets, match_args)
    gt_query = class_target_from_matches(
        resized_labels[0], targets, query_ids, target_ids, num_queries
    )
    target_count = int(targets.shape[0])

    gt_compact = torch.full_like(gt_query, target_count)
    for compact_id, query_id in enumerate(query_ids):
        gt_compact[gt_query == int(query_id)] = compact_id

    class_logits = torch.cat(
        (output["mask_logits"], output["background_logits"]), dim=0
    )
    predicted = class_logits.argmax(dim=0)
    background_query_id = num_queries
    raw_iou = compute_raw_iou(predicted, gt_compact, query_ids, target_count)

    assignment = summarize_assignment(
        predicted,
        gt_compact,
        num_queries,
        target_count,
        args.min_query_pixels,
        args.min_query_purity,
    )
    merged, merge_groups, suppressed = oracle_merge_prediction(
        predicted, assignment, target_count, background_query_id
    )
    merged_iou = compute_compact_iou(merged, gt_compact, target_count)

    valid_plane = gt_compact < target_count
    raw_wrong = valid_plane & (predicted < num_queries) & (predicted != gt_query)
    raw_miss = valid_plane & (predicted == num_queries)
    merged_wrong = valid_plane & (merged < target_count) & (merged != gt_compact)
    merged_miss = valid_plane & (merged == target_count)
    background = gt_compact == target_count
    merged_background_error = background & (merged < target_count)

    final_k = sum(1 for query_list in merge_groups.values() if query_list)
    return {
        "metrics": {
            "gt_plane_count": target_count,
            "raw_mean_iou": raw_iou,
            "merged_mean_iou": merged_iou,
            "iou_delta": merged_iou - raw_iou,
            "raw_active_query_count": len(assignment["active_queries"]),
            "final_k_pred": final_k,
            "raw_split_gt_count": assignment["split_gt_count"],
            "raw_redundant_query_count": assignment["redundant_query_count"],
            "raw_max_split_parts": assignment["max_split_parts"],
            "merged_leakage_rate": float(
                merged_wrong.sum() / valid_plane.sum().clamp_min(1)
            ),
            "raw_leakage_rate": float(
                raw_wrong.sum() / valid_plane.sum().clamp_min(1)
            ),
            "raw_miss_rate": float(raw_miss.sum() / valid_plane.sum().clamp_min(1)),
            "merged_miss_rate": float(
                merged_miss.sum() / valid_plane.sum().clamp_min(1)
            ),
            "merged_background_error_rate": float(
                merged_background_error.sum() / background.sum().clamp_min(1)
            ),
            "suppressed_pixels": int(suppressed.sum().item()),
        },
        "predicted": predicted.detach().cpu().numpy().astype(np.int16),
        "merged": merged.detach().cpu().numpy().astype(np.int16),
        "gt_compact": gt_compact.detach().cpu().numpy().astype(np.int16),
        "merge_groups": merge_groups,
    }


def render_case(case, path, num_queries):
    figure, axes = plt.subplots(2, 5, figsize=(22, 9))
    for row_index, view_name in enumerate(("view1", "view2")):
        view = case[view_name]
        rgb = case[f"rgb{row_index + 1}"]
        gt = view["gt_compact"]
        raw = view["predicted"]
        merged = view["merged"]
        target_count = view["metrics"]["gt_plane_count"]

        raw_error = np.zeros((*raw.shape, 3), dtype=np.float32)
        raw_error[(gt < target_count) & (raw < num_queries)] = np.asarray([0.0, 0.7, 0.2])
        raw_error[(gt < target_count) & (raw < num_queries) & (raw != gt)] = np.asarray(
            [1.0, 0.0, 1.0]
        )
        raw_error[(gt < target_count) & (raw == num_queries)] = np.asarray([1.0, 0.8, 0.0])

        merged_error = np.zeros((*raw.shape, 3), dtype=np.float32)
        merged_error[(gt < target_count) & (merged < target_count)] = np.asarray(
            [0.0, 0.7, 0.2]
        )
        merged_error[(gt < target_count) & (merged < target_count) & (merged != gt)] = np.asarray(
            [1.0, 0.0, 1.0]
        )
        merged_error[(gt < target_count) & (merged == target_count)] = np.asarray(
            [1.0, 0.8, 0.0]
        )

        images = (
            rgb,
            colorize_class_map(gt, num_queries),
            colorize_class_map(raw, num_queries),
            colorize_class_map(merged, num_queries),
            merged_error,
        )
        titles = (
            "RGB",
            "GT compact",
            "Raw Kmax query argmax",
            "Dynamic-K oracle merge",
            "Merged errors",
        )
        for column, (image, title) in enumerate(zip(images, titles)):
            axis = axes[row_index, column]
            axis.imshow(image)
            axis.axis("off")
            if column == 0:
                m = view["metrics"]
                axis.set_title(
                    f"{view_name} {title}\n"
                    f"raw={m['raw_mean_iou']:.3f} merged={m['merged_mean_iou']:.3f} "
                    f"split={m['raw_split_gt_count']} K={m['final_k_pred']}"
                )
            else:
                axis.set_title(title)

    row = case["metrics"]
    figure.suptitle(
        f"cache_idx={row['sample_idx']} scene={row.get('scene_name', '')} "
        f"raw={row['raw_mean_iou']:.3f} merged={row['merged_mean_iou']:.3f} "
        f"delta={row['iou_delta']:.3f} rawK={row['raw_active_query_count']:.2f} "
        f"finalK={row['final_k_pred']:.2f}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = safe_load_torch(args.checkpoint)
    cache = safe_load_torch(args.feature_cache_path)
    cache_config = cache.get("config", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head, config = build_multiscale_head(checkpoint, cache_config, device)
    samples = cache[args.split]
    metadata = load_cache_metadata(cache_config, args.split, len(samples))
    num_queries = int(config.get("num_queries", 8))

    rows = []
    cases = {}
    for start in range(0, len(samples), args.batch_size):
        items = samples[start : start + args.batch_size]
        pair_batch = len(items)
        features1 = {
            stage: torch.cat([item["features1"][stage] for item in items], dim=0)
            .to(device=device, dtype=torch.float32)
            for stage in STAGE_NAMES
        }
        features2 = {
            stage: torch.cat([item["features2"][stage] for item in items], dim=0)
            .to(device=device, dtype=torch.float32)
            for stage in STAGE_NAMES
        }
        features = {
            stage: torch.cat((features1[stage], features2[stage]), dim=0)
            for stage in STAGE_NAMES
        }
        rgb = torch.cat(
            (
                torch.cat([item["rgb1"] for item in items], dim=0),
                torch.cat([item["rgb2"] for item in items], dim=0),
            ),
            dim=0,
        ).to(device=device, dtype=torch.float32)
        output = head(features, rgb)

        for local_index, item in enumerate(items):
            sample_idx = start + local_index
            view1 = analyze_view(
                output_slice(output, local_index),
                item["gt_plane1"][0].to(device),
                config,
                args,
            )
            view2 = analyze_view(
                output_slice(output, pair_batch + local_index),
                item["gt_plane2"][0].to(device),
                config,
                args,
            )
            row = {"sample_idx": sample_idx, **metadata[sample_idx]}
            for metric_name in (
                "raw_mean_iou",
                "merged_mean_iou",
                "iou_delta",
                "raw_active_query_count",
                "final_k_pred",
                "raw_split_gt_count",
                "raw_redundant_query_count",
                "raw_max_split_parts",
                "raw_leakage_rate",
                "merged_leakage_rate",
                "raw_miss_rate",
                "merged_miss_rate",
                "merged_background_error_rate",
            ):
                row[f"view1_{metric_name}"] = view1["metrics"][metric_name]
                row[f"view2_{metric_name}"] = view2["metrics"][metric_name]
                row[metric_name] = 0.5 * (
                    float(view1["metrics"][metric_name])
                    + float(view2["metrics"][metric_name])
                )
            row["gt_plane_count"] = 0.5 * (
                view1["metrics"]["gt_plane_count"] + view2["metrics"]["gt_plane_count"]
            )
            rows.append(row)
            cases[sample_idx] = {
                "metrics": row,
                "rgb1": tensor_image_to_numpy(item["rgb1"][0]),
                "rgb2": tensor_image_to_numpy(item["rgb2"][0]),
                "view1": view1,
                "view2": view2,
            }

        processed = min(start + pair_batch, len(samples))
        if start == 0 or processed == len(samples) or processed % 32 == 0:
            print(f"[Dynamic query postprocess] {processed}/{len(samples)}", flush=True)

    rows_by_delta = sorted(rows, key=lambda row: row["iou_delta"], reverse=True)
    rows_by_bad = sorted(rows, key=lambda row: row["merged_mean_iou"])
    summary = {
        "checkpoint": args.checkpoint,
        "feature_cache_path": args.feature_cache_path,
        "split": args.split,
        "pair_count": len(rows),
        "raw_mean_iou": float(np.mean([row["raw_mean_iou"] for row in rows])),
        "merged_mean_iou": float(np.mean([row["merged_mean_iou"] for row in rows])),
        "mean_iou_delta": float(np.mean([row["iou_delta"] for row in rows])),
        "raw_redundant_query_count": float(
            np.mean([row["raw_redundant_query_count"] for row in rows])
        ),
        "raw_split_gt_count": float(np.mean([row["raw_split_gt_count"] for row in rows])),
        "raw_active_query_count": float(
            np.mean([row["raw_active_query_count"] for row in rows])
        ),
        "final_k_pred": float(np.mean([row["final_k_pred"] for row in rows])),
        "raw_leakage_rate": float(np.mean([row["raw_leakage_rate"] for row in rows])),
        "merged_leakage_rate": float(
            np.mean([row["merged_leakage_rate"] for row in rows])
        ),
        "improved_pairs": int(sum(row["iou_delta"] > 1e-6 for row in rows)),
        "degraded_pairs": int(sum(row["iou_delta"] < -1e-6 for row in rows)),
        "same_pairs": int(sum(abs(row["iou_delta"]) <= 1e-6 for row in rows)),
        "top_improved": rows_by_delta[:20],
        "worst_after_merge": rows_by_bad[:20],
    }
    (output_dir / "dynamic_query_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(output_dir / "dynamic_query_metrics.csv", rows)

    focus = set()
    if args.focus_indices:
        focus.update(
            int(value.strip())
            for value in args.focus_indices.split(",")
            if value.strip()
        )
    focus.update(int(row["sample_idx"]) for row in rows_by_delta[: args.top_k_visuals])
    focus.update(int(row["sample_idx"]) for row in rows_by_bad[: min(args.top_k_visuals, 8)])
    for sample_idx in sorted(focus):
        if sample_idx in cases:
            render_case(
                cases[sample_idx],
                output_dir / "visuals" / f"cache_{sample_idx:06d}_dynamic_query.png",
                num_queries,
            )

    report = [
        "# Dynamic Query Postprocess Ablation",
        "",
        "This is an oracle merge ablation: query groups are merged by their GT-majority attribution.",
        "It estimates whether dynamic K can fix current query splitting before implementing a no-GT rule.",
        "",
        f"- pair_count: {summary['pair_count']}",
        f"- raw_mean_iou: {summary['raw_mean_iou']:.6f}",
        f"- merged_mean_iou: {summary['merged_mean_iou']:.6f}",
        f"- mean_iou_delta: {summary['mean_iou_delta']:.6f}",
        f"- raw_active_query_count: {summary['raw_active_query_count']:.4f}",
        f"- final_k_pred: {summary['final_k_pred']:.4f}",
        f"- raw_redundant_query_count: {summary['raw_redundant_query_count']:.4f}",
        f"- raw_split_gt_count: {summary['raw_split_gt_count']:.4f}",
        f"- raw_leakage_rate: {summary['raw_leakage_rate']:.6f}",
        f"- merged_leakage_rate: {summary['merged_leakage_rate']:.6f}",
        f"- improved/degraded/same pairs: {summary['improved_pairs']}/"
        f"{summary['degraded_pairs']}/{summary['same_pairs']}",
        "",
        "## Top Improved Cases",
        "",
        "| sample_idx | raw_iou | merged_iou | delta | rawK | finalK | raw_split |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_by_delta[:20]:
        report.append(
            f"| {row['sample_idx']} | {row['raw_mean_iou']:.3f} "
            f"| {row['merged_mean_iou']:.3f} | {row['iou_delta']:.3f} "
            f"| {row['raw_active_query_count']:.2f} | {row['final_k_pred']:.2f} "
            f"| {row['raw_split_gt_count']:.2f} |"
        )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
