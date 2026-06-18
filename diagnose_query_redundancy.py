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
        "Diagnose Stage1 query redundancy and plane over-splitting"
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
        "--metrics_path",
        default="local_outputs/vis_multiscale_balanced512/all_metrics.json",
    )
    parser.add_argument(
        "--output_dir",
        default="local_outputs/query_redundancy_diagnostics_v1",
    )
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--min_query_pixels", type=int, default=16)
    parser.add_argument("--dominance_threshold", type=float, default=0.25)
    parser.add_argument("--focus_indices", default="")
    parser.add_argument("--top_k_visuals", type=int, default=16)
    return parser.parse_args()


def tensor_image_to_numpy(image_tensor):
    image = image_tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def safe_load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_baseline_metrics(path):
    path = Path(path)
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["sample_idx"]): row for row in rows}


def build_view_analysis(output, labels, config, min_query_pixels, dominance_threshold):
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
    if masks[0]:
        targets = torch.stack(masks[0])
    else:
        targets = output["mask_logits"].new_zeros((0, *target_hw))
    query_ids, target_ids = match_queries(output["mask_logits"], targets, match_args)
    gt_query = class_target_from_matches(
        resized_labels[0], targets, query_ids, target_ids, num_queries
    )

    class_logits = torch.cat(
        (output["mask_logits"], output["background_logits"]), dim=0
    )
    predicted = class_logits.argmax(dim=0)
    plane_count = int(targets.shape[0])
    background_id = plane_count

    confusion = torch.zeros(
        (num_queries + 1, plane_count + 1), dtype=torch.long, device=predicted.device
    )
    gt_compact = torch.full_like(gt_query, background_id)
    for compact_id, query_id in enumerate(query_ids):
        gt_compact[gt_query == int(query_id)] = compact_id
    pred_compact = predicted.clamp(max=num_queries)
    for pred_id in range(num_queries + 1):
        pred_mask = pred_compact == pred_id
        if not pred_mask.any():
            continue
        for gt_id in range(plane_count + 1):
            confusion[pred_id, gt_id] = (pred_mask & (gt_compact == gt_id)).sum()

    query_rows = []
    gt_to_queries = {gt_id: [] for gt_id in range(plane_count)}
    for query_id in range(num_queries):
        row = confusion[query_id]
        pixels = int(row.sum().item())
        if pixels < min_query_pixels:
            continue
        main_gt = int(row.argmax().item())
        main_pixels = int(row[main_gt].item())
        purity = main_pixels / max(pixels, 1)
        plane_fraction = float(row[:plane_count].sum().item() / max(pixels, 1))
        query_rows.append(
            {
                "query_id": query_id,
                "pixels": pixels,
                "main_gt": main_gt,
                "main_pixels": main_pixels,
                "purity": purity,
                "plane_fraction": plane_fraction,
            }
        )
        if main_gt < plane_count and purity >= dominance_threshold:
            gt_to_queries[main_gt].append(query_id)

    gt_rows = []
    split_gt_count = 0
    max_split_parts = 0
    redundant_query_count = 0
    for gt_id in range(plane_count):
        gt_pixels = int(confusion[:num_queries, gt_id].sum().item())
        query_hits = []
        for query_id in range(num_queries):
            hit = int(confusion[query_id, gt_id].item())
            if hit >= min_query_pixels:
                query_hits.append((query_id, hit))
        query_hits.sort(key=lambda item: item[1], reverse=True)
        dominant_queries = gt_to_queries[gt_id]
        split_parts = len(dominant_queries)
        if split_parts > 1:
            split_gt_count += 1
            redundant_query_count += split_parts - 1
        max_split_parts = max(max_split_parts, split_parts)
        gt_rows.append(
            {
                "gt_id": gt_id,
                "gt_pixels_captured": gt_pixels,
                "dominant_query_count": split_parts,
                "dominant_queries": dominant_queries,
                "query_hits": query_hits[:5],
            }
        )

    wrong_plane = (
        (gt_query < num_queries)
        & (predicted < num_queries)
        & (predicted != gt_query)
    )
    valid_plane = gt_query < num_queries
    leakage_rate = float(wrong_plane.sum() / valid_plane.sum().clamp_min(1))

    active_queries = len(query_rows)
    oversegmentation_ratio = redundant_query_count / max(plane_count, 1)
    split_gt_rate = split_gt_count / max(plane_count, 1)

    return {
        "metrics": {
            "gt_plane_count": plane_count,
            "active_query_count": active_queries,
            "split_gt_count": split_gt_count,
            "split_gt_rate": split_gt_rate,
            "max_split_parts": max_split_parts,
            "redundant_query_count": redundant_query_count,
            "oversegmentation_ratio": oversegmentation_ratio,
            "leakage_rate": leakage_rate,
        },
        "predicted": predicted.detach().cpu().numpy().astype(np.int16),
        "gt_query": gt_query.detach().cpu().numpy().astype(np.int16),
        "gt_compact": gt_compact.detach().cpu().numpy().astype(np.int16),
        "confusion": confusion.detach().cpu().numpy().astype(np.int64),
        "query_rows": query_rows,
        "gt_rows": gt_rows,
    }


def render_case(case, path, num_queries):
    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    for row_index, view_name in enumerate(("view1", "view2")):
        view = case[view_name]
        rgb = case[f"rgb{row_index + 1}"]
        predicted = view["predicted"]
        gt_query = view["gt_query"]
        wrong = (gt_query < num_queries) & (predicted < num_queries) & (
            gt_query != predicted
        )
        split_heat = np.zeros((*predicted.shape, 3), dtype=np.float32)
        split_heat[predicted < num_queries] = QUERY_COLORS[
            predicted[predicted < num_queries] % len(QUERY_COLORS)
        ]
        split_heat[wrong] = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)

        confusion = view["confusion"][:num_queries, : max(view["metrics"]["gt_plane_count"], 1)]
        if confusion.size:
            conf = confusion.astype(np.float32)
            col_sum = np.maximum(conf.sum(axis=0, keepdims=True), 1.0)
            conf = conf / col_sum
        else:
            conf = np.zeros((num_queries, 1), dtype=np.float32)

        images = (
            rgb,
            colorize_class_map(gt_query, num_queries),
            colorize_class_map(predicted, num_queries),
            split_heat,
        )
        titles = (
            "RGB",
            "GT matched query colors",
            "Pred query argmax",
            "Wrong plane pixels magenta",
        )
        for col, (image, title) in enumerate(zip(images, titles)):
            axis = axes[row_index, col]
            axis.imshow(image)
            axis.axis("off")
            if col == 0:
                m = view["metrics"]
                axis.set_title(
                    f"{view_name} {title}\n"
                    f"gt={m['gt_plane_count']} active={m['active_query_count']} "
                    f"split_gt={m['split_gt_count']} max_parts={m['max_split_parts']} "
                    f"leak={m['leakage_rate']:.3f}"
                )
            else:
                axis.set_title(title)

        inset = axes[row_index, 3].inset_axes([0.58, 0.05, 0.38, 0.38])
        inset.imshow(conf, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        inset.set_title("query/GT share", fontsize=8)
        inset.set_xlabel("GT", fontsize=7)
        inset.set_ylabel("Q", fontsize=7)
        inset.tick_params(labelsize=6)

    metrics = case["metrics"]
    figure.suptitle(
        f"cache_idx={metrics['sample_idx']} scene={metrics.get('scene_name', '')} "
        f"IoU={metrics.get('mean_iou', 0.0):.3f} leak={metrics.get('leakage', 0.0):.3f} "
        f"redundant={metrics['redundant_query_count']:.2f} "
        f"split_gt={metrics['split_gt_count']:.2f}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
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
    baseline_metrics = load_baseline_metrics(args.metrics_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head, config = build_multiscale_head(checkpoint, cache_config, device)
    samples = cache[args.split]
    metadata = load_cache_metadata(cache_config, args.split, len(samples))
    num_queries = int(config.get("num_queries", 8))

    rows = []
    cases = []
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
            view1 = build_view_analysis(
                output_slice(output, local_index),
                item["gt_plane1"][0].to(device),
                config,
                args.min_query_pixels,
                args.dominance_threshold,
            )
            view2 = build_view_analysis(
                output_slice(output, pair_batch + local_index),
                item["gt_plane2"][0].to(device),
                config,
                args.min_query_pixels,
                args.dominance_threshold,
            )
            base = baseline_metrics.get(sample_idx, {})
            row = {
                "sample_idx": sample_idx,
                **metadata[sample_idx],
                "mean_iou": base.get("mean_iou", ""),
                "leakage": base.get("leakage", ""),
                "view_gap": base.get("view_gap", ""),
            }
            for prefix, view in (("view1", view1), ("view2", view2)):
                for key, value in view["metrics"].items():
                    row[f"{prefix}_{key}"] = value
            row["redundant_query_count"] = 0.5 * (
                view1["metrics"]["redundant_query_count"]
                + view2["metrics"]["redundant_query_count"]
            )
            row["split_gt_count"] = 0.5 * (
                view1["metrics"]["split_gt_count"] + view2["metrics"]["split_gt_count"]
            )
            row["max_split_parts"] = max(
                view1["metrics"]["max_split_parts"], view2["metrics"]["max_split_parts"]
            )
            rows.append(row)
            cases.append(
                {
                    "metrics": row,
                    "rgb1": tensor_image_to_numpy(item["rgb1"][0]),
                    "rgb2": tensor_image_to_numpy(item["rgb2"][0]),
                    "view1": view1,
                    "view2": view2,
                }
            )

        processed = min(start + pair_batch, len(samples))
        if start == 0 or processed == len(samples) or processed % 32 == 0:
            print(f"[Query redundancy] {processed}/{len(samples)}", flush=True)

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            float(row.get("redundant_query_count", 0.0)),
            float(row.get("leakage", 0.0) or 0.0),
        ),
        reverse=True,
    )
    summary = {
        "checkpoint": args.checkpoint,
        "feature_cache_path": args.feature_cache_path,
        "split": args.split,
        "pair_count": len(rows),
        "mean_redundant_query_count": float(
            np.mean([row["redundant_query_count"] for row in rows])
        ),
        "mean_split_gt_count": float(np.mean([row["split_gt_count"] for row in rows])),
        "pairs_with_split": int(sum(row["split_gt_count"] > 0 for row in rows)),
        "pairs_with_max_split_ge3": int(sum(row["max_split_parts"] >= 3 for row in rows)),
        "top_redundant": rows_sorted[:20],
    }
    (output_dir / "query_redundancy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(output_dir / "query_redundancy.csv", rows)

    focus = set()
    if args.focus_indices:
        focus.update(
            int(value.strip())
            for value in args.focus_indices.split(",")
            if value.strip()
        )
    focus.update(int(row["sample_idx"]) for row in rows_sorted[: args.top_k_visuals])
    case_by_index = {int(case["metrics"]["sample_idx"]): case for case in cases}
    for sample_idx in sorted(focus):
        if sample_idx not in case_by_index:
            continue
        render_case(
            case_by_index[sample_idx],
            output_dir / "visuals" / f"cache_{sample_idx:06d}_query_redundancy.png",
            num_queries,
        )

    report = [
        "# Query Redundancy Diagnostic",
        "",
        f"- pair_count: {summary['pair_count']}",
        f"- mean_redundant_query_count: {summary['mean_redundant_query_count']:.4f}",
        f"- mean_split_gt_count: {summary['mean_split_gt_count']:.4f}",
        f"- pairs_with_split: {summary['pairs_with_split']}",
        f"- pairs_with_max_split_ge3: {summary['pairs_with_max_split_ge3']}",
        "",
        "## Top Redundant Cases",
        "",
        "| sample_idx | mean_iou | leakage | redundant_query_count | split_gt_count | max_split_parts |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_sorted[:20]:
        report.append(
            f"| {row['sample_idx']} | {float(row.get('mean_iou') or 0.0):.3f} "
            f"| {float(row.get('leakage') or 0.0):.3f} "
            f"| {row['redundant_query_count']:.2f} "
            f"| {row['split_gt_count']:.2f} | {row['max_split_parts']} |"
        )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
