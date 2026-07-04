import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from visualize_predictions import (
    STAGE_NAMES,
    analyze_view,
    build_multiscale_head,
    load_cache_metadata,
    output_slice,
    render_case,
    safe_load_checkpoint,
    tensor_image_to_numpy,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Build two-axis Stage1 data quality splits for manual review and training"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_cache_path", default="")
    parser.add_argument("--feature_cache_glob", default="")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--review_samples_per_bucket", type=int, default=20)
    parser.add_argument("--device", default="cuda")

    # Reliability thresholds. These are intentionally conservative: low IoU
    # alone makes a sample hard, not unreliable.
    parser.add_argument("--weak_leakage", type=float, default=0.16)
    parser.add_argument("--reject_leakage", type=float, default=0.28)
    parser.add_argument("--weak_background_error", type=float, default=0.03)
    parser.add_argument("--reject_background_error", type=float, default=0.08)
    parser.add_argument("--weak_view_gap", type=float, default=0.35)
    parser.add_argument("--reject_view_gap", type=float, default=0.55)
    parser.add_argument("--weak_count_error", type=float, default=2.0)
    parser.add_argument("--reject_count_error", type=float, default=3.5)
    parser.add_argument("--weak_tiny_plane_ratio", type=float, default=0.35)
    parser.add_argument("--reject_tiny_plane_ratio", type=float, default=0.55)

    # Difficulty thresholds.
    parser.add_argument("--hard_iou", type=float, default=0.75)
    parser.add_argument("--hard_leakage", type=float, default=0.12)
    parser.add_argument("--hard_count_error", type=float, default=1.5)
    parser.add_argument("--hard_view_gap", type=float, default=0.20)
    parser.add_argument("--hard_gt_planes", type=float, default=5.0)
    parser.add_argument("--hard_tiny_plane_ratio", type=float, default=0.20)
    parser.add_argument("--hard_boundary_complexity", type=float, default=0.20)

    # Fixed first-pass training policy.
    parser.add_argument("--clean_easy_sampling_weight", type=float, default=1.0)
    parser.add_argument("--clean_hard_sampling_weight", type=float, default=2.0)
    parser.add_argument("--weak_sampling_weight", type=float, default=0.5)
    parser.add_argument("--weak_gt_loss_weight", type=float, default=0.3)
    parser.add_argument("--reject_sampling_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260704)
    return parser.parse_args()


def cache_paths_from_args(args):
    if args.feature_cache_glob:
        paths = [Path(path) for path in sorted(glob.glob(args.feature_cache_glob))]
    elif args.feature_cache_path:
        paths = [Path(args.feature_cache_path)]
    else:
        paths = []
    if not paths:
        raise FileNotFoundError("Provide --feature_cache_path or --feature_cache_glob")
    return paths


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_index_file(path, rows, key):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(int(row[key])) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def cat_feature_batch(items, key, device):
    return {
        stage: torch.cat([item[key][stage] for item in items], dim=0).to(
            device=device,
            dtype=torch.float32,
        )
        for stage in STAGE_NAMES
    }


def cat_tensor_batch(items, key, device):
    return torch.cat([item[key] for item in items], dim=0).to(
        device=device,
        dtype=torch.float32,
    )


def mask_stats(label, min_plane_pixels=4):
    label_np = label.detach().cpu().numpy()
    plane_ids = [int(value) for value in np.unique(label_np) if value > 0 and value != 255]
    total_pixels = float(label_np.size)
    if not plane_ids:
        return {
            "gt_plane_count": 0,
            "largest_plane_ratio": 0.0,
            "very_small_plane_ratio": 0.0,
            "boundary_complexity": 0.0,
        }
    areas = []
    boundary_pixels = 0
    for plane_id in plane_ids:
        mask = label_np == plane_id
        area = int(mask.sum())
        areas.append(area)
        vertical = np.zeros_like(mask, dtype=bool)
        horizontal = np.zeros_like(mask, dtype=bool)
        vertical[:, 1:] = mask[:, 1:] != mask[:, :-1]
        horizontal[1:, :] = mask[1:, :] != mask[:-1, :]
        boundary_pixels += int((vertical | horizontal).sum())
    plane_area = max(sum(areas), 1)
    very_small = sum(area < max(min_plane_pixels * 4, 0.01 * total_pixels) for area in areas)
    return {
        "gt_plane_count": len(plane_ids),
        "largest_plane_ratio": float(max(areas) / max(plane_area, 1)),
        "very_small_plane_ratio": float(very_small / max(len(areas), 1)),
        "boundary_complexity": float(boundary_pixels / max(plane_area, 1)),
    }


def classify_reliability(row, args):
    weak_reasons = []
    reject_reasons = []
    if row["leakage"] >= args.reject_leakage:
        reject_reasons.append("leakage>=reject")
    elif row["leakage"] >= args.weak_leakage:
        weak_reasons.append("leakage>=weak")
    if row["background_error"] >= args.reject_background_error:
        reject_reasons.append("background_error>=reject")
    elif row["background_error"] >= args.weak_background_error:
        weak_reasons.append("background_error>=weak")
    if row["view_gap"] >= args.reject_view_gap:
        reject_reasons.append("view_gap>=reject")
    elif row["view_gap"] >= args.weak_view_gap:
        weak_reasons.append("view_gap>=weak")
    if row["plane_count_abs_error"] >= args.reject_count_error:
        reject_reasons.append("plane_count_error>=reject")
    elif row["plane_count_abs_error"] >= args.weak_count_error:
        weak_reasons.append("plane_count_error>=weak")
    if row["very_small_plane_ratio"] >= args.reject_tiny_plane_ratio:
        reject_reasons.append("tiny_plane_ratio>=reject")
    elif row["very_small_plane_ratio"] >= args.weak_tiny_plane_ratio:
        weak_reasons.append("tiny_plane_ratio>=weak")

    if reject_reasons:
        return "reject", reject_reasons
    if weak_reasons:
        return "weak", weak_reasons
    return "clean", ["passed_reliability_checks"]


def classify_difficulty(row, args):
    reasons = []
    if row["gt_plane_count"] >= args.hard_gt_planes:
        reasons.append("gt_plane_count>=hard")
    if row["very_small_plane_ratio"] >= args.hard_tiny_plane_ratio:
        reasons.append("tiny_plane_ratio>=hard")
    if row["boundary_complexity"] >= args.hard_boundary_complexity:
        reasons.append("boundary_complexity>=hard")
    if row["mean_iou"] < args.hard_iou:
        reasons.append("mean_iou<hard")
    if row["leakage"] > args.hard_leakage:
        reasons.append("leakage>hard")
    if row["plane_count_abs_error"] >= args.hard_count_error:
        reasons.append("plane_count_error>=hard")
    if row["view_gap"] > args.hard_view_gap:
        reasons.append("view_gap>hard")
    if len(reasons) >= 2:
        return "hard", reasons
    return "easy", reasons or ["passed_difficulty_checks"]


def policy_for(reliability, difficulty, args):
    if reliability == "reject":
        return args.reject_sampling_weight, 0.0
    if reliability == "weak":
        return args.weak_sampling_weight, args.weak_gt_loss_weight
    if difficulty == "hard":
        return args.clean_hard_sampling_weight, 1.0
    return args.clean_easy_sampling_weight, 1.0


@torch.no_grad()
def score_cache(head, cache, split, cache_name, global_start, args, device):
    samples = cache[split]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    metadata = load_cache_metadata(cache.get("config", {}), split, len(samples))
    rows = []
    cases = []
    use_geometry = bool(head._quality_config.get("use_geometry", False))

    for start in range(0, len(samples), args.batch_size):
        items = samples[start : start + args.batch_size]
        pair_batch = len(items)
        features1 = cat_feature_batch(items, "features1", device)
        features2 = cat_feature_batch(items, "features2", device)
        features = {
            stage: torch.cat((features1[stage], features2[stage]), dim=0)
            for stage in STAGE_NAMES
        }
        rgb1 = cat_tensor_batch(items, "rgb1", device)
        rgb2 = cat_tensor_batch(items, "rgb2", device)
        geometry = None
        if use_geometry:
            if "geometry1" not in items[0] or "geometry2" not in items[0]:
                raise RuntimeError("Checkpoint requires geometry cache fields.")
            geometry1 = cat_tensor_batch(items, "geometry1", device)
            geometry2 = cat_tensor_batch(items, "geometry2", device)
            geometry = torch.cat((geometry1, geometry2), dim=0)

        output = head(features, torch.cat((rgb1, rgb2), dim=0), geometry=geometry)
        for local_index, item in enumerate(items):
            local_cache_idx = start + local_index
            global_cache_idx = global_start + local_cache_idx
            label1 = item["gt_plane1"][0].to(device)
            label2 = item["gt_plane2"][0].to(device)
            view1 = analyze_view(
                output_slice(output, local_index),
                label1,
                head._quality_config,
            )
            view2 = analyze_view(
                output_slice(output, pair_batch + local_index),
                label2,
                head._quality_config,
            )
            metrics1 = view1["metrics"]
            metrics2 = view2["metrics"]
            stats1 = mask_stats(item["gt_plane1"][0])
            stats2 = mask_stats(item["gt_plane2"][0])
            gt_planes = 0.5 * (stats1["gt_plane_count"] + stats2["gt_plane_count"])
            pred_regions = 0.5 * (
                metrics1["pred_region_count"] + metrics2["pred_region_count"]
            )
            row = {
                "cache_name": cache_name,
                "local_index": local_cache_idx,
                "global_index": global_cache_idx,
                "sample_idx": global_cache_idx,
                **metadata[local_cache_idx],
                "mean_iou": 0.5 * (metrics1["mean_iou"] + metrics2["mean_iou"]),
                "view1_mean_iou": metrics1["mean_iou"],
                "view2_mean_iou": metrics2["mean_iou"],
                "view_gap": abs(metrics1["mean_iou"] - metrics2["mean_iou"]),
                "leakage": 0.5
                * (metrics1["leakage_rate"] + metrics2["leakage_rate"]),
                "view1_leakage": metrics1["leakage_rate"],
                "view2_leakage": metrics2["leakage_rate"],
                "plane_miss": 0.5
                * (metrics1["plane_miss_rate"] + metrics2["plane_miss_rate"]),
                "background_error": 0.5
                * (
                    metrics1["background_error_rate"]
                    + metrics2["background_error_rate"]
                ),
                "gt_plane_count": gt_planes,
                "view1_gt_plane_count": stats1["gt_plane_count"],
                "view2_gt_plane_count": stats2["gt_plane_count"],
                "pred_region_count": pred_regions,
                "view1_pred_region_count": metrics1["pred_region_count"],
                "view2_pred_region_count": metrics2["pred_region_count"],
                "plane_count_abs_error": abs(pred_regions - gt_planes),
                "largest_plane_ratio": 0.5
                * (stats1["largest_plane_ratio"] + stats2["largest_plane_ratio"]),
                "very_small_plane_ratio": 0.5
                * (stats1["very_small_plane_ratio"] + stats2["very_small_plane_ratio"]),
                "boundary_complexity": 0.5
                * (stats1["boundary_complexity"] + stats2["boundary_complexity"]),
            }
            reliability, reliability_reasons = classify_reliability(row, args)
            difficulty, difficulty_reasons = classify_difficulty(row, args)
            sampling_weight, gt_loss_weight = policy_for(reliability, difficulty, args)
            row.update(
                {
                    "reliability": reliability,
                    "difficulty": difficulty,
                    "bucket": f"{reliability}_{difficulty}"
                    if reliability != "reject"
                    else "reject",
                    "sampling_weight": sampling_weight,
                    "gt_loss_weight": gt_loss_weight,
                    "reasons": ";".join(reliability_reasons + difficulty_reasons),
                }
            )
            rows.append(row)
            cases.append(
                {
                    "metrics": {
                        **row,
                        "view1_plane_miss": metrics1["plane_miss_rate"],
                        "view2_plane_miss": metrics2["plane_miss_rate"],
                    },
                    "rgb1": tensor_image_to_numpy(item["rgb1"][0]),
                    "rgb2": tensor_image_to_numpy(item["rgb2"][0]),
                    "view1": view1,
                    "view2": view2,
                }
            )

        processed = min(start + pair_batch, len(samples))
        if processed == len(samples) or processed % 64 == 0:
            print(f"[score] {cache_name} {processed}/{len(samples)}", flush=True)
    return rows, cases


def summarize(rows):
    numeric_keys = [
        "mean_iou",
        "leakage",
        "plane_count_abs_error",
        "view_gap",
        "gt_plane_count",
        "boundary_complexity",
        "very_small_plane_ratio",
    ]
    result = {"count": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        result[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
        result[f"{key}_median"] = float(np.median(values)) if values else 0.0
    return result


def save_distribution_plots(rows, output_dir):
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets = [row["bucket"] for row in rows]
    bucket_names = ["clean_easy", "clean_hard", "weak_easy", "weak_hard", "reject"]
    counts = [buckets.count(name) for name in bucket_names]
    plt.figure(figsize=(10, 5))
    plt.bar(bucket_names, counts)
    plt.xticks(rotation=20)
    plt.title("Quality and difficulty distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "quality_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(
        [row["gt_plane_count"] for row in rows],
        [row["mean_iou"] for row in rows],
        s=10,
        alpha=0.7,
    )
    plt.xlabel("GT plane count")
    plt.ylabel("Old-best mean IoU")
    plt.title("Plane count vs IoU")
    plt.tight_layout()
    plt.savefig(output_dir / "plane_count_vs_iou.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(
        [row["boundary_complexity"] for row in rows],
        [row["mean_iou"] for row in rows],
        s=10,
        alpha=0.7,
    )
    plt.xlabel("Boundary complexity")
    plt.ylabel("Old-best mean IoU")
    plt.title("Boundary complexity vs IoU")
    plt.tight_layout()
    plt.savefig(output_dir / "boundary_complexity_vs_iou.png", dpi=150)
    plt.close()


def save_review_cases(cases, output_dir, checkpoint_epoch, num_queries, split, count, seed):
    if count <= 0:
        return
    rng = np.random.default_rng(seed)
    by_bucket = defaultdict(list)
    for case in cases:
        by_bucket[case["metrics"]["bucket"]].append(case)
    for bucket, bucket_cases in by_bucket.items():
        indices = np.arange(len(bucket_cases))
        if len(indices) > count:
            indices = rng.choice(indices, size=count, replace=False)
        for rank, index in enumerate(indices, start=1):
            case = bucket_cases[int(index)]
            metrics = case["metrics"]
            safe_reason = str(metrics["reasons"]).replace(";", "_").replace(">", "gt").replace("<", "lt")
            path = (
                output_dir
                / "review"
                / bucket
                / f"{rank:02d}_g{int(metrics['global_index']):06d}_l{int(metrics['local_index']):04d}_{safe_reason[:80]}.png"
            )
            render_case(case, path, checkpoint_epoch, num_queries, split)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = cache_paths_from_args(args)

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    first_cache = torch.load(cache_paths[0], map_location="cpu", weights_only=False)
    checkpoint = safe_load_checkpoint(args.checkpoint, "cpu")
    head, config = build_multiscale_head(checkpoint, first_cache.get("config", {}), device)
    head._quality_config = config
    checkpoint_epoch = checkpoint.get("epoch", "unknown")
    num_queries = int(config.get("num_queries", 8))

    all_rows = []
    all_cases = []
    per_cache_rows = {}
    global_start = 0
    for cache_path in cache_paths:
        cache = (
            first_cache
            if cache_path == cache_paths[0]
            else torch.load(cache_path, map_location="cpu", weights_only=False)
        )
        cache_name = cache_path.stem
        rows, cases = score_cache(
            head,
            cache,
            args.split,
            cache_name,
            global_start,
            args,
            device,
        )
        all_rows.extend(rows)
        all_cases.extend(cases)
        per_cache_rows[cache_name] = rows
        global_start += len(cache[args.split])

    bucket_names = ["clean_easy", "clean_hard", "weak_easy", "weak_hard", "reject"]
    buckets = {name: [row for row in all_rows if row["bucket"] == name] for name in bucket_names}
    write_csv(output_dir / "all_samples.csv", all_rows)
    for name, rows in buckets.items():
        write_csv(output_dir / f"{name}.csv", rows)
        write_index_file(output_dir / f"{name}_global_indices.txt", rows, "global_index")

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for shard_id, (cache_name, rows) in enumerate(per_cache_rows.items()):
        samples = [
            {
                "local_index": int(row["local_index"]),
                "global_index": int(row["global_index"]),
                "scene_name": row.get("scene_name", ""),
                "reliability": row["reliability"],
                "difficulty": row["difficulty"],
                "bucket": row["bucket"],
                "sampling_weight": float(row["sampling_weight"]),
                "gt_loss_weight": float(row["gt_loss_weight"]),
                "metrics": {
                    "mean_iou": float(row["mean_iou"]),
                    "leakage": float(row["leakage"]),
                    "plane_count_abs_error": float(row["plane_count_abs_error"]),
                    "view_gap": float(row["view_gap"]),
                    "gt_plane_count": float(row["gt_plane_count"]),
                    "boundary_complexity": float(row["boundary_complexity"]),
                    "very_small_plane_ratio": float(row["very_small_plane_ratio"]),
                },
                "reasons": row["reasons"].split(";"),
            }
            for row in rows
        ]
        payload = {
            "shard_id": shard_id,
            "cache_name": cache_name,
            "samples": samples,
        }
        (shard_dir / f"shard_{shard_id:03d}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Default active training set excludes reject.
        active = [row for row in rows if row["sampling_weight"] > 0.0]
        write_index_file(
            output_dir / "per_cache" / f"{cache_name}_cache_indices.txt",
            active,
            "local_index",
        )
        for name in bucket_names:
            write_index_file(
                output_dir / "per_cache" / f"{cache_name}_{name}_cache_indices.txt",
                [row for row in rows if row["bucket"] == name],
                "local_index",
            )

    summary_rows = []
    for name in bucket_names:
        row = {"bucket": name, **summarize(buckets[name])}
        summary_rows.append(row)
    write_csv(output_dir / "summary.csv", summary_rows)

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint_epoch,
        "split": args.split,
        "cache_paths": [str(path) for path in cache_paths],
        "total": len(all_rows),
        "buckets": {name: summarize(rows) for name, rows in buckets.items()},
        "policy": {
            "clean_easy": {
                "sampling_weight": args.clean_easy_sampling_weight,
                "gt_loss_weight": 1.0,
            },
            "clean_hard": {
                "sampling_weight": args.clean_hard_sampling_weight,
                "gt_loss_weight": 1.0,
            },
            "weak_easy": {
                "sampling_weight": args.weak_sampling_weight,
                "gt_loss_weight": args.weak_gt_loss_weight,
            },
            "weak_hard": {
                "sampling_weight": args.weak_sampling_weight,
                "gt_loss_weight": args.weak_gt_loss_weight,
            },
            "reject": {
                "sampling_weight": args.reject_sampling_weight,
                "gt_loss_weight": 0.0,
            },
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_distribution_plots(all_rows, output_dir)
    save_review_cases(
        all_cases,
        output_dir,
        checkpoint_epoch,
        num_queries,
        args.split,
        args.review_samples_per_bucket,
        args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
