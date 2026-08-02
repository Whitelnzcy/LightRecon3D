import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize_predictions import (
    STAGE_NAMES,
    analyze_view,
    build_multiscale_head,
    flatten_cached_samples,
    load_cache_metadata,
    output_slice,
    safe_load_checkpoint,
    tensor_image_to_numpy,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Diagnose Stage1 plane-mask failure modes without changing training"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_cache_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_visuals", type=int, default=24)
    parser.add_argument("--bad_iou_threshold", type=float, default=0.80)
    parser.add_argument("--bad_leakage_threshold", type=float, default=0.12)
    parser.add_argument("--significant_overlap", type=float, default=0.08)
    parser.add_argument("--edge_percentile", type=float, default=85.0)
    return parser.parse_args()


def dilate(mask, radius=2):
    mask = mask.astype(bool)
    output = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(mask)
            y_src0 = max(0, -dy)
            y_src1 = mask.shape[0] - max(0, dy)
            x_src0 = max(0, -dx)
            x_src1 = mask.shape[1] - max(0, dx)
            y_dst0 = max(0, dy)
            y_dst1 = mask.shape[0] - max(0, -dy)
            x_dst0 = max(0, dx)
            x_dst1 = mask.shape[1] - max(0, -dx)
            shifted[y_dst0:y_dst1, x_dst0:x_dst1] = mask[y_src0:y_src1, x_src0:x_src1]
            output |= shifted
    return output


def boundary_map(labels):
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def rgb_edge_map(rgb, percentile):
    gray = rgb.mean(axis=-1)
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    edge = gx + gy
    threshold = np.percentile(edge, percentile)
    return edge >= threshold, edge


def geometry_edge_map(geometry, percentile):
    if geometry is None:
        return None, None
    geom = geometry.detach().float().cpu().numpy()
    if geom.ndim == 4:
        geom = geom[0]
    if geom.shape[0] < 9:
        return None, None
    edge = geom[8]
    threshold = np.percentile(edge, percentile)
    return edge >= threshold, edge


def safe_ratio(numerator, denominator):
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator


def confusion_stats(gt_query, predicted, num_queries, significant_overlap):
    valid = gt_query < num_queries
    confusion = np.zeros((num_queries, num_queries), dtype=np.int64)
    for gt_id in range(num_queries):
        gt_mask = valid & (gt_query == gt_id)
        if not gt_mask.any():
            continue
        for pred_id in range(num_queries):
            confusion[gt_id, pred_id] = int((gt_mask & (predicted == pred_id)).sum())

    gt_totals = confusion.sum(axis=1)
    pred_totals = confusion.sum(axis=0)

    split_count = 0
    split_score = 0.0
    for gt_id, total in enumerate(gt_totals):
        if total <= 0:
            continue
        active = confusion[gt_id] / max(float(total), 1.0) >= significant_overlap
        count = int(active.sum())
        if count > 1:
            split_count += 1
            split_score += count - 1

    merge_count = 0
    merge_score = 0.0
    for pred_id, total in enumerate(pred_totals):
        if total <= 0:
            continue
        active = confusion[:, pred_id] / max(float(total), 1.0) >= significant_overlap
        count = int(active.sum())
        if count > 1:
            merge_count += 1
            merge_score += count - 1

    return confusion, split_count, split_score, merge_count, merge_score


def classify_view(view, rgb, geometry, config, args):
    num_queries = int(config.get("num_queries", 8))
    predicted = view["predicted"]
    gt_query = view["gt_query"]
    error_codes = view["error_codes"]
    metrics = view["metrics"]

    valid = gt_query < num_queries
    wrong_plane = error_codes == 2
    plane_miss = error_codes == 3
    bg_to_plane = error_codes == 4
    error = wrong_plane | plane_miss | bg_to_plane

    confusion, split_count, split_score, merge_count, merge_score = confusion_stats(
        gt_query,
        predicted,
        num_queries,
        args.significant_overlap,
    )

    gt_boundary = dilate(boundary_map(gt_query), radius=2)
    rgb_edge, rgb_edge_strength = rgb_edge_map(rgb, args.edge_percentile)
    geom_edge, geom_edge_strength = geometry_edge_map(geometry, args.edge_percentile)

    error_pixels = int(error.sum())
    wrong_pixels = int(wrong_plane.sum())
    error_near_gt_boundary = safe_ratio((error & gt_boundary).sum(), error_pixels)
    error_on_rgb_edge = safe_ratio((error & rgb_edge).sum(), error_pixels)
    rgb_edge_not_gt_boundary = safe_ratio(
        (error & rgb_edge & ~gt_boundary).sum(), error_pixels
    )
    if geom_edge is not None:
        error_on_geom_edge = safe_ratio((error & geom_edge).sum(), error_pixels)
        geom_edge_not_gt_boundary = safe_ratio(
            (error & geom_edge & ~gt_boundary).sum(), error_pixels
        )
    else:
        error_on_geom_edge = 0.0
        geom_edge_not_gt_boundary = 0.0

    is_bad = (
        metrics["mean_iou"] < args.bad_iou_threshold
        or metrics["leakage_rate"] > args.bad_leakage_threshold
        or metrics["plane_miss_rate"] > 0.05
        or metrics["background_error_rate"] > 0.05
    )
    if not is_bad:
        failure_type = "ok"
    elif merge_score >= max(1.0, split_score + 0.5):
        failure_type = "merge"
    elif split_score >= max(1.0, merge_score + 0.5):
        failure_type = "split"
    elif rgb_edge_not_gt_boundary >= 0.35 and error_near_gt_boundary < 0.55:
        failure_type = "texture_light_boundary"
    elif geom_edge_not_gt_boundary >= 0.35 and error_near_gt_boundary < 0.55:
        failure_type = "geometry_conflict"
    elif error_near_gt_boundary >= 0.60:
        failure_type = "boundary_localization"
    elif metrics["plane_miss_rate"] > 0.05:
        failure_type = "missing_plane"
    elif metrics["background_error_rate"] > 0.05:
        failure_type = "background_leak"
    else:
        failure_type = "unclear"

    return {
        "failure_type": failure_type,
        "is_bad": bool(is_bad),
        "split_count": int(split_count),
        "split_score": float(split_score),
        "merge_count": int(merge_count),
        "merge_score": float(merge_score),
        "error_pixels": error_pixels,
        "wrong_plane_pixels": wrong_pixels,
        "error_near_gt_boundary": float(error_near_gt_boundary),
        "error_on_rgb_edge": float(error_on_rgb_edge),
        "rgb_edge_not_gt_boundary": float(rgb_edge_not_gt_boundary),
        "error_on_geom_edge": float(error_on_geom_edge),
        "geom_edge_not_gt_boundary": float(geom_edge_not_gt_boundary),
        "confusion": confusion.tolist(),
        "rgb_edge": rgb_edge,
        "rgb_edge_strength": rgb_edge_strength,
        "geom_edge": geom_edge,
        "geom_edge_strength": geom_edge_strength,
    }


def pair_failure_type(view1_type, view2_type):
    if view1_type == view2_type:
        return view1_type
    if view1_type == "ok":
        return view2_type
    if view2_type == "ok":
        return view1_type
    priority = [
        "merge",
        "split",
        "texture_light_boundary",
        "geometry_conflict",
        "boundary_localization",
        "missing_plane",
        "background_leak",
        "unclear",
    ]
    for item in priority:
        if item in (view1_type, view2_type):
            return item
    return "unclear"


@torch.no_grad()
def collect_diagnostics(head, samples, metadata, config, device, args):
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
        rgb1 = torch.cat([item["rgb1"] for item in items], dim=0).to(
            device=device, dtype=torch.float32
        )
        rgb2 = torch.cat([item["rgb2"] for item in items], dim=0).to(
            device=device, dtype=torch.float32
        )
        if bool(config.get("use_geometry", False)):
            geometry1 = torch.cat([item["geometry1"] for item in items], dim=0).to(
                device=device, dtype=torch.float32
            )
            geometry2 = torch.cat([item["geometry2"] for item in items], dim=0).to(
                device=device, dtype=torch.float32
            )
            geometry = torch.cat((geometry1, geometry2), dim=0)
        else:
            geometry1 = geometry2 = None
            geometry = None

        output = head(features, torch.cat((rgb1, rgb2), dim=0), geometry=geometry)

        for local_index, item in enumerate(items):
            sample_index = start + local_index
            view1 = analyze_view(
                output_slice(output, local_index), item["gt_plane1"][0].to(device), config
            )
            view2 = analyze_view(
                output_slice(output, pair_batch + local_index),
                item["gt_plane2"][0].to(device),
                config,
            )
            rgb1_np = tensor_image_to_numpy(item["rgb1"][0])
            rgb2_np = tensor_image_to_numpy(item["rgb2"][0])
            diag1 = classify_view(
                view1,
                rgb1_np,
                item.get("geometry1", None),
                config,
                args,
            )
            diag2 = classify_view(
                view2,
                rgb2_np,
                item.get("geometry2", None),
                config,
                args,
            )
            pair_type = pair_failure_type(diag1["failure_type"], diag2["failure_type"])
            row = {
                "sample_idx": sample_index,
                **metadata[sample_index],
                "pair_failure_type": pair_type,
                "view1_failure_type": diag1["failure_type"],
                "view2_failure_type": diag2["failure_type"],
                "mean_iou": 0.5
                * (view1["metrics"]["mean_iou"] + view2["metrics"]["mean_iou"]),
                "leakage": 0.5
                * (
                    view1["metrics"]["leakage_rate"]
                    + view2["metrics"]["leakage_rate"]
                ),
                "plane_miss": 0.5
                * (
                    view1["metrics"]["plane_miss_rate"]
                    + view2["metrics"]["plane_miss_rate"]
                ),
                "background_error": 0.5
                * (
                    view1["metrics"]["background_error_rate"]
                    + view2["metrics"]["background_error_rate"]
                ),
                "view1_iou": view1["metrics"]["mean_iou"],
                "view2_iou": view2["metrics"]["mean_iou"],
                "view1_gt_planes": view1["metrics"]["gt_plane_count"],
                "view2_gt_planes": view2["metrics"]["gt_plane_count"],
                "view1_pred_regions": view1["metrics"]["pred_region_count"],
                "view2_pred_regions": view2["metrics"]["pred_region_count"],
                "view1_merge_score": diag1["merge_score"],
                "view2_merge_score": diag2["merge_score"],
                "view1_split_score": diag1["split_score"],
                "view2_split_score": diag2["split_score"],
                "view1_error_near_gt_boundary": diag1["error_near_gt_boundary"],
                "view2_error_near_gt_boundary": diag2["error_near_gt_boundary"],
                "view1_rgb_edge_not_gt_boundary": diag1["rgb_edge_not_gt_boundary"],
                "view2_rgb_edge_not_gt_boundary": diag2["rgb_edge_not_gt_boundary"],
                "view1_geom_edge_not_gt_boundary": diag1["geom_edge_not_gt_boundary"],
                "view2_geom_edge_not_gt_boundary": diag2["geom_edge_not_gt_boundary"],
            }
            rows.append(row)
            cases.append(
                {
                    "row": row,
                    "rgb1": rgb1_np,
                    "rgb2": rgb2_np,
                    "view1": view1,
                    "view2": view2,
                    "diag1": diag1,
                    "diag2": diag2,
                }
            )

        processed = min(start + pair_batch, len(samples))
        if start == 0 or processed == len(samples) or processed % 32 == 0:
            print(f"[diagnose] {processed}/{len(samples)}", flush=True)
    return rows, cases


def save_rows(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "failure_modes.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output_dir / "failure_modes.csv"
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def save_summary(rows, output_dir):
    type_counts = Counter(row["pair_failure_type"] for row in rows)
    bad_rows = [row for row in rows if row["pair_failure_type"] != "ok"]
    summary = {
        "num_samples": len(rows),
        "mean_iou": float(np.mean([row["mean_iou"] for row in rows])),
        "mean_leakage": float(np.mean([row["leakage"] for row in rows])),
        "failure_type_counts": dict(type_counts),
        "bad_sample_count": len(bad_rows),
        "bad_sample_rate": safe_ratio(len(bad_rows), len(rows)),
        "mean_merge_score": float(
            np.mean(
                [
                    0.5 * (row["view1_merge_score"] + row["view2_merge_score"])
                    for row in rows
                ]
            )
        ),
        "mean_split_score": float(
            np.mean(
                [
                    0.5 * (row["view1_split_score"] + row["view2_split_score"])
                    for row in rows
                ]
            )
        ),
    }
    path = output_dir / "failure_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary, path


def error_rgb(error_codes):
    colors = np.asarray(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.8, 0.2],
            [0.9, 0.1, 0.1],
            [1.0, 0.8, 0.0],
            [0.85, 0.0, 0.85],
        ],
        dtype=np.float32,
    )
    return colors[error_codes]


def label_rgb(labels, num_queries):
    palette = np.asarray(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [160, 160, 160],
        ],
        dtype=np.float32,
    ) / 255.0
    image = np.zeros((*labels.shape, 3), dtype=np.float32)
    for query_id in range(num_queries + 1):
        image[labels == query_id] = palette[query_id % len(palette)]
    return image


def render_case(case, output_path, num_queries):
    row = case["row"]
    figure, axes = plt.subplots(2, 7, figsize=(24, 7))
    for r, (rgb, view, diag, prefix) in enumerate(
        [
            (case["rgb1"], case["view1"], case["diag1"], "view1"),
            (case["rgb2"], case["view2"], case["diag2"], "view2"),
        ]
    ):
        geom_strength = diag["geom_edge_strength"]
        if geom_strength is None:
            geom_strength = np.zeros_like(diag["rgb_edge_strength"])
        images = [
            rgb,
            label_rgb(view["gt_query"], num_queries),
            label_rgb(view["predicted"], num_queries),
            error_rgb(view["error_codes"]),
            diag["rgb_edge_strength"],
            geom_strength,
            np.asarray(diag["confusion"], dtype=np.float32),
        ]
        titles = [
            f"{prefix} RGB",
            "GT",
            "Pred",
            "Error",
            "RGB edge",
            "Geom edge",
            "Confusion",
        ]
        for c, (image, title) in enumerate(zip(images, titles)):
            ax = axes[r, c]
            if c in (4, 5):
                ax.imshow(image, cmap="gray")
            elif c == 6:
                ax.imshow(image, cmap="magma")
            else:
                ax.imshow(image)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    figure.suptitle(
        f"idx={row['sample_idx']} type={row['pair_failure_type']} "
        f"iou={row['mean_iou']:.3f} leak={row['leakage']:.3f} "
        f"v1={row['view1_failure_type']} v2={row['view2_failure_type']}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def save_visuals(cases, output_dir, max_visuals, num_queries):
    bad_cases = [case for case in cases if case["row"]["pair_failure_type"] != "ok"]
    bad_cases.sort(key=lambda case: (case["row"]["mean_iou"], -case["row"]["leakage"]))
    visual_dir = output_dir / "failure_visuals"
    for rank, case in enumerate(bad_cases[:max_visuals], start=1):
        row = case["row"]
        name = (
            f"rank_{rank:02d}_idx_{row['sample_idx']:06d}_"
            f"{row['pair_failure_type']}_iou_{row['mean_iou']:.3f}.png"
        )
        render_case(case, visual_dir / name, num_queries)
    return visual_dir, min(max_visuals, len(bad_cases))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = safe_load_checkpoint(args.checkpoint, device="cpu")
    cache = torch.load(args.feature_cache_path, map_location="cpu", weights_only=False)
    cache_config = cache.get("config", {})
    head, config = build_multiscale_head(checkpoint, cache_config, device)
    samples = flatten_cached_samples(cache[args.split])
    metadata = load_cache_metadata(cache_config, args.split, len(samples))
    rows, cases = collect_diagnostics(head, samples, metadata, config, device, args)

    _, csv_path = save_rows(rows, output_dir)
    summary, summary_path = save_summary(rows, output_dir)
    visual_dir, visual_count = save_visuals(
        cases,
        output_dir,
        args.max_visuals,
        int(config.get("num_queries", 8)),
    )

    print("=" * 76)
    print(f"mean_iou          = {summary['mean_iou']:.6f}")
    print(f"mean_leakage      = {summary['mean_leakage']:.6f}")
    print(f"bad_sample_count  = {summary['bad_sample_count']}/{summary['num_samples']}")
    print("failure_type_counts:")
    for key, value in sorted(summary["failure_type_counts"].items()):
        print(f"  {key:<24} {value}")
    print("=" * 76)
    print(f"Saved CSV     : {csv_path}")
    print(f"Saved summary : {summary_path}")
    print(f"Saved visuals : {visual_dir} ({visual_count})")


if __name__ == "__main__":
    main()
