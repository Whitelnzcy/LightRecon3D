import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_stage1_clean_baseline import apply_query_class_scores
from visualize_predictions import (
    STAGE_NAMES,
    analyze_view,
    build_multiscale_head,
    colorize_class_map,
    load_cache_metadata,
    output_slice,
    safe_load_checkpoint,
    tensor_image_to_numpy,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Diagnose Stage1 masks with Stage2-style 3D geometry consistency"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_cache_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_visuals", type=int, default=40)
    parser.add_argument("--min_pixels", type=int, default=64)
    parser.add_argument("--adjacency_radius", type=int, default=2)
    parser.add_argument("--merge_angle_deg", type=float, default=12.0)
    parser.add_argument("--merge_offset", type=float, default=0.12)
    parser.add_argument("--merge_residual", type=float, default=0.08)
    parser.add_argument("--merge_max_boundary_geom_edge", type=float, default=1.0)
    parser.add_argument("--merge_max_boundary_rgb_edge", type=float, default=1.0)
    parser.add_argument("--merge_min_area_ratio", type=float, default=0.0)
    parser.add_argument("--merge_max_area_ratio", type=float, default=1.0)
    parser.add_argument("--split_angle_deg", type=float, default=28.0)
    parser.add_argument("--split_residual", type=float, default=0.16)
    parser.add_argument("--bad_iou_threshold", type=float, default=0.82)
    parser.add_argument(
        "--apply_merge_correction",
        action="store_true",
        help="Apply high-confidence geometry merge pairs to produce corrected masks.",
    )
    return parser.parse_args()


def dilate(mask, radius):
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(mask)
            ys0 = max(0, -dy)
            ys1 = mask.shape[0] - max(0, dy)
            xs0 = max(0, -dx)
            xs1 = mask.shape[1] - max(0, dx)
            yd0 = max(0, dy)
            yd1 = mask.shape[0] - max(0, -dy)
            xd0 = max(0, dx)
            xd1 = mask.shape[1] - max(0, -dx)
            shifted[yd0:yd1, xd0:xd1] = mask[ys0:ys1, xs0:xs1]
            out |= shifted
    return out


def fit_plane(points):
    if len(points) < 3:
        return None
    points = points.astype(np.float64, copy=False)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    centered = points - centroid[None, :]
    covariance = np.empty((3, 3), dtype=np.float64)
    scale = 1.0 / max(len(points), 1)
    for row in range(3):
        for col in range(3):
            covariance[row, col] = float((centered[:, row] * centered[:, col]).sum() * scale)
    covariance_t = torch.from_numpy(covariance).to(dtype=torch.float64)
    try:
        _, vectors = torch.linalg.eigh(covariance_t)
    except RuntimeError:
        return None
    normal = vectors[:, 0].cpu().numpy().astype(np.float64)
    norm = math.sqrt(float((normal * normal).sum()))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    normal = normal / norm
    offset = -float((normal * centroid).sum())
    signed = (
        points[:, 0] * normal[0]
        + points[:, 1] * normal[1]
        + points[:, 2] * normal[2]
        + offset
    )
    residual = np.abs(signed)
    return {
        "normal": normal,
        "offset": offset,
        "centroid": centroid,
        "mean_residual": float(residual.mean()) if len(residual) else 0.0,
        "p90_residual": float(np.quantile(residual, 0.90)) if len(residual) else 0.0,
    }


def normal_angle_deg(a, b):
    dot = abs(float((a * b).sum()))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(math.acos(dot))


def mutual_residual(points_a, plane_a, points_b, plane_b):
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0
    normal_a = plane_a["normal"]
    normal_b = plane_b["normal"]
    rb_signed = (
        points_b[:, 0] * normal_a[0]
        + points_b[:, 1] * normal_a[1]
        + points_b[:, 2] * normal_a[2]
        + plane_a["offset"]
    )
    ra_signed = (
        points_a[:, 0] * normal_b[0]
        + points_a[:, 1] * normal_b[1]
        + points_a[:, 2] * normal_b[2]
        + plane_b["offset"]
    )
    rb = np.abs(rb_signed).mean()
    ra = np.abs(ra_signed).mean()
    return float(0.5 * (ra + rb))


def overlap_distribution(pred_mask, gt_query, num_queries):
    valid = (gt_query >= 0) & (gt_query < num_queries)
    total = int((pred_mask & valid).sum())
    if total <= 0:
        return []
    rows = []
    for gt_id in range(num_queries):
        count = int((pred_mask & (gt_query == gt_id)).sum())
        if count:
            rows.append((gt_id, count / total))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def union_find_groups(num_queries, edges):
    parent = list(range(num_queries))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for a, b in edges:
        ra = find(int(a))
        rb = find(int(b))
        if ra == rb:
            continue
        parent[max(ra, rb)] = min(ra, rb)

    return {query: find(query) for query in range(num_queries)}


def apply_merge_edges(predicted, edges, num_queries):
    groups = union_find_groups(num_queries, edges)
    corrected = predicted.copy()
    for query_id, root in groups.items():
        corrected[predicted == query_id] = root
    return corrected, groups


def label_agnostic_metrics(predicted, gt_query, num_queries):
    valid_gt = (gt_query >= 0) & (gt_query < num_queries)
    gt_ids = sorted(int(v) for v in np.unique(gt_query[valid_gt]))
    pred_ids = sorted(int(v) for v in np.unique(predicted[predicted < num_queries]))
    ious = []
    leakage_pixels = 0
    valid_pixels = int(valid_gt.sum())
    for gt_id in gt_ids:
        gt_mask = gt_query == gt_id
        best_iou = 0.0
        best_pred = None
        for pred_id in pred_ids:
            pred_mask = predicted == pred_id
            union = int((gt_mask | pred_mask).sum())
            if union <= 0:
                continue
            iou = float((gt_mask & pred_mask).sum() / union)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred_id
        ious.append(best_iou)
        if best_pred is not None:
            leakage_pixels += int((gt_mask & (predicted != best_pred)).sum())
        else:
            leakage_pixels += int(gt_mask.sum())
    return {
        "label_agnostic_iou": float(np.mean(ious)) if ious else 0.0,
        "label_agnostic_leakage": float(leakage_pixels / max(valid_pixels, 1)),
        "pred_region_count": len(pred_ids),
    }


def rgb_edge_strength(rgb):
    gray = rgb.mean(axis=-1)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    edge = gx + gy
    max_value = float(edge.max())
    return edge / max(max_value, 1e-6)


def mean_on_mask(value, mask):
    if not mask.any():
        return 0.0
    return float(value[mask].mean())


def view_geometry_pairs(output, labels, geometry, rgb, config, args):
    num_queries = int(config.get("num_queries", 8))
    view = analyze_view(output, labels, config)
    mask_logits = output["mask_logits"]
    background_logits = output["background_logits"]
    existence_logits = output["existence_logits"]
    class_logits = torch.cat(
        (
            apply_query_class_scores(
                mask_logits,
                existence_logits,
                argparse.Namespace(
                    class_score_weight=float(config.get("class_score_weight", 0.0))
                ),
            ),
            background_logits,
        ),
        dim=0,
    )
    probs = torch.softmax(class_logits, dim=0)
    top2 = torch.topk(probs, k=2, dim=0).values
    confidence = top2[0].detach().cpu().numpy()
    margin = (top2[0] - top2[1]).detach().cpu().numpy()
    rgb_np = rgb.detach().float().cpu().numpy().transpose(1, 2, 0)
    rgb_edge = rgb_edge_strength(rgb_np)

    predicted = view["predicted"]
    gt_query = view["gt_query"]
    points = geometry[:3].detach().float().cpu().numpy().transpose(1, 2, 0)
    geom_edge = geometry[8].detach().float().cpu().numpy()
    valid = np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) > 1e-6)

    planes = {}
    query_points = {}
    for query_id in range(num_queries):
        mask = (predicted == query_id) & valid
        if int(mask.sum()) < args.min_pixels:
            continue
        # Use high-confidence interior when possible; fall back to full support.
        boundary = dilate(mask, 1) & (~mask)
        core = mask & (confidence > 0.55) & (margin > 0.12) & (~boundary)
        fit_mask = core if int(core.sum()) >= args.min_pixels else mask
        pts = points[fit_mask]
        plane = fit_plane(pts)
        if plane is None:
            continue
        planes[query_id] = {
            **plane,
            "pixel_count": int(mask.sum()),
            "fit_count": int(fit_mask.sum()),
            "dominant_gt": overlap_distribution(mask, gt_query, num_queries),
        }
        query_points[query_id] = points[mask]

    pair_rows = []
    merge_edges = []
    split_edges = []
    query_ids = sorted(planes)
    for i, q_a in enumerate(query_ids):
        mask_a = predicted == q_a
        dilated_a = dilate(mask_a, args.adjacency_radius)
        for q_b in query_ids[i + 1 :]:
            mask_b = predicted == q_b
            adjacent = bool((dilated_a & mask_b).any())
            if not adjacent:
                continue
            boundary = (dilate(mask_a, 1) & mask_b) | (dilate(mask_b, 1) & mask_a)
            area_a = int(mask_a.sum())
            area_b = int(mask_b.sum())
            area_ratio = min(area_a, area_b) / max(max(area_a, area_b), 1)
            boundary_rgb = mean_on_mask(rgb_edge, boundary)
            boundary_geom = mean_on_mask(geom_edge, boundary)
            mean_conf = 0.5 * (
                mean_on_mask(confidence, mask_a) + mean_on_mask(confidence, mask_b)
            )
            mean_margin = 0.5 * (
                mean_on_mask(margin, mask_a) + mean_on_mask(margin, mask_b)
            )
            plane_a = planes[q_a]
            plane_b = planes[q_b]
            angle = normal_angle_deg(plane_a["normal"], plane_b["normal"])
            offset = abs(abs(float(plane_a["offset"])) - abs(float(plane_b["offset"])))
            residual = mutual_residual(
                query_points[q_a],
                plane_a,
                query_points[q_b],
                plane_b,
            )
            gt_a = plane_a["dominant_gt"][0][0] if plane_a["dominant_gt"] else -1
            gt_b = plane_b["dominant_gt"][0][0] if plane_b["dominant_gt"] else -1
            same_gt = gt_a >= 0 and gt_a == gt_b
            should_merge = (
                angle <= args.merge_angle_deg
                and offset <= args.merge_offset
                and residual <= args.merge_residual
                and boundary_geom <= args.merge_max_boundary_geom_edge
                and boundary_rgb <= args.merge_max_boundary_rgb_edge
                and area_ratio >= args.merge_min_area_ratio
                and area_ratio <= args.merge_max_area_ratio
            )
            should_split = angle >= args.split_angle_deg or residual >= args.split_residual
            if should_merge:
                merge_edges.append((q_a, q_b))
            if should_split:
                split_edges.append((q_a, q_b))
            pair_rows.append(
                {
                    "query_a": q_a,
                    "query_b": q_b,
                    "angle_deg": angle,
                    "offset_diff": offset,
                    "mutual_residual": residual,
                    "area_a": area_a,
                    "area_b": area_b,
                    "area_ratio": area_ratio,
                    "boundary_rgb_edge": boundary_rgb,
                    "boundary_geom_edge": boundary_geom,
                    "mean_confidence": mean_conf,
                    "mean_margin": mean_margin,
                    "same_dominant_gt": same_gt,
                    "dominant_gt_a": gt_a,
                    "dominant_gt_b": gt_b,
                    "should_merge_by_geometry": should_merge,
                    "should_split_by_geometry": should_split,
                }
            )

    view["geometry"] = {
        "planes": planes,
        "pairs": pair_rows,
        "merge_edges": merge_edges,
        "split_edges": split_edges,
        "merge_pair_count": len(merge_edges),
        "split_pair_count": len(split_edges),
    }
    corrected, groups = apply_merge_edges(predicted, merge_edges, num_queries)
    view["corrected"] = corrected
    view["geometry"]["merge_groups"] = groups
    view["geometry"]["original_eval"] = label_agnostic_metrics(
        predicted, gt_query, num_queries
    )
    view["geometry"]["corrected_eval"] = label_agnostic_metrics(
        corrected, gt_query, num_queries
    )
    return view


def edge_overlay(predicted, edges, num_queries):
    image = colorize_class_map(predicted, num_queries)
    for q_a, q_b in edges:
        boundary = dilate(predicted == q_a, 1) & (predicted == q_b)
        boundary |= dilate(predicted == q_b, 1) & (predicted == q_a)
        image[boundary] = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    return image


def pair_matrix(view, key, num_queries):
    matrix = np.full((num_queries, num_queries), np.nan, dtype=np.float32)
    for row in view["geometry"]["pairs"]:
        a = int(row["query_a"])
        b = int(row["query_b"])
        matrix[a, b] = matrix[b, a] = float(row[key])
    return matrix


def render_case(case, output_path, num_queries):
    fig, axes = plt.subplots(2, 7, figsize=(29, 8))
    for row, prefix in enumerate(("view1", "view2")):
        rgb = case[f"{prefix}_rgb"]
        view = case[prefix]
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"{prefix} RGB")
        axes[row, 1].imshow(colorize_class_map(view["gt_query"], num_queries))
        axes[row, 1].set_title("GT")
        axes[row, 2].imshow(colorize_class_map(view["predicted"], num_queries))
        axes[row, 2].set_title("Stage1 pred")
        axes[row, 3].imshow(
            edge_overlay(
                view["predicted"],
                view["geometry"]["merge_edges"],
                num_queries,
            )
        )
        axes[row, 3].set_title("Geom merge edges")
        axes[row, 4].imshow(
            edge_overlay(
                view["predicted"],
                view["geometry"]["split_edges"],
                num_queries,
            )
        )
        axes[row, 4].set_title("Geom split edges")
        axes[row, 5].imshow(colorize_class_map(view["corrected"], num_queries))
        original = view["geometry"]["original_eval"]["label_agnostic_iou"]
        corrected = view["geometry"]["corrected_eval"]["label_agnostic_iou"]
        axes[row, 5].set_title(f"Corrected {original:.3f}->{corrected:.3f}")
        matrix = pair_matrix(view, "angle_deg", num_queries)
        masked = np.ma.masked_invalid(matrix)
        axes[row, 6].imshow(masked, vmin=0.0, vmax=60.0, cmap="magma")
        axes[row, 6].set_title("Pair normal angle")
        for col in range(7):
            axes[row, col].axis("off")
    metrics = case["metrics"]
    fig.suptitle(
        f"idx={metrics['sample_idx']} dataset={metrics.get('dataset_index', '')} "
        f"iou={metrics['mean_iou']:.3f} mergePairs={metrics['geom_merge_pairs']} "
        f"splitPairs={metrics['geom_split_pairs']}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    visual_dir = output_dir / "geometry_pair_visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = safe_load_checkpoint(args.checkpoint, device="cpu")
    cache = torch.load(args.feature_cache_path, map_location="cpu", weights_only=False)
    cache_config = cache.get("config", {})
    samples = cache[args.split]
    metadata = load_cache_metadata(cache_config, args.split, len(samples))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head, config = build_multiscale_head(checkpoint, cache_config, device)
    num_queries = int(config.get("num_queries", 8))

    rows = []
    cases = []
    pair_rows = []
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
        rgb1 = torch.cat([item["rgb1"] for item in items], dim=0).to(device=device, dtype=torch.float32)
        rgb2 = torch.cat([item["rgb2"] for item in items], dim=0).to(device=device, dtype=torch.float32)
        geometry1 = torch.cat([item["geometry1"] for item in items], dim=0).to(device=device, dtype=torch.float32)
        geometry2 = torch.cat([item["geometry2"] for item in items], dim=0).to(device=device, dtype=torch.float32)
        geometry = torch.cat((geometry1, geometry2), dim=0)
        output = head(features, torch.cat((rgb1, rgb2), dim=0), geometry=geometry)

        for local_index, item in enumerate(items):
            sample_idx = start + local_index
            view1 = view_geometry_pairs(
                output_slice(output, local_index),
                item["gt_plane1"][0].to(device),
                geometry1[local_index],
                rgb1[local_index],
                config,
                args,
            )
            view2 = view_geometry_pairs(
                output_slice(output, pair_batch + local_index),
                item["gt_plane2"][0].to(device),
                geometry2[local_index],
                rgb2[local_index],
                config,
                args,
            )
            metrics = {
                "sample_idx": sample_idx,
                **metadata[sample_idx],
                "view1_iou": view1["metrics"]["mean_iou"],
                "view2_iou": view2["metrics"]["mean_iou"],
                "mean_iou": 0.5
                * (view1["metrics"]["mean_iou"] + view2["metrics"]["mean_iou"]),
                "view1_leakage": view1["metrics"]["leakage_rate"],
                "view2_leakage": view2["metrics"]["leakage_rate"],
                "mean_leakage": 0.5
                * (view1["metrics"]["leakage_rate"] + view2["metrics"]["leakage_rate"]),
                "view1_geom_merge_pairs": view1["geometry"]["merge_pair_count"],
                "view2_geom_merge_pairs": view2["geometry"]["merge_pair_count"],
                "view1_geom_split_pairs": view1["geometry"]["split_pair_count"],
                "view2_geom_split_pairs": view2["geometry"]["split_pair_count"],
                "view1_original_geom_iou": view1["geometry"]["original_eval"][
                    "label_agnostic_iou"
                ],
                "view2_original_geom_iou": view2["geometry"]["original_eval"][
                    "label_agnostic_iou"
                ],
                "view1_corrected_geom_iou": view1["geometry"]["corrected_eval"][
                    "label_agnostic_iou"
                ],
                "view2_corrected_geom_iou": view2["geometry"]["corrected_eval"][
                    "label_agnostic_iou"
                ],
                "view1_original_geom_leakage": view1["geometry"]["original_eval"][
                    "label_agnostic_leakage"
                ],
                "view2_original_geom_leakage": view2["geometry"]["original_eval"][
                    "label_agnostic_leakage"
                ],
                "view1_corrected_geom_leakage": view1["geometry"]["corrected_eval"][
                    "label_agnostic_leakage"
                ],
                "view2_corrected_geom_leakage": view2["geometry"]["corrected_eval"][
                    "label_agnostic_leakage"
                ],
                "view1_original_regions": view1["geometry"]["original_eval"][
                    "pred_region_count"
                ],
                "view2_original_regions": view2["geometry"]["original_eval"][
                    "pred_region_count"
                ],
                "view1_corrected_regions": view1["geometry"]["corrected_eval"][
                    "pred_region_count"
                ],
                "view2_corrected_regions": view2["geometry"]["corrected_eval"][
                    "pred_region_count"
                ],
            }
            metrics["geom_merge_pairs"] = (
                metrics["view1_geom_merge_pairs"] + metrics["view2_geom_merge_pairs"]
            )
            metrics["geom_split_pairs"] = (
                metrics["view1_geom_split_pairs"] + metrics["view2_geom_split_pairs"]
            )
            metrics["original_geom_iou"] = 0.5 * (
                metrics["view1_original_geom_iou"] + metrics["view2_original_geom_iou"]
            )
            metrics["corrected_geom_iou"] = 0.5 * (
                metrics["view1_corrected_geom_iou"] + metrics["view2_corrected_geom_iou"]
            )
            metrics["geom_iou_delta"] = (
                metrics["corrected_geom_iou"] - metrics["original_geom_iou"]
            )
            metrics["original_geom_leakage"] = 0.5 * (
                metrics["view1_original_geom_leakage"]
                + metrics["view2_original_geom_leakage"]
            )
            metrics["corrected_geom_leakage"] = 0.5 * (
                metrics["view1_corrected_geom_leakage"]
                + metrics["view2_corrected_geom_leakage"]
            )
            metrics["corrected_region_delta"] = (
                metrics["view1_corrected_regions"]
                + metrics["view2_corrected_regions"]
                - metrics["view1_original_regions"]
                - metrics["view2_original_regions"]
            )
            rows.append(metrics)
            case = {
                "metrics": metrics,
                "view1_rgb": tensor_image_to_numpy(item["rgb1"][0]),
                "view2_rgb": tensor_image_to_numpy(item["rgb2"][0]),
                "view1": view1,
                "view2": view2,
            }
            cases.append(case)
            for view_name, view in (("view1", view1), ("view2", view2)):
                for pair in view["geometry"]["pairs"]:
                    pair_rows.append(
                        {
                            "sample_idx": sample_idx,
                            **metadata[sample_idx],
                            "view": view_name,
                            **pair,
                            "mean_iou": metrics["mean_iou"],
                        }
                    )

        done = min(start + pair_batch, len(samples))
        if start == 0 or done == len(samples) or done % 32 == 0:
            print(f"[geometry correction] {done}/{len(samples)}", flush=True)

    rows_sorted = sorted(rows, key=lambda row: row["mean_iou"])
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "geometry_correction_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    pair_fieldnames = sorted({key for row in pair_rows for key in row})
    with (output_dir / "geometry_pair_decisions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pair_fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)

    visual_cases = sorted(
        cases,
        key=lambda case: (
            -case["metrics"]["geom_iou_delta"],
            -case["metrics"]["geom_merge_pairs"],
            case["metrics"]["mean_iou"],
        ),
    )[: args.max_visuals]
    for rank, case in enumerate(visual_cases, start=1):
        metrics = case["metrics"]
        render_case(
            case,
            visual_dir
            / (
                f"rank_{rank:02d}_idx_{metrics['sample_idx']:06d}_"
                f"gm_{metrics['geom_merge_pairs']}_gs_{metrics['geom_split_pairs']}_"
                f"diou_{metrics['geom_iou_delta']:+.3f}_"
                f"iou_{metrics['mean_iou']:.3f}.png"
            ),
            num_queries,
        )

    summary = {
        "num_samples": len(rows),
        "mean_iou": float(np.mean([row["mean_iou"] for row in rows])) if rows else 0.0,
        "mean_leakage": float(np.mean([row["mean_leakage"] for row in rows])) if rows else 0.0,
        "samples_with_geom_merge": int(sum(row["geom_merge_pairs"] > 0 for row in rows)),
        "samples_with_geom_split": int(sum(row["geom_split_pairs"] > 0 for row in rows)),
        "total_geom_merge_pairs": int(sum(row["geom_merge_pairs"] for row in rows)),
        "total_geom_split_pairs": int(sum(row["geom_split_pairs"] for row in rows)),
        "mean_original_geom_iou": float(np.mean([row["original_geom_iou"] for row in rows])) if rows else 0.0,
        "mean_corrected_geom_iou": float(np.mean([row["corrected_geom_iou"] for row in rows])) if rows else 0.0,
        "mean_geom_iou_delta": float(np.mean([row["geom_iou_delta"] for row in rows])) if rows else 0.0,
        "samples_improved_by_merge": int(sum(row["geom_iou_delta"] > 1e-4 for row in rows)),
        "samples_hurt_by_merge": int(sum(row["geom_iou_delta"] < -1e-4 for row in rows)),
        "mean_original_geom_leakage": float(np.mean([row["original_geom_leakage"] for row in rows])) if rows else 0.0,
        "mean_corrected_geom_leakage": float(np.mean([row["corrected_geom_leakage"] for row in rows])) if rows else 0.0,
        "bad_samples_with_geom_merge": int(
            sum(row["mean_iou"] < args.bad_iou_threshold and row["geom_merge_pairs"] > 0 for row in rows)
        ),
        "pair_decision_counts": dict(
            Counter(
                "merge"
                if row["should_merge_by_geometry"]
                else "split"
                if row["should_split_by_geometry"]
                else "neutral"
                for row in pair_rows
            )
        ),
        "args": vars(args),
    }
    (output_dir / "geometry_correction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("=" * 76)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
