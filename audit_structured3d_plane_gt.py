import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from dataloaders.s3d_dataset import Structured3DDataset


COLORS = np.asarray(
    [
        [230, 57, 53],
        [33, 150, 243],
        [67, 160, 71],
        [255, 143, 0],
        [142, 36, 170],
        [0, 137, 123],
        [244, 81, 30],
        [117, 117, 117],
        [121, 85, 72],
        [0, 188, 212],
        [205, 220, 57],
        [233, 30, 99],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser("Audit Structured3D plane polygon masks")
    parser.add_argument("--root_dir", default="data/Structured3D")
    parser.add_argument(
        "--cache_path",
        default="local_outputs/feature_cache/multiscale_symref_balanced_train512_val128.pt",
    )
    parser.add_argument(
        "--metrics_path",
        default="local_outputs/vis_multiscale_balanced512/all_metrics.json",
        help="Optional model metrics JSON used only for cross-reference; leave empty for GT-only audit.",
    )
    parser.add_argument(
        "--output_dir",
        default="local_outputs/gt_audit_structured3d",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--pair_strategy", default=None)
    parser.add_argument("--pair_max_view_id_gap", type=int, default=None)
    parser.add_argument(
        "--focus_indices",
        default="74,6,44,72,23,35,31,102",
        help="Cache indices to render in detail",
    )
    parser.add_argument("--max_focus", type=int, default=16)
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Audit only the first N selected cache pairs; 0 means all selected pairs.",
    )
    parser.add_argument(
        "--no_visuals",
        action="store_true",
        help="Skip PNG rendering so the audit can run in minimal Python environments.",
    )
    return parser.parse_args()


def colorize_labels(labels):
    image = np.full((*labels.shape, 3), 245, dtype=np.uint8)
    for value in np.unique(labels):
        if value <= 0:
            continue
        if value == 255:
            image[labels == value] = np.asarray([30, 30, 30], dtype=np.uint8)
        else:
            image[labels == value] = COLORS[(int(value) - 1) % len(COLORS)]
    return image


def load_cache_config(path):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict) or "config" not in cache:
        raise ValueError(f"Cannot find cache config in {path}")
    return cache["config"]


def valid_polygon_from_indices(junctions, indices):
    points = [
        junctions[index]["coordinate"]
        for index in indices
        if isinstance(index, int) and 0 <= index < len(junctions)
    ]
    if len(points) < 3:
        return None
    polygon = np.asarray(points, dtype=np.float32)
    if not np.isfinite(polygon).all():
        return None
    return np.rint(polygon).astype(np.int32)


def rasterize_individual_planes(layout, height, width):
    planes = layout.get("planes", [])
    junctions = layout.get("junctions", [])
    individual = []
    valid_polygon_count = 0
    empty_polygon_planes = 0
    for plane_index, plane in enumerate(planes, start=1):
        mask = np.zeros((height, width), dtype=np.uint8)
        valid_for_plane = 0
        for indices in plane.get("visible_mask", []):
            polygon = valid_polygon_from_indices(junctions, indices)
            if polygon is None:
                continue
            valid_polygon_count += 1
            valid_for_plane += 1
            cv2.fillPoly(mask, [polygon], color=1)
        if valid_for_plane == 0:
            empty_polygon_planes += 1
        individual.append(mask.astype(bool))
    return individual, valid_polygon_count, empty_polygon_planes


def legacy_from_individual(individual):
    if not individual:
        return np.zeros((0, 0), dtype=np.int32)
    height, width = individual[0].shape
    label = np.zeros((height, width), dtype=np.int32)
    for plane_index, mask in enumerate(individual, start=1):
        label[mask] = plane_index
    return label


def overlap_ignore_from_individual(individual):
    if not individual:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.uint16)
    stack = np.stack(individual, axis=0)
    overlap_count = stack.sum(axis=0).astype(np.uint16)
    label = np.zeros(overlap_count.shape, dtype=np.int32)
    for plane_index, mask in enumerate(individual, start=1):
        unique = mask & (overlap_count == 1)
        label[unique] = plane_index
    label[overlap_count >= 2] = 255
    return label, overlap_count


def resize_label(label, image_size):
    if label.shape == (image_size, image_size):
        return label
    return cv2.resize(label, (image_size, image_size), interpolation=cv2.INTER_NEAREST)


def audit_view(view_info, image_size):
    rgb_bgr = cv2.imread(view_info["rgb_path"])
    if rgb_bgr is None:
        raise RuntimeError(f"Failed to read image: {view_info['rgb_path']}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    with open(view_info["json_path"], "r", encoding="utf-8") as handle:
        layout = json.load(handle)

    individual, valid_polygon_count, empty_polygon_planes = rasterize_individual_planes(
        layout, height, width
    )
    legacy = legacy_from_individual(individual)
    overlap_ignore, overlap_count = overlap_ignore_from_individual(individual)
    total = int(height * width)
    raw_plane_count = len(individual)
    individual_pixels = np.asarray([int(mask.sum()) for mask in individual], dtype=np.int64)
    legacy_ids = [int(value) for value in np.unique(legacy) if value > 0]
    visible_legacy_count = len(legacy_ids)
    disappeared = [
        plane_index
        for plane_index, pixels in enumerate(individual_pixels, start=1)
        if pixels > 0 and plane_index not in legacy_ids
    ]
    covered0 = int((overlap_count == 0).sum())
    covered1 = int((overlap_count == 1).sum())
    covered2 = int((overlap_count >= 2).sum())
    overlap_ratio = covered2 / max(total, 1)
    largest_individual_ratio = (
        float(individual_pixels.max()) / total if len(individual_pixels) else 0.0
    )
    legacy_counts = np.asarray(
        [int((legacy == plane_id).sum()) for plane_id in legacy_ids], dtype=np.int64
    )
    legacy_largest_ratio = float(legacy_counts.max()) / total if len(legacy_counts) else 0.0

    legacy512 = resize_label(legacy, image_size)
    overlap_ignore512 = resize_label(overlap_ignore, image_size)
    overlap_count512 = resize_label(overlap_count.astype(np.int32), image_size)
    rgb512 = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    independent_color = np.zeros((height, width, 3), dtype=np.float32)
    alpha = np.zeros((height, width), dtype=np.float32)
    for plane_index, mask in enumerate(individual, start=1):
        color = COLORS[(plane_index - 1) % len(COLORS)].astype(np.float32)
        independent_color[mask] += color
        alpha[mask] += 1.0
    visible = alpha > 0
    independent_color[visible] /= alpha[visible, None]
    independent_color[~visible] = 245
    independent512 = cv2.resize(
        independent_color.astype(np.uint8),
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    )

    metrics = {
        "raw_plane_count": raw_plane_count,
        "valid_polygon_count": int(valid_polygon_count),
        "empty_polygon_planes": int(empty_polygon_planes),
        "individual_visible_plane_count": int((individual_pixels > 0).sum()),
        "legacy_visible_plane_count": visible_legacy_count,
        "fully_covered_plane_count": len(disappeared),
        "overlap_pixel_ratio": overlap_ratio,
        "covered_by_0_ratio": covered0 / max(total, 1),
        "covered_by_1_ratio": covered1 / max(total, 1),
        "covered_by_ge2_ratio": covered2 / max(total, 1),
        "largest_individual_plane_ratio": largest_individual_ratio,
        "legacy_largest_plane_ratio": legacy_largest_ratio,
        "legacy_single_plane_dominance": legacy_largest_ratio,
        "overlap_ignore_ratio": overlap_ratio,
        "disappeared_plane_ids": disappeared,
    }
    visuals = {
        "rgb": rgb512,
        "independent": independent512,
        "overlap_count": overlap_count512,
        "legacy": legacy512,
        "overlap_ignore": overlap_ignore512,
    }
    return metrics, visuals


def save_view_audit_figure(path, title, metrics, visuals):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    panels = [
        (visuals["rgb"], "RGB"),
        (visuals["independent"], "Individual plane masks"),
        (visuals["overlap_count"], "Overlap count"),
        (colorize_labels(visuals["legacy"]), "Legacy final GT"),
        (colorize_labels(visuals["overlap_ignore"]), "Overlap-ignore GT"),
    ]
    text = "\n".join(
        [
            title,
            f"raw planes: {metrics['raw_plane_count']}",
            f"legacy visible: {metrics['legacy_visible_plane_count']}",
            f"fully covered: {metrics['fully_covered_plane_count']}",
            f"overlap ratio: {metrics['overlap_pixel_ratio']:.3f}",
            f"legacy largest: {metrics['legacy_largest_plane_ratio']:.3f}",
        ]
    )
    panels.append((text, "Stats"))
    for axis, (image, name) in zip(axes.flat, panels):
        if isinstance(image, str):
            axis.text(0.02, 0.98, image, va="top", ha="left", family="monospace")
        else:
            cmap = "magma" if name == "Overlap count" else None
            axis.imshow(image, cmap=cmap)
        axis.set_title(name)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    def mean(key):
        values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
        return sum(values) / len(values) if values else 0.0

    severe_overlap = [row for row in rows if row["overlap_pixel_ratio"] >= 0.10]
    covered_planes = [row for row in rows if row["fully_covered_plane_count"] > 0]
    dominant = [row for row in rows if row["legacy_largest_plane_ratio"] >= 0.80]
    return {
        "view_count": len(rows),
        "mean_overlap_pixel_ratio": mean("overlap_pixel_ratio"),
        "median_overlap_pixel_ratio": float(
            np.median([float(row["overlap_pixel_ratio"]) for row in rows])
        )
        if rows
        else 0.0,
        "views_overlap_ge_10pct": len(severe_overlap),
        "views_with_fully_covered_planes": len(covered_planes),
        "views_legacy_largest_ge_80pct": len(dominant),
        "mean_raw_plane_count": mean("raw_plane_count"),
        "mean_legacy_visible_plane_count": mean("legacy_visible_plane_count"),
        "mean_fully_covered_plane_count": mean("fully_covered_plane_count"),
        "mean_legacy_largest_plane_ratio": mean("legacy_largest_plane_ratio"),
    }


def save_overview(path, rows):
    if not rows:
        return
    import matplotlib.pyplot as plt

    overlap = np.asarray([float(row["overlap_pixel_ratio"]) for row in rows])
    covered = np.asarray([float(row["fully_covered_plane_count"]) for row in rows])
    dominance = np.asarray([float(row["legacy_largest_plane_ratio"]) for row in rows])
    visible = np.asarray([float(row["legacy_visible_plane_count"]) for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(overlap, bins=30)
    axes[0, 0].set_title("Overlap pixel ratio")
    axes[0, 1].hist(covered, bins=np.arange(covered.max() + 2) - 0.5)
    axes[0, 1].set_title("Fully covered plane count")
    axes[1, 0].hist(dominance, bins=30)
    axes[1, 0].set_title("Legacy largest plane ratio")
    axes[1, 1].hist(visible, bins=np.arange(visible.max() + 2) - 0.5)
    axes[1, 1].set_title("Legacy visible plane count")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visuals"
    if not args.no_visuals:
        visual_dir.mkdir(parents=True, exist_ok=True)

    config = load_cache_config(args.cache_path)
    selected_key = f"selected_{args.split}_indices"
    selected_indices = config[selected_key]
    if args.max_pairs > 0:
        selected_indices = selected_indices[: args.max_pairs]
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio if args.train_ratio is not None else float(config.get("train_ratio", 0.9)),
        image_size=(512, 512),
        input_mode="pair",
        pair_strategy=args.pair_strategy or config.get("pair_strategy", "adjacent"),
        pair_max_view_id_gap=args.pair_max_view_id_gap
        if args.pair_max_view_id_gap is not None
        else config.get("pair_max_view_id_gap"),
    )
    metrics_by_idx = {}
    if args.metrics_path and args.metrics_path.lower() not in ("none", "null"):
        metrics_path = Path(args.metrics_path)
        if metrics_path.exists():
            metrics_rows = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics_by_idx = {int(row["sample_idx"]): row for row in metrics_rows}
    focus_indices = [
        int(value.strip())
        for value in args.focus_indices.split(",")
        if value.strip()
    ][: args.max_focus]

    rows = []
    pair_rows = []
    for cache_idx, dataset_idx in enumerate(selected_indices):
        sample = dataset.samples[int(dataset_idx)]
        pair_metric = metrics_by_idx.get(cache_idx, {})
        pair_summary = {
            "cache_idx": cache_idx,
            "dataset_idx": int(dataset_idx),
            "scene_name": sample["scene_name"],
            "pair_group": sample["pair_group"],
            "mean_iou": pair_metric.get("mean_iou", ""),
            "leakage": pair_metric.get("leakage", ""),
            "view_gap": pair_metric.get("view_gap", ""),
        }
        view_summaries = []
        for view_name in ("view1", "view2"):
            view_info = sample[view_name]
            metrics, visuals = audit_view(view_info, args.image_size)
            row = {
                **pair_summary,
                "view": view_name,
                "view_id": view_info["view_id"],
                "rgb_path": view_info["rgb_path"],
                "json_path": view_info["json_path"],
                **{
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in metrics.items()
                },
            }
            rows.append(row)
            view_summaries.append(metrics)
            if not args.no_visuals and cache_idx in focus_indices:
                title = (
                    f"cache_idx={cache_idx} dataset_idx={dataset_idx} "
                    f"{view_name} view_id={view_info['view_id']}"
                )
                save_view_audit_figure(
                    visual_dir / f"cache_{cache_idx:06d}_{view_name}_gt_audit.png",
                    title,
                    metrics,
                    visuals,
                )
        pair_rows.append(
            {
                **pair_summary,
                "view1_overlap_ratio": view_summaries[0]["overlap_pixel_ratio"],
                "view2_overlap_ratio": view_summaries[1]["overlap_pixel_ratio"],
                "view1_fully_covered": view_summaries[0]["fully_covered_plane_count"],
                "view2_fully_covered": view_summaries[1]["fully_covered_plane_count"],
                "view1_legacy_largest": view_summaries[0]["legacy_largest_plane_ratio"],
                "view2_legacy_largest": view_summaries[1]["legacy_largest_plane_ratio"],
                "view1_legacy_visible": view_summaries[0]["legacy_visible_plane_count"],
                "view2_legacy_visible": view_summaries[1]["legacy_visible_plane_count"],
            }
        )

    write_csv(output_dir / "audit_summary.csv", rows)
    write_csv(output_dir / "pair_audit_summary.csv", pair_rows)
    summary = {
        "config": {
            "root_dir": args.root_dir,
            "cache_path": args.cache_path,
            "metrics_path": args.metrics_path,
            "split": args.split,
            "selected_count": len(selected_indices),
        },
        "view_summary": summarize(rows),
        "focus_indices": focus_indices,
        "worst_pairs": [
            row
            for row in sorted(
                pair_rows,
                key=lambda item: float(item["mean_iou"]) if item["mean_iou"] != "" else 1.0,
            )[:16]
        ],
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps({"summary": summary, "views": rows, "pairs": pair_rows}, indent=2),
        encoding="utf-8",
    )
    if not args.no_visuals:
        save_overview(output_dir / "gt_audit_overview.png", rows)

    report = [
        "# Structured3D Plane GT Audit",
        "",
        "This audit is read-only. It reconstructs individual plane masks from layout.json and does not change Structured3DDataset.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary["view_summary"].items():
        report.append(f"- {key}: {value}")
    report.extend(
        [
            "",
            "## Worst Pairs",
            "",
            "| cache_idx | mean_iou | leakage | view_gap | overlap v1/v2 | fully covered v1/v2 | legacy largest v1/v2 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["worst_pairs"]:
        report.append(
            f"| {row['cache_idx']} | {row['mean_iou']} | {row['leakage']} | {row['view_gap']} | "
            f"{row['view1_overlap_ratio']:.3f}/{row['view2_overlap_ratio']:.3f} | "
            f"{row['view1_fully_covered']}/{row['view2_fully_covered']} | "
            f"{row['view1_legacy_largest']:.3f}/{row['view2_legacy_largest']:.3f} |"
        )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "- High overlap or fully covered plane counts indicate that legacy fillPoly ordering can hide planes in the final label map.",
            "- Low IoU with low overlap is more likely a model-side failure.",
            "- overlap-ignore labels are diagnostic only; large ignored areas can remove useful supervision.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary["view_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
