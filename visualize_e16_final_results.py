"""Render reporting-only figures from frozen E16 artifacts.

This module never runs B16/M16, never changes predictions or GT, and never
recomputes the registered evaluator. It only visualizes frozen metrics and
point/view assignments after E16 has completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = (
    ROOT / "results" / "targeted_GT_sidecar_and_final_BM_e16_20260730_v1"
)
DEFAULT_CONFIG = ROOT / "configs" / "e16_final_bm.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_scene_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method = {
        method: {row["scene_id"]: row for row in metrics if row["method_id"] == method}
        for method in ("B16", "M16")
    }
    if set(by_method["B16"]) != set(by_method["M16"]):
        raise ValueError("B16/M16 scene sets differ")
    rows = []
    for scene_id in sorted(by_method["B16"]):
        b_row = by_method["B16"][scene_id]
        m_row = by_method["M16"][scene_id]
        rows.append(
            {
                "scene_id": scene_id,
                "B16": b_row,
                "M16": m_row,
                "delta_pair_f1": (
                    m_row["partition_pair_f1"] - b_row["partition_pair_f1"]
                ),
                "delta_plane_count": (
                    m_row["predicted_plane_count"] - b_row["predicted_plane_count"]
                ),
            }
        )
    return rows


def select_scene_groups(
    paired_rows: list[dict[str, Any]],
    similar_count: int = 4,
    extra_split_count: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    same_count = [row for row in paired_rows if row["delta_plane_count"] == 0]
    same_count.sort(key=lambda row: (abs(row["delta_pair_f1"]), row["scene_id"]))
    remaining = sorted(
        paired_rows,
        key=lambda row: (
            abs(row["delta_pair_f1"]),
            abs(row["delta_plane_count"]),
            row["scene_id"],
        ),
    )
    similar = same_count[:similar_count]
    for row in remaining:
        if len(similar) >= similar_count:
            break
        if row not in similar:
            similar.append(row)
    extra_split = sorted(
        (
            row
            for row in paired_rows
            if row["delta_plane_count"] > 0 and row["delta_pair_f1"] <= 0
        ),
        key=lambda row: (
            -row["delta_plane_count"],
            row["delta_pair_f1"],
            row["scene_id"],
        ),
    )[:extra_split_count]
    return similar, extra_split


def prediction_index(manifest: dict[str, Any]) -> dict[tuple[str, str], Path]:
    return {
        (row["scene_id"], row["method_id"]): Path(row["prediction_path"])
        for row in manifest["predictions"]
    }


def load_prediction_view(path: Path, view_index: int | None = None):
    with np.load(path, allow_pickle=False) as source:
        views = source["view_indices"].astype(np.int32)
        pixels = source["pixel_xy"].astype(np.int32)
        labels = source["plane_ids"].astype(np.int32)
        global_indices = source["global_point_indices"].astype(np.int64)
    if view_index is None:
        values, counts = np.unique(views, return_counts=True)
        view_index = int(values[np.argmax(counts)])
    selected = views == view_index
    return (
        int(view_index),
        pixels[selected],
        labels[selected],
        global_indices[selected],
    )


def label_raster(pixels: np.ndarray, labels: np.ndarray, height: int, width: int):
    raster = np.full((height, width), np.nan, dtype=np.float32)
    valid = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    raster[pixels[valid, 1], pixels[valid, 0]] = labels[valid]
    return raster


def align_partition_labels(
    reference: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    """Map candidate IDs to reference colors by maximum support overlap."""
    from scipy.optimize import linear_sum_assignment

    reference_ids = np.unique(reference)
    candidate_ids = np.unique(candidate)
    overlap = np.zeros((len(candidate_ids), len(reference_ids)), dtype=np.int64)
    for row, candidate_id in enumerate(candidate_ids):
        selected = candidate == candidate_id
        values, counts = np.unique(reference[selected], return_counts=True)
        lookup = {int(value): int(count) for value, count in zip(values, counts)}
        for column, reference_id in enumerate(reference_ids):
            overlap[row, column] = lookup.get(int(reference_id), 0)
    rows, columns = linear_sum_assignment(-overlap)
    mapping = {
        int(candidate_ids[row]): int(reference_ids[column])
        for row, column in zip(rows, columns)
        if overlap[row, column] > 0
    }
    next_id = int(reference_ids.max()) + 1 if len(reference_ids) else 0
    for candidate_id in candidate_ids:
        if int(candidate_id) not in mapping:
            mapping[int(candidate_id)] = next_id
            next_id += 1
    return np.asarray([mapping[int(value)] for value in candidate], dtype=np.int32)


def scene_shape(global_path: Path, view_index: int) -> tuple[int, int]:
    with np.load(global_path, allow_pickle=False) as source:
        views = source["view_indices"].astype(np.int32)
        pixels = source["pixel_xy"].astype(np.int32)
    selected = pixels[views == view_index]
    return int(selected[:, 1].max()) + 1, int(selected[:, 0].max()) + 1


def render_main_table(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    macro = summary["macro"]
    aggregate = summary["aggregate"]
    rows = [
        ("Plane F1", macro["B16"]["Plane_F1"], macro["M16"]["Plane_F1"]),
        (
            "Matched IoU",
            macro["B16"]["matched_iou_mean"],
            macro["M16"]["matched_iou_mean"],
        ),
        (
            "Partition pair F1",
            macro["B16"]["partition_pair_f1"],
            macro["M16"]["partition_pair_f1"],
        ),
        (
            "GT-support assignment",
            macro["B16"]["gt_support_assignment_rate"],
            macro["M16"]["gt_support_assignment_rate"],
        ),
        (
            "Full-cache label rate",
            macro["B16"]["full_cache_label_rate"],
            macro["M16"]["full_cache_label_rate"],
        ),
        (
            "Predicted planes (total)",
            aggregate["B16"]["predicted_plane_count"],
            aggregate["M16"]["predicted_plane_count"],
        ),
    ]
    cell_text = []
    for label, b_value, m_value in rows:
        integer = label.startswith("Predicted")
        formatter = (lambda value: f"{int(value)}") if integer else (
            lambda value: f"{value:.6f}"
        )
        delta = m_value - b_value
        cell_text.append(
            [
                label,
                formatter(b_value),
                formatter(m_value),
                f"{int(delta):+d}" if integer else f"{delta:+.6f}",
            ]
        )
    figure, axis = plt.subplots(figsize=(9.4, 3.7))
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        colLabels=["Metric", "B16", "M16", "M16 - B16"],
        colLoc="center",
        cellLoc="right",
        colWidths=[0.40, 0.19, 0.19, 0.22],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.65)
    for (row_index, column_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#dce6f2")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f2f4f7")
        if column_index == 0:
            cell.set_text_props(ha="left")
    axis.set_title(
        "E16 final B16/M16 comparison (128 frozen scenes)",
        fontsize=13,
        pad=12,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_delta_distribution(
    paired_rows: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    values = np.asarray([row["delta_pair_f1"] for row in paired_rows]) * 1000.0
    effect = next(
        row for row in bootstrap["effects"] if row["metric"] == "partition_pair_f1"
    )
    mean = effect["mean"] * 1000.0
    low = effect["ci95_lower"] * 1000.0
    high = effect["ci95_upper"] * 1000.0
    figure, axis = plt.subplots(figsize=(9.4, 4.8))
    bins = np.linspace(values.min(), values.max(), 25)
    axis.hist(values, bins=bins, color="#4c78a8", edgecolor="white", alpha=0.9)
    axis.axvspan(low, high, color="#f2cf5b", alpha=0.35, label="95% CI of mean")
    axis.axvline(0, color="#444444", linewidth=1.4, label="No change")
    axis.axvline(mean, color="#e45756", linewidth=2.0, label=f"Mean {mean:+.4f}")
    axis.set_xlabel("Per-scene M16 - B16 partition pair F1 (×10⁻³)")
    axis.set_ylabel("Scene count")
    axis.set_title("Paired identity effect is centered near zero")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    axis.text(
        0.99,
        0.96,
        (
            f"95% CI [{low:+.4f}, {high:+.4f}] ×10⁻³\n"
            f"W/T/L = {effect['positive_scene_count']}/"
            f"{effect['zero_scene_count']}/{effect['negative_scene_count']}"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_scene_gallery(
    selected_rows: list[dict[str, Any]],
    predictions: dict[tuple[str, str], Path],
    global_root: Path,
    output: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    palette = ListedColormap(plt.get_cmap("tab20").colors)
    figure, axes = plt.subplots(
        len(selected_rows), 2, figsize=(9.4, 2.25 * len(selected_rows)), squeeze=False
    )
    for row_index, row in enumerate(selected_rows):
        scene = row["scene_id"]
        b_path = predictions[(scene, "B16")]
        m_path = predictions[(scene, "M16")]
        with np.load(b_path, allow_pickle=False) as b_source:
            b_global = b_source["global_point_indices"].astype(np.int64)
            b_all_labels = b_source["plane_ids"].astype(np.int32)
        with np.load(m_path, allow_pickle=False) as m_source:
            m_global = m_source["global_point_indices"].astype(np.int64)
            m_all_labels = m_source["plane_ids"].astype(np.int32)
        if not np.array_equal(b_global, m_global):
            raise ValueError(f"{scene}: paired support order differs")
        m_display_labels = align_partition_labels(b_all_labels, m_all_labels)
        view_index, b_pixels, b_labels, b_view_global = load_prediction_view(b_path)
        _, m_pixels, _, m_view_global = load_prediction_view(m_path, view_index)
        b_lookup = {int(index): int(label) for index, label in zip(b_global, b_all_labels)}
        m_lookup = {
            int(index): int(label) for index, label in zip(m_global, m_display_labels)
        }
        b_labels = np.asarray(
            [b_lookup[int(index)] for index in b_view_global], dtype=np.int32
        )
        m_labels = np.asarray(
            [m_lookup[int(index)] for index in m_view_global], dtype=np.int32
        )
        global_path = global_root / scene / "global_cache_e14.npz"
        height, width = scene_shape(global_path, view_index)
        for column_index, (method, pixels, labels) in enumerate(
            (("B16", b_pixels, b_labels), ("M16", m_pixels, m_labels))
        ):
            axis = axes[row_index, column_index]
            raster = label_raster(pixels, labels, height, width)
            axis.imshow(raster, cmap=palette, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            metrics = row[method]
            axis.set_title(
                (
                    f"{scene} · {method} · view {view_index}\n"
                    f"Pair F1 {metrics['partition_pair_f1']:.6f} · "
                    f"planes {metrics['predicted_plane_count']}"
                ),
                fontsize=9.5,
            )
    figure.suptitle(title, fontsize=13)
    figure.text(
        0.5,
        0.005,
        "M16 display colors are matched to B16 by maximum frozen-support overlap.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.97))
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_support_gap(
    paired_rows: list[dict[str, Any]],
    predictions: dict[tuple[str, str], Path],
    gt_manifest: dict[str, Any],
    global_root: Path,
    macro_assignment: float,
    macro_gt_rate: float,
    output: Path,
) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    candidates = [row for row in paired_rows if row["B16"]["view_count"] == 5]
    representative = min(
        candidates,
        key=lambda row: abs(
            row["B16"]["gt_support_assignment_rate"] - macro_assignment
        ),
    )
    scene = representative["scene_id"]
    b_path = predictions[(scene, "B16")]
    view_index, b_pixels, _, b_global = load_prediction_view(b_path)
    gt_receipt = next(row for row in gt_manifest["receipts"] if row["scene_id"] == scene)
    global_path = global_root / scene / "global_cache_e14.npz"
    with np.load(global_path, allow_pickle=False) as cache:
        cache_views = cache["view_indices"].astype(np.int32)
        cache_pixels = cache["pixel_xy"].astype(np.int32)
    height, width = scene_shape(global_path, view_index)
    with np.load(gt_receipt["output_npz"], allow_pickle=False) as gt:
        gt_global = gt["source_cache_indices"].astype(np.int64)
        gt_labels = gt["point_plane_ids"].astype(np.int32)
    gt_selected = (cache_views[gt_global] == view_index) & (gt_labels >= 0)
    gt_pixels = cache_pixels[gt_global[gt_selected]]
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    gt_mask[gt_pixels[:, 1], gt_pixels[:, 0]] = 1
    pred_mask = np.zeros((height, width), dtype=np.uint8)
    pred_mask[b_pixels[:, 1], b_pixels[:, 0]] = 1
    overlay = np.zeros((height, width), dtype=np.uint8)
    overlay[gt_mask > 0] = 1
    overlay[pred_mask > 0] = 2

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    axes[0].imshow(gt_mask, cmap=ListedColormap(["#f1f3f5", "#4c78a8"]))
    axes[0].set_title("Complete visible GT support")
    axes[1].imshow(pred_mask, cmap=ListedColormap(["#f1f3f5", "#f58518"]))
    axes[1].set_title("B16 sparse direct support")
    axes[2].imshow(
        overlay,
        cmap=ListedColormap(["#f1f3f5", "#c8d5e5", "#f58518"]),
        vmin=0,
        vmax=2,
    )
    axes[2].set_title("Overlay: prediction inside GT domain")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        (
            f"{scene} · view {view_index} · macro label rate "
            f"{representative['B16']['full_cache_label_rate']:.1%} vs "
            f"GT plane rate {representative['B16']['gt_plane_label_rate']:.1%}\n"
            f"128-scene macro: assigned GT support {macro_assignment:.1%}, "
            f"GT plane rate {macro_gt_rate:.1%}"
        ),
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    results = args.results.resolve()
    output = results / "figures"
    output.mkdir(parents=True, exist_ok=True)
    summary = load_json(results / "summary.json")
    metrics = load_json(results / "per_scene_metrics.json")
    bootstrap = load_json(results / "bootstrap_statistics.json")
    prediction_manifest = load_json(results / "prediction_manifest.json")
    gt_manifest = load_json(results / "point_aligned_gt_manifest.json")
    config = load_json(args.config)
    paired_rows = paired_scene_rows(metrics)
    similar, extra_split = select_scene_groups(paired_rows)
    predictions = prediction_index(prediction_manifest)
    global_root = Path(config["global_cache_root"])

    paths = {
        "main_results_table": output / "e16_main_results_table.png",
        "pair_f1_delta_distribution": output / "e16_pair_f1_delta_distribution.png",
        "similar_scene_gallery": output / "e16_similar_scene_gallery.png",
        "extra_split_gallery": output / "e16_extra_split_gallery.png",
        "support_domain_gap": output / "e16_support_domain_gap.png",
    }
    render_main_table(summary, paths["main_results_table"])
    render_delta_distribution(
        paired_rows, bootstrap, paths["pair_f1_delta_distribution"]
    )
    render_scene_gallery(
        similar,
        predictions,
        global_root,
        paths["similar_scene_gallery"],
        "B16 and M16 are nearly identical on these mechanically selected scenes",
    )
    render_scene_gallery(
        extra_split,
        predictions,
        global_root,
        paths["extra_split_gallery"],
        "M16 creates more planes without improving partition pair F1",
    )
    support_scene = render_support_gap(
        paired_rows,
        predictions,
        gt_manifest,
        global_root,
        summary["macro"]["B16"]["gt_support_assignment_rate"],
        summary["macro"]["B16"]["gt_plane_label_rate"],
        paths["support_domain_gap"],
    )
    manifest = {
        "registration_id": summary["registration_id"],
        "reporting_only": True,
        "scientific_outputs_modified": False,
        "source_summary_sha256": sha256_file(results / "summary.json"),
        "source_metrics_sha256": sha256_file(results / "per_scene_metrics.json"),
        "source_prediction_manifest_sha256": sha256_file(
            results / "prediction_manifest.json"
        ),
        "source_gt_manifest_sha256": sha256_file(
            results / "point_aligned_gt_manifest.json"
        ),
        "similar_scenes": [row["scene_id"] for row in similar],
        "extra_split_no_gain_scenes": [row["scene_id"] for row in extra_split],
        "support_gap_scene": support_scene,
        "figures": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
