"""Overlap-controlled audit of view-conditioned DUSt3R plane residuals.

The audit uses point-aligned Structured3D GT only to identify a physical plane
and a metric tangent coordinate for each observation.  Within each GT plane,
metric tangent coordinates are quantized into spatial cells.  Only cells
observed by at least two connected views are retained.

Two held-out residual models are compared:

1. spatial-cell effects only;
2. spatial-cell effects plus a regularized source-view effect.

The second model can only improve consistently when source-view provenance
explains residual variation beyond which part of a plane was observed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


OUTPUT_SCHEMA_VERSION = 1


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("points must have shape (N,3) with N >= 3")
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(points), 1)
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return normal, -float(normal @ center)


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    axis = np.zeros((3,), dtype=np.float64)
    axis[int(np.argmin(np.abs(normal)))] = 1.0
    first = np.cross(normal, axis)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.cross(normal, first)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    return first, second


def packed_cell_ids(metric_points: np.ndarray, normal: np.ndarray, cell_size_m: float) -> np.ndarray:
    first, second = tangent_basis(normal)
    coordinates = np.column_stack((metric_points @ first, metric_points @ second))
    cells = np.floor(coordinates / float(cell_size_m)).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def largest_connected_view_component(
    cell_ids: np.ndarray,
    view_indices: np.ndarray,
    min_cell_view_points: int,
) -> np.ndarray:
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    view_indices = np.asarray(view_indices, dtype=np.int64)
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for cell_id, view_index in zip(cell_ids, view_indices):
        pair_counts[(int(cell_id), int(view_index))] += 1

    cell_views: dict[int, list[int]] = defaultdict(list)
    for (cell_id, view_index), count in pair_counts.items():
        if count >= int(min_cell_view_points):
            cell_views[cell_id].append(view_index)

    graph: dict[int, set[int]] = defaultdict(set)
    for views in cell_views.values():
        unique_views = sorted(set(views))
        if len(unique_views) < 2:
            continue
        for first in unique_views:
            graph[first].update(second for second in unique_views if second != first)

    components = []
    unseen = set(graph)
    while unseen:
        start = min(unseen)
        queue = deque([start])
        component = set()
        while queue:
            value = queue.popleft()
            if value in component:
                continue
            component.add(value)
            unseen.discard(value)
            queue.extend(graph[value] - component)
        components.append(component)
    if not components:
        return np.empty((0,), dtype=np.int64)
    components.sort(key=lambda values: (-len(values), tuple(sorted(values))))
    return np.asarray(sorted(components[0]), dtype=np.int64)


def fit_cell_only(
    values: np.ndarray,
    cell_ids: np.ndarray,
) -> dict[int, float]:
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for value, cell_id in zip(values, cell_ids):
        sums[int(cell_id)] += float(value)
        counts[int(cell_id)] += 1
    return {cell_id: sums[cell_id] / counts[cell_id] for cell_id in counts}


def fit_cell_plus_view(
    values: np.ndarray,
    cell_ids: np.ndarray,
    view_indices: np.ndarray,
    view_ridge_count: float,
    iterations: int = 30,
) -> tuple[dict[int, float], dict[int, float]]:
    values = np.asarray(values, dtype=np.float64)
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    view_indices = np.asarray(view_indices, dtype=np.int64)
    cells = np.unique(cell_ids)
    views = np.unique(view_indices)
    view_effect = {int(view): 0.0 for view in views}
    cell_effect = fit_cell_only(values, cell_ids)

    for _ in range(int(iterations)):
        for cell in cells:
            mask = cell_ids == cell
            correction = np.asarray([view_effect[int(view)] for view in view_indices[mask]])
            cell_effect[int(cell)] = float(np.mean(values[mask] - correction))

        for view in views:
            mask = view_indices == view
            correction = np.asarray([cell_effect[int(cell)] for cell in cell_ids[mask]])
            numerator = float(np.sum(values[mask] - correction))
            denominator = float(mask.sum()) + float(view_ridge_count)
            view_effect[int(view)] = numerator / max(denominator, 1e-12)

        counts = np.asarray([(view_indices == view).sum() for view in views], dtype=np.float64)
        effects = np.asarray([view_effect[int(view)] for view in views], dtype=np.float64)
        weighted_mean = float(np.sum(counts * effects) / max(np.sum(counts), 1.0))
        for view in views:
            view_effect[int(view)] -= weighted_mean
        for cell in cells:
            cell_effect[int(cell)] += weighted_mean

    return cell_effect, view_effect


def _predict(
    cell_effect: dict[int, float],
    view_effect: dict[int, float] | None,
    cell_ids: np.ndarray,
    view_indices: np.ndarray,
) -> np.ndarray:
    prediction = np.asarray([cell_effect[int(cell)] for cell in cell_ids], dtype=np.float64)
    if view_effect is not None:
        prediction += np.asarray([view_effect[int(view)] for view in view_indices])
    return prediction


def evaluate_plane(
    points: np.ndarray,
    metric_points: np.ndarray,
    pixel_xy: np.ndarray,
    view_indices: np.ndarray,
    metric_normal: np.ndarray,
    *,
    cell_size_m: float,
    min_cell_view_points: int,
    min_connected_views: int,
    view_ridge_count: float,
    return_details: bool = False,
) -> dict | None:
    points = np.asarray(points, dtype=np.float64)
    metric_points = np.asarray(metric_points, dtype=np.float64)
    pixel_xy = np.asarray(pixel_xy, dtype=np.int64)
    view_indices = np.asarray(view_indices, dtype=np.int64)
    cell_ids = packed_cell_ids(metric_points, metric_normal, cell_size_m)
    component = largest_connected_view_component(
        cell_ids, view_indices, min_cell_view_points)
    if len(component) < int(min_connected_views):
        return None

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for cell_id, view_index in zip(cell_ids, view_indices):
        pair_counts[(int(cell_id), int(view_index))] += 1
    eligible_cells = {
        int(cell_id)
        for cell_id in np.unique(cell_ids)
        if sum(
            pair_counts[(int(cell_id), int(view))] >= int(min_cell_view_points)
            for view in component
        )
        >= 2
    }
    keep = np.isin(view_indices, component) & np.isin(cell_ids, list(eligible_cells))
    if int(keep.sum()) < 3:
        return None
    points = points[keep]
    metric_points = metric_points[keep]
    pixel_xy = pixel_xy[keep]
    view_indices = view_indices[keep]
    cell_ids = cell_ids[keep]

    pixel_hash = (
        pixel_xy[:, 0] * np.int64(73856093)
        ^ pixel_xy[:, 1] * np.int64(19349663)
    )
    train = (pixel_hash & 1) == 0
    test = ~train
    if int(train.sum()) < 3 or int(test.sum()) < 3:
        return None

    train_pairs = set(zip(cell_ids[train].tolist(), view_indices[train].tolist()))
    test_supported = np.asarray(
        [
            (int(cell), int(view)) in train_pairs
            for cell, view in zip(cell_ids[test], view_indices[test])
        ],
        dtype=bool,
    )
    if int(test_supported.sum()) < 3:
        return None

    normal, offset = fit_plane(points[train])
    train_values = points[train] @ normal + offset
    test_values = (points[test] @ normal + offset)[test_supported]
    train_cells = cell_ids[train]
    test_cells = cell_ids[test][test_supported]
    train_views = view_indices[train]
    test_views = view_indices[test][test_supported]

    base_cell_effect = fit_cell_only(train_values, train_cells)
    full_cell_effect, view_effect = fit_cell_plus_view(
        train_values,
        train_cells,
        train_views,
        view_ridge_count,
    )
    known = np.asarray(
        [
            int(cell) in base_cell_effect
            and int(cell) in full_cell_effect
            and int(view) in view_effect
            for cell, view in zip(test_cells, test_views)
        ],
        dtype=bool,
    )
    if int(known.sum()) < 3:
        return None
    test_values = test_values[known]
    test_cells = test_cells[known]
    test_views = test_views[known]

    base_prediction = _predict(base_cell_effect, None, test_cells, test_views)
    full_prediction = _predict(full_cell_effect, view_effect, test_cells, test_views)
    base_error = test_values - base_prediction
    full_error = test_values - full_prediction
    base_rmse = float(np.sqrt(np.mean(base_error**2)))
    full_rmse = float(np.sqrt(np.mean(full_error**2)))
    ratio = full_rmse / max(base_rmse, 1e-12)
    view_values = np.asarray(list(view_effect.values()), dtype=np.float64)
    result = {
        "cell_size_m": float(cell_size_m),
        "connected_view_count": int(len(component)),
        "connected_views": [int(value) for value in component],
        "overlap_cell_count": int(len(np.unique(test_cells))),
        "train_point_count": int(train.sum()),
        "test_point_count": int(len(test_values)),
        "cell_only_rmse": base_rmse,
        "cell_plus_view_rmse": full_rmse,
        "rmse_ratio": ratio,
        "incremental_variance_explained": float(1.0 - ratio**2),
        "view_effect_std": float(view_values.std()),
        "view_effect_range": float(np.ptp(view_values)),
        "view_effects": {str(key): float(value) for key, value in view_effect.items()},
    }
    if return_details:
        first, second = tangent_basis(normal)
        result["details"] = {
            "local_xyz": np.column_stack(
                (points @ first, points @ second, points @ normal + offset)
            ),
            "view_indices": view_indices,
            "test_views": test_views,
            "cell_only_error": base_error,
            "cell_plus_view_error": full_error,
        }
    return result


def _artifact_path(root: Path, item: dict, name: str) -> Path:
    artifact = item["artifacts"][name]
    reference = artifact["path"] if isinstance(artifact, dict) else artifact
    return root / reference


def load_plane_data(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as data:
        points = data["points"].astype(np.float64)
        metric_points = data["metric_points_world_m"].astype(np.float64)
        metric_valid = data["metric_valid_mask"].astype(bool)
        pixel_xy = data["pixel_xy"].astype(np.int64)
        view_indices = data["view_indices"].astype(np.int64)
        labels = data["point_plane_ids"].astype(np.int64)
        plane_ids = data["plane_ids"].astype(np.int64)
        source_plane_ids = data["source_gt_plane_ids"].astype(np.int64)
        metric_normals = data["structured3d_world_plane_normals"].astype(np.float64)

    rows = []
    for index, plane_id in enumerate(plane_ids):
        mask = metric_valid & (labels == int(plane_id))
        if int(mask.sum()) < 3:
            continue
        rows.append(
            {
                "plane_id": int(plane_id),
                "source_gt_plane_id": int(source_plane_ids[index]),
                "points": points[mask],
                "metric_points": metric_points[mask],
                "pixel_xy": pixel_xy[mask],
                "view_indices": view_indices[mask],
                "metric_normal": metric_normals[index],
            }
        )
    return rows


def _scene_name(item: dict, gt_path: Path) -> str:
    for key in ("item_id", "case_id", "name", "scene_key"):
        if key in item:
            return str(item[key])
    stem = gt_path.stem
    suffix = "_point_aligned_plane_gt"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def summarize(rows: list[dict], cell_sizes: list[float]) -> dict:
    by_size = {}
    for cell_size in cell_sizes:
        selected = [row for row in rows if row["cell_size_m"] == float(cell_size)]
        ratios = np.asarray([row["rmse_ratio"] for row in selected], dtype=np.float64)
        if not len(ratios):
            by_size[str(cell_size)] = {"plane_count": 0}
            continue
        scene_ratios = []
        for scene in sorted({row["scene"] for row in selected}):
            values = [row["rmse_ratio"] for row in selected if row["scene"] == scene]
            scene_ratios.append(float(np.mean(values)))
        by_size[str(cell_size)] = {
            "plane_count": int(len(selected)),
            "scene_count": int(len(scene_ratios)),
            "median_rmse_ratio": float(np.median(ratios)),
            "mean_rmse_ratio": float(np.mean(ratios)),
            "median_incremental_variance_explained": float(
                np.median([row["incremental_variance_explained"] for row in selected])
            ),
            "plane_fraction_rmse_improves": float(np.mean(ratios < 1.0)),
            "plane_fraction_rmse_improves_5pct": float(np.mean(ratios <= 0.95)),
            "plane_fraction_rmse_improves_10pct": float(np.mean(ratios <= 0.90)),
            "scene_macro_mean_rmse_ratio": float(np.mean(scene_ratios)),
            "scene_macro_median_rmse_ratio": float(np.median(scene_ratios)),
        }
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "model_comparison": "spatial_cell_effects_vs_spatial_cell_plus_view_effects",
        "by_cell_size_m": by_size,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "scene",
        "plane_id",
        "source_gt_plane_id",
        "cell_size_m",
        "connected_view_count",
        "overlap_cell_count",
        "train_point_count",
        "test_point_count",
        "cell_only_rmse",
        "cell_plus_view_rmse",
        "rmse_ratio",
        "incremental_variance_explained",
        "view_effect_std",
        "view_effect_range",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def select_visual_rows(rows: list[dict], cell_size: float, count_each: int) -> list[dict]:
    selected = [row for row in rows if row["cell_size_m"] == float(cell_size)]
    selected.sort(key=lambda value: value["rmse_ratio"])
    if not selected:
        return []
    count_each = max(int(count_each), 1)
    middle = len(selected) // 2
    candidates = (
        selected[:count_each]
        + selected[max(0, middle - count_each // 2) : middle + math.ceil(count_each / 2)]
        + selected[-count_each:]
    )
    unique = {}
    for row in candidates:
        unique[(row["scene"], row["plane_id"], row["cell_size_m"])] = row
    return list(unique.values())


def render_visual(path: Path, title: str, result: dict, max_points: int = 50000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    details = result["details"]
    local_xyz = details["local_xyz"]
    views = details["view_indices"]
    if len(local_xyz) > int(max_points):
        indices = np.linspace(0, len(local_xyz) - 1, int(max_points), dtype=np.int64)
        local_xyz = local_xyz[indices]
        views = views[indices]

    figure = plt.figure(figsize=(15, 5.6), dpi=150)
    grid = figure.add_gridspec(1, 3, wspace=0.28)
    axis3d = figure.add_subplot(grid[0, 0], projection="3d")
    axis_base = figure.add_subplot(grid[0, 1])
    axis_full = figure.add_subplot(grid[0, 2])
    colors = plt.get_cmap("tab10")

    for index, view in enumerate(np.unique(views)):
        mask = views == view
        axis3d.scatter(
            local_xyz[mask, 0],
            local_xyz[mask, 1],
            local_xyz[mask, 2],
            s=0.4,
            alpha=0.38,
            color=colors(index % 10),
            label=f"view {int(view)}",
            rasterized=True,
        )
    axis3d.set_title("Aligned observations colored by source view")
    axis3d.set_xlabel("plane tangent u")
    axis3d.set_ylabel("plane tangent v")
    axis3d.set_zlabel("canonical residual")
    axis3d.legend(markerscale=8, fontsize=7, loc="best")

    test_views = details["test_views"]
    for index, view in enumerate(np.unique(test_views)):
        mask = test_views == view
        axis_base.hist(
            details["cell_only_error"][mask],
            bins=80,
            histtype="step",
            density=True,
            color=colors(index % 10),
            label=f"view {int(view)}",
        )
        axis_full.hist(
            details["cell_plus_view_error"][mask],
            bins=80,
            histtype="step",
            density=True,
            color=colors(index % 10),
            label=f"view {int(view)}",
        )
    axis_base.set_title("After spatial-cell correction only")
    axis_full.set_title("After spatial-cell + view correction")
    for axis in (axis_base, axis_full):
        axis.set_xlabel("held-out residual")
        axis.set_ylabel("density")
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{title}\nRMSE ratio={result['rmse_ratio']:.3f}, "
        f"incremental explained variance={result['incremental_variance_explained']:.1%}",
        fontsize=13,
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether source-view effects explain DUSt3R plane thickness after spatial control."
    )
    parser.add_argument("--batch_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cell_sizes_m", default="0.05,0.10,0.20")
    parser.add_argument("--min_cell_view_points", type=int, default=8)
    parser.add_argument("--min_connected_views", type=int, default=3)
    parser.add_argument("--view_ridge_count", type=float, default=64.0)
    parser.add_argument("--visual_cell_size_m", type=float, default=0.10)
    parser.add_argument("--visual_count_each", type=int, default=2)
    args = parser.parse_args()

    batch_path = Path(args.batch_json).resolve()
    root = batch_path
    while root.parent != root and not (root / ".git").exists():
        root = root.parent
    if not (root / ".git").exists():
        raise ValueError("Could not infer repository root from batch_json")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    visual_dir = output_dir / "visualizations"
    visual_dir.mkdir()

    cell_sizes = [float(value) for value in args.cell_sizes_m.split(",") if value.strip()]
    if not cell_sizes or any(value <= 0 for value in cell_sizes):
        raise ValueError("cell_sizes_m must contain positive comma-separated values")
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    rows = []
    scene_inputs = {}
    for item in batch["items"]:
        gt_path = _artifact_path(root, item, "point_aligned_gt")
        scene = _scene_name(item, gt_path)
        scene_inputs[scene] = gt_path
        for plane in load_plane_data(gt_path):
            for cell_size in cell_sizes:
                result = evaluate_plane(
                    plane["points"],
                    plane["metric_points"],
                    plane["pixel_xy"],
                    plane["view_indices"],
                    plane["metric_normal"],
                    cell_size_m=cell_size,
                    min_cell_view_points=args.min_cell_view_points,
                    min_connected_views=args.min_connected_views,
                    view_ridge_count=args.view_ridge_count,
                )
                if result is None:
                    continue
                result.update(
                    {
                        "scene": scene,
                        "plane_id": plane["plane_id"],
                        "source_gt_plane_id": plane["source_gt_plane_id"],
                        "point_aligned_gt": str(gt_path),
                    }
                )
                rows.append(result)

    summary = summarize(rows, cell_sizes)
    summary["config"] = {
        "batch_json": str(batch_path),
        "cell_sizes_m": cell_sizes,
        "min_cell_view_points": int(args.min_cell_view_points),
        "min_connected_views": int(args.min_connected_views),
        "view_ridge_count": float(args.view_ridge_count),
        "train_test_split": "parity of deterministic source-pixel hash",
        "spatial_coordinate": "Structured3D metric world plane tangent coordinates",
    }
    summary["outputs"] = {
        "per_plane_csv": str(output_dir / "per_plane_metrics.csv"),
        "per_plane_json": str(output_dir / "per_plane_metrics.json"),
        "summary_json": str(output_dir / "summary.json"),
        "visualizations": str(visual_dir),
    }
    write_csv(output_dir / "per_plane_metrics.csv", rows)
    (output_dir / "per_plane_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")

    visual_rows = select_visual_rows(
        rows, args.visual_cell_size_m, args.visual_count_each)
    visual_manifest = []
    for row in visual_rows:
        planes = load_plane_data(scene_inputs[row["scene"]])
        plane = next(value for value in planes if value["plane_id"] == row["plane_id"])
        detailed = evaluate_plane(
            plane["points"],
            plane["metric_points"],
            plane["pixel_xy"],
            plane["view_indices"],
            plane["metric_normal"],
            cell_size_m=row["cell_size_m"],
            min_cell_view_points=args.min_cell_view_points,
            min_connected_views=args.min_connected_views,
            view_ridge_count=args.view_ridge_count,
            return_details=True,
        )
        filename = (
            f"{row['scene']}_plane_{row['plane_id']:03d}"
            f"_cell_{row['cell_size_m']:.2f}m.png"
        )
        render_visual(
            visual_dir / filename,
            f"{row['scene']} | plane {row['plane_id']} | cell {row['cell_size_m']:.2f} m",
            detailed,
        )
        visual_manifest.append(
            {
                "scene": row["scene"],
                "plane_id": row["plane_id"],
                "cell_size_m": row["cell_size_m"],
                "rmse_ratio": row["rmse_ratio"],
                "path": str(visual_dir / filename),
            }
        )
    summary["visualization_manifest"] = visual_manifest
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
