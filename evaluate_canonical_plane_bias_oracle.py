"""GT-identity oracle for provenance-aware canonical plane estimation.

The script keeps the DUSt3R pointmaps frozen.  A single scene-level Sim(3) is
used only for metric evaluation.  Given GT plane identity, it compares pooled
SVD, view-balanced SVD, and alternating canonical-plane fits with regularized
per-view observation offsets and reliability.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_structured3d_metric_geometry import apply_similarity, trimmed_similarity


OUTPUT_SCHEMA_VERSION = 1


def weighted_fit_plane(
    points: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("points must have shape (N,3) with N >= 3")
    if weights is None:
        weights = np.ones((len(points),), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(weights) != len(points):
        raise ValueError("weights must match points")
    weights = np.clip(weights, 1e-12, None)
    weights /= weights.sum()
    center = np.sum(points * weights[:, None], axis=0)
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return normal, -float(normal @ center)


def view_balanced_weights(
    view_indices: np.ndarray,
    confidence: np.ndarray | None = None,
    view_totals: dict[int, float] | None = None,
) -> np.ndarray:
    view_indices = np.asarray(view_indices, dtype=np.int64)
    if confidence is None:
        confidence = np.ones((len(view_indices),), dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    positive = confidence[confidence > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    confidence = np.clip(confidence / max(scale, 1e-12), 0.1, 10.0)
    weights = np.zeros((len(view_indices),), dtype=np.float64)
    views = np.unique(view_indices)
    for view in views:
        mask = view_indices == view
        local = confidence[mask]
        total = 1.0 if view_totals is None else float(view_totals[int(view)])
        weights[mask] = total * local / max(float(local.sum()), 1e-12)
    weights /= max(float(weights.sum()), 1e-12)
    return weights


def _weighted_view_mean(
    values: np.ndarray,
    weights: np.ndarray,
    view_indices: np.ndarray,
) -> dict[int, float]:
    result = {}
    for view in np.unique(view_indices):
        mask = view_indices == view
        result[int(view)] = float(
            np.sum(weights[mask] * values[mask])
            / max(float(weights[mask].sum()), 1e-12)
        )
    return result


def fit_canonical_view_bias(
    points: np.ndarray,
    view_indices: np.ndarray,
    confidence: np.ndarray,
    *,
    bias_ridge: float,
    use_reliability: bool,
    iterations: int = 20,
) -> dict:
    points = np.asarray(points, dtype=np.float64)
    view_indices = np.asarray(view_indices, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float64)
    views = np.unique(view_indices)
    view_totals = {int(view): 1.0 for view in views}
    weights = view_balanced_weights(view_indices, confidence, view_totals)
    normal, offset = weighted_fit_plane(points, weights)
    biases = {int(view): 0.0 for view in views}
    reliability = {int(view): 1.0 for view in views}
    sigmas = {int(view): 0.0 for view in views}

    for _ in range(int(iterations)):
        residual = points @ normal + offset
        raw_bias = _weighted_view_mean(residual, weights, view_indices)
        for view in views:
            support = float(weights[view_indices == view].sum())
            shrink = support / max(support + float(bias_ridge), 1e-12)
            biases[int(view)] = shrink * raw_bias[int(view)]

        view_mass = np.asarray(
            [weights[view_indices == view].sum() for view in views],
            dtype=np.float64,
        )
        bias_values = np.asarray([biases[int(view)] for view in views], dtype=np.float64)
        gauge = float(np.sum(view_mass * bias_values) / max(float(view_mass.sum()), 1e-12))
        for view in views:
            biases[int(view)] -= gauge
        offset += gauge

        bias_per_point = np.asarray([biases[int(view)] for view in view_indices])
        corrected_points = points - bias_per_point[:, None] * normal[None]
        updated_normal, updated_offset = weighted_fit_plane(corrected_points, weights)
        if float(updated_normal @ normal) < 0:
            updated_normal = -updated_normal
            updated_offset = -updated_offset
            for view in views:
                biases[int(view)] = -biases[int(view)]
        normal, offset = updated_normal, updated_offset

        adjusted = points @ normal + offset - np.asarray(
            [biases[int(view)] for view in view_indices]
        )
        if use_reliability:
            sigma_values = []
            for view in views:
                mask = view_indices == view
                local = adjusted[mask]
                median = float(np.median(local))
                sigma = 1.4826 * float(np.median(np.abs(local - median)))
                sigma = max(sigma, 1e-6)
                sigmas[int(view)] = sigma
                sigma_values.append(sigma)
            sigma_reference = max(float(np.median(sigma_values)), 1e-6)
            for view in views:
                reliability[int(view)] = float(
                    np.clip(sigma_reference / sigmas[int(view)], 0.25, 4.0)
                )
            weights = view_balanced_weights(
                view_indices, confidence, reliability)

            robust_scale = max(1.4826 * float(np.median(np.abs(adjusted))), 1e-6)
            huber = np.minimum(1.0, (2.5 * robust_scale) / np.maximum(np.abs(adjusted), 1e-12))
            weights *= huber
            weights /= max(float(weights.sum()), 1e-12)

    return {
        "normal": normal,
        "offset": float(offset),
        "view_biases": biases,
        "view_sigmas": sigmas,
        "view_reliability": reliability,
    }


def plane_parameter_error(
    normal: np.ndarray,
    offset: float,
    gt_normal: np.ndarray,
    gt_offset: float,
) -> tuple[float, float, np.ndarray, float]:
    normal = np.asarray(normal, dtype=np.float64)
    gt_normal = np.asarray(gt_normal, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    gt_normal /= max(float(np.linalg.norm(gt_normal)), 1e-12)
    if float(normal @ gt_normal) < 0:
        normal = -normal
        offset = -float(offset)
    cosine = float(np.clip(normal @ gt_normal, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    return angle, abs(float(offset) - float(gt_offset)), normal, float(offset)


def evaluate_model(
    name: str,
    model: dict,
    test_points: np.ndarray,
    test_views: np.ndarray,
    gt_normal: np.ndarray,
    gt_offset: float,
) -> dict:
    angle, offset_error, normal, offset = plane_parameter_error(
        model["normal"], model["offset"], gt_normal, gt_offset)
    canonical_residual = test_points @ normal + offset
    biases = model.get("view_biases", {})
    signed_bias = np.asarray([biases.get(int(view), 0.0) for view in test_views])
    if float(model["normal"] @ normal) < 0:
        signed_bias = -signed_bias
    corrected_residual = canonical_residual - signed_bias
    return {
        "method": name,
        "normal_error_deg": angle,
        "offset_error_m": offset_error,
        "heldout_canonical_rmse_m": float(
            np.sqrt(np.mean(canonical_residual**2))
        ),
        "heldout_bias_corrected_rmse_m": float(
            np.sqrt(np.mean(corrected_residual**2))
        ),
        "view_bias_std_m": float(np.std(list(biases.values()))) if biases else 0.0,
        "view_bias_range_m": float(np.ptp(list(biases.values()))) if biases else 0.0,
        "view_biases": {str(key): float(value) for key, value in biases.items()},
        "view_reliability": {
            str(key): float(value)
            for key, value in model.get("view_reliability", {}).items()
        },
    }


def _pooled_model(points: np.ndarray) -> dict:
    normal, offset = weighted_fit_plane(points)
    return {"normal": normal, "offset": offset}


def _view_balanced_model(
    points: np.ndarray,
    views: np.ndarray,
    confidence: np.ndarray,
) -> dict:
    weights = view_balanced_weights(views, confidence)
    normal, offset = weighted_fit_plane(points, weights)
    return {"normal": normal, "offset": offset}


def evaluate_scene(
    gt_path: Path,
    min_points_per_view: int,
    min_views: int,
    bias_ridge: float,
    trim_quantile: float,
) -> tuple[list[dict], dict]:
    with np.load(gt_path, allow_pickle=False) as data:
        points = data["points"].astype(np.float64)
        confidence = data["confidence"].astype(np.float64)
        views = data["view_indices"].astype(np.int64)
        pixels = data["pixel_xy"].astype(np.int64)
        labels = data["point_plane_ids"].astype(np.int64)
        metric_valid = data["metric_valid_mask"].astype(bool)
        metric_points = data["metric_points_world_m"].astype(np.float64)
        plane_ids = data["plane_ids"].astype(np.int64)
        source_plane_ids = data["source_gt_plane_ids"].astype(np.int64)
        gt_normals = data["structured3d_world_plane_normals"].astype(np.float64)
        gt_offsets = data["structured3d_world_plane_offsets_m"].astype(np.float64)

    transform, transform_keep = trimmed_similarity(
        points[metric_valid],
        metric_points[metric_valid],
        keep_quantile=trim_quantile,
        iterations=3,
    )
    aligned_points = apply_similarity(points, transform)
    scene_rows = []
    for plane_index, plane_id in enumerate(plane_ids):
        plane_mask = metric_valid & (labels == int(plane_id))
        eligible_views = []
        for view, count in zip(*np.unique(views[plane_mask], return_counts=True)):
            if int(count) >= int(min_points_per_view):
                eligible_views.append(int(view))
        if len(eligible_views) < int(min_views):
            continue
        selected = plane_mask & np.isin(views, eligible_views)
        local_points = aligned_points[selected]
        local_confidence = confidence[selected]
        local_views = views[selected]
        local_pixels = pixels[selected]
        pixel_hash = (
            local_pixels[:, 0] * np.int64(73856093)
            ^ local_pixels[:, 1] * np.int64(19349663)
        )
        train = (pixel_hash & 1) == 0
        test = ~train
        if int(train.sum()) < 3 or int(test.sum()) < 3:
            continue

        train_points = local_points[train]
        train_views = local_views[train]
        train_confidence = local_confidence[train]
        test_points = local_points[test]
        test_views = local_views[test]
        models = {
            "pooled_svd": _pooled_model(train_points),
            "view_balanced_svd": _view_balanced_model(
                train_points, train_views, train_confidence),
            "view_bias_free": fit_canonical_view_bias(
                train_points,
                train_views,
                train_confidence,
                bias_ridge=0.0,
                use_reliability=False,
            ),
            "view_bias_shrinkage": fit_canonical_view_bias(
                train_points,
                train_views,
                train_confidence,
                bias_ridge=bias_ridge,
                use_reliability=False,
            ),
            "view_bias_reliability": fit_canonical_view_bias(
                train_points,
                train_views,
                train_confidence,
                bias_ridge=bias_ridge,
                use_reliability=True,
            ),
        }
        for name, model in models.items():
            row = evaluate_model(
                name,
                model,
                test_points,
                test_views,
                gt_normals[plane_index],
                float(gt_offsets[plane_index]),
            )
            row.update(
                {
                    "plane_id": int(plane_id),
                    "source_gt_plane_id": int(source_plane_ids[plane_index]),
                    "view_count": int(len(eligible_views)),
                    "train_point_count": int(train.sum()),
                    "test_point_count": int(test.sum()),
                }
            )
            scene_rows.append(row)
    scale, rotation, translation = transform
    transform_row = {
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation_m": translation.tolist(),
        "trimmed_alignment_inliers": int(transform_keep.sum()),
        "metric_valid_points": int(metric_valid.sum()),
    }
    return scene_rows, transform_row


def summarize(rows: list[dict]) -> dict:
    methods = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        methods[method] = {
            "plane_count": len(selected),
            "scene_count": len({row["scene"] for row in selected}),
        }
        for metric in (
            "normal_error_deg",
            "offset_error_m",
            "heldout_canonical_rmse_m",
            "heldout_bias_corrected_rmse_m",
            "view_bias_std_m",
        ):
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            methods[method][f"mean_{metric}"] = float(np.mean(values))
            methods[method][f"median_{metric}"] = float(np.median(values))
    paired = {}
    lookup = {
        (row["scene"], row["plane_id"], row["method"]): row
        for row in rows
    }
    metrics = (
        "normal_error_deg",
        "offset_error_m",
        "heldout_canonical_rmse_m",
        "heldout_bias_corrected_rmse_m",
    )
    for candidate in sorted(method for method in methods if method != "pooled_svd"):
        contrast = {}
        keys = sorted(
            (scene, plane_id)
            for scene, plane_id, method in lookup
            if method == "pooled_svd"
            and (scene, plane_id, candidate) in lookup
        )
        for metric in metrics:
            deltas = np.asarray(
                [
                    lookup[(scene, plane_id, candidate)][metric]
                    - lookup[(scene, plane_id, "pooled_svd")][metric]
                    for scene, plane_id in keys
                ],
                dtype=np.float64,
            )
            scene_deltas = []
            for scene in sorted({scene for scene, _ in keys}):
                selected = [
                    delta
                    for delta, (row_scene, _) in zip(deltas, keys)
                    if row_scene == scene
                ]
                scene_deltas.append(float(np.mean(selected)))
            contrast[metric] = {
                "paired_planes": int(len(deltas)),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "win_fraction_lower_is_better": float(np.mean(deltas < 0)),
                "scene_macro_mean_delta": float(np.mean(scene_deltas)),
                "scene_win_fraction_lower_is_better": float(
                    np.mean(np.asarray(scene_deltas) < 0)
                ),
            }
        paired[f"{candidate}_vs_pooled_svd"] = contrast
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "oracle_identity": True,
        "pointmaps_modified": False,
        "methods": methods,
        "paired_contrasts": paired,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "scene",
        "method",
        "plane_id",
        "source_gt_plane_id",
        "view_count",
        "train_point_count",
        "test_point_count",
        "normal_error_deg",
        "offset_error_m",
        "heldout_canonical_rmse_m",
        "heldout_bias_corrected_rmse_m",
        "view_bias_std_m",
        "view_bias_range_m",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _artifact_path(root: Path, item: dict) -> Path:
    artifact = item["artifacts"]["point_aligned_gt"]
    reference = artifact["path"] if isinstance(artifact, dict) else artifact
    return root / reference


def _scene_name(item: dict, path: Path) -> str:
    for key in ("item_id", "case_id", "name", "scene_key"):
        if key in item:
            return str(item[key])
    suffix = "_point_aligned_plane_gt"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate canonical plane and per-view observation bias with GT identity."
    )
    parser.add_argument("--batch_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_points_per_view", type=int, default=256)
    parser.add_argument("--min_views", type=int, default=3)
    parser.add_argument("--bias_ridge", type=float, default=0.25)
    parser.add_argument("--alignment_trim_quantile", type=float, default=0.9)
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

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    rows = []
    scene_transforms = {}
    for item in batch["items"]:
        gt_path = _artifact_path(root, item)
        scene = _scene_name(item, gt_path)
        scene_rows, transform = evaluate_scene(
            gt_path,
            args.min_points_per_view,
            args.min_views,
            args.bias_ridge,
            args.alignment_trim_quantile,
        )
        for row in scene_rows:
            row["scene"] = scene
            row["point_aligned_gt"] = str(gt_path)
        rows.extend(scene_rows)
        scene_transforms[scene] = transform

    result = summarize(rows)
    result["config"] = {
        "batch_json": str(batch_path),
        "min_points_per_view": int(args.min_points_per_view),
        "min_views": int(args.min_views),
        "bias_ridge": float(args.bias_ridge),
        "alignment_trim_quantile": float(args.alignment_trim_quantile),
        "train_test_split": "parity of deterministic source-pixel hash",
        "metric_alignment": "one trimmed scene-level Sim(3), shared by all plane methods",
    }
    result["scene_similarity_transforms"] = scene_transforms
    result["limitations"] = [
        "GT plane identity is provided and no plane assignment is inferred.",
        "The scene-level metric Sim(3) is evaluation-only and uses GT correspondences.",
        "Zero-mean per-view bias fixes gauge but does not prove the canonical offset is unbiased.",
    ]
    result["outputs"] = {
        "summary_json": str(output_dir / "summary.json"),
        "per_plane_csv": str(output_dir / "per_plane_metrics.csv"),
        "per_plane_json": str(output_dir / "per_plane_metrics.json"),
    }
    write_csv(output_dir / "per_plane_metrics.csv", rows)
    (output_dir / "per_plane_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
