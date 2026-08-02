"""Test whether plane-view offsets have a predictable shared view component.

For every GT plane/view observation, the signed DUSt3R residual to the metric
GT plane is measured on interleaved train/test pixels.  A leave-one-plane-out
test then asks whether offsets from other planes in the same view predict the
held-out offset through a shared 3D view translation:

    b_vk ~= n_k^T t_v

This is an oracle-identity diagnostic.  It does not move the pointmap and does
not infer plane assignments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_structured3d_metric_geometry import apply_similarity, trimmed_similarity


OUTPUT_SCHEMA_VERSION = 1


def solve_view_translation(
    normals: np.ndarray,
    biases: np.ndarray,
    ridge: float,
) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    biases = np.asarray(biases, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3 or len(normals) != len(biases):
        raise ValueError("normals must be (N,3) and match biases")
    hessian = normals.T @ normals + float(ridge) * np.eye(3)
    gradient = normals.T @ biases
    try:
        return np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(hessian, gradient, rcond=None)[0]


def measure_plane_view_biases(
    gt_path: Path,
    min_points: int,
    trim_quantile: float,
) -> tuple[list[dict], dict]:
    with np.load(gt_path, allow_pickle=False) as data:
        points = data["points"].astype(np.float64)
        metric_points = data["metric_points_world_m"].astype(np.float64)
        metric_valid = data["metric_valid_mask"].astype(bool)
        pixels = data["pixel_xy"].astype(np.int64)
        views = data["view_indices"].astype(np.int64)
        labels = data["point_plane_ids"].astype(np.int64)
        plane_ids = data["plane_ids"].astype(np.int64)
        source_plane_ids = data["source_gt_plane_ids"].astype(np.int64)
        normals = data["structured3d_world_plane_normals"].astype(np.float64)
        offsets = data["structured3d_world_plane_offsets_m"].astype(np.float64)

    transform, keep = trimmed_similarity(
        points[metric_valid],
        metric_points[metric_valid],
        keep_quantile=trim_quantile,
        iterations=3,
    )
    aligned = apply_similarity(points, transform)
    rows = []
    for plane_index, plane_id in enumerate(plane_ids):
        plane_mask = metric_valid & (labels == int(plane_id))
        for view in np.unique(views[plane_mask]):
            mask = plane_mask & (views == int(view))
            if int(mask.sum()) < int(min_points):
                continue
            local_pixels = pixels[mask]
            pixel_hash = (
                local_pixels[:, 0] * np.int64(73856093)
                ^ local_pixels[:, 1] * np.int64(19349663)
            )
            train = (pixel_hash & 1) == 0
            test = ~train
            if int(train.sum()) < 3 or int(test.sum()) < 3:
                continue
            residual = (
                aligned[mask] @ normals[plane_index]
                + float(offsets[plane_index])
            )
            train_values = residual[train]
            test_values = residual[test]
            rows.append(
                {
                    "view_index": int(view),
                    "plane_id": int(plane_id),
                    "source_gt_plane_id": int(source_plane_ids[plane_index]),
                    "normal": normals[plane_index].tolist(),
                    "train_bias_m": float(np.median(train_values)),
                    "test_bias_m": float(np.median(test_values)),
                    "train_mean_bias_m": float(np.mean(train_values)),
                    "test_mean_bias_m": float(np.mean(test_values)),
                    "train_point_count": int(train.sum()),
                    "test_point_count": int(test.sum()),
                    "test_within_view_mad_m": float(
                        np.median(np.abs(test_values - np.median(test_values)))
                    ),
                }
            )
    scale, rotation, translation = transform
    return rows, {
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation_m": translation.tolist(),
        "trimmed_alignment_inliers": int(keep.sum()),
        "metric_valid_points": int(metric_valid.sum()),
    }


def leave_one_plane_out(rows: list[dict], ridge: float, min_other_planes: int) -> list[dict]:
    output = []
    for heldout in rows:
        same_view = [
            row
            for row in rows
            if row["scene"] == heldout["scene"]
            and row["view_index"] == heldout["view_index"]
            and row["plane_id"] != heldout["plane_id"]
        ]
        if len(same_view) < int(min_other_planes):
            continue
        normals = np.asarray([row["normal"] for row in same_view], dtype=np.float64)
        biases = np.asarray([row["train_bias_m"] for row in same_view], dtype=np.float64)
        translation = solve_view_translation(normals, biases, ridge)
        target_normal = np.asarray(heldout["normal"], dtype=np.float64)
        translation_prediction = float(target_normal @ translation)
        view_mean_prediction = float(np.mean(biases))

        same_plane_other_views = [
            row["train_bias_m"]
            for row in rows
            if row["scene"] == heldout["scene"]
            and row["plane_id"] == heldout["plane_id"]
            and row["view_index"] != heldout["view_index"]
        ]
        plane_mean_prediction = (
            float(np.mean(same_plane_other_views))
            if same_plane_other_views
            else 0.0
        )
        target = float(heldout["test_bias_m"])
        row = {
            **heldout,
            "other_plane_count": int(len(same_view)),
            "other_plane_normal_rank": int(np.linalg.matrix_rank(normals)),
            "shared_translation_x_m": float(translation[0]),
            "shared_translation_y_m": float(translation[1]),
            "shared_translation_z_m": float(translation[2]),
            "target_test_bias_m": target,
            "zero_prediction_m": 0.0,
            "view_mean_prediction_m": view_mean_prediction,
            "plane_mean_prediction_m": plane_mean_prediction,
            "shared_translation_prediction_m": translation_prediction,
        }
        for name in ("zero", "view_mean", "plane_mean", "shared_translation"):
            error = float(row[f"{name}_prediction_m"] - target)
            row[f"{name}_error_m"] = error
            row[f"{name}_absolute_error_m"] = abs(error)
        output.append(row)
    return output


def summarize(rows: list[dict]) -> dict:
    methods = {}
    for method in ("zero", "view_mean", "plane_mean", "shared_translation"):
        errors = np.asarray([row[f"{method}_error_m"] for row in rows], dtype=np.float64)
        absolute = np.abs(errors)
        scene_rmse = []
        for scene in sorted({row["scene"] for row in rows}):
            local = np.asarray(
                [
                    row[f"{method}_error_m"]
                    for row in rows
                    if row["scene"] == scene
                ],
                dtype=np.float64,
            )
            scene_rmse.append(float(np.sqrt(np.mean(local**2))))
        methods[method] = {
            "observation_count": int(len(errors)),
            "scene_count": int(len(scene_rmse)),
            "rmse_m": float(np.sqrt(np.mean(errors**2))),
            "mae_m": float(np.mean(absolute)),
            "median_absolute_error_m": float(np.median(absolute)),
            "scene_macro_mean_rmse_m": float(np.mean(scene_rmse)),
            "scene_macro_median_rmse_m": float(np.median(scene_rmse)),
        }
    target = np.asarray([row["target_test_bias_m"] for row in rows], dtype=np.float64)
    shared_error = np.asarray(
        [row["shared_translation_error_m"] for row in rows], dtype=np.float64)
    zero_error = -target
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "oracle_identity": True,
        "pointmaps_modified": False,
        "model": "leave-one-plane-out b_vk ~= n_k^T t_v",
        "methods": methods,
        "shared_translation_vs_zero": {
            "rmse_ratio": float(
                np.sqrt(np.mean(shared_error**2))
                / max(float(np.sqrt(np.mean(zero_error**2))), 1e-12)
            ),
            "absolute_error_win_fraction": float(
                np.mean(np.abs(shared_error) < np.abs(zero_error))
            ),
            "incremental_variance_explained": float(
                1.0
                - np.mean(shared_error**2)
                / max(float(np.mean(zero_error**2)), 1e-24)
            ),
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "scene",
        "view_index",
        "plane_id",
        "source_gt_plane_id",
        "train_point_count",
        "test_point_count",
        "train_bias_m",
        "target_test_bias_m",
        "test_within_view_mad_m",
        "other_plane_count",
        "other_plane_normal_rank",
        "zero_prediction_m",
        "view_mean_prediction_m",
        "plane_mean_prediction_m",
        "shared_translation_prediction_m",
        "zero_absolute_error_m",
        "view_mean_absolute_error_m",
        "plane_mean_absolute_error_m",
        "shared_translation_absolute_error_m",
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
        description="Test whether view-plane biases have a shared view translation component."
    )
    parser.add_argument("--batch_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_points_per_plane_view", type=int, default=256)
    parser.add_argument("--min_other_planes", type=int, default=3)
    parser.add_argument("--translation_ridge", type=float, default=0.1)
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
    measured = []
    transforms = {}
    for item in batch["items"]:
        gt_path = _artifact_path(root, item)
        scene = _scene_name(item, gt_path)
        rows, transform = measure_plane_view_biases(
            gt_path,
            args.min_points_per_plane_view,
            args.alignment_trim_quantile,
        )
        for row in rows:
            row["scene"] = scene
            row["point_aligned_gt"] = str(gt_path)
        measured.extend(rows)
        transforms[scene] = transform

    evaluated = leave_one_plane_out(
        measured, args.translation_ridge, args.min_other_planes)
    result = summarize(evaluated)
    result["config"] = {
        "batch_json": str(batch_path),
        "min_points_per_plane_view": int(args.min_points_per_plane_view),
        "min_other_planes": int(args.min_other_planes),
        "translation_ridge": float(args.translation_ridge),
        "alignment_trim_quantile": float(args.alignment_trim_quantile),
        "bias_statistic": "median signed residual on interleaved train/test pixels",
    }
    result["scene_similarity_transforms"] = transforms
    result["limitations"] = [
        "GT plane identity and metric plane equations are used.",
        "A shared translation is a diagnostic latent factor and is not applied to the pointmaps.",
        "Repeated/parallel plane normals can make a view translation weakly identifiable.",
    ]
    result["outputs"] = {
        "summary_json": str(output_dir / "summary.json"),
        "plane_view_csv": str(output_dir / "plane_view_predictions.csv"),
        "plane_view_json": str(output_dir / "plane_view_predictions.json"),
    }
    write_csv(output_dir / "plane_view_predictions.csv", evaluated)
    (output_dir / "plane_view_predictions.json").write_text(
        json.dumps(evaluated, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
