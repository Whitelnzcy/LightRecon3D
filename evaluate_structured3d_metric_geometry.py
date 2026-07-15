"""Evaluate cached reconstructions against exact Structured3D metric rays.

Predictions are joined to GT only by ``(alignment_view_index, x, y)``.  A
single global Sim(3) removes the arbitrary DUSt3R gauge before correspondence
and GT-plane residuals are reported.  The reference prediction's transform is
also reused for corrected methods so gauge-invariant and fixed-gauge changes
are both visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_keys(view_indices, pixel_xy):
    view_indices = np.asarray(view_indices, dtype=np.int64).reshape(-1)
    pixel_xy = np.asarray(pixel_xy, dtype=np.int64)
    if pixel_xy.shape != (len(view_indices), 2):
        raise ValueError("pixel_xy must match view_indices and have shape (N,2)")
    if (view_indices < 0).any() or (pixel_xy < 0).any() or (pixel_xy >= (1 << 21)).any():
        raise ValueError("View indices and pointmap pixels must be in [0, 2^21)")
    return (view_indices << 42) | (pixel_xy[:, 0] << 21) | pixel_xy[:, 1]


def prediction_provenance(raw):
    if "alignment_view_indices" in raw and "pointmap_pixel_xy" in raw:
        return raw["alignment_view_indices"], raw["pointmap_pixel_xy"]
    if "source_views" in raw and "pixel_xy" in raw:
        return raw["source_views"], raw["pixel_xy"]
    if "view_indices" in raw and "pixel_xy" in raw:
        return raw["view_indices"], raw["pixel_xy"]
    raise ValueError(
        "Prediction needs canonical view/pixel provenance; nearest-XYZ joins are forbidden"
    )


def load_metric_gt(path):
    with np.load(path, allow_pickle=False) as raw:
        required = {
            "metric_points_world_m", "metric_valid_mask", "view_indices", "pixel_xy",
            "point_plane_ids", "structured3d_world_plane_normals",
            "structured3d_world_plane_offsets_m",
        }
        missing = sorted(required - set(raw.files))
        if missing:
            raise ValueError(f"Metric GT is missing fields: {missing}")
        points = raw["metric_points_world_m"].astype(np.float64)
        views = raw["view_indices"].astype(np.int32).reshape(-1)
        pixels = raw["pixel_xy"].astype(np.int32)
        plane_ids = raw["point_plane_ids"].astype(np.int32).reshape(-1)
        valid = raw["metric_valid_mask"].astype(bool).reshape(-1)
        valid &= np.isfinite(points).all(axis=1) & (plane_ids >= 0)
        normals = raw["structured3d_world_plane_normals"].astype(np.float64)
        offsets = raw["structured3d_world_plane_offsets_m"].astype(np.float64)
        source_cache_sha = (
            str(raw["source_global_cloud_sha256"].item())
            if "source_global_cloud_sha256" in raw else ""
        )
    keys = pack_keys(views[valid], pixels[valid])
    if len(np.unique(keys)) != len(keys):
        raise ValueError("Metric GT contains duplicate canonical view/pixel keys")
    return {
        "points": points[valid],
        "views": views[valid],
        "pixels": pixels[valid],
        "plane_ids": plane_ids[valid],
        "plane_normals": normals,
        "plane_offsets": offsets,
        "keys": keys,
        "source_cache_sha256": source_cache_sha,
    }


def load_prediction(path, name):
    with np.load(path, allow_pickle=False) as raw:
        if "points" not in raw:
            raise ValueError(f"Prediction {path} has no points array")
        points = raw["points"].astype(np.float64)
        views, pixels = prediction_provenance(raw)
        views = np.asarray(views, dtype=np.int32).reshape(-1)
        pixels = np.asarray(pixels, dtype=np.int32)
        method = str(raw["method"].item()) if "method" in raw else name
    if points.shape != (len(views), 3) or pixels.shape != (len(views), 2):
        raise ValueError(f"Prediction {path} arrays have inconsistent shapes")
    valid = np.isfinite(points).all(axis=1) & (np.max(np.abs(points), axis=1) < 1e5)
    points, views, pixels = points[valid], views[valid], pixels[valid]
    keys = pack_keys(views, pixels)
    if len(np.unique(keys)) != len(keys):
        raise ValueError(f"Prediction {path} contains duplicate canonical view/pixel keys")
    return {
        "name": name,
        "method": method,
        "path": str(path),
        "sha256": file_sha256(path),
        "points": points,
        "views": views,
        "pixels": pixels,
        "keys": keys,
    }


def match_prediction_to_gt(prediction, gt):
    order = np.argsort(prediction["keys"])
    sorted_keys = prediction["keys"][order]
    positions = np.searchsorted(sorted_keys, gt["keys"])
    matched = positions < len(sorted_keys)
    matched[matched] &= sorted_keys[positions[matched]] == gt["keys"][matched]
    return {
        "pred_points": prediction["points"][order[positions[matched]]],
        "gt_points": gt["points"][matched],
        "gt_views": gt["views"][matched],
        "gt_plane_ids": gt["plane_ids"][matched],
        "gt_mask": matched,
    }


def estimate_similarity(source, target):
    """Closed-form Umeyama map ``target ~= scale * R * source + t``."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Similarity inputs must have matching (N,3) shapes")
    if len(source) < 3:
        raise ValueError("At least three correspondences are required for Sim(3)")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if float(np.linalg.det(u @ vt)) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if variance <= 1e-15:
        raise ValueError("Prediction variance is too small for Sim(3) alignment")
    scale = float((singular * sign).sum() / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_similarity(points, transform):
    scale, rotation, translation = transform
    return scale * (np.asarray(points, dtype=np.float64) @ rotation.T) + translation


def trimmed_similarity(source, target, keep_quantile=0.9, iterations=3):
    keep = np.ones(len(source), dtype=bool)
    for _ in range(max(1, int(iterations))):
        transform = estimate_similarity(source[keep], target[keep])
        if float(keep_quantile) >= 1.0:
            break
        error = np.linalg.norm(apply_similarity(source, transform) - target, axis=1)
        threshold = float(np.quantile(error, float(keep_quantile)))
        updated = error <= threshold
        if int(updated.sum()) < 3 or np.array_equal(updated, keep):
            break
        keep = updated
    return estimate_similarity(source[keep], target[keep]), keep


def numeric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def rotation_angle_deg(rotation):
    cosine = np.clip((float(np.trace(rotation)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def transform_json(transform, inlier_count):
    scale, rotation, translation = transform
    return {
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "rotation_angle_deg": rotation_angle_deg(rotation),
        "translation_m": translation.tolist(),
        "alignment_inliers": int(inlier_count),
    }


def metric_rows(aligned, match, gt):
    point_error = np.linalg.norm(aligned - match["gt_points"], axis=1)
    normals = gt["plane_normals"][match["gt_plane_ids"]]
    offsets = gt["plane_offsets"][match["gt_plane_ids"]]
    plane_residual = np.abs(np.einsum("ij,ij->i", aligned, normals) + offsets)
    per_view = []
    for view_index in sorted(int(value) for value in np.unique(match["gt_views"])):
        mask = match["gt_views"] == view_index
        per_view.append({
            "view_index": view_index,
            "points": int(mask.sum()),
            "correspondence_error_m": numeric_summary(point_error[mask]),
            "gt_plane_residual_m": numeric_summary(plane_residual[mask]),
        })
    return {
        "correspondence_error_m": numeric_summary(point_error),
        "gt_plane_residual_m": numeric_summary(plane_residual),
        "per_view": per_view,
    }


def evaluate_predictions(gt_path, prediction_specs, reference_name, trim_quantile=0.9):
    gt = load_metric_gt(gt_path)
    predictions = [load_prediction(path, name) for name, path in prediction_specs]
    if len({row["name"] for row in predictions}) != len(predictions):
        raise ValueError("Prediction names must be unique")
    matches = {row["name"]: match_prediction_to_gt(row, gt) for row in predictions}
    for name, match in matches.items():
        if len(match["pred_points"]) < 3:
            raise ValueError(f"Prediction {name} has fewer than three exact GT matches")
    if reference_name not in matches:
        raise ValueError(f"Reference prediction {reference_name!r} is not present")
    reference_prediction = next(
        row for row in predictions if row["name"] == reference_name
    )
    if (
        gt["source_cache_sha256"]
        and reference_prediction["sha256"] != gt["source_cache_sha256"]
    ):
        raise ValueError(
            "Reference prediction checksum does not match the metric GT source cache"
        )

    reference_transform, reference_keep = trimmed_similarity(
        matches[reference_name]["pred_points"],
        matches[reference_name]["gt_points"],
        keep_quantile=trim_quantile,
    )
    methods = []
    for prediction in predictions:
        match = matches[prediction["name"]]
        transform, keep = trimmed_similarity(
            match["pred_points"], match["gt_points"], keep_quantile=trim_quantile
        )
        independent = metric_rows(apply_similarity(match["pred_points"], transform), match, gt)
        shared = metric_rows(
            apply_similarity(match["pred_points"], reference_transform), match, gt
        )
        methods.append({
            "name": prediction["name"],
            "method": prediction["method"],
            "prediction": prediction["path"],
            "prediction_sha256": prediction["sha256"],
            "matched_metric_points": int(len(match["pred_points"])),
            "metric_gt_coverage": float(len(match["pred_points"]) / len(gt["points"])),
            "independent_similarity": transform_json(transform, int(keep.sum())),
            "independent_alignment_metrics": independent,
            "shared_reference_alignment_metrics": shared,
        })
    reference_row = next(row for row in methods if row["name"] == reference_name)
    reference_rmse = reference_row["independent_alignment_metrics"]["correspondence_error_m"]["rmse"]
    reference_plane = reference_row["independent_alignment_metrics"]["gt_plane_residual_m"]["mean"]
    for row in methods:
        row["delta_vs_reference"] = {
            "independent_correspondence_rmse_m": (
                row["independent_alignment_metrics"]["correspondence_error_m"]["rmse"]
                - reference_rmse
            ),
            "independent_gt_plane_mean_residual_m": (
                row["independent_alignment_metrics"]["gt_plane_residual_m"]["mean"]
                - reference_plane
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_gt": str(gt_path),
        "metric_gt_sha256": file_sha256(gt_path),
        "source_global_cloud_sha256": gt["source_cache_sha256"],
        "metric_gt_points": int(len(gt["points"])),
        "reference_prediction": reference_name,
        "alignment": {
            "type": "global_sim3_umeyama_iterative_trim",
            "keep_quantile": float(trim_quantile),
            "iterations": 3,
            "reference_similarity": transform_json(
                reference_transform, int(reference_keep.sum())
            ),
        },
        "methods": methods,
        "limitations": [
            "GT covers structural layout planes reconstructed by calibrated ray-plane intersection.",
            "Furniture and other non-planar scene geometry are not evaluated without depth.png.",
            "Camera pose error is not reported because the retained DUSt3R cache did not store optimized poses.",
        ],
    }


def parse_prediction(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--prediction expects NAME=NPZ_PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Prediction name and path must be non-empty")
    return name.strip(), Path(path.strip())


def main():
    parser = argparse.ArgumentParser("Evaluate metric Structured3D structural geometry")
    parser.add_argument("--metric_gt_npz", required=True, type=Path)
    parser.add_argument("--prediction", action="append", required=True, type=parse_prediction)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--alignment_trim_quantile", type=float, default=0.9)
    parser.add_argument("--output_json", required=True, type=Path)
    args = parser.parse_args()
    if not 0.5 <= args.alignment_trim_quantile <= 1.0:
        parser.error("--alignment_trim_quantile must be in [0.5, 1.0]")
    result = evaluate_predictions(
        args.metric_gt_npz,
        args.prediction,
        args.reference,
        trim_quantile=args.alignment_trim_quantile,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json),
        "metric_gt_points": result["metric_gt_points"],
        "methods": len(result["methods"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
