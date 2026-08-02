"""Build prediction-only fragment graph features with GT identity labels.

The graph nodes are Stage2 direct-support fragments after DUSt3R global
alignment.  GT is used only to select the high-purity association audit
population and to label edges; every exported feature is computed from the
prediction support, RGB samples, pixel provenance, or previously computed
geometry/bias costs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from evaluate_bias_aware_fragment_association import (
    _artifact_path,
    _scene_name,
    exact_gt_join,
)
from evaluate_canonical_plane_bias_oracle import weighted_fit_plane


OUTPUT_SCHEMA_VERSION = 1


def _normalize_colors(colors: np.ndarray) -> np.ndarray:
    values = np.asarray(colors, dtype=np.float64)
    if values.size and float(np.nanmax(values)) > 1.5:
        values = values / 255.0
    return np.clip(values, 0.0, 1.0)


def _bbox_gap(first_min, first_max, second_min, second_max) -> float:
    first_min = np.asarray(first_min, dtype=np.float64)
    first_max = np.asarray(first_max, dtype=np.float64)
    second_min = np.asarray(second_min, dtype=np.float64)
    second_max = np.asarray(second_max, dtype=np.float64)
    axis_gap = np.maximum(
        np.maximum(first_min - second_max, second_min - first_max),
        0.0,
    )
    return float(np.linalg.norm(axis_gap))


def _bbox_iou(first: dict, second: dict) -> float:
    first_min = np.asarray(first["min"], dtype=np.float64)
    first_max = np.asarray(first["max"], dtype=np.float64)
    second_min = np.asarray(second["min"], dtype=np.float64)
    second_max = np.asarray(second["max"], dtype=np.float64)
    intersection = np.maximum(
        np.minimum(first_max, second_max) - np.maximum(first_min, second_min) + 1.0,
        0.0,
    )
    intersection_area = float(np.prod(intersection))
    first_area = float(np.prod(first_max - first_min + 1.0))
    second_area = float(np.prod(second_max - second_min + 1.0))
    return intersection_area / max(first_area + second_area - intersection_area, 1e-12)


def _robust_interval_relation(
    first_values: np.ndarray,
    second_values: np.ndarray,
    trim_quantile: float = 0.02,
) -> tuple[float, float]:
    """Return interval gap and overlap/min-span after robust endpoint trimming."""

    first_low, first_high = np.quantile(
        np.asarray(first_values, dtype=np.float64),
        [trim_quantile, 1.0 - trim_quantile],
    )
    second_low, second_high = np.quantile(
        np.asarray(second_values, dtype=np.float64),
        [trim_quantile, 1.0 - trim_quantile],
    )
    gap = max(first_low - second_high, second_low - first_high, 0.0)
    overlap = max(min(first_high, second_high) - max(first_low, second_low), 0.0)
    minimum_span = max(
        min(first_high - first_low, second_high - second_low),
        1e-12,
    )
    return float(gap), float(overlap / minimum_span)


def _tangent_support_relations(
    first_points: np.ndarray,
    second_points: np.ndarray,
    normal: np.ndarray,
) -> dict[str, float]:
    """Describe separation and overlap of two finite supports in a shared plane."""

    first_points = np.asarray(first_points, dtype=np.float64)
    second_points = np.asarray(second_points, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    centroid_delta = second_points.mean(axis=0) - first_points.mean(axis=0)
    primary = centroid_delta - float(centroid_delta @ normal) * normal
    primary_norm = float(np.linalg.norm(primary))
    if primary_norm <= 1e-10:
        centered = np.concatenate((first_points, second_points), axis=0)
        centered = centered - centered.mean(axis=0)
        centered = centered - np.outer(centered @ normal, normal)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        primary = vh[0] - float(vh[0] @ normal) * normal
        primary_norm = float(np.linalg.norm(primary))
    if primary_norm <= 1e-10:
        reference = np.asarray([1.0, 0.0, 0.0])
        if abs(float(reference @ normal)) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0])
        primary = np.cross(normal, reference)
        primary_norm = float(np.linalg.norm(primary))
    primary = primary / max(primary_norm, 1e-12)
    secondary = np.cross(normal, primary)
    secondary = secondary / max(float(np.linalg.norm(secondary)), 1e-12)
    primary_gap, primary_overlap = _robust_interval_relation(
        first_points @ primary, second_points @ primary
    )
    secondary_gap, secondary_overlap = _robust_interval_relation(
        first_points @ secondary, second_points @ secondary
    )
    return {
        "primary_gap": primary_gap,
        "primary_overlap": primary_overlap,
        "secondary_gap": secondary_gap,
        "secondary_overlap": secondary_overlap,
        "bbox_gap": math.hypot(primary_gap, secondary_gap),
        "bbox_overlap": primary_overlap * secondary_overlap,
    }


def build_scene_descriptors(
    support_path: Path,
    gt_path: Path,
    *,
    min_fragment_points: int,
    min_gt_purity: float,
    max_geometry_points: int,
    filter_by_gt: bool = True,
) -> tuple[dict[int, dict], float, dict]:
    with np.load(support_path, allow_pickle=True) as support:
        points = support["points"].astype(np.float64)
        colors_key = "original_colors" if "original_colors" in support else "colors"
        colors = _normalize_colors(support[colors_key])
        fragment_ids = support["point_plane_ids"].astype(np.int64)
        views = support["alignment_view_indices"].astype(np.int64)
        pixels = support["pointmap_pixel_xy"].astype(np.int64)
    with np.load(gt_path, allow_pickle=False) as gt:
        gt_views = gt["view_indices"].astype(np.int64)
        gt_pixels = gt["pixel_xy"].astype(np.int64)
        gt_labels = gt["point_plane_ids"].astype(np.int64)
    matched_labels = exact_gt_join(
        views, pixels, gt_views, gt_pixels, gt_labels
    )
    center = np.median(points, axis=0)
    scene_scale = max(
        float(np.median(np.linalg.norm(points - center, axis=1))),
        1e-8,
    )
    descriptors = {}
    rejected_impure = 0
    rejected_small = 0
    for fragment_id in np.unique(fragment_ids[fragment_ids >= 0]):
        mask = fragment_ids == int(fragment_id)
        local_gt = matched_labels[mask]
        valid_gt = local_gt >= 0
        support_count_for_filter = int(valid_gt.sum()) if filter_by_gt else int(mask.sum())
        if support_count_for_filter < int(min_fragment_points):
            rejected_small += 1
            continue
        if valid_gt.any():
            values, counts = np.unique(local_gt[valid_gt], return_counts=True)
            dominant_index = int(np.argmax(counts))
            dominant_gt = int(values[dominant_index])
            purity = float(counts[dominant_index] / valid_gt.sum())
        else:
            dominant_gt = -1
            purity = 0.0
        if filter_by_gt and purity < float(min_gt_purity):
            rejected_impure += 1
            continue
        local_points = points[mask]
        local_colors = colors[mask]
        local_views = views[mask]
        local_pixels = pixels[mask]
        if len(local_points) > int(max_geometry_points):
            sample_indices = np.linspace(
                0,
                len(local_points) - 1,
                int(max_geometry_points),
                dtype=np.int64,
            )
            geometry_points = local_points[sample_indices]
            geometry_views = local_views[sample_indices]
        else:
            geometry_points = local_points
            geometry_views = local_views
        normal, offset = weighted_fit_plane(geometry_points)
        centroid = local_points.mean(axis=0)
        centered = geometry_points - geometry_points.mean(axis=0)
        covariance = centered.T @ centered / max(len(centered), 1)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        view_boxes = {}
        for view in np.unique(local_views):
            view_mask = local_views == int(view)
            view_pixels = local_pixels[view_mask]
            view_boxes[int(view)] = {
                "min": view_pixels.min(axis=0).astype(float).tolist(),
                "max": view_pixels.max(axis=0).astype(float).tolist(),
                "centroid": view_pixels.mean(axis=0).astype(float).tolist(),
                "count": int(view_mask.sum()),
            }
        descriptors[int(fragment_id)] = {
            "fragment_id": int(fragment_id),
            "gt_plane": dominant_gt,
            "gt_purity": purity,
            "matched_gt_points": int(valid_gt.sum()),
            "point_count": int(mask.sum()),
            "points": geometry_points,
            "views": geometry_views,
            "normal": normal,
            "offset": float(offset),
            "centroid": centroid,
            "bbox_min": local_points.min(axis=0),
            "bbox_max": local_points.max(axis=0),
            "color_mean": local_colors.mean(axis=0),
            "color_std": local_colors.std(axis=0),
            "eigenvalues": eigenvalues,
            "view_set": set(int(value) for value in np.unique(local_views)),
            "view_boxes": view_boxes,
        }
    audit = {
        "accepted_fragments": len(descriptors),
        "rejected_small": rejected_small,
        "rejected_impure": rejected_impure,
        "filter_by_gt": bool(filter_by_gt),
        "support_points": int(len(points)),
        "scene_scale": scene_scale,
    }
    return descriptors, scene_scale, audit


def pair_features(
    pair: dict,
    descriptors: dict[int, dict],
    scene_scale: float,
    pixel_diagonal: float,
) -> dict | None:
    first = descriptors.get(int(pair["first_fragment_id"]))
    second = descriptors.get(int(pair["second_fragment_id"]))
    if first is None or second is None:
        return None
    first_views = first["view_set"]
    second_views = second["view_set"]
    shared_views = sorted(first_views & second_views)
    union_views = first_views | second_views
    centroid_delta = second["centroid"] - first["centroid"]
    mean_normal = first["normal"] + np.sign(
        first["normal"] @ second["normal"]
    ) * second["normal"]
    mean_normal_norm = float(np.linalg.norm(mean_normal))
    if mean_normal_norm < 1e-8:
        mean_normal = first["normal"]
    else:
        mean_normal = mean_normal / mean_normal_norm
    normal_centroid_separation = abs(float(centroid_delta @ mean_normal))
    tangent_centroid_distance = math.sqrt(
        max(float(centroid_delta @ centroid_delta) - normal_centroid_separation**2, 0.0)
    )
    residual_first_to_second = (
        first["points"] @ second["normal"] + second["offset"]
    )
    residual_second_to_first = (
        second["points"] @ first["normal"] + first["offset"]
    )
    cross_plane_rmse = math.sqrt(
        0.5
        * (
            float(np.mean(residual_first_to_second**2))
            + float(np.mean(residual_second_to_first**2))
        )
    )
    tangent_relations = _tangent_support_relations(
        first["points"],
        second["points"],
        mean_normal,
    )
    if shared_views:
        ious = [
            _bbox_iou(first["view_boxes"][view], second["view_boxes"][view])
            for view in shared_views
        ]
        pixel_gaps = [
            _bbox_gap(
                first["view_boxes"][view]["min"],
                first["view_boxes"][view]["max"],
                second["view_boxes"][view]["min"],
                second["view_boxes"][view]["max"],
            )
            for view in shared_views
        ]
        pixel_centroid_distances = [
            float(
                np.linalg.norm(
                    np.asarray(first["view_boxes"][view]["centroid"])
                    - np.asarray(second["view_boxes"][view]["centroid"])
                )
            )
            for view in shared_views
        ]
        pixel_bbox_iou_max = max(ious)
        pixel_bbox_iou_mean = float(np.mean(ious))
        pixel_bbox_gap_min = min(pixel_gaps) / pixel_diagonal
        pixel_centroid_distance_min = min(pixel_centroid_distances) / pixel_diagonal
    else:
        pixel_bbox_iou_max = 0.0
        pixel_bbox_iou_mean = 0.0
        pixel_bbox_gap_min = 1.0
        pixel_centroid_distance_min = 1.0
    first_eigen = first["eigenvalues"] / (scene_scale**2)
    second_eigen = second["eigenvalues"] / (scene_scale**2)
    pooled_cost = float(pair["pooled_rmse_normalized"])
    bias_cost = float(pair["bias_corrected_rmse_normalized"])
    features = {
        "normal_angle_deg": float(pair["normal_angle_deg"]),
        "pooled_rmse_normalized": pooled_cost,
        "bias_corrected_rmse_normalized": bias_cost,
        "bias_rmse_gain_normalized": pooled_cost - bias_cost,
        "required_bias_std_normalized": float(pair["required_bias_std_normalized"]),
        "required_bias_range_normalized": float(pair["required_bias_range_normalized"]),
        "centroid_distance_normalized": float(np.linalg.norm(centroid_delta)) / scene_scale,
        "normal_centroid_separation_normalized": normal_centroid_separation / scene_scale,
        "tangent_centroid_distance_normalized": tangent_centroid_distance / scene_scale,
        "tangent_support_primary_gap_normalized": tangent_relations["primary_gap"]
        / scene_scale,
        "tangent_support_primary_overlap": tangent_relations["primary_overlap"],
        "tangent_support_secondary_gap_normalized": tangent_relations["secondary_gap"]
        / scene_scale,
        "tangent_support_secondary_overlap": tangent_relations["secondary_overlap"],
        "tangent_support_bbox_gap_normalized": tangent_relations["bbox_gap"]
        / scene_scale,
        "tangent_support_bbox_overlap": tangent_relations["bbox_overlap"],
        "bbox_3d_gap_normalized": _bbox_gap(
            first["bbox_min"],
            first["bbox_max"],
            second["bbox_min"],
            second["bbox_max"],
        )
        / scene_scale,
        "cross_plane_rmse_normalized": cross_plane_rmse / scene_scale,
        "log_point_count_ratio_abs": abs(
            math.log(max(first["point_count"], 1) / max(second["point_count"], 1))
        ),
        "min_point_count_log": math.log1p(
            min(first["point_count"], second["point_count"])
        ),
        "color_mean_l2": float(
            np.linalg.norm(first["color_mean"] - second["color_mean"])
        ),
        "color_std_l2": float(
            np.linalg.norm(first["color_std"] - second["color_std"])
        ),
        "shared_view_count": float(len(shared_views)),
        "union_view_count": float(len(union_views)),
        "view_jaccard": len(shared_views) / max(len(union_views), 1),
        "disjoint_view_sets": float(not shared_views),
        "pixel_bbox_iou_max": pixel_bbox_iou_max,
        "pixel_bbox_iou_mean": pixel_bbox_iou_mean,
        "pixel_bbox_gap_min_normalized": pixel_bbox_gap_min,
        "pixel_centroid_distance_min_normalized": pixel_centroid_distance_min,
    }
    for index in range(3):
        features[f"eigenvalue_{index}_difference"] = abs(
            float(first_eigen[index] - second_eigen[index])
        )
    return {
        "scene": str(pair["scene"]),
        "first_fragment_id": int(pair["first_fragment_id"]),
        "second_fragment_id": int(pair["second_fragment_id"]),
        "first_gt_plane": int(pair["first_gt_plane"]),
        "second_gt_plane": int(pair["second_gt_plane"]),
        "same_gt_plane": bool(pair["same_gt_plane"]),
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Stage2 fragment graph features for learned identity baselines."
    )
    parser.add_argument("--batch_json", required=True)
    parser.add_argument("--pair_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_fragment_points", type=int, default=64)
    parser.add_argument("--min_gt_purity", type=float, default=0.8)
    parser.add_argument("--max_geometry_points", type=int, default=512)
    parser.add_argument("--pixel_diagonal", type=float, default=724.0773439350247)
    args = parser.parse_args()

    batch_path = Path(args.batch_json).resolve()
    pair_path = Path(args.pair_json).resolve()
    root = batch_path
    while root.parent != root and not (root / ".git").exists():
        root = root.parent
    if not (root / ".git").exists():
        raise ValueError("Could not infer repository root from batch_json")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    pairs = json.loads(pair_path.read_text(encoding="utf-8"))
    pairs_by_scene = {}
    for pair in pairs:
        pairs_by_scene.setdefault(str(pair["scene"]), []).append(pair)
    exported = []
    scene_audits = {}
    for item in batch["items"]:
        gt_path = _artifact_path(root, item, "point_aligned_gt")
        scene = _scene_name(item, gt_path)
        support_path = _artifact_path(root, item, "direct_support_records")
        descriptors, scene_scale, audit = build_scene_descriptors(
            support_path,
            gt_path,
            min_fragment_points=args.min_fragment_points,
            min_gt_purity=args.min_gt_purity,
            max_geometry_points=args.max_geometry_points,
        )
        scene_rows = []
        for pair in pairs_by_scene.get(scene, []):
            row = pair_features(
                pair,
                descriptors,
                scene_scale,
                float(args.pixel_diagonal),
            )
            if row is not None:
                scene_rows.append(row)
        exported.extend(scene_rows)
        audit["exported_edges"] = len(scene_rows)
        scene_audits[scene] = audit
    if len(exported) != len(pairs):
        raise ValueError(
            f"Feature export lost pair rows: {len(exported)} != {len(pairs)}"
        )
    feature_names = sorted(exported[0]["features"]) if exported else []
    summary = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "node_source": "stage2_support_direct_global_svd",
        "edge_count": len(exported),
        "scene_count": len(scene_audits),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "label_contract": "GT used only for audit population selection and edge labels",
        "feature_contract": "all features are prediction/provenance derived",
        "coordinate_contract": {
            "pointmap_pixel_order": "(x,y)",
            "pointmap_pixel_space": "DUSt3R input pointmap",
            "pixel_distance_normalizer": float(args.pixel_diagonal),
            "metric_normalizer": "per-scene median point radius; prediction-only",
        },
        "scene_audits": scene_audits,
        "config": {
            "batch_json": str(batch_path),
            "pair_json": str(pair_path),
            "min_fragment_points": int(args.min_fragment_points),
            "min_gt_purity": float(args.min_gt_purity),
            "max_geometry_points": int(args.max_geometry_points),
        },
        "outputs": {
            "edges_json": str(output_dir / "fragment_graph_edges.json"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "fragment_graph_edges.json").write_text(
        json.dumps(exported, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
