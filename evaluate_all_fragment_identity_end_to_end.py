"""Apply scene-held-out identity learning to every predicted Stage2 fragment.

Training and threshold calibration use only high-purity fragments from the
other scenes.  The held-out scene is inferred over all Stage2 fragments
without any GT purity filter, then mapped back to every repeated 3D support
record for point-level partition evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from build_fragment_graph_features import (
    build_scene_descriptors,
    pair_features,
)
from evaluate_bias_aware_fragment_association import (
    _artifact_path,
    _scene_name,
    angle_deg,
    exact_gt_join,
)
from evaluate_canonical_plane_bias_oracle import weighted_fit_plane
from evaluate_fragment_partition_linkage import (
    cluster_labels_from_context,
    linkage_context,
    tune_linkage_threshold,
)
from evaluate_global_plane_baselines import evaluate_support_conditioned_partition
from evaluate_learned_fragment_identity import (
    BIAS_EXCLUSIVE_FEATURES,
    cross_fitted_training_scores,
    feature_matrix,
    fit_model,
    label_vector,
    make_model,
    score_rows,
    training_sample_weights,
)


OUTPUT_SCHEMA_VERSION = 1


def all_fragment_pair_rows(
    scene: str,
    descriptors: dict[int, dict],
    scene_scale: float,
    pixel_diagonal: float,
) -> list[dict]:
    """Build inference edges without consulting descriptor GT labels."""
    fragment_ids = sorted(descriptors)
    rows = []
    for first_index, first_id in enumerate(fragment_ids):
        first = descriptors[first_id]
        for second_id in fragment_ids[first_index + 1 :]:
            second = descriptors[second_id]
            points = np.concatenate((first["points"], second["points"]))
            pooled_normal, pooled_offset = weighted_fit_plane(points)
            pooled_residual = points @ pooled_normal + pooled_offset
            pooled_rmse = float(np.sqrt(np.mean(pooled_residual**2)))
            pair = {
                "scene": scene,
                "first_fragment_id": first_id,
                "second_fragment_id": second_id,
                # Labels below are exported only for evaluator bookkeeping.
                # pair_features never uses them to compute a feature.
                "first_gt_plane": first["gt_plane"],
                "second_gt_plane": second["gt_plane"],
                "same_gt_plane": bool(
                    first["gt_plane"] >= 0
                    and first["gt_plane"] == second["gt_plane"]
                ),
                "normal_angle_deg": angle_deg(first["normal"], second["normal"]),
                "pooled_rmse_normalized": pooled_rmse / scene_scale,
                # The fast end-to-end gate excludes all four bias variables.
                "bias_corrected_rmse_normalized": pooled_rmse / scene_scale,
                "required_bias_std_normalized": 0.0,
                "required_bias_range_normalized": 0.0,
            }
            row = pair_features(
                pair, descriptors, scene_scale, pixel_diagonal
            )
            if row is not None:
                rows.append(row)
    return rows


def assign_fragment_clusters(
    fragment_ids: np.ndarray,
    clustered_nodes: list[int],
    cluster_labels: np.ndarray,
) -> np.ndarray:
    unique_clusters = sorted(int(value) for value in np.unique(cluster_labels))
    cluster_remap = {
        cluster: index for index, cluster in enumerate(unique_clusters)
    }
    fragment_to_cluster = {
        int(node): cluster_remap[int(cluster)]
        for node, cluster in zip(clustered_nodes, cluster_labels)
    }
    result = np.full(np.asarray(fragment_ids).shape, -1, dtype=np.int32)
    for fragment_id, cluster_id in fragment_to_cluster.items():
        result[np.asarray(fragment_ids) == fragment_id] = cluster_id
    return result


def fit_cluster_planes(
    points: np.ndarray, cluster_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cluster_ids = sorted(
        int(value) for value in np.unique(cluster_labels) if value >= 0
    )
    normals = np.zeros((len(cluster_ids), 3), dtype=np.float32)
    offsets = np.zeros((len(cluster_ids),), dtype=np.float32)
    counts = np.zeros((len(cluster_ids),), dtype=np.int32)
    if cluster_ids != list(range(len(cluster_ids))):
        raise ValueError("Cluster labels must be contiguous and zero-based")
    for cluster_id in cluster_ids:
        mask = cluster_labels == cluster_id
        normal, offset = weighted_fit_plane(np.asarray(points)[mask])
        normals[cluster_id] = normal.astype(np.float32)
        offsets[cluster_id] = float(offset)
        counts[cluster_id] = int(mask.sum())
    return normals, offsets, counts


def metric_subset(metrics: dict) -> dict:
    keys = [
        "assigned_point_count",
        "assigned_gt_labeled_point_count",
        "support_conditioned_pred_plane_count",
        "support_conditioned_gt_plane_count",
        "support_conditioned_true_positive_planes",
        "support_conditioned_plane_precision",
        "support_conditioned_plane_recall_observed",
        "support_conditioned_plane_recall_all_gt",
        "support_conditioned_matched_iou",
        "support_conditioned_normal_angular_error_deg",
        "support_conditioned_fragmentation_excess",
        "support_conditioned_overmerge_excess",
        "support_partition_pairwise_precision",
        "support_partition_pairwise_recall",
        "support_partition_pairwise_f1",
        "support_partition_pred_purity",
        "support_partition_gt_completeness",
        "support_partition_purity_completeness_f1",
    ]
    return {key: metrics[key] for key in keys}


def mean_metrics(scene_rows: list[dict], method: str) -> dict:
    rows = [row[method] for row in scene_rows]
    keys = rows[0]
    result = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = (
            float(np.nanmean(values))
            if np.isfinite(values).any()
            else float("nan")
        )
    return result


def save_prediction(
    path: Path,
    support: dict,
    labels: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    counts: np.ndarray,
    runtime_seconds: float,
    method_name: str,
) -> None:
    np.savez_compressed(
        path,
        points=support["points"],
        colors=support["colors"],
        original_colors=support["original_colors"],
        point_plane_ids=labels.astype(np.int32),
        plane_ids=np.arange(len(normals), dtype=np.int32),
        plane_normals=normals.astype(np.float32),
        plane_offsets=offsets.astype(np.float32),
        plane_inlier_counts=counts.astype(np.int32),
        alignment_view_indices=support["alignment_view_indices"].astype(np.int32),
        pointmap_pixel_xy=support["pointmap_pixel_xy"].astype(np.int32),
        pointmap_pixel_coordinate_order=np.asarray("xy"),
        pointmap_pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
        method=np.asarray(method_name),
        runtime_seconds=np.asarray(float(runtime_seconds), dtype=np.float64),
        schema_version=np.asarray(1, dtype=np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all-fragment scene-held-out identity inference and point evaluation."
    )
    parser.add_argument("--batch_json", required=True)
    parser.add_argument("--audit_edges_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_fragment_points", type=int, default=64)
    parser.add_argument("--max_geometry_points", type=int, default=512)
    parser.add_argument("--pixel_diagonal", type=float, default=724.0773439350247)
    parser.add_argument("--regularization_c", type=float, default=0.2)
    parser.add_argument(
        "--model_kind",
        choices=("logistic", "hist_gradient_boosting"),
        default="logistic",
    )
    parser.add_argument("--hard_negative_weight", type=float, default=1.0)
    parser.add_argument("--threshold_count", type=int, default=128)
    parser.add_argument("--calibration_folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--match_iou", type=float, default=0.5)
    parser.add_argument("--fragmentation_iou", type=float, default=0.1)
    parser.add_argument("--min_observed_plane_points", type=int, default=64)
    args = parser.parse_args()

    batch_path = Path(args.batch_json).resolve()
    edges_path = Path(args.audit_edges_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir()
    root = batch_path
    while root.parent != root and not (root / ".git").exists():
        root = root.parent
    if not (root / ".git").exists():
        raise ValueError("Could not infer repository root from batch_json")

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    audit_rows = json.loads(edges_path.read_text(encoding="utf-8"))
    audit_by_scene = {
        scene: [row for row in audit_rows if row["scene"] == scene]
        for scene in sorted({str(row["scene"]) for row in audit_rows})
    }
    all_feature_names = sorted(audit_rows[0]["features"])
    feature_names = [
        name for name in all_feature_names if name not in BIAS_EXCLUSIVE_FEATURES
    ]
    item_by_scene = {}
    for item in batch["items"]:
        gt_path = _artifact_path(root, item, "point_aligned_gt")
        item_by_scene[_scene_name(item, gt_path)] = item
    scenes = sorted(audit_by_scene)
    scene_rows = []
    total_started = time.perf_counter()
    for heldout_index, heldout in enumerate(scenes):
        scene_started = time.perf_counter()
        training_scenes = [scene for scene in scenes if scene != heldout]
        calibration_rows, _ = cross_fitted_training_scores(
            audit_by_scene,
            training_scenes,
            feature_names,
            regularization_c=args.regularization_c,
            calibration_folds=args.calibration_folds,
            seed=args.seed + heldout_index * 100,
            model_kind=args.model_kind,
            hard_negative_weight=args.hard_negative_weight,
        )
        threshold, calibration_f1 = tune_linkage_threshold(
            calibration_rows,
            training_scenes,
            "learned_cost",
            args.threshold_count,
        )
        training_rows = [
            row for scene in training_scenes for row in audit_by_scene[scene]
        ]
        model = make_model(
            args.regularization_c,
            args.seed + heldout_index,
            model_kind=args.model_kind,
        )
        fit_model(
            model,
            feature_matrix(training_rows, feature_names),
            label_vector(training_rows),
            training_sample_weights(
                training_rows, args.hard_negative_weight
            ),
        )

        item = item_by_scene[heldout]
        support_path = _artifact_path(root, item, "direct_support_records")
        gt_path = _artifact_path(root, item, "point_aligned_gt")
        descriptors, scene_scale, descriptor_audit = build_scene_descriptors(
            support_path,
            gt_path,
            min_fragment_points=args.min_fragment_points,
            min_gt_purity=0.0,
            max_geometry_points=args.max_geometry_points,
            filter_by_gt=False,
        )
        inference_rows = all_fragment_pair_rows(
            heldout,
            descriptors,
            scene_scale,
            args.pixel_diagonal,
        )
        probabilities = model.predict_proba(
            feature_matrix(inference_rows, feature_names)
        )[:, 1]
        scored_rows = score_rows(
            inference_rows, probabilities, "learned_cost"
        )
        context = linkage_context(scored_rows, "learned_cost")
        clusters = cluster_labels_from_context(context, threshold)

        with np.load(support_path, allow_pickle=True) as raw:
            support = {key: raw[key] for key in raw.files}
        with np.load(gt_path, allow_pickle=False) as raw:
            gt = {key: raw[key] for key in raw.files}
        learned_labels = assign_fragment_clusters(
            support["point_plane_ids"].astype(np.int64),
            context["nodes"],
            clusters,
        )
        learned_normals, learned_offsets, learned_counts = fit_cluster_planes(
            support["points"], learned_labels
        )
        gt_record_labels = exact_gt_join(
            support["alignment_view_indices"].astype(np.int64),
            support["pointmap_pixel_xy"].astype(np.int64),
            gt["view_indices"].astype(np.int64),
            gt["pixel_xy"].astype(np.int64),
            gt["point_plane_ids"].astype(np.int64),
        )
        direct_metrics = evaluate_support_conditioned_partition(
            support["point_plane_ids"],
            support["plane_normals"],
            gt_record_labels,
            gt["plane_normals"],
            args.match_iou,
            args.fragmentation_iou,
            args.min_observed_plane_points,
        )
        learned_metrics = evaluate_support_conditioned_partition(
            learned_labels,
            learned_normals,
            gt_record_labels,
            gt["plane_normals"],
            args.match_iou,
            args.fragmentation_iou,
            args.min_observed_plane_points,
        )
        scene_runtime = time.perf_counter() - scene_started
        prediction_path = prediction_dir / f"{heldout}_all_fragment_identity.npz"
        save_prediction(
            prediction_path,
            support,
            learned_labels,
            learned_normals,
            learned_offsets,
            learned_counts,
            scene_runtime,
            (
                "learned_topology_"
                f"{args.model_kind}_average_linkage_all_fragments"
            ),
        )
        scene_rows.append(
            {
                "scene": heldout,
                "threshold": threshold,
                "calibration_fragment_f1": calibration_f1,
                "all_fragment_count": len(descriptors),
                "all_edge_count": len(inference_rows),
                "descriptor_audit": descriptor_audit,
                "direct": metric_subset(direct_metrics),
                "learned": metric_subset(learned_metrics),
                "runtime_seconds": scene_runtime,
                "prediction_npz": str(prediction_path),
            }
        )
        print(
            json.dumps(
                {
                    "scene": heldout,
                    "fragments": len(descriptors),
                    "direct_pair_f1": direct_metrics[
                        "support_partition_pairwise_f1"
                    ],
                    "learned_pair_f1": learned_metrics[
                        "support_partition_pairwise_f1"
                    ],
                    "runtime_seconds": scene_runtime,
                }
            ),
            flush=True,
        )
    direct_mean = mean_metrics(scene_rows, "direct")
    learned_mean = mean_metrics(scene_rows, "learned")
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "protocol": "leave-one-scene-out training; all held-out Stage2 fragments inferred",
        "gt_filter_at_inference": False,
        "feature_group": "joint_without_bias_with_tangent_support",
        "model_kind": args.model_kind,
        "hard_negative_weight": float(args.hard_negative_weight),
        "clustering": "average_linkage",
        "scene_count": len(scene_rows),
        "direct_mean": direct_mean,
        "learned_mean": learned_mean,
        "mean_delta": {
            key: learned_mean[key] - direct_mean[key]
            for key in direct_mean
        },
        "scenes": scene_rows,
        "runtime_seconds": time.perf_counter() - total_started,
        "config": {
            "batch_json": str(batch_path),
            "audit_edges_json": str(edges_path),
            "min_fragment_points": int(args.min_fragment_points),
            "max_geometry_points": int(args.max_geometry_points),
            "regularization_c": float(args.regularization_c),
            "model_kind": args.model_kind,
            "hard_negative_weight": float(args.hard_negative_weight),
            "threshold_count": int(args.threshold_count),
            "calibration_folds": int(args.calibration_folds),
        },
        "limitations": [
            "The model is trained on high-purity fragments from other scenes, but inference contains every sufficiently large predicted fragment.",
            "The fast gate excludes the four view-bias variables.",
            "Evaluation is on repeated 3D support observations, not bounded continuous plane polygons.",
        ],
        "outputs": {
            "summary_json": str(output_dir / "summary.json"),
            "prediction_dir": str(prediction_dir),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "direct_mean": direct_mean,
                "learned_mean": learned_mean,
                "mean_delta": result["mean_delta"],
                "runtime_seconds": result["runtime_seconds"],
                "summary_json": result["outputs"]["summary_json"],
            },
            indent=2,
            allow_nan=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
