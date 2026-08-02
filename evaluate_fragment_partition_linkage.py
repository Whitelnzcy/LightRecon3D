"""Component-consistent fragment partitioning for learned identity scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from analyze_bias_aware_fragment_clustering import candidate_thresholds
from evaluate_learned_fragment_identity import (
    BIAS_EXCLUSIVE_FEATURES,
    GEOMETRY_FEATURES,
    cross_fitted_training_scores,
    feature_matrix,
    label_vector,
    make_model,
    paired_vs_baseline,
    score_rows,
)


OUTPUT_SCHEMA_VERSION = 1


def linkage_context(rows: list[dict], cost_key: str) -> dict:
    gt_by_node = {}
    for row in rows:
        gt_by_node[int(row["first_fragment_id"])] = int(row["first_gt_plane"])
        gt_by_node[int(row["second_fragment_id"])] = int(row["second_gt_plane"])
    nodes = sorted(gt_by_node)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    distances = np.ones((len(nodes), len(nodes)), dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    for row in rows:
        first = node_to_index[int(row["first_fragment_id"])]
        second = node_to_index[int(row["second_fragment_id"])]
        cost = float(row[cost_key])
        distances[first, second] = min(distances[first, second], cost)
        distances[second, first] = distances[first, second]
    tree = (
        linkage(squareform(distances, checks=False), method="average")
        if len(nodes) > 1
        else np.empty((0, 4), dtype=np.float64)
    )
    return {
        "nodes": nodes,
        "gt_labels": np.asarray([gt_by_node[node] for node in nodes], dtype=np.int64),
        "tree": tree,
    }


def metrics_from_context(context: dict, threshold: float) -> dict:
    gt_labels = context["gt_labels"]
    cluster_labels = cluster_labels_from_context(context, threshold)
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for first in range(len(gt_labels)):
        for second in range(first + 1, len(gt_labels)):
            truth = gt_labels[first] == gt_labels[second]
            prediction = cluster_labels[first] == cluster_labels[second]
            if truth and prediction:
                true_positive += 1
            elif prediction:
                false_positive += 1
            elif truth:
                false_negative += 1
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "nodes": int(len(gt_labels)),
        "true_positive_pairs": true_positive,
        "false_positive_pairs": false_positive,
        "false_negative_pairs": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "component_count": int(len(np.unique(cluster_labels))),
    }


def cluster_labels_from_context(context: dict, threshold: float) -> np.ndarray:
    gt_labels = context["gt_labels"]
    if len(gt_labels) == 1:
        return np.ones((1,), dtype=np.int64)
    return fcluster(
        context["tree"], t=float(threshold), criterion="distance"
    ).astype(np.int64)


def tune_linkage_threshold(
    rows_by_scene: dict[str, list[dict]],
    training_scenes: list[str],
    cost_key: str,
    threshold_count: int,
) -> tuple[float, float]:
    training_rows = [
        row for scene in training_scenes for row in rows_by_scene[scene]
    ]
    thresholds = candidate_thresholds(
        training_rows, cost_key, int(threshold_count)
    )
    contexts = {
        scene: linkage_context(rows_by_scene[scene], cost_key)
        for scene in training_scenes
    }
    mean_f1 = []
    for threshold in thresholds:
        mean_f1.append(
            float(
                np.mean(
                    [
                        metrics_from_context(contexts[scene], threshold)["f1"]
                        for scene in training_scenes
                    ]
                )
            )
        )
    best_index = int(np.argmax(mean_f1))
    return float(thresholds[best_index]), float(mean_f1[best_index])


def leave_one_scene_out_linkage(
    rows_by_scene: dict[str, list[dict]],
    cost_key: str,
    threshold_count: int,
) -> dict:
    heldout_results = []
    scenes = sorted(rows_by_scene)
    for heldout in scenes:
        training_scenes = [scene for scene in scenes if scene != heldout]
        threshold, training_f1 = tune_linkage_threshold(
            rows_by_scene, training_scenes, cost_key, threshold_count
        )
        metrics = metrics_from_context(
            linkage_context(rows_by_scene[heldout], cost_key), threshold
        )
        heldout_results.append(
            {
                "scene": heldout,
                "threshold": threshold,
                "training_scene_mean_f1": training_f1,
                **metrics,
            }
        )
    return summarize_partitions(heldout_results)


def summarize_partitions(heldout_results: list[dict]) -> dict:
    return {
        "mean_precision": float(
            np.mean([row["precision"] for row in heldout_results])
        ),
        "mean_recall": float(
            np.mean([row["recall"] for row in heldout_results])
        ),
        "mean_f1": float(np.mean([row["f1"] for row in heldout_results])),
        "median_f1": float(np.median([row["f1"] for row in heldout_results])),
        "heldout_scenes": heldout_results,
    }


def evaluate_learned_linkage(
    rows_by_scene: dict[str, list[dict]],
    feature_names: list[str],
    *,
    regularization_c: float,
    threshold_count: int,
    calibration_folds: int,
    seed: int,
) -> dict:
    scenes = sorted(rows_by_scene)
    heldout_results = []
    for heldout_index, heldout in enumerate(scenes):
        training_scenes = [scene for scene in scenes if scene != heldout]
        calibration_rows, _ = cross_fitted_training_scores(
            rows_by_scene,
            training_scenes,
            feature_names,
            regularization_c=regularization_c,
            calibration_folds=calibration_folds,
            seed=seed + heldout_index * 100,
        )
        threshold, training_f1 = tune_linkage_threshold(
            calibration_rows,
            training_scenes,
            "learned_cost",
            threshold_count,
        )
        training_rows = [
            row for scene in training_scenes for row in rows_by_scene[scene]
        ]
        model = make_model(regularization_c, seed + heldout_index)
        model.fit(
            feature_matrix(training_rows, feature_names),
            label_vector(training_rows),
        )
        heldout_rows = rows_by_scene[heldout]
        probabilities = model.predict_proba(
            feature_matrix(heldout_rows, feature_names)
        )[:, 1]
        scored = score_rows(heldout_rows, probabilities, "learned_cost")
        metrics = metrics_from_context(
            linkage_context(scored, "learned_cost"), threshold
        )
        heldout_results.append(
            {
                "scene": heldout,
                "threshold": threshold,
                "training_scene_mean_f1": training_f1,
                **metrics,
            }
        )
    result = summarize_partitions(heldout_results)
    result["feature_names"] = feature_names
    result["feature_count"] = len(feature_names)
    return result


def flatten_pooled_rows(rows_by_scene: dict[str, list[dict]]) -> dict[str, list[dict]]:
    return {
        scene: [
            {
                "scene": scene,
                "first_fragment_id": int(row["first_fragment_id"]),
                "second_fragment_id": int(row["second_fragment_id"]),
                "first_gt_plane": int(row["first_gt_plane"]),
                "second_gt_plane": int(row["second_gt_plane"]),
                "pooled_cost": float(row["features"]["pooled_rmse_normalized"]),
            }
            for row in rows
        ]
        for scene, rows in rows_by_scene.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate average-linkage component consistency for fragment identity."
    )
    parser.add_argument("--edges_json", required=True)
    parser.add_argument("--connected_summary_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--regularization_c", type=float, default=0.2)
    parser.add_argument("--threshold_count", type=int, default=128)
    parser.add_argument("--calibration_folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    edges_path = Path(args.edges_json).resolve()
    connected_path = Path(args.connected_summary_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    rows = json.loads(edges_path.read_text(encoding="utf-8"))
    rows_by_scene = {
        scene: [row for row in rows if row["scene"] == scene]
        for scene in sorted({str(row["scene"]) for row in rows})
    }
    all_features = sorted(rows[0]["features"])
    feature_groups = {
        "geometry_only": GEOMETRY_FEATURES,
        "joint_without_bias": [
            name for name in all_features if name not in BIAS_EXCLUSIVE_FEATURES
        ],
        "joint_all": all_features,
    }
    pooled = leave_one_scene_out_linkage(
        flatten_pooled_rows(rows_by_scene),
        "pooled_cost",
        args.threshold_count,
    )
    learned = {
        name: evaluate_learned_linkage(
            rows_by_scene,
            feature_names,
            regularization_c=args.regularization_c,
            threshold_count=args.threshold_count,
            calibration_folds=args.calibration_folds,
            seed=args.seed,
        )
        for name, feature_names in feature_groups.items()
    }
    connected = json.loads(connected_path.read_text(encoding="utf-8"))
    connected_pooled = connected["baseline"]
    connected_joint = connected["methods"]["joint_all"]
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "clustering": "average_linkage",
        "pooled_average_linkage": pooled,
        "learned_average_linkage": learned,
        "contrasts": {
            "pooled_average_vs_pooled_connected": paired_vs_baseline(
                pooled, connected_pooled
            ),
            "joint_average_vs_pooled_connected": paired_vs_baseline(
                learned["joint_all"], connected_pooled
            ),
            "joint_average_vs_joint_connected": paired_vs_baseline(
                learned["joint_all"], connected_joint
            ),
            "joint_all_vs_joint_without_bias_average": paired_vs_baseline(
                learned["joint_all"], learned["joint_without_bias"]
            ),
        },
        "config": {
            "edges_json": str(edges_path),
            "connected_summary_json": str(connected_path),
            "regularization_c": float(args.regularization_c),
            "threshold_count": int(args.threshold_count),
            "calibration_folds": int(args.calibration_folds),
            "seed": int(args.seed),
        },
        "limitations": [
            "Average linkage is a deterministic component-consistency baseline, not a trained graph partition objective.",
            "The high-purity fragment audit population is GT-selected.",
            "Missing fragment edges receive maximum cost 1.0.",
        ],
        "outputs": {"summary_json": str(output_dir / "summary.json")},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    compact = {
        "pooled_average_linkage": {
            key: pooled[key]
            for key in ("mean_precision", "mean_recall", "mean_f1", "median_f1")
        },
        "learned_average_linkage": {
            name: {
                key: method[key]
                for key in ("mean_precision", "mean_recall", "mean_f1", "median_f1")
            }
            for name, method in learned.items()
        },
        "contrasts": result["contrasts"],
        "summary_json": result["outputs"]["summary_json"],
    }
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
