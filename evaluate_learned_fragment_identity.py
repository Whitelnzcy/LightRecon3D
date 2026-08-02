"""Leave-one-scene-out learned fragment identity and clustering baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_bias_aware_fragment_clustering import (
    leave_one_scene_out,
    scene_partition_metrics,
    tune_threshold,
)
from evaluate_bias_aware_fragment_association import (
    average_precision_lower,
    binary_auc_lower,
)


OUTPUT_SCHEMA_VERSION = 1

GEOMETRY_FEATURES = [
    "normal_angle_deg",
    "pooled_rmse_normalized",
    "centroid_distance_normalized",
    "normal_centroid_separation_normalized",
    "tangent_centroid_distance_normalized",
    "tangent_support_primary_gap_normalized",
    "tangent_support_primary_overlap",
    "tangent_support_secondary_gap_normalized",
    "tangent_support_secondary_overlap",
    "tangent_support_bbox_gap_normalized",
    "tangent_support_bbox_overlap",
    "bbox_3d_gap_normalized",
    "cross_plane_rmse_normalized",
    "log_point_count_ratio_abs",
    "min_point_count_log",
    "eigenvalue_0_difference",
    "eigenvalue_1_difference",
    "eigenvalue_2_difference",
]

APPEARANCE_CONNECTIVITY_FEATURES = [
    "color_mean_l2",
    "color_std_l2",
    "shared_view_count",
    "union_view_count",
    "view_jaccard",
    "disjoint_view_sets",
    "pixel_bbox_iou_max",
    "pixel_bbox_iou_mean",
    "pixel_bbox_gap_min_normalized",
    "pixel_centroid_distance_min_normalized",
]

PROVENANCE_BIAS_FEATURES = [
    "bias_corrected_rmse_normalized",
    "bias_rmse_gain_normalized",
    "required_bias_std_normalized",
    "required_bias_range_normalized",
    "shared_view_count",
    "union_view_count",
    "view_jaccard",
    "disjoint_view_sets",
]

BIAS_EXCLUSIVE_FEATURES = [
    "bias_corrected_rmse_normalized",
    "bias_rmse_gain_normalized",
    "required_bias_std_normalized",
    "required_bias_range_normalized",
]


def feature_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    matrix = np.asarray(
        [
            [float(row["features"][name]) for name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Fragment graph feature matrix contains non-finite values")
    return matrix


def label_vector(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["same_gt_plane"] for row in rows], dtype=np.int64)


def score_rows(rows: list[dict], probabilities: np.ndarray, cost_key: str) -> list[dict]:
    if len(rows) != len(probabilities):
        raise ValueError("Rows and probabilities have different lengths")
    return [
        {
            "scene": str(row["scene"]),
            "first_fragment_id": int(row["first_fragment_id"]),
            "second_fragment_id": int(row["second_fragment_id"]),
            "first_gt_plane": int(row["first_gt_plane"]),
            "second_gt_plane": int(row["second_gt_plane"]),
            cost_key: 1.0 - float(probability),
        }
        for row, probability in zip(rows, probabilities)
    ]


def _mean_finite(values: list[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def make_model(
    regularization_c: float,
    seed: int,
    model_kind: str = "logistic",
):
    if model_kind == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=160,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=int(seed),
        )
    if model_kind != "logistic":
        raise ValueError(f"Unsupported model kind: {model_kind}")
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(regularization_c),
            class_weight="balanced",
            max_iter=2000,
            random_state=int(seed),
            solver="lbfgs",
        ),
    )


def training_sample_weights(
    rows: list[dict],
    hard_negative_weight: float,
) -> np.ndarray | None:
    """Upweight rare coplanar-looking negative edges during supervised fitting."""

    weight = float(hard_negative_weight)
    if weight <= 1.0:
        return None
    weights = np.ones((len(rows),), dtype=np.float64)
    for index, row in enumerate(rows):
        features = row["features"]
        is_hard_negative = (
            not bool(row["same_gt_plane"])
            and float(features["normal_angle_deg"]) < 10.0
            and float(features["pooled_rmse_normalized"]) < 0.1
        )
        if is_hard_negative:
            weights[index] = weight
    return weights


def fit_model(
    model,
    matrix: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray | None,
) -> None:
    if sample_weights is None:
        model.fit(matrix, labels)
    elif hasattr(model, "steps"):
        final_step = model.steps[-1][0]
        model.fit(
            matrix,
            labels,
            **{f"{final_step}__sample_weight": sample_weights},
        )
    else:
        model.fit(matrix, labels, sample_weight=sample_weights)


def cross_fitted_training_scores(
    rows_by_scene: dict[str, list[dict]],
    training_scenes: list[str],
    feature_names: list[str],
    *,
    regularization_c: float,
    calibration_folds: int,
    seed: int,
    model_kind: str = "logistic",
    hard_negative_weight: float = 1.0,
) -> tuple[dict[str, list[dict]], float]:
    fold_count = min(max(int(calibration_folds), 2), len(training_scenes))
    scored_by_scene = {}
    fit_seconds = 0.0
    for fold in range(fold_count):
        validation_scenes = [
            scene
            for index, scene in enumerate(training_scenes)
            if index % fold_count == fold
        ]
        fitting_scenes = [
            scene for scene in training_scenes if scene not in validation_scenes
        ]
        fitting_rows = [
            row for scene in fitting_scenes for row in rows_by_scene[scene]
        ]
        model = make_model(
            regularization_c,
            seed + fold,
            model_kind=model_kind,
        )
        started = time.perf_counter()
        fit_model(
            model,
            feature_matrix(fitting_rows, feature_names),
            label_vector(fitting_rows),
            training_sample_weights(fitting_rows, hard_negative_weight),
        )
        fit_seconds += time.perf_counter() - started
        for scene in validation_scenes:
            validation_rows = rows_by_scene[scene]
            probabilities = model.predict_proba(
                feature_matrix(validation_rows, feature_names)
            )[:, 1]
            scored_by_scene[scene] = score_rows(
                validation_rows, probabilities, "learned_cost"
            )
    if set(scored_by_scene) != set(training_scenes):
        raise AssertionError("Cross-fitted calibration did not score every training scene")
    return scored_by_scene, fit_seconds


def evaluate_feature_group(
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
    total_fit_seconds = 0.0
    for heldout_index, heldout in enumerate(scenes):
        training_scenes = [scene for scene in scenes if scene != heldout]
        training_rows = [
            row for scene in training_scenes for row in rows_by_scene[scene]
        ]
        heldout_rows = rows_by_scene[heldout]
        training_scored_by_scene, calibration_fit_seconds = (
            cross_fitted_training_scores(
                rows_by_scene,
                training_scenes,
                feature_names,
                regularization_c=regularization_c,
                calibration_folds=calibration_folds,
                seed=seed + heldout_index * 100,
            )
        )
        total_fit_seconds += calibration_fit_seconds
        threshold, training_partition_f1 = tune_threshold(
            training_scored_by_scene,
            training_scenes,
            "learned_cost",
            int(threshold_count),
        )
        model = make_model(regularization_c, seed + heldout_index)
        started = time.perf_counter()
        model.fit(
            feature_matrix(training_rows, feature_names),
            label_vector(training_rows),
        )
        total_fit_seconds += time.perf_counter() - started
        heldout_probabilities = model.predict_proba(
            feature_matrix(heldout_rows, feature_names)
        )[:, 1]
        heldout_scored = score_rows(
            heldout_rows, heldout_probabilities, "learned_cost"
        )
        partition = scene_partition_metrics(
            heldout_scored, "learned_cost", threshold
        )
        labels = label_vector(heldout_rows).astype(bool)
        costs = 1.0 - heldout_probabilities
        heldout_results.append(
            {
                "scene": heldout,
                "threshold": float(threshold),
                "training_scene_mean_partition_f1": float(training_partition_f1),
                "pair_roc_auc": binary_auc_lower(labels, costs),
                "pair_average_precision": average_precision_lower(labels, costs),
                **partition,
            }
        )
    return {
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "model": "StandardScaler + class-balanced LogisticRegression",
        "total_fit_seconds": total_fit_seconds,
        "mean_pair_roc_auc": _mean_finite(
            [row["pair_roc_auc"] for row in heldout_results]
        ),
        "mean_pair_average_precision": _mean_finite(
            [row["pair_average_precision"] for row in heldout_results]
        ),
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


def pooled_baseline(rows_by_scene: dict[str, list[dict]], threshold_count: int) -> dict:
    flattened = {}
    for scene, rows in rows_by_scene.items():
        flattened[scene] = [
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
    return leave_one_scene_out(flattened, "pooled_cost", int(threshold_count))


def paired_vs_baseline(method: dict, baseline: dict) -> dict:
    baseline_rows = {
        row["scene"]: row for row in baseline["heldout_scenes"]
    }
    deltas = np.asarray(
        [
            row["f1"] - baseline_rows[row["scene"]]["f1"]
            for row in method["heldout_scenes"]
        ],
        dtype=np.float64,
    )
    return {
        "mean_f1_delta": float(np.mean(deltas)),
        "median_f1_delta": float(np.median(deltas)),
        "win_tie_loss": [
            int(np.sum(deltas > 1e-12)),
            int(np.sum(np.abs(deltas) <= 1e-12)),
            int(np.sum(deltas < -1e-12)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate lightweight learned fragment identity with scene-held-out clustering."
    )
    parser.add_argument("--edges_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--regularization_c", type=float, default=0.2)
    parser.add_argument("--threshold_count", type=int, default=128)
    parser.add_argument("--calibration_folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    edges_path = Path(args.edges_json).resolve()
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
    groups = {
        "geometry_only": GEOMETRY_FEATURES,
        "appearance_connectivity_only": APPEARANCE_CONNECTIVITY_FEATURES,
        "provenance_bias_only": PROVENANCE_BIAS_FEATURES,
        "geometry_plus_bias": sorted(
            set(GEOMETRY_FEATURES + PROVENANCE_BIAS_FEATURES)
        ),
        "joint_without_bias": [
            name for name in all_features if name not in BIAS_EXCLUSIVE_FEATURES
        ],
        "joint_all": all_features,
    }
    for name, feature_names in groups.items():
        missing = sorted(set(feature_names) - set(all_features))
        if missing:
            raise ValueError(f"Feature group {name} is missing: {missing}")

    baseline = pooled_baseline(rows_by_scene, args.threshold_count)
    methods = {}
    for name, feature_names in groups.items():
        methods[name] = evaluate_feature_group(
            rows_by_scene,
            feature_names,
            regularization_c=args.regularization_c,
            threshold_count=args.threshold_count,
            calibration_folds=args.calibration_folds,
            seed=args.seed,
        )
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "protocol": "leave-one-scene-out model evaluation",
        "threshold_protocol": (
            "global threshold selected on scene-grouped cross-fitted predictions "
            "from the other 16 scenes; held-out scene labels are never used"
        ),
        "baseline": baseline,
        "methods": methods,
        "paired_vs_pooled": {
            name: paired_vs_baseline(method, baseline)
            for name, method in methods.items()
        },
        "config": {
            "edges_json": str(edges_path),
            "regularization_c": float(args.regularization_c),
            "threshold_count": int(args.threshold_count),
            "calibration_folds": int(args.calibration_folds),
            "seed": int(args.seed),
        },
        "limitations": [
            "The audit population excludes fragments with GT purity below the configured threshold.",
            "RGB statistics are not deep appearance embeddings.",
            "The classifier is pairwise and clustering is connected components, not a learned global partition objective.",
            "Threshold selection uses scene-grouped cross-fitted predictions from training scenes.",
        ],
        "compute": {
            "gpu_required": False,
            "model_family": "linear logistic regression",
            "feature_dimension": len(all_features),
        },
        "outputs": {"summary_json": str(output_dir / "summary.json")},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    compact = {
        "baseline": {
            key: baseline[key]
            for key in ("mean_precision", "mean_recall", "mean_f1", "median_f1")
        },
        "methods": {
            name: {
                key: method[key]
                for key in (
                    "feature_count",
                    "mean_pair_roc_auc",
                    "mean_pair_average_precision",
                    "mean_precision",
                    "mean_recall",
                    "mean_f1",
                    "median_f1",
                    "total_fit_seconds",
                )
            }
            for name, method in methods.items()
        },
        "paired_vs_pooled": result["paired_vs_pooled"],
        "summary_json": result["outputs"]["summary_json"],
    }
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
