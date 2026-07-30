"""Build frozen final GT after the prediction barrier and evaluate B16/M16."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from build_structured3d_point_aligned_gt import build_point_aligned_gt
from final_bm_methods_e16 import (
    OUT,
    REGISTRATION_ID,
    array_sha256,
    load_config,
    load_json,
    locate_npz,
    sha256_file,
    utc_now,
    write_json,
)


GT_ROOT = Path("F:/Structured3D_clean_e11/point_aligned_gt_e16")
ANNOTATION_ROOT = Path("F:/Structured3D_clean_e11/sealed_annotations")
SIDECAR_ROOT = Path("F:/Structured3D_clean_e11/sealed_view_gt_e16")
MATCH_IOU = 0.5
FRAGMENTATION_IOU = 0.1


def verify_prediction_barrier() -> tuple[dict[str, Any], str]:
    manifest_path = OUT / "prediction_manifest.json"
    sha_path = OUT / "prediction_manifest.sha256"
    if not manifest_path.is_file() or not sha_path.is_file():
        raise RuntimeError("prediction barrier files are absent")
    manifest = load_json(manifest_path)
    actual = sha256_file(manifest_path)
    recorded = sha_path.read_text(encoding="utf-8").split()[0]
    if (
        actual != recorded
        or manifest.get("prediction_count") != 256
        or not manifest.get("prediction_frozen_before_GT_evaluation")
        or manifest.get("final_layout_json_parse_count") != 0
        or manifest.get("final_camera_pose_numeric_parse_count") != 0
        or manifest.get("final_annotation_semantic_parse_count") != 0
    ):
        raise RuntimeError("prediction freeze barrier is invalid")
    for receipt in manifest["predictions"]:
        path = Path(receipt["prediction_path"])
        if sha256_file(path) != receipt["whole_file_sha256"]:
            raise RuntimeError(f"prediction whole-file hash mismatch: {path}")
    return manifest, actual


def build_final_gt(config: dict[str, Any]) -> dict[str, Any]:
    prediction_manifest, prediction_manifest_hash = verify_prediction_barrier()
    view_manifest = load_json(Path(config["e14_view_manifest"]))
    receipts = []
    total_views = 0
    for record in view_manifest["records"]:
        scene = record["scene_id"]
        global_path = locate_npz(
            Path(config["global_cache_root"]), scene, "global_cache_e14.npz"
        )
        output_dir = GT_ROOT / scene
        output = output_dir / "point_aligned_gt_e16.npz"
        manifest_path = output_dir / "point_aligned_gt_e16.json"
        existing = None
        if output.is_file() and manifest_path.is_file():
            existing = load_json(manifest_path)
        if (
            existing
            and existing.get("source_global_cloud_sha256") == sha256_file(global_path)
            and existing.get("output_sha256") == sha256_file(output)
        ):
            receipt = existing
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            temporary = output_dir / "point_aligned_gt_e16.tmp.npz"
            row = build_point_aligned_gt(
                global_path,
                temporary,
                min_conf=0.0,
                min_plane_points=64,
                image_size=512,
                patch_size=16,
                square_ok=False,
                boundary_ignore_radius=1,
                sidecar_root=SIDECAR_ROOT,
                annotation_root=ANNOTATION_ROOT,
            )
            with np.load(temporary, allow_pickle=False) as gt:
                required = {
                    "source_cache_indices",
                    "point_plane_ids",
                    "plane_normals",
                    "view_indices",
                    "pixel_xy",
                }
                if required - set(gt.files):
                    raise RuntimeError(f"{scene}: incomplete GT schema")
                if len(gt["source_cache_indices"]) != len(gt["point_plane_ids"]):
                    raise RuntimeError(f"{scene}: GT array length mismatch")
            os.replace(temporary, output)
            receipt = {
                **row,
                "scene_id": scene,
                "view_count": int(record["view_count"]),
                "output_npz": str(output),
                "source_global_cloud_sha256": sha256_file(global_path),
                "output_sha256": sha256_file(output),
                "prediction_manifest_sha256": prediction_manifest_hash,
                "completed_utc": utc_now(),
            }
            write_json(manifest_path, receipt)
        receipts.append(receipt)
        total_views += int(record["view_count"])
    if len(receipts) != 128 or total_views != 636:
        raise RuntimeError("final GT completion counts are invalid")
    result = {
        "registration_id": REGISTRATION_ID,
        "status": "frozen",
        "scene_count": len(receipts),
        "selected_view_count": total_views,
        "layout_json_parse_count": total_views,
        "camera_pose_numeric_parse_count": total_views,
        "annotation_semantic_parse_count": len(receipts),
        "prediction_manifest_sha256": prediction_manifest_hash,
        "receipts": receipts,
        "frozen_utc": utc_now(),
    }
    write_json(OUT / "point_aligned_gt_manifest.json", result)
    return result


def choose2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    return values * (values - 1) // 2


def partition_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    valid = truth >= 0
    pred = prediction[valid].astype(np.int64)
    gt = truth[valid].astype(np.int64)
    pred_ids, pred_inverse = np.unique(pred, return_inverse=True)
    gt_ids, gt_inverse = np.unique(gt, return_inverse=True)
    contingency = np.bincount(
        pred_inverse * len(gt_ids) + gt_inverse,
        minlength=len(pred_ids) * len(gt_ids),
    ).reshape(len(pred_ids), len(gt_ids))
    same_both = int(choose2(contingency).sum())
    same_pred = int(choose2(contingency.sum(axis=1)).sum())
    same_gt = int(choose2(contingency.sum(axis=0)).sum())
    false_positive = same_pred - same_both
    false_negative = same_gt - same_both
    precision = same_both / max(same_both + false_positive, 1)
    recall = same_both / max(same_both + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "partition_pair_precision": precision,
        "partition_pair_recall": recall,
        "partition_pair_f1": f1,
        "partition_true_positive_pairs": same_both,
        "partition_false_positive_pairs": false_positive,
        "partition_false_negative_pairs": false_negative,
    }


def iou_matrix(
    prediction: np.ndarray,
    truth: np.ndarray,
    pred_ids: np.ndarray,
    gt_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float64)
    intersections = np.zeros_like(matrix, dtype=np.int64)
    pred_counts = {int(value): int((prediction == value).sum()) for value in pred_ids}
    gt_counts = {int(value): int((truth == value).sum()) for value in gt_ids}
    for pred_index, pred_id in enumerate(pred_ids):
        pred_mask = prediction == pred_id
        values, counts = np.unique(truth[pred_mask & (truth >= 0)], return_counts=True)
        overlap = {int(value): int(count) for value, count in zip(values, counts)}
        for gt_index, gt_id in enumerate(gt_ids):
            intersection = overlap.get(int(gt_id), 0)
            union = pred_counts[int(pred_id)] + gt_counts[int(gt_id)] - intersection
            intersections[pred_index, gt_index] = intersection
            matrix[pred_index, gt_index] = intersection / max(union, 1)
    return matrix, intersections


def evaluate_one(
    prediction_receipt: dict[str, Any],
    gt_receipt: dict[str, Any],
    global_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction_path = Path(prediction_receipt["prediction_path"])
    before_hash = sha256_file(prediction_path)
    if before_hash != prediction_receipt["whole_file_sha256"]:
        raise RuntimeError("prediction changed before evaluation")
    with np.load(global_path, allow_pickle=False) as cache:
        cache_points = cache["points"].astype(np.float32)
        cache_views = cache["view_indices"].astype(np.int16)
        cache_pixels = cache["pixel_xy"].astype(np.int32)
    with np.load(gt_receipt["output_npz"], allow_pickle=False) as gt:
        source_indices = gt["source_cache_indices"].astype(np.int64)
        gt_labels_sparse = gt["point_plane_ids"].astype(np.int32)
        gt_normals = gt["plane_normals"].astype(np.float32)
    gt_labels = np.full(len(cache_points), -1, dtype=np.int32)
    gt_labels[source_indices] = gt_labels_sparse
    with np.load(prediction_path, allow_pickle=False) as prediction:
        global_indices = prediction["global_point_indices"].astype(np.int64)
        if len(np.unique(global_indices)) != len(global_indices):
            raise RuntimeError("prediction support contains duplicate global points")
        if not np.array_equal(cache_views[global_indices], prediction["view_indices"]):
            raise RuntimeError("prediction view provenance mismatch")
        if not np.array_equal(cache_pixels[global_indices], prediction["pixel_xy"]):
            raise RuntimeError("prediction pixel provenance mismatch")
        observation_plane_ids = prediction["plane_ids"].astype(np.int32)
        pred_normals = prediction["plane_normals"].astype(np.float32)
        pred_offsets = prediction["plane_offsets"].astype(np.float32)
        frozen_array_hashes = {
            "support_sha256": array_sha256(
                np.column_stack(
                    (
                        prediction["view_indices"],
                        prediction["pixel_xy"],
                        prediction["global_point_indices"],
                    )
                )
            ),
            "plane_id_sha256": array_sha256(observation_plane_ids),
            "plane_parameter_sha256": array_sha256(
                np.column_stack((pred_normals, pred_offsets))
            ),
            "provenance_sha256": hashlib.sha256(
                str(prediction["fragment_provenance_json"].item()).encode("utf-8")
            ).hexdigest(),
        }
    for key, value in frozen_array_hashes.items():
        if value != prediction_receipt[key]:
            raise RuntimeError(f"prediction {key} mismatch")
    pred_labels = np.full(len(cache_points), -1, dtype=np.int32)
    pred_labels[global_indices] = observation_plane_ids
    pred_ids = np.asarray(
        sorted(int(value) for value in np.unique(pred_labels) if value >= 0),
        dtype=np.int32,
    )
    gt_ids = np.asarray(
        sorted(int(value) for value in np.unique(gt_labels) if value >= 0),
        dtype=np.int32,
    )
    matrix, intersections = iou_matrix(pred_labels, gt_labels, pred_ids, gt_ids)
    row_indices, column_indices = (
        linear_sum_assignment(1.0 - matrix)
        if matrix.size
        else (np.asarray([], dtype=int), np.asarray([], dtype=int))
    )
    matches = [
        (
            int(pred_ids[row]),
            int(gt_ids[column]),
            float(matrix[row, column]),
            int(intersections[row, column]),
        )
        for row, column in zip(row_indices, column_indices)
    ]
    accepted = [row for row in matches if row[2] >= MATCH_IOU]
    true_positive = len(accepted)
    precision = true_positive / max(len(pred_ids), 1)
    recall = true_positive / max(len(gt_ids), 1)
    plane_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accepted_iou = np.asarray([row[2] for row in accepted], dtype=np.float64)
    fragments = [
        int((matrix[:, column] >= FRAGMENTATION_IOU).sum())
        for column in range(len(gt_ids))
    ]
    overmerges = [
        int((matrix[row, :] >= FRAGMENTATION_IOU).sum())
        for row in range(len(pred_ids))
    ]
    gt_support = gt_labels >= 0
    assigned = pred_labels >= 0
    matched_points = np.zeros(len(cache_points), dtype=bool)
    accepted_mapping = {pred: gt for pred, gt, _, _ in accepted}
    for pred_id, gt_id in accepted_mapping.items():
        matched_points |= (pred_labels == pred_id) & (gt_labels == gt_id)
    nonplane = ~gt_support
    assigned_gt_count = int((assigned & gt_support).sum())
    matched_gt_count = int(matched_points.sum())
    assigned_count = int(assigned.sum())
    gt_support_count = int(gt_support.sum())
    nonplane_count = int(nonplane.sum())
    nonplane_false_assigned_count = int((assigned & nonplane).sum())
    metrics = {
        "scene_id": prediction_receipt["scene_id"],
        "method_id": prediction_receipt["method_id"],
        "view_count": prediction_receipt["view_count"],
        "cache_point_count": len(cache_points),
        "Plane_F1": plane_f1,
        "true_positive_planes": true_positive,
        "predicted_plane_count": len(pred_ids),
        "gt_plane_count": len(gt_ids),
        "plane_precision": precision,
        "plane_recall": recall,
        "matched_iou_mean": float(accepted_iou.mean()) if len(accepted_iou) else 0.0,
        "matched_iou_median": float(np.median(accepted_iou))
        if len(accepted_iou)
        else 0.0,
        "matched_iou_minimum": float(accepted_iou.min()) if len(accepted_iou) else 0.0,
        "fragmentation_excess": int(sum(max(value - 1, 0) for value in fragments)),
        "overmerge_excess": int(sum(max(value - 1, 0) for value in overmerges)),
        "assigned_gt_support_point_count": assigned_gt_count,
        "matched_gt_support_point_count": matched_gt_count,
        "assigned_cache_point_count": assigned_count,
        "gt_support_point_count": gt_support_count,
        "nonplane_cache_point_count": nonplane_count,
        "nonplane_false_assigned_point_count": nonplane_false_assigned_count,
        "accepted_matched_iou_sum": float(accepted_iou.sum()),
        "gt_support_assignment_rate": assigned_gt_count / max(gt_support_count, 1),
        "gt_support_matched_coverage": matched_gt_count / max(gt_support_count, 1),
        "full_cache_label_rate": assigned_count / max(len(assigned), 1),
        "gt_plane_label_rate": gt_support_count / max(len(gt_support), 1),
        "nonplane_false_assignment_rate": nonplane_false_assigned_count
        / max(nonplane_count, 1),
        **partition_metrics(pred_labels, gt_labels),
        "prediction_hash_before": before_hash,
    }
    plane_rows = [
        {
            "scene_id": prediction_receipt["scene_id"],
            "method_id": prediction_receipt["method_id"],
            "prediction_plane_id": pred_id,
            "gt_plane_id": gt_id,
            "matched_iou": score,
            "intersection_point_count": intersection,
            "accepted_tp": score >= MATCH_IOU,
        }
        for pred_id, gt_id, score, intersection in matches
    ]
    after_hash = sha256_file(prediction_path)
    if after_hash != before_hash:
        raise RuntimeError("prediction changed during evaluation")
    metrics["prediction_hash_after"] = after_hash
    metrics["immutability_passed"] = True
    return metrics, plane_rows


def finite_mean(rows: list[dict], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return float(np.nanmean(values))


def paired_bootstrap(
    rows_b: list[dict], rows_m: list[dict], key: str
) -> dict[str, Any]:
    delta = np.asarray(
        [m[key] - b[key] for b, m in zip(rows_b, rows_m, strict=True)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260729)
    indices = rng.integers(0, len(delta), size=(10000, len(delta)))
    samples = delta[indices].mean(axis=1)
    return {
        "metric": key,
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "positive_scene_count": int((delta > 0).sum()),
        "zero_scene_count": int((delta == 0).sum()),
        "negative_scene_count": int((delta < 0).sum()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_all(config: dict[str, Any]) -> dict[str, Any]:
    prediction_manifest, prediction_manifest_hash = verify_prediction_barrier()
    gt_manifest = load_json(OUT / "point_aligned_gt_manifest.json")
    if (
        gt_manifest.get("status") != "frozen"
        or gt_manifest.get("scene_count") != 128
        or gt_manifest.get("prediction_manifest_sha256") != prediction_manifest_hash
    ):
        raise RuntimeError("final GT manifest is invalid")
    gt_by_scene = {row["scene_id"]: row for row in gt_manifest["receipts"]}
    metrics = []
    plane_rows = []
    for receipt in prediction_manifest["predictions"]:
        scene = receipt["scene_id"]
        global_path = locate_npz(
            Path(config["global_cache_root"]), scene, "global_cache_e14.npz"
        )
        scene_metrics, scene_planes = evaluate_one(
            receipt, gt_by_scene[scene], global_path
        )
        metrics.append(scene_metrics)
        plane_rows.extend(scene_planes)
    rows_b = [row for row in metrics if row["method_id"] == "B16"]
    rows_m = [row for row in metrics if row["method_id"] == "M16"]
    if [row["scene_id"] for row in rows_b] != [row["scene_id"] for row in rows_m]:
        raise RuntimeError("paired scene order mismatch")
    bootstrap = {
        "repetitions": 10000,
        "seed": 20260729,
        "confidence_interval": 0.95,
        "effects": [
            paired_bootstrap(rows_b, rows_m, "Plane_F1"),
            paired_bootstrap(rows_b, rows_m, "matched_iou_mean"),
            paired_bootstrap(rows_b, rows_m, "partition_pair_f1"),
        ],
    }
    effect = {row["metric"]: row for row in bootstrap["effects"]}
    delta_f1 = effect["Plane_F1"]["mean"]
    delta_pair = effect["partition_pair_f1"]["mean"]
    delta_iou = effect["matched_iou_mean"]["mean"]
    fragment_b = finite_mean(rows_b, "fragmentation_excess")
    fragment_m = finite_mean(rows_m, "fragmentation_excess")
    overmerge_b = finite_mean(rows_b, "overmerge_excess")
    overmerge_m = finite_mean(rows_m, "overmerge_excess")
    aggregate_tp_b = sum(row["true_positive_planes"] for row in rows_b)
    aggregate_tp_m = sum(row["true_positive_planes"] for row in rows_m)
    if (
        delta_f1 > 0
        and effect["Plane_F1"]["ci95_lower"] > 0
        and delta_iou >= 0
        and delta_pair >= 0
        and fragment_m + overmerge_m <= fragment_b + overmerge_b
        and aggregate_tp_m >= aggregate_tp_b
    ):
        verdict = "M_Primary_Method"
    elif (
        (delta_f1 > 0 or delta_pair > 0)
        and (fragment_m < fragment_b or overmerge_m < overmerge_b)
    ):
        verdict = "M_Useful_Extension"
    elif (
        delta_f1 <= 0
        and delta_pair <= 0
        and fragment_m >= fragment_b
        and overmerge_m >= overmerge_b
    ):
        verdict = "B_Primary_M_Negative_Result"
    else:
        verdict = "Mixed_Result"
    macro = {
        method: {
            key: finite_mean(rows, key)
            for key in (
                "Plane_F1",
                "matched_iou_mean",
                "partition_pair_precision",
                "partition_pair_recall",
                "partition_pair_f1",
                "fragmentation_excess",
                "overmerge_excess",
                "gt_support_assignment_rate",
                "gt_support_matched_coverage",
                "full_cache_label_rate",
                "gt_plane_label_rate",
                "nonplane_false_assignment_rate",
            )
        }
        for method, rows in (("B16", rows_b), ("M16", rows_m))
    }
    aggregate = {}
    for method, rows in (("B16", rows_b), ("M16", rows_m)):
        tp = sum(row["true_positive_planes"] for row in rows)
        pred = sum(row["predicted_plane_count"] for row in rows)
        gt = sum(row["gt_plane_count"] for row in rows)
        same_both = sum(row["partition_true_positive_pairs"] for row in rows)
        false_positive_pairs = sum(
            row["partition_false_positive_pairs"] for row in rows
        )
        false_negative_pairs = sum(
            row["partition_false_negative_pairs"] for row in rows
        )
        pair_precision = same_both / max(same_both + false_positive_pairs, 1)
        pair_recall = same_both / max(same_both + false_negative_pairs, 1)
        assigned_gt = sum(row["assigned_gt_support_point_count"] for row in rows)
        matched_gt = sum(row["matched_gt_support_point_count"] for row in rows)
        assigned_cache = sum(row["assigned_cache_point_count"] for row in rows)
        gt_support_points = sum(row["gt_support_point_count"] for row in rows)
        cache_points = sum(row["cache_point_count"] for row in rows)
        nonplane_points = sum(row["nonplane_cache_point_count"] for row in rows)
        false_nonplane = sum(
            row["nonplane_false_assigned_point_count"] for row in rows
        )
        precision = tp / max(pred, 1)
        recall = tp / max(gt, 1)
        aggregate[method] = {
            "true_positive_planes": tp,
            "predicted_plane_count": pred,
            "gt_plane_count": gt,
            "plane_precision": precision,
            "plane_recall": recall,
            "Plane_F1": 2 * precision * recall / max(precision + recall, 1e-12),
            "matched_iou_mean": sum(
                row["accepted_matched_iou_sum"] for row in rows
            )
            / max(tp, 1),
            "partition_pair_precision": pair_precision,
            "partition_pair_recall": pair_recall,
            "partition_pair_f1": 2
            * pair_precision
            * pair_recall
            / max(pair_precision + pair_recall, 1e-12),
            "fragmentation_excess": sum(
                row["fragmentation_excess"] for row in rows
            ),
            "overmerge_excess": sum(row["overmerge_excess"] for row in rows),
            "gt_support_assignment_rate": assigned_gt
            / max(gt_support_points, 1),
            "gt_support_matched_coverage": matched_gt
            / max(gt_support_points, 1),
            "full_cache_label_rate": assigned_cache / max(cache_points, 1),
            "gt_plane_label_rate": gt_support_points / max(cache_points, 1),
            "nonplane_false_assignment_rate": false_nonplane
            / max(nonplane_points, 1),
        }
    summary = {
        "registration_id": REGISTRATION_ID,
        "status": "completed",
        "verdict": verdict,
        "scene_count": 128,
        "macro": macro,
        "aggregate": aggregate,
        "five_view_descriptive": {
            method: {
                "scene_count": sum(row["view_count"] == 5 for row in rows),
                "Plane_F1": finite_mean(
                    [row for row in rows if row["view_count"] == 5], "Plane_F1"
                ),
            }
            for method, rows in (("B16", rows_b), ("M16", rows_m))
        },
        "four_view_descriptive": {
            method: {
                "scene_count": sum(row["view_count"] == 4 for row in rows),
                "Plane_F1": finite_mean(
                    [row for row in rows if row["view_count"] == 4], "Plane_F1"
                ),
            }
            for method, rows in (("B16", rows_b), ("M16", rows_m))
        },
        "paired_effects": bootstrap["effects"],
        "fragmentation_mean_B": fragment_b,
        "fragmentation_mean_M": fragment_m,
        "overmerge_mean_B": overmerge_b,
        "overmerge_mean_M": overmerge_m,
        "prediction_immutability_passed": all(
            row["immutability_passed"] for row in metrics
        ),
        "forbidden_evaluator_calls": {
            "expansion": 0,
            "support_addition_removal_reassignment": 0,
            "plane_fit_refit": 0,
            "weighted_fit": 0,
            "RANSAC": 0,
            "plane_ID_modification": 0,
            "plane_parameter_modification": 0,
            "GT_guided_prediction_repair": 0,
        },
        "completed_utc": utc_now(),
    }
    write_json(OUT / "per_scene_metrics.json", metrics)
    write_csv(OUT / "per_scene_metrics.csv", metrics)
    write_json(OUT / "per_plane_metrics.json", plane_rows)
    write_csv(OUT / "per_plane_metrics.csv", plane_rows)
    paired_rows = [
        {
            "scene_id": b["scene_id"],
            "delta_plane_f1": m["Plane_F1"] - b["Plane_F1"],
            "delta_matched_iou": m["matched_iou_mean"] - b["matched_iou_mean"],
            "delta_partition_pair_f1": m["partition_pair_f1"]
            - b["partition_pair_f1"],
        }
        for b, m in zip(rows_b, rows_m, strict=True)
    ]
    write_json(OUT / "paired_effects.json", paired_rows)
    write_csv(OUT / "paired_effects.csv", paired_rows)
    write_json(OUT / "bootstrap_statistics.json", bootstrap)
    write_json(OUT / "summary.json", summary)
    for receipt in prediction_manifest["predictions"]:
        if sha256_file(Path(receipt["prediction_path"])) != receipt["whole_file_sha256"]:
            raise RuntimeError("prediction changed after evaluation")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("build-gt", "evaluate"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = build_final_gt(config) if args.phase == "build-gt" else evaluate_all(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
