"""Evaluate full-support split and fragment-identity diagnostic oracles.

GT is used only to select or score offline oracle decisions.  The exported
support variants retain the exact prediction-only partition candidates saved
by ``apply_intra_fragment_splitter.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from apply_intra_fragment_splitter import (
    component_statistics,
    file_sha256,
    refit_planes,
)
from evaluate_all_fragment_identity_end_to_end import (
    save_prediction,
)
from evaluate_bias_aware_fragment_association import exact_gt_join
from evaluate_global_plane_baselines import (
    evaluate_support_conditioned_partition,
    pair_count,
    public_partition_metrics,
)
from train_intra_fragment_splitter import fit_fragment_plane


OUTPUT_SCHEMA_VERSION = 1


def contiguous_nonnegative_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    result = np.full(labels.shape, -1, dtype=np.int32)
    valid_ids = sorted(int(value) for value in np.unique(labels) if value >= 0)
    for new_id, old_id in enumerate(valid_ids):
        result[labels == old_id] = new_id
    return result


def pairwise_metrics(
    pred_labels: np.ndarray, gt_labels: np.ndarray
) -> dict[str, float | int]:
    pred_labels = np.asarray(pred_labels, dtype=np.int64).reshape(-1)
    gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    if pred_labels.shape != gt_labels.shape:
        raise ValueError("prediction and GT labels must have identical shapes")
    valid = (pred_labels >= 0) & (gt_labels >= 0)
    pred = pred_labels[valid]
    gt = gt_labels[valid]
    if not len(pred):
        return {
            "labeled_point_count": 0,
            "predicted_component_count": 0,
            "gt_component_count": 0,
            "same_pred_pair_count": 0,
            "same_gt_pair_count": 0,
            "same_both_pair_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    pred_ids, pred_inverse = np.unique(pred, return_inverse=True)
    gt_ids, gt_inverse = np.unique(gt, return_inverse=True)
    flat = pred_inverse.astype(np.int64) * len(gt_ids) + gt_inverse
    contingency = np.bincount(
        flat,
        minlength=len(pred_ids) * len(gt_ids),
    ).reshape(len(pred_ids), len(gt_ids))
    same_both = int(sum(pair_count(value) for value in contingency.ravel()))
    same_pred = int(sum(pair_count(value) for value in contingency.sum(axis=1)))
    same_gt = int(sum(pair_count(value) for value in contingency.sum(axis=0)))
    precision = same_both / same_pred if same_pred else 0.0
    recall = same_both / same_gt if same_gt else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "labeled_point_count": int(len(pred)),
        "predicted_component_count": int(len(pred_ids)),
        "gt_component_count": int(len(gt_ids)),
        "same_pred_pair_count": same_pred,
        "same_gt_pair_count": same_gt,
        "same_both_pair_count": same_both,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compose_group_labels(
    group_indices: np.ndarray,
    candidate_component_labels: np.ndarray,
    split_decisions: np.ndarray,
) -> np.ndarray:
    group_indices = np.asarray(group_indices, dtype=np.int64).reshape(-1)
    candidate_component_labels = np.asarray(
        candidate_component_labels, dtype=np.int64
    ).reshape(-1)
    split_decisions = np.asarray(split_decisions, dtype=bool).reshape(-1)
    if group_indices.shape != candidate_component_labels.shape:
        raise ValueError("group and candidate label arrays must align")
    if np.any((group_indices < 0) != (candidate_component_labels < 0)):
        raise ValueError("unassigned group and candidate labels must agree")
    group_ids = sorted(
        int(value) for value in np.unique(group_indices) if value >= 0
    )
    if group_ids and max(group_ids) >= len(split_decisions):
        raise ValueError("split decision array does not cover every group")
    result = np.full(group_indices.shape, -1, dtype=np.int32)
    next_id = 0
    for group_id in group_ids:
        mask = group_indices == group_id
        local = (
            candidate_component_labels[mask]
            if split_decisions[group_id]
            else np.zeros(int(mask.sum()), dtype=np.int64)
        )
        for component in np.unique(local):
            result_indices = np.flatnonzero(mask)[local == component]
            result[result_indices] = next_id
            next_id += 1
    if np.any(result[group_indices >= 0] < 0):
        raise RuntimeError("not every grouped point received a composed label")
    if np.any(result[group_indices < 0] >= 0):
        raise RuntimeError("unassigned support received a composed label")
    return result


def dominant_gt_identity_labels(
    fragment_labels: np.ndarray,
    gt_labels: np.ndarray,
    *,
    low_purity_threshold: float = 0.8,
    min_fragment_points: int = 1,
) -> tuple[np.ndarray, dict]:
    fragment_labels = np.asarray(fragment_labels, dtype=np.int64).reshape(-1)
    gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    if fragment_labels.shape != gt_labels.shape:
        raise ValueError("fragment and GT labels must align")
    result = np.full(fragment_labels.shape, -1, dtype=np.int32)
    identity_by_key: dict[tuple[str, int], int] = {}
    rows = []
    rejected_small_count = 0
    rejected_small_points = 0
    for fragment_id in sorted(
        int(value) for value in np.unique(fragment_labels) if value >= 0
    ):
        mask = fragment_labels == fragment_id
        point_count = int(mask.sum())
        local_gt = gt_labels[mask]
        valid = local_gt >= 0
        if valid.any():
            values, counts = np.unique(local_gt[valid], return_counts=True)
            dominant_index = int(np.argmax(counts))
            dominant_gt = int(values[dominant_index])
            purity = float(counts[dominant_index] / valid.sum())
            identity_key = ("gt", dominant_gt)
        else:
            dominant_gt = -1
            purity = 0.0
            identity_key = ("unmatched_fragment", fragment_id)
        eligible = point_count >= int(min_fragment_points)
        if eligible:
            if identity_key not in identity_by_key:
                identity_by_key[identity_key] = len(identity_by_key)
            result[mask] = identity_by_key[identity_key]
            oracle_identity = int(identity_by_key[identity_key])
        else:
            rejected_small_count += 1
            rejected_small_points += point_count
            oracle_identity = -1
        rows.append(
            {
                "fragment_id": fragment_id,
                "point_count": point_count,
                "matched_gt_point_count": int(valid.sum()),
                "dominant_gt": dominant_gt,
                "purity": purity,
                "low_purity": bool(purity < low_purity_threshold),
                "eligible": bool(eligible),
                "oracle_identity": oracle_identity,
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    low_rows = [row for row in eligible_rows if row["low_purity"]]
    matched_points = sum(
        row["matched_gt_point_count"] for row in eligible_rows
    )
    low_matched_points = sum(
        row["matched_gt_point_count"] for row in low_rows
    )
    audit = {
        "fragment_count": int(len(rows)),
        "eligible_fragment_count": int(len(eligible_rows)),
        "min_fragment_points": int(min_fragment_points),
        "rejected_small_fragment_count": int(rejected_small_count),
        "rejected_small_point_count": int(rejected_small_points),
        "oracle_identity_count": int(len(identity_by_key)),
        "low_purity_threshold": float(low_purity_threshold),
        "low_purity_fragment_count": int(len(low_rows)),
        "low_purity_fragment_rate": (
            float(len(low_rows) / len(eligible_rows)) if eligible_rows else 0.0
        ),
        "low_purity_matched_point_rate": (
            float(low_matched_points / matched_points)
            if matched_points
            else 0.0
        ),
        "mean_fragment_purity": (
            float(np.mean([row["purity"] for row in eligible_rows]))
            if eligible_rows
            else 0.0
        ),
        "point_weighted_fragment_purity": (
            float(
                sum(
                    row["purity"] * row["matched_gt_point_count"]
                    for row in rows
                )
                / matched_points
            )
            if matched_points
            else 0.0
        ),
        "fragments": rows,
    }
    return result, audit


def partition_metrics(
    points: np.ndarray,
    labels: np.ndarray,
    gt_record_labels: np.ndarray,
    gt_normals: np.ndarray,
    *,
    match_iou: float,
    fragmentation_iou: float,
    min_observed_plane_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    labels = contiguous_nonnegative_labels(labels)
    cluster_ids = sorted(
        int(value) for value in np.unique(labels) if value >= 0
    )
    if cluster_ids != list(range(len(cluster_ids))):
        raise ValueError("partition labels must be contiguous and zero-based")
    normals = np.zeros((len(cluster_ids), 3), dtype=np.float32)
    offsets = np.zeros((len(cluster_ids),), dtype=np.float32)
    counts = np.zeros((len(cluster_ids),), dtype=np.int32)
    for cluster_id in cluster_ids:
        selected = np.asarray(points)[labels == cluster_id]
        normal, offset = fit_fragment_plane(selected)
        normals[cluster_id] = normal
        offsets[cluster_id] = offset
        counts[cluster_id] = len(selected)
    metrics = evaluate_support_conditioned_partition(
        labels,
        normals,
        gt_record_labels,
        gt_normals,
        match_iou,
        fragmentation_iou,
        min_observed_plane_points,
    )
    precision = float(metrics["support_conditioned_plane_precision"])
    recall = float(metrics["support_conditioned_plane_recall_observed"])
    metrics["support_conditioned_plane_f1"] = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    metrics.update(public_partition_metrics(labels, gt_record_labels))
    metrics["assignment_rate"] = float((labels >= 0).mean())
    assigned = labels >= 0
    if assigned.any():
        residual = np.abs(
            np.sum(points[assigned] * normals[labels[assigned]], axis=1)
            + offsets[labels[assigned]]
        )
        metrics["point_to_plane_residual"] = float(residual.mean())
    else:
        metrics["point_to_plane_residual"] = float("nan")
    return labels, normals, offsets, counts, metrics


def _group_weighted_mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if np.isfinite(row[key])]
    return float(np.mean(values)) if values else float("nan")


def build_split_oracle(
    support: dict,
    gt_record_labels: np.ndarray,
    *,
    split_gain_margin: float,
) -> tuple[np.ndarray, list[dict], dict]:
    metadata = json.loads(
        str(support["intra_fragment_split_metadata_json"].item())
    )
    if int(metadata.get("schema_version", 0)) < 2:
        raise ValueError("split diagnostics schema v2 or newer is required")
    group_indices = support["intra_fragment_candidate_group_indices"].astype(
        np.int64
    )
    candidate_labels = support[
        "intra_fragment_candidate_component_labels"
    ].astype(np.int64)
    rows_by_index = {
        int(row["group_index"]): row for row in metadata["groups"]
    }
    group_count = len(rows_by_index)
    oracle_decisions = np.zeros((group_count,), dtype=bool)
    learned_decisions = np.zeros((group_count,), dtype=bool)
    group_rows = []
    for group_index in range(group_count):
        metadata_row = rows_by_index[group_index]
        mask = group_indices == group_index
        local_gt = gt_record_labels[mask]
        unsplit = pairwise_metrics(
            np.zeros(int(mask.sum()), dtype=np.int32), local_gt
        )
        candidate = pairwise_metrics(candidate_labels[mask], local_gt)
        learned_split = bool(
            metadata_row["gate_decision"]
            and int(metadata_row["candidate_component_count"]) > 1
        )
        learned_decisions[group_index] = learned_split
        gain = float(candidate["f1"] - unsplit["f1"])
        oracle_split = bool(
            metadata_row["eligible"]
            and int(metadata_row["candidate_component_count"]) > 1
            and gain > float(split_gain_margin)
        )
        oracle_decisions[group_index] = oracle_split
        learned = candidate if learned_split else unsplit
        group_rows.append(
            {
                **metadata_row,
                "gt_labeled_point_count": int(candidate["labeled_point_count"]),
                "gt_component_count": int(candidate["gt_component_count"]),
                "unsplit_pairwise_precision": float(unsplit["precision"]),
                "unsplit_pairwise_recall": float(unsplit["recall"]),
                "unsplit_pairwise_f1": float(unsplit["f1"]),
                "candidate_pairwise_precision": float(candidate["precision"]),
                "candidate_pairwise_recall": float(candidate["recall"]),
                "candidate_pairwise_f1": float(candidate["f1"]),
                "learned_pairwise_precision": float(learned["precision"]),
                "learned_pairwise_recall": float(learned["recall"]),
                "learned_pairwise_f1": float(learned["f1"]),
                "split_gain": gain,
                "candidate_component_count_error": abs(
                    int(candidate["predicted_component_count"])
                    - int(candidate["gt_component_count"])
                ),
                "oracle_split_decision": oracle_split,
            }
        )
    unsplit_labels = compose_group_labels(
        group_indices,
        candidate_labels,
        np.zeros(group_count, dtype=bool),
    )
    candidate_all_labels = compose_group_labels(
        group_indices,
        candidate_labels,
        np.ones(group_count, dtype=bool),
    )
    learned_labels = compose_group_labels(
        group_indices,
        candidate_labels,
        learned_decisions,
    )
    oracle_labels = compose_group_labels(
        group_indices,
        candidate_labels,
        oracle_decisions,
    )
    aggregate = {}
    for name, labels, row_prefix in (
        ("unsplit_groups", unsplit_labels, "unsplit"),
        ("candidate_all", candidate_all_labels, "candidate"),
        ("learned_gate", learned_labels, "learned"),
        ("split_gain_oracle", oracle_labels, None),
    ):
        weighted = pairwise_metrics(labels, gt_record_labels)
        if row_prefix is None:
            group_precision = _group_weighted_mean(
                [
                    {
                        "value": (
                            row["candidate_pairwise_precision"]
                            if row["oracle_split_decision"]
                            else row["unsplit_pairwise_precision"]
                        )
                    }
                    for row in group_rows
                ],
                "value",
            )
            group_recall = _group_weighted_mean(
                [
                    {
                        "value": (
                            row["candidate_pairwise_recall"]
                            if row["oracle_split_decision"]
                            else row["unsplit_pairwise_recall"]
                        )
                    }
                    for row in group_rows
                ],
                "value",
            )
            group_f1 = _group_weighted_mean(
                [
                    {
                        "value": (
                            row["candidate_pairwise_f1"]
                            if row["oracle_split_decision"]
                            else row["unsplit_pairwise_f1"]
                        )
                    }
                    for row in group_rows
                ],
                "value",
            )
        else:
            group_precision = _group_weighted_mean(
                group_rows, f"{row_prefix}_pairwise_precision"
            )
            group_recall = _group_weighted_mean(
                group_rows, f"{row_prefix}_pairwise_recall"
            )
            group_f1 = _group_weighted_mean(
                group_rows, f"{row_prefix}_pairwise_f1"
            )
        group_component_stats = []
        for group_index in range(group_count):
            mask = group_indices == group_index
            group_component_stats.append(component_statistics(labels[mask]))
        aggregate[name] = {
            "point_pair_weighted": weighted,
            "group_weighted": {
                "precision": group_precision,
                "recall": group_recall,
                "f1": group_f1,
            },
            "output_fragment_count": int(len(np.unique(labels))),
            "tiny_component_count_lt16": int(
                sum(
                    row["tiny_component_count_lt16"]
                    for row in group_component_stats
                )
            ),
            "tiny_component_count_lt32": int(
                sum(
                    row["tiny_component_count_lt32"]
                    for row in group_component_stats
                )
            ),
            "tiny_component_count_lt64": int(
                sum(
                    row["tiny_component_count_lt64"]
                    for row in group_component_stats
                )
            ),
        }
    aggregate["decision_counts"] = {
        "group_count": group_count,
        "learned_split_count": int(learned_decisions.sum()),
        "oracle_split_count": int(oracle_decisions.sum()),
        "learned_oracle_disagreement_count": int(
            (learned_decisions != oracle_decisions).sum()
        ),
        "learned_false_split_vs_oracle": int(
            (learned_decisions & ~oracle_decisions).sum()
        ),
        "learned_false_no_split_vs_oracle": int(
            (~learned_decisions & oracle_decisions).sum()
        ),
    }
    return oracle_labels, group_rows, aggregate


def save_support_variant(
    path: Path,
    source: dict,
    labels: np.ndarray,
    metadata: dict,
) -> None:
    arrays = dict(source)
    labels = contiguous_nonnegative_labels(labels)
    arrays["point_plane_ids"] = labels
    arrays.update(refit_planes(arrays["points"], labels))
    arrays["split_gain_oracle_metadata_json"] = np.asarray(
        json.dumps(metadata), dtype=object
    )
    np.savez_compressed(path, **arrays)


def replace_batch_support(
    batch: dict,
    scene_to_support: dict[str, Path],
) -> dict:
    result = copy.deepcopy(batch)
    for item in result["items"]:
        if item["id"] not in scene_to_support:
            continue
        path = scene_to_support[item["id"]].resolve()
        item["artifacts"]["direct_support_records"] = {
            "path": str(path),
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "expected_sha256": "",
            "checksum_matches": None,
        }
    return result


def _mean_metric(scene_rows: list[dict], variant: str, key: str) -> float:
    values = np.asarray(
        [row["variants"][variant]["metrics"][key] for row in scene_rows],
        dtype=np.float64,
    )
    return (
        float(np.nanmean(values))
        if np.isfinite(values).any()
        else float("nan")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate split-gain and dominant-GT identity oracles."
    )
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--diagnostic_support_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--split_gain_margin", type=float, default=0.0)
    parser.add_argument("--low_purity_threshold", type=float, default=0.8)
    parser.add_argument("--min_identity_fragment_points", type=int, default=64)
    parser.add_argument("--match_iou", type=float, default=0.5)
    parser.add_argument("--fragmentation_iou", type=float, default=0.1)
    parser.add_argument("--min_observed_plane_points", type=int, default=64)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_json).resolve()
    diagnostic_dir = Path(args.diagnostic_support_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    split_support_dir = output_dir / "split_oracle_support"
    split_support_dir.mkdir()
    prediction_root = output_dir / "support_predictions"
    prediction_root.mkdir()
    variant_names = (
        "direct_support",
        "learned_gate_support",
        "split_oracle_support",
        "identity_oracle_direct",
        "identity_oracle_learned_gate",
        "identity_oracle_split_oracle",
        "pointwise_gt_evaluator_oracle",
    )
    prediction_dirs = {}
    for name in variant_names:
        prediction_dirs[name] = prediction_root / name
        prediction_dirs[name].mkdir()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = [
        scene for scene in manifest["scenes"] if scene["split"] == args.split
    ]
    base_batch_path = Path(manifest["config"]["batch_json"]).resolve()
    base_batch = json.loads(base_batch_path.read_text(encoding="utf-8"))
    item_by_scene = {item["id"]: item for item in base_batch["items"]}
    scene_rows = []
    all_group_rows = []
    split_support_by_scene = {}
    for scene in selected:
        scene_name = scene["scene"]
        diagnostic_path = (
            diagnostic_dir / f"{scene_name}_split_support.npz"
        )
        if not diagnostic_path.exists():
            raise FileNotFoundError(diagnostic_path)
        with np.load(diagnostic_path, allow_pickle=True) as raw:
            support = {key: raw[key] for key in raw.files}
        with np.load(Path(scene["gt_npz"]), allow_pickle=False) as raw:
            gt = {key: raw[key] for key in raw.files}
        gt_record_labels = exact_gt_join(
            support["alignment_view_indices"].astype(np.int64),
            support["pointmap_pixel_xy"].astype(np.int64),
            gt["view_indices"].astype(np.int64),
            gt["pixel_xy"].astype(np.int64),
            gt["point_plane_ids"].astype(np.int64),
        )
        split_oracle_labels, group_rows, split_audit = build_split_oracle(
            support,
            gt_record_labels,
            split_gain_margin=args.split_gain_margin,
        )
        for row in group_rows:
            row["scene"] = scene_name
        all_group_rows.extend(group_rows)
        direct_labels = support[
            "original_point_plane_ids_before_split"
        ].astype(np.int64)
        learned_labels = support["point_plane_ids"].astype(np.int64)
        base_variants = {
            "direct_support": direct_labels,
            "learned_gate_support": learned_labels,
            "split_oracle_support": split_oracle_labels,
        }
        variants = {}
        oracle_variant_labels = {}
        for name, labels in base_variants.items():
            (
                contiguous,
                normals,
                offsets,
                counts,
                metrics,
            ) = partition_metrics(
                support["points"],
                labels,
                gt_record_labels,
                gt["plane_normals"],
                match_iou=args.match_iou,
                fragmentation_iou=args.fragmentation_iou,
                min_observed_plane_points=args.min_observed_plane_points,
            )
            identity_labels, purity_audit = dominant_gt_identity_labels(
                contiguous,
                gt_record_labels,
                low_purity_threshold=args.low_purity_threshold,
                min_fragment_points=args.min_identity_fragment_points,
            )
            (
                identity_labels,
                identity_normals,
                identity_offsets,
                identity_counts,
                identity_metrics,
            ) = partition_metrics(
                support["points"],
                identity_labels,
                gt_record_labels,
                gt["plane_normals"],
                match_iou=args.match_iou,
                fragmentation_iou=args.fragmentation_iou,
                min_observed_plane_points=args.min_observed_plane_points,
            )
            oracle_name = f"identity_oracle_{name.removesuffix('_support')}"
            variants[name] = {
                "metrics": metrics,
                "fragment_purity": purity_audit,
            }
            variants[oracle_name] = {
                "metrics": identity_metrics,
                "fragment_purity": purity_audit,
            }
            oracle_variant_labels[oracle_name] = identity_labels
            save_prediction(
                prediction_dirs[name]
                / f"{scene_name}_all_fragment_identity.npz",
                support,
                contiguous,
                normals,
                offsets,
                counts,
                0.0,
                name,
            )
            save_prediction(
                prediction_dirs[oracle_name]
                / f"{scene_name}_all_fragment_identity.npz",
                support,
                identity_labels,
                identity_normals,
                identity_offsets,
                identity_counts,
                0.0,
                oracle_name,
            )
        (
            pointwise_gt_labels,
            pointwise_gt_normals,
            pointwise_gt_offsets,
            pointwise_gt_counts,
            pointwise_gt_metrics,
        ) = partition_metrics(
            support["points"],
            gt_record_labels,
            gt_record_labels,
            gt["plane_normals"],
            match_iou=args.match_iou,
            fragmentation_iou=args.fragmentation_iou,
            min_observed_plane_points=args.min_observed_plane_points,
        )
        variants["pointwise_gt_evaluator_oracle"] = {
            "metrics": pointwise_gt_metrics,
            "fragment_purity": None,
            "role": "evaluator_sanity_check_only",
        }
        save_prediction(
            prediction_dirs["pointwise_gt_evaluator_oracle"]
            / f"{scene_name}_all_fragment_identity.npz",
            support,
            pointwise_gt_labels,
            pointwise_gt_normals,
            pointwise_gt_offsets,
            pointwise_gt_counts,
            0.0,
            "pointwise_gt_evaluator_oracle",
        )
        split_path = (
            split_support_dir / f"{scene_name}_split_oracle_support.npz"
        )
        save_support_variant(
            split_path,
            support,
            split_oracle_labels,
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "selection": "candidate_pairwise_f1_minus_unsplit_pairwise_f1",
                "split_gain_margin": float(args.split_gain_margin),
                "gt_use": "offline_oracle_only",
            },
        )
        split_support_by_scene[scene_name] = split_path
        scene_rows.append(
            {
                "scene": scene_name,
                "diagnostic_support_npz": str(diagnostic_path),
                "split_oracle_support_npz": str(split_path),
                "split_audit": split_audit,
                "variants": variants,
            }
        )
        print(
            json.dumps(
                {
                    "scene": scene_name,
                    "learned_splits": split_audit["decision_counts"][
                        "learned_split_count"
                    ],
                    "oracle_splits": split_audit["decision_counts"][
                        "oracle_split_count"
                    ],
                    "learned_pair_f1": variants["learned_gate_support"][
                        "metrics"
                    ]["support_partition_pairwise_f1"],
                    "split_oracle_pair_f1": variants[
                        "split_oracle_support"
                    ]["metrics"]["support_partition_pairwise_f1"],
                }
            ),
            flush=True,
        )

    split_batch = replace_batch_support(base_batch, split_support_by_scene)
    split_batch["batch_name"] = (
        f"{base_batch.get('batch_name', 'batch')}_split_gain_oracle_{args.split}"
    )
    split_batch["derived_from"] = str(base_batch_path)
    split_batch_path = output_dir / "split_oracle_batch_execution.json"
    split_batch_path.write_text(
        json.dumps(split_batch, indent=2), encoding="utf-8"
    )
    pressure_batch = copy.deepcopy(base_batch)
    pressure_batch["items"] = [
        copy.deepcopy(item_by_scene[scene["scene"]]) for scene in selected
    ]
    pressure_batch["batch_name"] = (
        f"{base_batch.get('batch_name', 'batch')}_{args.split}_pressure_only"
    )
    pressure_batch["derived_from"] = str(base_batch_path)
    pressure_batch_path = output_dir / "pressure_batch_execution.json"
    pressure_batch_path.write_text(
        json.dumps(pressure_batch, indent=2), encoding="utf-8"
    )
    group_audit_path = output_dir / "group_audit.json"
    group_audit_path.write_text(
        json.dumps(all_group_rows, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    metric_keys = sorted(
        scene_rows[0]["variants"]["direct_support"]["metrics"]
    )
    mean = {
        variant: {
            key: _mean_metric(scene_rows, variant, key)
            for key in metric_keys
        }
        for variant in variant_names
    }
    summary = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "task": "full_support_split_gain_and_dominant_gt_identity_oracles",
        "gt_in_inference_features": False,
        "oracle_gt_use": "offline decision and evaluation only",
        "scene_count": len(scene_rows),
        "mean": mean,
        "scenes": scene_rows,
        "config": {
            **vars(args),
            "manifest_json": str(manifest_path),
            "diagnostic_support_dir": str(diagnostic_dir),
            "output_dir": str(output_dir),
        },
        "outputs": {
            "summary_json": str(output_dir / "summary.json"),
            "group_audit_json": str(group_audit_path),
            "split_oracle_batch_json": str(split_batch_path),
            "pressure_batch_json": str(pressure_batch_path),
            "split_oracle_support_dir": str(split_support_dir),
            "support_prediction_dirs": {
                key: str(value) for key, value in prediction_dirs.items()
            },
        },
        "limitations": [
            "The split oracle selects only among the frozen prediction-only partition candidates.",
            "The identity oracle merges whole fragments by dominant GT and cannot repair impurity inside a fragment.",
            "These support-domain metrics precede the frozen common-cache bounded evaluator.",
            "The pointwise-GT variant is only an evaluator sanity check and is not a permissible inference method or distillation target.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summary["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
