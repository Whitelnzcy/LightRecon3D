"""Evaluate plane primitives against point-aligned Structured3D GT labels."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def linear_sum_assignment(cost):
    """Exact rectangular Hungarian matching without a SciPy dependency."""

    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("assignment cost must be a matrix")
    original_rows, original_cols = cost.shape
    transposed = original_rows > original_cols
    if transposed:
        cost = cost.T
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(cols + 1, dtype=np.float64)
    p = np.zeros(cols + 1, dtype=np.int64)
    way = np.zeros(cols + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        p[0] = row
        min_values = np.full(cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(cols + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, cols + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if current < min_values[column]:
                    min_values[column] = current
                    way[column] = column0
                if min_values[column] < delta:
                    delta = min_values[column]
                    column1 = column
            for column in range(cols + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_values[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    matched_rows = []
    matched_cols = []
    for column in range(1, cols + 1):
        if p[column] != 0:
            matched_rows.append(int(p[column] - 1))
            matched_cols.append(int(column - 1))
    matched_rows = np.asarray(matched_rows, dtype=np.int64)
    matched_cols = np.asarray(matched_cols, dtype=np.int64)
    order = np.argsort(matched_rows)
    matched_rows, matched_cols = matched_rows[order], matched_cols[order]
    if transposed:
        return matched_cols, matched_rows
    return matched_rows, matched_cols


def iou_matrix(pred, gt, pred_ids, gt_ids):
    result = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float64)
    for i, pid in enumerate(pred_ids):
        pm = pred == pid
        for j, gid in enumerate(gt_ids):
            gm = gt == gid
            union = int((pm | gm).sum())
            result[i, j] = int((pm & gm).sum()) / union if union else 0.0
    return result


def pair_count(count):
    count = int(count)
    return count * (count - 1) // 2


def public_partition_metrics(pred_labels, gt_labels):
    """Compute standard segmentation metrics on all GT-planar points.

    The public plane-reconstruction literature commonly reports Variation of
    Information (VOI), Rand Index (RI), and Segmentation Covering (SC).  This
    point-aligned implementation uses every point with a valid GT plane label.
    Predicted ``-1`` labels are retained as one unassigned segment instead of
    being dropped, so incomplete methods cannot improve their score by omitting
    difficult points.

    VOI is measured with natural logarithms (nats).  SC is reported in both
    directions, plus their arithmetic mean, to make the convention explicit.
    """

    pred_labels = np.asarray(pred_labels, dtype=np.int64).reshape(-1)
    gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    if pred_labels.shape != gt_labels.shape:
        raise ValueError("prediction and GT labels must have identical shapes")

    valid = gt_labels >= 0
    pred = pred_labels[valid]
    gt = gt_labels[valid]
    point_count = int(len(gt))
    if point_count == 0:
        return {
            "segmentation_evaluation_points": 0,
            "segmentation_pred_segment_count": 0,
            "segmentation_gt_segment_count": 0,
            "segmentation_voi_nats": float("nan"),
            "segmentation_rand_index": float("nan"),
            "segmentation_covering_gt_by_pred": float("nan"),
            "segmentation_covering_pred_by_gt": float("nan"),
            "segmentation_covering_symmetric": float("nan"),
        }

    pred_ids, pred_inverse = np.unique(pred, return_inverse=True)
    gt_ids, gt_inverse = np.unique(gt, return_inverse=True)
    gt_count = len(gt_ids)
    flat = pred_inverse.astype(np.int64) * gt_count + gt_inverse
    contingency = np.bincount(
        flat,
        minlength=len(pred_ids) * gt_count,
    ).reshape(len(pred_ids), gt_count).astype(np.int64, copy=False)

    row_counts = contingency.sum(axis=1)
    column_counts = contingency.sum(axis=0)
    probability = contingency.astype(np.float64) / point_count
    row_probability = row_counts.astype(np.float64) / point_count
    column_probability = column_counts.astype(np.float64) / point_count

    nonzero = probability > 0
    mutual_information = float(
        np.sum(
            probability[nonzero]
            * np.log(
                probability[nonzero]
                / (
                    row_probability[:, None]
                    * column_probability[None, :]
                )[nonzero]
            )
        )
    )
    pred_entropy = float(-np.sum(row_probability * np.log(row_probability)))
    gt_entropy = float(-np.sum(column_probability * np.log(column_probability)))
    voi = max(0.0, pred_entropy + gt_entropy - 2.0 * mutual_information)

    total_pairs = pair_count(point_count)
    same_both = sum(pair_count(value) for value in contingency.ravel())
    same_pred = sum(pair_count(value) for value in row_counts)
    same_gt = sum(pair_count(value) for value in column_counts)
    different_both = total_pairs - same_pred - same_gt + same_both
    rand_index = (
        float((same_both + different_both) / total_pairs)
        if total_pairs
        else 1.0
    )

    union = row_counts[:, None] + column_counts[None, :] - contingency
    iou = np.divide(
        contingency,
        union,
        out=np.zeros_like(contingency, dtype=np.float64),
        where=union > 0,
    )
    covering_gt_by_pred = float(
        np.sum(column_counts * iou.max(axis=0)) / point_count
    )
    covering_pred_by_gt = float(
        np.sum(row_counts * iou.max(axis=1)) / point_count
    )

    return {
        "segmentation_evaluation_points": point_count,
        "segmentation_pred_segment_count": int(len(pred_ids)),
        "segmentation_gt_segment_count": int(len(gt_ids)),
        "segmentation_voi_nats": float(voi),
        "segmentation_rand_index": rand_index,
        "segmentation_covering_gt_by_pred": covering_gt_by_pred,
        "segmentation_covering_pred_by_gt": covering_pred_by_gt,
        "segmentation_covering_symmetric": float(
            0.5 * (covering_gt_by_pred + covering_pred_by_gt)
        ),
    }


def evaluate_support_conditioned_partition(
    pred_labels,
    pred_normals,
    gt_labels,
    gt_normals,
    match_iou=0.5,
    fragmentation_iou=0.1,
    min_observed_plane_points=1,
):
    """Evaluate identity only where a sparse method emitted support.

    Full-cache coverage and IoU remain separate metrics. Restricting the domain
    to predicted support separates partition quality from sparse sampling,
    while the reported coverage keeps a method from obtaining a good identity
    score by predicting very few points.
    """

    pred_labels = np.asarray(pred_labels, dtype=np.int32).reshape(-1)
    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    if pred_labels.shape != gt_labels.shape:
        raise ValueError("prediction and GT labels must have identical shapes")
    if min_observed_plane_points < 1:
        raise ValueError("min_observed_plane_points must be positive")

    pred_ids = np.asarray(
        sorted(int(value) for value in np.unique(pred_labels) if value >= 0),
        dtype=np.int32,
    )
    all_gt_ids = np.asarray(
        sorted(int(value) for value in np.unique(gt_labels) if value >= 0),
        dtype=np.int32,
    )
    assigned = pred_labels >= 0
    labeled_assigned = assigned & (gt_labels >= 0)
    observed_gt_ids = np.asarray(
        [
            int(gt_id)
            for gt_id in all_gt_ids
            if int((labeled_assigned & (gt_labels == gt_id)).sum())
            >= int(min_observed_plane_points)
        ],
        dtype=np.int32,
    )

    conditioned_matrix = iou_matrix(
        pred_labels[assigned],
        gt_labels[assigned],
        pred_ids,
        observed_gt_ids,
    )
    if conditioned_matrix.size:
        rows, cols = linear_sum_assignment(1.0 - conditioned_matrix)
    else:
        rows, cols = np.array([], int), np.array([], int)
    matched = [
        (
            int(pred_ids[row]),
            int(observed_gt_ids[column]),
            float(conditioned_matrix[row, column]),
        )
        for row, column in zip(rows, cols)
    ]
    accepted = [item for item in matched if item[2] >= match_iou]
    angular = []
    for pred_id, gt_id, _ in accepted:
        angular.append(
            float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            abs(float(pred_normals[pred_id] @ gt_normals[gt_id])),
                            0,
                            1,
                        )
                    )
                )
            )
        )

    fragments = [
        int((conditioned_matrix[:, column] >= fragmentation_iou).sum())
        for column in range(len(observed_gt_ids))
    ]
    overmerges = [
        int((conditioned_matrix[row, :] >= fragmentation_iou).sum())
        for row in range(len(pred_ids))
    ]

    contingency = np.zeros((len(pred_ids), len(all_gt_ids)), dtype=np.int64)
    labeled_pred = pred_labels[labeled_assigned]
    labeled_gt = gt_labels[labeled_assigned]
    for row, pred_id in enumerate(pred_ids):
        pred_mask = labeled_pred == pred_id
        for column, gt_id in enumerate(all_gt_ids):
            contingency[row, column] = int(
                (pred_mask & (labeled_gt == gt_id)).sum()
            )
    same_both = sum(pair_count(value) for value in contingency.ravel())
    same_pred = sum(pair_count(value) for value in contingency.sum(axis=1))
    same_gt = sum(pair_count(value) for value in contingency.sum(axis=0))
    pairwise_precision = same_both / same_pred if same_pred else 0.0
    pairwise_recall = same_both / same_gt if same_gt else 0.0
    pairwise_f1 = (
        2.0 * pairwise_precision * pairwise_recall
        / (pairwise_precision + pairwise_recall)
        if pairwise_precision + pairwise_recall
        else 0.0
    )
    labeled_count = int(labeled_assigned.sum())
    pred_purity = (
        float(contingency.max(axis=1).sum()) / labeled_count
        if labeled_count and contingency.shape[1]
        else 0.0
    )
    gt_completeness = (
        float(contingency.max(axis=0).sum()) / labeled_count
        if labeled_count and contingency.shape[0]
        else 0.0
    )
    partition_f1 = (
        2.0 * pred_purity * gt_completeness / (pred_purity + gt_completeness)
        if pred_purity + gt_completeness
        else 0.0
    )
    assigned_count = int(assigned.sum())
    true_positives = len(accepted)
    return {
        "assigned_point_count": assigned_count,
        "assigned_gt_labeled_point_count": labeled_count,
        "assigned_gt_label_rate": (
            labeled_count / assigned_count if assigned_count else 0.0
        ),
        "support_conditioned_min_gt_points": int(min_observed_plane_points),
        "support_conditioned_pred_plane_count": int(len(pred_ids)),
        "support_conditioned_gt_plane_count": int(len(observed_gt_ids)),
        "support_conditioned_true_positive_planes": int(true_positives),
        "support_conditioned_plane_precision": (
            true_positives / len(pred_ids) if len(pred_ids) else 0.0
        ),
        "support_conditioned_plane_recall_observed": (
            true_positives / len(observed_gt_ids) if len(observed_gt_ids) else 0.0
        ),
        "support_conditioned_plane_recall_all_gt": (
            true_positives / len(all_gt_ids) if len(all_gt_ids) else 0.0
        ),
        "support_conditioned_matched_iou": (
            float(np.mean([item[2] for item in accepted])) if accepted else 0.0
        ),
        "support_conditioned_normal_angular_error_deg": (
            float(np.mean(angular)) if angular else float("nan")
        ),
        "support_conditioned_fragmentation_excess": int(
            sum(max(0, count - 1) for count in fragments)
        ),
        "support_conditioned_overmerge_excess": int(
            sum(max(0, count - 1) for count in overmerges)
        ),
        "support_partition_pairwise_precision": float(pairwise_precision),
        "support_partition_pairwise_recall": float(pairwise_recall),
        "support_partition_pairwise_f1": float(pairwise_f1),
        "support_partition_pred_purity": float(pred_purity),
        "support_partition_gt_completeness": float(gt_completeness),
        "support_partition_purity_completeness_f1": float(partition_f1),
    }


def evaluate_arrays(points, pred_labels, pred_normals, pred_offsets, gt_labels,
                    gt_normals, match_iou=0.5, fragmentation_iou=0.1,
                    min_observed_plane_points=1):
    pred_ids = np.asarray(sorted(int(x) for x in np.unique(pred_labels) if x >= 0))
    gt_ids = np.asarray(sorted(int(x) for x in np.unique(gt_labels) if x >= 0))
    matrix = iou_matrix(pred_labels, gt_labels, pred_ids, gt_ids)
    rows, cols = linear_sum_assignment(1.0 - matrix) if matrix.size else (np.array([], int), np.array([], int))
    matched = [(int(pred_ids[r]), int(gt_ids[c]), float(matrix[r, c])) for r, c in zip(rows, cols)]
    accepted = [(p, g, score) for p, g, score in matched if score >= match_iou]
    angular, residual, matched_iou = [], [], []
    for pid, gid, score in accepted:
        pn, gn = pred_normals[pid], gt_normals[gid]
        angular.append(float(np.degrees(np.arccos(np.clip(abs(float(pn @ gn)), 0, 1)))))
        support = pred_labels == pid
        residual.append(float(np.abs(points[support] @ pn + pred_offsets[pid]).mean()))
        matched_iou.append(score)
    fragments = [int((matrix[:, j] >= fragmentation_iou).sum()) for j in range(len(gt_ids))]
    overmerges = [int((matrix[i, :] >= fragmentation_iou).sum()) for i in range(len(pred_ids))]
    gt_support = gt_labels >= 0
    covered = gt_support & (pred_labels >= 0)
    result = {
        "pred_plane_count": int(len(pred_ids)), "gt_plane_count": int(len(gt_ids)),
        "plane_count_error": int(len(pred_ids) - len(gt_ids)),
        "plane_count_abs_error": int(abs(len(pred_ids) - len(gt_ids))),
        "true_positive_planes": len(accepted),
        "plane_precision": len(accepted) / len(pred_ids) if len(pred_ids) else 0.0,
        "plane_recall": len(accepted) / len(gt_ids) if len(gt_ids) else 0.0,
        "normal_angular_error_deg": float(np.mean(angular)) if angular else float("nan"),
        "point_to_plane_residual": float(np.mean(residual)) if residual else float("nan"),
        "support_matched_iou": float(np.mean(matched_iou)) if matched_iou else 0.0,
        "support_coverage": int(covered.sum()) / int(gt_support.sum()) if gt_support.any() else 0.0,
        "fragmentation_excess": int(sum(max(0, count - 1) for count in fragments)),
        "fragmented_gt_planes": int(sum(count > 1 for count in fragments)),
        "overmerge_excess": int(sum(max(0, count - 1) for count in overmerges)),
        "overmerged_pred_planes": int(sum(count > 1 for count in overmerges)),
    }
    result.update(
        evaluate_support_conditioned_partition(
            pred_labels,
            pred_normals,
            gt_labels,
            gt_normals,
            match_iou,
            fragmentation_iou,
            min_observed_plane_points,
        )
    )
    result.update(public_partition_metrics(pred_labels, gt_labels))
    return result


def main():
    parser = argparse.ArgumentParser("Evaluate globally aligned plane primitive NPZ files")
    parser.add_argument("--gt_npz", required=True,
                        help="Point-aligned NPZ with points, point_plane_ids and plane_normals")
    parser.add_argument("--pred_npz", nargs="+", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--match_iou", type=float, default=.5)
    parser.add_argument("--fragmentation_iou", type=float, default=.1)
    parser.add_argument("--min_observed_plane_points", type=int, default=64)
    args = parser.parse_args()
    gt = np.load(args.gt_npz, allow_pickle=False)
    rows = []
    for path_text in args.pred_npz:
        pred = np.load(path_text, allow_pickle=False)
        if pred["points"].shape != gt["points"].shape or not np.allclose(pred["points"], gt["points"], atol=1e-5):
            raise ValueError(f"prediction and GT are not indexed on the identical global cloud: {path_text}")
        row = evaluate_arrays(
            gt["points"], pred["point_plane_ids"], pred["plane_normals"], pred["plane_offsets"],
            gt["point_plane_ids"], gt["plane_normals"], args.match_iou,
            args.fragmentation_iou, args.min_observed_plane_points)
        row.update({"prediction": path_text,
                    "method": str(pred["method"].item()) if "method" in pred else Path(path_text).stem,
                    "runtime_seconds": float(pred["runtime_seconds"]) if "runtime_seconds" in pred else float("nan")})
        rows.append(row)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = output.with_suffix(".json")
    summary.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"csv": str(output), "json": str(summary), "methods": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
