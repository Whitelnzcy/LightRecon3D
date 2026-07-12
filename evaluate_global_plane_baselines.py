"""Evaluate plane primitives against point-aligned Structured3D GT labels."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    def linear_sum_assignment(cost):
        # Deterministic greedy fallback; SciPy uses exact Hungarian matching when available.
        candidates = sorted((float(cost[i, j]), i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1]))
        rows, cols, used_r, used_c = [], [], set(), set()
        for _, i, j in candidates:
            if i not in used_r and j not in used_c:
                rows.append(i); cols.append(j); used_r.add(i); used_c.add(j)
        return np.asarray(rows), np.asarray(cols)


def iou_matrix(pred, gt, pred_ids, gt_ids):
    result = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float64)
    for i, pid in enumerate(pred_ids):
        pm = pred == pid
        for j, gid in enumerate(gt_ids):
            gm = gt == gid
            union = int((pm | gm).sum())
            result[i, j] = int((pm & gm).sum()) / union if union else 0.0
    return result


def evaluate_arrays(points, pred_labels, pred_normals, pred_offsets, gt_labels,
                    gt_normals, match_iou=0.5, fragmentation_iou=0.1):
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
    return {
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


def main():
    parser = argparse.ArgumentParser("Evaluate globally aligned plane primitive NPZ files")
    parser.add_argument("--gt_npz", required=True,
                        help="Point-aligned NPZ with points, point_plane_ids and plane_normals")
    parser.add_argument("--pred_npz", nargs="+", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--match_iou", type=float, default=.5)
    parser.add_argument("--fragmentation_iou", type=float, default=.1)
    args = parser.parse_args()
    gt = np.load(args.gt_npz, allow_pickle=False)
    rows = []
    for path_text in args.pred_npz:
        pred = np.load(path_text, allow_pickle=False)
        if pred["points"].shape != gt["points"].shape or not np.allclose(pred["points"], gt["points"], atol=1e-5):
            raise ValueError(f"prediction and GT are not indexed on the identical global cloud: {path_text}")
        row = evaluate_arrays(
            gt["points"], pred["point_plane_ids"], pred["plane_normals"], pred["plane_offsets"],
            gt["point_plane_ids"], gt["plane_normals"], args.match_iou, args.fragmentation_iou)
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
