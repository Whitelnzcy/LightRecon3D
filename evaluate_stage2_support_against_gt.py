import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_summary(json_path):
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def evaluate_assignment(path):
    with np.load(path) as raw:
        if "assignment" in raw:
            assignment = raw["assignment"].astype(np.int32)
        else:
            assignment = raw["point_plane_ids"].astype(np.int32)
        gt = raw.get("gt_point_plane_ids")
        if gt is None:
            raise KeyError(f"{path} does not contain gt_point_plane_ids")
        gt = gt.astype(np.int32)
    valid_pred = assignment >= 0
    valid_gt = gt >= 0
    plane_rows = []
    for plane_id in sorted(int(value) for value in np.unique(assignment[valid_pred])):
        pred_mask = assignment == plane_id
        gt_values = gt[pred_mask & valid_gt]
        if len(gt_values) == 0:
            source_gt = -1
            iou = 0.0
        else:
            counts = np.bincount(gt_values)
            source_gt = int(np.argmax(counts))
            gt_mask = gt == source_gt
            union = np.count_nonzero(pred_mask | gt_mask)
            iou = float(np.count_nonzero(pred_mask & gt_mask) / union) if union else 0.0
        plane_rows.append(
            {
                "plane_id": plane_id,
                "source_gt_plane_id": source_gt,
                "assigned_points": int(pred_mask.sum()),
                "majority_gt_iou": iou,
            }
        )

    if valid_pred.any():
        mapped = np.full_like(assignment, -9999, dtype=np.int32)
        for row in plane_rows:
            mapped[assignment == row["plane_id"]] = row["source_gt_plane_id"]
        leakage = float(np.mean(mapped[valid_pred] != gt[valid_pred]))
    else:
        leakage = 0.0
    background_error = float(np.mean((assignment < 0) & valid_gt)) if valid_gt.any() else 0.0
    return plane_rows, leakage, background_error


def main():
    parser = argparse.ArgumentParser("Evaluate Stage2 support assignment against GT point labels")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*.npz")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    plane_ious = []
    leakages = []
    background_errors = []
    residuals = []
    for npz_path in sorted(input_dir.glob(args.pattern)):
        plane_rows, leakage, background_error = evaluate_assignment(npz_path)
        stem = npz_path.name.replace("_assignment.npz", "")
        summary = load_summary(input_dir / f"{stem}.json")
        json_planes = summary.get("planes", [])
        for plane in json_planes:
            if plane.get("active", True) and plane.get("mean_abs_distance_normalized") is not None:
                residuals.append(float(plane["mean_abs_distance_normalized"]))
        sample_iou = float(np.mean([row["majority_gt_iou"] for row in plane_rows])) if plane_rows else 0.0
        plane_ious.extend(row["majority_gt_iou"] for row in plane_rows)
        leakages.append(leakage)
        background_errors.append(background_error)
        rows.append(
            {
                "sample": stem,
                "planes": len(plane_rows),
                "mean_majority_gt_iou": sample_iou,
                "leakage": leakage,
                "background_error": background_error,
                "active_planes": summary.get("active_planes", len(plane_rows)),
                "background_ratio": summary.get("background_ratio", float(np.mean(np.asarray([]))) if False else None),
            }
        )

    fields = [
        "sample",
        "planes",
        "active_planes",
        "mean_majority_gt_iou",
        "leakage",
        "background_error",
        "background_ratio",
    ]
    with (output_dir / "stage2_support_vs_gt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_dir": str(input_dir),
        "samples": len(rows),
        "mean_support_iou": float(np.mean(plane_ious)) if plane_ious else 0.0,
        "median_support_iou": float(np.median(plane_ious)) if plane_ious else 0.0,
        "mean_leakage": float(np.mean(leakages)) if leakages else 0.0,
        "median_leakage": float(np.median(leakages)) if leakages else 0.0,
        "mean_background_error": float(np.mean(background_errors)) if background_errors else 0.0,
        "mean_plane_residual": float(np.mean(residuals)) if residuals else 0.0,
        "median_plane_residual": float(np.median(residuals)) if residuals else 0.0,
        "p90_plane_residual": float(np.quantile(residuals, 0.9)) if residuals else 0.0,
        "worst_samples": sorted(
            rows,
            key=lambda row: row["mean_majority_gt_iou"],
        )[:10],
    }
    (output_dir / "stage2_support_vs_gt_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
