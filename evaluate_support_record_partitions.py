"""Evaluate plane identity on every sparse support observation record.

Unlike a unique-cache lift, this audit preserves repeated `(view, x, y)`
records.  Disagreeing labels for the same pixel therefore remain visible in
the score instead of being dropped before evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_global_plane_baselines import evaluate_support_conditioned_partition
from lift_support_prediction_to_global_cache import file_sha256, pack_registry_keys


def scalar_text(value):
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f"expected scalar text, got shape {value.shape}")
    return str(value.reshape(()).item())


def validate_coordinate_convention(
    raw, order_key, space_key, source, allow_legacy_cache_xy=False
):
    missing = [key for key in (order_key, space_key) if key not in raw]
    if missing:
        if not allow_legacy_cache_xy:
            raise ValueError(
                f"{source}: missing coordinate convention fields {missing}; "
                "use --allow_legacy_cache_xy only for outputs from the known "
                "global_plane_baselines.py legacy writer"
            )
        return True
    if order_key in raw and scalar_text(raw[order_key]) != "xy":
        raise ValueError(f"{source}: pixel coordinate order must be xy")
    if (
        space_key in raw
        and scalar_text(raw[space_key]) != "dust3r_aligned_pointmap"
    ):
        raise ValueError(
            f"{source}: pixels must be in dust3r_aligned_pointmap space"
        )
    return False


def load_keyed_prediction(path, allow_legacy_cache_xy=False):
    path = Path(path)
    with np.load(path, allow_pickle=False) as raw:
        required = {"point_plane_ids", "plane_normals"}
        missing = sorted(required - set(raw.files))
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        if {"alignment_view_indices", "pointmap_pixel_xy"} <= set(raw.files):
            legacy_coordinate_override = validate_coordinate_convention(
                raw,
                "pointmap_pixel_coordinate_order",
                "pointmap_pixel_coordinate_space",
                path,
                allow_legacy_cache_xy,
            )
            views = raw["alignment_view_indices"].astype(np.int32)
            pixel_xy = raw["pointmap_pixel_xy"].astype(np.int32)
            registry_kind = "support_records"
        elif {"view_indices", "pixel_xy"} <= set(raw.files):
            legacy_coordinate_override = validate_coordinate_convention(
                raw,
                "pixel_coordinate_order",
                "pixel_coordinate_space",
                path,
                allow_legacy_cache_xy,
            )
            views = raw["view_indices"].astype(np.int32)
            pixel_xy = raw["pixel_xy"].astype(np.int32)
            registry_kind = "unique_cache"
        elif {"source_views", "pixel_xy"} <= set(raw.files):
            legacy_coordinate_override = validate_coordinate_convention(
                raw,
                "pixel_coordinate_order",
                "pixel_coordinate_space",
                path,
                allow_legacy_cache_xy,
            )
            views = raw["source_views"].astype(np.int32)
            pixel_xy = raw["pixel_xy"].astype(np.int32)
            registry_kind = "unique_cache"
        else:
            raise ValueError(f"{path}: no explicit DUSt3R view/pixel registry")
        labels = raw["point_plane_ids"].astype(np.int32).reshape(-1)
        if len(labels) != len(np.asarray(views).reshape(-1)):
            raise ValueError(f"{path}: labels and registry have different lengths")
        method = (
            scalar_text(raw["method"])
            if "method" in raw
            else path.stem
        )
        runtime = (
            float(raw["runtime_seconds"].item())
            if "runtime_seconds" in raw
            else float("nan")
        )
        return {
            "path": path,
            "method": method,
            "runtime_seconds": runtime,
            "labels": labels,
            "normals": raw["plane_normals"].astype(np.float32),
            "views": np.asarray(views, dtype=np.int32).reshape(-1),
            "pixel_xy": np.asarray(pixel_xy, dtype=np.int32),
            "keys": pack_registry_keys(views, pixel_xy, str(path)),
            "registry_kind": registry_kind,
            "legacy_coordinate_override": legacy_coordinate_override,
        }


def map_unique_labels(source_keys, source_labels, target_keys, source):
    source_keys = np.asarray(source_keys, dtype=np.int64)
    source_labels = np.asarray(source_labels, dtype=np.int32)
    order = np.argsort(source_keys, kind="stable")
    sorted_keys = source_keys[order]
    if len(sorted_keys) > 1 and np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError(f"{source}: expected a unique cache registry")
    positions = np.searchsorted(sorted_keys, target_keys)
    matched = positions < len(sorted_keys)
    if matched.any():
        matched_positions = positions[matched]
        matched[matched] = sorted_keys[matched_positions] == target_keys[matched]
    result = np.full(len(target_keys), -1, dtype=np.int32)
    result[matched] = source_labels[order[positions[matched]]]
    return result, matched


def repeated_key_diagnostics(keys, labels):
    positive = np.asarray(labels) >= 0
    keys = np.asarray(keys, dtype=np.int64)[positive]
    labels = np.asarray(labels, dtype=np.int32)[positive]
    order = np.argsort(keys, kind="stable")
    keys, labels = keys[order], labels[order]
    if len(keys):
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        ends = np.r_[starts[1:], len(keys)]
    else:
        starts = ends = np.zeros((0,), dtype=np.int64)
    conflicts = 0
    conflicting_records = 0
    for start, end in zip(starts, ends):
        if len(np.unique(labels[start:end])) > 1:
            conflicts += 1
            conflicting_records += int(end - start)
    return {
        "positive_record_count": int(len(keys)),
        "unique_positive_key_count": int(len(starts)),
        "duplicate_positive_record_count": int(len(keys) - len(starts)),
        "conflicting_positive_key_count": int(conflicts),
        "conflicting_positive_record_count": int(conflicting_records),
    }


def per_plane_rows(labels, gt_labels, views, keys, normals, gt_normals):
    gt_ids = sorted(int(value) for value in np.unique(gt_labels) if value >= 0)
    rows = []
    for plane_id in sorted(int(value) for value in np.unique(labels) if value >= 0):
        mask = labels == plane_id
        labeled = mask & (gt_labels >= 0)
        counts = {
            str(gt_id): int((labeled & (gt_labels == gt_id)).sum())
            for gt_id in gt_ids
        }
        labeled_count = int(labeled.sum())
        dominant_gt_id = (
            max(gt_ids, key=lambda value: counts[str(value)])
            if gt_ids and labeled_count
            else -1
        )
        dominant_count = counts.get(str(dominant_gt_id), 0)
        angular = float("nan")
        if dominant_gt_id >= 0 and dominant_count and plane_id < len(normals):
            angular = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            abs(float(normals[plane_id] @ gt_normals[dominant_gt_id])),
                            0,
                            1,
                        )
                    )
                )
            )
        rows.append(
            {
                "pred_plane_id": plane_id,
                "record_count": int(mask.sum()),
                "gt_labeled_record_count": labeled_count,
                "unique_support_keys": int(len(np.unique(keys[mask]))),
                "view_count": int(len(np.unique(views[mask]))),
                "dominant_gt_plane_id": int(dominant_gt_id),
                "dominant_gt_record_count": int(dominant_count),
                "dominant_gt_purity": (
                    dominant_count / labeled_count if labeled_count else 0.0
                ),
                "dominant_gt_normal_error_deg": angular,
                "gt_record_counts": counts,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        "Evaluate plane partitions on repeated sparse support records"
    )
    parser.add_argument("--gt_npz", required=True)
    parser.add_argument("--support_reference_npz", required=True)
    parser.add_argument("--pred_npz", nargs="+", required=True)
    parser.add_argument(
        "--method_names",
        nargs="*",
        default=[],
        help="Optional explicit method name for each --pred_npz entry",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--match_iou", type=float, default=0.5)
    parser.add_argument("--fragmentation_iou", type=float, default=0.1)
    parser.add_argument("--min_observed_plane_points", type=int, default=64)
    parser.add_argument(
        "--allow_legacy_cache_xy",
        action="store_true",
        help=(
            "Explicitly interpret legacy global_plane_baselines.py "
            "source_views/pixel_xy fields as DUSt3R view indices and xy pixels"
        ),
    )
    args = parser.parse_args()
    if args.method_names and len(args.method_names) != len(args.pred_npz):
        parser.error("--method_names must match the number of --pred_npz entries")

    reference = load_keyed_prediction(
        args.support_reference_npz, args.allow_legacy_cache_xy
    )
    if reference["registry_kind"] != "support_records":
        raise ValueError("support reference must contain Stage3 support records")
    gt = load_keyed_prediction(args.gt_npz, args.allow_legacy_cache_xy)
    if gt["registry_kind"] != "unique_cache":
        raise ValueError("GT must use a unique global-cache registry")
    gt_record_labels, gt_matched = map_unique_labels(
        gt["keys"], gt["labels"], reference["keys"], gt["path"]
    )
    gt_labeled_reference = int((gt_record_labels >= 0).sum())

    rows = []
    for prediction_index, path in enumerate(args.pred_npz):
        prediction = load_keyed_prediction(path, args.allow_legacy_cache_xy)
        if args.method_names:
            prediction["method"] = args.method_names[prediction_index]
        if prediction["registry_kind"] == "support_records":
            if not np.array_equal(prediction["keys"], reference["keys"]):
                raise ValueError(
                    f"{prediction['path']}: support record registry/order differs "
                    "from the reference"
                )
            record_labels = prediction["labels"]
            registry_matches = np.ones(len(record_labels), dtype=bool)
        else:
            record_labels, registry_matches = map_unique_labels(
                prediction["keys"],
                prediction["labels"],
                reference["keys"],
                prediction["path"],
            )
        metrics = evaluate_support_conditioned_partition(
            record_labels,
            prediction["normals"],
            gt_record_labels,
            gt["normals"],
            args.match_iou,
            args.fragmentation_iou,
            args.min_observed_plane_points,
        )
        labeled_assigned = (record_labels >= 0) & (gt_record_labels >= 0)
        metrics.update(
            {
                "method": prediction["method"],
                "prediction": str(prediction["path"]),
                "prediction_sha256": file_sha256(prediction["path"]),
                "runtime_seconds": prediction["runtime_seconds"],
                "legacy_coordinate_override": prediction[
                    "legacy_coordinate_override"
                ],
                "reference_record_count": int(len(reference["keys"])),
                "registry_matched_record_count": int(registry_matches.sum()),
                "gt_registry_matched_record_count": int(gt_matched.sum()),
                "gt_labeled_reference_record_count": gt_labeled_reference,
                "record_assignment_rate": float((record_labels >= 0).mean()),
                "gt_labeled_record_coverage": (
                    int(labeled_assigned.sum()) / gt_labeled_reference
                    if gt_labeled_reference
                    else 0.0
                ),
                **repeated_key_diagnostics(reference["keys"], record_labels),
                "per_plane": per_plane_rows(
                    record_labels,
                    gt_record_labels,
                    reference["views"],
                    reference["keys"],
                    prediction["normals"],
                    gt["normals"],
                ),
            }
        )
        rows.append(metrics)

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gt_npz": str(gt["path"]),
        "gt_sha256": file_sha256(gt["path"]),
        "support_reference_npz": str(reference["path"]),
        "support_reference_sha256": file_sha256(reference["path"]),
        "coordinate_order": "xy",
        "coordinate_space": "dust3r_aligned_pointmap",
        "record_join_key": "(alignment_view_index,x,y)",
        "duplicate_records_preserved": True,
        "match_iou": args.match_iou,
        "fragmentation_iou": args.fragmentation_iou,
        "min_observed_plane_points": args.min_observed_plane_points,
        "allow_legacy_cache_xy": args.allow_legacy_cache_xy,
        "methods": rows,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    scalar_keys = [key for key in rows[0] if key != "per_plane"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows(
            [{key: row[key] for key in scalar_keys} for row in rows]
        )
    print(
        json.dumps(
            {"json": str(output), "csv": str(csv_path), "methods": len(rows)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
