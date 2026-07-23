"""Recompute publication-style plane partition metrics from a frozen batch.

This is a read-only post-processing step.  It reuses the point-aligned GT and
prediction NPZ artifacts already recorded in ``batch_execution.json``; no
DUSt3R inference, Stage1 inference, or RANSAC fitting is rerun.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

from evaluate_global_plane_baselines import evaluate_arrays


SCHEMA_VERSION = 1
METHOD_ARTIFACTS = (
    ("ordinary_ransac", "global_ransac"),
    ("learning_guided_ransac", "learning_guided_ransac"),
)
METRICS = (
    ("pairwise_f1", "support_partition_pairwise_f1", True),
    ("matched_iou", "support_conditioned_matched_iou", True),
    ("overmerge_excess", "support_conditioned_overmerge_excess", False),
    ("voi_nats", "segmentation_voi_nats", False),
    ("rand_index", "segmentation_rand_index", True),
    ("segmentation_covering", "segmentation_covering_symmetric", True),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_artifact(item: dict, key: str) -> Path:
    value = item.get("artifacts", {}).get(key)
    path_text = value.get("path") if isinstance(value, dict) else value
    if not path_text:
        raise ValueError(f"item {item.get('id')} has no {key!r} artifact")
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"item {item.get('id')} artifact is missing: {path}")
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        return {key: raw[key] for key in raw.files}


def evaluate_item(item: dict) -> list[dict]:
    gt_path = require_artifact(item, "point_aligned_gt")
    gt = load_npz(gt_path)
    required_gt = {"points", "point_plane_ids", "plane_normals"}
    missing = sorted(required_gt - set(gt))
    if missing:
        raise ValueError(f"GT {gt_path} is missing fields: {missing}")

    rows = []
    for method, artifact_key in METHOD_ARTIFACTS:
        prediction_path = require_artifact(item, artifact_key)
        prediction = load_npz(prediction_path)
        required_prediction = {
            "points",
            "point_plane_ids",
            "plane_normals",
            "plane_offsets",
        }
        missing = sorted(required_prediction - set(prediction))
        if missing:
            raise ValueError(
                f"prediction {prediction_path} is missing fields: {missing}"
            )
        if prediction["points"].shape != gt["points"].shape or not np.allclose(
            prediction["points"], gt["points"], atol=1e-5
        ):
            raise ValueError(
                f"prediction and GT are not the identical global cloud: "
                f"{prediction_path}"
            )
        metrics = evaluate_arrays(
            gt["points"],
            prediction["point_plane_ids"],
            prediction["plane_normals"],
            prediction["plane_offsets"],
            gt["point_plane_ids"],
            gt["plane_normals"],
            match_iou=0.5,
            fragmentation_iou=0.1,
            min_observed_plane_points=64,
        )
        row = {
            "item_id": str(item["id"]),
            "scene_name": str(item["scene_name"]),
            "method": method,
            "prediction_npz": str(prediction_path),
            "prediction_sha256": file_sha256(prediction_path),
            "gt_npz": str(gt_path),
            "gt_sha256": file_sha256(gt_path),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def finite_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_method = {method: [] for method, _ in METHOD_ARTIFACTS}
    for row in rows:
        by_method[row["method"]].append(row)
    method_summary = []
    for method, method_rows in by_method.items():
        summary = {
            "method": method,
            "scenes": len(method_rows),
        }
        for short_name, key, _ in METRICS:
            values = finite_values(method_rows, key)
            summary[f"{short_name}_mean"] = float(statistics.fmean(values))
            summary[f"{short_name}_median"] = float(statistics.median(values))
        method_summary.append(summary)

    paired = []
    ordinary = {
        row["item_id"]: row
        for row in by_method["ordinary_ransac"]
    }
    guided = {
        row["item_id"]: row
        for row in by_method["learning_guided_ransac"]
    }
    for short_name, key, higher_is_better in METRICS:
        deltas = []
        wins = 0
        for item_id in sorted(set(ordinary) & set(guided)):
            left = float(ordinary[item_id][key])
            right = float(guided[item_id][key])
            if not (math.isfinite(left) and math.isfinite(right)):
                continue
            delta = right - left
            deltas.append(delta)
            wins += int(delta > 0 if higher_is_better else delta < 0)
        paired.append(
            {
                "metric": short_name,
                "source_key": key,
                "higher_is_better": higher_is_better,
                "valid_scene_pairs": len(deltas),
                "guided_scene_wins": wins,
                "mean_delta_guided_minus_ordinary": float(
                    statistics.fmean(deltas)
                ),
                "median_delta_guided_minus_ordinary": float(
                    statistics.median(deltas)
                ),
            }
        )
    return method_summary, paired


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = set().union(*(row.keys() for row in rows))
    preferred = ["item_id", "scene_name", "method"]
    fieldnames = preferred + sorted(keys - set(preferred))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown(result: dict) -> str:
    summaries = {row["method"]: row for row in result["method_summary"]}
    ordinary = summaries["ordinary_ransac"]
    guided = summaries["learning_guided_ransac"]
    lines = [
        "# Publication-style plane partition metrics",
        "",
        (
            f"Recomputed {result['passed_items']} frozen scenes from identical "
            "point-aligned caches; no model inference or plane fitting was rerun."
        ),
        "",
        "| Metric | Ordinary RANSAC | Guided RANSAC | Delta |",
        "|---|---:|---:|---:|",
    ]
    for short_name, _, _ in METRICS:
        left = ordinary[f"{short_name}_mean"]
        right = guided[f"{short_name}_mean"]
        lines.append(
            f"| {short_name} | {left:.6f} | {right:.6f} | "
            f"{right - left:+.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "VOI uses natural logarithms and is lower-is-better. RI and SC "
                "are higher-is-better. SC is the mean of both weighted covering "
                "directions. Unassigned prediction points are retained as one "
                "segment rather than discarded."
            ),
            "",
            (
                "These point-aligned Structured3D scores make the internal "
                "methods comparable under one protocol. They are not directly "
                "comparable to published ScanNet mesh scores until all methods "
                "are evaluated on the same dataset, visibility mask, surface "
                "sampling, and pose condition."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def recompute(batch_execution_json: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    batch = json.loads(batch_execution_json.read_text(encoding="utf-8"))
    items = [item for item in batch.get("items", []) if item.get("status") == "pass"]
    if not items:
        raise ValueError("batch ledger contains no passed items")

    rows = []
    failures = []
    for item in items:
        try:
            rows.extend(evaluate_item(item))
        except Exception as error:
            failures.append(
                {
                    "item_id": str(item.get("id", "")),
                    "scene_name": str(item.get("scene_name", "")),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if failures:
        raise RuntimeError(
            "public metric recomputation failed: "
            + json.dumps(failures, ensure_ascii=False)
        )

    method_summary, paired = aggregate(rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "research_practice_public_plane_partition_metrics",
        "source_batch_execution_json": str(batch_execution_json),
        "source_batch_execution_sha256": file_sha256(batch_execution_json),
        "passed_items": len(items),
        "independent_scenes": len({item["scene_name"] for item in items}),
        "protocol": {
            "domain": "all point-aligned points with GT plane_id >= 0",
            "unassigned_prediction_policy": "retain_as_one_segment",
            "voi_log_base": "natural",
            "segmentation_covering": "mean(gt_by_pred, pred_by_gt)",
            "match_iou": 0.5,
            "fragmentation_iou": 0.1,
            "min_observed_plane_points": 64,
        },
        "method_summary": method_summary,
        "paired_diagnostics": paired,
        "rows": rows,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "public_plane_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    write_csv(output_dir / "public_plane_metrics.csv", rows)
    (output_dir / "public_plane_metrics.md").write_text(
        markdown(result), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        "Recompute publication-style metrics from a frozen batch ledger"
    )
    parser.add_argument("--batch_execution_json", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    result = recompute(args.batch_execution_json, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "passed_items": result["passed_items"],
                "independent_scenes": result["independent_scenes"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
