"""Complete B0--B5/O1/O2 rows on immutable 17-scene caches."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from execute_guided_ransac_mechanism_ablation import (
    file_sha256,
    mechanism_command,
    resolve_project_path,
    run_command,
    verified_artifact,
)
from execute_research_practice_batch import FROZEN_CONFIG, exporter_command, single_fusion_row


SCHEMA_VERSION = 1
METHODS = (
    "O2_gt_identity",
    "B0_global_ransac",
    "B1_stage1_direct_svd",
    "B2_stage2_local_refit",
    "B3_stage2_manual_merge",
    "B4_guided_ransac",
)
SUMMARY_METRICS = (
    "support_partition_pairwise_f1",
    "support_partition_purity_completeness_f1",
    "support_conditioned_matched_iou",
    "support_conditioned_plane_precision",
    "support_conditioned_plane_recall_all_gt",
    "support_coverage",
    "support_conditioned_normal_angular_error_deg",
    "support_conditioned_fragmentation_excess",
    "support_conditioned_overmerge_excess",
    "runtime_seconds",
)


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def checked_reuse(record: Any) -> Path | None:
    if not isinstance(record, dict):
        return None
    path = Path(str(record.get("path", "")))
    expected = str(record.get("sha256", ""))
    if not path.is_file() or not expected or file_sha256(path) != expected:
        return None
    return path


def validate_b5_contract(prediction: Path, lines_manifest: Path) -> dict[str, Any]:
    with np.load(prediction, allow_pickle=False) as payload:
        required = {
            "point_plane_ids", "plane_ids", "plane_normals", "plane_offsets",
            "source_views", "pixel_xy", "pixel_coordinate_order", "pixel_coordinate_space",
        }
        missing = sorted(required - set(payload.files))
        exact_registry = not missing and payload["pixel_xy"].shape == (
            len(payload["point_plane_ids"]), 2
        )
        bounded_assignments = (
            "point_plane_ids" in payload.files
            and "plane_ids" in payload.files
            and len(payload["plane_ids"]) > 0
            and np.count_nonzero(payload["point_plane_ids"] >= 0) > 0
        )
    line_payload = json.loads(lines_manifest.read_text(encoding="utf-8"))
    checks = {
        "prediction_checksum_verified": bool(file_sha256(prediction)),
        "exact_registry_present": bool(exact_registry),
        "bounded_component_assignments_present": bool(bounded_assignments),
        "structural_lines_present": int(line_payload.get("line_count", 0)) > 0,
        "missing_prediction_fields": missing,
        "structural_line_count": int(line_payload.get("line_count", 0)),
    }
    if not all(checks[key] for key in (
        "prediction_checksum_verified", "exact_registry_present",
        "bounded_component_assignments_present", "structural_lines_present",
    )):
        raise ValueError(f"B5 contract failed: {checks}")
    return checks


def load_reuse(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            if isinstance(item, dict):
                result[str(item.get("id"))] = item
    return result


def structural_line_command(
    project_dir: Path, cache: Path, prediction: Path, output_dir: Path
) -> list[str]:
    config = FROZEN_CONFIG["structural_lines"]
    return [
        sys.executable,
        str(project_dir / "extract_structural_lines.py"),
        "--global_cloud_npz", str(cache),
        "--plane_prediction_npz", str(prediction),
        "--output_dir", str(output_dir),
        "--min_conf", str(FROZEN_CONFIG["evaluation_min_conf"]),
        "--min_length_px", str(config["min_length_px"]),
        "--max_lines_per_view", str(config["max_lines_per_view"]),
        "--sample_step_px", str(config["sample_step_px"]),
        "--min_valid_samples", str(config["min_valid_samples"]),
        "--plane_side_offset_px", str(config["plane_side_offset_px"]),
        "--max_3d_gap_factor", str(config["max_3d_gap_factor"]),
        "--association_filter", str(config["association_filter"]),
    ]


def support_evaluation_command(
    project_dir: Path,
    gt: Path,
    reference: Path,
    predictions: list[Path],
    method_names: list[str],
    output_json: Path,
) -> list[str]:
    config = FROZEN_CONFIG["metrics"]
    return [
        sys.executable,
        str(project_dir / "evaluate_support_record_partitions.py"),
        "--gt_npz", str(gt),
        "--support_reference_npz", str(reference),
        "--pred_npz", *[str(path) for path in predictions],
        "--method_names", *method_names,
        "--output_json", str(output_json),
        "--match_iou", str(config["match_iou"]),
        "--fragmentation_iou", str(config["fragmentation_iou"]),
        "--min_observed_plane_points", str(config["min_observed_plane_points"]),
        "--allow_legacy_cache_xy",
    ]


def run_required(command: list[str], project_dir: Path, log_path: Path) -> dict[str, Any]:
    stage = run_command(command, project_dir, log_path)
    if stage["return_code"] != 0:
        raise RuntimeError(f"command failed: {stage['command_text']}")
    return stage


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        summary: dict[str, Any] = {
            "method": method,
            "rows": len(selected),
            "scenes": len({row["scene_name"] for row in selected}),
        }
        for metric in SUMMARY_METRICS:
            values = []
            for row in selected:
                try:
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if value == value:
                    values.append(value)
            if values:
                summary[f"{metric}_mean"] = statistics.fmean(values)
                summary[f"{metric}_median"] = statistics.median(values)
        result.append(summary)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def execute(
    source_batch_json: Path,
    output_dir: Path,
    seeds: list[int],
    item_ids: list[str],
    reuse_ledgers: list[Path],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    project_dir = Path.cwd().resolve()
    source = json.loads(source_batch_json.read_text(encoding="utf-8"))
    source_items = [item for item in source["items"] if item.get("status") == "pass"]
    if item_ids:
        wanted = set(item_ids)
        source_items = [item for item in source_items if str(item.get("id")) in wanted]
        missing = wanted - {str(item.get("id")) for item in source_items}
        if missing:
            raise ValueError(f"unknown or failed item IDs: {sorted(missing)}")
    reused = load_reuse(reuse_ledgers)
    output_dir.mkdir(parents=True, exist_ok=False)
    execution: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "main_component_ablation",
        "source_batch_json": str(source_batch_json),
        "source_batch_sha256": file_sha256(source_batch_json),
        "seeds": seeds,
        "platform": platform.platform(),
        "python": sys.version,
        "items": [],
        "summary": {},
    }
    ledger = output_dir / "component_ablation_execution.json"

    def checkpoint() -> None:
        ledger.write_text(json.dumps(execution, indent=2, ensure_ascii=False), encoding="utf-8")

    aggregate_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for item_index, source_item in enumerate(source_items):
        item_id = str(source_item["id"])
        scene_name = str(source_item["scene_name"])
        # Keep intermediate paths deliberately short.  The exported Stage3 filenames
        # are descriptive and can otherwise exceed the legacy Windows MAX_PATH limit.
        # The full item ID remains in the ledger, so compact directories do not cost
        # provenance or make checkpoint recovery ambiguous.
        item_dir = output_dir / "i" / f"{item_index:03d}"
        item: dict[str, Any] = {
            "id": item_id,
            "scene_name": scene_name,
            "status": "running",
            "stages": [],
            "artifacts": {},
            "error": "",
        }
        execution["items"].append(item)
        checkpoint()
        try:
            cache, _ = verified_artifact(project_dir, source_item, "global_cloud_cache")
            gt, _ = verified_artifact(project_dir, source_item, "point_aligned_gt")
            b0, _ = verified_artifact(project_dir, source_item, "global_ransac")
            b2, _ = verified_artifact(project_dir, source_item, "direct_support_records")
            b3, _ = verified_artifact(project_dir, source_item, "manual_support_records")
            b4, _ = verified_artifact(project_dir, source_item, "learning_guided_ransac")
            prior = reused.get(item_id, {})

            b1 = checked_reuse(prior.get("artifacts", {}).get("b1_stage1_direct"))
            if b1 is None:
                stage2_dir = resolve_project_path(
                    project_dir, str(source_item["input_preflight"]["input_dir"])
                )
                stage1_dir = stage2_dir.parent / "stage1_teacher"
                if not (stage1_dir / "stage1_pred_support_teacher_manifest.json").is_file():
                    raise FileNotFoundError(f"missing recorded Stage1 sibling: {stage1_dir}")
                b1_dir = item_dir / "b1"
                command_item = {
                    "input_dir": str(stage1_dir),
                    "pattern": "*_stage1_teacher_full_pointcloud_editable_planes_data.npz",
                }
                item["stages"].append(run_required(
                    exporter_command(sys.executable, project_dir, command_item, b1_dir, "none", Path(), cache),
                    project_dir, item_dir / "x" / "b1.log",
                ))
                b1 = resolve_project_path(project_dir, str(single_fusion_row(b1_dir)["npz"]))
            item["artifacts"]["b1_stage1_direct"] = artifact(b1)
            checkpoint()

            gt_support = checked_reuse(prior.get("artifacts", {}).get("gt_support"))
            if gt_support is None:
                gt_support = item_dir / "g" / "support.npz"
                item["stages"].append(run_required(
                    [sys.executable, str(project_dir / "build_gt_support_guidance.py"), "--gt_npz", str(gt), "--output_npz", str(gt_support)],
                    project_dir, item_dir / "x" / "gt.log",
                ))
            item["artifacts"]["gt_support"] = artifact(gt_support)
            checkpoint()

            o1_predictions: list[Path] = []
            prior_o1 = prior.get("artifacts", {}).get("o1_gt_support_guided", {})
            for seed in seeds:
                prediction = checked_reuse(prior_o1.get(str(seed)) if isinstance(prior_o1, dict) else None)
                if prediction is None:
                    o1_dir = item_dir / f"o{seed}"
                    item["stages"].append(run_required(
                        mechanism_command(Path(sys.executable), project_dir, cache, gt_support, o1_dir, item_id, "proposal_consensus", seed, FROZEN_CONFIG, 1.0),
                        project_dir, item_dir / "x" / f"o{seed}.log",
                    ))
                    manifest = json.loads((o1_dir / "guided_plane_ransac_manifest.json").read_text(encoding="utf-8"))
                    prediction = resolve_project_path(project_dir, str(manifest["npz"]))
                o1_predictions.append(prediction)
            item["artifacts"]["o1_gt_support_guided"] = {
                str(seed): artifact(path) for seed, path in zip(seeds, o1_predictions, strict=True)
            }
            checkpoint()

            b5_lines = checked_reuse(prior.get("artifacts", {}).get("b5_structural_lines"))
            if b5_lines is None:
                lines_dir = item_dir / "l"
                item["stages"].append(run_required(
                    structural_line_command(project_dir, cache, b4, lines_dir),
                    project_dir, item_dir / "x" / "b5.log",
                ))
                b5_lines = lines_dir / "structural_lines_manifest.json"
            item["artifacts"]["b5_structural_lines"] = artifact(b5_lines)
            item["b5_quality_alias"] = "B4_guided_ransac"
            item["b5_checks"] = validate_b5_contract(b4, b5_lines)

            predictions = [gt, b0, b1, b2, b3, b4, *o1_predictions]
            names = [*METHODS, *[f"O1_gt_support_guided_seed{seed}" for seed in seeds]]
            metrics_json = item_dir / "e" / "metrics.json"
            item["stages"].append(run_required(
                support_evaluation_command(project_dir, gt, b3, predictions, names, metrics_json),
                project_dir, item_dir / "x" / "eval.log",
            ))
            metrics = json.loads(metrics_json.read_text(encoding="utf-8"))["methods"]
            for row in metrics:
                row.update({"item_id": item_id, "scene_name": scene_name})
                aggregate_rows.append(row)
            item["artifacts"]["metrics"] = artifact(metrics_json)
            item["status"] = "pass"
        except Exception as error:
            item["status"] = "fail"
            item["error"] = f"{type(error).__name__}: {error}"
        checkpoint()

    execution["summary"] = {
        "items": len(execution["items"]),
        "passed_items": sum(item["status"] == "pass" for item in execution["items"]),
        "failed_items": sum(item["status"] != "pass" for item in execution["items"]),
        "metric_rows": len(aggregate_rows),
        "runtime_seconds": time.perf_counter() - started,
    }
    execution["metric_summary"] = summarize(aggregate_rows)
    checkpoint()
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    (output_dir / "metric_summary.json").write_text(
        json.dumps(execution["metric_summary"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(output_dir / "metric_summary.csv", execution["metric_summary"])
    return execution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_batch_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--item_ids", nargs="*", default=[])
    parser.add_argument("--reuse_execution_json", action="append", default=[])
    args = parser.parse_args()
    result = execute(
        Path(args.source_batch_json), Path(args.output_dir), args.seeds, args.item_ids,
        [Path(path) for path in args.reuse_execution_json],
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return 0 if result["summary"]["failed_items"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
