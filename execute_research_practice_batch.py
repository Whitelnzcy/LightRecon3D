"""Run the retained research-practice baselines on identical DUSt3R caches.

The executor is intentionally deterministic and manifest driven.  It streams
each subprocess to a stage log, preserves partial/failure rows, and writes
machine-readable aggregate tables after every item.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from research_practice_batch import (
    DEFAULT_PATTERN,
    file_record,
    file_sha256,
    load_manifest,
    preflight_item,
)


SCHEMA_VERSION = 1
Runner = Callable[[str, list[str], Path, Path], dict[str, Any]]

FROZEN_CONFIG = {
    "global_alignment": {
        "image_size": 512,
        "scene_graph": "complete",
        "batch_size": 1,
        "niter": 300,
        "lr": 0.01,
        "schedule": "cosine",
        "global_min_conf": 0.0,
    },
    "evaluation_min_conf": 1.0,
    "min_plane_points": 64,
    "ransac": {
        "distance_threshold": 0.03,
        "iterations": 300,
        "min_inliers": 2000,
        "cluster_radius": 0.08,
        "min_component_points": 1000,
        "max_planes": 32,
        "seed": 0,
        "hypothesis_max_points": 50000,
        "component_exact_max_points": 20000,
    },
    "guided_ransac": {
        "proposal_iterations": 64,
        "proposal_min_points": 64,
        "proposal_min_inlier_ratio": 0.60,
        "proposal_max_points": 4000,
        "support_score_weight": 1.0,
        "fallback_iterations": 96,
    },
    "support_join": {
        "conflict_policy": "drop",
        "min_points_per_plane": 3,
    },
    "metrics": {
        "match_iou": 0.5,
        "fragmentation_iou": 0.1,
        "min_observed_plane_points": 64,
    },
    "structural_lines": {
        "min_length_px": 24,
        "max_lines_per_view": 256,
        "sample_step_px": 2,
        "min_valid_samples": 6,
        "plane_side_offset_px": 2,
        "max_3d_gap_factor": 8,
        "association_filter": "all",
    },
}


class StageFailure(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def command_text(command: list[str]) -> str:
    return shlex.join(str(value) for value in command)


def run_logged_stage(
    stage: str, command: list[str], project_dir: Path, log_path: Path
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"\n[{stage}] {command_text(command)}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"stage={stage}\ncommand={command_text(command)}\n")
        log.flush()
        process = subprocess.Popen(
            [str(value) for value in command],
            cwd=str(project_dir),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = int(process.wait())
    return {
        "stage": stage,
        "command": [str(value) for value in command],
        "command_text": command_text(command),
        "log": str(log_path),
        "return_code": return_code,
        "runtime_seconds": float(time.perf_counter() - started),
        "status": "pass" if return_code == 0 else "fail",
    }


def run_required(
    row: dict[str, Any],
    stage: str,
    command: list[str],
    project_dir: Path,
    item_dir: Path,
    runner: Runner,
) -> dict[str, Any]:
    result = runner(stage, command, project_dir, item_dir / "logs" / f"{stage}.log")
    row["stages"].append(result)
    if int(result["return_code"]) != 0:
        raise StageFailure(stage, f"stage {stage} exited with {result['return_code']}")
    return result


def exporter_command(
    python_bin: str,
    project_dir: Path,
    item: dict[str, Any],
    output_dir: Path,
    merge_mode: str,
    weights_path: Path,
    global_cloud_cache: Path | None,
) -> list[str]:
    alignment = FROZEN_CONFIG["global_alignment"]
    command = [
        python_bin,
        str(project_dir / "export_stage3_scene_plane_fusion.py"),
        "--input_dir",
        str(item["input_dir"]),
        "--output_dir",
        str(output_dir),
        "--pattern",
        str(item.get("pattern", DEFAULT_PATTERN)),
        "--group_by",
        "pair_group",
        "--fusion_mode",
        "dust3r_global",
        "--merge_mode",
        merge_mode,
        "--image_size",
        str(alignment["image_size"]),
        "--scene_graph",
        str(alignment["scene_graph"]),
        "--batch_size",
        str(alignment["batch_size"]),
        "--niter",
        str(alignment["niter"]),
        "--lr",
        str(alignment["lr"]),
        "--schedule",
        str(alignment["schedule"]),
        "--global_min_conf",
        str(alignment["global_min_conf"]),
        "--include_second_view",
        "--min_points",
        str(FROZEN_CONFIG["min_plane_points"]),
        "--max_display_points",
        "32000",
        "--mesh_grid_resolution",
        "48",
        "--dust3r_mesh_min_conf",
        str(FROZEN_CONFIG["evaluation_min_conf"]),
    ]
    if global_cloud_cache is None:
        command.extend(["--weights_path", str(weights_path)])
    else:
        command.extend(["--global_cloud_cache", str(global_cloud_cache)])
    return command


def single_fusion_row(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "stage3_scene_fusion_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing fusion manifest: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"expected one fusion row in {path}, found {len(rows)}")
    return rows[0]


def require_file(path: Path, stage: str) -> Path:
    if not path.is_file():
        raise StageFailure(stage, f"missing expected output: {path}")
    return path


def verify_reused_cache(item: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    text = str(item.get("global_cloud_cache", "")).strip()
    if not text:
        return None, None
    path = Path(text)
    expected = str(item.get("global_cloud_sha256", "")).strip().lower()
    record = file_record(path, expected)
    if not record["exists"]:
        raise StageFailure("cache_precheck", f"reused cache is missing: {path}")
    if expected and record["checksum_matches"] is not True:
        raise StageFailure("cache_precheck", f"reused cache checksum mismatch: {path}")
    return path, record


def artifact(path: Path) -> dict[str, Any]:
    record = file_record(path)
    if not record["exists"]:
        raise FileNotFoundError(path)
    return record


def hardware_record() -> dict[str, Any]:
    row: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "nvidia_smi": "",
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        row["nvidia_smi"] = completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        row["nvidia_smi_error"] = str(error)
    return row


def execute_item(
    item: dict[str, Any],
    *,
    project_dir: Path,
    python_bin: str,
    weights_path: Path,
    output_dir: Path,
    runner: Runner = run_logged_stage,
) -> dict[str, Any]:
    item_id = str(item["id"])
    item_dir = output_dir / "items" / item_id
    item_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    row: dict[str, Any] = {
        "id": item_id,
        "scene_name": str(item.get("expected_scene_name", "")),
        "pair_group": str(item.get("expected_pair_group", "")),
        "status": "running",
        "failure_stage": "",
        "error": "",
        "stages": [],
        "artifacts": {},
    }
    try:
        validation = preflight_item(
            item,
            default_pattern=str(item.get("pattern", DEFAULT_PATTERN)),
            default_min_views=int(item.get("min_views", 5)),
            check_image_files=True,
        )
        row["input_preflight"] = validation
        if validation["status"] != "pass":
            raise StageFailure("input_preflight", " | ".join(validation["errors"]))
        row["scene_name"] = validation["scene_name"]
        row["pair_group"] = validation["pair_group"]

        reused_cache, reused_record = verify_reused_cache(item)
        row["reused_global_cloud_cache"] = reused_record

        direct_dir = item_dir / "direct_support"
        run_required(
            row,
            "direct_support",
            exporter_command(
                python_bin,
                project_dir,
                item,
                direct_dir,
                "none",
                weights_path,
                reused_cache,
            ),
            project_dir,
            item_dir,
            runner,
        )
        direct_row = single_fusion_row(direct_dir)
        direct_npz = require_file(Path(direct_row["npz"]), "direct_support")
        cache_path = require_file(
            Path(direct_row["global_cloud_cache"]), "direct_support"
        )
        if reused_cache is not None and cache_path.resolve() != reused_cache.resolve():
            raise StageFailure(
                "direct_support",
                f"fusion reported cache {cache_path}, expected {reused_cache}",
            )

        gt_dir = item_dir / "gt"
        gt_dir.mkdir()
        gt_npz = gt_dir / f"{item_id}_point_aligned_plane_gt.npz"
        run_required(
            row,
            "point_aligned_gt",
            [
                python_bin,
                str(project_dir / "build_structured3d_point_aligned_gt.py"),
                "--global_cloud_npz",
                str(cache_path),
                "--output_npz",
                str(gt_npz),
                "--output_ply",
                str(gt_dir / f"{item_id}_point_aligned_plane_gt.ply"),
                "--output_manifest",
                str(gt_dir / f"{item_id}_point_aligned_plane_gt_manifest.json"),
                "--min_conf",
                str(FROZEN_CONFIG["evaluation_min_conf"]),
                "--min_plane_points",
                str(FROZEN_CONFIG["min_plane_points"]),
                "--image_size",
                "512",
                "--patch_size",
                "16",
                "--boundary_ignore_radius",
                "1",
            ],
            project_dir,
            item_dir,
            runner,
        )
        require_file(gt_npz, "point_aligned_gt")

        manual_dir = item_dir / "manual_support"
        run_required(
            row,
            "manual_support",
            exporter_command(
                python_bin,
                project_dir,
                item,
                manual_dir,
                "manual",
                weights_path,
                cache_path,
            ),
            project_dir,
            item_dir,
            runner,
        )
        manual_row = single_fusion_row(manual_dir)
        manual_npz = require_file(Path(manual_row["npz"]), "manual_support")

        ransac_dir = item_dir / "ransac"
        ransac_dir.mkdir()
        ransac = FROZEN_CONFIG["ransac"]
        run_required(
            row,
            "global_ransac",
            [
                python_bin,
                str(project_dir / "global_plane_baselines.py"),
                "--global_cloud_npz",
                str(cache_path),
                "--output_dir",
                str(ransac_dir),
                "--scene_key",
                item_id,
                "--min_conf",
                str(FROZEN_CONFIG["evaluation_min_conf"]),
                "--distance_threshold",
                str(ransac["distance_threshold"]),
                "--iterations",
                str(ransac["iterations"]),
                "--min_inliers",
                str(ransac["min_inliers"]),
                "--cluster_radius",
                str(ransac["cluster_radius"]),
                "--min_component_points",
                str(ransac["min_component_points"]),
                "--max_planes",
                str(ransac["max_planes"]),
                "--seed",
                str(ransac["seed"]),
                "--hypothesis_max_points",
                str(ransac["hypothesis_max_points"]),
                "--component_exact_max_points",
                str(ransac["component_exact_max_points"]),
            ],
            project_dir,
            item_dir,
            runner,
        )
        ransac_npz = require_file(
            ransac_dir
            / f"{item_id}_global_ransac_cc_full_pointcloud_editable_planes_data.npz",
            "global_ransac",
        )

        guided_dir = item_dir / "guided_ransac"
        guided = FROZEN_CONFIG["guided_ransac"]
        run_required(
            row,
            "learning_guided_ransac",
            [
                python_bin,
                str(project_dir / "guided_plane_ransac.py"),
                "--global_cloud_npz",
                str(cache_path),
                "--support_npz",
                str(direct_npz),
                "--output_dir",
                str(guided_dir),
                "--scene_key",
                item_id,
                "--min_conf",
                str(FROZEN_CONFIG["evaluation_min_conf"]),
                "--distance_threshold",
                str(ransac["distance_threshold"]),
                "--proposal_iterations",
                str(guided["proposal_iterations"]),
                "--proposal_min_points",
                str(guided["proposal_min_points"]),
                "--proposal_min_inlier_ratio",
                str(guided["proposal_min_inlier_ratio"]),
                "--proposal_max_points",
                str(guided["proposal_max_points"]),
                "--support_score_weight",
                str(guided["support_score_weight"]),
                "--fallback_iterations",
                str(guided["fallback_iterations"]),
                "--min_inliers",
                str(ransac["min_inliers"]),
                "--cluster_radius",
                str(ransac["cluster_radius"]),
                "--min_component_points",
                str(ransac["min_component_points"]),
                "--max_planes",
                str(ransac["max_planes"]),
                "--seed",
                str(ransac["seed"]),
                "--hypothesis_max_points",
                str(ransac["hypothesis_max_points"]),
                "--component_exact_max_points",
                str(ransac["component_exact_max_points"]),
            ],
            project_dir,
            item_dir,
            runner,
        )
        guided_npz = require_file(
            guided_dir
            / f"{item_id}_learning_guided_ransac_cc_full_pointcloud_editable_planes_data.npz",
            "learning_guided_ransac",
        )

        lifted_paths: dict[str, Path] = {}
        for label, source_npz, method in (
            (
                "direct_unique",
                direct_npz,
                "stage2_direct_svd_unique_conflict_drop",
            ),
            (
                "manual_unique",
                manual_npz,
                "stage2_manual_merge_unique_conflict_drop",
            ),
        ):
            lift_dir = item_dir / label
            lift_dir.mkdir()
            run_required(
                row,
                label,
                [
                    python_bin,
                    str(project_dir / "lift_support_prediction_to_global_cache.py"),
                    "--global_cloud_npz",
                    str(cache_path),
                    "--support_npz",
                    str(source_npz),
                    "--output_dir",
                    str(lift_dir),
                    "--scene_key",
                    item_id,
                    "--method",
                    method,
                    "--min_conf",
                    str(FROZEN_CONFIG["evaluation_min_conf"]),
                    "--conflict_policy",
                    str(FROZEN_CONFIG["support_join"]["conflict_policy"]),
                    "--min_points_per_plane",
                    str(FROZEN_CONFIG["support_join"]["min_points_per_plane"]),
                ],
                project_dir,
                item_dir,
                runner,
            )
            lifted_paths[label] = require_file(
                lift_dir / f"{item_id}_{method}_full_pointcloud_editable_planes_data.npz",
                label,
            )

        line_dir = item_dir / "structural_lines"
        line = FROZEN_CONFIG["structural_lines"]
        run_required(
            row,
            "structural_lines",
            [
                python_bin,
                str(project_dir / "extract_structural_lines.py"),
                "--global_cloud_npz",
                str(cache_path),
                "--plane_prediction_npz",
                str(manual_npz),
                "--output_dir",
                str(line_dir),
                "--min_conf",
                str(FROZEN_CONFIG["evaluation_min_conf"]),
                "--min_length_px",
                str(line["min_length_px"]),
                "--max_lines_per_view",
                str(line["max_lines_per_view"]),
                "--sample_step_px",
                str(line["sample_step_px"]),
                "--min_valid_samples",
                str(line["min_valid_samples"]),
                "--plane_side_offset_px",
                str(line["plane_side_offset_px"]),
                "--max_3d_gap_factor",
                str(line["max_3d_gap_factor"]),
                "--association_filter",
                str(line["association_filter"]),
            ],
            project_dir,
            item_dir,
            runner,
        )
        line_manifest = require_file(
            line_dir / "structural_lines_manifest.json", "structural_lines"
        )

        evaluation_dir = item_dir / "evaluation"
        evaluation_dir.mkdir()
        metric = FROZEN_CONFIG["metrics"]
        full_csv = evaluation_dir / "full_cache_metrics.csv"
        run_required(
            row,
            "full_cache_metrics",
            [
                python_bin,
                str(project_dir / "evaluate_global_plane_baselines.py"),
                "--gt_npz",
                str(gt_npz),
                "--pred_npz",
                str(gt_npz),
                str(ransac_npz),
                str(guided_npz),
                str(lifted_paths["direct_unique"]),
                str(lifted_paths["manual_unique"]),
                "--output_csv",
                str(full_csv),
                "--match_iou",
                str(metric["match_iou"]),
                "--fragmentation_iou",
                str(metric["fragmentation_iou"]),
                "--min_observed_plane_points",
                str(metric["min_observed_plane_points"]),
            ],
            project_dir,
            item_dir,
            runner,
        )
        full_json = require_file(full_csv.with_suffix(".json"), "full_cache_metrics")

        support_json = evaluation_dir / "support_record_metrics.json"
        run_required(
            row,
            "support_record_metrics",
            [
                python_bin,
                str(project_dir / "evaluate_support_record_partitions.py"),
                "--gt_npz",
                str(gt_npz),
                "--support_reference_npz",
                str(manual_npz),
                "--pred_npz",
                str(gt_npz),
                str(ransac_npz),
                str(guided_npz),
                str(direct_npz),
                str(manual_npz),
                str(lifted_paths["direct_unique"]),
                str(lifted_paths["manual_unique"]),
                "--method_names",
                "point_aligned_gt_oracle",
                "global_ransac_cc",
                "learning_guided_ransac_cc",
                "stage2_support_direct_global_svd",
                "stage2_manual_merge_support",
                "stage2_direct_svd_unique_conflict_drop",
                "stage2_manual_merge_unique_conflict_drop",
                "--output_json",
                str(support_json),
                "--match_iou",
                str(metric["match_iou"]),
                "--fragmentation_iou",
                str(metric["fragmentation_iou"]),
                "--min_observed_plane_points",
                str(metric["min_observed_plane_points"]),
                "--allow_legacy_cache_xy",
            ],
            project_dir,
            item_dir,
            runner,
        )
        require_file(support_json, "support_record_metrics")

        row["artifacts"] = {
            "global_cloud_cache": artifact(cache_path),
            "point_aligned_gt": artifact(gt_npz),
            "direct_support_records": artifact(direct_npz),
            "manual_support_records": artifact(manual_npz),
            "global_ransac": artifact(ransac_npz),
            "learning_guided_ransac": artifact(guided_npz),
            "direct_unique_conflict_drop": artifact(lifted_paths["direct_unique"]),
            "manual_unique_conflict_drop": artifact(lifted_paths["manual_unique"]),
            "structural_lines_manifest": artifact(line_manifest),
            "full_cache_metrics": artifact(full_json),
            "support_record_metrics": artifact(support_json),
        }
        line_payload = json.loads(line_manifest.read_text(encoding="utf-8"))
        row["line_summary"] = {
            "line_count": int(line_payload["line_count"]),
            "runtime_seconds": float(line_payload["runtime_seconds"]),
            "association_counts": line_payload["association_counts"],
        }
        row["status"] = "pass"
    except StageFailure as error:
        row["status"] = "fail"
        row["failure_stage"] = error.stage
        row["error"] = str(error)
    except Exception as error:  # preserve unexpected failures in the batch ledger
        row["status"] = "fail"
        row["failure_stage"] = row["stages"][-1]["stage"] if row["stages"] else "setup"
        row["error"] = f"{type(error).__name__}: {error}"
    row["runtime_seconds"] = float(time.perf_counter() - started)
    (item_dir / "item_execution.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    return row


def metric_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if item["status"] != "pass":
            continue
        item_id = item["id"]
        full_path = Path(item["artifacts"]["full_cache_metrics"]["path"])
        for metric in json.loads(full_path.read_text(encoding="utf-8")):
            rows.append(
                {
                    "item_id": item_id,
                    "scene_name": item["scene_name"],
                    "metric_family": "full_cache",
                    **metric,
                }
            )
        support_path = Path(item["artifacts"]["support_record_metrics"]["path"])
        support = json.loads(support_path.read_text(encoding="utf-8"))
        for metric in support["methods"]:
            rows.append(
                {
                    "item_id": item_id,
                    "scene_name": item["scene_name"],
                    "metric_family": "support_records",
                    **{key: value for key, value in metric.items() if key != "per_plane"},
                }
            )
    return rows


def aggregate_method_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("metric_family", "")), str(row.get("method", "")))
        grouped.setdefault(key, []).append(row)
    aggregate: list[dict[str, Any]] = []
    excluded = {"item_id", "scene_name", "metric_family", "method", "prediction"}
    for (family, method), method_rows in sorted(grouped.items()):
        output: dict[str, Any] = {
            "metric_family": family,
            "method": method,
            "view_groups": len({str(row["item_id"]) for row in method_rows}),
            "unique_scenes": len({str(row["scene_name"]) for row in method_rows}),
        }
        keys = set().union(*(row.keys() for row in method_rows)) - excluded
        for key in sorted(keys):
            values = []
            for row in method_rows:
                value = row.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                numeric = float(value)
                if math.isfinite(numeric):
                    values.append(numeric)
            if values:
                output[f"{key}_mean"] = float(statistics.fmean(values))
                output[f"{key}_median"] = float(statistics.median(values))
        aggregate.append(output)
    return aggregate


def write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = set().union(*(row.keys() for row in rows))
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def batch_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [item for item in items if item["status"] == "pass"]
    return {
        "items": len(items),
        "passed_items": len(passed),
        "failed_items": len(items) - len(passed),
        "unique_view_groups": len({item["pair_group"] for item in passed}),
        "unique_scene_names": len({item["scene_name"] for item in passed}),
        "runtime_seconds": float(sum(item["runtime_seconds"] for item in items)),
        "line_count": int(sum(item.get("line_summary", {}).get("line_count", 0) for item in passed)),
    }


def markdown_summary(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        f"# {result['batch_name']} execution",
        "",
        f"Git SHA: `{result['git_sha']}`",
        "",
        (
            f"Passed {summary['passed_items']}/{summary['items']} view groups "
            f"from {summary['unique_scene_names']} independent scene IDs."
        ),
        "",
        "| Item | Scene | Status | Failure stage | Runtime (s) | Lines |",
        "|---|---|---:|---|---:|---:|",
    ]
    for item in result["items"]:
        lines.append(
            f"| {item['id']} | {item['scene_name']} | {item['status']} | "
            f"{item['failure_stage']} | {item['runtime_seconds']:.3f} | "
            f"{item.get('line_summary', {}).get('line_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "Raw manual/direct support-record metrics preserve repeated and conflicting observations. The unique-cache conflict-drop variants are reported only as ablations.",
            "",
        ]
    )
    return "\n".join(lines)


def write_batch_outputs(result: dict[str, Any], output_dir: Path) -> None:
    result["summary"] = batch_summary(result["items"])
    metrics = metric_rows(result["items"])
    method_summary = aggregate_method_rows(metrics)
    (output_dir / "batch_execution.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    item_rows = [
        {
            "id": item["id"],
            "scene_name": item["scene_name"],
            "pair_group": item["pair_group"],
            "status": item["status"],
            "failure_stage": item["failure_stage"],
            "error": item["error"],
            "runtime_seconds": item["runtime_seconds"],
            "line_count": item.get("line_summary", {}).get("line_count", 0),
        }
        for item in result["items"]
    ]
    write_csv(
        output_dir / "batch_items.csv",
        item_rows,
        ["id", "scene_name", "pair_group", "status", "failure_stage", "error"],
    )
    write_csv(
        output_dir / "aggregate_metrics.csv",
        metrics,
        ["item_id", "scene_name", "metric_family", "method", "prediction"],
    )
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "aggregate_method_summary.csv",
        method_summary,
        ["metric_family", "method", "view_groups", "unique_scenes"],
    )
    (output_dir / "aggregate_method_summary.json").write_text(
        json.dumps(method_summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    (output_dir / "batch_execution.md").write_text(
        markdown_summary(result), encoding="utf-8"
    )


def execute_batch(
    manifest_path: Path,
    output_dir: Path,
    *,
    project_dir: Path,
    python_bin: str,
    weights_path: Path,
    git_sha: str,
    resume: bool = False,
    runner: Runner = run_logged_stage,
) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    manifest = load_manifest(manifest_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"missing DUSt3R weights: {weights_path}")
    manifest_sha256 = file_sha256(manifest_path)
    weights_sha256 = file_sha256(weights_path)
    ledger_path = output_dir / "batch_execution.json"
    if output_dir.exists():
        if not ledger_path.is_file():
            raise FileNotFoundError(f"resume ledger is missing: {ledger_path}")
        result = json.loads(ledger_path.read_text(encoding="utf-8"))
        if result.get("kind") != "research_practice_identical_cache_batch":
            raise ValueError("resume ledger has the wrong kind")
        if result.get("manifest_sha256") != manifest_sha256:
            raise ValueError("resume manifest checksum mismatch")
        if result.get("weights_sha256") != weights_sha256:
            raise ValueError("resume DUSt3R weights checksum mismatch")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "research_practice_identical_cache_batch",
            "batch_name": str(manifest.get("name", manifest_path.stem)),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "git_sha": git_sha,
            "project_dir": str(project_dir),
            "python_bin": python_bin,
            "weights_path": str(weights_path),
            "weights_sha256": weights_sha256,
            "hardware": hardware_record(),
            "coordinate_contract": {
                "pixel_order": "xy",
                "global_space": "dust3r_aligned_pointmap",
                "join_key": "(alignment_view_index,x,y)",
                "nearest_neighbor_join": False,
                "raw_support_conflicts_preserved": True,
                "conflict_drop_is_ablation_only": True,
            },
            "frozen_config": FROZEN_CONFIG,
            "items": [],
            "summary": {},
        }
        write_batch_outputs(result, output_dir)
    planned_ids = {str(item["id"]) for item in manifest["items"]}
    recorded_ids = [str(item.get("id", "")) for item in result["items"]]
    if len(recorded_ids) != len(set(recorded_ids)):
        raise ValueError("resume batch ledger contains duplicate item IDs")
    if not set(recorded_ids).issubset(planned_ids):
        raise ValueError("resume batch ledger contains items absent from the manifest")
    completed = set(recorded_ids)
    for item in manifest["items"]:
        item_id = str(item["id"])
        if item_id in completed:
            print(f"[resume] batch item already recorded: {item_id}", flush=True)
            continue
        item_dir = output_dir / "items" / item_id
        if item_dir.exists():
            item_ledger = item_dir / "item_execution.json"
            if item_ledger.is_file():
                row = json.loads(item_ledger.read_text(encoding="utf-8"))
                if str(row.get("id", "")) != item_id:
                    raise ValueError(f"resume item ledger ID mismatch: {item_ledger}")
                row["resume_recovered"] = True
            else:
                row = {
                    "id": item_id,
                    "scene_name": str(item.get("expected_scene_name", "")),
                    "pair_group": str(item.get("expected_pair_group", "")),
                    "status": "fail",
                    "failure_stage": "interrupted_partial_output",
                    "error": f"unrecorded partial item directory preserved: {item_dir}",
                    "stages": [],
                    "artifacts": {},
                    "runtime_seconds": 0.0,
                    "resume_recovered": True,
                }
            result["items"].append(row)
            write_batch_outputs(result, output_dir)
            print(f"[resume] preserved existing item directory: {item_id}", flush=True)
            continue
        print(f"\n=== {item['id']} ===", flush=True)
        row = execute_item(
            item,
            project_dir=project_dir,
            python_bin=python_bin,
            weights_path=weights_path,
            output_dir=output_dir,
            runner=runner,
        )
        result["items"].append(row)
        write_batch_outputs(result, output_dir)
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "status": row["status"],
                    "failure_stage": row["failure_stage"],
                    "runtime_seconds": row["runtime_seconds"],
                },
                indent=2,
            ),
            flush=True,
        )
    write_batch_outputs(result, output_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        "Run LightRecon3D research-practice identical-cache batch"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--git_sha", default=os.environ.get("GIT_SHA", ""))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = execute_batch(
        Path(args.manifest),
        Path(args.output_dir),
        project_dir=Path(args.project_dir),
        python_bin=str(args.python_bin),
        weights_path=Path(args.weights_path),
        git_sha=str(args.git_sha),
        resume=bool(args.resume),
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False), flush=True)
    return 0 if result["summary"]["failed_items"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
