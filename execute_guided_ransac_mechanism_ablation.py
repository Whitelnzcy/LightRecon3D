"""Execute support-mechanism ablations on immutable final-batch caches.

The executor never runs DUSt3R or Stage1/Stage2. It reads the exact global
cache, predicted-support records, point-aligned GT, and historical B0/B4 paths
from a completed research-practice batch ledger. Every new prediction is
written below a fresh output root, evaluated on the identical point ordering,
and recorded in a resumable failure ledger.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from guided_plane_ransac import MECHANISM_MODES, METHOD_BY_MODE


SCHEMA_VERSION = 1
DEFAULT_MODES = (
    "none",
    "proposal_only",
    "consensus_only",
    "refit_only",
    "proposal_consensus",
    "all",
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
LOWER_IS_BETTER = {
    "support_conditioned_normal_angular_error_deg",
    "support_conditioned_fragmentation_excess",
    "support_conditioned_overmerge_excess",
    "runtime_seconds",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve_project_path(project_dir: Path, text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else project_dir / path


def verified_artifact(
    project_dir: Path, item: dict[str, Any], key: str
) -> tuple[Path, str]:
    record = item.get("artifacts", {}).get(key)
    if not isinstance(record, dict):
        raise ValueError(f"{item.get('id')} missing artifact record {key}")
    path = resolve_project_path(project_dir, str(record.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    expected = str(record.get("sha256", ""))
    if expected and actual != expected:
        raise ValueError(
            f"{item.get('id')} {key} SHA-256 mismatch: {actual} != {expected}"
        )
    return path, actual


def prediction_semantic_equivalence(
    current_path: Path, historical_path: Path
) -> dict[str, Any]:
    current = np.load(current_path, allow_pickle=False)
    historical = np.load(historical_path, allow_pickle=False)
    exact_fields = ("points", "point_plane_ids", "source_views", "pixel_xy")
    close_fields = ("plane_normals", "plane_offsets", "plane_inlier_counts")
    exact = {
        field: bool(
            field in current.files
            and field in historical.files
            and np.array_equal(current[field], historical[field])
        )
        for field in exact_fields
    }
    close = {
        field: bool(
            field in current.files
            and field in historical.files
            and current[field].shape == historical[field].shape
            and np.allclose(current[field], historical[field], rtol=0, atol=1e-6)
        )
        for field in close_fields
    }
    return {
        "current_path": str(current_path),
        "current_sha256": file_sha256(current_path),
        "historical_path": str(historical_path),
        "historical_sha256": file_sha256(historical_path),
        "exact_fields": exact,
        "close_fields": close,
        "equivalent": bool(all(exact.values()) and all(close.values())),
    }


def run_command(
    command: list[str], project_dir: Path, log_path: Path
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=project_dir,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    runtime = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "command": command,
        "command_text": subprocess.list2cmdline(command),
        "return_code": int(completed.returncode),
        "runtime_seconds": float(runtime),
        "log": str(log_path),
        "status": "pass" if completed.returncode == 0 else "fail",
    }


def mechanism_command(
    python_bin: Path,
    project_dir: Path,
    cache_path: Path,
    support_path: Path,
    output_dir: Path,
    scene_key: str,
    mode: str,
    seed: int,
    frozen_config: dict[str, Any],
    support_refit_weight: float,
) -> list[str]:
    ransac = frozen_config["ransac"]
    guided = frozen_config["guided_ransac"]
    return [
        str(python_bin),
        str(project_dir / "guided_plane_ransac.py"),
        "--global_cloud_npz",
        str(cache_path),
        "--support_npz",
        str(support_path),
        "--output_dir",
        str(output_dir),
        "--scene_key",
        scene_key,
        "--mechanism_mode",
        mode,
        "--min_conf",
        str(frozen_config["evaluation_min_conf"]),
        "--distance_threshold",
        str(ransac["distance_threshold"]),
        "--global_proposal_iterations",
        str(ransac["iterations"]),
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
        "--support_refit_weight",
        str(support_refit_weight),
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
        str(seed),
        "--hypothesis_max_points",
        str(ransac["hypothesis_max_points"]),
        "--component_exact_max_points",
        str(ransac["component_exact_max_points"]),
    ]


def evaluation_command(
    python_bin: Path,
    project_dir: Path,
    gt_path: Path,
    prediction_paths: list[Path],
    output_csv: Path,
    frozen_config: dict[str, Any],
) -> list[str]:
    metrics = frozen_config["metrics"]
    return [
        str(python_bin),
        str(project_dir / "evaluate_global_plane_baselines.py"),
        "--gt_npz",
        str(gt_path),
        "--pred_npz",
        *[str(path) for path in prediction_paths],
        "--output_csv",
        str(output_csv),
        "--match_iou",
        str(metrics["match_iou"]),
        "--fragmentation_iou",
        str(metrics["fragmentation_iou"]),
        "--min_observed_plane_points",
        str(metrics["min_observed_plane_points"]),
    ]


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    result = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            result.append(value)
    return result


def summarize_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    summary: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        for metric in SUMMARY_METRICS:
            values = finite_values(method_rows, metric)
            summary.append(
                {
                    "method": method,
                    "metric": metric,
                    "valid_scenes": int(len(values)),
                    "mean": float(statistics.fmean(values)) if values else float("nan"),
                    "median": float(statistics.median(values)) if values else float("nan"),
                }
            )
    return summary


def paired_summary(
    rows: list[dict[str, Any]], reference_method: str
) -> list[dict[str, Any]]:
    by_method = {
        method: {str(row["item_id"]): row for row in rows if row["method"] == method}
        for method in sorted({str(row["method"]) for row in rows})
    }
    reference = by_method.get(reference_method, {})
    result: list[dict[str, Any]] = []
    for method, method_rows in by_method.items():
        if method == reference_method:
            continue
        common = sorted(set(reference) & set(method_rows))
        for metric in SUMMARY_METRICS:
            deltas = []
            for item_id in common:
                try:
                    current = float(method_rows[item_id][metric])
                    base = float(reference[item_id][metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(current) and math.isfinite(base):
                    deltas.append(current - base)
            result.append(
                {
                    "reference_method": reference_method,
                    "method": method,
                    "metric": metric,
                    "valid_scene_pairs": int(len(deltas)),
                    "mean_delta": float(statistics.fmean(deltas))
                    if deltas
                    else float("nan"),
                    "median_delta": float(statistics.median(deltas))
                    if deltas
                    else float("nan"),
                    "direction": "lower_is_better"
                    if metric in LOWER_IS_BETTER
                    else "higher_is_better",
                    "improvements": int(
                        sum(value < 0 for value in deltas)
                        if metric in LOWER_IS_BETTER
                        else sum(value > 0 for value in deltas)
                    ),
                    "ties": int(sum(value == 0 for value in deltas)),
                    "regressions": int(
                        sum(value > 0 for value in deltas)
                        if metric in LOWER_IS_BETTER
                        else sum(value < 0 for value in deltas)
                    ),
                }
            )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_reusable_mode_stages(
    execution_paths: list[Path],
    *,
    source_batch_sha256: str,
    modes: list[str],
    seeds: list[int],
    support_refit_weight: float,
) -> tuple[dict[tuple[str, int, str], dict[str, Any]], list[dict[str, str]]]:
    """Validate and index completed mode stages from immutable prior ledgers."""

    reusable: dict[tuple[str, int, str], dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    for execution_path in execution_paths:
        execution_path = execution_path.resolve()
        payload = load_json(execution_path)
        if payload.get("kind") != "guided_ransac_mechanism_ablation":
            raise ValueError(f"not a mechanism-ablation ledger: {execution_path}")
        if payload.get("source_batch_sha256") != source_batch_sha256:
            raise ValueError(
                f"reuse ledger refers to a different source batch: {execution_path}"
            )
        if float(payload.get("support_refit_weight", float("nan"))) != float(
            support_refit_weight
        ):
            raise ValueError(
                f"reuse ledger support_refit_weight mismatch: {execution_path}"
            )
        if not set(modes).issubset(set(payload.get("modes", []))):
            raise ValueError(f"reuse ledger lacks requested modes: {execution_path}")
        if not set(seeds).issubset(set(payload.get("seeds", []))):
            raise ValueError(f"reuse ledger lacks requested seeds: {execution_path}")
        sources.append(
            {"path": str(execution_path), "sha256": file_sha256(execution_path)}
        )
        for item in payload.get("items", []):
            item_id = str(item.get("id", ""))
            seed = int(item.get("seed", 0))
            if seed not in seeds:
                continue
            for stage in item.get("modes", []):
                mode = str(stage.get("mode", ""))
                if mode not in modes or stage.get("status") != "pass":
                    continue
                prediction_text = str(stage.get("prediction", ""))
                if not prediction_text:
                    continue
                prediction_path = Path(prediction_text)
                if not prediction_path.is_file():
                    raise FileNotFoundError(prediction_path)
                actual_sha256 = file_sha256(prediction_path)
                if actual_sha256 != str(stage.get("prediction_sha256", "")):
                    raise ValueError(
                        f"reused prediction SHA-256 mismatch: {prediction_path}"
                    )
                key = (item_id, seed, mode)
                if key not in reusable:
                    reusable[key] = {
                        **stage,
                        "reused": True,
                        "reused_from_execution": str(execution_path),
                    }
    return reusable, sources


def markdown_summary(result: dict[str, Any]) -> str:
    summary_lookup = {
        (row["method"], row["metric"]): row for row in result["metric_summary"]
    }
    lines = [
        "# Guided-RANSAC mechanism ablation",
        "",
        (
            f"Passed {result['summary']['passed_items']}/"
            f"{result['summary']['items']} scene/seed items. "
            f"Modes: {', '.join(result['modes'])}."
        ),
        "",
        "| Method | Pairwise F1 mean | median | Coverage mean | Runtime mean, s |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in sorted({row["method"] for row in result["metric_summary"]}):
        f1 = summary_lookup.get((method, "support_partition_pairwise_f1"), {})
        coverage = summary_lookup.get((method, "support_coverage"), {})
        runtime = summary_lookup.get((method, "runtime_seconds"), {})
        lines.append(
            f"| {method} | {f1.get('mean', float('nan')):.6f} | "
            f"{f1.get('median', float('nan')):.6f} | "
            f"{coverage.get('mean', float('nan')):.6f} | "
            f"{runtime.get('mean', float('nan')):.3f} |"
        )
    equivalence = result["summary"].get("historical_b4_equivalence", {})
    lines.extend(
        [
            "",
            (
                "Historical B4 semantic equivalence checks: "
                f"{equivalence.get('passed', 0)}/{equivalence.get('checks', 0)} passed. "
                "Checks cover assignments, plane parameters, point order, view registry, "
                "and pixel registry."
            ),
            "",
            "Paired deltas against both archived B0 and the matched no-support control "
            "are stored in `paired_summary.json` and `paired_summary.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def execute_ablation(
    source_batch_json: Path,
    output_dir: Path,
    modes: list[str],
    seeds: list[int],
    item_ids: list[str],
    support_refit_weight: float,
    reuse_execution_jsons: list[Path] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    project_dir = Path(__file__).resolve().parent
    source_batch_json = source_batch_json.resolve()
    batch = load_json(source_batch_json)
    source_batch_sha256 = file_sha256(source_batch_json)
    frozen_config = batch.get("frozen_config")
    if not isinstance(frozen_config, dict):
        raise ValueError("source batch lacks frozen_config")
    available_items = [
        item for item in batch.get("items", []) if item.get("status") == "pass"
    ]
    if item_ids:
        requested = set(item_ids)
        available_items = [item for item in available_items if item.get("id") in requested]
        missing = sorted(requested - {str(item.get("id")) for item in available_items})
        if missing:
            raise ValueError(f"requested items are not passed source items: {missing}")
    if not available_items:
        raise ValueError("no passed source items selected")

    reusable_stages, reuse_sources = load_reusable_mode_stages(
        list(reuse_execution_jsons or []),
        source_batch_sha256=source_batch_sha256,
        modes=modes,
        seeds=seeds,
        support_refit_weight=support_refit_weight,
    )

    preflight: list[dict[str, Any]] = []
    for item in available_items:
        record = {"id": str(item["id"]), "scene_name": str(item["scene_name"])}
        for key in (
            "global_cloud_cache",
            "direct_support_records",
            "point_aligned_gt",
            "global_ransac",
            "learning_guided_ransac",
        ):
            path, sha256 = verified_artifact(project_dir, item, key)
            record[key] = {"path": str(path), "sha256": sha256}
        preflight.append(record)

    output_dir.mkdir(parents=True, exist_ok=False)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=project_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout
    execution: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "guided_ransac_mechanism_ablation",
        "source_batch_json": str(source_batch_json),
        "source_batch_sha256": source_batch_sha256,
        "reuse_execution_ledgers": reuse_sources,
        "source_git_sha": str(batch.get("git_sha", "")),
        "execution_git_sha": git_head,
        "dirty_worktree": bool(git_status.strip()),
        "python_bin": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "modes": modes,
        "mechanism_flags": {mode: MECHANISM_MODES[mode] for mode in modes},
        "seeds": seeds,
        "support_refit_weight": float(support_refit_weight),
        "preflight": preflight,
        "items": [],
        "summary": {},
    }
    execution_path = output_dir / "mechanism_ablation_execution.json"

    def checkpoint() -> None:
        execution_path.write_text(
            json.dumps(execution, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf-8",
        )

    started = time.perf_counter()
    aggregate_rows: list[dict[str, Any]] = []
    for source_item in available_items:
        source_record = next(row for row in preflight if row["id"] == source_item["id"])
        for seed in seeds:
            item_id = str(source_item["id"])
            run_id = f"{item_id}_seed_{seed}"
            item_output = output_dir / "items" / run_id
            item: dict[str, Any] = {
                "id": item_id,
                "run_id": run_id,
                "scene_name": str(source_item["scene_name"]),
                "seed": int(seed),
                "status": "running",
                "failure_stage": "",
                "error": "",
                "modes": [],
            }
            execution["items"].append(item)
            checkpoint()
            try:
                predictions: list[Path] = [Path(source_record["global_ransac"]["path"])]
                for mode in modes:
                    reuse_key = (item_id, int(seed), mode)
                    stage = reusable_stages.get(reuse_key)
                    if stage is None:
                        mode_output = item_output / mode
                        command = mechanism_command(
                            Path(sys.executable),
                            project_dir,
                            Path(source_record["global_cloud_cache"]["path"]),
                            Path(source_record["direct_support_records"]["path"]),
                            mode_output,
                            item_id,
                            mode,
                            seed,
                            frozen_config,
                            support_refit_weight,
                        )
                        stage = run_command(
                            command,
                            project_dir,
                            item_output / "logs" / f"{mode}.log",
                        )
                        stage["mode"] = mode
                        item["modes"].append(stage)
                        checkpoint()
                        if stage["return_code"] != 0:
                            raise RuntimeError(f"mechanism mode failed: {mode}")
                        manifest_path = (
                            mode_output / "guided_plane_ransac_manifest.json"
                        )
                        manifest = load_json(manifest_path)
                        prediction_path = resolve_project_path(
                            project_dir, str(manifest["npz"])
                        )
                        if not prediction_path.is_file():
                            raise FileNotFoundError(prediction_path)
                        stage["manifest"] = str(manifest_path)
                        stage["manifest_sha256"] = file_sha256(manifest_path)
                        stage["prediction"] = str(prediction_path)
                        stage["prediction_sha256"] = file_sha256(prediction_path)
                    else:
                        stage = dict(stage)
                        item["modes"].append(stage)
                        prediction_path = Path(str(stage["prediction"]))
                    predictions.append(prediction_path)
                    if mode == "proposal_consensus" and seed == 0:
                        equivalence = prediction_semantic_equivalence(
                            prediction_path,
                            Path(source_record["learning_guided_ransac"]["path"]),
                        )
                        stage["historical_b4_equivalence"] = equivalence
                        if not equivalence["equivalent"]:
                            raise ValueError(
                                f"historical B4 semantic mismatch for {item_id}"
                            )
                    checkpoint()

                metrics_csv = item_output / "evaluation" / "mechanism_metrics.csv"
                evaluation = run_command(
                    evaluation_command(
                        Path(sys.executable),
                        project_dir,
                        Path(source_record["point_aligned_gt"]["path"]),
                        predictions,
                        metrics_csv,
                        frozen_config,
                    ),
                    project_dir,
                    item_output / "logs" / "evaluation.log",
                )
                item["evaluation"] = evaluation
                if evaluation["return_code"] != 0:
                    raise RuntimeError("mechanism evaluation failed")
                metric_rows = json.loads(
                    metrics_csv.with_suffix(".json").read_text(encoding="utf-8")
                )
                method_to_mode = {METHOD_BY_MODE[mode]: mode for mode in modes}
                method_to_mode["global_ransac_cc"] = "archived_b0"
                for row in metric_rows:
                    row.update(
                        {
                            "item_id": item_id,
                            "scene_name": str(source_item["scene_name"]),
                            "seed": int(seed),
                            "mode": method_to_mode.get(str(row["method"]), "unknown"),
                        }
                    )
                    aggregate_rows.append(row)
                item["metric_rows"] = int(len(metric_rows))
                item["status"] = "pass"
            except Exception as error:
                item["status"] = "fail"
                if not item["failure_stage"]:
                    item["failure_stage"] = (
                        item["modes"][-1].get("mode", "evaluation")
                        if item["modes"]
                        else "preflight"
                    )
                item["error"] = f"{type(error).__name__}: {error}"
            checkpoint()

    equivalence_checks = [
        mode["historical_b4_equivalence"]
        for item in execution["items"]
        for mode in item.get("modes", [])
        if "historical_b4_equivalence" in mode
    ]
    execution["summary"] = {
        "items": int(len(execution["items"])),
        "passed_items": int(sum(item["status"] == "pass" for item in execution["items"])),
        "failed_items": int(sum(item["status"] != "pass" for item in execution["items"])),
        "unique_scenes": int(len({item["scene_name"] for item in execution["items"]})),
        "metric_rows": int(len(aggregate_rows)),
        "runtime_seconds": float(time.perf_counter() - started),
        "historical_b4_equivalence": {
            "checks": int(len(equivalence_checks)),
            "passed": int(sum(row["equivalent"] for row in equivalence_checks)),
        },
    }
    metric_summary = summarize_metrics(aggregate_rows)
    paired_b0 = paired_summary(aggregate_rows, "global_ransac_cc")
    paired_none = paired_summary(aggregate_rows, METHOD_BY_MODE["none"])
    execution["metric_summary"] = metric_summary
    execution["paired_summary_references"] = [
        "global_ransac_cc",
        METHOD_BY_MODE["none"],
    ]
    checkpoint()
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate_rows, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    (output_dir / "metric_summary.json").write_text(
        json.dumps(metric_summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    write_csv(output_dir / "metric_summary.csv", metric_summary)
    paired = paired_b0 + paired_none
    (output_dir / "paired_summary.json").write_text(
        json.dumps(paired, indent=2, allow_nan=True), encoding="utf-8"
    )
    write_csv(output_dir / "paired_summary.csv", paired)
    result = {**execution, "metric_summary": metric_summary}
    (output_dir / "mechanism_ablation_summary.md").write_text(
        markdown_summary(result), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        "Run support proposal/consensus/refit ablations on immutable caches"
    )
    parser.add_argument("--source_batch_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MECHANISM_MODES),
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--item_ids", nargs="*", default=[])
    parser.add_argument(
        "--reuse_execution_json",
        action="append",
        default=[],
        help=(
            "Immutable prior mechanism-ablation ledger. Completed mode outputs "
            "are checksum-verified and referenced read-only in the new run."
        ),
    )
    parser.add_argument("--support_refit_weight", type=float, default=1.0)
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must not contain duplicates")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if args.support_refit_weight < 0:
        parser.error("--support_refit_weight must be non-negative")
    result = execute_ablation(
        Path(args.source_batch_json),
        Path(args.output_dir),
        list(args.modes),
        list(args.seeds),
        list(args.item_ids),
        float(args.support_refit_weight),
        [Path(value) for value in args.reuse_execution_json],
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return 0 if result["summary"]["failed_items"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
