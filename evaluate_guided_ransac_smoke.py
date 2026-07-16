"""Run and gate the guided-RANSAC method on an archived identical-cache batch."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_global_plane_baselines import evaluate_arrays
from execute_research_practice_batch import FROZEN_CONFIG, run_logged_stage
from guided_plane_ransac import METHOD
from research_practice_batch import file_record, file_sha256


SCHEMA_VERSION = 1


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot take median of an empty list")
    return float(statistics.median(values))


def checked_artifact(item: dict[str, Any], key: str) -> Path:
    record = item.get("artifacts", {}).get(key, {})
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"{item.get('id')} missing {key}: {path}")
    expected = str(record.get("sha256", "")).lower()
    if expected and file_sha256(path) != expected:
        raise RuntimeError(f"{item.get('id')} checksum mismatch for {key}: {path}")
    return path


def guided_command(
    python_bin: str,
    project_dir: Path,
    *,
    item_id: str,
    cache_path: Path,
    support_path: Path,
    output_dir: Path,
) -> list[str]:
    ransac = FROZEN_CONFIG["ransac"]
    guided = FROZEN_CONFIG["guided_ransac"]
    return [
        python_bin,
        str(project_dir / "guided_plane_ransac.py"),
        "--global_cloud_npz",
        str(cache_path),
        "--support_npz",
        str(support_path),
        "--output_dir",
        str(output_dir),
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
    ]


def prediction_metrics(gt_path: Path, prediction_path: Path) -> dict[str, Any]:
    with np.load(gt_path, allow_pickle=False) as gt, np.load(
        prediction_path, allow_pickle=False
    ) as pred:
        if pred["points"].shape != gt["points"].shape or not np.allclose(
            pred["points"], gt["points"], atol=1e-5
        ):
            raise ValueError(
                f"prediction is not indexed on the identical cache: {prediction_path}"
            )
        metrics = evaluate_arrays(
            gt["points"],
            pred["point_plane_ids"],
            pred["plane_normals"],
            pred["plane_offsets"],
            gt["point_plane_ids"],
            gt["plane_normals"],
            FROZEN_CONFIG["metrics"]["match_iou"],
            FROZEN_CONFIG["metrics"]["fragmentation_iou"],
            FROZEN_CONFIG["metrics"]["min_observed_plane_points"],
        )
        metrics["method"] = (
            str(pred["method"].item())
            if "method" in pred.files
            else prediction_path.stem
        )
        metrics["runtime_seconds"] = (
            float(pred["runtime_seconds"])
            if "runtime_seconds" in pred.files
            else float("nan")
        )
        metrics["prediction"] = str(prediction_path)
    return metrics


def compact_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_f1": float(metrics["support_partition_pairwise_f1"]),
        f"{prefix}_coverage": float(metrics["support_coverage"]),
        f"{prefix}_plane_precision": float(
            metrics["support_conditioned_plane_precision"]
        ),
        f"{prefix}_plane_recall": float(
            metrics["support_conditioned_plane_recall_observed"]
        ),
        f"{prefix}_fragmentation": float(
            metrics["support_conditioned_fragmentation_excess"]
        ),
        f"{prefix}_overmerge": float(
            metrics["support_conditioned_overmerge_excess"]
        ),
        f"{prefix}_runtime_seconds": float(metrics["runtime_seconds"]),
        f"{prefix}_planes": int(metrics["pred_plane_count"]),
    }


def check(name: str, value: float, rule: str, threshold: float) -> dict[str, Any]:
    if rule == ">=":
        passed = value >= threshold
    elif rule == "<=":
        passed = value <= threshold
    else:
        raise ValueError(rule)
    return {
        "name": name,
        "value": float(value),
        "rule": rule,
        "threshold": float(threshold),
        "passed": bool(passed),
    }


def gate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [row for row in rows if row["status"] == "pass"]
    if len(passed) != len(rows):
        return {
            "decision": "guided_smoke_execution_failed",
            "method_gate_passed": False,
            "quality_gate_passed": False,
            "efficiency_gate_passed": False,
            "quality_checks": [],
            "efficiency_checks": [],
            "diagnostics": {"groups": len(rows), "passed_groups": len(passed)},
        }
    f1_deltas = [row["guided_f1"] - row["ransac_f1"] for row in passed]
    win_rate = sum(delta > 0 for delta in f1_deltas) / len(f1_deltas)
    guided_coverage = median([row["guided_coverage"] for row in passed])
    guided_precision = median([row["guided_plane_precision"] for row in passed])
    ransac_precision = median([row["ransac_plane_precision"] for row in passed])
    guided_overmerge = median([row["guided_overmerge"] for row in passed])
    ransac_overmerge = median([row["ransac_overmerge"] for row in passed])
    runtime_ratios = [
        row["guided_runtime_seconds"] / row["ransac_runtime_seconds"]
        for row in passed
        if row["ransac_runtime_seconds"] > 0
    ]
    median_delta = median(f1_deltas)
    median_runtime_ratio = median(runtime_ratios)
    common = [
        check("median_guided_coverage", guided_coverage, ">=", 0.90),
        check(
            "median_guided_plane_precision_not_worse",
            guided_precision,
            ">=",
            ransac_precision,
        ),
        check(
            "median_guided_overmerge_not_worse",
            guided_overmerge,
            "<=",
            ransac_overmerge,
        ),
    ]
    quality_checks = [
        check("median_guided_f1_gain", median_delta, ">=", 0.02),
        check("guided_group_win_rate", win_rate, ">=", 2.0 / 3.0),
        *common,
    ]
    efficiency_checks = [
        check("median_guided_f1_noninferiority", median_delta, ">=", -0.01),
        check("median_guided_runtime_ratio", median_runtime_ratio, "<=", 0.75),
        *common,
    ]
    quality_passed = all(row["passed"] for row in quality_checks)
    efficiency_passed = all(row["passed"] for row in efficiency_checks)
    method_passed = quality_passed or efficiency_passed
    return {
        "decision": (
            "guided_smoke_signal_expand_independent_scenes"
            if method_passed
            else "keep_guided_as_ablation_ransac_primary"
        ),
        "method_gate_passed": method_passed,
        "quality_gate_passed": quality_passed,
        "efficiency_gate_passed": efficiency_passed,
        "quality_checks": quality_checks,
        "efficiency_checks": efficiency_checks,
        "diagnostics": {
            "groups": len(passed),
            "independent_scenes": len({row["scene_name"] for row in passed}),
            "median_ransac_f1": median([row["ransac_f1"] for row in passed]),
            "median_guided_f1": median([row["guided_f1"] for row in passed]),
            "median_guided_f1_gain": median_delta,
            "guided_group_wins": int(sum(delta > 0 for delta in f1_deltas)),
            "guided_group_win_rate": float(win_rate),
            "median_guided_coverage": guided_coverage,
            "median_ransac_plane_precision": ransac_precision,
            "median_guided_plane_precision": guided_precision,
            "median_ransac_overmerge": ransac_overmerge,
            "median_guided_overmerge": guided_overmerge,
            "median_ransac_runtime_seconds": median(
                [row["ransac_runtime_seconds"] for row in passed]
            ),
            "median_guided_runtime_seconds": median(
                [row["guided_runtime_seconds"] for row in passed]
            ),
            "median_guided_runtime_ratio": median_runtime_ratio,
        },
    }


def markdown_report(result: dict[str, Any]) -> str:
    gate = result["gate"]
    lines = [
        "# Learning-guided RANSAC smoke",
        "",
        f"Decision: `{gate['decision']}`",
        "",
        "The learned Stage1/Stage2 supports generate proposals; all scoring and refitting use the identical frozen DUSt3R global cache.",
        "",
        "| Group | Scene | RANSAC F1 | Guided F1 | Delta | RANSAC s | Guided s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["items"]:
        if row["status"] != "pass":
            lines.append(
                f"| {row['id']} | {row['scene_name']} | fail | fail |  |  |  |"
            )
            continue
        lines.append(
            f"| {row['id']} | {row['scene_name']} | {row['ransac_f1']:.6f} | "
            f"{row['guided_f1']:.6f} | {row['guided_f1'] - row['ransac_f1']:.6f} | "
            f"{row['ransac_runtime_seconds']:.3f} | {row['guided_runtime_seconds']:.3f} |"
        )
    for title, checks in (
        ("Quality path", gate["quality_checks"]),
        ("Efficiency path", gate["efficiency_checks"]),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Check | Value | Rule | Pass |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in checks:
            lines.append(
                f"| {row['name']} | {row['value']:.6f} | "
                f"{row['rule']} {row['threshold']:.6f} | {row['passed']} |"
            )
    lines.extend(
        [
            "",
            "Passing this smoke is not final promotion: the method must still be evaluated on at least eight independent scene IDs.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "guided_smoke.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    rows = result["items"]
    keys = set().union(*(row.keys() for row in rows))
    with (output_dir / "guided_smoke_per_group.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(keys), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "guided_smoke.md").write_text(
        markdown_report(result), encoding="utf-8"
    )


def run_smoke(
    batch_execution_json: Path,
    output_dir: Path,
    *,
    project_dir: Path,
    python_bin: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    batch = json.loads(batch_execution_json.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "learning_guided_ransac_identical_cache_smoke",
        "source_batch_execution_json": str(batch_execution_json),
        "source_batch_execution_sha256": file_sha256(batch_execution_json),
        "source_git_sha": str(batch.get("git_sha", "")),
        "method": METHOD,
        "frozen_config": {
            "ransac": FROZEN_CONFIG["ransac"],
            "guided_ransac": FROZEN_CONFIG["guided_ransac"],
            "metrics": FROZEN_CONFIG["metrics"],
        },
        "items": [],
        "gate": {},
    }
    for item in batch.get("items", []):
        row: dict[str, Any] = {
            "id": str(item.get("id", "")),
            "scene_name": str(item.get("scene_name", "")),
            "status": "running",
            "error": "",
        }
        try:
            if item.get("status") != "pass":
                raise ValueError("source batch item did not pass")
            cache_path = checked_artifact(item, "global_cloud_cache")
            support_path = checked_artifact(item, "direct_support_records")
            gt_path = checked_artifact(item, "point_aligned_gt")
            ransac_path = checked_artifact(item, "global_ransac")
            item_dir = output_dir / "items" / row["id"]
            guided_dir = item_dir / "guided_ransac"
            command = guided_command(
                python_bin,
                project_dir,
                item_id=row["id"],
                cache_path=cache_path,
                support_path=support_path,
                output_dir=guided_dir,
            )
            stage = run_logged_stage(
                "learning_guided_ransac",
                command,
                project_dir,
                item_dir / "learning_guided_ransac.log",
            )
            row["stage"] = stage
            if stage["return_code"] != 0:
                raise RuntimeError(
                    f"guided RANSAC exited with {stage['return_code']}"
                )
            guided_path = (
                guided_dir
                / f"{row['id']}_{METHOD}_full_pointcloud_editable_planes_data.npz"
            )
            ransac_metrics = prediction_metrics(gt_path, ransac_path)
            guided_metrics = prediction_metrics(gt_path, guided_path)
            row.update(compact_metrics(ransac_metrics, "ransac"))
            row.update(compact_metrics(guided_metrics, "guided"))
            row["guided_artifact"] = file_record(guided_path)
            row["guided_manifest"] = file_record(
                guided_dir / "guided_plane_ransac_manifest.json"
            )
            row["status"] = "pass"
        except Exception as error:
            row["status"] = "fail"
            row["error"] = f"{type(error).__name__}: {error}"
        result["items"].append(row)
        result["gate"] = gate_rows(result["items"])
        write_outputs(result, output_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        "Run learning-guided RANSAC on an archived research-practice smoke batch"
    )
    parser.add_argument("--batch_execution_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--python_bin", default=sys.executable)
    args = parser.parse_args()
    result = run_smoke(
        Path(args.batch_execution_json),
        Path(args.output_dir),
        project_dir=Path(args.project_dir),
        python_bin=str(args.python_bin),
    )
    print(json.dumps(result["gate"], indent=2, ensure_ascii=False))
    return 0 if all(row["status"] == "pass" for row in result["items"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
