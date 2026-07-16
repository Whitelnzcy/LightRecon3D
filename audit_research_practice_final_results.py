"""Audit the frozen eight-scene guided-RANSAC result without recomputation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
FULL_CACHE = "full_cache"
SUPPORT_RECORDS = "support_records"
RANSAC = "global_ransac_cc"
GUIDED = "learning_guided_ransac_cc"
MANUAL_RAW = "stage2_manual_merge_support"
MANUAL_DROP = "stage2_manual_merge_unique_conflict_drop"
DIRECT_DROP = "stage2_direct_svd_unique_conflict_drop"

MINIMUM_INDEPENDENT_SCENES = 8
MINIMUM_MEDIAN_F1_GAIN = 0.02
MINIMUM_WIN_RATE = 2.0 / 3.0
MINIMUM_COVERAGE = 0.90
MAXIMUM_EFFICIENCY_RUNTIME_RATIO = 0.75
MINIMUM_EFFICIENCY_F1_DELTA = -0.01
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260716

REPORT_METRICS = (
    ("pairwise_f1", "support_partition_pairwise_f1", True),
    ("purity_completeness_f1", "support_partition_purity_completeness_f1", True),
    ("matched_iou", "support_conditioned_matched_iou", True),
    ("plane_precision", "support_conditioned_plane_precision", True),
    ("plane_recall_all_gt", "support_conditioned_plane_recall_all_gt", True),
    ("coverage", "support_coverage", True),
    ("normal_error_deg", "support_conditioned_normal_angular_error_deg", False),
    ("fragmentation_excess", "support_conditioned_fragmentation_excess", False),
    ("overmerge_excess", "support_conditioned_overmerge_excess", False),
    ("runtime_seconds", "runtime_seconds", False),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot take median of an empty list")
    return float(statistics.median(values))


def finite_metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{row.get('item_id')} {row.get('method')} missing numeric {key}"
        )
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(
            f"{row.get('item_id')} {row.get('method')} has non-finite {key}"
        )
    return value


def finite_metric_or_none(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot bootstrap an empty list")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    rng = random.Random(seed)
    count = len(values)
    estimates = []
    for _ in range(samples):
        estimates.append(
            statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        )
    estimates.sort()
    low = estimates[int(0.025 * (samples - 1))]
    high = estimates[int(0.975 * (samples - 1))]
    return {
        "samples": int(samples),
        "seed": int(seed),
        "mean": float(statistics.fmean(values)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def exact_two_sided_sign_pvalue(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return float(min(1.0, 2.0 * probability))


def index_rows(
    rows: list[dict[str, Any]], family: str
) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("metric_family") != family:
            continue
        item_id = str(row.get("item_id", ""))
        method = str(row.get("method", ""))
        if not item_id or not method:
            raise ValueError(f"{family} rows require item_id and method")
        methods = indexed.setdefault(item_id, {})
        if method in methods:
            raise ValueError(f"duplicate metric row for {family}/{item_id}/{method}")
        methods[method] = row
    return indexed


def paired_rows(
    rows: list[dict[str, Any]], batch: dict[str, Any]
) -> list[dict[str, Any]]:
    indexed = index_rows(rows, FULL_CACHE)
    items = batch.get("items", [])
    failed = [str(item.get("id", "")) for item in items if item.get("status") != "pass"]
    if failed:
        raise ValueError(f"batch contains failed items: {failed}")
    output: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    seen_items: set[str] = set()
    for item in items:
        item_id = str(item.get("id", ""))
        scene_name = str(item.get("scene_name", ""))
        if not item_id or not scene_name:
            raise ValueError("passed batch items require id and scene_name")
        if item_id in seen_items:
            raise ValueError(f"duplicate batch item: {item_id}")
        if scene_name in seen_scenes:
            raise ValueError(f"duplicate independent scene: {scene_name}")
        seen_items.add(item_id)
        seen_scenes.add(scene_name)
        methods = indexed.get(item_id, {})
        missing = [method for method in (RANSAC, GUIDED) if method not in methods]
        if missing:
            raise ValueError(f"{item_id} missing full-cache methods: {missing}")
        ransac = methods[RANSAC]
        guided = methods[GUIDED]
        for source in (ransac, guided):
            if str(source.get("scene_name", "")) != scene_name:
                raise ValueError(f"scene mismatch for {item_id}/{source.get('method')}")
        row: dict[str, Any] = {"item_id": item_id, "scene_name": scene_name}
        for short_name, metric_name, _ in REPORT_METRICS:
            reader = (
                finite_metric_or_none
                if short_name == "normal_error_deg"
                else finite_metric
            )
            row[f"ransac_{short_name}"] = reader(ransac, metric_name)
            row[f"guided_{short_name}"] = reader(guided, metric_name)
            if (
                row[f"ransac_{short_name}"] is None
                or row[f"guided_{short_name}"] is None
            ):
                row[f"delta_{short_name}"] = None
            else:
                row[f"delta_{short_name}"] = (
                    row[f"guided_{short_name}"] - row[f"ransac_{short_name}"]
                )
        row["guided_f1_win"] = row["delta_pairwise_f1"] > 0
        row["runtime_ratio"] = (
            row["guided_runtime_seconds"] / row["ransac_runtime_seconds"]
            if row["ransac_runtime_seconds"] > 0
            else math.inf
        )
        output.append(row)
    metric_item_ids = set(indexed)
    if metric_item_ids != seen_items:
        raise ValueError(
            "full-cache metric item IDs do not match passed batch items: "
            f"metrics_only={sorted(metric_item_ids - seen_items)}, "
            f"batch_only={sorted(seen_items - metric_item_ids)}"
        )
    return output


def check(name: str, value: float, rule: str, threshold: float) -> dict[str, Any]:
    if rule == ">=":
        passed = value >= threshold
    elif rule == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"unsupported rule: {rule}")
    return {
        "name": name,
        "value": float(value),
        "rule": rule,
        "threshold": float(threshold),
        "passed": bool(passed),
    }


def final_gate(per_scene: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_scene:
        raise ValueError("no paired scene rows")
    scenes = len({row["scene_name"] for row in per_scene})
    f1_deltas = [row["delta_pairwise_f1"] for row in per_scene]
    wins = sum(delta > 0 for delta in f1_deltas)
    losses = sum(delta < 0 for delta in f1_deltas)
    ties = len(f1_deltas) - wins - losses
    win_rate = wins / len(f1_deltas)
    median_delta = median(f1_deltas)
    guided_coverage = median([row["guided_coverage"] for row in per_scene])
    ransac_precision = median([row["ransac_plane_precision"] for row in per_scene])
    guided_precision = median([row["guided_plane_precision"] for row in per_scene])
    ransac_overmerge = median([row["ransac_overmerge_excess"] for row in per_scene])
    guided_overmerge = median([row["guided_overmerge_excess"] for row in per_scene])
    runtime_ratio = median([row["runtime_ratio"] for row in per_scene])
    scene_check = check(
        "independent_scene_count",
        float(scenes),
        ">=",
        float(MINIMUM_INDEPENDENT_SCENES),
    )
    common = [
        check("median_guided_coverage", guided_coverage, ">=", MINIMUM_COVERAGE),
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
        check(
            "median_guided_f1_gain",
            median_delta,
            ">=",
            MINIMUM_MEDIAN_F1_GAIN,
        ),
        check("guided_scene_win_rate", win_rate, ">=", MINIMUM_WIN_RATE),
        *common,
    ]
    efficiency_checks = [
        check(
            "median_guided_f1_noninferiority",
            median_delta,
            ">=",
            MINIMUM_EFFICIENCY_F1_DELTA,
        ),
        check(
            "median_guided_runtime_ratio",
            runtime_ratio,
            "<=",
            MAXIMUM_EFFICIENCY_RUNTIME_RATIO,
        ),
        *common,
    ]
    quality_passed = scene_check["passed"] and all(
        row["passed"] for row in quality_checks
    )
    efficiency_passed = scene_check["passed"] and all(
        row["passed"] for row in efficiency_checks
    )
    method_passed = quality_passed or efficiency_passed
    if not scene_check["passed"]:
        decision = "insufficient_independent_scenes_for_final_decision"
    elif method_passed:
        decision = "promote_learning_guided_ransac_final"
    else:
        decision = "retain_global_ransac_primary_guided_ablation"
    return {
        "decision": decision,
        "method_gate_passed": bool(method_passed),
        "quality_gate_passed": bool(quality_passed),
        "efficiency_gate_passed": bool(efficiency_passed),
        "scene_check": scene_check,
        "quality_checks": quality_checks,
        "efficiency_checks": efficiency_checks,
        "diagnostics": {
            "view_groups": len(per_scene),
            "independent_scenes": scenes,
            "median_ransac_f1": median(
                [row["ransac_pairwise_f1"] for row in per_scene]
            ),
            "median_guided_f1": median(
                [row["guided_pairwise_f1"] for row in per_scene]
            ),
            "median_guided_f1_gain": median_delta,
            "mean_guided_f1_gain": float(statistics.fmean(f1_deltas)),
            "guided_scene_wins": int(wins),
            "guided_scene_losses": int(losses),
            "guided_scene_ties": int(ties),
            "guided_scene_win_rate": float(win_rate),
            "f1_sign_test_two_sided_p": exact_two_sided_sign_pvalue(wins, losses),
            "median_guided_coverage": guided_coverage,
            "median_ransac_plane_precision": ransac_precision,
            "median_guided_plane_precision": guided_precision,
            "median_ransac_overmerge": ransac_overmerge,
            "median_guided_overmerge": guided_overmerge,
            "median_ransac_runtime_seconds": median(
                [row["ransac_runtime_seconds"] for row in per_scene]
            ),
            "median_guided_runtime_seconds": median(
                [row["guided_runtime_seconds"] for row in per_scene]
            ),
            "median_guided_runtime_ratio": runtime_ratio,
            "paired_f1_mean_delta_bootstrap": bootstrap_mean_ci(f1_deltas),
        },
    }


def metric_summary(per_scene: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, (short_name, _, higher_is_better) in enumerate(REPORT_METRICS):
        valid_rows = [
            row
            for row in per_scene
            if row[f"ransac_{short_name}"] is not None
            and row[f"guided_{short_name}"] is not None
        ]
        if not valid_rows:
            continue
        ransac = [row[f"ransac_{short_name}"] for row in valid_rows]
        guided = [row[f"guided_{short_name}"] for row in valid_rows]
        deltas = [row[f"delta_{short_name}"] for row in valid_rows]
        wins = sum(
            (delta > 0 if higher_is_better else delta < 0) for delta in deltas
        )
        output.append(
            {
                "metric": short_name,
                "higher_is_better": bool(higher_is_better),
                "valid_scene_pairs": len(valid_rows),
                "ransac_mean": float(statistics.fmean(ransac)),
                "guided_mean": float(statistics.fmean(guided)),
                "mean_delta_guided_minus_ransac": float(statistics.fmean(deltas)),
                "ransac_median": median(ransac),
                "guided_median": median(guided),
                "median_paired_delta_guided_minus_ransac": median(deltas),
                "guided_scene_wins": int(wins),
                "paired_mean_delta_bootstrap": bootstrap_mean_ci(
                    deltas, seed=BOOTSTRAP_SEED + index
                ),
            }
        )
    return output


def optional_median(
    rows: list[dict[str, Any]], family: str, method: str, key: str
) -> float | None:
    values = [
        finite_metric(row, key)
        for row in rows
        if row.get("metric_family") == family and row.get("method") == method
    ]
    return median(values) if values else None


def coverage_diagnostics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {
        "manual_raw_support_gt_coverage_median": optional_median(
            rows, SUPPORT_RECORDS, MANUAL_RAW, "gt_labeled_record_coverage"
        ),
        "manual_drop_support_gt_coverage_median": optional_median(
            rows, SUPPORT_RECORDS, MANUAL_DROP, "gt_labeled_record_coverage"
        ),
        "direct_drop_support_gt_coverage_median": optional_median(
            rows, SUPPORT_RECORDS, DIRECT_DROP, "gt_labeled_record_coverage"
        ),
        "manual_drop_full_cache_coverage_median": optional_median(
            rows, FULL_CACHE, MANUAL_DROP, "support_coverage"
        ),
        "direct_drop_full_cache_coverage_median": optional_median(
            rows, FULL_CACHE, DIRECT_DROP, "support_coverage"
        ),
    }


def markdown_report(result: dict[str, Any]) -> str:
    gate = result["gate"]
    diagnostics = gate["diagnostics"]
    lines = [
        "# Final eight-scene learning-guided RANSAC audit",
        "",
        f"Decision: `{gate['decision']}`",
        "",
        (
            f"Evaluated {diagnostics['view_groups']} view groups from "
            f"{diagnostics['independent_scenes']} independent scene IDs on the "
            "identical frozen DUSt3R caches."
        ),
        "",
        "The final decision reuses the frozen smoke thresholds. Bootstrap intervals are report diagnostics and do not change the promotion gate.",
        "",
        "| Scene | RANSAC F1 | Guided F1 | Delta | RANSAC s | Guided s | Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_scene"]:
        lines.append(
            f"| {row['scene_name']} | {row['ransac_pairwise_f1']:.6f} | "
            f"{row['guided_pairwise_f1']:.6f} | {row['delta_pairwise_f1']:+.6f} | "
            f"{row['ransac_runtime_seconds']:.3f} | "
            f"{row['guided_runtime_seconds']:.3f} | {row['runtime_ratio']:.3f} |"
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
            "## Report-ready aggregate",
            "",
            "| Metric | RANSAC mean | Guided mean | Mean delta | RANSAC median | Guided median | Guided wins | Valid pairs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["metric_summary"]:
        lines.append(
            f"| {row['metric']} | {row['ransac_mean']:.6f} | "
            f"{row['guided_mean']:.6f} | "
            f"{row['mean_delta_guided_minus_ransac']:+.6f} | "
            f"{row['ransac_median']:.6f} | {row['guided_median']:.6f} | "
            f"{row['guided_scene_wins']}/{row['valid_scene_pairs']} | "
            f"{row['valid_scene_pairs']} |"
        )
    ci = diagnostics["paired_f1_mean_delta_bootstrap"]
    lines.extend(
        [
            "",
            "## Uncertainty and coverage guardrails",
            "",
            (
                "Paired scene-bootstrap mean F1 delta: "
                f"{ci['mean']:+.6f}, exploratory 95% interval "
                f"[{ci['ci95_low']:+.6f}, {ci['ci95_high']:+.6f}] "
                f"({ci['samples']} resamples, seed {ci['seed']})."
            ),
            (
                "Exact two-sided sign-test p-value: "
                f"{diagnostics['f1_sign_test_two_sided_p']:.6f}. With eight scenes, "
                "this is descriptive evidence rather than a broad generalization claim."
            ),
            "",
        ]
    )
    coverage = result["coverage_diagnostics"]
    if coverage["manual_drop_full_cache_coverage_median"] is not None:
        lines.append(
            "Conflict-drop variants remain coverage-collapse ablations: median full-cache "
            f"coverage is {coverage['manual_drop_full_cache_coverage_median']:.6f} "
            "for manual conflict-drop and "
            f"{coverage['direct_drop_full_cache_coverage_median']:.6f} for direct conflict-drop."
        )
        lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = set().union(*(row.keys() for row in rows))
    preferred = ["item_id", "scene_name", "metric"]
    fields = [key for key in preferred if key in keys]
    fields.extend(sorted(keys - set(fields)))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_final_results(
    aggregate_metrics_json: Path,
    batch_execution_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    rows = json.loads(aggregate_metrics_json.read_text(encoding="utf-8"))
    batch = json.loads(batch_execution_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not isinstance(batch, dict):
        raise ValueError("unexpected final-batch JSON schema")
    per_scene = paired_rows(rows, batch)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "research_practice_final_guided_ransac_audit",
        "source_aggregate_metrics_json": str(aggregate_metrics_json),
        "source_aggregate_metrics_sha256": file_sha256(aggregate_metrics_json),
        "source_batch_execution_json": str(batch_execution_json),
        "source_batch_execution_sha256": file_sha256(batch_execution_json),
        "source_git_sha": str(batch.get("git_sha", "")),
        "frozen_thresholds": {
            "minimum_independent_scenes": MINIMUM_INDEPENDENT_SCENES,
            "minimum_median_f1_gain": MINIMUM_MEDIAN_F1_GAIN,
            "minimum_win_rate": MINIMUM_WIN_RATE,
            "minimum_coverage": MINIMUM_COVERAGE,
            "maximum_efficiency_runtime_ratio": MAXIMUM_EFFICIENCY_RUNTIME_RATIO,
            "minimum_efficiency_f1_delta": MINIMUM_EFFICIENCY_F1_DELTA,
        },
        "per_scene": per_scene,
        "gate": final_gate(per_scene),
        "metric_summary": metric_summary(per_scene),
        "coverage_diagnostics": coverage_diagnostics(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "final_method_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "final_method_per_scene.csv", per_scene)
    write_csv(output_dir / "final_method_summary.csv", result["metric_summary"])
    (output_dir / "final_method_audit.md").write_text(
        markdown_report(result), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        "Audit final eight-scene guided RANSAC metrics without GPU recomputation"
    )
    parser.add_argument("--aggregate_metrics_json", required=True)
    parser.add_argument("--batch_execution_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    result = audit_final_results(
        Path(args.aggregate_metrics_json),
        Path(args.batch_execution_json),
        Path(args.output_dir),
    )
    print(
        json.dumps(
            {
                "decision": result["gate"]["decision"],
                **result["gate"]["diagnostics"],
                "coverage_diagnostics": result["coverage_diagnostics"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
