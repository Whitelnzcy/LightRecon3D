"""Apply the pre-registered identity/support gate to batch metric rows."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from research_practice_batch import file_sha256


SCHEMA_VERSION = 1
FAMILY = "support_records"
RANSAC = "global_ransac_cc"
MANUAL_RAW = "stage2_manual_merge_support"
MANUAL_DROP = "stage2_manual_merge_unique_conflict_drop"
DIRECT_RAW = "stage2_support_direct_global_svd"
DIRECT_DROP = "stage2_direct_svd_unique_conflict_drop"
REQUIRED_METHODS = (RANSAC, MANUAL_RAW, MANUAL_DROP, DIRECT_RAW, DIRECT_DROP)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot take a median of an empty list")
    return float(statistics.median(values))


def metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{row.get('item_id')} {row.get('method')} missing numeric {key}")
    return float(value)


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("metric_family") != FAMILY:
            continue
        item_id = str(row.get("item_id", ""))
        method = str(row.get("method", ""))
        if not item_id or not method:
            raise ValueError("support-record metric rows require item_id and method")
        if method in indexed.setdefault(item_id, {}):
            raise ValueError(f"duplicate metric row for {item_id}/{method}")
        indexed[item_id][method] = row
    if not indexed:
        raise ValueError("no support_records metric rows found")
    for item_id, methods in indexed.items():
        missing = [method for method in REQUIRED_METHODS if method not in methods]
        if missing:
            raise ValueError(f"{item_id} missing methods: {missing}")
    return indexed


def gate_row(item_id: str, methods: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ransac = methods[RANSAC]
    manual = methods[MANUAL_RAW]
    manual_drop = methods[MANUAL_DROP]
    direct = methods[DIRECT_RAW]
    direct_drop = methods[DIRECT_DROP]
    manual_f1 = metric(manual, "support_partition_pairwise_f1")
    ransac_f1 = metric(ransac, "support_partition_pairwise_f1")
    return {
        "item_id": item_id,
        "scene_name": str(manual.get("scene_name", "")),
        "ransac_f1": ransac_f1,
        "manual_raw_f1": manual_f1,
        "manual_raw_delta_vs_ransac": manual_f1 - ransac_f1,
        "manual_raw_win": manual_f1 > ransac_f1,
        "ransac_gt_coverage": metric(ransac, "gt_labeled_record_coverage"),
        "manual_raw_gt_coverage": metric(manual, "gt_labeled_record_coverage"),
        "manual_raw_assignment_rate": metric(manual, "record_assignment_rate"),
        "ransac_plane_precision": metric(ransac, "support_conditioned_plane_precision"),
        "manual_raw_plane_precision": metric(manual, "support_conditioned_plane_precision"),
        "ransac_plane_recall_observed": metric(
            ransac, "support_conditioned_plane_recall_observed"
        ),
        "manual_raw_plane_recall_observed": metric(
            manual, "support_conditioned_plane_recall_observed"
        ),
        "ransac_fragmentation_excess": metric(
            ransac, "support_conditioned_fragmentation_excess"
        ),
        "manual_raw_fragmentation_excess": metric(
            manual, "support_conditioned_fragmentation_excess"
        ),
        "ransac_overmerge_excess": metric(
            ransac, "support_conditioned_overmerge_excess"
        ),
        "manual_raw_overmerge_excess": metric(
            manual, "support_conditioned_overmerge_excess"
        ),
        "manual_raw_conflicting_keys": metric(
            manual, "conflicting_positive_key_count"
        ),
        "manual_raw_conflicting_records": metric(
            manual, "conflicting_positive_record_count"
        ),
        "manual_drop_f1": metric(
            manual_drop, "support_partition_pairwise_f1"
        ),
        "manual_drop_gt_coverage": metric(
            manual_drop, "gt_labeled_record_coverage"
        ),
        "manual_drop_assignment_rate": metric(manual_drop, "record_assignment_rate"),
        "direct_raw_f1": metric(direct, "support_partition_pairwise_f1"),
        "direct_raw_gt_coverage": metric(direct, "gt_labeled_record_coverage"),
        "direct_drop_f1": metric(direct_drop, "support_partition_pairwise_f1"),
        "direct_drop_gt_coverage": metric(
            direct_drop, "gt_labeled_record_coverage"
        ),
    }


def gate_check(name: str, value: float, threshold: float, passed: bool, rule: str) -> dict[str, Any]:
    return {
        "name": name,
        "value": float(value),
        "threshold": float(threshold),
        "passed": bool(passed),
        "rule": rule,
    }


def audit_rows(
    rows: list[dict[str, Any]],
    *,
    minimum_independent_scenes: int = 8,
    minimum_median_f1_gain: float = 0.05,
    minimum_win_rate: float = 0.70,
    minimum_gt_coverage: float = 0.90,
) -> dict[str, Any]:
    indexed = index_rows(rows)
    per_group = [gate_row(item_id, indexed[item_id]) for item_id in sorted(indexed)]
    group_count = len(per_group)
    scene_count = len({row["scene_name"] for row in per_group if row["scene_name"]})
    f1_deltas = [row["manual_raw_delta_vs_ransac"] for row in per_group]
    win_rate = sum(row["manual_raw_win"] for row in per_group) / group_count
    manual_coverage = median([row["manual_raw_gt_coverage"] for row in per_group])
    manual_assignment = median([row["manual_raw_assignment_rate"] for row in per_group])
    manual_overmerge = median([row["manual_raw_overmerge_excess"] for row in per_group])
    ransac_overmerge = median([row["ransac_overmerge_excess"] for row in per_group])
    manual_fragmentation = median(
        [row["manual_raw_fragmentation_excess"] for row in per_group]
    )
    ransac_fragmentation = median(
        [row["ransac_fragmentation_excess"] for row in per_group]
    )
    median_gain = median(f1_deltas)
    checks = [
        gate_check(
            "median_manual_raw_f1_gain_vs_ransac",
            median_gain,
            minimum_median_f1_gain,
            median_gain >= minimum_median_f1_gain,
            ">=",
        ),
        gate_check(
            "manual_raw_group_win_rate",
            win_rate,
            minimum_win_rate,
            win_rate >= minimum_win_rate,
            ">=",
        ),
        gate_check(
            "manual_raw_median_gt_coverage",
            manual_coverage,
            minimum_gt_coverage,
            manual_coverage >= minimum_gt_coverage,
            ">=",
        ),
        gate_check(
            "manual_raw_median_assignment_rate",
            manual_assignment,
            minimum_gt_coverage,
            manual_assignment >= minimum_gt_coverage,
            ">=",
        ),
        gate_check(
            "manual_raw_median_overmerge_not_worse_than_ransac",
            manual_overmerge,
            ransac_overmerge,
            manual_overmerge <= ransac_overmerge,
            "<=",
        ),
    ]
    method_gate_passed = all(check["passed"] for check in checks)
    scene_count_passed = scene_count >= minimum_independent_scenes
    if not method_gate_passed:
        decision = "stop_identity_method_promotion_use_strongest_baseline"
    elif not scene_count_passed:
        decision = "smoke_signal_only_expand_independent_scenes"
    else:
        decision = "identity_signal_passed_final_gate"

    manual_drop_coverage = median(
        [row["manual_drop_gt_coverage"] for row in per_group]
    )
    direct_drop_coverage = median(
        [row["direct_drop_gt_coverage"] for row in per_group]
    )
    diagnostics = {
        "groups": group_count,
        "independent_scenes": scene_count,
        "minimum_independent_scenes": int(minimum_independent_scenes),
        "scene_count_passed": scene_count_passed,
        "method_gate_passed": method_gate_passed,
        "median_manual_raw_f1": median([row["manual_raw_f1"] for row in per_group]),
        "median_ransac_f1": median([row["ransac_f1"] for row in per_group]),
        "median_manual_raw_f1_gain_vs_ransac": median_gain,
        "manual_raw_group_wins": int(sum(row["manual_raw_win"] for row in per_group)),
        "manual_raw_group_win_rate": float(win_rate),
        "median_manual_raw_gt_coverage": manual_coverage,
        "median_manual_raw_assignment_rate": manual_assignment,
        "median_manual_raw_plane_precision": median(
            [row["manual_raw_plane_precision"] for row in per_group]
        ),
        "median_ransac_plane_precision": median(
            [row["ransac_plane_precision"] for row in per_group]
        ),
        "median_manual_raw_plane_recall_observed": median(
            [row["manual_raw_plane_recall_observed"] for row in per_group]
        ),
        "median_ransac_plane_recall_observed": median(
            [row["ransac_plane_recall_observed"] for row in per_group]
        ),
        "median_manual_raw_fragmentation_excess": manual_fragmentation,
        "median_ransac_fragmentation_excess": ransac_fragmentation,
        "median_manual_raw_overmerge_excess": manual_overmerge,
        "median_ransac_overmerge_excess": ransac_overmerge,
        "median_manual_raw_conflicting_keys": median(
            [row["manual_raw_conflicting_keys"] for row in per_group]
        ),
        "median_manual_raw_conflicting_records": median(
            [row["manual_raw_conflicting_records"] for row in per_group]
        ),
        "median_manual_drop_f1": median(
            [row["manual_drop_f1"] for row in per_group]
        ),
        "median_manual_drop_gt_coverage": manual_drop_coverage,
        "manual_drop_coverage_collapse": manual_drop_coverage < minimum_gt_coverage,
        "median_direct_raw_f1": median([row["direct_raw_f1"] for row in per_group]),
        "median_direct_drop_f1": median(
            [row["direct_drop_f1"] for row in per_group]
        ),
        "median_direct_drop_gt_coverage": direct_drop_coverage,
        "direct_drop_coverage_collapse": direct_drop_coverage < minimum_gt_coverage,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "checks": checks,
        "diagnostics": diagnostics,
        "per_group": per_group,
    }


def markdown_report(result: dict[str, Any]) -> str:
    diagnostics = result["diagnostics"]
    lines = [
        "# Research-practice smoke gate",
        "",
        f"Decision: `{result['decision']}`",
        "",
        (
            f"Evaluated {diagnostics['groups']} view groups from "
            f"{diagnostics['independent_scenes']} independent scene IDs."
        ),
        "",
        "| Check | Value | Threshold | Pass |",
        "|---|---:|---:|---:|",
    ]
    for check in result["checks"]:
        lines.append(
            f"| {check['name']} | {check['value']:.6f} | "
            f"{check['rule']} {check['threshold']:.6f} | {check['passed']} |"
        )
    lines.extend(
        [
            "",
            "| Group | Scene | RANSAC F1 | Manual raw F1 | Delta | Manual coverage |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["per_group"]:
        lines.append(
            f"| {row['item_id']} | {row['scene_name']} | {row['ransac_f1']:.6f} | "
            f"{row['manual_raw_f1']:.6f} | {row['manual_raw_delta_vs_ransac']:.6f} | "
            f"{row['manual_raw_gt_coverage']:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "Conflict-drop is not a primary result. Manual conflict-drop median "
                f"GT coverage is {diagnostics['median_manual_drop_gt_coverage']:.6f}; "
                "direct conflict-drop median GT coverage is "
                f"{diagnostics['median_direct_drop_gt_coverage']:.6f}."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    aggregate_metrics_json: Path,
    batch_execution_json: Path,
    output_dir: Path,
    *,
    minimum_independent_scenes: int = 8,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    rows = json.loads(aggregate_metrics_json.read_text(encoding="utf-8"))
    batch = json.loads(batch_execution_json.read_text(encoding="utf-8"))
    failed = [item["id"] for item in batch.get("items", []) if item.get("status") != "pass"]
    if failed:
        raise ValueError(f"batch contains failed items: {failed}")
    result = audit_rows(rows, minimum_independent_scenes=minimum_independent_scenes)
    result.update(
        {
            "source_aggregate_metrics_json": str(aggregate_metrics_json),
            "source_aggregate_metrics_sha256": file_sha256(aggregate_metrics_json),
            "source_batch_execution_json": str(batch_execution_json),
            "source_batch_execution_sha256": file_sha256(batch_execution_json),
            "source_git_sha": str(batch.get("git_sha", "")),
            "batch_summary": batch.get("summary", {}),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "smoke_gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "smoke_gate_per_group.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["per_group"][0]))
        writer.writeheader()
        writer.writerows(result["per_group"])
    (output_dir / "smoke_gate.md").write_text(
        markdown_report(result), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser("Audit the research-practice batch identity gate")
    parser.add_argument("--aggregate_metrics_json", required=True)
    parser.add_argument("--batch_execution_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--minimum_independent_scenes", type=int, default=8)
    args = parser.parse_args()
    result = run_audit(
        Path(args.aggregate_metrics_json),
        Path(args.batch_execution_json),
        Path(args.output_dir),
        minimum_independent_scenes=args.minimum_independent_scenes,
    )
    print(json.dumps({"decision": result["decision"], **result["diagnostics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
