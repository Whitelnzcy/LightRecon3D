"""Paired scene-level statistics for the B0--B5/O1/O2 component table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from analyze_guided_ransac_mechanism_ablation import holm_adjust, wilcoxon_pvalue
from audit_research_practice_final_results import (
    bootstrap_mean_ci,
    exact_two_sided_sign_pvalue,
)


SCHEMA_VERSION = 1
PAIRWISE_F1 = "support_partition_pairwise_f1"
METHODS = (
    "B0_global_ransac",
    "B1_stage1_direct_svd",
    "B2_stage2_local_refit",
    "B3_stage2_manual_merge",
    "B4_guided_ransac",
    "O1_gt_support_guided_seed0",
    "O2_gt_identity",
)
COMPARISONS = (
    ("B2_vs_B1_local_refit", "B1_stage1_direct_svd", "B2_stage2_local_refit"),
    ("B3_vs_B2_manual_merge", "B2_stage2_local_refit", "B3_stage2_manual_merge"),
    ("B4_vs_B0_guided_gain", "B0_global_ransac", "B4_guided_ransac"),
    ("B4_vs_B3_learned_vs_manual", "B3_stage2_manual_merge", "B4_guided_ransac"),
    ("O1_vs_B4_support_gap", "B4_guided_ransac", "O1_gt_support_guided_seed0"),
    ("O2_vs_O1_identity_gap", "O1_gt_support_guided_seed0", "O2_gt_identity"),
)


def read_index(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("aggregate metrics CSV is empty")
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        method = row["method"]
        if method not in METHODS:
            continue
        key = (row["item_id"], method)
        if key in indexed:
            raise ValueError(f"duplicate aggregate row: {key}")
        indexed[key] = row
    item_ids = sorted({item_id for item_id, _ in indexed})
    missing = [
        (item_id, method)
        for item_id in item_ids
        for method in METHODS
        if (item_id, method) not in indexed
    ]
    if missing:
        raise ValueError(f"incomplete component table; first missing row: {missing[0]}")
    return indexed


def method_summary(indexed: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    item_ids = sorted({item_id for item_id, _ in indexed})
    for method in METHODS:
        values = [float(indexed[(item_id, method)][PAIRWISE_F1]) for item_id in item_ids]
        runtimes = [float(indexed[(item_id, method)]["runtime_seconds"]) for item_id in item_ids]
        finite_runtimes = [value for value in runtimes if math.isfinite(value)]
        row: dict[str, Any] = {
            "method": method,
            "scenes": len(values),
            "mean_pairwise_f1": statistics.fmean(values),
            "median_pairwise_f1": statistics.median(values),
        }
        if finite_runtimes:
            row["mean_runtime_seconds"] = statistics.fmean(finite_runtimes)
            row["median_runtime_seconds"] = statistics.median(finite_runtimes)
        result.append(row)
    return result


def comparison_stats(
    indexed: dict[tuple[str, str], dict[str, str]],
    name: str,
    reference: str,
    method: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    pairs = []
    for item_id in sorted({candidate for candidate, _ in indexed}):
        reference_row = indexed[(item_id, reference)]
        method_row = indexed[(item_id, method)]
        reference_value = float(reference_row[PAIRWISE_F1])
        method_value = float(method_row[PAIRWISE_F1])
        if not math.isfinite(reference_value) or not math.isfinite(method_value):
            raise ValueError(f"non-finite Pairwise F1 for {item_id}/{name}")
        pairs.append(
            {
                "item_id": item_id,
                "scene_name": method_row["scene_name"],
                "reference_value": reference_value,
                "method_value": method_value,
                "delta": method_value - reference_value,
            }
        )
    deltas = [row["delta"] for row in pairs]
    wins = sum(value > 0.0 for value in deltas)
    losses = sum(value < 0.0 for value in deltas)
    ties = len(deltas) - wins - losses
    return {
        "comparison": name,
        "reference_method": reference,
        "method": method,
        "metric": PAIRWISE_F1,
        "pairs": len(pairs),
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_two_sided_p": exact_two_sided_sign_pvalue(wins, losses),
        "wilcoxon_two_sided_p": wilcoxon_pvalue(deltas),
        "bootstrap": bootstrap_mean_ci(deltas, samples=bootstrap_samples, seed=bootstrap_seed),
        "per_scene": pairs,
    }


def analyze(
    aggregate_metrics_csv: Path,
    output_dir: Path,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260722,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    indexed = read_index(aggregate_metrics_csv)
    comparisons = [
        comparison_stats(
            indexed,
            name,
            reference,
            method,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for name, reference, method in COMPARISONS
    ]
    for row, adjusted in zip(
        comparisons,
        holm_adjust([row["sign_test_two_sided_p"] for row in comparisons]),
        strict=True,
    ):
        row["sign_test_holm_p"] = adjusted
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "main_component_paired_statistics",
        "source_aggregate_metrics_csv": str(aggregate_metrics_csv.resolve()),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "b5_quality_alias": "B4_guided_ransac",
        "method_summary": method_summary(indexed),
        "comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "component_paired_statistics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    flat = []
    for row in comparisons:
        summary = {key: value for key, value in row.items() if key not in {"bootstrap", "per_scene"}}
        summary["bootstrap_ci95_low"] = row["bootstrap"]["ci95_low"]
        summary["bootstrap_ci95_high"] = row["bootstrap"]["ci95_high"]
        flat.append(summary)
    with (output_dir / "component_paired_statistics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = [
        "# Main component paired statistics",
        "",
        "Positive deltas favor the named method. Holm correction covers the six planned Pairwise-F1 contrasts.",
        "B5 uses B4 quality metrics and adds verified provenance, bounded-component, and structural-line contracts.",
        "",
        "| Contrast | Mean delta | Median delta | Bootstrap 95% CI | W/T/L | Sign p | Holm p | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        ci = row["bootstrap"]
        wilcoxon = row["wilcoxon_two_sided_p"]
        wilcoxon_text = "n/a" if wilcoxon is None else f"{wilcoxon:.6f}"
        lines.append(
            f"| {row['comparison']} | {row['mean_delta']:+.6f} | {row['median_delta']:+.6f} | "
            f"[{ci['ci95_low']:+.6f}, {ci['ci95_high']:+.6f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['sign_test_two_sided_p']:.6f} | {row['sign_test_holm_p']:.6f} | "
            f"{wilcoxon_text} |"
        )
    (output_dir / "component_paired_statistics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate_metrics_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260722)
    args = parser.parse_args()
    result = analyze(
        Path(args.aggregate_metrics_csv),
        Path(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"comparisons": len(result["comparisons"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
