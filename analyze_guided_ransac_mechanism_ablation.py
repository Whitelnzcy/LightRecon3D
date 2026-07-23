"""Paired scene-level statistics for the guided-RANSAC mechanism ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from audit_research_practice_final_results import (
    bootstrap_mean_ci,
    exact_two_sided_sign_pvalue,
)


SCHEMA_VERSION = 2
PAIRWISE_F1 = "support_partition_pairwise_f1"
COMPARISONS = (
    ("proposal_vs_none", "support_mechanism_none_cc", "support_mechanism_proposal_only_cc"),
    ("consensus_vs_none", "support_mechanism_none_cc", "support_mechanism_consensus_only_cc"),
    ("refit_vs_none", "support_mechanism_none_cc", "support_mechanism_refit_only_cc"),
    ("consensus_after_proposal", "support_mechanism_proposal_only_cc", "learning_guided_ransac_cc"),
    ("refit_after_proposal_consensus", "learning_guided_ransac_cc", "support_mechanism_proposal_consensus_refit_cc"),
    ("all_vs_none", "support_mechanism_none_cc", "support_mechanism_proposal_consensus_refit_cc"),
)
PLANNED_METHODS = tuple(
    sorted({method for _, reference, method in COMPARISONS for method in (reference, method)})
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("aggregate metrics CSV is empty")
    return rows


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, str]]:
    indexed: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rows:
        if row["method"] not in PLANNED_METHODS:
            continue
        key = (row["item_id"], int(row["seed"]), row["method"])
        if key in indexed:
            raise ValueError(f"duplicate aggregate row: {key}")
        indexed[key] = row
    return indexed


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * pvalues[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def wilcoxon_pvalue(deltas: list[float]) -> float | None:
    nonzero = [value for value in deltas if value != 0.0]
    if len(nonzero) < 5:
        return None
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    result = wilcoxon(nonzero, alternative="two-sided", method="auto")
    return float(result.pvalue)


def comparison_stats(
    indexed: dict[tuple[str, int, str], dict[str, str]],
    name: str,
    reference: str,
    method: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    item_ids = sorted({item_id for item_id, _, _ in indexed})
    expected_seeds: tuple[int, ...] | None = None
    for item_id in item_ids:
        reference_seeds = {
            seed for candidate, seed, candidate_method in indexed
            if candidate == item_id and candidate_method == reference
        }
        method_seeds = {
            seed for candidate, seed, candidate_method in indexed
            if candidate == item_id and candidate_method == method
        }
        if not reference_seeds or reference_seeds != method_seeds:
            raise ValueError(f"missing or mismatched seeds for {item_id}/{name}")
        seeds = tuple(sorted(reference_seeds))
        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise ValueError(f"inconsistent seed set for {item_id}/{name}: {seeds}")
        reference_values = [float(indexed[(item_id, seed, reference)][PAIRWISE_F1]) for seed in seeds]
        method_values = [float(indexed[(item_id, seed, method)][PAIRWISE_F1]) for seed in seeds]
        if not all(math.isfinite(value) for value in reference_values + method_values):
            raise ValueError(f"non-finite F1 for {item_id}/{name}")
        reference_value = statistics.fmean(reference_values)
        method_value = statistics.fmean(method_values)
        method_row = indexed[(item_id, seeds[0], method)]
        pairs.append(
            {
                "item_id": item_id,
                "scene_name": method_row["scene_name"],
                "seeds": list(seeds),
                "seed_count": len(seeds),
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
        "seeds": list(expected_seeds or ()),
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_two_sided_p": exact_two_sided_sign_pvalue(wins, losses),
        "wilcoxon_two_sided_p": wilcoxon_pvalue(deltas),
        "bootstrap": bootstrap_mean_ci(
            deltas, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "per_scene": pairs,
    }


def mode_summary(
    indexed: dict[tuple[str, int, str], dict[str, str]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in PLANNED_METHODS:
        item_ids = sorted({item_id for item_id, _, candidate in indexed if candidate == method})
        scene_f1: list[float] = []
        scene_runtime: list[float] = []
        seeds: set[int] = set()
        for item_id in item_ids:
            rows = []
            for (candidate, seed, candidate_method), row in indexed.items():
                if candidate == item_id and candidate_method == method:
                    rows.append(row)
                    seeds.add(seed)
            scene_f1.append(statistics.fmean(float(row[PAIRWISE_F1]) for row in rows))
            scene_runtime.append(statistics.fmean(float(row["runtime_seconds"]) for row in rows))
        summaries.append(
            {
                "method": method,
                "scenes": len(item_ids),
                "seeds": sorted(seeds),
                "mean_pairwise_f1": statistics.fmean(scene_f1),
                "median_scene_mean_pairwise_f1": statistics.median(scene_f1),
                "mean_runtime_seconds": statistics.fmean(scene_runtime),
            }
        )
    return summaries


def analyze(
    aggregate_metrics_csv: Path | list[Path] | tuple[Path, ...],
    output_dir: Path,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260719,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    paths = [aggregate_metrics_csv] if isinstance(aggregate_metrics_csv, Path) else list(aggregate_metrics_csv)
    if not paths:
        raise ValueError("at least one aggregate metrics CSV is required")
    indexed = index_rows([row for path in paths for row in read_rows(path)])
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
    adjusted = holm_adjust([row["sign_test_two_sided_p"] for row in comparisons])
    for row, pvalue in zip(comparisons, adjusted, strict=True):
        row["sign_test_holm_p"] = pvalue
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "guided_ransac_mechanism_paired_statistics",
        "source_aggregate_metrics_csv": [str(path.resolve()) for path in paths],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "mode_summary": mode_summary(indexed),
        "comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "mechanism_paired_statistics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    summary_rows = [{key: value for key, value in row.items() if key not in {"per_scene", "bootstrap"}} | {
        "bootstrap_ci95_low": row["bootstrap"]["ci95_low"],
        "bootstrap_ci95_high": row["bootstrap"]["ci95_high"],
    } for row in comparisons]
    with (output_dir / "mechanism_paired_statistics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    lines = [
        "# Guided-RANSAC mechanism paired statistics",
        "",
        "Positive deltas favor the method. Holm correction covers the six planned Pairwise-F1 mechanism contrasts.",
        "",
        "| Contrast | Mean delta | Median delta | Bootstrap 95% CI | W/T/L | Sign p | Holm p | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        ci = row["bootstrap"]
        wilcoxon = row["wilcoxon_two_sided_p"]
        lines.append(
            f"| {row['comparison']} | {row['mean_delta']:+.6f} | "
            f"{row['median_delta']:+.6f} | [{ci['ci95_low']:+.6f}, "
            f"{ci['ci95_high']:+.6f}] | {row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['sign_test_two_sided_p']:.6f} | {row['sign_test_holm_p']:.6f} | "
            f"{wilcoxon:.6f} |" if wilcoxon is not None else
            f"| {row['comparison']} | {row['mean_delta']:+.6f} | "
            f"{row['median_delta']:+.6f} | [{ci['ci95_low']:+.6f}, "
            f"{ci['ci95_high']:+.6f}] | {row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['sign_test_two_sided_p']:.6f} | {row['sign_test_holm_p']:.6f} | n/a |"
        )
    (output_dir / "mechanism_paired_statistics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate_metrics_csv", required=True, action="append")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260719)
    args = parser.parse_args()
    result = analyze(
        [Path(path) for path in args.aggregate_metrics_csv],
        Path(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"comparisons": len(result["comparisons"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
