import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HIGHER_IS_BETTER = (
    "support_f1",
    "support_precision",
    "support_recall",
    "matched_mask_iou",
)
LOWER_IS_BETTER = ("plane_count_abs_error",)


def parse_args():
    parser = argparse.ArgumentParser("Summarize paired Stage1 feature ablation runs")
    parser.add_argument("--dust3r_dirs", nargs="+", required=True)
    parser.add_argument("--rgb_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260802)
    return parser.parse_args()


def load_run(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "per_record_metrics.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    return summary, rows


def numeric(row, key):
    return float(row[key])


def cluster_bootstrap(scene_deltas, samples, seed):
    scene_names = sorted(scene_deltas)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw_id in range(samples):
        selected = rng.choice(scene_names, size=len(scene_names), replace=True)
        values = [value for scene in selected for value in scene_deltas[scene]]
        draws[draw_id] = float(np.mean(values))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main():
    args = parse_args()
    if len(args.dust3r_dirs) != len(args.rgb_dirs):
        raise ValueError("dust3r_dirs and rgb_dirs must contain the same number of seeds")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = {"dust3r": {}, "rgb": {}}
    for expected, directories in (("dust3r", args.dust3r_dirs), ("rgb", args.rgb_dirs)):
        for directory in directories:
            summary, rows = load_run(directory)
            if summary["condition"] != expected:
                raise ValueError(f"{directory} is {summary['condition']}, expected {expected}")
            seed = int(summary["seed"])
            if seed in runs[expected]:
                raise ValueError(f"Duplicate {expected} seed: {seed}")
            runs[expected][seed] = {"summary": summary, "rows": rows, "path": directory}

    seeds = sorted(runs["dust3r"])
    if seeds != sorted(runs["rgb"]):
        raise ValueError("Conditions do not contain the same seeds")

    paired_rows = []
    for seed in seeds:
        dust_run = runs["dust3r"][seed]
        rgb_run = runs["rgb"][seed]
        if (
            dust_run["summary"]["val_records_csv_sha256"]
            != rgb_run["summary"]["val_records_csv_sha256"]
        ):
            raise ValueError(f"Validation manifest mismatch for seed {seed}")
        if (
            dust_run["summary"]["trainable_parameters"]
            != rgb_run["summary"]["trainable_parameters"]
        ):
            raise ValueError(f"Trainable parameter mismatch for seed {seed}")
        dust_rows = {int(row["sample_idx"]): row for row in dust_run["rows"]}
        rgb_rows = {int(row["sample_idx"]): row for row in rgb_run["rows"]}
        if set(dust_rows) != set(rgb_rows):
            raise ValueError(f"Per-record sample mismatch for seed {seed}")
        for sample_idx in sorted(dust_rows):
            dust = dust_rows[sample_idx]
            rgb = rgb_rows[sample_idx]
            if dust.get("scene_name", "") != rgb.get("scene_name", ""):
                raise ValueError(f"Scene mismatch for seed={seed}, sample={sample_idx}")
            row = {
                "seed": seed,
                "sample_idx": sample_idx,
                "scene_name": dust.get("scene_name", ""),
            }
            for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
                row[f"dust3r_{metric}"] = numeric(dust, metric)
                row[f"rgb_{metric}"] = numeric(rgb, metric)
                row[f"delta_{metric}"] = numeric(dust, metric) - numeric(rgb, metric)
            paired_rows.append(row)

    with (output_dir / "paired_per_record.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    metrics = {}
    for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        deltas = [row[f"delta_{metric}"] for row in paired_rows]
        scene_deltas = defaultdict(list)
        for row in paired_rows:
            scene_deltas[row["scene_name"]].append(row[f"delta_{metric}"])
        per_scene = {
            scene: float(np.mean(values)) for scene, values in sorted(scene_deltas.items())
        }
        metrics[metric] = {
            "dust3r_mean": float(np.mean([row[f"dust3r_{metric}"] for row in paired_rows])),
            "rgb_mean": float(np.mean([row[f"rgb_{metric}"] for row in paired_rows])),
            "dust3r_minus_rgb": float(np.mean(deltas)),
            "scene_cluster_bootstrap_95ci": cluster_bootstrap(
                scene_deltas,
                args.bootstrap_samples,
                args.bootstrap_seed,
            ),
            "positive_scenes": int(sum(value > 0 for value in per_scene.values())),
            "scene_count": len(per_scene),
            "per_scene_delta": per_scene,
        }

    primary = metrics["support_f1"]
    iou = metrics["matched_mask_iou"]
    if (
        primary["dust3r_minus_rgb"] >= 0.03
        and primary["scene_cluster_bootstrap_95ci"][0] > 0.0
        and iou["dust3r_minus_rgb"] >= 0.02
        and primary["positive_scenes"] >= 6
    ):
        verdict = "Strong_DUSt3R_Feature_Advantage"
    elif primary["dust3r_minus_rgb"] > 0.0:
        verdict = "Weak_or_Inconclusive_DUSt3R_Feature_Advantage"
    else:
        verdict = "No_DUSt3R_Feature_Advantage"

    summary = {
        "seeds": seeds,
        "paired_records": len(paired_rows),
        "trainable_parameters": runs["dust3r"][seeds[0]]["summary"][
            "trainable_parameters"
        ],
        "verdict": verdict,
        "metrics": metrics,
        "rules": {
            "primary_metric": "support_f1",
            "strong_min_support_f1_delta": 0.03,
            "strong_min_matched_mask_iou_delta": 0.02,
            "strong_min_positive_scenes": 6,
            "strong_requires_positive_scene_cluster_bootstrap_lower_bound": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        "# Stage1 feature ablation summary\n\n"
        f"Verdict: {verdict}\n\n"
        "| Metric | RGB | DUSt3R | Delta | 95% scene-cluster CI |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {metric} | {values['rgb_mean']:.6f} | {values['dust3r_mean']:.6f} | "
            f"{values['dust3r_minus_rgb']:+.6f} | "
            f"[{values['scene_cluster_bootstrap_95ci'][0]:+.6f}, "
            f"{values['scene_cluster_bootstrap_95ci'][1]:+.6f}] |"
            for metric, values in metrics.items()
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
