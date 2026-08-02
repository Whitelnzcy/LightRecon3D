"""Leave-one-scene-out clustering gate for bias-aware fragment costs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


OUTPUT_SCHEMA_VERSION = 1


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


class LabelAwareUnionFind(UnionFind):
    def __init__(self, gt_labels: dict[int, int]):
        super().__init__(gt_labels)
        self.size = {node: 1 for node in gt_labels}
        self.label_counts = {
            node: Counter([label]) for node, label in gt_labels.items()
        }

    def union_with_pair_counts(self, first: int, second: int) -> tuple[int, int]:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return 0, 0
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        predicted_cross_pairs = self.size[first_root] * self.size[second_root]
        true_positive_cross_pairs = sum(
            count * self.label_counts[second_root].get(label, 0)
            for label, count in self.label_counts[first_root].items()
        )
        self.parent[second_root] = first_root
        self.size[first_root] += self.size.pop(second_root)
        self.label_counts[first_root].update(
            self.label_counts.pop(second_root)
        )
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1
        return true_positive_cross_pairs, predicted_cross_pairs


def _node_gt_labels(rows: list[dict]) -> dict[int, int]:
    gt_labels = {}
    for row in rows:
        gt_labels[int(row["first_fragment_id"])] = int(row["first_gt_plane"])
        gt_labels[int(row["second_fragment_id"])] = int(row["second_gt_plane"])
    return gt_labels


def scene_partition_metrics(rows: list[dict], cost_key: str, threshold: float) -> dict:
    gt_labels = _node_gt_labels(rows)
    nodes = set(gt_labels)
    union_find = UnionFind(nodes)
    for row in rows:
        if float(row[cost_key]) <= float(threshold):
            union_find.union(
                int(row["first_fragment_id"]),
                int(row["second_fragment_id"]),
            )
    nodes = sorted(nodes)
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for index, first in enumerate(nodes):
        for second in nodes[index + 1 :]:
            truth = gt_labels[first] == gt_labels[second]
            prediction = union_find.find(first) == union_find.find(second)
            if truth and prediction:
                true_positive += 1
            elif prediction:
                false_positive += 1
            elif truth:
                false_negative += 1
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "nodes": len(nodes),
        "true_positive_pairs": true_positive,
        "false_positive_pairs": false_positive,
        "false_negative_pairs": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "component_count": len({union_find.find(node) for node in nodes}),
    }


def scene_partition_f1_curve(
    rows: list[dict],
    cost_key: str,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Evaluate all thresholds with one incremental connected-component pass."""
    gt_labels = _node_gt_labels(rows)
    union_find = LabelAwareUnionFind(gt_labels)
    label_totals = Counter(gt_labels.values())
    true_pairs = sum(count * (count - 1) // 2 for count in label_totals.values())
    predicted_pairs = 0
    true_positive_pairs = 0
    sorted_rows = sorted(rows, key=lambda row: float(row[cost_key]))
    edge_index = 0
    f1_values = []
    for threshold in thresholds:
        while (
            edge_index < len(sorted_rows)
            and float(sorted_rows[edge_index][cost_key]) <= float(threshold)
        ):
            row = sorted_rows[edge_index]
            true_delta, predicted_delta = union_find.union_with_pair_counts(
                int(row["first_fragment_id"]),
                int(row["second_fragment_id"]),
            )
            true_positive_pairs += true_delta
            predicted_pairs += predicted_delta
            edge_index += 1
        precision = true_positive_pairs / max(predicted_pairs, 1)
        recall = true_positive_pairs / max(true_pairs, 1)
        f1_values.append(
            2.0 * precision * recall / max(precision + recall, 1e-12)
        )
    return np.asarray(f1_values, dtype=np.float64)


def candidate_thresholds(rows: list[dict], cost_key: str, count: int) -> np.ndarray:
    values = np.asarray([row[cost_key] for row in rows], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, max(int(count), 2))
    thresholds = np.unique(np.quantile(values, quantiles))
    return np.concatenate(([float(values.min()) - 1e-12], thresholds))


def tune_threshold(
    rows_by_scene: dict[str, list[dict]],
    training_scenes: list[str],
    cost_key: str,
    threshold_count: int,
) -> tuple[float, float]:
    training_rows = [
        row for scene in training_scenes for row in rows_by_scene[scene]
    ]
    thresholds = candidate_thresholds(
        training_rows, cost_key, threshold_count
    )
    mean_f1 = np.mean(
        np.stack(
            [
                scene_partition_f1_curve(
                    rows_by_scene[scene], cost_key, thresholds
                )
                for scene in training_scenes
            ],
            axis=0,
        ),
        axis=0,
    )
    best_index = int(np.argmax(mean_f1))
    return float(thresholds[best_index]), float(mean_f1[best_index])


def leave_one_scene_out(
    rows_by_scene: dict[str, list[dict]],
    cost_key: str,
    threshold_count: int,
) -> dict:
    scenes = sorted(rows_by_scene)
    heldout_rows = []
    for heldout in scenes:
        training = [scene for scene in scenes if scene != heldout]
        threshold, training_f1 = tune_threshold(
            rows_by_scene, training, cost_key, threshold_count)
        metrics = scene_partition_metrics(
            rows_by_scene[heldout], cost_key, threshold)
        heldout_rows.append(
            {
                "scene": heldout,
                "threshold": threshold,
                "training_scene_mean_f1": training_f1,
                **metrics,
            }
        )
    return {
        "cost_key": cost_key,
        "scene_count": len(heldout_rows),
        "mean_precision": float(np.mean([row["precision"] for row in heldout_rows])),
        "mean_recall": float(np.mean([row["recall"] for row in heldout_rows])),
        "mean_f1": float(np.mean([row["f1"] for row in heldout_rows])),
        "median_f1": float(np.median([row["f1"] for row in heldout_rows])),
        "heldout_scenes": heldout_rows,
    }


def add_costs(rows: list[dict], penalty: float) -> None:
    for row in rows:
        row["pooled_cost"] = float(row["pooled_rmse_normalized"])
        row["bias_corrected_cost"] = float(row["bias_corrected_rmse_normalized"])
        row["bias_penalized_cost"] = (
            float(row["bias_corrected_rmse_normalized"])
            + float(penalty) * float(row["required_bias_std_normalized"])
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leave-one-scene-out connected-component clustering on fragment costs."
    )
    parser.add_argument("--pair_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bias_penalty", type=float, default=2.0)
    parser.add_argument("--threshold_count", type=int, default=128)
    args = parser.parse_args()

    pair_path = Path(args.pair_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    rows = json.loads(pair_path.read_text(encoding="utf-8"))
    add_costs(rows, args.bias_penalty)
    rows_by_scene = {
        scene: [row for row in rows if row["scene"] == scene]
        for scene in sorted({row["scene"] for row in rows})
    }
    methods = {
        key: leave_one_scene_out(rows_by_scene, key, args.threshold_count)
        for key in ("pooled_cost", "bias_corrected_cost", "bias_penalized_cost")
    }
    pooled = {
        row["scene"]: row
        for row in methods["pooled_cost"]["heldout_scenes"]
    }
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "selection": "leave-one-scene-out global threshold",
        "clustering": "connected_components",
        "methods": methods,
        "paired_vs_pooled": {},
        "config": {
            "pair_json": str(pair_path),
            "bias_penalty": float(args.bias_penalty),
            "threshold_count": int(args.threshold_count),
        },
    }
    for name in ("bias_corrected_cost", "bias_penalized_cost"):
        rows_method = methods[name]["heldout_scenes"]
        deltas = np.asarray(
            [row["f1"] - pooled[row["scene"]]["f1"] for row in rows_method],
            dtype=np.float64,
        )
        result["paired_vs_pooled"][name] = {
            "mean_f1_delta": float(np.mean(deltas)),
            "median_f1_delta": float(np.median(deltas)),
            "win_tie_loss": [
                int(np.sum(deltas > 1e-12)),
                int(np.sum(np.abs(deltas) <= 1e-12)),
                int(np.sum(deltas < -1e-12)),
            ],
        }
    result["limitations"] = [
        "GT labels selected the high-purity fragment audit population.",
        "Connected components is a simple transitive clustering baseline, not a learned partitioner.",
        "The threshold is tuned only on the other 16 development scenes for each held-out scene.",
    ]
    result["outputs"] = {"summary_json": str(output_dir / "summary.json")}
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
