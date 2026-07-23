import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from execute_guided_ransac_mechanism_ablation import (
    mechanism_command,
    load_reusable_mode_stages,
    paired_summary,
    prediction_semantic_equivalence,
    summarize_metrics,
)


class MechanismAblationExecutorTests(unittest.TestCase):
    def test_reuse_ledger_verifies_prediction_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "prediction.npz"
            np.savez(prediction, value=np.asarray([1], np.int32))
            sha256 = hashlib.sha256(prediction.read_bytes()).hexdigest()
            ledger = root / "execution.json"
            ledger.write_text(
                json.dumps(
                    {
                        "kind": "guided_ransac_mechanism_ablation",
                        "source_batch_sha256": "batch",
                        "support_refit_weight": 1.0,
                        "modes": ["none"],
                        "seeds": [0],
                        "items": [
                            {
                                "id": "item",
                                "seed": 0,
                                "modes": [
                                    {
                                        "mode": "none",
                                        "status": "pass",
                                        "prediction": str(prediction),
                                        "prediction_sha256": sha256,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            reusable, sources = load_reusable_mode_stages(
                [ledger],
                source_batch_sha256="batch",
                modes=["none"],
                seeds=[0],
                support_refit_weight=1.0,
            )
            self.assertIn(("item", 0, "none"), reusable)
            self.assertTrue(reusable[("item", 0, "none")]["reused"])
            self.assertEqual(len(sources), 1)

    def test_mechanism_command_reuses_frozen_batch_config(self):
        frozen = {
            "evaluation_min_conf": 1.0,
            "ransac": {
                "distance_threshold": 0.03,
                "iterations": 300,
                "min_inliers": 2000,
                "cluster_radius": 0.08,
                "min_component_points": 1000,
                "max_planes": 32,
                "hypothesis_max_points": 50000,
                "component_exact_max_points": 20000,
            },
            "guided_ransac": {
                "proposal_iterations": 64,
                "proposal_min_points": 64,
                "proposal_min_inlier_ratio": 0.6,
                "proposal_max_points": 4000,
                "support_score_weight": 1.0,
                "fallback_iterations": 96,
            },
        }
        command = mechanism_command(
            Path("python"),
            Path("project"),
            Path("cache.npz"),
            Path("support.npz"),
            Path("output"),
            "scene",
            "all",
            2,
            frozen,
            1.5,
        )
        self.assertEqual(command[command.index("--mechanism_mode") + 1], "all")
        self.assertEqual(
            command[command.index("--global_proposal_iterations") + 1], "300"
        )
        self.assertEqual(command[command.index("--seed") + 1], "2")
        self.assertEqual(
            command[command.index("--support_refit_weight") + 1], "1.5"
        )

    def test_prediction_equivalence_checks_semantics_not_compression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = {
                "points": np.asarray([[0, 0, 0], [1, 0, 0]], np.float32),
                "point_plane_ids": np.asarray([0, -1], np.int32),
                "source_views": np.asarray([0, 1], np.int32),
                "pixel_xy": np.asarray([[0, 0], [1, 0]], np.int32),
                "plane_normals": np.asarray([[0, 0, 1]], np.float32),
                "plane_offsets": np.asarray([0], np.float32),
                "plane_inlier_counts": np.asarray([1], np.int32),
            }
            first = root / "first.npz"
            second = root / "second.npz"
            np.savez(first, **fields)
            np.savez_compressed(second, **fields)
            result = prediction_semantic_equivalence(first, second)
            self.assertTrue(result["equivalent"])
            self.assertNotEqual(result["current_sha256"], result["historical_sha256"])

    def test_metric_and_paired_summaries_use_scene_rows(self):
        rows = [
            {
                "item_id": "a",
                "method": "base",
                "support_partition_pairwise_f1": 0.4,
                "support_coverage": 1.0,
                "runtime_seconds": 2.0,
            },
            {
                "item_id": "b",
                "method": "base",
                "support_partition_pairwise_f1": 0.6,
                "support_coverage": 1.0,
                "runtime_seconds": 4.0,
            },
            {
                "item_id": "a",
                "method": "guided",
                "support_partition_pairwise_f1": 0.5,
                "support_coverage": 1.0,
                "runtime_seconds": 3.0,
            },
            {
                "item_id": "b",
                "method": "guided",
                "support_partition_pairwise_f1": 0.55,
                "support_coverage": 1.0,
                "runtime_seconds": 5.0,
            },
        ]
        summary = summarize_metrics(rows)
        f1 = next(
            row
            for row in summary
            if row["method"] == "base"
            and row["metric"] == "support_partition_pairwise_f1"
        )
        self.assertAlmostEqual(f1["mean"], 0.5)
        paired = paired_summary(rows, "base")
        f1_delta = next(
            row
            for row in paired
            if row["method"] == "guided"
            and row["metric"] == "support_partition_pairwise_f1"
        )
        self.assertAlmostEqual(f1_delta["mean_delta"], 0.025)
        self.assertEqual(
            (f1_delta["improvements"], f1_delta["regressions"]), (1, 1)
        )


if __name__ == "__main__":
    unittest.main()
