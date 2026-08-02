import unittest

import numpy as np

from analyze_bias_aware_fragment_clustering import (
    add_costs,
    scene_partition_f1_curve,
    scene_partition_metrics,
    tune_threshold,
)


class BiasAwareFragmentClusteringTest(unittest.TestCase):
    def test_partition_metrics_recover_two_components(self):
        rows = [
            {
                "first_fragment_id": 0,
                "second_fragment_id": 1,
                "first_gt_plane": 0,
                "second_gt_plane": 0,
                "cost": 0.1,
            },
            {
                "first_fragment_id": 0,
                "second_fragment_id": 2,
                "first_gt_plane": 0,
                "second_gt_plane": 1,
                "cost": 0.9,
            },
            {
                "first_fragment_id": 1,
                "second_fragment_id": 2,
                "first_gt_plane": 0,
                "second_gt_plane": 1,
                "cost": 0.8,
            },
        ]
        result = scene_partition_metrics(rows, "cost", threshold=0.5)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["component_count"], 2)

    def test_incremental_curve_matches_direct_evaluation(self):
        rows = [
            {
                "first_fragment_id": 0,
                "second_fragment_id": 1,
                "first_gt_plane": 0,
                "second_gt_plane": 0,
                "cost": 0.1,
            },
            {
                "first_fragment_id": 0,
                "second_fragment_id": 2,
                "first_gt_plane": 0,
                "second_gt_plane": 1,
                "cost": 0.4,
            },
            {
                "first_fragment_id": 1,
                "second_fragment_id": 2,
                "first_gt_plane": 0,
                "second_gt_plane": 1,
                "cost": 0.5,
            },
        ]
        thresholds = np.asarray([0.0, 0.2, 0.45, 0.6])
        incremental = scene_partition_f1_curve(rows, "cost", thresholds)
        direct = np.asarray(
            [
                scene_partition_metrics(rows, "cost", threshold)["f1"]
                for threshold in thresholds
            ]
        )
        np.testing.assert_allclose(incremental, direct)

    def test_threshold_tuning_returns_scalar_score(self):
        rows_by_scene = {
            "scene_a": [
                {
                    "first_fragment_id": 0,
                    "second_fragment_id": 1,
                    "first_gt_plane": 0,
                    "second_gt_plane": 0,
                    "cost": 0.1,
                }
            ],
            "scene_b": [
                {
                    "first_fragment_id": 0,
                    "second_fragment_id": 1,
                    "first_gt_plane": 0,
                    "second_gt_plane": 0,
                    "cost": 0.2,
                }
            ],
        }
        threshold, score = tune_threshold(
            rows_by_scene, ["scene_a", "scene_b"], "cost", 8
        )
        self.assertIsInstance(threshold, float)
        self.assertEqual(score, 1.0)

    def test_bias_penalty_is_added(self):
        rows = [
            {
                "pooled_rmse_normalized": 0.2,
                "bias_corrected_rmse_normalized": 0.1,
                "required_bias_std_normalized": 0.05,
            }
        ]
        add_costs(rows, penalty=2.0)
        self.assertAlmostEqual(rows[0]["pooled_cost"], 0.2)
        self.assertAlmostEqual(rows[0]["bias_corrected_cost"], 0.1)
        self.assertAlmostEqual(rows[0]["bias_penalized_cost"], 0.2)


if __name__ == "__main__":
    unittest.main()
