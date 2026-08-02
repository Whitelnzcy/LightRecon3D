import unittest

import numpy as np

from evaluate_learned_fragment_identity import (
    cross_fitted_training_scores,
    feature_matrix,
    paired_vs_baseline,
    score_rows,
)


class LearnedFragmentIdentityTest(unittest.TestCase):
    def test_feature_matrix_preserves_requested_order(self):
        rows = [{"features": {"a": 1.0, "b": 2.0}}]
        matrix = feature_matrix(rows, ["b", "a"])
        np.testing.assert_allclose(matrix, [[2.0, 1.0]])

    def test_score_rows_converts_probability_to_cost(self):
        rows = [
            {
                "scene": "s",
                "first_fragment_id": 1,
                "second_fragment_id": 2,
                "first_gt_plane": 3,
                "second_gt_plane": 3,
            }
        ]
        scored = score_rows(rows, np.asarray([0.8]), "cost")
        self.assertAlmostEqual(scored[0]["cost"], 0.2)

    def test_paired_delta(self):
        baseline = {
            "heldout_scenes": [
                {"scene": "a", "f1": 0.4},
                {"scene": "b", "f1": 0.6},
            ]
        }
        method = {
            "heldout_scenes": [
                {"scene": "a", "f1": 0.5},
                {"scene": "b", "f1": 0.5},
            ]
        }
        result = paired_vs_baseline(method, baseline)
        self.assertAlmostEqual(result["mean_f1_delta"], 0.0)
        self.assertEqual(result["win_tie_loss"], [1, 0, 1])

    def test_cross_fitted_scores_cover_each_scene(self):
        rows_by_scene = {}
        for scene_index in range(4):
            scene = f"s{scene_index}"
            rows_by_scene[scene] = [
                {
                    "scene": scene,
                    "first_fragment_id": 0,
                    "second_fragment_id": 1,
                    "first_gt_plane": 0,
                    "second_gt_plane": 0,
                    "same_gt_plane": True,
                    "features": {"x": 0.0},
                },
                {
                    "scene": scene,
                    "first_fragment_id": 0,
                    "second_fragment_id": 2,
                    "first_gt_plane": 0,
                    "second_gt_plane": 1,
                    "same_gt_plane": False,
                    "features": {"x": 1.0},
                },
            ]
        scored, _ = cross_fitted_training_scores(
            rows_by_scene,
            sorted(rows_by_scene),
            ["x"],
            regularization_c=0.2,
            calibration_folds=2,
            seed=0,
        )
        self.assertEqual(set(scored), set(rows_by_scene))
        self.assertTrue(all("learned_cost" in rows[0] for rows in scored.values()))


if __name__ == "__main__":
    unittest.main()
