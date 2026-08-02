import unittest

import numpy as np

from evaluate_all_fragment_identity_end_to_end import (
    assign_fragment_clusters,
    fit_cluster_planes,
)
from evaluate_learned_fragment_identity import training_sample_weights


class AllFragmentIdentityEndToEndTest(unittest.TestCase):
    def test_fragment_clusters_map_back_to_support_records(self):
        fragment_ids = np.asarray([2, 2, 5, 7, -1])
        result = assign_fragment_clusters(
            fragment_ids,
            [2, 5, 7],
            np.asarray([3, 3, 9]),
        )
        np.testing.assert_array_equal(result, [0, 0, 0, 1, -1])

    def test_cluster_plane_fit_is_contiguous(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        normals, offsets, counts = fit_cluster_planes(points, labels)
        self.assertEqual(normals.shape, (2, 3))
        np.testing.assert_array_equal(counts, [3, 3])
        self.assertAlmostEqual(abs(float(offsets[1])), 1.0)

    def test_hard_negative_weight_targets_coplanar_lookalikes(self):
        rows = [
            {
                "same_gt_plane": False,
                "features": {
                    "normal_angle_deg": 2.0,
                    "pooled_rmse_normalized": 0.02,
                },
            },
            {
                "same_gt_plane": False,
                "features": {
                    "normal_angle_deg": 60.0,
                    "pooled_rmse_normalized": 0.2,
                },
            },
            {
                "same_gt_plane": True,
                "features": {
                    "normal_angle_deg": 2.0,
                    "pooled_rmse_normalized": 0.02,
                },
            },
        ]
        np.testing.assert_array_equal(
            training_sample_weights(rows, 4.0),
            [4.0, 1.0, 1.0],
        )
        self.assertIsNone(training_sample_weights(rows, 1.0))


if __name__ == "__main__":
    unittest.main()
