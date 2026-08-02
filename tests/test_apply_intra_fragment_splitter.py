import unittest

import numpy as np

from apply_intra_fragment_splitter import (
    component_statistics,
    merge_small_anchor_clusters,
    nearest_anchor_assignment,
    nearest_anchor_labels,
    refit_planes,
)


class ApplyIntraFragmentSplitterTest(unittest.TestCase):
    def test_nearest_anchor_labels_follow_pixel_support(self):
        labels = nearest_anchor_labels(
            np.asarray([[0.0, 0.0], [9.0, 0.0]]),
            np.asarray([[0.0, 0.0], [10.0, 0.0]]),
            np.asarray([3, 7]),
        )
        np.testing.assert_array_equal(labels, np.asarray([3, 7]))

    def test_small_anchor_cluster_merges_by_affinity(self):
        labels = np.asarray([0, 0, 1, 1, 2])
        affinity = np.eye(5)
        affinity[4, :2] = affinity[:2, 4] = 0.8
        affinity[4, 2:4] = affinity[2:4, 4] = 0.2
        merged = merge_small_anchor_clusters(labels, affinity, 2)
        self.assertEqual(merged[4], merged[0])
        self.assertNotEqual(merged[4], merged[2])

    def test_nearest_anchor_assignment_reports_pixel_distance(self):
        labels, distances = nearest_anchor_assignment(
            np.asarray([[1.0, 0.0], [7.0, 0.0]]),
            np.asarray([[0.0, 0.0], [10.0, 0.0]]),
            np.asarray([3, 7]),
        )
        np.testing.assert_array_equal(labels, [3, 7])
        np.testing.assert_allclose(distances, [1.0, 3.0])

    def test_component_statistics_exposes_long_tail(self):
        stats = component_statistics(
            np.asarray([0] * 64 + [1] * 31 + [2] * 5)
        )
        self.assertEqual(stats["component_sizes"], [64, 31, 5])
        self.assertEqual(stats["tiny_component_count_lt16"], 1)
        self.assertEqual(stats["tiny_component_count_lt32"], 2)
        self.assertEqual(stats["tiny_component_count_lt64"], 2)
        self.assertAlmostEqual(stats["largest_component_ratio"], 0.64)

    def test_refit_planes_ignores_unassigned_label(self):
        result = refit_planes(
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [9.0, 9.0, 9.0],
                ]
            ),
            np.asarray([0, 0, 0, -1]),
        )
        np.testing.assert_array_equal(result["plane_ids"], [0])
        np.testing.assert_array_equal(result["plane_inlier_counts"], [3])


if __name__ == "__main__":
    unittest.main()
