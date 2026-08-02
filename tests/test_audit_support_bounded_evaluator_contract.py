import unittest

import numpy as np

from audit_support_bounded_evaluator_contract import (
    exact_unique_join,
    matching_rows,
    median_eight_neighbor_distance,
)


class EvaluatorContractAuditTest(unittest.TestCase):
    def test_exact_join_preserves_unique_labels(self):
        views = np.asarray([0, 0, 1], dtype=np.int32)
        pixels = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
        labels = np.asarray([0, 1, -1], dtype=np.int32)
        assignment, rows = exact_unique_join(
            views,
            pixels,
            views,
            pixels,
            labels,
            scene="scene",
            variant="E0",
        )
        np.testing.assert_array_equal(assignment, labels)
        aggregate = rows[-1]
        self.assertEqual(aggregate["unique_input_key_count"], 2)
        self.assertEqual(aggregate["matched_unique_global_point_count"], 2)
        self.assertEqual(aggregate["duplicate_key_count"], 0)

    def test_same_key_duplicates_are_deduplicated(self):
        cache_views = np.asarray([0], dtype=np.int32)
        cache_pixels = np.asarray([[1, 2]], dtype=np.int32)
        assignment, rows = exact_unique_join(
            cache_views,
            cache_pixels,
            np.asarray([0, 0, 0], dtype=np.int32),
            np.asarray([[1, 2], [1, 2], [1, 2]], dtype=np.int32),
            np.asarray([4, 4, 4], dtype=np.int32),
            scene="scene",
            variant="support",
        )
        np.testing.assert_array_equal(assignment, np.asarray([4]))
        aggregate = rows[-1]
        self.assertEqual(aggregate["duplicate_key_count"], 2)
        self.assertEqual(
            aggregate["duplicate_same_global_point_count"], 2
        )
        self.assertEqual(aggregate["ambiguous_key_count"], 0)

    def test_ambiguous_key_is_dropped(self):
        assignment, rows = exact_unique_join(
            np.asarray([0], dtype=np.int32),
            np.asarray([[1, 2]], dtype=np.int32),
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([[1, 2], [1, 2]], dtype=np.int32),
            np.asarray([4, 5], dtype=np.int32),
            scene="scene",
            variant="support",
        )
        np.testing.assert_array_equal(assignment, np.asarray([-1]))
        self.assertEqual(rows[-1]["ambiguous_key_count"], 1)

    def test_unmatched_key_has_no_fallback(self):
        assignment, rows = exact_unique_join(
            np.asarray([0], dtype=np.int32),
            np.asarray([[1, 2]], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([[1, 3]], dtype=np.int32),
            np.asarray([4], dtype=np.int32),
            scene="scene",
            variant="support",
        )
        np.testing.assert_array_equal(assignment, np.asarray([-1]))
        self.assertEqual(rows[-1]["unmatched_key_count"], 1)

    def test_median_eight_neighbor_distance_is_positive(self):
        points = np.stack(
            [np.arange(10, dtype=np.float64), np.zeros(10), np.zeros(10)],
            axis=1,
        )
        self.assertGreater(median_eight_neighbor_distance(points), 0.0)

    def test_matched_coverage_counts_only_correct_matched_instances(self):
        points = np.zeros((4, 3), dtype=np.float64)
        pred = np.asarray([0, 0, 1, -1], dtype=np.int32)
        gt = np.asarray([0, 0, 1, 1], dtype=np.int32)
        normals = np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.float64)
        offsets = np.zeros(2, dtype=np.float64)
        _, summary = matching_rows(
            "scene", points, pred, normals, offsets, gt, normals
        )
        self.assertEqual(summary["gt_support_matched_point_count"], 3)


if __name__ == "__main__":
    unittest.main()
