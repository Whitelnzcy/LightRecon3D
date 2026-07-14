import unittest

import numpy as np

from lift_support_prediction_to_global_cache import resolve_support_assignments


class LiftSupportPredictionTest(unittest.TestCase):
    def setUp(self):
        self.cache_views = np.asarray([0, 0, 1, 1], dtype=np.int32)
        self.cache_xy = np.asarray([[0, 0], [1, 0], [0, 0], [1, 0]], dtype=np.int32)

    def test_exact_view_xy_join_preserves_cache_order(self):
        labels, stats = resolve_support_assignments(
            self.cache_views,
            self.cache_xy,
            np.asarray([1, 0], dtype=np.int32),
            np.asarray([[1, 0], [0, 0]], dtype=np.int32),
            np.asarray([7, 3], dtype=np.int32),
        )
        self.assertEqual(labels.tolist(), [3, -1, -1, 7])
        self.assertEqual(stats["matched_support_keys"], 2)

    def test_duplicate_same_label_is_kept_once(self):
        labels, stats = resolve_support_assignments(
            self.cache_views,
            self.cache_xy,
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([[1, 0], [1, 0]], dtype=np.int32),
            np.asarray([4, 4], dtype=np.int32),
        )
        self.assertEqual(labels.tolist(), [-1, 4, -1, -1])
        self.assertEqual(stats["duplicate_positive_support_records"], 1)
        self.assertEqual(stats["conflicting_support_keys"], 0)

    def test_conflicting_duplicate_is_explicitly_dropped(self):
        labels, stats = resolve_support_assignments(
            self.cache_views,
            self.cache_xy,
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([[1, 0], [1, 0]], dtype=np.int32),
            np.asarray([4, 5], dtype=np.int32),
        )
        self.assertTrue(np.all(labels == -1))
        self.assertEqual(stats["conflicting_support_keys"], 1)
        self.assertEqual(stats["conflicting_support_records"], 2)

    def test_unmatched_key_is_counted(self):
        labels, stats = resolve_support_assignments(
            self.cache_views,
            self.cache_xy,
            np.asarray([2], dtype=np.int32),
            np.asarray([[9, 9]], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
        )
        self.assertTrue(np.all(labels == -1))
        self.assertEqual(stats["unmatched_support_keys"], 1)
        self.assertEqual(stats["matched_cache_points"], 0)

    def test_unassigned_records_do_not_create_a_phantom_key(self):
        labels, stats = resolve_support_assignments(
            self.cache_views,
            self.cache_xy,
            np.asarray([0], dtype=np.int32),
            np.asarray([[1, 0]], dtype=np.int32),
            np.asarray([-1], dtype=np.int32),
        )
        self.assertTrue(np.all(labels == -1))
        self.assertEqual(stats["positive_support_records"], 0)
        self.assertEqual(stats["unique_positive_support_keys"], 0)

    def test_duplicate_cache_registry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not one-to-one"):
            resolve_support_assignments(
                np.asarray([0, 0]),
                np.asarray([[1, 2], [1, 2]]),
                np.asarray([0]),
                np.asarray([[1, 2]]),
                np.asarray([3]),
            )


if __name__ == "__main__":
    unittest.main()
