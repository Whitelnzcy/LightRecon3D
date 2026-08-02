import unittest

import numpy as np

from build_intra_fragment_split_dataset import summarize_fragments


class IntraFragmentSplitDatasetTest(unittest.TestCase):
    def test_fragment_purity_and_hard_impurity(self):
        fragments = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        gt = np.asarray([2, 2, 3, 3, 4, 4, 4, 4])
        rows = summarize_fragments(
            fragments,
            gt,
            min_fragment_points=3,
            min_labeled_rate=0.75,
            impurity_threshold=0.8,
        )
        self.assertEqual(rows[0]["gt_plane_count"], 2)
        self.assertAlmostEqual(rows[0]["gt_purity"], 0.5)
        self.assertTrue(rows[0]["hard_impure"])
        self.assertFalse(rows[1]["hard_impure"])

    def test_low_labeled_rate_is_not_eligible(self):
        rows = summarize_fragments(
            np.asarray([0, 0, 0, 0]),
            np.asarray([1, -1, -1, -1]),
            min_fragment_points=3,
            min_labeled_rate=0.5,
            impurity_threshold=0.8,
        )
        self.assertFalse(rows[0]["eligible"])


if __name__ == "__main__":
    unittest.main()
