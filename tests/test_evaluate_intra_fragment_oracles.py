import unittest

import numpy as np

from evaluate_intra_fragment_oracles import (
    compose_group_labels,
    dominant_gt_identity_labels,
    pairwise_metrics,
)


class EvaluateIntraFragmentOraclesTest(unittest.TestCase):
    def test_pairwise_metrics_distinguish_split_gain(self):
        gt = np.asarray([0, 0, 1, 1])
        unsplit = pairwise_metrics(np.zeros(4, dtype=np.int32), gt)
        split = pairwise_metrics(np.asarray([0, 0, 1, 1]), gt)
        self.assertAlmostEqual(unsplit["precision"], 1.0 / 3.0)
        self.assertAlmostEqual(unsplit["recall"], 1.0)
        self.assertAlmostEqual(split["f1"], 1.0)
        self.assertGreater(split["f1"], unsplit["f1"])

    def test_compose_group_labels_selects_candidates_per_group(self):
        labels = compose_group_labels(
            np.asarray([0, 0, 0, 1, 1, 1]),
            np.asarray([0, 0, 1, 0, 1, 1]),
            np.asarray([True, False]),
        )
        self.assertEqual(len(np.unique(labels)), 3)
        np.testing.assert_array_equal(labels[:3], [0, 0, 1])
        np.testing.assert_array_equal(labels[3:], [2, 2, 2])

    def test_compose_group_labels_preserves_unassigned_support(self):
        labels = compose_group_labels(
            np.asarray([0, 0, -1]),
            np.asarray([0, 1, -1]),
            np.asarray([True]),
        )
        np.testing.assert_array_equal(labels, [0, 1, -1])

    def test_dominant_gt_oracle_merges_whole_fragments(self):
        fragments = np.asarray([0, 0, 1, 1, 2, 2])
        gt = np.asarray([3, 3, 3, 4, 4, 4])
        labels, audit = dominant_gt_identity_labels(fragments, gt)
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[4], labels[5])
        self.assertNotEqual(labels[0], labels[4])
        self.assertAlmostEqual(
            audit["fragments"][1]["purity"],
            0.5,
        )
        self.assertEqual(audit["low_purity_fragment_count"], 1)
        # The impure fragment is merged as a whole; pointwise GT is not copied.
        self.assertEqual(labels[2], labels[3])

    def test_dominant_gt_oracle_matches_identity_minimum_size_gate(self):
        fragments = np.asarray([0, 0, 1, 1, 1])
        gt = np.asarray([3, 3, 3, 3, 3])
        labels, audit = dominant_gt_identity_labels(
            fragments,
            gt,
            min_fragment_points=3,
        )
        np.testing.assert_array_equal(labels[:2], [-1, -1])
        np.testing.assert_array_equal(labels[2:], [0, 0, 0])
        self.assertEqual(audit["rejected_small_fragment_count"], 1)
        self.assertEqual(audit["rejected_small_point_count"], 2)


if __name__ == "__main__":
    unittest.main()
