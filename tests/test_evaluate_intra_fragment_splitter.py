import unittest

import numpy as np

from evaluate_intra_fragment_splitter import (
    average_linkage_components,
    average_linkage_tree,
    connected_components,
    partition_pair_metrics,
)


class EvaluateIntraFragmentSplitterTest(unittest.TestCase):
    def test_connected_components_respect_affinity(self):
        neighbors = np.asarray([[1], [0], [3], [2]])
        probability = np.asarray([[0.9], [0.9], [0.9], [0.9]])
        labels = connected_components(
            4,
            neighbors,
            probability,
            np.zeros(4),
            affinity_threshold=0.5,
            boundary_threshold=0.5,
        )
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_pair_metrics_detect_perfect_split(self):
        truth = np.asarray([0, 0, 1, 1])
        metrics = partition_pair_metrics(truth, truth)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_average_linkage_rejects_a_single_weak_bridge(self):
        affinity = np.asarray(
            [
                [1.0, 0.9, 0.2, 0.2],
                [0.9, 1.0, 0.7, 0.2],
                [0.2, 0.7, 1.0, 0.9],
                [0.2, 0.2, 0.9, 1.0],
            ]
        )
        labels = average_linkage_components(
            average_linkage_tree(affinity),
            4,
            affinity_threshold=0.65,
        )
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[1], labels[2])

    def test_average_linkage_handles_singleton(self):
        labels = average_linkage_components(
            average_linkage_tree(np.ones((1, 1))),
            1,
            affinity_threshold=0.5,
        )
        np.testing.assert_array_equal(labels, np.asarray([0]))


if __name__ == "__main__":
    unittest.main()
