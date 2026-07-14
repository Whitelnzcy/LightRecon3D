import unittest

import numpy as np

from evaluate_global_plane_baselines import evaluate_arrays, linear_sum_assignment


class GlobalPlaneEvaluationTest(unittest.TestCase):
    def test_rectangular_hungarian_finds_non_greedy_optimum(self):
        cost = np.asarray([[1.0, 2.0, 9.0], [1.1, 9.0, 9.0]], dtype=np.float64)
        rows, cols = linear_sum_assignment(cost)
        self.assertEqual(rows.tolist(), [0, 1])
        self.assertEqual(cols.tolist(), [1, 0])
        self.assertAlmostEqual(float(cost[rows, cols].sum()), 3.1)

    def test_perfect_prediction(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], np.float32)
        labels = np.array([0, 0, 1, 1], np.int32)
        normals = np.array([[0, 1, 0], [0, 1, 0]], np.float32)
        metrics = evaluate_arrays(points, labels, normals, np.zeros(2), labels, normals)
        self.assertEqual(metrics["plane_precision"], 1.0)
        self.assertEqual(metrics["plane_recall"], 1.0)
        self.assertEqual(metrics["support_matched_iou"], 1.0)
        self.assertEqual(metrics["plane_count_error"], 0)
        self.assertEqual(metrics["support_partition_pairwise_f1"], 1.0)

    def test_fragmentation_is_reported(self):
        points = np.zeros((4, 3), np.float32)
        pred = np.array([0, 0, 1, 1], np.int32)
        gt = np.zeros(4, np.int32)
        metrics = evaluate_arrays(points, pred, np.array([[1, 0, 0]] * 2, np.float32),
                                  np.zeros(2), gt, np.array([[1, 0, 0]], np.float32),
                                  match_iou=.4, fragmentation_iou=.1)
        self.assertEqual(metrics["fragmentation_excess"], 1)
        self.assertEqual(metrics["fragmented_gt_planes"], 1)

    def test_sparse_correct_partition_is_separated_from_dense_coverage(self):
        points = np.zeros((100, 3), np.float32)
        gt = np.repeat(np.asarray([0, 1], np.int32), 50)
        pred = np.full(100, -1, np.int32)
        pred[[0, 1, 50, 51]] = [0, 0, 1, 1]
        normals = np.asarray([[1, 0, 0], [0, 1, 0]], np.float32)
        metrics = evaluate_arrays(
            points,
            pred,
            normals,
            np.zeros(2),
            gt,
            normals,
            min_observed_plane_points=2,
        )
        self.assertEqual(metrics["true_positive_planes"], 0)
        self.assertEqual(metrics["support_coverage"], 0.04)
        self.assertEqual(metrics["support_conditioned_true_positive_planes"], 2)
        self.assertEqual(metrics["support_conditioned_matched_iou"], 1.0)
        self.assertEqual(metrics["support_partition_pairwise_f1"], 1.0)

    def test_support_partition_pairwise_metrics_detect_overmerge(self):
        points = np.zeros((6, 3), np.float32)
        pred = np.zeros(6, np.int32)
        gt = np.asarray([0, 0, 0, 1, 1, 1], np.int32)
        metrics = evaluate_arrays(
            points,
            pred,
            np.asarray([[1, 0, 0]], np.float32),
            np.zeros(1),
            gt,
            np.asarray([[1, 0, 0], [0, 1, 0]], np.float32),
        )
        self.assertLess(metrics["support_partition_pairwise_precision"], 1.0)
        self.assertEqual(metrics["support_partition_pairwise_recall"], 1.0)
        self.assertLess(metrics["support_partition_pairwise_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
