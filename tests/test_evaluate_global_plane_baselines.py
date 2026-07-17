import unittest

import numpy as np

from evaluate_global_plane_baselines import (
    evaluate_arrays,
    linear_sum_assignment,
    public_partition_metrics,
)


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
        self.assertEqual(metrics["segmentation_voi_nats"], 0.0)
        self.assertEqual(metrics["segmentation_rand_index"], 1.0)
        self.assertEqual(metrics["segmentation_covering_symmetric"], 1.0)

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

    def test_public_metrics_report_known_overmerge(self):
        pred = np.zeros(4, np.int32)
        gt = np.asarray([0, 0, 1, 1], np.int32)
        metrics = public_partition_metrics(pred, gt)
        self.assertAlmostEqual(metrics["segmentation_voi_nats"], np.log(2.0))
        self.assertAlmostEqual(metrics["segmentation_rand_index"], 2.0 / 6.0)
        self.assertAlmostEqual(
            metrics["segmentation_covering_gt_by_pred"], 0.5
        )
        self.assertAlmostEqual(
            metrics["segmentation_covering_pred_by_gt"], 0.5
        )
        self.assertAlmostEqual(metrics["segmentation_covering_symmetric"], 0.5)

    def test_public_metrics_do_not_drop_unassigned_points(self):
        gt = np.asarray([0, 0, 1, 1], np.int32)
        complete = public_partition_metrics(gt, gt)
        sparse = public_partition_metrics(
            np.asarray([0, -1, 1, -1], np.int32),
            gt,
        )
        self.assertEqual(sparse["segmentation_evaluation_points"], 4)
        self.assertEqual(sparse["segmentation_pred_segment_count"], 3)
        self.assertGreater(sparse["segmentation_voi_nats"], 0.0)
        self.assertLess(
            sparse["segmentation_covering_symmetric"],
            complete["segmentation_covering_symmetric"],
        )


if __name__ == "__main__":
    unittest.main()
