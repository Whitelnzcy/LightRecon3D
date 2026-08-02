import unittest

from rejudge_support_bounded_e0_metrics import rejudge_scene


class RejudgeE0MetricsTest(unittest.TestCase):
    def test_exact_e0_passes_explicit_denominators(self):
        old = {
            "scene": "scene",
            "cache_point_count": 100,
            "gt_labeled_point_count": 90,
            "pred_plane_count": 1,
            "gt_plane_count": 1,
            "true_positive_planes": 1,
            "plane_precision": 1.0,
            "plane_recall": 1.0,
            "plane_f1": 1.0,
            "plane_weighted_matched_iou_mean": 1.0,
            "plane_weighted_matched_iou_median": 1.0,
            "minimum_matched_iou": 1.0,
            "point_weighted_matched_iou": 1.0,
            "assignment_rate_gt_support": 1.0,
            "fragmentation_excess": 0,
            "overmerge_excess": 0,
        }
        planes = [
            {
                "plane_tp": True,
                "matched_iou": 1.0,
                "pred_point_count": 90,
                "gt_point_count": 90,
            }
        ]
        result = rejudge_scene(old, planes)
        self.assertTrue(result["e0_all_pass"])
        self.assertEqual(result["assigned_cache_point_count"], 90)
        self.assertEqual(result["gt_plane_point_count"], 90)
        self.assertEqual(result["nonplane_false_assigned_point_count"], 0)
        self.assertEqual(result["gt_support_matched_coverage"], 1.0)
        self.assertEqual(result["full_cache_label_rate"], 0.9)
        self.assertEqual(result["gt_plane_label_rate"], 0.9)

    def test_nonplane_false_assignment_fails_integer_contract(self):
        old = {
            "scene": "scene",
            "cache_point_count": 100,
            "gt_labeled_point_count": 90,
            "pred_plane_count": 1,
            "gt_plane_count": 1,
            "true_positive_planes": 1,
            "plane_precision": 1.0,
            "plane_recall": 1.0,
            "plane_f1": 1.0,
            "plane_weighted_matched_iou_mean": 1.0,
            "plane_weighted_matched_iou_median": 1.0,
            "minimum_matched_iou": 1.0,
            "point_weighted_matched_iou": 1.0,
            "assignment_rate_gt_support": 1.0,
            "fragmentation_excess": 0,
            "overmerge_excess": 0,
        }
        planes = [
            {
                "plane_tp": True,
                "matched_iou": 1.0,
                "pred_point_count": 91,
                "gt_point_count": 91,
            }
        ]
        result = rejudge_scene(old, planes)
        self.assertEqual(result["nonplane_false_assigned_point_count"], 1)
        self.assertFalse(result["e0_integer_count_pass"])
        self.assertFalse(result["e0_all_pass"])


if __name__ == "__main__":
    unittest.main()
