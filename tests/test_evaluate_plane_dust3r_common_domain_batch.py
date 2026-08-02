import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_plane_dust3r_common_domain_batch import (
    aggregate,
    normalized_metric_row,
    paired_comparisons,
    plane_f1,
)


class PlaneDust3rCommonDomainBatchTests(unittest.TestCase):
    def test_plane_f1_handles_zero_and_nonzero(self):
        self.assertEqual(plane_f1(0.0, 0.0), 0.0)
        self.assertAlmostEqual(plane_f1(1.0, 0.5), 2.0 / 3.0)

    def test_normalized_row_adds_scene_and_f1(self):
        result = normalized_metric_row(
            {
                "method": "method",
                "plane_precision": 0.8,
                "plane_recall": 0.5,
            },
            scene_name="scene_1",
            item_id="item_1",
        )
        self.assertEqual(result["scene_name"], "scene_1")
        self.assertAlmostEqual(result["plane_f1"], 8.0 / 13.0)

    def test_aggregate_reports_mean_and_median_per_method(self):
        rows = []
        for value in (0.2, 0.8):
            row = {
                "method": "a",
                "plane_precision": value,
                "plane_recall": value,
                "plane_f1": value,
            }
            rows.append(row)
        result = aggregate(rows)
        self.assertEqual(result[0]["scenes"], 2)
        self.assertAlmostEqual(result[0]["plane_f1_mean"], 0.5)
        self.assertAlmostEqual(result[0]["plane_f1_median"], 0.5)

    def test_paired_comparison_respects_lower_is_better_metrics(self):
        rows = []
        values = {
            "global_ransac_cc": (0.2, 1.0),
            "learning_guided_ransac_cc": (0.5, 0.7),
            "plane_dust3r_same_input_common_point_cache": (0.8, 0.3),
        }
        for method, (f1, voi) in values.items():
            rows.append(
                {
                    "method": method,
                    "scene_name": "scene",
                    "plane_f1": f1,
                    "support_matched_iou": f1,
                    "segmentation_voi_nats": voi,
                    "segmentation_rand_index": f1,
                    "segmentation_covering_symmetric": f1,
                }
            )
        result = paired_comparisons(rows)
        guided_voi = next(
            row
            for row in result
            if row["baseline_method"] == "learning_guided_ransac_cc"
            and row["metric"] == "segmentation_voi_nats"
        )
        self.assertEqual(guided_voi["wins"], 1)
        self.assertAlmostEqual(guided_voi["improvement_mean"], 0.4)


if __name__ == "__main__":
    unittest.main()
