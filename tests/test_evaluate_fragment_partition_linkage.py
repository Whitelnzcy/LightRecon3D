import unittest

from analyze_bias_aware_fragment_clustering import scene_partition_metrics
from evaluate_fragment_partition_linkage import (
    linkage_context,
    metrics_from_context,
)


class FragmentPartitionLinkageTest(unittest.TestCase):
    def test_average_linkage_rejects_single_bridge(self):
        rows = [
            {
                "first_fragment_id": 0,
                "second_fragment_id": 1,
                "first_gt_plane": 0,
                "second_gt_plane": 0,
                "cost": 0.1,
            },
            {
                "first_fragment_id": 0,
                "second_fragment_id": 2,
                "first_gt_plane": 0,
                "second_gt_plane": 1,
                "cost": 0.2,
            },
            {
                "first_fragment_id": 1,
                "second_fragment_id": 2,
                "first_gt_plane": 0,
                "second_gt_plane": 1,
                "cost": 0.9,
            },
        ]
        connected = scene_partition_metrics(rows, "cost", 0.3)
        average = metrics_from_context(linkage_context(rows, "cost"), 0.3)
        self.assertLess(connected["f1"], 1.0)
        self.assertEqual(average["f1"], 1.0)

    def test_missing_edges_get_maximum_cost(self):
        rows = [
            {
                "first_fragment_id": 0,
                "second_fragment_id": 1,
                "first_gt_plane": 0,
                "second_gt_plane": 0,
                "cost": 0.1,
            },
            {
                "first_fragment_id": 1,
                "second_fragment_id": 2,
                "first_gt_plane": 0,
                "second_gt_plane": 1,
                "cost": 0.2,
            },
        ]
        result = metrics_from_context(linkage_context(rows, "cost"), 0.3)
        self.assertEqual(result["component_count"], 2)


if __name__ == "__main__":
    unittest.main()
