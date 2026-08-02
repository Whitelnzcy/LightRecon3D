import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualize_common_point_cache_partitions import prediction_to_gt_color_ids


class CommonPointCacheVisualizationTests(unittest.TestCase):
    def test_prediction_colors_follow_hungarian_gt_match(self):
        gt = np.array([0, 0, 1, 1, 2, 2])
        prediction = np.array([5, 5, 3, 3, 8, -1])
        mapping = prediction_to_gt_color_ids(prediction, gt)
        self.assertEqual(mapping[5], 0)
        self.assertEqual(mapping[3], 1)
        self.assertEqual(mapping[8], 2)


if __name__ == "__main__":
    unittest.main()
