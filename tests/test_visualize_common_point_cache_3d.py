import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualize_common_point_cache_3d import matched_plane_colors


class CommonPointCache3DVisualizationTests(unittest.TestCase):
    def test_matched_colors_share_gt_identity(self):
        gt = np.array([0, 0, 1, 1], dtype=np.int32)
        prediction = np.array([5, 5, 3, 3], dtype=np.int32)
        colors = matched_plane_colors(prediction, gt)
        self.assertTrue(np.array_equal(colors[0], colors[1]))
        self.assertTrue(np.array_equal(colors[2], colors[3]))
        self.assertFalse(np.array_equal(colors[0], colors[2]))


if __name__ == "__main__":
    unittest.main()
