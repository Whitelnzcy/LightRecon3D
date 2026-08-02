import unittest

import numpy as np

from build_fragment_graph_features import (
    _bbox_gap,
    _bbox_iou,
    _normalize_colors,
    _robust_interval_relation,
    _tangent_support_relations,
)


class FragmentGraphFeatureTest(unittest.TestCase):
    def test_color_normalization_handles_uint8(self):
        values = _normalize_colors(np.asarray([[0, 128, 255]], dtype=np.uint8))
        np.testing.assert_allclose(values[0, [0, 2]], [0.0, 1.0])

    def test_bbox_gap(self):
        self.assertEqual(
            _bbox_gap([0, 0], [2, 2], [4, 2], [5, 3]),
            2.0,
        )

    def test_bbox_iou(self):
        first = {"min": [0, 0], "max": [1, 1]}
        second = {"min": [1, 0], "max": [2, 1]}
        self.assertAlmostEqual(_bbox_iou(first, second), 1.0 / 3.0)

    def test_robust_interval_relation_detects_gap_and_overlap(self):
        gap, overlap = _robust_interval_relation(
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([2.0, 2.5, 3.0]),
            trim_quantile=0.0,
        )
        self.assertAlmostEqual(gap, 1.0)
        self.assertEqual(overlap, 0.0)
        gap, overlap = _robust_interval_relation(
            np.asarray([0.0, 1.0, 2.0]),
            np.asarray([1.0, 2.0, 3.0]),
            trim_quantile=0.0,
        )
        self.assertEqual(gap, 0.0)
        self.assertAlmostEqual(overlap, 0.5)

    def test_tangent_support_relations_ignore_normal_displacement(self):
        first = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        second = first + np.asarray([3.0, 0.0, 0.2])
        relation = _tangent_support_relations(
            first, second, np.asarray([0.0, 0.0, 1.0])
        )
        self.assertGreater(relation["primary_gap"], 1.0)
        self.assertAlmostEqual(relation["secondary_gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
