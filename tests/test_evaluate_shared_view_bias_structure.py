import unittest

import numpy as np

from evaluate_shared_view_bias_structure import (
    leave_one_plane_out,
    solve_view_translation,
)


class SharedViewBiasStructureTest(unittest.TestCase):
    def test_solve_view_translation(self):
        normals = np.eye(3, dtype=np.float64)
        expected = np.asarray([0.03, -0.02, 0.01])
        biases = normals @ expected
        actual = solve_view_translation(normals, biases, ridge=0.0)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_leave_one_plane_out_predicts_shared_translation(self):
        translation = np.asarray([0.03, -0.02, 0.01])
        normals = [
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([1.0, 1.0, 1.0]) / np.sqrt(3.0),
        ]
        rows = []
        for plane_id, normal in enumerate(normals):
            bias = float(normal @ translation)
            rows.append(
                {
                    "scene": "synthetic",
                    "view_index": 0,
                    "plane_id": plane_id,
                    "source_gt_plane_id": plane_id,
                    "normal": normal.tolist(),
                    "train_bias_m": bias,
                    "test_bias_m": bias,
                    "train_mean_bias_m": bias,
                    "test_mean_bias_m": bias,
                    "train_point_count": 100,
                    "test_point_count": 100,
                    "test_within_view_mad_m": 0.0,
                }
            )
        result = leave_one_plane_out(rows, ridge=0.0, min_other_planes=3)
        self.assertEqual(len(result), 4)
        for row in result:
            self.assertAlmostEqual(
                row["shared_translation_prediction_m"],
                row["target_test_bias_m"],
                places=8,
            )


if __name__ == "__main__":
    unittest.main()
