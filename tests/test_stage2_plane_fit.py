import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_stage2_region_merge_net import fit_plane_np
from export_stage3_scene_plane_fusion import (
    fit_plane_np as fit_stage3_plane_np,
    plane_bbox_area,
    plane_quality_rows,
    safe_filename,
)


class Stage2PlaneFitTests(unittest.TestCase):
    def _assert_known_plane(self, fit_function):
        xx, yy = np.meshgrid(
            np.linspace(-1.0, 1.0, 17, dtype=np.float32),
            np.linspace(-0.75, 0.75, 13, dtype=np.float32),
            indexing="xy",
        )
        zz = 0.25 * xx - 0.5 * yy + 1.25
        points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3).astype(np.float32)

        normal, offset, residual = fit_function(points)

        expected = np.asarray([-0.25, 0.5, 1.0], dtype=np.float32)
        expected /= np.sqrt(np.sum(expected * expected))
        self.assertAlmostEqual(float(np.sqrt(np.sum(normal * normal))), 1.0, places=5)
        self.assertGreater(float(abs(np.sum(normal * expected))), 0.99999)
        self.assertLess(float(residual.max()), 1e-5)
        point_residual = np.abs(np.sum(points * normal[None, :], axis=1) + offset)
        self.assertLess(float(point_residual.max()), 1e-5)

    def test_stage2_covariance_fit_recovers_known_plane(self):
        self._assert_known_plane(fit_plane_np)

    def test_stage3_covariance_fit_recovers_known_plane(self):
        self._assert_known_plane(fit_stage3_plane_np)

    def test_stage3_quality_after_torch_initialization(self):
        xx, yy = np.meshgrid(
            np.linspace(-1.0, 1.0, 17, dtype=np.float32),
            np.linspace(-0.75, 0.75, 13, dtype=np.float32),
            indexing="xy",
        )
        zz = 0.25 * xx - 0.5 * yy + 1.25
        points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3).astype(np.float32)
        normal, offset, _ = fit_stage3_plane_np(points)

        rows = plane_quality_rows(
            points,
            np.zeros(len(points), dtype=np.int32),
            normal[None],
            np.asarray([offset], dtype=np.float32),
            np.zeros(len(points), dtype=np.int8),
            np.zeros(len(points), dtype=np.int32),
        )

        self.assertEqual(len(rows), 1)
        self.assertLess(rows[0]["residual_p95"], 1e-5)
        self.assertGreater(plane_bbox_area(points, normal), 0.0)

    def test_stage3_safe_filename_is_short_and_collision_resistant(self):
        common = "E:/very/long/prefix/" * 8
        name_a = safe_filename(common + "scene_a/perspective/empty")
        name_b = safe_filename(common + "scene_b/perspective/empty")

        self.assertLessEqual(len(name_a), 64)
        self.assertLessEqual(len(name_b), 64)
        self.assertNotEqual(name_a, name_b)


if __name__ == "__main__":
    unittest.main()
