import tempfile
import unittest
from pathlib import Path

import numpy as np

from planegraph_ba import apply_sim3, load_inputs, optimize_planegraph_ba, rodrigues


class PlaneGraphBATest(unittest.TestCase):
    def test_rodrigues_is_a_rotation(self):
        rotation = rodrigues(np.asarray([0.1, -0.2, 0.05]))
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=10)

    def test_load_inputs_joins_by_view_and_pointmap_pixel(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cache_path = directory / "cache.npz"
            support_path = directory / "support.npz"
            points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
            views = np.asarray([0, 0, 1, 1], np.int32)
            pixels = np.asarray([[0, 0], [1, 0], [0, 0], [1, 0]], np.int32)
            np.savez_compressed(
                cache_path, points=points, colors=np.zeros((4, 3), np.uint8),
                confidence=np.ones(4, np.float32), view_indices=views, pixel_xy=pixels)
            np.savez_compressed(
                support_path, point_plane_ids=np.asarray([3, 3, 3, -1], np.int32),
                alignment_view_indices=views, pointmap_pixel_xy=pixels)
            cache, support = load_inputs(cache_path, support_path, min_plane_points=3)
            self.assertEqual(len(cache["points"]), 4)
            self.assertEqual(support["cache_indices"].tolist(), [0, 1, 2])
            self.assertEqual(support["plane_ids"].tolist(), [0, 0, 0])
            self.assertEqual(support["source_plane_ids"].tolist(), [3])

    def test_multiview_planes_reduce_structural_residual(self):
        rng = np.random.default_rng(7)
        count = 240
        yz = rng.uniform(-1.0, 1.0, (count, 2))
        xz = rng.uniform(-1.0, 1.0, (count, 2))
        xy = rng.uniform(-1.0, 1.0, (count, 2))
        base = np.vstack((
            np.column_stack((np.zeros(count), yz)),
            np.column_stack((xz[:, 0], np.zeros(count), xz[:, 1])),
            np.column_stack((xy, np.zeros(count))),
        )).astype(np.float64)
        base += rng.normal(0.0, 0.001, base.shape)
        plane_ids = np.repeat(np.arange(3, dtype=np.int32), count)
        bad_alignment = np.asarray([0.045, -0.06, 0.035, 0.09, -0.07, 0.06, 0.04])
        moving = apply_sim3(base, bad_alignment)
        points = np.vstack((base, moving))
        cache = {
            "points": points,
            "colors": np.zeros((len(points), 3), np.uint8),
            "confidence": np.ones(len(points)),
            "view_indices": np.concatenate((np.zeros(len(base), np.int32), np.ones(len(base), np.int32))),
            "pixel_xy": np.column_stack((np.arange(len(points)), np.zeros(len(points), np.int32))),
        }
        support = {
            "cache_indices": np.arange(len(points), dtype=np.int64),
            "plane_ids": np.concatenate((plane_ids, plane_ids)),
            "source_plane_ids": np.arange(3, dtype=np.int64),
        }
        result = optimize_planegraph_ba(
            cache, support, iterations=12, huber_delta=0.15, min_plane_views=2,
            min_view_observations=64, rotation_anchor=0.02, translation_anchor=0.02,
            scale_anchor=0.02, max_rotation_deg=12.0,
            max_translation_fraction=0.2, max_scale_fraction=0.15,
            reference_view=0, tolerance=1e-8)
        before = float(np.mean(np.abs(result["initial_residual"])))
        after = float(np.mean(np.abs(result["final_residual"])))
        self.assertLess(after, before * 0.45)
        np.testing.assert_allclose(result["parameters"][0], np.zeros(7), atol=1e-12)
        self.assertEqual(int(result["active_planes"].sum()), 3)


if __name__ == "__main__":
    unittest.main()
