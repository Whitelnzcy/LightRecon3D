import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_stage3_scene_plane_fusion import map_stage2_points_to_global, path_key


class Stage3ScenePlaneFusionMappingTest(unittest.TestCase):
    def test_support_source_view_selects_registered_view(self):
        raw = {
            "point_plane_ids": np.asarray([0, 0], dtype=np.int32),
            "support_source_view": np.asarray([1, 2], dtype=np.int8),
            "rgb_path1": np.asarray("view_a.png"),
            "rgb_path2": np.asarray("view_b.png"),
            "pixel_xy1": np.asarray([[-1.0, -1.0], [0.0, 0.0]], dtype=np.float32),
            "pixel_xy2": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        }
        view_a_points = np.zeros((2, 2, 3), dtype=np.float32)
        view_b_points = np.zeros((2, 2, 3), dtype=np.float32)
        view_a_points[0, 0] = [1.0, 0.0, 0.0]
        view_b_points[1, 1] = [0.0, 2.0, 0.0]
        global_views = {
            path_key("view_a.png"): {
                "alignment_view_index": 0,
                "points": view_a_points,
                "conf": np.ones((2, 2), dtype=np.float32),
                "colors": np.zeros((2, 2, 3), dtype=np.uint8),
            },
            path_key("view_b.png"): {
                "alignment_view_index": 1,
                "points": view_b_points,
                "conf": np.ones((2, 2), dtype=np.float32),
                "colors": np.zeros((2, 2, 3), dtype=np.uint8),
            },
        }

        points, _, keep, stats = map_stage2_points_to_global(raw, global_views, SimpleNamespace(global_min_conf=0.0))

        self.assertEqual(keep.tolist(), [True, True])
        np.testing.assert_allclose(points, np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32))
        self.assertEqual(stats["source_counts"], {"1": 1, "2": 1})
        self.assertEqual(stats["kept_counts"], {"1": 1, "2": 1})
        self.assertEqual(stats["registered_view_indices"], {"1": 0, "2": 1})

        points, _, keep, _, view_indices, pixel_xy = map_stage2_points_to_global(
            raw, global_views, SimpleNamespace(global_min_conf=0.0), return_provenance=True)
        self.assertEqual(len(points), int(keep.sum()))
        self.assertEqual(view_indices.tolist(), [0, 1])
        self.assertEqual(pixel_xy.tolist(), [[0, 0], [1, 1]])


if __name__ == "__main__":
    unittest.main()
