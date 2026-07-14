import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_stage3_scene_plane_fusion import (
    gather_pointmap_points,
    map_stage2_points_to_global,
    parse_path_prefix_maps,
    path_key,
    remap_path,
)


class Stage3ScenePlaneFusionMappingTest(unittest.TestCase):
    def test_pointmap_gather_uses_explicit_view_and_xy_provenance(self):
        first = np.zeros((2, 3, 3), dtype=np.float32)
        second = np.zeros((1, 2, 3), dtype=np.float32)
        first[1, 2] = [1.0, 2.0, 3.0]
        second[0, 1] = [4.0, 5.0, 6.0]

        points = gather_pointmap_points(
            [first, second],
            np.asarray([1, 0], dtype=np.int32),
            np.asarray([[1, 0], [2, 1]], dtype=np.int32),
        )

        np.testing.assert_allclose(points, [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])

        with self.assertRaisesRegex(ValueError, "out of range"):
            gather_pointmap_points(
                [first],
                np.asarray([0], dtype=np.int32),
                np.asarray([[3, 0]], dtype=np.int32),
            )

    def test_path_prefix_remap_supports_cross_machine_npz_paths(self):
        mappings = parse_path_prefix_maps(
            ["/gemini/data-1/Structured3D=E:/Study/code/LightRecon3D/data/Structured3D"]
        )
        remapped = remap_path(
            "/gemini/data-1/Structured3D/scene_00180/view/rgb.png", mappings
        )
        self.assertEqual(
            path_key(remapped),
            "E:/Study/code/LightRecon3D/data/Structured3D/scene_00180/view/rgb.png",
        )

        with self.assertRaisesRegex(ValueError, "expected SOURCE_PREFIX"):
            parse_path_prefix_maps(["missing_separator"])

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

    def test_support_mapping_uses_remapped_rgb_path(self):
        raw = {
            "point_plane_ids": np.asarray([0], dtype=np.int32),
            "support_source_view": np.asarray([1], dtype=np.int8),
            "rgb_path1": np.asarray("/server/data/view_a.png"),
            "pixel_xy1": np.asarray([[-1.0, -1.0]], dtype=np.float32),
        }
        view_points = np.zeros((1, 1, 3), dtype=np.float32)
        view_points[0, 0] = [1.0, 2.0, 3.0]
        global_views = {
            path_key("E:/local/data/view_a.png"): {
                "alignment_view_index": 4,
                "points": view_points,
                "conf": np.ones((1, 1), dtype=np.float32),
                "colors": np.zeros((1, 1, 3), dtype=np.uint8),
            }
        }
        args = SimpleNamespace(
            global_min_conf=0.0,
            path_prefix_maps=parse_path_prefix_maps(["/server/data=E:/local/data"]),
        )

        points, _, keep, stats = map_stage2_points_to_global(raw, global_views, args)

        self.assertEqual(keep.tolist(), [True])
        np.testing.assert_allclose(points, [[1.0, 2.0, 3.0]])
        self.assertEqual(stats["registered_view_indices"], {"1": 4})


if __name__ == "__main__":
    unittest.main()
