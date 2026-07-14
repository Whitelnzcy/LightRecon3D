import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from build_structured3d_point_aligned_gt import (
    build_point_aligned_gt,
    dust3r_resize_crop_transform,
)


class Structured3DPointAlignedGTTest(unittest.TestCase):
    def test_resize_crop_matches_dust3r_geometry(self):
        transform = dust3r_resize_crop_transform((720, 1280), image_size=512)
        self.assertEqual(transform["resized_hw"], [288, 512])
        self.assertEqual(transform["crop_xyxy"], [0, 0, 512, 288])
        self.assertEqual(transform["pointmap_hw"], [288, 512])

        square = dust3r_resize_crop_transform((800, 800), image_size=512)
        self.assertEqual(square["crop_xyxy"], [0, 64, 512, 448])
        self.assertEqual(square["pointmap_hw"], [384, 512])

    def test_builds_labels_on_exact_cache_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "rgb_rawlight.png"
            layout_path = root / "layout.json"
            cache_path = root / "cache.npz"
            output_path = root / "gt.npz"
            cv2.imwrite(str(image_path), np.zeros((4, 8, 3), dtype=np.uint8))
            layout = {
                "junctions": [
                    {"coordinate": [0, 0]}, {"coordinate": [7, 0]},
                    {"coordinate": [7, 3]}, {"coordinate": [0, 3]},
                ],
                "planes": [{"ID": 7, "visible_mask": [[0, 1, 2, 3, 0]]}],
            }
            layout_path.write_text(json.dumps(layout), encoding="utf-8")
            yy, xx = np.indices((4, 8), dtype=np.int32)
            pixels = np.stack((xx, yy), axis=-1).reshape(-1, 2)
            points = np.column_stack((pixels, np.zeros(len(pixels)))).astype(np.float32)
            registry = [{
                "alignment_view_index": 0,
                "image_path": str(image_path),
                "points_hw": [4, 8],
            }]
            np.savez_compressed(
                cache_path,
                points=points,
                colors=np.zeros((len(points), 3), dtype=np.uint8),
                confidence=np.ones(len(points), dtype=np.float32),
                view_indices=np.zeros(len(points), dtype=np.int32),
                pixel_xy=pixels,
                pixel_coordinate_order=np.asarray("xy"),
                pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
                dust3r_view_registry_json=np.asarray(json.dumps(registry)),
                scene_key=np.asarray("synthetic"),
            )

            row = build_point_aligned_gt(
                cache_path,
                output_path,
                image_size=8,
                patch_size=2,
                boundary_ignore_radius=0,
                min_plane_points=3,
            )
            with np.load(output_path, allow_pickle=False) as result:
                self.assertEqual(row["planes"], 1)
                self.assertEqual(row["labeled_points"], len(points))
                self.assertEqual(result["source_gt_plane_ids"].tolist(), [7])
                self.assertEqual(result["point_plane_ids"].tolist(), [0] * len(points))
                np.testing.assert_array_equal(result["points"], points)
                np.testing.assert_array_equal(result["pixel_xy"], pixels)


if __name__ == "__main__":
    unittest.main()
