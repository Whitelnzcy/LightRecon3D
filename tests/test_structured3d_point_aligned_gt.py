import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from build_structured3d_point_aligned_gt import (
    build_point_aligned_gt,
    camera_plane_to_world,
    dust3r_resize_crop_transform,
    parse_camera_pose,
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
            camera_pose_path = root / "camera_pose.txt"
            annotation_path = root / "annotation_3d.json"
            cache_path = root / "cache.npz"
            output_path = root / "gt.npz"
            metric_ply_path = root / "metric_gt.ply"
            cv2.imwrite(str(image_path), np.zeros((4, 8, 3), dtype=np.uint8))
            layout = {
                "junctions": [
                    {"coordinate": [0, 0]}, {"coordinate": [7, 0]},
                    {"coordinate": [7, 3]}, {"coordinate": [0, 3]},
                ],
                "planes": [{
                    "ID": 7,
                    "visible_mask": [[0, 1, 2, 3, 0]],
                    "normal": [0.0, 0.0, -1.0],
                    "offset": 2000.0,
                }],
            }
            layout_path.write_text(json.dumps(layout), encoding="utf-8")
            camera_pose_path.write_text(
                "0 0 1000 0 0 1 0 1 0 0.7853981633974483 0.7853981633974483 1\n",
                encoding="utf-8",
            )
            annotation_path.write_text(json.dumps({
                "planes": [{
                    "ID": 7,
                    "normal": [0.0, 0.0, -1.0],
                    "offset": 3000.0,
                }]
            }), encoding="utf-8")
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
                output_metric_ply=metric_ply_path,
            )
            with np.load(output_path, allow_pickle=False) as result:
                self.assertEqual(row["planes"], 1)
                self.assertEqual(row["labeled_points"], len(points))
                self.assertEqual(result["source_gt_plane_ids"].tolist(), [7])
                self.assertEqual(result["point_plane_ids"].tolist(), [0] * len(points))
                np.testing.assert_array_equal(result["points"], points)
                np.testing.assert_array_equal(result["pixel_xy"], pixels)
                self.assertEqual(int(result["metric_valid_mask"].sum()), len(points))
                np.testing.assert_allclose(
                    result["metric_points_world_m"][:, 2], 3.0, atol=1e-6
                )
                self.assertEqual(result["metric_length_unit"].item(), "metre")
                self.assertLess(row["max_plane_normal_consistency_error_deg"], 1e-6)
                self.assertLess(row["max_plane_offset_consistency_error_mm"], 1e-6)
                self.assertTrue(metric_ply_path.is_file())

    def test_camera_plane_transform_matches_world_equation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera_pose.txt"
            path.write_text(
                "10 20 1000 0 0 1 0 1 0 0.5 0.4 1\n", encoding="utf-8"
            )
            pose = parse_camera_pose(path)
            normal, offset = camera_plane_to_world(
                [0.0, 0.0, -1.0], 2000.0, pose
            )
            np.testing.assert_allclose(normal, [0.0, 0.0, -1.0], atol=1e-12)
            self.assertAlmostEqual(offset, 3000.0)


if __name__ == "__main__":
    unittest.main()
