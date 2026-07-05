import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_stage3_view_registry import validate_input_dir


def write_npz(path, scene="scene_00001", pair_group="room_a", rgb1="a.png", rgb2="b.png", view1="1", view2="2", xy=None):
    if xy is None:
        xy = np.asarray([[0.0, 0.0], [0.5, -0.5]], dtype=np.float32)
    n = len(xy)
    np.savez_compressed(
        path,
        schema_version=np.asarray(2, dtype=np.int32),
        scene_name=np.asarray(scene),
        pair_group=np.asarray(pair_group),
        rgb_path1=np.asarray(rgb1),
        rgb_path2=np.asarray(rgb2),
        json_path1=np.asarray(rgb1.replace(".png", ".json")),
        json_path2=np.asarray(rgb2.replace(".png", ".json")),
        view_id1=np.asarray(view1),
        view_id2=np.asarray(view2),
        original_hw1=np.asarray([720, 1280], dtype=np.int32),
        original_hw2=np.asarray([720, 1280], dtype=np.int32),
        stage1_input_hw1=np.asarray([512, 512], dtype=np.int32),
        stage1_input_hw2=np.asarray([512, 512], dtype=np.int32),
        stage1_mask_hw1=np.asarray([512, 512], dtype=np.int32),
        stage1_mask_hw2=np.asarray([512, 512], dtype=np.int32),
        pixel_coordinate_space=np.asarray("stage1_pointmap_normalized"),
        pixel_coordinate_order=np.asarray("xy"),
        pixel_coordinate_range=np.asarray("[-1,1]"),
        pixel_xy=xy,
        pixel_xy1=xy,
        pixel_xy2=np.zeros((0, 2), dtype=np.float32),
        support_source_view=np.ones((n,), dtype=np.int8),
        point_plane_ids=np.zeros((n,), dtype=np.int32),
        points=np.zeros((n, 3), dtype=np.float32),
        colors=np.zeros((n, 3), dtype=np.uint8),
        original_colors=np.zeros((n, 3), dtype=np.uint8),
        plane_ids=np.asarray([0], dtype=np.int32),
        plane_normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        plane_offsets=np.asarray([0.0], dtype=np.float32),
        plane_inlier_counts=np.asarray([n], dtype=np.int32),
    )


class Stage3ViewRegistryValidatorTest(unittest.TestCase):
    def test_deduplicates_overlapping_pair_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_npz(root / "pair_ab_learned_region_merge_full_pointcloud_editable_planes_data.npz", rgb1="a.png", rgb2="b.png", view1="1", view2="2")
            write_npz(root / "pair_bc_learned_region_merge_full_pointcloud_editable_planes_data.npz", rgb1="b.png", rgb2="c.png", view1="2", view2="3")
            result = validate_input_dir(root, "*.npz")
        summary = result["summary"]
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["groups"], 1)
        self.assertEqual(summary["unique_views"], 3)
        self.assertEqual(summary["groups_with_3plus_views"], 1)
        self.assertEqual(summary["duplicate_view_conflict_count"], 0)
        self.assertEqual(summary["unmapped_support_records"], 0)
        views = result["groups"][0]["view_registry"]
        self.assertEqual([row["alignment_view_index"] for row in views], [0, 1, 2])
        self.assertEqual([row["source_view_id"] for row in views], ["1", "2", "3"])

    def test_invalid_pixel_coordinates_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_xy = np.asarray([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
            write_npz(root / "bad_learned_region_merge_full_pointcloud_editable_planes_data.npz", xy=bad_xy)
            result = validate_input_dir(root, "*.npz")
        self.assertEqual(result["summary"]["records_with_errors"], 1)
        errors = result["groups"][0]["records"][0]["errors"]
        self.assertTrue(any("invalid/out-of-range" in row for row in errors))

    def test_missing_required_metadata_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez_compressed(
                root / "missing_learned_region_merge_full_pointcloud_editable_planes_data.npz",
                point_plane_ids=np.zeros((1,), dtype=np.int32),
            )
            result = validate_input_dir(root, "*.npz")
        self.assertEqual(result["summary"]["missing_required_records"], 1)
        self.assertIn("scene_name", result["groups"][0]["records"][0]["missing_fields"])


if __name__ == "__main__":
    unittest.main()
