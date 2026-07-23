import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_gt_support_guidance import build_gt_support
from lift_support_prediction_to_global_cache import load_support_prediction


class BuildGtSupportGuidanceTests(unittest.TestCase):
    def test_builds_exact_guided_support_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "gt.npz"
            output = root / "support.npz"
            labels = np.asarray([0, 1, -1], dtype=np.int32)
            views = np.asarray([0, 0, 1], dtype=np.int32)
            pixels = np.asarray([[2, 3], [4, 5], [6, 7]], dtype=np.int32)
            np.savez_compressed(
                source,
                point_plane_ids=labels,
                view_indices=views,
                pixel_xy=pixels,
                pixel_coordinate_order=np.asarray("xy"),
                pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
            )
            result = build_gt_support(source, output)
            support = load_support_prediction(output)
            np.testing.assert_array_equal(support["labels"], labels)
            np.testing.assert_array_equal(support["view_indices"], views)
            np.testing.assert_array_equal(support["pixel_xy"], pixels)
            self.assertEqual(result["assigned_records"], 2)
            self.assertTrue(output.with_suffix(".json").is_file())
            with self.assertRaises(FileExistsError):
                build_gt_support(source, output)

    def test_rejects_duplicate_registry_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "gt.npz"
            np.savez_compressed(
                source,
                point_plane_ids=np.asarray([0, 1], dtype=np.int32),
                view_indices=np.asarray([0, 0], dtype=np.int32),
                pixel_xy=np.asarray([[2, 3], [2, 3]], dtype=np.int32),
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                build_gt_support(source, root / "support.npz")


if __name__ == "__main__":
    unittest.main()
