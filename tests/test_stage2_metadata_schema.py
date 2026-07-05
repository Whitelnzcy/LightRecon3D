import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_stage2_learned_region_merge_editables import (
    PASSTHROUGH_METADATA_KEYS,
    STAGE2_SCHEMA_VERSION,
    validate_metadata_lengths,
)


class Stage2MetadataSchemaTest(unittest.TestCase):
    def test_required_global_alignment_metadata_is_passthrough(self):
        expected = {
            "scene_name",
            "pair_group",
            "rgb_path1",
            "rgb_path2",
            "json_path1",
            "json_path2",
            "view_id1",
            "view_id2",
            "original_hw1",
            "original_hw2",
            "stage1_input_hw1",
            "stage1_input_hw2",
            "stage1_mask_hw1",
            "stage1_mask_hw2",
            "pixel_coordinate_space",
            "pixel_coordinate_order",
            "pixel_coordinate_range",
            "pixel_xy",
            "pixel_xy1",
            "pixel_xy2",
            "support_source_view",
        }
        self.assertEqual(STAGE2_SCHEMA_VERSION, 2)
        self.assertTrue(expected.issubset(set(PASSTHROUGH_METADATA_KEYS)))

    def test_per_point_metadata_length_validation_accepts_matching_rows(self):
        raw = {
            "pixel_xy1": np.zeros((4, 2), dtype=np.float32),
            "support_source_view": np.ones((4,), dtype=np.int8),
            "pixel_xy2": np.zeros((0, 2), dtype=np.float32),
        }
        validate_metadata_lengths(raw, 4, "case.npz")

    def test_per_point_metadata_length_validation_rejects_mismatched_rows(self):
        raw = {"pixel_xy1": np.zeros((3, 2), dtype=np.float32)}
        with self.assertRaisesRegex(RuntimeError, "pixel_xy1 has 3 rows, expected 4"):
            validate_metadata_lengths(raw, 4, "case.npz")


if __name__ == "__main__":
    unittest.main()
