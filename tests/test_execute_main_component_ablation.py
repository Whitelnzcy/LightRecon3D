import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from execute_main_component_ablation import summarize, validate_b5_contract


class MainComponentAblationTests(unittest.TestCase):
    def test_summarize_groups_scene_rows(self):
        rows = [
            {"method": "B0", "scene_name": "a", "support_partition_pairwise_f1": 0.4},
            {"method": "B0", "scene_name": "b", "support_partition_pairwise_f1": 0.8},
        ]
        result = summarize(rows)
        self.assertEqual(result[0]["scenes"], 2)
        self.assertAlmostEqual(result[0]["support_partition_pairwise_f1_mean"], 0.6)

    def test_b5_contract_requires_registry_assignments_and_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "prediction.npz"
            np.savez_compressed(
                prediction,
                point_plane_ids=np.asarray([0, 0, -1], dtype=np.int32),
                plane_ids=np.asarray([0], dtype=np.int32),
                plane_normals=np.asarray([[0, 0, 1]], dtype=np.float32),
                plane_offsets=np.asarray([0], dtype=np.float32),
                source_views=np.asarray([0, 0, 1], dtype=np.int32),
                pixel_xy=np.asarray([[1, 2], [2, 2], [3, 4]], dtype=np.int32),
                pixel_coordinate_order=np.asarray("xy"),
                pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
            )
            lines = root / "lines.json"
            lines.write_text(json.dumps({"line_count": 3}), encoding="utf-8")
            result = validate_b5_contract(prediction, lines)
            self.assertTrue(result["exact_registry_present"])
            self.assertEqual(result["structural_line_count"], 3)


if __name__ == "__main__":
    unittest.main()
