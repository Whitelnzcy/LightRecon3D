import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_plane_dust3r_same_input_batch import (
    aggregate_items,
    parse_metric_results,
    parse_reuse_specifications,
    select_items,
    validate_manifest,
)


class PlaneDust3rBatchTests(unittest.TestCase):
    def test_parses_native_metric_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metric_results.txt"
            path.write_text(
                "[np.float64(0.5), np.float64(0.1), np.float64(2.0), "
                "np.float64(0.03), np.float64(0.04)]\n"
                "[np.float64(0.7), np.float64(0.2), np.float64(4.0), "
                "np.float64(0.05), np.float64(0.06)]\n"
                "precision: 0.8, recall: 0.6\n",
                encoding="utf-8",
            )
            result = parse_metric_results(path)
        self.assertEqual(result["view_count"], 2)
        self.assertAlmostEqual(result["mean"]["layout_iou"], 0.6)
        self.assertAlmostEqual(result["mean"]["edge_error"], 3.0)
        self.assertAlmostEqual(result["plane_precision"], 0.8)
        self.assertAlmostEqual(result["plane_recall"], 0.6)

    def test_aggregate_separates_room_macro_and_view_micro(self):
        items = [
            {
                "status": "complete",
                "metrics": {
                    "view_rows": [[1, 2, 3, 4, 5]],
                    "mean": dict(
                        zip(
                            (
                                "layout_iou",
                                "pixel_error",
                                "edge_error",
                                "depth_rmse",
                                "scale_aligned_rmse",
                            ),
                            [1, 2, 3, 4, 5],
                        )
                    ),
                    "plane_precision": 1.0,
                    "plane_recall": 0.5,
                },
            },
            {"status": "missing"},
        ]
        result = aggregate_items(items)
        self.assertEqual(result["requested_rooms"], 2)
        self.assertEqual(result["complete_rooms"], 1)
        self.assertEqual(result["complete_views"], 1)
        self.assertEqual(result["view_micro_mean"]["layout_iou"], 1)

    def test_manifest_protocol_and_scene_selection_are_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = {
                "dataset_root": str(dataset),
                "protocol": {
                    "comparison_class": "same_input_external_model_reproduction",
                    "native_protocol_claim_allowed": False,
                },
                "items": [
                    {"scene_name": "scene_1"},
                    {"scene_name": "scene_2"},
                ],
            }
            found_root, items = validate_manifest(manifest)
            self.assertEqual(found_root, dataset.resolve())
            self.assertEqual(
                [item["scene_name"] for item in select_items(items, ["scene_2"])],
                ["scene_2"],
            )
            with self.assertRaisesRegex(ValueError, "unknown requested scenes"):
                select_items(items, ["scene_3"])

    def test_reuse_specs_require_unique_scene_mapping(self):
        result = parse_reuse_specifications(["scene_1=C:/room"])
        self.assertIn("scene_1", result)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_reuse_specifications(
                ["scene_1=C:/first", "scene_1=C:/second"]
            )
        with self.assertRaisesRegex(ValueError, "must be SCENE"):
            parse_reuse_specifications(["scene_1"])


if __name__ == "__main__":
    unittest.main()
