import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluate_support_record_partitions import (
    load_keyed_prediction,
    main,
    map_unique_labels,
    per_plane_rows,
    repeated_key_diagnostics,
)
from lift_support_prediction_to_global_cache import pack_registry_keys


class SupportRecordPartitionTest(unittest.TestCase):
    def test_cli_writes_duplicate_preserving_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_path = root / "gt.npz"
            manual_path = root / "manual.npz"
            direct_path = root / "direct.npz"
            output_path = root / "audit.json"
            cache_views = np.asarray([0, 0, 1, 1], dtype=np.int32)
            cache_xy = np.asarray([[0, 0], [1, 0], [0, 0], [1, 0]], dtype=np.int32)
            normals = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            np.savez_compressed(
                gt_path,
                point_plane_ids=np.asarray([0, 0, 1, 1], dtype=np.int32),
                plane_normals=normals,
                view_indices=cache_views,
                pixel_xy=cache_xy,
                pixel_coordinate_order=np.asarray("xy"),
                pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
            )
            record_views = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
            record_xy = np.asarray(
                [[0, 0], [0, 0], [1, 0], [0, 0], [1, 0], [1, 0]],
                dtype=np.int32,
            )
            common = {
                "alignment_view_indices": record_views,
                "pointmap_pixel_xy": record_xy,
                "pointmap_pixel_coordinate_order": np.asarray("xy"),
                "pointmap_pixel_coordinate_space": np.asarray(
                    "dust3r_aligned_pointmap"
                ),
            }
            np.savez_compressed(
                manual_path,
                point_plane_ids=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
                plane_normals=normals,
                **common,
            )
            np.savez_compressed(
                direct_path,
                point_plane_ids=np.asarray([0, 0, 2, 1, 1, 3], dtype=np.int32),
                plane_normals=np.asarray(
                    [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]],
                    dtype=np.float32,
                ),
                **common,
            )
            argv = [
                "evaluate_support_record_partitions.py",
                "--gt_npz",
                str(gt_path),
                "--support_reference_npz",
                str(manual_path),
                "--pred_npz",
                str(gt_path),
                str(manual_path),
                str(direct_path),
                "--method_names",
                "gt",
                "manual",
                "direct",
                "--output_json",
                str(output_path),
                "--min_observed_plane_points",
                "1",
            ]
            with patch("sys.argv", argv):
                main()
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["duplicate_records_preserved"])
            self.assertEqual(
                [row["method"] for row in payload["methods"]],
                ["gt", "manual", "direct"],
            )
            self.assertEqual(
                payload["methods"][1]["duplicate_positive_record_count"], 2
            )
            self.assertTrue(output_path.with_suffix(".csv").is_file())

    def test_legacy_cache_coordinates_require_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez_compressed(
                path,
                point_plane_ids=np.asarray([0], dtype=np.int32),
                plane_normals=np.asarray([[1, 0, 0]], dtype=np.float32),
                source_views=np.asarray([2], dtype=np.int32),
                pixel_xy=np.asarray([[3, 4]], dtype=np.int32),
            )
            with self.assertRaisesRegex(ValueError, "missing coordinate convention"):
                load_keyed_prediction(path)
            loaded = load_keyed_prediction(path, allow_legacy_cache_xy=True)
            self.assertTrue(loaded["legacy_coordinate_override"])

    def test_unique_cache_labels_expand_to_repeated_records(self):
        cache_keys = pack_registry_keys(
            np.asarray([0, 1]), np.asarray([[2, 3], [4, 5]]), "cache"
        )
        record_keys = np.asarray(
            [cache_keys[1], cache_keys[0], cache_keys[1]], dtype=np.int64
        )
        labels, matched = map_unique_labels(
            cache_keys, np.asarray([7, 9]), record_keys, "cache"
        )
        self.assertEqual(labels.tolist(), [9, 7, 9])
        self.assertTrue(matched.all())

    def test_repeated_conflicting_labels_are_counted_not_dropped(self):
        keys = np.asarray([10, 10, 11, 11, 11], dtype=np.int64)
        labels = np.asarray([2, 3, 4, 4, 4], dtype=np.int32)
        stats = repeated_key_diagnostics(keys, labels)
        self.assertEqual(stats["positive_record_count"], 5)
        self.assertEqual(stats["unique_positive_key_count"], 2)
        self.assertEqual(stats["conflicting_positive_key_count"], 1)
        self.assertEqual(stats["conflicting_positive_record_count"], 2)

    def test_per_plane_audit_exposes_dominant_gt_and_views(self):
        labels = np.asarray([0, 0, 0, 1], dtype=np.int32)
        gt = np.asarray([2, 2, 3, 3], dtype=np.int32)
        rows = per_plane_rows(
            labels,
            gt,
            np.asarray([0, 1, 1, 1]),
            np.asarray([10, 11, 12, 13]),
            np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            np.asarray(
                [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]],
                dtype=np.float32,
            ),
        )
        self.assertEqual(rows[0]["dominant_gt_plane_id"], 2)
        self.assertAlmostEqual(rows[0]["dominant_gt_purity"], 2 / 3)
        self.assertEqual(rows[0]["view_count"], 2)
        self.assertEqual(rows[1]["dominant_gt_plane_id"], 3)


if __name__ == "__main__":
    unittest.main()
