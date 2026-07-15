import json
import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_structural_lines import (
    associate_line_with_planes,
    build_plane_label_maps,
    lift_line_to_3d,
    load_plane_label_records,
    sample_line_pixels,
    write_edge_ply,
)
import extract_structural_lines


def fake_views(height=5, width=8):
    return {
        "view.png": {
            "alignment_view_index": 0,
            "image_path": "view.png",
            "points": np.zeros((height, width, 3), dtype=np.float32),
            "colors": np.zeros((height, width, 3), dtype=np.uint8),
            "conf": np.ones((height, width), dtype=np.float32),
        }
    }


class StructuralLineTest(unittest.TestCase):
    def test_plane_label_rasterization_keeps_agreement_and_drops_conflict(self):
        records = {
            "labels": np.asarray([2, 2, 3, 4, -1], dtype=np.int32),
            "view_indices": np.asarray([0, 0, 0, 0, 0], dtype=np.int32),
            "pixel_xy": np.asarray(
                [[1, 1], [1, 1], [2, 1], [2, 1], [3, 1]], dtype=np.int32
            ),
        }
        maps, stats = build_plane_label_maps(fake_views(), records)
        self.assertEqual(int(maps[0][1, 1]), 2)
        self.assertEqual(int(maps[0][1, 2]), -1)
        self.assertEqual(stats["duplicate_positive_records"], 2)
        self.assertEqual(stats["conflicting_keys"], 1)
        self.assertEqual(stats["assigned_keys"], 1)

    def test_load_plane_records_accepts_full_cache_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "planes.npz"
            np.savez_compressed(
                path,
                point_plane_ids=np.asarray([1, -1], dtype=np.int32),
                source_views=np.asarray([0, 1], dtype=np.int32),
                pixel_xy=np.asarray([[2, 3], [4, 5]], dtype=np.int32),
                pixel_coordinate_order=np.asarray("xy"),
                pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
            )
            records = load_plane_label_records(path)
            self.assertEqual(records["labels"].tolist(), [1, -1])
            self.assertEqual(records["view_indices"].tolist(), [0, 1])
            self.assertEqual(records["pixel_xy"].tolist(), [[2, 3], [4, 5]])

    def test_line_sampling_is_xy_and_clipped(self):
        xy = sample_line_pixels([-2.0, 1.0, 9.0, 1.0], 4, 6, step_px=1.0)
        self.assertTrue(np.all(xy[:, 0] >= 0))
        self.assertTrue(np.all(xy[:, 0] < 6))
        self.assertTrue(np.all(xy[:, 1] == 1))
        self.assertEqual(xy[0].tolist(), [0, 1])
        self.assertEqual(xy[-1].tolist(), [5, 1])

    def test_plane_side_association_detects_boundary(self):
        labels = np.full((7, 10), -1, dtype=np.int32)
        labels[:3] = 4
        labels[4:] = 9
        left, right, left_fraction, right_fraction, code = associate_line_with_planes(
            [1.0, 3.0, 8.0, 3.0], labels, step_px=1.0, side_offset_px=2.0
        )
        self.assertEqual({left, right}, {4, 9})
        self.assertEqual(code, 2)
        self.assertEqual(left_fraction, 1.0)
        self.assertEqual(right_fraction, 1.0)

    def test_3d_lift_uses_longest_contiguous_valid_run(self):
        height, width = 3, 12
        points = np.zeros((height, width, 3), dtype=np.float32)
        for x in range(width):
            points[1, x] = [float(x), 0.0, 2.0]
        confidence = np.ones((height, width), dtype=np.float32)
        confidence[1, 5] = 0.0
        lifted = lift_line_to_3d(
            [0.0, 1.0, 11.0, 1.0],
            points,
            confidence,
            min_conf=0.5,
            step_px=1.0,
            min_valid_samples=4,
        )
        self.assertIsNotNone(lifted)
        self.assertEqual(lifted["retained_sample_count"], 6)
        self.assertGreater(lifted["length_3d"], 4.5)
        self.assertLess(lifted["fit_residual_mean"], 1e-6)

    def test_edge_ply_contains_explicit_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lines.ply"
            write_edge_ply(
                path,
                np.asarray([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32),
                np.asarray([[255, 0, 0]], dtype=np.uint8),
            )
            text = path.read_text(encoding="ascii")
            self.assertIn("element vertex 2", text)
            self.assertIn("element edge 1", text)
            self.assertTrue(text.rstrip().endswith("0 1"))

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "opencv is not installed")
    def test_synthetic_cli_writes_complete_artifact_set(self):
        height, width = 64, 64
        yy, xx = np.indices((height, width), dtype=np.float32)
        points = np.stack((xx * 0.01, yy * 0.01, np.ones_like(xx)), axis=-1)
        colors = np.zeros((height, width, 3), dtype=np.uint8)
        colors[31:34, 6:58] = 255
        confidence = np.full((height, width), 2.0, dtype=np.float32)
        pixel_xy = np.stack((xx.astype(np.int32), yy.astype(np.int32)), axis=-1)
        plane_ids = np.where(yy < 32, 4, 9).astype(np.int32).reshape(-1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "cache.npz"
            prediction_path = root / "planes.npz"
            output_dir = root / "output"
            np.savez_compressed(
                cache_path,
                schema_version=np.asarray(1, dtype=np.int32),
                points=points.reshape(-1, 3),
                colors=colors.reshape(-1, 3),
                confidence=confidence.reshape(-1),
                view_indices=np.zeros(height * width, dtype=np.int32),
                pixel_xy=pixel_xy.reshape(-1, 2),
                pixel_coordinate_order=np.asarray("xy"),
                pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
                scene_key=np.asarray("synthetic"),
                dust3r_global_alignment_loss=np.asarray(0.1, dtype=np.float32),
                dust3r_view_registry_json=np.asarray(
                    json.dumps(
                        [
                            {
                                "alignment_view_index": 0,
                                "image_path": "synthetic.png",
                                "points_hw": [height, width],
                            }
                        ]
                    )
                ),
            )
            np.savez_compressed(
                prediction_path,
                point_plane_ids=plane_ids,
                source_views=np.zeros(height * width, dtype=np.int32),
                pixel_xy=pixel_xy.reshape(-1, 2),
                pixel_coordinate_order=np.asarray("xy"),
                pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
            )
            argv = [
                "extract_structural_lines.py",
                "--global_cloud_npz",
                str(cache_path),
                "--plane_prediction_npz",
                str(prediction_path),
                "--output_dir",
                str(output_dir),
                "--min_length_px",
                "12",
                "--min_conf",
                "1",
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                extract_structural_lines.main()

            manifest = json.loads(
                (output_dir / "structural_lines_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertGreater(manifest["line_count"], 0)
            self.assertEqual(manifest["coordinate_order"], "xy")
            self.assertEqual(manifest["views"][0]["alignment_view_index"], 0)
            self.assertTrue((output_dir / "structural_lines.npz").is_file())
            self.assertTrue((output_dir / "structural_lines.ply").is_file())
            self.assertTrue((output_dir / "overlays" / "view_000_lines.png").is_file())


if __name__ == "__main__":
    unittest.main()
