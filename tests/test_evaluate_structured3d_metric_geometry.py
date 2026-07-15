import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_structured3d_metric_geometry import (
    apply_similarity,
    estimate_similarity,
    evaluate_predictions,
    oracle_view_switch,
)


class Structured3DMetricGeometryTest(unittest.TestCase):
    def test_umeyama_recovers_similarity(self):
        source = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0], [1.0, 2.0, 3.0],
        ])
        angle = 0.3
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        target = 2.5 * (source @ rotation.T) + np.asarray([1.0, -2.0, 0.5])
        transform = estimate_similarity(source, target)
        np.testing.assert_allclose(apply_similarity(source, transform), target, atol=1e-10)

    def test_exact_key_evaluation_and_reference_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_path = root / "gt.npz"
            reference_path = root / "reference.npz"
            moved_path = root / "moved.npz"
            gt_points = np.asarray([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0],
                [0.0, 2.0, 1.0], [0.0, 2.0, 2.0],
            ], dtype=np.float64)
            views = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], np.int32)
            pixels = np.column_stack((np.arange(8), np.zeros(8))).astype(np.int32)
            plane_ids = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], np.int32)
            np.savez_compressed(
                gt_path,
                metric_points_world_m=gt_points.astype(np.float32),
                metric_valid_mask=np.ones(8, bool),
                view_indices=views,
                pixel_xy=pixels,
                point_plane_ids=plane_ids,
                structured3d_world_plane_normals=np.asarray(
                    [[0, 0, 1], [1, 0, 0]], np.float32
                ),
                structured3d_world_plane_offsets_m=np.asarray([0, 0], np.float32),
                source_global_cloud_sha256=np.asarray(""),
            )
            angle = -0.2
            rotation = np.asarray([
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ])
            scale = 1.7
            translation = np.asarray([0.3, -0.4, 0.8])
            prediction = ((gt_points - translation) @ rotation) / scale
            np.savez_compressed(
                reference_path, points=prediction.astype(np.float32),
                view_indices=views, pixel_xy=pixels, method=np.asarray("original"),
            )
            moved = prediction.copy()
            moved[views == 1] += np.asarray([0.08, -0.03, 0.04])
            np.savez_compressed(
                moved_path, points=moved.astype(np.float32),
                source_views=views, pixel_xy=pixels, method=np.asarray("moved"),
            )

            result = evaluate_predictions(
                gt_path,
                [("original", reference_path), ("moved", moved_path)],
                "original",
                trim_quantile=1.0,
            )
            rows = {row["name"]: row for row in result["methods"]}
            self.assertLess(
                rows["original"]["independent_alignment_metrics"]
                ["correspondence_error_m"]["rmse"],
                1e-6,
            )
            self.assertGreater(
                rows["moved"]["independent_alignment_metrics"]
                ["correspondence_error_m"]["rmse"],
                1e-3,
            )
            self.assertEqual(rows["moved"]["matched_metric_points"], 8)
            self.assertEqual(result["source_global_cloud_sha256"], "")
            self.assertEqual(
                rows["moved"]["oracle_view_switch_upper_bound"]
                ["joint_pareto_oracle"]["selected_corrected_views"],
                [],
            )

    def test_oracle_view_switch_keeps_only_jointly_helpful_view(self):
        gt_points = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0],
            [0.0, 2.0, 1.0], [0.0, 2.0, 2.0],
        ], dtype=np.float64)
        views = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], np.int32)
        plane_ids = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], np.int32)
        match = {
            "gt_points": gt_points,
            "gt_views": views,
            "gt_plane_ids": plane_ids,
            "gt_mask": np.ones(8, bool),
        }
        gt = {
            "plane_normals": np.asarray([[0, 0, 1], [1, 0, 0]], np.float64),
            "plane_offsets": np.asarray([0, 0], np.float64),
        }
        reference = gt_points.copy()
        reference[views == 1, 0] += 0.1
        candidate = reference.copy()
        candidate[views == 1] = gt_points[views == 1]
        result = oracle_view_switch(candidate, match, reference, match, gt)
        oracle = result["joint_pareto_oracle"]
        self.assertEqual(oracle["selected_corrected_views"], [1])
        self.assertLess(oracle["delta_vs_reference"]["correspondence_rmse_m"], 0.0)
        self.assertLess(oracle["delta_vs_reference"]["gt_plane_mean_residual_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
