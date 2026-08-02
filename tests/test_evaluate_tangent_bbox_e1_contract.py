import unittest
from unittest import mock

import numpy as np

import evaluate_all_fragment_common_cache
from evaluate_all_fragment_common_cache import (
    build_tangent_bounds,
    expand_plane_seeds,
)
from evaluate_tangent_bbox_e1_contract import (
    _convex_hull_area,
    _directions_consistent,
    expand_tangent_bbox_once_no_refit,
)


class TangentBBoxE1ContractTest(unittest.TestCase):
    def test_no_refit_path_matches_current_first_pass_assignment(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [1.0, 0.0, 0.1],
                [1.2, 0.0, 0.1],
                [1.0, 0.2, 0.1],
                [0.1, 0.1, 0.01],
                [1.1, 0.1, 0.11],
            ],
            dtype=np.float64,
        )
        seeds = np.asarray([0, 0, 0, 1, 1, 1, -1, -1], dtype=np.int32)
        normals = np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.float64)
        offsets = np.asarray([0.0, -0.1], dtype=np.float64)
        bounds = build_tangent_bounds(
            points, seeds, normals, trim_quantile=0.0
        )
        current_assignment, _, _, _ = expand_plane_seeds(
            points,
            normals,
            offsets,
            distance_threshold=0.02,
            refit_iterations=0,
            min_points_per_plane=1,
            tangent_bounds=bounds,
            bound_margin=0.0,
        )
        with mock.patch.object(
            evaluate_all_fragment_common_cache,
            "weighted_fit_plane",
            side_effect=AssertionError("refit must not be called"),
        ):
            result = expand_tangent_bbox_once_no_refit(
                points,
                seeds,
                normals,
                offsets,
                bounds,
                distance_threshold=0.02,
                bound_margin=0.0,
                min_points_per_plane=1,
            )
        np.testing.assert_array_equal(
            result["assignment"], current_assignment
        )
        np.testing.assert_allclose(result["plane_normals"], normals)
        np.testing.assert_allclose(result["plane_offsets"], offsets)
        self.assertEqual(result["refit_call_count"], 0)

    def test_conflict_uses_minimum_residual_then_lowest_id(self):
        points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
        seeds = np.asarray([-1], dtype=np.int32)
        normals = np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.float64)
        offsets = np.asarray([0.0, 0.0], dtype=np.float64)
        common_bound = {
            "origin": np.zeros(3),
            "axes": np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float64),
            "lower": np.asarray([-1, -1], dtype=np.float64),
            "upper": np.asarray([1, 1], dtype=np.float64),
        }
        result = expand_tangent_bbox_once_no_refit(
            points,
            seeds,
            normals,
            offsets,
            [common_bound, common_bound],
            distance_threshold=0.03,
            bound_margin=0.0,
            min_points_per_plane=1,
        )
        self.assertEqual(int(result["multiplicity"][0]), 2)
        self.assertEqual(int(result["assignment_source_plane_ids"][0]), 0)

    def test_dropped_plane_points_are_not_reassigned(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]], dtype=np.float64
        )
        seeds = np.asarray([-1, -1], dtype=np.int32)
        normals = np.asarray([[0, 0, 1]], dtype=np.float64)
        offsets = np.asarray([0.0], dtype=np.float64)
        bound = {
            "origin": np.zeros(3),
            "axes": np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float64),
            "lower": np.asarray([-1, -1], dtype=np.float64),
            "upper": np.asarray([1, 1], dtype=np.float64),
        }
        result = expand_tangent_bbox_once_no_refit(
            points,
            seeds,
            normals,
            offsets,
            [bound],
            distance_threshold=0.03,
            bound_margin=0.0,
            min_points_per_plane=3,
        )
        np.testing.assert_array_equal(result["assignment"], [-1, -1])
        self.assertEqual(len(result["plane_normals"]), 0)

    def test_convex_hull_records_degenerate_reason(self):
        area, status = _convex_hull_area(
            np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        )
        self.assertIsNone(area)
        self.assertIn("not_computable", status)

    def test_direction_consistency(self):
        self.assertTrue(_directions_consistent(-0.1, -0.2))
        self.assertTrue(_directions_consistent(0.0, -0.2))
        self.assertFalse(_directions_consistent(-0.1, 0.2))


if __name__ == "__main__":
    unittest.main()
