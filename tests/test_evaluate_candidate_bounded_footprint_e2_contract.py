import unittest
from unittest import mock

import numpy as np

import evaluate_candidate_bounded_footprint_e2_contract as e2
from audit_support_bounded_evaluator_contract import matching_rows
from evaluate_global_plane_baselines import evaluate_arrays


class CandidateBoundedFootprintE2Test(unittest.TestCase):
    def test_deterministic_tangent_basis_and_axis_tie(self):
        status, n, u, v = e2.deterministic_tangent_basis([0, 0, 1])
        self.assertEqual(status, "valid")
        np.testing.assert_array_equal(n, [0, 0, 1])
        # x and y axes tie; x wins.
        np.testing.assert_array_equal(u, [1, 0, 0])
        np.testing.assert_array_equal(v, [0, 1, 0])

    def test_tangent_sign_canonicalization(self):
        status, _, u, _ = e2.deterministic_tangent_basis([0.2, 0.9, 0.3])
        self.assertEqual(status, "valid")
        dominant = int(np.argmax(np.abs(u)))
        self.assertGreaterEqual(u[dominant], 0.0)

    def test_duplicate_projection_keeps_smallest_point_order(self):
        points = np.asarray(
            [[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]],
            dtype=np.float64,
        )
        projected, original = e2.unique_seed_projection(
            points,
            np.asarray([0, 1, 2, 3]),
            np.asarray([1, 0, 0], dtype=np.float64),
            np.asarray([0, 1, 0], dtype=np.float64),
        )
        self.assertEqual(len(projected), 3)
        self.assertIn(0, original.tolist())
        self.assertNotIn(1, original.tolist())

    def test_delaunay_degeneracy_has_no_hull_fallback(self):
        points = np.column_stack(
            (np.arange(10, dtype=np.float64), np.zeros(10))
        )
        result = e2.accepted_delaunay_triangles(points, 1.0)
        self.assertIsNone(result["triangulation"])
        self.assertEqual(result["triangle_count_accepted"], 0)
        self.assertIn("no_convex_hull_fallback", result["status"])

    def test_disk_radius_clipping(self):
        radii = e2.local_disk_radii(
            np.asarray([0.01, 1.0, 10.0]), local_scale=1.0
        )
        np.testing.assert_allclose(radii, [0.5, 1.5, 2.0])

    def test_tau_exact_formula(self):
        seed = np.column_stack(
            (
                np.zeros(100),
                np.zeros(100),
                np.linspace(0.0, 0.0099, 100),
            )
        )
        result = e2.adaptive_normal_threshold(
            seed, np.asarray([0, 0, 1]), 0.0, 0.02
        )
        residual = seed[:, 2]
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        q99 = np.quantile(residual, 0.99, method="linear")
        expected = min(
            0.03, 0.01, max(1e-6, q99 + 3.0 * 1.4826 * mad)
        )
        self.assertEqual(result["tau"], expected)

    def test_seed_immutability_unique_acceptance_and_multi_rejection(self):
        seeds = np.asarray([0, 1, -1, -1, -1], dtype=np.int32)
        membership = np.asarray(
            [
                [False, True],
                [True, False],
                [True, False],
                [True, True],
                [False, False],
            ],
            dtype=bool,
        )
        result = e2.resolve_e2_assignment(seeds, membership)
        np.testing.assert_array_equal(
            result["final_assignment"], [0, 1, 0, -1, -1]
        )
        self.assertTrue(result["conflict_rejected_mask"][3])
        self.assertEqual(result["seed_deletion_count"], 0)
        self.assertEqual(result["seed_reassignment_count"], 0)
        self.assertEqual(result["final_multi_assignment_count"], 0)

    def test_rejected_candidate_is_not_second_reassigned(self):
        seeds = np.asarray([-1], dtype=np.int32)
        membership = np.asarray([[True, True]], dtype=bool)
        result = e2.resolve_e2_assignment(seeds, membership)
        self.assertEqual(int(result["final_assignment"][0]), -1)
        self.assertEqual(
            int(result["accepted_candidate_assignment"][0]), -1
        )

    def test_cross_seed_intrusion_remains_visible(self):
        seeds = np.asarray([0, 1], dtype=np.int32)
        raw = np.asarray([[False, True], [True, False]], dtype=bool)
        cross = [
            int((raw[:, p] & (seeds >= 0) & (seeds != p)).sum())
            for p in range(2)
        ]
        self.assertEqual(cross, [1, 1])
        result = e2.resolve_e2_assignment(seeds, raw)
        np.testing.assert_array_equal(result["final_assignment"], seeds)

    def test_freeze_barrier_blocks_gt_scoring(self):
        barrier = e2.FreezeBarrier(1, {})
        with self.assertRaises(RuntimeError):
            barrier.assert_gt_scoring_allowed()
        barrier.record_prediction("scene", "hash")
        barrier.freeze_all()
        barrier.assert_gt_scoring_allowed()

    def test_zero_fit_calls_and_no_fit_function_use(self):
        self.assertEqual(e2.PLANE_FIT_CALL_COUNT, 0)
        self.assertEqual(e2.WEIGHTED_FIT_PLANE_CALL_COUNT, 0)
        self.assertEqual(e2.REFIT_CALL_COUNT, 0)
        self.assertEqual(e2.RANSAC_CALL_COUNT, 0)
        with mock.patch(
            "evaluate_canonical_plane_bias_oracle.weighted_fit_plane",
            side_effect=AssertionError("fit forbidden"),
        ):
            result = e2.resolve_e2_assignment(
                np.asarray([-1]), np.asarray([[True]], dtype=bool)
            )
        self.assertEqual(int(result["final_assignment"][0]), 0)

    def test_exact_matcher_and_metric_reuse(self):
        self.assertIs(e2.E2_MATCHER, matching_rows)
        self.assertIs(e2.E2_METRIC, evaluate_arrays)


if __name__ == "__main__":
    unittest.main()
