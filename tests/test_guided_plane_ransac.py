import unittest

import numpy as np

from guided_plane_ransac import (
    build_guided_hypotheses,
    fit_weighted_plane,
    guided_sequential_plane_ransac,
    map_support_records_to_cache,
)


class GuidedPlaneRansacTests(unittest.TestCase):
    def test_exact_mapping_preserves_competing_conflict_records(self):
        cache_indices, labels, diagnostics = map_support_records_to_cache(
            np.asarray([0, 0], np.int32),
            np.asarray([[0, 0], [1, 0]], np.int32),
            np.asarray([0, 0, 0, 0], np.int32),
            np.asarray([[0, 0], [0, 0], [0, 0], [1, 0]], np.int32),
            np.asarray([4, 4, 9, 9], np.int32),
        )
        self.assertEqual(cache_indices.tolist(), [0, 0, 0, 1])
        self.assertEqual(labels.tolist(), [4, 4, 9, 9])
        self.assertEqual(diagnostics["conflicting_support_keys"], 1)
        self.assertEqual(diagnostics["conflicting_support_records"], 3)
        self.assertEqual(diagnostics["duplicate_positive_support_records"], 2)
        self.assertTrue(
            diagnostics["duplicate_conflicts_preserved_as_competing_hypotheses"]
        )

    def test_confidence_weighted_refit_downweights_bad_point(self):
        yy, xx = np.mgrid[:5, :5]
        plane = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(25)))
        points = np.vstack((plane, [[2.0, 2.0, 4.0]])).astype(np.float32)
        weights = np.ones(len(points), np.float32)
        weights[-1] = 1e-4
        normal, offset, _ = fit_weighted_plane(points, weights)
        self.assertGreater(abs(float(normal[2])), 0.999)
        self.assertLess(abs(float(offset)), 0.01)

    def test_learned_proposals_drive_two_global_consensus_planes(self):
        values = np.linspace(0.0, 0.9, 10, dtype=np.float32)
        yy, xx = np.meshgrid(values, values, indexing="ij")
        horizontal = np.column_stack(
            (xx.ravel(), yy.ravel(), np.zeros(xx.size, np.float32))
        )
        zz, yy2 = np.meshgrid(values, values, indexing="ij")
        vertical = np.column_stack(
            (np.full(zz.size, 2.0, np.float32), yy2.ravel(), zz.ravel())
        )
        points = np.vstack((horizontal, vertical)).astype(np.float32)
        cache = {
            "points": points,
            "colors": np.zeros((len(points), 3), np.uint8),
            "confidence": np.ones(len(points), np.float32),
            "view_indices": np.zeros(len(points), np.int32),
            "pixel_xy": np.column_stack(
                (np.arange(len(points), dtype=np.int32), np.zeros(len(points), np.int32))
            ),
        }
        selected = np.r_[np.arange(0, 100, 2), np.arange(100, 200, 2)]
        support = {
            "labels": np.r_[np.zeros(50, np.int32), np.ones(50, np.int32)],
            "view_indices": cache["view_indices"][selected],
            "pixel_xy": cache["pixel_xy"][selected],
        }
        candidates, proposal = build_guided_hypotheses(
            cache,
            support,
            distance_threshold=0.01,
            proposal_iterations=8,
            proposal_min_points=20,
            proposal_min_inlier_ratio=0.95,
            proposal_max_points=100,
            seed=3,
        )
        self.assertEqual(proposal["accepted_hypotheses"], 2)
        supports, extraction = guided_sequential_plane_ransac(
            cache,
            candidates,
            distance_threshold=0.01,
            min_inliers=50,
            cluster_radius=0.15,
            min_component_points=40,
            max_planes=4,
            seed=3,
            hypothesis_max_points=200,
            component_exact_max_points=1000,
            support_score_weight=1.0,
            fallback_iterations=0,
        )
        self.assertEqual(len(supports), 2)
        # Ten vertical-plane points also lie on z=0.  Sequential consensus
        # removes those infinite-plane inliers in the first round, exactly as
        # the retained global RANSAC baseline does.
        self.assertEqual(sorted(len(value) for value in supports), [90, 100])
        self.assertEqual(extraction["guided_components"], 2)
        self.assertEqual(extraction["fallback_components"], 0)


if __name__ == "__main__":
    unittest.main()
