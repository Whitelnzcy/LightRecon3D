import unittest

import numpy as np

from guided_plane_ransac import (
    MECHANISM_MODES,
    METHOD_BY_MODE,
    best_support_group_for_plane,
    build_support_groups,
    build_guided_hypotheses,
    fit_weighted_plane,
    guided_sequential_plane_ransac,
    map_support_records_to_cache,
    pack_support_groups,
    refit_global_inliers,
)


class GuidedPlaneRansacTests(unittest.TestCase):
    def test_mechanism_modes_are_orthogonal_and_preserve_historical_b4(self):
        self.assertEqual(
            MECHANISM_MODES["proposal_consensus"],
            {
                "proposal_guidance": True,
                "consensus_guidance": True,
                "refit_guidance": False,
            },
        )
        self.assertEqual(
            MECHANISM_MODES["all"],
            {
                "proposal_guidance": True,
                "consensus_guidance": True,
                "refit_guidance": True,
            },
        )
        self.assertEqual(
            METHOD_BY_MODE["proposal_consensus"], "learning_guided_ransac_cc"
        )

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

    def test_support_groups_collapse_repeats_but_preserve_conflicts(self):
        cache = {
            "points": np.zeros((2, 3), np.float32),
            "confidence": np.ones(2, np.float32),
            "view_indices": np.asarray([0, 0], np.int32),
            "pixel_xy": np.asarray([[0, 0], [1, 0]], np.int32),
        }
        support = {
            "view_indices": np.asarray([0, 0, 0, 0], np.int32),
            "pixel_xy": np.asarray([[0, 0], [0, 0], [0, 0], [1, 0]], np.int32),
            "labels": np.asarray([4, 4, 9, 9], np.int32),
        }
        groups, diagnostics = build_support_groups(cache, support)
        self.assertEqual([group["source_plane_id"] for group in groups], [4, 9])
        self.assertEqual(groups[0]["support_indices"].tolist(), [0])
        self.assertEqual(groups[1]["support_indices"].tolist(), [0, 1])
        self.assertEqual(diagnostics["unique_label_cache_memberships"], 3)

    def test_best_support_group_uses_one_coherent_label(self):
        horizontal = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32
        )
        vertical = np.asarray(
            [[2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1]], np.float32
        )
        points = np.vstack((horizontal, vertical))
        groups = [
            {"source_plane_id": 4, "support_indices": np.arange(4)},
            {"source_plane_id": 9, "support_indices": np.arange(4, 8)},
        ]
        cache_indices, group_indices, source_ids = pack_support_groups(groups)
        group_index, source_id, count = best_support_group_for_plane(
            points,
            cache_indices,
            group_indices,
            source_ids,
            np.ones(len(points), dtype=bool),
            np.asarray([0, 0, 1], np.float32),
            0.0,
            0.01,
        )
        self.assertEqual((group_index, source_id, count), (0, 4, 4))

    def test_support_guided_refit_reduces_selected_support_residual(self):
        yy, xx = np.mgrid[:4, :4]
        support_plane = np.column_stack(
            (xx.ravel(), yy.ravel(), np.zeros(xx.size))
        ).astype(np.float32)
        yy2, xx2 = np.mgrid[:6, :6]
        competing = np.column_stack(
            (xx2.ravel(), yy2.ravel(), 0.18 * xx2.ravel() + 0.1)
        ).astype(np.float32)
        points = np.vstack((support_plane, competing))
        cache = {
            "points": points,
            "confidence": np.ones(len(points), np.float32),
        }
        indices = np.arange(len(points), dtype=np.int64)
        base_normal, base_offset, base_diagnostics = refit_global_inliers(
            cache, indices
        )
        guided_normal, guided_offset, guided_diagnostics = refit_global_inliers(
            cache,
            indices,
            support_indices=np.arange(len(support_plane), dtype=np.int64),
            support_refit_weight=20.0,
        )
        base_residual = np.abs(
            np.sum(support_plane * base_normal[None], axis=1) + base_offset
        ).mean()
        guided_residual = np.abs(
            np.sum(support_plane * guided_normal[None], axis=1) + guided_offset
        ).mean()
        self.assertLess(float(guided_residual), float(base_residual))
        self.assertEqual(base_diagnostics["support_guided_inliers"], 0)
        self.assertEqual(
            guided_diagnostics["support_guided_inliers"], len(support_plane)
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
