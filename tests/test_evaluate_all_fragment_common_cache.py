import unittest

import numpy as np

from evaluate_all_fragment_common_cache import (
    assignment_to_primitives,
    build_tangent_bounds,
    expand_plane_seeds,
)


class AllFragmentCommonCacheTest(unittest.TestCase):
    def test_small_plane_is_dropped_and_ids_are_compact(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        source = np.asarray([5, 5, 5, 9], dtype=np.int32)
        assignment, normals, _, counts, source_ids = assignment_to_primitives(
            points, source, min_points_per_plane=3
        )
        np.testing.assert_array_equal(assignment, [0, 0, 0, -1])
        self.assertEqual(normals.shape, (1, 3))
        np.testing.assert_array_equal(counts, [3])
        self.assertEqual(source_ids, [5])

    def test_seed_expansion_assigns_nearby_plane_points(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.01],
                [1.0, 0.0, -0.01],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.2],
            ],
            dtype=np.float32,
        )
        assignment, normals, _, counts = expand_plane_seeds(
            points,
            np.asarray([[0.0, 0.0, 1.0]]),
            np.asarray([0.0]),
            distance_threshold=0.03,
            refit_iterations=1,
            min_points_per_plane=3,
        )
        np.testing.assert_array_equal(assignment, [0, 0, 0, -1])
        self.assertEqual(normals.shape, (1, 3))
        np.testing.assert_array_equal(counts, [3])

    def test_tangent_bounds_reject_coplanar_points_outside_seed_support(self):
        seed_points = np.asarray(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [-0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        points = np.vstack([seed_points, [[3.0, 0.0, 0.0]]]).astype(np.float32)
        bounds = build_tangent_bounds(
            seed_points,
            np.zeros((len(seed_points),), dtype=np.int32),
            np.asarray([[0.0, 0.0, 1.0]]),
            trim_quantile=0.0,
        )
        unbounded, _, _, _ = expand_plane_seeds(
            points,
            np.asarray([[0.0, 0.0, 1.0]]),
            np.asarray([0.0]),
            distance_threshold=0.03,
            refit_iterations=0,
            min_points_per_plane=3,
        )
        bounded, _, _, _ = expand_plane_seeds(
            points,
            np.asarray([[0.0, 0.0, 1.0]]),
            np.asarray([0.0]),
            distance_threshold=0.03,
            refit_iterations=0,
            min_points_per_plane=3,
            tangent_bounds=bounds,
            bound_margin=0.0,
        )
        self.assertEqual(int(unbounded[-1]), 0)
        self.assertEqual(int(bounded[-1]), -1)


if __name__ == "__main__":
    unittest.main()
