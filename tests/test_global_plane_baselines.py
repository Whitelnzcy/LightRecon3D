import unittest

import numpy as np

from global_plane_baselines import euclidean_components, sequential_plane_ransac, supports_to_primitives


class GlobalPlaneBaselineTest(unittest.TestCase):
    def test_coplanar_disconnected_regions_stay_separate(self):
        rng = np.random.default_rng(4)
        left = np.column_stack((rng.uniform(-2, -1, 180), rng.uniform(-1, 1, 180), rng.normal(0, .001, 180)))
        right = np.column_stack((rng.uniform(1, 2, 180), rng.uniform(-1, 1, 180), rng.normal(0, .001, 180)))
        supports = sequential_plane_ransac(
            np.vstack((left, right)), distance_threshold=.01, iterations=100,
            min_inliers=200, cluster_radius=.3, min_component_points=100, seed=2)
        self.assertEqual(len(supports), 2)
        self.assertEqual(sorted(map(len, supports)), [180, 180])

    def test_components_drop_tiny_island(self):
        points = np.array([[0, 0, 0], [.01, 0, 0], [1, 1, 0]], np.float32)
        components = euclidean_components(points, radius=.05, min_points=2)
        self.assertEqual(len(components), 1)
        self.assertEqual(set(components[0]), {0, 1})

    def test_output_assignment_and_plane_equation(self):
        yy, xx = np.mgrid[:10, :10]
        points = np.column_stack((xx.ravel(), yy.ravel(), np.full(100, 2.0))).astype(np.float32)
        assignment, normals, offsets, counts = supports_to_primitives(points, [np.arange(100)])
        self.assertTrue(np.all(assignment == 0))
        self.assertEqual(counts.tolist(), [100])
        self.assertLess(float(np.abs(points @ normals[0] + offsets[0]).max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
