import unittest

import numpy as np

from analyze_provenance_view_bias import (
    evaluate_plane,
    largest_connected_view_component,
    tangent_basis,
)


class ProvenanceViewBiasTest(unittest.TestCase):
    def test_tangent_basis_is_orthonormal(self):
        normal = np.asarray([0.2, -0.4, 0.9], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        first, second = tangent_basis(normal)
        self.assertAlmostEqual(float(first @ normal), 0.0, places=7)
        self.assertAlmostEqual(float(second @ normal), 0.0, places=7)
        self.assertAlmostEqual(float(first @ second), 0.0, places=7)
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=7)
        self.assertAlmostEqual(float(np.linalg.norm(second)), 1.0, places=7)

    def test_largest_connected_view_component_uses_shared_cells(self):
        cells = np.asarray([0] * 8 + [0] * 8 + [1] * 8 + [1] * 8 + [2] * 8)
        views = np.asarray([0] * 8 + [1] * 8 + [1] * 8 + [2] * 8 + [4] * 8)
        component = largest_connected_view_component(cells, views, min_cell_view_points=8)
        np.testing.assert_array_equal(component, np.asarray([0, 1, 2]))

    def test_overlap_control_recovers_view_effect(self):
        rng = np.random.default_rng(7)
        point_rows = []
        metric_rows = []
        pixels = []
        views = []
        view_bias = {0: -0.03, 1: 0.01, 2: 0.02}
        index = 0
        for cell_x in range(5):
            for cell_y in range(4):
                spatial_warp = 0.015 * np.sin(cell_x) - 0.01 * np.cos(cell_y)
                for view in range(3):
                    for repeat in range(30):
                        u = cell_x * 0.1 + rng.uniform(0.005, 0.095)
                        v = cell_y * 0.1 + rng.uniform(0.005, 0.095)
                        z = spatial_warp + view_bias[view] + rng.normal(0.0, 0.001)
                        point_rows.append([u, v, z])
                        metric_rows.append([u, v, 0.0])
                        pixels.append([index % 640, index // 640])
                        views.append(view)
                        index += 1
        result = evaluate_plane(
            np.asarray(point_rows),
            np.asarray(metric_rows),
            np.asarray(pixels),
            np.asarray(views),
            np.asarray([0.0, 0.0, 1.0]),
            cell_size_m=0.1,
            min_cell_view_points=8,
            min_connected_views=3,
            view_ridge_count=8.0,
        )
        self.assertIsNotNone(result)
        self.assertLess(result["rmse_ratio"], 0.25)
        self.assertGreater(result["incremental_variance_explained"], 0.9)

    def test_overlap_control_does_not_invent_view_effect(self):
        rng = np.random.default_rng(11)
        point_rows = []
        metric_rows = []
        pixels = []
        views = []
        index = 0
        for cell_x in range(5):
            for cell_y in range(4):
                spatial_warp = 0.02 * np.sin(cell_x + cell_y)
                for view in range(3):
                    for repeat in range(30):
                        u = cell_x * 0.1 + rng.uniform(0.005, 0.095)
                        v = cell_y * 0.1 + rng.uniform(0.005, 0.095)
                        z = spatial_warp + rng.normal(0.0, 0.003)
                        point_rows.append([u, v, z])
                        metric_rows.append([u, v, 0.0])
                        pixels.append([index % 640, index // 640])
                        views.append(view)
                        index += 1
        result = evaluate_plane(
            np.asarray(point_rows),
            np.asarray(metric_rows),
            np.asarray(pixels),
            np.asarray(views),
            np.asarray([0.0, 0.0, 1.0]),
            cell_size_m=0.1,
            min_cell_view_points=8,
            min_connected_views=3,
            view_ridge_count=64.0,
        )
        self.assertIsNotNone(result)
        self.assertGreater(result["rmse_ratio"], 0.95)


if __name__ == "__main__":
    unittest.main()
