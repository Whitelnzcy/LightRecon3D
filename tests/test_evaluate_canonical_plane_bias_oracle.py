import unittest

import numpy as np

from evaluate_canonical_plane_bias_oracle import (
    fit_canonical_view_bias,
    plane_parameter_error,
    summarize,
    view_balanced_weights,
)


class CanonicalPlaneBiasOracleTest(unittest.TestCase):
    def test_view_balanced_weights_equalize_view_mass(self):
        views = np.asarray([0] * 100 + [1] * 10 + [2] * 4)
        confidence = np.ones((len(views),), dtype=np.float64)
        weights = view_balanced_weights(views, confidence)
        masses = [float(weights[views == view].sum()) for view in range(3)]
        np.testing.assert_allclose(masses, np.full((3,), 1.0 / 3.0))

    def test_view_bias_recovers_zero_mean_canonical_plane(self):
        rng = np.random.default_rng(5)
        points = []
        views = []
        biases = {0: -0.03, 1: 0.01, 2: 0.02}
        counts = {0: 600, 1: 180, 2: 80}
        for view, count in counts.items():
            xy = rng.uniform(-1.0, 1.0, size=(count, 2))
            z = biases[view] + rng.normal(0.0, 0.001, size=(count, 1))
            points.append(np.column_stack((xy, z)))
            views.extend([view] * count)
        points = np.concatenate(points)
        views = np.asarray(views)
        model = fit_canonical_view_bias(
            points,
            views,
            np.ones((len(points),)),
            bias_ridge=0.0,
            use_reliability=False,
        )
        angle, offset_error, _, _ = plane_parameter_error(
            model["normal"], model["offset"], np.asarray([0.0, 0.0, 1.0]), 0.0)
        self.assertLess(angle, 0.1)
        self.assertLess(offset_error, 0.001)
        recovered = model["view_biases"]
        for view, expected in biases.items():
            self.assertAlmostEqual(recovered[view], expected, delta=0.002)

    def test_reliability_downweights_noisy_view(self):
        rng = np.random.default_rng(9)
        points = []
        views = []
        for view, sigma in ((0, 0.001), (1, 0.001), (2, 0.05)):
            xy = rng.uniform(-1.0, 1.0, size=(500, 2))
            z = rng.normal(0.0, sigma, size=(500, 1))
            points.append(np.column_stack((xy, z)))
            views.extend([view] * len(xy))
        model = fit_canonical_view_bias(
            np.concatenate(points),
            np.asarray(views),
            np.ones((1500,)),
            bias_ridge=0.25,
            use_reliability=True,
        )
        self.assertLess(model["view_reliability"][2], model["view_reliability"][0])
        self.assertLess(model["view_reliability"][2], model["view_reliability"][1])

    def test_summary_reports_paired_win_fraction(self):
        rows = []
        for scene, base, candidate in (("a", 3.0, 2.0), ("b", 1.0, 1.5)):
            common = {
                "scene": scene,
                "plane_id": 0,
                "offset_error_m": base,
                "heldout_canonical_rmse_m": base,
                "heldout_bias_corrected_rmse_m": base,
                "view_bias_std_m": 0.0,
            }
            rows.append({**common, "method": "pooled_svd", "normal_error_deg": base})
            rows.append(
                {
                    **common,
                    "method": "candidate",
                    "normal_error_deg": candidate,
                    "offset_error_m": candidate,
                    "heldout_canonical_rmse_m": candidate,
                    "heldout_bias_corrected_rmse_m": candidate,
                }
            )
        result = summarize(rows)
        contrast = result["paired_contrasts"]["candidate_vs_pooled_svd"]
        self.assertEqual(
            contrast["normal_error_deg"]["win_fraction_lower_is_better"], 0.5)


if __name__ == "__main__":
    unittest.main()
