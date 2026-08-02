import unittest

import numpy as np

from evaluate_bias_aware_fragment_association import (
    average_precision_lower,
    binary_auc_lower,
)
from evaluate_canonical_plane_bias_oracle import (
    fit_canonical_view_bias,
    weighted_fit_plane,
)


class BiasAwareFragmentAssociationTest(unittest.TestCase):
    def test_rank_metrics_prefer_lower_positive_cost(self):
        labels = np.asarray([True, True, False, False])
        perfect = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(binary_auc_lower(labels, perfect), 1.0)
        self.assertAlmostEqual(average_precision_lower(labels, perfect), 1.0)

    def test_view_bias_explains_cross_view_layer_offset(self):
        rng = np.random.default_rng(13)
        xy0 = rng.uniform(-1.0, 1.0, size=(500, 2))
        xy1 = rng.uniform(-1.0, 1.0, size=(500, 2))
        first = np.column_stack((xy0, rng.normal(-0.03, 0.001, size=500)))
        second = np.column_stack((xy1, rng.normal(0.03, 0.001, size=500)))
        points = np.concatenate((first, second))
        views = np.asarray([0] * len(first) + [1] * len(second))
        normal, offset = weighted_fit_plane(points)
        pooled = float(np.sqrt(np.mean((points @ normal + offset) ** 2)))
        model = fit_canonical_view_bias(
            points,
            views,
            np.ones((len(points),)),
            bias_ridge=0.0,
            use_reliability=False,
        )
        bias = np.asarray([model["view_biases"][int(view)] for view in views])
        corrected = float(
            np.sqrt(
                np.mean(
                    (
                        points @ model["normal"]
                        + model["offset"]
                        - bias
                    )
                    ** 2
                )
            )
        )
        self.assertLess(corrected, pooled * 0.1)

    def test_view_bias_cannot_explain_two_layers_in_same_view(self):
        rng = np.random.default_rng(17)
        xy0 = rng.uniform(-1.0, 1.0, size=(500, 2))
        xy1 = rng.uniform(-1.0, 1.0, size=(500, 2))
        first = np.column_stack((xy0, rng.normal(-0.03, 0.001, size=500)))
        second = np.column_stack((xy1, rng.normal(0.03, 0.001, size=500)))
        points = np.concatenate((first, second))
        views = np.zeros((len(points),), dtype=np.int64)
        model = fit_canonical_view_bias(
            points,
            views,
            np.ones((len(points),)),
            bias_ridge=0.0,
            use_reliability=False,
        )
        bias = np.asarray([model["view_biases"][0]] * len(points))
        corrected = float(
            np.sqrt(
                np.mean(
                    (
                        points @ model["normal"]
                        + model["offset"]
                        - bias
                    )
                    ** 2
                )
            )
        )
        self.assertGreater(corrected, 0.02)


if __name__ == "__main__":
    unittest.main()
