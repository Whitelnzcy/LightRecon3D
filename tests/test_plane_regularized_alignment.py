import unittest

import numpy as np
import torch

from plane_regularized_alignment import optimize_scene_with_plane_feedback


class FakeScene(torch.nn.Module):
    def __init__(self):
        super().__init__()
        xy = torch.tensor(
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
            dtype=torch.float32,
        )
        first = torch.cat((xy, torch.full((4, 1), -0.08)), dim=1)
        second = torch.cat((xy, torch.full((4, 1), 0.08)), dim=1)
        anchor = torch.stack((first, second))
        self.points = torch.nn.Parameter(anchor.clone())
        self.register_buffer("anchor", anchor)
        self.register_buffer("confidence", torch.ones((2, 2, 2)))
        self.imshapes = [(2, 2), (2, 2)]
        self.n_imgs = 2

    @property
    def device(self):
        return self.points.device

    def get_pts3d(self, raw=False):
        if raw:
            return self.points
        return [self.points[index].view(2, 2, 3) for index in range(2)]

    def get_conf(self, mode=None):
        return [self.confidence[index] for index in range(2)]

    def forward(self):
        return 0.1 + 5.0 * (self.points - self.anchor).square().mean()


class PlaneRegularizedAlignmentTest(unittest.TestCase):
    def test_multiview_plane_feedback_changes_scene_and_reduces_residual(self):
        scene = FakeScene()
        views = np.repeat(np.arange(2), 4)
        pixels = np.tile(np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]), (2, 1))
        labels = np.zeros((8,), dtype=np.int64)
        result = optimize_scene_with_plane_feedback(
            scene,
            views,
            pixels,
            labels,
            niter=80,
            lr=0.02,
            plane_weight=2.0,
            min_plane_views=2,
            min_plane_points=4,
            max_base_loss_increase=10.0,
            log_every=1000,
        )
        self.assertTrue(result["accepted"])
        self.assertLess(
            result["residual_after"]["mean"],
            result["residual_before"]["mean"],
        )
        self.assertGreater(result["support_displacement"]["mean"], 0.0)

    def test_single_view_plane_is_not_allowed_to_drive_alignment(self):
        scene = FakeScene()
        with self.assertRaisesRegex(ValueError, "distinct views"):
            optimize_scene_with_plane_feedback(
                scene,
                np.zeros((4,), dtype=np.int64),
                np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]),
                np.zeros((4,), dtype=np.int64),
                min_plane_views=2,
                min_plane_points=4,
            )

    def test_guard_rolls_back_when_base_objective_degrades(self):
        scene = FakeScene()
        original = scene.points.detach().clone()
        views = np.repeat(np.arange(2), 4)
        pixels = np.tile(np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]), (2, 1))
        result = optimize_scene_with_plane_feedback(
            scene,
            views,
            pixels,
            np.zeros((8,), dtype=np.int64),
            niter=40,
            lr=0.02,
            plane_weight=10.0,
            min_plane_views=2,
            min_plane_points=4,
            max_base_loss_increase=0.0,
            log_every=1000,
        )
        self.assertFalse(result["accepted"])
        torch.testing.assert_close(scene.points, original)


if __name__ == "__main__":
    unittest.main()
