import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapt_plane_dust3r_to_point_cache import (
    closest_reference_pose,
    refit_predictions,
    render_global_label_maps,
)


class PlaneDust3rPointCacheAdapterTests(unittest.TestCase):
    def test_closest_reference_pose_uses_rotation_then_translation(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[0, :3, :3] *= 0.5
        poses[1, :3, 3] = [2, 0, 0]
        poses[2, :3, 3] = [1, 0, 0]
        self.assertEqual(closest_reference_pose(poses), 2)

    def test_render_retains_global_wall_ids_across_views(self):
        node_info = {
            "global_plane_info": [
                {"index": 0, "pparam": [1, 0, 0, 1]},
                {"index": 1, "pparam": [0, 0, 1, 1]},
            ],
            "floor_pparam": [0, -1, 0, 1],
            "ceiling_pparam": [0, 1, 0, 1],
            "planes": {"0": [1, 0]},
            "wall_relationship": [[1]],
        }
        detection = {"0": {"line": []}}
        poses = np.eye(4)[None]

        def reconstruction(*_args):
            upper = [[0, 100], [320, 100], [640, 100]]
            lower = [[0, 300], [320, 300], [640, 300]]
            return upper, lower, [np.ones(4), np.ones(4)]

        maps, manifest = render_global_label_maps(
            node_info, detection, poses, reconstruction
        )
        labels = maps["0"]
        self.assertEqual(labels.shape, (288, 512))
        self.assertEqual(int(labels[160, 100]), 1)
        self.assertEqual(int(labels[160, 400]), 0)
        self.assertEqual(manifest["floor_plane_id"], 2)
        self.assertEqual(manifest["ceiling_plane_id"], 3)

    def test_refit_drops_tiny_planes_without_reindexing_identity(self):
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        labels = np.array([2, 2, 2, 5], dtype=np.int32)
        output, normals, offsets, counts = refit_predictions(points, labels, 3)
        np.testing.assert_array_equal(output, [2, 2, 2, -1])
        self.assertEqual(normals.shape, (3, 3))
        self.assertEqual(int(counts[2]), 3)
        self.assertAlmostEqual(float(abs(normals[2, 2])), 1.0)
        self.assertAlmostEqual(float(offsets[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
