import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from recompute_public_plane_metrics import recompute


class RecomputePublicPlaneMetricsTest(unittest.TestCase):
    def test_recomputes_frozen_artifacts_without_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            points = np.asarray(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                np.float32,
            )
            gt_labels = np.asarray([0, 0, 1, 1], np.int32)
            normals = np.asarray([[0, 0, 1], [0, 0, 1]], np.float32)
            gt_path = root / "gt.npz"
            ordinary_path = root / "ordinary.npz"
            guided_path = root / "guided.npz"
            np.savez_compressed(
                gt_path,
                points=points,
                point_plane_ids=gt_labels,
                plane_normals=normals,
            )
            np.savez_compressed(
                ordinary_path,
                points=points,
                point_plane_ids=np.zeros(4, np.int32),
                plane_normals=normals[:1],
                plane_offsets=np.zeros(1, np.float32),
            )
            np.savez_compressed(
                guided_path,
                points=points,
                point_plane_ids=gt_labels,
                plane_normals=normals,
                plane_offsets=np.zeros(2, np.float32),
            )
            ledger = root / "batch_execution.json"
            ledger.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "item_0",
                                "scene_name": "scene_0",
                                "status": "pass",
                                "artifacts": {
                                    "point_aligned_gt": {"path": str(gt_path)},
                                    "global_ransac": {"path": str(ordinary_path)},
                                    "learning_guided_ransac": {
                                        "path": str(guided_path)
                                    },
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            result = recompute(ledger, output)
            summaries = {row["method"]: row for row in result["method_summary"]}
            self.assertEqual(result["passed_items"], 1)
            self.assertEqual(result["independent_scenes"], 1)
            self.assertAlmostEqual(
                summaries["ordinary_ransac"]["voi_nats_mean"], np.log(2.0)
            )
            self.assertEqual(
                summaries["learning_guided_ransac"]["voi_nats_mean"], 0.0
            )
            self.assertGreater(
                summaries["learning_guided_ransac"]["rand_index_mean"],
                summaries["ordinary_ransac"]["rand_index_mean"],
            )
            self.assertTrue((output / "public_plane_metrics.csv").is_file())
            self.assertTrue((output / "public_plane_metrics.md").is_file())

    def test_refuses_to_overwrite_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                recompute(root / "missing.json", output)


if __name__ == "__main__":
    unittest.main()
