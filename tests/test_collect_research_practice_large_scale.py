import json
import tempfile
import unittest
from pathlib import Path

from collect_research_practice_large_scale import collect_bundle
from research_practice_batch import file_sha256


class LargeScaleCollectorTests(unittest.TestCase):
    def test_collects_scene_artifacts_and_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "summary": {"selected_scenes": 2},
                        "dataset_inventory": {"eligible_scenes": 2},
                    }
                ),
                encoding="utf-8",
            )
            materialization = root / "materialization.json"
            materialization.write_text(
                json.dumps(
                    {
                        "selection_plan_sha256": file_sha256(selection),
                        "summary": {"passed_items": 2},
                        "items": [],
                    }
                ),
                encoding="utf-8",
            )
            preflight = root / "preflight.json"
            preflight.write_text(json.dumps({"summary": {"passed_items": 2}}), encoding="utf-8")
            batch = root / "batch.json"
            batch.write_text(
                json.dumps(
                    {
                        "summary": {"items": 2, "passed_items": 1},
                        "items": [
                            {
                                "id": "a",
                                "scene_name": "scene_a",
                                "status": "pass",
                                "artifacts": {
                                    "global_cloud_cache": {"path": "/cache/a.npz"},
                                    "global_ransac": {"path": "/pred/ransac.npz"},
                                    "learning_guided_ransac": {"path": "/pred/guided.npz"},
                                },
                            },
                            {
                                "id": "b",
                                "scene_name": "scene_b",
                                "status": "fail",
                                "failure_stage": "alignment",
                                "error": "failed",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            batch_sha = file_sha256(batch)
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "source_batch_execution_sha256": batch_sha,
                        "per_scene": [
                            {
                                "item_id": "a",
                                "scene_name": "scene_a",
                                "ransac_pairwise_f1": 0.7,
                                "guided_pairwise_f1": 0.75,
                                "delta_pairwise_f1": 0.05,
                                "ransac_runtime_seconds": 2.0,
                                "guided_runtime_seconds": 3.0,
                            }
                        ],
                        "gate": {
                            "decision": "promote_learning_guided_ransac_final",
                            "diagnostics": {
                                "guided_scene_wins": 1,
                                "view_groups": 1,
                                "median_guided_f1_gain": 0.05,
                            },
                        },
                        "metric_summary": [
                            {
                                "metric": "pairwise_f1",
                                "ransac_mean": 0.7,
                                "guided_mean": 0.75,
                                "mean_delta_guided_minus_ransac": 0.05,
                                "valid_scene_pairs": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            visualization = root / "visualization.json"
            visualization.write_text(
                json.dumps(
                    {
                        "source_batch_execution_sha256": batch_sha,
                        "contact_sheet_png": "/figures/all.png",
                        "scenes": [
                            {
                                "item_id": "a",
                                "multiview_png": "/figures/a.png",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "bundle"
            result = collect_bundle(
                selection,
                materialization,
                preflight,
                batch,
                audit,
                visualization,
                output,
            )
            self.assertEqual(len(result["scenes"]), 1)
            self.assertEqual(len(result["failures"]), 1)
            self.assertTrue((output / "large_scale_summary.md").is_file())
            self.assertTrue((output / "scene_artifact_index.csv").is_file())
            self.assertTrue((output / "failures.csv").is_file())
            with self.assertRaises(FileExistsError):
                collect_bundle(
                    selection,
                    materialization,
                    preflight,
                    batch,
                    audit,
                    visualization,
                    output,
                )


if __name__ == "__main__":
    unittest.main()
