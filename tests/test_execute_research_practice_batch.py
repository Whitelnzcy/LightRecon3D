import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execute_research_practice_batch import (
    aggregate_method_rows,
    batch_summary,
    execute_item,
    execute_batch,
    exporter_command,
    metric_rows,
    single_fusion_row,
    verify_reused_cache,
)


class ResearchPracticeBatchExecutionTests(unittest.TestCase):
    def test_exporter_command_uses_exactly_one_cache_source(self):
        project = Path("/project")
        item = {"input_dir": "/input", "pattern": "*.npz"}
        weights = Path("/weights/model.pth")
        generated = exporter_command(
            "/env/python", project, item, Path("/output"), "none", weights, None
        )
        reused = exporter_command(
            "/env/python",
            project,
            item,
            Path("/output"),
            "manual",
            weights,
            Path("/cache/global.npz"),
        )
        self.assertIn("--weights_path", generated)
        self.assertNotIn("--global_cloud_cache", generated)
        self.assertIn("--global_cloud_cache", reused)
        self.assertNotIn("--weights_path", reused)
        self.assertNotIn("--plane_feedback", generated)
        self.assertNotIn("--plane_feedback", reused)

    def test_single_fusion_manifest_requires_one_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "stage3_scene_fusion_manifest.json"
            path.write_text(json.dumps([{"npz": "one.npz"}]), encoding="utf-8")
            self.assertEqual(single_fusion_row(root)["npz"], "one.npz")
            path.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected one fusion row"):
                single_fusion_row(root)

    def test_reused_cache_checksum_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.npz"
            path.write_bytes(b"cache")
            checksum = hashlib.sha256(b"cache").hexdigest()
            cache, record = verify_reused_cache(
                {
                    "global_cloud_cache": str(path),
                    "global_cloud_sha256": checksum,
                }
            )
            self.assertEqual(cache, path)
            self.assertTrue(record["checksum_matches"])
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                verify_reused_cache(
                    {
                        "global_cloud_cache": str(path),
                        "global_cloud_sha256": "0" * 64,
                    }
                )

    def test_failed_preflight_becomes_item_failure_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = execute_item(
                {
                    "id": "missing",
                    "input_dir": str(root / "does_not_exist"),
                    "expected_scene_name": "scene",
                    "expected_pair_group": "/room",
                },
                project_dir=root,
                python_bin=sys.executable,
                weights_path=root / "weights.pth",
                output_dir=root / "output",
            )
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["failure_stage"], "input_preflight")
            self.assertTrue((root / "output/items/missing/item_execution.json").is_file())

    def test_summary_counts_view_groups_and_scenes_separately(self):
        items = [
            {
                "id": "a",
                "status": "pass",
                "scene_name": "scene_1",
                "pair_group": "/room/a",
                "runtime_seconds": 1.0,
                "line_summary": {"line_count": 2},
            },
            {
                "id": "b",
                "status": "pass",
                "scene_name": "scene_1",
                "pair_group": "/room/b",
                "runtime_seconds": 2.0,
                "line_summary": {"line_count": 3},
            },
        ]
        summary = batch_summary(items)
        self.assertEqual(summary["unique_view_groups"], 2)
        self.assertEqual(summary["unique_scene_names"], 1)
        self.assertEqual(summary["line_count"], 5)

    def test_metric_rows_keeps_families_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = root / "full.json"
            support = root / "support.json"
            full.write_text(json.dumps([{"method": "ransac", "plane_recall": 0.5}]), encoding="utf-8")
            support.write_text(
                json.dumps(
                    {
                        "methods": [
                            {
                                "method": "manual",
                                "support_partition_pairwise_f1": 0.7,
                                "per_plane": [{"pred_plane_id": 0}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = metric_rows(
                [
                    {
                        "id": "group",
                        "status": "pass",
                        "scene_name": "scene",
                        "artifacts": {
                            "full_cache_metrics": {"path": str(full)},
                            "support_record_metrics": {"path": str(support)},
                        },
                    }
                ]
            )
            self.assertEqual([row["metric_family"] for row in rows], ["full_cache", "support_records"])
            self.assertNotIn("per_plane", rows[1])

    def test_aggregate_method_rows_reports_mean_median_and_scene_count(self):
        rows = [
            {
                "item_id": "a",
                "scene_name": "scene_1",
                "metric_family": "support_records",
                "method": "manual",
                "support_partition_pairwise_f1": 0.6,
            },
            {
                "item_id": "b",
                "scene_name": "scene_1",
                "metric_family": "support_records",
                "method": "manual",
                "support_partition_pairwise_f1": 0.8,
            },
        ]
        summary = aggregate_method_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["view_groups"], 2)
        self.assertEqual(summary[0]["unique_scenes"], 1)
        self.assertAlmostEqual(
            summary[0]["support_partition_pairwise_f1_mean"], 0.7
        )
        self.assertAlmostEqual(
            summary[0]["support_partition_pairwise_f1_median"], 0.7
        )

    def test_execute_batch_resume_skips_recorded_items(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "weights.pth"
            weights.write_bytes(b"weights")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "name": "resume_test",
                        "minimum_valid_items": 1,
                        "items": [
                            {
                                "id": "scene_a",
                                "input_dir": str(root / "input"),
                                "expected_scene_name": "scene_a",
                                "expected_pair_group": "/room/a",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            output.mkdir()
            recorded = {
                "id": "scene_a",
                "scene_name": "scene_a",
                "pair_group": "/room/a",
                "status": "fail",
                "failure_stage": "test",
                "error": "recorded",
                "runtime_seconds": 1.0,
                "stages": [],
                "artifacts": {},
            }
            (output / "batch_execution.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "research_practice_identical_cache_batch",
                        "batch_name": "resume_test",
                        "git_sha": "abc",
                        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                        "items": [recorded],
                        "summary": {"failed_items": 1},
                    }
                ),
                encoding="utf-8",
            )
            calls = []
            result = execute_batch(
                manifest,
                output,
                project_dir=root,
                python_bin=sys.executable,
                weights_path=weights,
                git_sha="abc",
                resume=True,
                runner=lambda *args: calls.append(args),
            )
            self.assertEqual(result["summary"]["failed_items"], 1)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
