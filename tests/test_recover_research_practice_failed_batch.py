import json
import tempfile
import unittest
from pathlib import Path

from recover_research_practice_failed_batch import merge_recovery, prepare_recovery
from research_practice_batch import file_sha256


def dump(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def selection_item(root: Path) -> dict:
    return {
        "id": "scene_a",
        "scene_name": "scene_a",
        "pair_group": "/room/a",
        "group_key": "scene_a::/room/a",
        "pair_count": 10,
        "selected_indices": list(range(10)),
        "input_dir": str(root / "original/scene_a/stage2_merge"),
        "materialization": "needs_stage1_stage2",
        "existing_group_name": "",
        "existing_stage2_records": 0,
        "reusable_cache": None,
    }


def material_row(root: Path, status: str) -> dict:
    return {
        "id": "scene_a",
        "scene_name": "scene_a",
        "pair_group": "/room/a",
        "materialization": "needs_stage1_stage2",
        "input_dir": str(root / "original/scene_a/stage2_merge"),
        "status": status,
        "failure_stage": "stage1_support" if status != "pass" else "",
        "error": "platform failed" if status != "pass" else "",
        "runtime_seconds": 1.0,
    }


def batch_row(root: Path, status: str) -> dict:
    row = {
        "id": "scene_a",
        "scene_name": "scene_a",
        "pair_group": "/room/a",
        "status": status,
        "failure_stage": "input_preflight" if status != "pass" else "",
        "error": "missing" if status != "pass" else "",
        "runtime_seconds": 2.0,
        "stages": [],
        "artifacts": {},
    }
    if status == "pass":
        full = root / "full.json"
        support = root / "support.json"
        full.write_text(json.dumps([{"method": "global_ransac_cc", "plane_recall": 0.5}]))
        support.write_text(
            json.dumps(
                {
                    "methods": [
                        {
                            "method": "learning_guided_ransac_cc",
                            "support_partition_pairwise_f1": 0.8,
                        }
                    ]
                }
            )
        )
        row["artifacts"] = {
            "full_cache_metrics": {"path": str(full)},
            "support_record_metrics": {"path": str(support)},
        }
    return row


def preflight_row(status: str) -> dict:
    return {
        "id": "scene_a",
        "status": status,
        "scene_name": "scene_a",
        "pair_group": "/room/a",
        "group_key": "scene_a::/room/a",
        "records": 10 if status == "pass" else 0,
        "unique_views": 5 if status == "pass" else 0,
        "input_bytes": 100 if status == "pass" else 0,
        "errors": [] if status == "pass" else ["missing"],
        "warnings": [],
    }


class FailedBatchRecoveryTests(unittest.TestCase):
    def test_prepare_recovery_redirects_only_failed_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = dump(
                root / "selection.json",
                {
                    "schema_version": 1,
                    "kind": "research_practice_final_scene_selection",
                    "items": [selection_item(root)],
                },
            )
            material = dump(
                root / "material.json",
                {
                    "selection_plan_sha256": file_sha256(selection),
                    "items": [material_row(root, "fail")],
                },
            )
            output = root / "plan"
            result = prepare_recovery(selection, material, root / "retry", output)
            retry = json.loads(
                (output / "recovery_selection_plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["failed_items"], ["scene_a"])
            self.assertEqual(len(retry["items"]), 1)
            self.assertEqual(
                retry["items"][0]["input_dir"],
                str(root / "retry/scene_a/stage2_merge"),
            )
            self.assertEqual(retry["items"][0]["selected_indices"], list(range(10)))

    def test_merge_replaces_recovered_rows_and_recomputes_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = dump(
                root / "selection.json",
                {
                    "schema_version": 1,
                    "kind": "research_practice_final_scene_selection",
                    "items": [selection_item(root)],
                },
            )
            selection_sha = file_sha256(selection)
            original_material = dump(
                root / "original_material.json",
                {
                    "selection_plan_sha256": selection_sha,
                    "items": [material_row(root, "fail")],
                },
            )
            recovered_material_row = material_row(root, "pass")
            recovered_material_row["input_dir"] = str(root / "retry/scene_a/stage2_merge")
            recovery_material = dump(
                root / "recovery_material.json", {"items": [recovered_material_row]}
            )
            original_preflight = dump(
                root / "original_preflight.json",
                {
                    "summary": {"minimum_valid_items": 1},
                    "items": [preflight_row("fail")],
                },
            )
            recovery_preflight = dump(
                root / "recovery_preflight.json", {"items": [preflight_row("pass")]}
            )
            original_batch = dump(
                root / "original_batch.json",
                {
                    "batch_name": "original",
                    "git_sha": "original-sha",
                    "weights_sha256": "weights",
                    "items": [batch_row(root, "fail")],
                },
            )
            recovery_batch = dump(
                root / "recovery_batch.json",
                {
                    "weights_sha256": "weights",
                    "items": [batch_row(root, "pass")],
                },
            )
            output = root / "merged"
            result = merge_recovery(
                selection,
                original_material,
                recovery_material,
                original_preflight,
                recovery_preflight,
                original_batch,
                recovery_batch,
                output,
            )
            self.assertEqual(result["recovered_batch_items"], 1)
            self.assertEqual(result["combined_batch_summary"]["passed_items"], 1)
            combined = json.loads(
                (output / "combined_batch/batch_execution.json").read_text(encoding="utf-8")
            )
            self.assertEqual(combined["items"][0]["status"], "pass")
            self.assertIn("recovered_from_failure", combined["items"][0])
            metrics = json.loads(
                (output / "combined_batch/aggregate_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(metrics), 2)


if __name__ == "__main__":
    unittest.main()
