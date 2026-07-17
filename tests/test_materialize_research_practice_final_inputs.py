import json
import tempfile
import unittest
from pathlib import Path

from materialize_research_practice_final_inputs import (
    load_selection_plan,
    materialize_item,
    materialize_plan,
    registry_command,
    stage1_command,
    stage2_command,
    summary,
)
from research_practice_batch import file_record, file_sha256


def plan_item(root, item_id="final_000_scene_1", scene="scene_1", group="/room/a"):
    return {
        "id": item_id,
        "scene_name": scene,
        "pair_group": group,
        "pair_count": 10,
        "selected_indices": list(range(10, 20)),
        "input_dir": str(Path(root) / item_id / "stage2_merge"),
        "materialization": "needs_stage1_stage2",
        "existing_group_name": "",
        "existing_stage2_records": 0,
        "reusable_cache": None,
    }


def write_plan(path, items):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "research_practice_final_scene_selection",
                "items": items,
            }
        ),
        encoding="utf-8",
    )


class FinalInputMaterializationTests(unittest.TestCase):
    def test_plan_rejects_duplicate_independent_scene(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "plan.json"
            write_plan(
                path,
                [
                    plan_item(root, "a", "scene_1", "/room/a"),
                    plan_item(root, "b", "scene_1", "/room/b"),
                ],
            )
            with self.assertRaisesRegex(ValueError, "duplicates"):
                load_selection_plan(path)

    def test_stage_commands_freeze_indices_and_check_registry(self):
        item = plan_item("/new")
        stage1 = stage1_command(
            "/env/python",
            Path("/project"),
            item,
            root_dir=Path("/data"),
            weights_path=Path("/weights.pth"),
            stage1_checkpoint=Path("/stage1.pt"),
            feature_cache_path=Path("/features.pt"),
            stage1_dir=Path("/output/stage1"),
        )
        self.assertIn("--indices", stage1)
        self.assertIn(",".join(map(str, range(10, 20))), stage1)
        self.assertIn("--export_second_view_support", stage1)
        self.assertIn("--pair_strategy", stage1)
        stage2 = stage2_command(
            "/env/python",
            Path("/project"),
            stage1_dir=Path("/output/stage1"),
            stage2_checkpoint=Path("/stage2.pt"),
            stage2_dir=Path("/output/stage2"),
        )
        self.assertIn("--use_safety_gate", stage2)
        registry = registry_command(
            "/env/python",
            Path("/project"),
            stage2_dir=Path("/output/stage2"),
            output_json=Path("/output/registry.json"),
        )
        self.assertIn("--check_files", registry)

    def test_existing_materialization_root_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = plan_item(root)
            Path(item["input_dir"]).parent.mkdir(parents=True)
            calls = []

            def runner(*args):
                calls.append(args)
                return {"return_code": 0}

            row = materialize_item(
                item,
                project_dir=root,
                python_bin="python",
                root_dir=root,
                weights_path=root / "weights",
                stage1_checkpoint=root / "stage1",
                feature_cache_path=root / "features",
                stage2_checkpoint=root / "stage2",
                runner=runner,
            )
            self.assertEqual(row["status"], "fail")
            self.assertEqual(row["failure_stage"], "refuse_overwrite")
            self.assertEqual(calls, [])

    def test_summary_separates_reused_and_new_groups(self):
        rows = [
            {
                "status": "pass",
                "scene_name": "scene_1",
                "materialization": "reuse_existing_stage2",
                "runtime_seconds": 1.0,
            },
            {
                "status": "pass",
                "scene_name": "scene_2",
                "materialization": "needs_stage1_stage2",
                "runtime_seconds": 2.0,
            },
            {
                "status": "fail",
                "scene_name": "scene_3",
                "materialization": "needs_stage1_stage2",
                "runtime_seconds": 3.0,
            },
        ]
        result = summary(rows)
        self.assertEqual(result["passed_items"], 2)
        self.assertEqual(result["failed_items"], 1)
        self.assertEqual(result["independent_scenes"], 2)
        self.assertEqual(result["reused_stage2_groups"], 1)
        self.assertEqual(result["materialized_stage2_groups"], 1)

    def test_materialize_plan_refuses_existing_ledger_before_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            write_plan(plan, [plan_item(root)])
            output = root / "ledger"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                materialize_plan(
                    plan,
                    output,
                    project_dir=root,
                    python_bin="python",
                    root_dir=root,
                    weights_path=root / "weights",
                    stage1_checkpoint=root / "stage1",
                    feature_cache_path=root / "features",
                    stage2_checkpoint=root / "stage2",
                    git_sha="abc",
                )

    def test_materialize_plan_resume_skips_recorded_items(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = plan_item(root)
            plan = root / "plan.json"
            write_plan(plan, [item])
            required = {}
            for name in ("weights", "stage1", "features", "stage2"):
                path = root / name
                path.write_bytes(name.encode())
                required[name] = path
            output = root / "ledger"
            output.mkdir()
            frozen = {
                "weights": file_record(required["weights"]),
                "stage1_checkpoint": file_record(required["stage1"]),
                "feature_cache": file_record(required["features"]),
                "stage2_checkpoint": file_record(required["stage2"]),
            }
            recorded = {
                "id": item["id"],
                "scene_name": item["scene_name"],
                "pair_group": item["pair_group"],
                "materialization": item["materialization"],
                "input_dir": item["input_dir"],
                "status": "pass",
                "failure_stage": "",
                "error": "",
                "runtime_seconds": 1.0,
            }
            (output / "materialization.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "research_practice_final_input_materialization",
                        "selection_plan_sha256": file_sha256(plan),
                        "frozen_inputs": frozen,
                        "git_sha": "abc",
                        "items": [recorded],
                        "summary": {},
                    }
                ),
                encoding="utf-8",
            )
            calls = []
            result = materialize_plan(
                plan,
                output,
                project_dir=root,
                python_bin="python",
                root_dir=root,
                weights_path=required["weights"],
                stage1_checkpoint=required["stage1"],
                feature_cache_path=required["features"],
                stage2_checkpoint=required["stage2"],
                git_sha="abc",
                resume=True,
                runner=lambda *args: calls.append(args),
            )
            self.assertEqual(result["summary"]["passed_items"], 1)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
