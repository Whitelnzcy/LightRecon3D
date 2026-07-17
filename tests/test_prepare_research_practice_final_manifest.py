import json
import tempfile
import unittest
from pathlib import Path

from prepare_research_practice_final_manifest import (
    DEFAULT_PATTERN,
    execution_manifest,
    group_dataset_samples,
    load_existing_stage2_groups,
    load_reusable_caches,
    select_unique_scene_groups,
    write_plan,
)


def group(scene, pair_group, start=0):
    return {
        "scene_name": scene,
        "pair_group": pair_group,
        "pair_count": 10,
        "selected_indices": list(range(start, start + 10)),
    }


class FinalManifestPreparationTests(unittest.TestCase):
    def test_dataset_grouping_keeps_indices_and_drops_short_groups(self):
        samples = [
            {"scene_name": "scene_1", "pair_group": "/room/a"}
            for _ in range(3)
        ]
        samples += [
            {"scene_name": "scene_2", "pair_group": "/room/b"}
            for _ in range(2)
        ]
        rows = group_dataset_samples(samples, min_pairs=3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scene_name"], "scene_1")
        self.assertEqual(rows[0]["selected_indices"], [0, 1, 2])

    def test_pair_group_cannot_cross_scene_boundary(self):
        samples = [
            {"scene_name": "scene_1", "pair_group": "/same"},
            {"scene_name": "scene_2", "pair_group": "/same"},
        ]
        with self.assertRaisesRegex(ValueError, "crosses scenes"):
            group_dataset_samples(samples, min_pairs=1)

    def test_existing_stage2_inventory_requires_complete_record_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tsv = root / "selected_groups.tsv"
            tsv.write_text("0\t10\t0,1,2\t/room/a\n", encoding="utf-8")
            stage2 = root / "group_000_pairs_10" / "stage2_merge"
            stage2.mkdir(parents=True)
            for index in range(3):
                (stage2 / f"{index}_learned_region_merge_full_pointcloud_editable_planes_data.npz").write_bytes(b"x")
            existing, warnings = load_existing_stage2_groups(
                root, pattern=DEFAULT_PATTERN, minimum_records=3
            )
            self.assertTrue(existing["/room/a"]["ready"])
            self.assertEqual(existing["/room/a"]["stage2_records"], 3)
            self.assertEqual(warnings, [])
            incomplete, warnings = load_existing_stage2_groups(
                root, pattern=DEFAULT_PATTERN, minimum_records=4
            )
            self.assertFalse(incomplete["/room/a"]["ready"])
            self.assertEqual(len(warnings), 1)

    def test_selection_is_one_per_scene_and_prefers_existing_without_metrics(self):
        eligible = [
            group("scene_1", "/room/a"),
            group("scene_1", "/room/b", 10),
            group("scene_2", "/room/c", 20),
            group("scene_3", "/room/d", 30),
        ]
        existing = {
            "/room/b": {
                "ready": True,
                "input_dir": "/existing/b/stage2_merge",
                "group_name": "group_001_pairs_10",
                "stage2_records": 10,
            }
        }
        caches = {
            "/room/b": {
                "global_cloud_cache": "/cache/b.npz",
                "global_cloud_sha256": "a" * 64,
            }
        }
        selected = select_unique_scene_groups(
            eligible,
            existing,
            caches,
            target_scenes=3,
            expansion_root=Path("/new"),
        )
        self.assertEqual([row["scene_name"] for row in selected], ["scene_1", "scene_2", "scene_3"])
        self.assertEqual(selected[0]["pair_group"], "/room/b")
        self.assertEqual(selected[0]["materialization"], "reuse_existing_stage2")
        self.assertEqual(selected[1]["materialization"], "needs_stage1_stage2")
        self.assertIsNotNone(selected[0]["reusable_cache"])

    def test_zero_target_selects_all_eligible_scenes(self):
        eligible = [
            group("scene_1", "/room/a"),
            group("scene_2", "/room/b", 10),
            group("scene_3", "/room/c", 20),
        ]
        selected = select_unique_scene_groups(
            eligible,
            {},
            {},
            target_scenes=0,
            expansion_root=Path("/new"),
        )
        self.assertEqual([row["scene_name"] for row in selected], ["scene_1", "scene_2", "scene_3"])
        manifest = execution_manifest(
            selected, pattern=DEFAULT_PATTERN, minimum_valid_items=2
        )
        self.assertEqual(manifest["minimum_valid_items"], 2)

    def test_reusable_cache_is_keyed_by_exact_pair_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "batch.json"
            path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "a",
                                "status": "pass",
                                "pair_group": "/room/a",
                                "artifacts": {
                                    "global_cloud_cache": {
                                        "path": "/cache/a.npz",
                                        "sha256": "b" * 64,
                                    }
                                },
                            },
                            {
                                "id": "failed",
                                "status": "fail",
                                "pair_group": "/room/b",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            caches, warnings = load_reusable_caches(path)
            self.assertEqual(list(caches), ["/room/a"])
            self.assertEqual(caches["/room/a"]["global_cloud_sha256"], "b" * 64)
            self.assertEqual(warnings, [])

    def test_written_manifest_refuses_overwrite_and_preserves_expected_metadata(self):
        selected = select_unique_scene_groups(
            [group("scene_1", "/room/a")],
            {},
            {},
            target_scenes=1,
            expansion_root=Path("/new"),
        )
        manifest = execution_manifest(selected, pattern=DEFAULT_PATTERN)
        self.assertEqual(manifest["items"][0]["expected_scene_name"], "scene_1")
        self.assertEqual(manifest["items"][0]["expected_pair_group"], "/room/a")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "selection"
            write_plan(
                output,
                selected=selected,
                dataset_inventory={"eligible_scenes": 1},
                manifest=manifest,
                warnings=[],
                configuration={},
            )
            self.assertTrue((output / "selection_plan.json").is_file())
            self.assertTrue((output / "final_unique_scenes_execute.json").is_file())
            with self.assertRaises(FileExistsError):
                write_plan(
                    output,
                    selected=selected,
                    dataset_inventory={},
                    manifest=manifest,
                    warnings=[],
                    configuration={},
                )


if __name__ == "__main__":
    unittest.main()
