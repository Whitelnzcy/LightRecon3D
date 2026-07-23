import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_practice_batch import load_manifest, run_preflight


def write_stage2(path, image1, image2, scene, pair_group):
    np.savez_compressed(
        path,
        schema_version=np.asarray(2, dtype=np.int32),
        scene_name=np.asarray(scene),
        pair_group=np.asarray(pair_group),
        rgb_path1=np.asarray(str(image1)),
        rgb_path2=np.asarray(str(image2)),
        view_id1=np.asarray("0"),
        view_id2=np.asarray("1"),
        pixel_xy1=np.asarray([[-0.5, -0.5], [0.5, 0.5]], dtype=np.float32),
        pixel_xy2=np.empty((0, 2), dtype=np.float32),
        point_plane_ids=np.asarray([0, 0], dtype=np.int32),
        support_source_view=np.asarray([1, 1], dtype=np.int32),
    )


class ResearchPracticeBatchTests(unittest.TestCase):
    def make_item(self, root, item_id, scene, pair_group):
        input_dir = root / item_id
        input_dir.mkdir()
        image1 = root / f"{item_id}_0.png"
        image2 = root / f"{item_id}_1.png"
        image1.write_bytes(b"image-zero")
        image2.write_bytes(b"image-one")
        write_stage2(input_dir / "sample_learned_region_merge_full_pointcloud_editable_planes_data.npz", image1, image2, scene, pair_group)
        return {
            "id": item_id,
            "input_dir": str(input_dir),
            "expected_scene_name": scene,
            "expected_pair_group": pair_group,
        }

    def write_manifest(self, root, items, **extra):
        manifest = {
            "schema_version": 1,
            "name": "synthetic_smoke",
            "minimum_valid_items": len(items),
            "min_views": 2,
            "check_image_files": True,
            "items": items,
            **extra,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_success_writes_json_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items = [
                self.make_item(root, "group_000", "scene_00180", "/room/a"),
                self.make_item(root, "group_001", "scene_00181", "/room/b"),
            ]
            manifest = self.write_manifest(root, items)
            output = root / "output"
            result = run_preflight(manifest, output, git_sha="abc123")
            self.assertTrue(result["summary"]["minimum_met"])
            self.assertEqual(result["summary"]["passed_items"], 2)
            self.assertEqual(result["summary"]["unique_scene_names"], 2)
            self.assertTrue((output / "batch_preflight.json").is_file())
            self.assertTrue((output / "batch_preflight.csv").is_file())
            self.assertIn("Passed 2/2", (output / "batch_preflight.md").read_text(encoding="utf-8"))

    def test_duplicate_group_keys_are_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items = [
                self.make_item(root, "first", "scene_00180", "/same/room"),
                self.make_item(root, "second", "scene_00180", "/same/room"),
            ]
            manifest = self.write_manifest(root, items)
            result = run_preflight(manifest, root / "output")
            self.assertEqual(result["summary"]["passed_items"], 0)
            self.assertTrue(all("duplicate group_key" in row["errors"][-1] for row in result["items"]))

    def test_required_artifact_checksum_mismatch_is_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = self.make_item(root, "group_000", "scene_00180", "/room/a")
            artifact = root / "cache.npz"
            artifact.write_bytes(b"frozen-cache")
            item["artifacts"] = {
                "global_cloud_cache": {
                    "path": str(artifact),
                    "required": True,
                    "sha256": "0" * 64,
                }
            }
            manifest = self.write_manifest(root, [item])
            result = run_preflight(manifest, root / "output")
            self.assertEqual(result["items"][0]["status"], "fail")
            self.assertIn("checksum mismatch", result["items"][0]["errors"][-1])

    def test_refuses_existing_output_and_duplicate_item_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = self.make_item(root, "group_000", "scene_00180", "/room/a")
            manifest = self.write_manifest(root, [item])
            output = root / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                run_preflight(manifest, output)
            duplicate_manifest = self.write_manifest(root, [item, dict(item)])
            with self.assertRaisesRegex(ValueError, "Duplicate item id"):
                load_manifest(duplicate_manifest)


if __name__ == "__main__":
    unittest.main()
