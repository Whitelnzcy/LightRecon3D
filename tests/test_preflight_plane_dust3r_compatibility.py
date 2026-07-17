import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preflight_plane_dust3r_compatibility import run_preflight


def touch_image_group(path: Path, positions=("0", "1", "2", "3", "4")):
    for position in positions:
        image = path / position / "rgb_rawlight.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"png")


def make_repo(path: Path):
    path.mkdir()
    for name in (
        "evaluate_planedust3r.py",
        "metric.py",
        "plane_merge_planedust3r.py",
    ):
        (path / name).write_text("# fixture\n", encoding="utf-8")
    (path / "MASt3R").mkdir()
    (path / "NonCuboidRoom").mkdir()


class PlaneDust3rCompatibilityTests(unittest.TestCase):
    def test_native_ready_but_common_partition_is_not_claimed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "scene_00180/2D_rendering/room/perspective/empty"
            full = empty.parent / "full"
            touch_image_group(empty)
            touch_image_group(full)
            batch = root / "batch.json"
            batch.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "final_000_scene_00180",
                                "scene_name": "scene_00180",
                                "pair_group": str(empty),
                                "status": "pass",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            repo = root / "Plane-DUSt3R"
            make_repo(repo)
            plane_checkpoint = root / "plane.pth"
            noncuboid_checkpoint = root / "layout.pt"
            plane_checkpoint.write_bytes(b"plane")
            noncuboid_checkpoint.write_bytes(b"layout")

            output = root / "output"
            result = run_preflight(
                batch,
                repo,
                plane_checkpoint,
                noncuboid_checkpoint,
                output,
                git_sha="abc123",
            )

            self.assertTrue(result["summary"]["native_smoke_ready"])
            self.assertTrue(result["summary"]["native_full_batch_ready"])
            self.assertFalse(result["summary"]["common_partition_ready"])
            self.assertEqual(result["summary"]["identical_input_scenes"], 0)
            self.assertEqual(result["scenes"][0]["shared_position_count"], 5)
            self.assertTrue((output / "plane_dust3r_compatibility.json").is_file())
            report = (output / "plane_dust3r_compatibility.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("separate table", report)

    def test_missing_external_assets_are_reported_without_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "scene/perspective/empty"
            touch_image_group(empty, positions=("0",))
            batch = root / "batch.json"
            batch.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "item",
                                "scene_name": "scene",
                                "pair_group": str(empty),
                                "status": "pass",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = run_preflight(
                batch,
                root / "missing_repo",
                root / "missing_plane.pth",
                root / "missing_layout.pt",
                root / "output",
            )
            self.assertFalse(result["summary"]["native_smoke_ready"])
            self.assertEqual(result["summary"]["native_scene_ready"], 0)
            self.assertGreaterEqual(len(result["summary"]["blocking_reasons"]), 4)

    def test_duplicate_scene_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch.json"
            batch.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "a",
                                "scene_name": "scene",
                                "pair_group": "/a/empty",
                            },
                            {
                                "id": "b",
                                "scene_name": "scene",
                                "pair_group": "/b/empty",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate independent scene"):
                run_preflight(
                    batch,
                    root / "repo",
                    root / "plane",
                    root / "layout",
                    root / "output",
                )


if __name__ == "__main__":
    unittest.main()
