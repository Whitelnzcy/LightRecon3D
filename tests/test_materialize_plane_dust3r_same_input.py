import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from materialize_plane_dust3r_same_input import materialize


def make_scene(root: Path, scene: str, room: str, positions=("0", "1")) -> Path:
    scene_root = root / scene
    scene_root.mkdir(parents=True)
    (scene_root / "annotation_3d.json").write_text("{}", encoding="utf-8")
    empty = scene_root / "2D_rendering" / room / "perspective" / "empty"
    for position in positions:
        position_dir = empty / position
        position_dir.mkdir(parents=True)
        (position_dir / "rgb_rawlight.png").write_bytes(b"rgb")
        (position_dir / "camera_pose.txt").write_text("pose", encoding="utf-8")
    return empty


def write_batch(path: Path, items):
    path.write_text(json.dumps({"items": items}), encoding="utf-8")


class SameInputMaterializationTests(unittest.TestCase):
    def test_copy_mode_builds_full_compatibility_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = make_scene(root, "scene_00180", "445895")
            batch = root / "batch.json"
            write_batch(
                batch,
                [
                    {
                        "id": "final_000_scene_00180",
                        "scene_name": "scene_00180",
                        "pair_group": str(empty),
                        "status": "pass",
                    }
                ],
            )
            output = root / "output"
            result = materialize(batch, output, link_mode="copy", git_sha="abc")

            self.assertEqual(result["summary"], {"scenes": 1, "images": 2, "link_mode": "copy"})
            self.assertFalse(result["protocol"]["native_protocol_claim_allowed"])
            full = (
                output
                / "dataset/scene_00180/2D_rendering/445895/perspective/full"
            )
            self.assertTrue((full / "0/rgb_rawlight.png").is_file())
            self.assertTrue((full / "1/camera_pose.txt").is_file())
            self.assertTrue(
                (output / "dataset/scene_00180/annotation_3d.json").is_file()
            )
            self.assertIn(
                "not the Plane-DUSt3R native",
                (output / "same_input_manifest.md").read_text(encoding="utf-8"),
            )

    def test_maximum_scenes_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = make_scene(root, "scene_00180", "room_a", positions=("0",))
            second = make_scene(root, "scene_00181", "room_b", positions=("0",))
            batch = root / "batch.json"
            write_batch(
                batch,
                [
                    {"id": "a", "scene_name": "scene_00180", "pair_group": str(first)},
                    {"id": "b", "scene_name": "scene_00181", "pair_group": str(second)},
                ],
            )
            result = materialize(
                batch, root / "output", link_mode="copy", maximum_scenes=1
            )
            self.assertEqual(result["summary"]["scenes"], 1)
            self.assertEqual(result["items"][0]["scene_name"], "scene_00180")

    def test_missing_annotation_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = make_scene(root, "scene_00180", "room")
            (root / "scene_00180/annotation_3d.json").unlink()
            batch = root / "batch.json"
            write_batch(
                batch,
                [{"id": "a", "scene_name": "scene_00180", "pair_group": str(empty)}],
            )
            output = root / "output"
            with self.assertRaisesRegex(FileNotFoundError, "annotation"):
                materialize(batch, output, link_mode="copy")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
