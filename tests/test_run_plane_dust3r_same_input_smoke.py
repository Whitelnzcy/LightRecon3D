import sys
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_plane_dust3r_same_input_smoke import (
    build_runtime_env,
    git_output,
    link_or_copy,
    patch_evaluator,
    patch_metric_paths,
    patch_noncuboid_config,
    seed_dust3r_output,
    seed_room_outputs,
    validate_official_worktree_status,
    write_mmcv_compat,
)


class PlaneDust3rSmokeTests(unittest.TestCase):
    def test_git_audit_scopes_safe_directory_to_the_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(
                ["git", "init", str(root)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                git_output(root, "rev-parse", "--is-inside-work-tree"),
                "true",
            )

    def test_patches_save_and_silent_exception_once(self):
        source = """import traceback
import argparse
import torch
args = object()
save = False
for room in rooms:
    try:
        run(room)
    except Exception as e:
        continue
"""
        patched = patch_evaluator(source)
        self.assertIn("save = bool(args.save_flag)", patched)
        self.assertIn("traceback.print_exc()", patched)
        self.assertIn("raise", patched)
        self.assertIn(
            "torch.serialization.add_safe_globals([argparse.Namespace])",
            patched,
        )
        self.assertNotIn("save = False", patched)
        self.assertNotIn("except Exception as e:\n        continue", patched)

    def test_refuses_source_drift(self):
        with self.assertRaisesRegex(ValueError, "hardcoded save flag"):
            patch_evaluator("save = True\n")
        with self.assertRaisesRegex(ValueError, "silent room exception"):
            patch_evaluator("import torch\nsave = False\n")
        with self.assertRaisesRegex(ValueError, "torch import"):
            patch_evaluator(
                "save = False\n"
                "try:\n"
                "    run()\n"
                "except Exception as e:\n"
                "    continue\n"
            )

    def test_minimal_mmcv_compat_initializers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_mmcv_compat(root)
            code = f"""
import sys
sys.path.insert(0, {str(root)!r})
import torch
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

layer = torch.nn.Conv2d(3, 2, 3, bias=True)
constant_init(layer, 2, bias=3)
assert torch.all(layer.weight == 2)
assert torch.all(layer.bias == 3)
kaiming_init(layer)
assert torch.isfinite(layer.weight).all()
assert callable(load_checkpoint)
"""
            completed = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )

    def test_link_or_copy_falls_back_when_symlink_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "value.txt").write_text("preserved", encoding="utf-8")
            destination = root / "destination"
            with mock.patch("os.symlink", side_effect=OSError("not permitted")):
                mode = link_or_copy(source, destination)
            self.assertEqual(mode, "copy")
            self.assertEqual(
                (destination / "value.txt").read_text(encoding="utf-8"),
                "preserved",
            )

    def test_runtime_env_uses_legacy_protobuf_implementation(self):
        with mock.patch.dict(
            "os.environ",
            {"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": ""},
            clear=False,
        ):
            environment = build_runtime_env()
        self.assertEqual(environment["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"], "")

        with mock.patch.dict(
            "os.environ",
            {},
            clear=True,
        ):
            environment = build_runtime_env()
        self.assertEqual(
            environment["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"],
            "python",
        )

    def test_official_worktree_allows_only_checkpoints_and_bytecode(self):
        allowed = validate_official_worktree_status(
            "?? checkpoints/\n"
            "?? NonCuboidRoom/noncuboid/models/__pycache__/\n"
        )
        self.assertEqual(
            allowed,
            [
                "checkpoints/",
                "NonCuboidRoom/noncuboid/models/__pycache__/",
            ],
        )
        with self.assertRaisesRegex(ValueError, "unexpected worktree changes"):
            validate_official_worktree_status(" M evaluate_planedust3r.py\n")
        with self.assertRaisesRegex(ValueError, "unexpected worktree changes"):
            validate_official_worktree_status("?? patched_model.py\n")

    def test_windows_noncuboid_config_disables_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "cfg.yaml"
            config.write_text("batch_size: 1\nnum_workers: 4\n", encoding="utf-8")
            with mock.patch("os.name", "nt"):
                patch_noncuboid_config(config)
            self.assertIn(
                "num_workers: 0",
                config.read_text(encoding="utf-8"),
            )

    def test_seed_dust3r_output_uses_official_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.npz"
            source.write_bytes(b"completed-cache")
            destination = seed_dust3r_output(
                source,
                root / "predictions",
                "scene_00180",
                "445895",
            )
            self.assertEqual(
                destination.relative_to(root / "predictions").as_posix(),
                "00180/445895/dust3r_output.npz",
            )
            self.assertEqual(destination.read_bytes(), b"completed-cache")

    def test_windows_metric_path_uses_native_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            metric = Path(tmp) / "metric.py"
            metric.write_text(
                (
                    '    room_path = "/".join(image_list[0].split(\'/\')[:-2])\n'
                    "    img_id = image_path.split('/')[-2]\n"
                    '    room_path = "/".join(image_list[0].split(\'/\')[:-2])\n'
                    "    img_id = image_path.split('/')[-2]\n"
                    '    room_path = "/".join(image_list[0].split(\'/\')[:-2])\n'
                    "    img_id = image_path.split('/')[-2]\n"
                ),
                encoding="utf-8",
            )
            patch_metric_paths(metric)
            self.assertEqual(
                metric.read_text(encoding="utf-8").count(
                    "os.path.dirname(os.path.dirname(image_list[0]))"
                ),
                3,
            )
            self.assertEqual(
                metric.read_text(encoding="utf-8").count(
                    "os.path.basename(os.path.dirname(image_path))"
                ),
                3,
            )

    def test_seed_room_outputs_requires_complete_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for name in (
                "dust3r_output.npz",
                "plane_detection.json",
                "node_data.json",
                "layout.png",
            ):
                (source / name).write_bytes(name.encode("utf-8"))
            copied = seed_room_outputs(
                source,
                root / "predictions",
                "scene_00180",
                "445895",
            )
            self.assertEqual(len(copied), 4)
            self.assertTrue(
                (
                    root
                    / "predictions"
                    / "00180"
                    / "445895"
                    / "node_data.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
