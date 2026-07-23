import json
import tempfile
import unittest
from pathlib import Path

from prepare_plane_dust3r_requirements import (
    COMPATIBILITY_PINS,
    requirement_name,
    write_outputs,
)


class PlaneDust3RRequirementsTest(unittest.TestCase):
    def make_repo(self, root: Path) -> Path:
        repo = root / "Plane-DUSt3R"
        contents = {
            "MASt3R/requirements.txt": "numpy\nscipy>=1.7\neinops\n",
            "MASt3R/dust3r/requirements.txt": (
                "torch\ntorchvision\ngradio\nopencv-python\nroma\n"
            ),
            "NonCuboidRoom/requirements.txt": (
                "scipy==1.3.1\n"
                "opencv_python==4.2.0.34\n"
                "Shapely==1.7.0\n"
                "matplotlib==3.1.1\n"
                "numpy==1.17.2\n"
                "mmcv==1.2.1\n"
                "easydict==1.9\n"
                "numba==0.51.0\n"
                "Pillow==8.1.0\n"
                "PyYAML==5.3.1\n"
                "tensorboardX==2.1\n"
            ),
        }
        for relative, text in contents.items():
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return repo

    def test_normalizes_package_names(self):
        self.assertEqual(requirement_name("opencv_python==4.2.0.34"), "opencv-python")
        self.assertEqual(requirement_name("PyYAML>=5; python_version>'3'"), "pyyaml")
        self.assertIsNone(requirement_name("# comment"))

    def test_rejects_hidden_nested_requirements(self):
        with self.assertRaisesRegex(ValueError, "nested or editable"):
            requirement_name("-r more.txt")

    def test_writes_sanitized_requirements_and_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self.make_repo(root)
            output = root / "sanitized"
            audit = write_outputs(repo, output)

            dust3r = (output / "MASt3R__dust3r__requirements.txt").read_text()
            noncuboid = (output / "NonCuboidRoom__requirements.txt").read_text()
            constraints = (output / "constraints.txt").read_text()
            pip_pins = (output / "python311_compatibility.txt").read_text()
            binary_pins = (
                output / "python311_binary_compatibility.txt"
            ).read_text()
            source_pins = (
                output / "python311_source_compatibility.txt"
            ).read_text()

            self.assertEqual(dust3r, "gradio\n")
            self.assertEqual(noncuboid, "")
            self.assertIn("torch==2.2.0", constraints)
            self.assertIn("scipy==1.11.4", constraints)
            self.assertNotIn("torch==2.2.0", pip_pins)
            self.assertIn("mmcv==1.7.2", pip_pins)
            self.assertIn("setuptools==80.9.0", binary_pins)
            self.assertNotIn("mmcv==1.7.2", binary_pins)
            self.assertEqual(source_pins, "mmcv==1.7.2\n")
            self.assertEqual(len(audit["files"]), 3)
            self.assertEqual(len(audit["replacements"]), 17)

            reloaded = json.loads((output / "requirements_audit.json").read_text())
            self.assertEqual(reloaded["compatibility_pins"], COMPATIBILITY_PINS)
            self.assertTrue((output / "requirements_audit.md").is_file())

    def test_launcher_uses_fresh_environment_and_constraints(self):
        project = Path(__file__).resolve().parents[1]
        launcher = (project / "prepare_plane_dust3r_external.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("planedust3r-py311-torch220-cu118-v2", launcher)
        self.assertIn("--constraint \"${constraints}\"", launcher)
        self.assertIn("prepare_plane_dust3r_requirements.py", launcher)
        self.assertIn("--no-build-isolation", launcher)
        self.assertIn("python311_source_compatibility.txt", launcher)
        self.assertIn("mkl=2024.0 intel-openmp=2024.0", launcher)
        self.assertIn("verify_torch_runtime", launcher)
        self.assertLess(
            launcher.index("verify_torch_runtime\n"),
            launcher.index('requirements_dir="${OUT_DIR}/sanitized_requirements"'),
        )
        self.assertNotIn(
            'pip install -r "${OFFICIAL_REPO}/NonCuboidRoom/requirements.txt"',
            launcher,
        )
        smoke = (project / "run_plane_dust3r_same_input_smoke.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("planedust3r-py311-torch220-cu118-v2", smoke)
        self.assertIn('torch.version.cuda == "11.8"', smoke)


if __name__ == "__main__":
    unittest.main()
