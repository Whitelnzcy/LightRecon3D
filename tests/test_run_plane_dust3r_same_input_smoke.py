import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_plane_dust3r_same_input_smoke import patch_evaluator


class PlaneDust3rSmokeTests(unittest.TestCase):
    def test_patches_save_and_silent_exception_once(self):
        source = """import traceback
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
        self.assertNotIn("save = False", patched)
        self.assertNotIn("except Exception as e:\n        continue", patched)

    def test_refuses_source_drift(self):
        with self.assertRaisesRegex(ValueError, "hardcoded save flag"):
            patch_evaluator("save = True\n")
        with self.assertRaisesRegex(ValueError, "silent room exception"):
            patch_evaluator("save = False\n")


if __name__ == "__main__":
    unittest.main()
