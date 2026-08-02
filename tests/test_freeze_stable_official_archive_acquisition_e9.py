import ast
import unittest
from pathlib import Path

from freeze_stable_official_archive_acquisition_e9 import (
    REGISTRATION_ID,
    RESULT_FILES,
    SCENE_RE,
    STATE,
    scan_scene_ids,
)


class E9PreregistrationTests(unittest.TestCase):
    def test_registration_state(self):
        self.assertEqual(
            REGISTRATION_ID,
            "E9_PREREG_stable_official_archive_acquisition_20260729_v1",
        )
        self.assertEqual(STATE, "PRE_REGISTERED_NOT_AUTHORIZED")

    def test_frozen_scanner(self):
        self.assertEqual(SCENE_RE.pattern, r"scene_(\d{5})(?!\d)")
        self.assertEqual(scan_scene_ids("scene_00184"), ["scene_00184"])
        self.assertEqual(scan_scene_ids("scene_20260728"), [])
        self.assertEqual(scan_scene_ids("scene_001840"), [])
        self.assertEqual(scan_scene_ids("scene_20260"), ["scene_20260"])

    def test_result_names_unique(self):
        self.assertEqual(len(RESULT_FILES), len(set(RESULT_FILES)))

    def test_no_network_client_import(self):
        source = Path(
            "freeze_stable_official_archive_acquisition_e9.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        self.assertTrue(
            {"requests", "urllib", "httpx", "aiohttp"}.isdisjoint(imports)
        )


if __name__ == "__main__":
    unittest.main()
