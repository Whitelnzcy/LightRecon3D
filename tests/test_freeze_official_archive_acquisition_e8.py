import ast
import unittest
from pathlib import Path

from freeze_official_archive_acquisition_e8 import (
    REGISTRATION_STATE,
    RESULT_FILES,
    scan_scene_id,
)


class OfficialArchiveAcquisitionE8PreregTest(unittest.TestCase):
    def test_exact_scene_scanner_contract(self):
        cases = {
            "scene_00184": "scene_00184",
            "scene_00184_room_20": "scene_00184",
            "scene_20260728": None,
            "scene_20260": "scene_20260",
            "scene_202607": None,
            "xscene_00184": "scene_00184",
            "scene_001840": None,
        }
        self.assertEqual({key: scan_scene_id(key) for key in cases}, cases)

    def test_registration_is_not_authorized(self):
        self.assertEqual(REGISTRATION_STATE, "PRE_REGISTERED_NOT_AUTHORIZED")

    def test_required_output_names_are_unique(self):
        self.assertEqual(len(RESULT_FILES), 36)
        self.assertEqual(len(set(RESULT_FILES)), len(RESULT_FILES))

    def test_freezer_imports_no_http_client(self):
        source = Path("freeze_official_archive_acquisition_e8.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(
            {"requests", "httpx", "urllib", "aiohttp"}.isdisjoint(imported)
        )


if __name__ == "__main__":
    unittest.main()
