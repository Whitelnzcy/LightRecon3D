import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from execute_official_archive_acquisition_e8 import (
    SCENE_RE,
    parse_central_directory,
    safe_member_relative,
    scene_ids,
)


class E8ExecutionTests(unittest.TestCase):
    def test_frozen_scene_scanner(self):
        cases = {
            "scene_00184": ["scene_00184"],
            "scene_00184_room_20": ["scene_00184"],
            "scene_20260728": [],
            "scene_20260": ["scene_20260"],
            "scene_202607": [],
            "xscene_00184": ["scene_00184"],
            "scene_001840": [],
        }
        self.assertEqual(SCENE_RE.pattern, r"scene_(\d{5})(?!\d)")
        for value, expected in cases.items():
            self.assertEqual(scene_ids(value), expected)

    def test_safe_member_path(self):
        self.assertEqual(
            safe_member_relative(
                "Structured3D/scene_01234/2D_rendering/a.png",
                "scene_01234",
            ),
            Path("2D_rendering/a.png"),
        )
        for unsafe in (
            "/scene_01234/a.png",
            "../scene_01234/a.png",
            "scene_01234/../../escape.png",
        ):
            with self.assertRaises(ValueError):
                safe_member_relative(unsafe, "scene_01234")

    def test_classic_central_directory_parser(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("scene_01234/a.txt", b"hello")
            archive.writestr("scene_01234/b.txt", b"world")
        payload = buffer.getvalue()
        eocd = payload.rfind(b"PK\x05\x06")
        entries = int.from_bytes(payload[eocd + 10 : eocd + 12], "little")
        size = int.from_bytes(payload[eocd + 12 : eocd + 16], "little")
        offset = int.from_bytes(payload[eocd + 16 : eocd + 20], "little")
        members = parse_central_directory(
            payload[offset : offset + size],
            entries,
        )
        self.assertEqual([row["path"] for row in members], [
            "scene_01234/a.txt",
            "scene_01234/b.txt",
        ])
        self.assertEqual([row["uncompressed_size"] for row in members], [5, 5])


if __name__ == "__main__":
    unittest.main()
