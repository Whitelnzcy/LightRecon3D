import tempfile
import unittest
import zlib
from pathlib import Path

import execute_stable_official_archive_acquisition_e9 as e9_executor
from execute_range_member_acquisition_e11 import (
    ALLOWED_VERDICTS,
    canonical_members,
    decode_local_name,
    scene_from_path,
    sha256_bytes,
)


class E11ExecutionTests(unittest.TestCase):
    def test_scene_from_member_path(self):
        self.assertEqual(
            scene_from_path(
                "Structured3D/scene_00426/2D_rendering/1/"
                "perspective/empty/0/rgb_rawlight.png"
            ),
            "scene_00426",
        )
        self.assertIsNone(scene_from_path("Structured3D/README.txt"))

    def test_local_name_encoding(self):
        self.assertEqual(decode_local_name(b"a/b.png", 0), "a/b.png")
        self.assertEqual(decode_local_name("目录.png".encode("utf-8"), 0x800), "目录.png")

    def test_hash_is_deterministic(self):
        self.assertEqual(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def test_registered_verdicts(self):
        self.assertIn("Ready_Clean_Raw_Pool", ALLOWED_VERDICTS)
        self.assertIn("Paused_Resumable", ALLOWED_VERDICTS)
        self.assertIn("Blocked_Source_Mapping", ALLOWED_VERDICTS)

    def test_frozen_central_directory_uses_e9_serialization(self):
        rows = canonical_members(
            "a.zip",
            [{
                "path": "Structured3D/scene_00001/a.png",
                "compressed_size": 1,
                "uncompressed_size": 2,
                "crc32": "12345678",
                "compression_method": 8,
                "local_header_offset": 3,
            }],
        )
        self.assertEqual(
            sha256_bytes(e9_executor.canonical_json(rows)),
            "3cd8464b5e303c16ae4a85af1063192ebe64a1fa8c1e780c263aad86a692bcc8",
        )


if __name__ == "__main__":
    unittest.main()
