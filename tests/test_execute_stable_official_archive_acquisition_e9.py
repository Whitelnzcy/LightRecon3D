import unittest

from execute_stable_official_archive_acquisition_e9 import (
    ALLOWED_VERDICTS,
    canonical_members,
    classify_reference,
    hard_identity,
)


class E9ExecutionTests(unittest.TestCase):
    def test_canonical_listing_includes_archive_name(self):
        members = [{
            "path": "scene_01234/a.txt",
            "compressed_size": 4,
            "uncompressed_size": 5,
            "crc32": "1234abcd",
            "compression_method": 8,
            "local_header_offset": 7,
        }]
        row = canonical_members("a.zip", members)[0]
        self.assertEqual(row["archive_name"], "a.zip")
        self.assertEqual(row["normalized_member_path"], "scene_01234/a.txt")

    def test_hard_identity_excludes_optional_headers(self):
        base = {
            "final_url": "https://example/a.zip",
            "total_length": 10,
            "strong_etag": '"abc"',
            "fixed_head_range_sha256": "1",
            "fixed_tail_range_sha256": "2",
            "central_directory_canonical_sha256": "3",
            "last_modified": "old",
            "content_type": "application/zip",
        }
        changed = dict(base, last_modified="new", content_type=None)
        self.assertEqual(hard_identity(base), hard_identity(changed))

    def test_plan_only_classification(self):
        classification, _reason = classify_reference(
            __import__("pathlib").Path(
                "results/official_archive_acquisition_e8_20260728_v1/"
                "candidate_order_128.json"
            ),
            "repository",
        )
        self.assertEqual(classification, "acquisition_plan_only")

    def test_verdicts_are_frozen(self):
        self.assertIn("Ready_Clean_Raw_Pool", ALLOWED_VERDICTS)
        self.assertIn("Blocked_Remote_Object_Unstable", ALLOWED_VERDICTS)
        self.assertNotIn("Blocked_Official_Metadata_Changed", ALLOWED_VERDICTS)


if __name__ == "__main__":
    unittest.main()
