import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_immutable_support_rebaseline_e4 import (
    PARAMETER_HASH_KEYS,
    SUPPORT_HASH_KEYS,
    audit_method_inputs,
    registered_hashes,
    sha256_arrays,
)


class ImmutableSupportRebaselineAuditTest(unittest.TestCase):
    def _prediction(self, root: Path, scene: str) -> Path:
        path = root / f"{scene}_all_fragment_identity.npz"
        np.savez_compressed(
            path,
            points=np.zeros((3, 3), dtype=np.float32),
            point_plane_ids=np.asarray([0, 0, -1], dtype=np.int32),
            plane_ids=np.asarray([0], dtype=np.int32),
            plane_normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            plane_offsets=np.asarray([0.0], dtype=np.float32),
            alignment_view_indices=np.asarray([0, 0, 1], dtype=np.int32),
            pointmap_pixel_xy=np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        )
        return path

    def test_current_hash_is_not_a_historical_hash(self):
        self.assertEqual(registered_hashes({"schema_version": 1}), {})

    def test_array_hash_changes_with_support_but_not_unrelated_arrays(self):
        arrays = {
            "alignment_view_indices": np.asarray([0, 1], dtype=np.int32),
            "pointmap_pixel_xy": np.asarray([[1, 2], [3, 4]], dtype=np.int32),
            "point_plane_ids": np.asarray([0, 1], dtype=np.int32),
            "plane_ids": np.asarray([0, 1], dtype=np.int32),
            "plane_normals": np.eye(2, 3, dtype=np.float32),
            "plane_offsets": np.asarray([0.0, 1.0], dtype=np.float32),
        }
        first = sha256_arrays(arrays, SUPPORT_HASH_KEYS)
        arrays["plane_offsets"][0] = 9.0
        self.assertEqual(first, sha256_arrays(arrays, SUPPORT_HASH_KEYS))
        self.assertNotEqual(
            sha256_arrays(arrays, SUPPORT_HASH_KEYS),
            sha256_arrays(
                {**arrays, "point_plane_ids": np.asarray([1, 1], dtype=np.int32)},
                SUPPORT_HASH_KEYS,
            ),
        )
        self.assertEqual(len(sha256_arrays(arrays, PARAMETER_HASH_KEYS)), 64)

    def test_missing_registered_hash_blocks_historical_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            self._prediction(prediction_dir, "scene_a")
            historical_summary = root / "summary.json"
            historical_summary.write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
            bounded_summary = root / "bounded_summary.json"
            bounded_summary.write_text(
                json.dumps(
                    {
                        "config": {
                            "support_prediction_dir": str(
                                prediction_dir.resolve()
                            )
                        }
                    }
                ),
                encoding="utf-8",
            )
            audit = audit_method_inputs(
                "B",
                prediction_dir,
                historical_summary,
                bounded_summary,
                expected_scene_count=1,
            )
            self.assertTrue(audit["bounded_summary_references_prediction_dir"])
            self.assertTrue(audit["all_required_keys_present"])
            self.assertTrue(audit["all_current_files_read_stable"])
            self.assertFalse(audit["all_historical_hashes_registered"])
            self.assertFalse(audit["all_historical_hashes_match"])

    def test_registered_exact_hash_passes_archival_hash_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            prediction = self._prediction(prediction_dir, "scene_a")
            file_hash = hashlib.sha256(prediction.read_bytes()).hexdigest()
            historical_summary = root / "summary.json"
            historical_summary.write_text(
                json.dumps(
                    {
                        "historical_input_hashes": {
                            "scene_a": {"file_sha256": file_hash}
                        }
                    }
                ),
                encoding="utf-8",
            )
            bounded_summary = root / "bounded_summary.json"
            bounded_summary.write_text(
                json.dumps(
                    {
                        "config": {
                            "support_prediction_dir": str(
                                prediction_dir.resolve()
                            )
                        }
                    }
                ),
                encoding="utf-8",
            )
            audit = audit_method_inputs(
                "M",
                prediction_dir,
                historical_summary,
                bounded_summary,
                expected_scene_count=1,
            )
            self.assertTrue(audit["all_historical_hashes_registered"])
            self.assertTrue(audit["all_historical_hashes_match"])


if __name__ == "__main__":
    unittest.main()
