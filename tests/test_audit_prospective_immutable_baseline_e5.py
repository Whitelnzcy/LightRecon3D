import json
import tempfile
import unittest
from pathlib import Path

from audit_prospective_immutable_baseline_e5 import (
    PRESSURE_SCENES,
    canonical_hash,
    scene_manifest,
)


class ProspectiveImmutableBaselineAuditTest(unittest.TestCase):
    def test_canonical_hash_ignores_mapping_order(self):
        self.assertEqual(
            canonical_hash({"a": 1, "b": 2}),
            canonical_hash({"b": 2, "a": 1}),
        )

    def test_scene_manifest_preserves_batch_order(self):
        batch = {
            "items": [
                {
                    "id": "final_000_scene_00180",
                    "scene_name": "scene_00180",
                    "pair_group": "room-a",
                },
                {
                    "id": "final_003_scene_00184",
                    "scene_name": "scene_00184",
                    "pair_group": "room-b",
                },
            ]
        }
        split = {
            "scenes": [
                {"scene": "final_003_scene_00184", "split": "val"},
                {"scene": "final_000_scene_00180", "split": "train"},
            ]
        }
        manifest = scene_manifest(batch, split)
        self.assertEqual(
            manifest["scene_order"],
            ["final_000_scene_00180", "final_003_scene_00184"],
        )
        self.assertFalse(manifest["scenes"][0]["pressure_scene"])
        self.assertTrue(manifest["scenes"][1]["pressure_scene"])

    def test_registered_pressure_scenes_are_exact(self):
        self.assertEqual(
            PRESSURE_SCENES,
            {
                "final_003_scene_00184",
                "final_013_scene_00195",
                "final_014_scene_00197",
            },
        )

    def test_blocked_schema_does_not_claim_prediction_freeze(self):
        payload = {
            "status": "Blocked_before_prediction",
            "actual_prediction_count": 0,
            "prediction_frozen_before_GT_evaluation": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction_manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["actual_prediction_count"], 0)
        self.assertIsNone(loaded["prediction_frozen_before_GT_evaluation"])


if __name__ == "__main__":
    unittest.main()
