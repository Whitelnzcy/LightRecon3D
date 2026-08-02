import unittest

from audit_prospective_freeze_e6 import (
    HISTORICAL_SCENES,
    PRESSURE_SCENES,
    REQUIRED_SPLIT_COUNTS,
    build_inventory,
    deterministic_split,
    quarantine_records,
    split_sort_key,
)


class ProspectiveFreezeE6AuditTest(unittest.TestCase):
    def test_registered_split_is_deterministic_and_disjoint(self):
        scenes = [f"scene_{index:05d}" for index in range(100)]
        first = deterministic_split(scenes)
        second = deterministic_split(reversed(scenes))
        self.assertEqual(first, second)
        self.assertEqual(
            {name: len(values) for name, values in first.items()},
            REQUIRED_SPLIT_COUNTS,
        )
        sets = [set(values) for values in first.values()]
        self.assertEqual(len(set.union(*sets)), 77)
        self.assertEqual(sum(map(len, sets)), 77)

    def test_registered_hash_sort_uses_canonical_scene_id(self):
        self.assertEqual(split_sort_key("scene_00042"), split_sort_key("x_scene_00042_y"))

    def test_fewer_than_77_candidates_is_blocked(self):
        with self.assertRaisesRegex(ValueError, "Need 77 clean candidates"):
            deterministic_split([f"scene_{index:05d}" for index in range(76)])

    def test_quarantine_includes_history_pressure_and_unknowns(self):
        rows = quarantine_records([*HISTORICAL_SCENES, "scene_00001"])
        by_scene = {row["scene_id"]: row for row in rows}
        self.assertTrue(set(PRESSURE_SCENES).issubset(by_scene))
        self.assertEqual(
            by_scene["scene_00184"]["evidence_class"],
            "explicit_historical_record",
        )
        self.assertEqual(
            by_scene["scene_00001"]["evidence_class"],
            "conservative_unproven_provenance",
        )
        self.assertTrue(all(row["quarantined"] for row in rows))

    def test_inventory_cannot_promote_quarantined_complete_input(self):
        quarantine = quarantine_records(["scene_00180"])
        rows = build_inventory(
            ["scene_00180"],
            quarantine,
            {"scene_00180": 5},
            {"scene_00180": ["cache.npz"]},
            {"scene_00180": ["support.npz"]},
        )
        self.assertTrue(rows[0]["required_input_presence_complete"])
        self.assertFalse(rows[0]["clean_candidate"])
        self.assertFalse(rows[0]["GT_content_loaded"])


if __name__ == "__main__":
    unittest.main()
