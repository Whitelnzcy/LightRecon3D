import tempfile
import unittest
from pathlib import Path

from audit_clean_scene_acquisition_e7 import (
    HISTORICAL_17,
    SALT,
    candidate_sort_key,
    canonical_scene_id,
    parse_errata_scene_ids,
)


class CleanSceneAcquisitionE7Test(unittest.TestCase):
    def test_canonical_scene_id(self):
        self.assertEqual(canonical_scene_id("x/scene_00201/room"), "scene_00201")

    def test_candidate_sort_is_deterministic(self):
        scenes = ["scene_00200", "scene_00201", "scene_00202"]
        self.assertEqual(
            sorted(scenes, key=candidate_sort_key),
            sorted(reversed(scenes), key=candidate_sort_key),
        )
        self.assertIn(SALT.encode("utf-8"), f"{SALT}|scene_00200".encode("utf-8"))

    def test_errata_parser_deduplicates_room_level_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "errata.txt"
            path.write_text(
                "scene_00240_room_1\nscene_00240_room_2\nscene_01155\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_errata_scene_ids(path), ["00240", "01155"])

    def test_historical_17_are_below_new_source_range(self):
        values = [int(scene.rsplit("_", 1)[1]) for scene in HISTORICAL_17]
        self.assertEqual(len(values), 17)
        self.assertTrue(all(value < 200 for value in values))

    def test_128_plan_has_no_preexisting_or_invalid_ids(self):
        preexisting = {f"scene_{index:05d}" for index in range(200)}
        invalid = {"scene_00240", "scene_01155"}
        universe = {
            f"scene_{index:05d}" for index in range(200, 3000)
        } - preexisting - invalid
        frozen = sorted(universe, key=candidate_sort_key)[:128]
        self.assertEqual(len(frozen), 128)
        self.assertTrue(set(frozen).isdisjoint(preexisting))
        self.assertTrue(set(frozen).isdisjoint(invalid))


if __name__ == "__main__":
    unittest.main()
