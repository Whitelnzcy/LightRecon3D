import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from materialize_variable_view_inputs_e14 import (
    E14Blocked,
    pair_indices,
    select_scene_views,
    stage1_to_global_index_map,
)


def write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(path)


class E14MaterializationTests(unittest.TestCase):
    def test_variable_view_selection_and_room_tie_break(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "scene_00001"
            for room in ("200", "100"):
                for view in range(4):
                    write_rgb(
                        root
                        / "2D_rendering"
                        / room
                        / "perspective"
                        / "empty"
                        / str(view)
                        / "rgb_rawlight.png"
                    )
            selected = select_scene_views(root)
            self.assertEqual(selected["view_count"], 4)
            self.assertEqual(selected["selected_room"], "2D_rendering/100")
            self.assertEqual(selected["view_ids"], ["0", "1", "2", "3"])

    def test_five_view_cap_and_insufficient_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "scene_00001"
            for view in range(6):
                write_rgb(
                    root
                    / "2D_rendering"
                    / "100"
                    / "perspective"
                    / "empty"
                    / str(view)
                    / "rgb_rawlight.png"
                )
            self.assertEqual(select_scene_views(root)["view_count"], 5)
            second = Path(temporary) / "scene_00002"
            for view in range(3):
                write_rgb(
                    second
                    / "2D_rendering"
                    / "100"
                    / "perspective"
                    / "empty"
                    / str(view)
                    / "rgb_rawlight.png"
                )
            with self.assertRaises(E14Blocked) as caught:
                select_scene_views(second)
            self.assertEqual(caught.exception.verdict, "Blocked_Insufficient_Views")

    def test_pair_indices_support_four_and_five_only(self):
        self.assertEqual(pair_indices(4).shape, (6, 2))
        self.assertEqual(pair_indices(5).shape, (10, 2))
        with self.assertRaises(ValueError):
            pair_indices(3)

    def test_stage1_global_provenance_is_in_range(self):
        mapped = stage1_to_global_index_map(
            (720, 1280), (288, 512), 1000, mask_hw=(128, 128)
        )
        self.assertEqual(mapped.shape, (128, 128))
        self.assertGreaterEqual(mapped.min(), 1000)
        self.assertLess(mapped.max(), 1000 + 288 * 512)
        self.assertEqual(len(np.unique(mapped)), 128 * 128)


if __name__ == "__main__":
    unittest.main()
