import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from visualize_research_practice_final_3d import (
    GUIDED_METHOD,
    RANSAC_METHOD,
    CameraView,
    load_prediction,
    parse_views,
    run_visualization,
    scene_inputs,
    validate_identical_cache,
)


def room_cloud(count: int = 1800) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260717)
    per_face = count // 3
    # Floor, back wall, and side wall make a visible indoor corner.
    floor = np.column_stack(
        (rng.uniform(-2, 2, per_face), rng.uniform(-1.5, 1.5, per_face), np.zeros(per_face))
    )
    back = np.column_stack(
        (rng.uniform(-2, 2, per_face), np.full(per_face, 1.5), rng.uniform(0, 2.5, per_face))
    )
    side = np.column_stack(
        (np.full(per_face, -2.0), rng.uniform(-1.5, 1.5, per_face), rng.uniform(0, 2.5, per_face))
    )
    points = np.concatenate((floor, back, side), axis=0).astype(np.float32)
    colors = np.clip(
        np.column_stack(
            (
                130 + 30 * points[:, 0],
                155 + 25 * points[:, 1],
                180 + 20 * points[:, 2],
            )
        ),
        0,
        255,
    ).astype(np.uint8)
    return points, colors


def write_prediction(
    path: Path,
    method: str,
    points: np.ndarray,
    colors: np.ndarray,
    assignments: np.ndarray,
) -> None:
    plane_ids = np.unique(assignments[assignments >= 0]).astype(np.int32)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1),
        points=points,
        colors=colors,
        original_colors=colors,
        point_plane_ids=assignments.astype(np.int32),
        plane_ids=plane_ids,
        method=np.asarray(method),
        runtime_seconds=np.asarray(1.25, dtype=np.float64),
    )


class FinalVisualizationTest(unittest.TestCase):
    def test_parse_views(self) -> None:
        views = parse_views("a:-35:20,b:55:18")
        self.assertEqual([view.name for view in views], ["a", "b"])
        self.assertEqual(views[0].yaw_deg, -35.0)
        self.assertEqual(parse_views("auto"), [])
        with self.assertRaises(ValueError):
            parse_views("missing-fields")

    def test_scene_inputs_uses_recorded_final_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            execution = root / "batch_execution.json"
            execution.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "final_000_scene_00180",
                                "scene_name": "scene_00180",
                                "status": "pass",
                                "artifacts": {
                                    "global_ransac": {"path": "baseline.npz"},
                                    "learning_guided_ransac": {"path": "guided.npz"},
                                },
                            },
                            {
                                "id": "failed",
                                "scene_name": "scene_failed",
                                "status": "fail",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            scenes = scene_inputs(execution)
            self.assertEqual(len(scenes), 1)
            self.assertEqual(scenes[0].baseline_path, root / "baseline.npz")
            self.assertEqual(scenes[0].guided_path, root / "guided.npz")

    def test_identical_cache_check_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            points, colors = room_cloud()
            assignments = np.arange(len(points), dtype=np.int32) % 3
            baseline_path = root / "baseline.npz"
            guided_path = root / "guided.npz"
            write_prediction(baseline_path, RANSAC_METHOD, points, colors, assignments)
            changed = points.copy()
            changed[10, 0] += 0.01
            write_prediction(guided_path, GUIDED_METHOD, changed, colors, assignments)
            baseline = load_prediction(baseline_path, RANSAC_METHOD)
            guided = load_prediction(guided_path, GUIDED_METHOD)
            with self.assertRaisesRegex(ValueError, "identical ordered global point cache"):
                validate_identical_cache(baseline, guided)

    def test_end_to_end_render_creates_report_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            points, colors = room_cloud()
            thirds = len(points) // 3
            baseline_ids = np.concatenate(
                (
                    np.zeros(thirds, dtype=np.int32),
                    np.ones(thirds, dtype=np.int32),
                    np.full(len(points) - 2 * thirds, 2, dtype=np.int32),
                )
            )
            guided_ids = baseline_ids.copy()
            guided_ids[len(points) - 120 :] = 3
            baseline_path = root / "baseline.npz"
            guided_path = root / "guided.npz"
            write_prediction(baseline_path, RANSAC_METHOD, points, colors, baseline_ids)
            write_prediction(guided_path, GUIDED_METHOD, points, colors, guided_ids)

            execution = root / "batch_execution.json"
            execution.write_text(
                json.dumps(
                    {
                        "git_sha": "test",
                        "items": [
                            {
                                "id": "final_000_scene_00180",
                                "scene_name": "scene_00180",
                                "status": "pass",
                                "artifacts": {
                                    "global_ransac": {"path": str(baseline_path)},
                                    "learning_guided_ransac": {"path": str(guided_path)},
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audit = root / "final_method_audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "per_scene": [
                            {
                                "item_id": "final_000_scene_00180",
                                "ransac_pairwise_f1": 0.70,
                                "guided_pairwise_f1": 0.76,
                                "delta_pairwise_f1": 0.06,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "render"
            result = run_visualization(
                execution,
                output,
                audit,
                [CameraView("overview", -35, 20), CameraView("top", -35, 58)],
                width=320,
                height=240,
                max_points=1200,
                point_radius=1,
            )
            self.assertEqual(len(result["scenes"]), 1)
            self.assertTrue((output / "visualization_manifest.json").is_file())
            self.assertTrue((output / "README.md").is_file())
            contact_sheet = output / "all_scenes_final_3d_contact_sheet.png"
            self.assertTrue(contact_sheet.is_file())
            with Image.open(contact_sheet) as image:
                self.assertGreater(image.size[0], 500)
            multiview = Path(result["scenes"][0]["multiview_png"])
            self.assertTrue(multiview.is_file())
            with Image.open(multiview) as image:
                self.assertEqual(image.size[0], 640)
            self.assertIn("0.700 -> 0.760", result["scenes"][0]["metric_caption"])
            with self.assertRaises(FileExistsError):
                run_visualization(
                    execution,
                    output,
                    audit,
                    [CameraView("overview", -35, 20)],
                    width=320,
                    height=240,
                    max_points=1200,
                    point_radius=1,
                )


if __name__ == "__main__":
    unittest.main()
