import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark_research_practice_efficiency import (
    aggregate_accuracy,
    archived_stage_timings,
    first_alignment_images,
    latency_summary,
    stage1_accuracy_rows,
    support_partition_metrics,
)


class EfficiencyBenchmarkTests(unittest.TestCase):
    def test_support_partition_perfect(self):
        metrics = support_partition_metrics(
            np.asarray([0, 0, 1, 1]), np.asarray([4, 4, 8, 8])
        )
        self.assertEqual(metrics["pairwise_f1"], 1.0)
        self.assertEqual(metrics["purity_completeness_f1"], 1.0)
        self.assertEqual(metrics["gt_labeled_coverage"], 1.0)

    def test_support_partition_separates_quality_from_coverage(self):
        metrics = support_partition_metrics(
            np.asarray([0, 0, -1, -1]), np.asarray([4, 4, 8, 8])
        )
        self.assertEqual(metrics["pairwise_f1"], 1.0)
        self.assertEqual(metrics["gt_labeled_coverage"], 0.5)
        fragmented = support_partition_metrics(
            np.asarray([0, 1, 2, 3]), np.asarray([4, 4, 8, 8])
        )
        self.assertEqual(fragmented["pairwise_recall"], 0.0)

    def test_latency_summary_uses_interpolated_percentiles(self):
        summary = latency_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["p50_ms"], 2.5)
        self.assertAlmostEqual(summary["p95_ms"], 3.85)

    def test_stage1_accuracy_discovers_sources_from_stage2_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items = []
            for index in range(8):
                stage1 = root / f"stage1_{index}.npz"
                np.savez(
                    stage1,
                    point_plane_ids=np.asarray([0, 0, 1, 1], dtype=np.int32),
                    gt_point_plane_ids=np.asarray([2, 2, 3, 3], dtype=np.int32),
                    sample_idx=np.asarray(index, dtype=np.int32),
                )
                stage2 = root / f"stage2_{index}"
                stage2.mkdir()
                (stage2 / "learned_region_merge_manifest.json").write_text(
                    json.dumps({"files": [{"input": str(stage1)}]}),
                    encoding="utf-8",
                )
                items.append(
                    {
                        "id": f"item_{index}",
                        "input_dir": str(stage2),
                        "expected_scene_name": f"scene_{index}",
                    }
                )
            manifest = root / "final.json"
            manifest.write_text(json.dumps({"items": items}), encoding="utf-8")
            rows, sources = stage1_accuracy_rows(manifest)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(sources), 8)
            self.assertEqual(rows[0]["pairwise_f1"], 1.0)
            overall = aggregate_accuracy(rows)[0]
            self.assertEqual(overall["records"], 8)
            self.assertEqual(overall["pairwise_f1_mean"], 1.0)

    def test_stage1_accuracy_rejects_duplicate_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "stage1.npz"
            np.savez(
                source,
                point_plane_ids=np.asarray([0, 0], dtype=np.int32),
                gt_point_plane_ids=np.asarray([1, 1], dtype=np.int32),
            )
            items = []
            for index in range(8):
                stage2 = root / f"stage2_{index}"
                stage2.mkdir()
                (stage2 / "learned_region_merge_manifest.json").write_text(
                    json.dumps({"files": [{"input": str(source)}]}),
                    encoding="utf-8",
                )
                items.append(
                    {
                        "id": f"item_{index}",
                        "input_dir": str(stage2),
                        "expected_scene_name": f"scene_{index}",
                    }
                )
            manifest = root / "final.json"
            manifest.write_text(json.dumps({"items": items}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate Stage1 source"):
                stage1_accuracy_rows(manifest)

    def test_archived_stage_timing_keeps_uncached_proxy_explicit(self):
        batch = {
            "items": [
                {
                    "id": "a",
                    "scene_name": "scene_a",
                    "reused_global_cloud_cache": None,
                    "stages": [
                        {"stage": "direct_support", "runtime_seconds": 10.0},
                        {"stage": "global_ransac", "runtime_seconds": 2.0},
                    ],
                },
                {
                    "id": "b",
                    "scene_name": "scene_b",
                    "reused_global_cloud_cache": {"path": "/cache"},
                    "stages": [
                        {"stage": "direct_support", "runtime_seconds": 1.0},
                        {"stage": "global_ransac", "runtime_seconds": 4.0},
                    ],
                },
            ]
        }
        rows, summary = archived_stage_timings(batch)
        self.assertEqual(len(rows), 4)
        proxy = next(
            row
            for row in summary
            if row["stage"] == "direct_support_uncached_alignment_plus_export_proxy"
        )
        self.assertEqual(proxy["scenes"], 1)
        self.assertEqual(proxy["mean_seconds"], 10.0)

    def test_alignment_images_use_global_cache_view_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = []
            registry = []
            for index in (1, 0):
                image = root / f"view_{index}.png"
                image.write_bytes(b"image")
                images.append(image)
                registry.append(
                    {
                        "alignment_view_index": index,
                        "image_path": str(image),
                        "points_hw": [10, 10],
                    }
                )
            cache = root / "cache.npz"
            np.savez(
                cache,
                dust3r_view_registry_json=np.asarray(json.dumps(registry)),
            )
            batch = {
                "items": [
                    {
                        "id": "item",
                        "scene_name": "scene",
                        "artifacts": {
                            "global_cloud_cache": {
                                "path": str(cache),
                                "sha256": "abc",
                            }
                        },
                    }
                ]
            }
            paths, source = first_alignment_images(batch)
            self.assertEqual(paths, [str(root / "view_0.png"), str(root / "view_1.png")])
            self.assertEqual(source["image_path_source"], "dust3r_view_registry_json")


if __name__ == "__main__":
    unittest.main()
