import unittest
from pathlib import Path

from evaluate_guided_ransac_smoke import gate_rows, guided_command


def row(item_id, *, ransac_f1, guided_f1, ransac_runtime, guided_runtime,
        ransac_precision=0.6, guided_precision=0.6,
        ransac_overmerge=2.0, guided_overmerge=2.0):
    return {
        "id": item_id,
        "scene_name": f"scene_{item_id}",
        "status": "pass",
        "ransac_f1": ransac_f1,
        "guided_f1": guided_f1,
        "guided_coverage": 0.98,
        "ransac_plane_precision": ransac_precision,
        "guided_plane_precision": guided_precision,
        "ransac_overmerge": ransac_overmerge,
        "guided_overmerge": guided_overmerge,
        "ransac_runtime_seconds": ransac_runtime,
        "guided_runtime_seconds": guided_runtime,
    }


class GuidedRansacSmokeTests(unittest.TestCase):
    def test_quality_path_requires_cross_group_gain(self):
        rows = [
            row(str(index), ransac_f1=0.6, guided_f1=0.63,
                ransac_runtime=10, guided_runtime=9,
                guided_precision=0.7, guided_overmerge=1)
            for index in range(3)
        ]
        result = gate_rows(rows)
        self.assertTrue(result["quality_gate_passed"])
        self.assertFalse(result["efficiency_gate_passed"])
        self.assertEqual(
            result["decision"], "guided_smoke_signal_expand_independent_scenes"
        )

    def test_efficiency_path_allows_small_bounded_f1_loss(self):
        rows = [
            row(str(index), ransac_f1=0.6, guided_f1=0.595,
                ransac_runtime=10, guided_runtime=5)
            for index in range(3)
        ]
        result = gate_rows(rows)
        self.assertFalse(result["quality_gate_passed"])
        self.assertTrue(result["efficiency_gate_passed"])
        self.assertTrue(result["method_gate_passed"])

    def test_failed_tradeoff_keeps_guided_method_as_ablation(self):
        rows = [
            row(str(index), ransac_f1=0.7, guided_f1=0.6,
                ransac_runtime=10, guided_runtime=9,
                guided_precision=0.5, guided_overmerge=3)
            for index in range(3)
        ]
        result = gate_rows(rows)
        self.assertFalse(result["method_gate_passed"])
        self.assertEqual(result["decision"], "keep_guided_as_ablation_ransac_primary")

    def test_command_uses_direct_support_and_frozen_cache(self):
        command = guided_command(
            "/env/python",
            Path("/project"),
            item_id="group",
            cache_path=Path("/cache/global.npz"),
            support_path=Path("/support/direct.npz"),
            output_dir=Path("/output"),
        )
        self.assertIn("--global_cloud_npz", command)
        self.assertIn(str(Path("/cache/global.npz")), command)
        self.assertIn("--support_npz", command)
        self.assertIn(str(Path("/support/direct.npz")), command)
        self.assertIn("--fallback_iterations", command)


if __name__ == "__main__":
    unittest.main()
