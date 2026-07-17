import json
import tempfile
import unittest
from pathlib import Path

from audit_research_practice_final_results import (
    DIRECT_DROP,
    FULL_CACHE,
    GUIDED,
    MANUAL_DROP,
    MANUAL_RAW,
    RANSAC,
    SUPPORT_RECORDS,
    audit_final_results,
    bootstrap_mean_ci,
    exact_two_sided_sign_pvalue,
    final_gate,
    metric_summary,
    paired_rows,
)


def full_row(item_id, scene, method, f1, runtime, *, coverage=0.99,
             precision=0.6, overmerge=1.0):
    return {
        "item_id": item_id,
        "scene_name": scene,
        "metric_family": FULL_CACHE,
        "method": method,
        "support_partition_pairwise_f1": f1,
        "support_partition_purity_completeness_f1": f1 + 0.05,
        "support_conditioned_matched_iou": f1 - 0.05,
        "support_conditioned_plane_precision": precision,
        "support_conditioned_plane_recall_all_gt": 0.55,
        "support_coverage": coverage,
        "support_conditioned_normal_angular_error_deg": 3.0,
        "support_conditioned_fragmentation_excess": 1.0,
        "support_conditioned_overmerge_excess": overmerge,
        "runtime_seconds": runtime,
    }


def batch(scene_count=8, *, failed=False):
    return {
        "git_sha": "abc123",
        "items": [
            {
                "id": f"item_{index}",
                "scene_name": f"scene_{index}",
                "status": "fail" if failed and index == 0 else "pass",
            }
            for index in range(scene_count)
        ],
    }


def rows(scene_count=8, *, guided_gain=0.03, guided_runtime=9.0):
    output = []
    for index in range(scene_count):
        item = f"item_{index}"
        scene = f"scene_{index}"
        output.extend(
            [
                full_row(item, scene, RANSAC, 0.60, 10.0, overmerge=2.0),
                full_row(
                    item,
                    scene,
                    GUIDED,
                    0.60 + guided_gain,
                    guided_runtime,
                    precision=0.7,
                    overmerge=1.0,
                ),
                {
                    "item_id": item,
                    "scene_name": scene,
                    "metric_family": SUPPORT_RECORDS,
                    "method": MANUAL_RAW,
                    "gt_labeled_record_coverage": 0.99,
                },
                {
                    "item_id": item,
                    "scene_name": scene,
                    "metric_family": SUPPORT_RECORDS,
                    "method": MANUAL_DROP,
                    "gt_labeled_record_coverage": 0.50,
                },
                {
                    "item_id": item,
                    "scene_name": scene,
                    "metric_family": SUPPORT_RECORDS,
                    "method": DIRECT_DROP,
                    "gt_labeled_record_coverage": 0.001,
                },
                {
                    "item_id": item,
                    "scene_name": scene,
                    "metric_family": FULL_CACHE,
                    "method": MANUAL_DROP,
                    "support_coverage": 0.01,
                },
                {
                    "item_id": item,
                    "scene_name": scene,
                    "metric_family": FULL_CACHE,
                    "method": DIRECT_DROP,
                    "support_coverage": 0.0001,
                },
            ]
        )
    return output


class FinalAuditTests(unittest.TestCase):
    def test_quality_path_promotes_only_after_eight_unique_scenes(self):
        per_scene = paired_rows(rows(), batch())
        gate = final_gate(per_scene)
        self.assertEqual(gate["decision"], "promote_learning_guided_ransac_final")
        self.assertTrue(gate["quality_gate_passed"])
        self.assertFalse(gate["efficiency_gate_passed"])
        self.assertEqual(gate["diagnostics"]["guided_scene_wins"], 8)

        short_gate = final_gate(paired_rows(rows(7), batch(7)))
        self.assertEqual(
            short_gate["decision"],
            "insufficient_independent_scenes_for_final_decision",
        )
        self.assertFalse(short_gate["method_gate_passed"])

    def test_failed_quality_and_efficiency_retains_baseline(self):
        gate = final_gate(
            paired_rows(rows(guided_gain=0.005, guided_runtime=9.0), batch())
        )
        self.assertEqual(
            gate["decision"], "retain_global_ransac_primary_guided_ablation"
        )
        self.assertFalse(gate["method_gate_passed"])

    def test_efficiency_path_can_promote_noninferior_method(self):
        gate = final_gate(
            paired_rows(rows(guided_gain=-0.005, guided_runtime=5.0), batch())
        )
        self.assertTrue(gate["efficiency_gate_passed"])
        self.assertEqual(gate["decision"], "promote_learning_guided_ransac_final")

    def test_duplicate_scene_and_failed_batch_are_rejected(self):
        duplicate = batch()
        duplicate["items"][1]["scene_name"] = duplicate["items"][0]["scene_name"]
        with self.assertRaisesRegex(ValueError, "duplicate independent scene"):
            paired_rows(rows(), duplicate)
        with self.assertRaisesRegex(ValueError, "failed items"):
            paired_rows(rows(), batch(failed=True))

    def test_failed_items_can_be_retained_for_large_batch_audit(self):
        metric_rows = rows()
        source_batch = batch()
        source_batch["items"].append(
            {
                "id": "failed_item",
                "scene_name": "failed_scene",
                "status": "fail",
                "failure_stage": "input_preflight",
            }
        )
        paired = paired_rows(
            metric_rows, source_batch, allow_failed_items=True
        )
        self.assertEqual(len(paired), 8)

    def test_bootstrap_and_sign_test_are_deterministic(self):
        first = bootstrap_mean_ci([0.01, 0.02, 0.03], samples=1000, seed=7)
        second = bootstrap_mean_ci([0.01, 0.02, 0.03], samples=1000, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], 0.02)
        self.assertAlmostEqual(exact_two_sided_sign_pvalue(8, 0), 0.0078125)

    def test_nonfinite_optional_normal_metric_does_not_block_gate(self):
        metric_rows = rows()
        guided = next(
            row
            for row in metric_rows
            if row.get("item_id") == "item_0" and row.get("method") == GUIDED
        )
        guided["support_conditioned_normal_angular_error_deg"] = float("nan")
        per_scene = paired_rows(metric_rows, batch())
        self.assertIsNone(per_scene[0]["guided_normal_error_deg"])
        normal = next(
            row for row in metric_summary(per_scene) if row["metric"] == "normal_error_deg"
        )
        self.assertEqual(normal["valid_scene_pairs"], 7)
        self.assertTrue(final_gate(per_scene)["quality_gate_passed"])

    def test_run_writes_auditable_outputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "aggregate_metrics.json"
            execution = root / "batch_execution.json"
            output = root / "audit"
            metrics.write_text(json.dumps(rows()), encoding="utf-8")
            execution.write_text(json.dumps(batch()), encoding="utf-8")
            result = audit_final_results(metrics, execution, output)
            self.assertTrue(result["gate"]["method_gate_passed"])
            self.assertEqual(
                result["coverage_diagnostics"][
                    "manual_drop_full_cache_coverage_median"
                ],
                0.01,
            )
            self.assertTrue((output / "final_method_audit.md").is_file())
            self.assertTrue((output / "final_method_per_scene.csv").is_file())
            with self.assertRaises(FileExistsError):
                audit_final_results(metrics, execution, output)


if __name__ == "__main__":
    unittest.main()
