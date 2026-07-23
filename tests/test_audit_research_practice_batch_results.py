import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_research_practice_batch_results import (
    DIRECT_DROP,
    DIRECT_RAW,
    MANUAL_DROP,
    MANUAL_RAW,
    RANSAC,
    audit_rows,
    index_rows,
    run_audit,
)


def method_row(item_id, scene, method, f1, coverage, assignment, overmerge):
    return {
        "item_id": item_id,
        "scene_name": scene,
        "metric_family": "support_records",
        "method": method,
        "support_partition_pairwise_f1": f1,
        "gt_labeled_record_coverage": coverage,
        "record_assignment_rate": assignment,
        "support_conditioned_plane_precision": 0.6,
        "support_conditioned_plane_recall_observed": 0.5,
        "support_conditioned_fragmentation_excess": 1,
        "support_conditioned_overmerge_excess": overmerge,
        "conflicting_positive_key_count": 10 if method == MANUAL_RAW else 0,
        "conflicting_positive_record_count": 40 if method == MANUAL_RAW else 0,
    }


def item_rows(item_id, scene, ransac_f1=0.6, manual_f1=0.7):
    return [
        method_row(item_id, scene, RANSAC, ransac_f1, 0.99, 0.8, 2),
        method_row(item_id, scene, MANUAL_RAW, manual_f1, 0.98, 0.99, 1),
        method_row(item_id, scene, MANUAL_DROP, 0.85, 0.55, 0.42, 1),
        method_row(item_id, scene, DIRECT_RAW, 0.1, 0.98, 0.99, 0),
        method_row(item_id, scene, DIRECT_DROP, 0.9, 0.001, 0.001, 0),
    ]


class SmokeGateTests(unittest.TestCase):
    def test_promising_smoke_is_not_promoted_before_eight_scenes(self):
        rows = []
        rows.extend(item_rows("a", "scene_1"))
        rows.extend(item_rows("b", "scene_1"))
        rows.extend(item_rows("c", "scene_2"))
        result = audit_rows(rows, minimum_independent_scenes=8)
        self.assertEqual(result["decision"], "smoke_signal_only_expand_independent_scenes")
        self.assertTrue(result["diagnostics"]["method_gate_passed"])
        self.assertFalse(result["diagnostics"]["scene_count_passed"])
        self.assertAlmostEqual(
            result["diagnostics"]["median_manual_raw_f1_gain_vs_ransac"], 0.1
        )
        self.assertTrue(result["diagnostics"]["manual_drop_coverage_collapse"])
        self.assertTrue(result["diagnostics"]["direct_drop_coverage_collapse"])

    def test_losing_most_groups_stops_method_promotion(self):
        rows = []
        rows.extend(item_rows("a", "scene_1", 0.6, 0.7))
        rows.extend(item_rows("b", "scene_2", 0.7, 0.6))
        rows.extend(item_rows("c", "scene_3", 0.8, 0.7))
        result = audit_rows(rows, minimum_independent_scenes=3)
        self.assertEqual(
            result["decision"], "stop_identity_method_promotion_use_strongest_baseline"
        )
        self.assertAlmostEqual(result["diagnostics"]["manual_raw_group_win_rate"], 1 / 3)

    def test_missing_required_method_is_rejected(self):
        rows = item_rows("a", "scene")
        rows = [row for row in rows if row["method"] != MANUAL_DROP]
        with self.assertRaisesRegex(ValueError, "missing methods"):
            index_rows(rows)

    def test_run_audit_rejects_failed_batch_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "metrics.json"
            batch = root / "batch.json"
            metrics.write_text(json.dumps(item_rows("a", "scene")), encoding="utf-8")
            batch.write_text(
                json.dumps(
                    {
                        "git_sha": "abc",
                        "items": [{"id": "a", "status": "fail"}],
                        "summary": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "failed items"):
                run_audit(metrics, batch, root / "output")
            output = root / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                run_audit(metrics, batch, output)


if __name__ == "__main__":
    unittest.main()
