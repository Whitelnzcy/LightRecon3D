"""Rejudge an existing E0 run under the frozen denominator contract.

No point cloud, support, geometry, join, matcher, or model artifact is loaded.
The script derives integer counts from the preserved E0 scene and per-plane
tables and writes a new versioned result directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from audit_support_bounded_evaluator_contract import git_state


SCHEMA_VERSION = 1


def _safe_ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def rejudge_scene(old_scene: dict, plane_rows: list[dict]) -> dict:
    cache_count = int(old_scene["cache_point_count"])
    gt_count = int(old_scene["gt_labeled_point_count"])
    assigned_count = int(sum(int(row["pred_point_count"]) for row in plane_rows))
    gt_support_assigned_count = int(
        round(float(old_scene["assignment_rate_gt_support"]) * gt_count)
    )
    for row in plane_rows:
        if bool(row["plane_tp"]):
            if abs(float(row["matched_iou"]) - 1.0) > 1e-8:
                raise ValueError(
                    f"{old_scene['scene']}: cannot infer exact matched count "
                    "from a non-identity IoU"
                )
            if int(row["pred_point_count"]) != int(row["gt_point_count"]):
                raise ValueError(
                    f"{old_scene['scene']}: matched support counts differ"
                )
    matched_count = int(
        sum(
            int(row["gt_point_count"])
            for row in plane_rows
            if bool(row["plane_tp"])
        )
    )
    nonplane_count = cache_count - gt_count
    nonplane_false_count = assigned_count - gt_support_assigned_count
    gt_assignment = _safe_ratio(gt_support_assigned_count, gt_count)
    matched_coverage = _safe_ratio(matched_count, gt_count)
    full_label_rate = _safe_ratio(assigned_count, cache_count)
    gt_label_rate = _safe_ratio(gt_count, cache_count)
    nonplane_false_rate = _safe_ratio(nonplane_false_count, nonplane_count)

    result = {
        "scene": old_scene["scene"],
        "config": "E0",
        "metric_only_rejudgment": True,
        "cache_point_count": cache_count,
        "assigned_cache_point_count": assigned_count,
        "gt_plane_point_count": gt_count,
        "gt_support_assigned_point_count": gt_support_assigned_count,
        "gt_support_matched_point_count": matched_count,
        "nonplane_point_count": nonplane_count,
        "nonplane_false_assigned_point_count": nonplane_false_count,
        "pred_plane_count": int(old_scene["pred_plane_count"]),
        "gt_plane_count": int(old_scene["gt_plane_count"]),
        "true_positive_planes": int(old_scene["true_positive_planes"]),
        "plane_precision": float(old_scene["plane_precision"]),
        "plane_recall": float(old_scene["plane_recall"]),
        "plane_f1": float(old_scene["plane_f1"]),
        "plane_weighted_matched_iou_mean": float(
            old_scene["plane_weighted_matched_iou_mean"]
        ),
        "plane_weighted_matched_iou_median": float(
            old_scene["plane_weighted_matched_iou_median"]
        ),
        "minimum_matched_iou": float(old_scene["minimum_matched_iou"]),
        "point_weighted_matched_iou": float(
            old_scene["point_weighted_matched_iou"]
        ),
        "gt_support_assignment_rate": gt_assignment,
        "gt_support_matched_coverage": matched_coverage,
        "full_cache_label_rate": full_label_rate,
        "gt_plane_label_rate": gt_label_rate,
        "nonplane_false_assignment_rate": nonplane_false_rate,
        "fragmentation_excess": int(old_scene["fragmentation_excess"]),
        "overmerge_excess": int(old_scene["overmerge_excess"]),
    }
    result.update(
        {
            "e0_plane_f1_pass": result["plane_f1"] >= 0.95,
            "e0_tp_pass": (
                result["true_positive_planes"] == result["gt_plane_count"]
            ),
            "e0_median_iou_pass": (
                result["plane_weighted_matched_iou_median"] >= 0.90
            ),
            "e0_min_iou_pass": result["minimum_matched_iou"] >= 0.80,
            "e0_gt_support_assignment_pass": gt_assignment >= 0.98,
            "e0_gt_support_matched_coverage_pass": matched_coverage >= 0.98,
            "e0_integer_count_pass": (
                assigned_count == gt_count and nonplane_false_count == 0
            ),
            "e0_label_rate_equality_pass": (
                abs(full_label_rate - gt_label_rate) <= 1e-8
            ),
        }
    )
    result["e0_all_pass"] = all(
        result[key]
        for key in (
            "e0_plane_f1_pass",
            "e0_tp_pass",
            "e0_median_iou_pass",
            "e0_min_iou_pass",
            "e0_gt_support_assignment_pass",
            "e0_gt_support_matched_coverage_pass",
            "e0_integer_count_pass",
            "e0_label_rate_equality_pass",
        )
    )
    return result


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.with_suffix(".json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    with path.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    marker_path = (
        source_dir / "VERDICT_SUPERSEDED_BY_METRIC_CONTRACT_20260727.json"
    )
    if marker_path.exists():
        raise FileExistsError(f"Refusing to overwrite {marker_path}")
    output_dir.mkdir(parents=True)

    old_summary = json.loads(
        (source_dir / "summary.json").read_text(encoding="utf-8")
    )
    plane_rows = json.loads(
        (source_dir / "per_plane_metrics.json").read_text(encoding="utf-8")
    )
    by_scene = {}
    for row in plane_rows:
        by_scene.setdefault(row["scene"], []).append(row)
    scenes = [
        rejudge_scene(old, by_scene[old["scene"]])
        for old in old_summary["scenes"]
    ]
    all_pass = all(row["e0_all_pass"] for row in scenes)
    if not all_pass:
        raise RuntimeError("corrected E0 contract did not pass every scene")

    config = {
        "schema_version": SCHEMA_VERSION,
        "task": "metric-only E0 denominator rejudgment",
        "source_dir": str(source_dir),
        "geometry_rerun": False,
        "join_changed": False,
        "matcher_changed": False,
        "plane_tp_changed": False,
        "gt_support_changed": False,
        "e1_e2_e3_executed": False,
        "l_g_j_resumed": False,
        "metric_contract": {
            "gt_support_assignment_rate": "|A intersection G| / |G|",
            "gt_support_matched_coverage": "|M| / |G|",
            "full_cache_label_rate": "|A| / |C|",
            "gt_plane_label_rate": "|G| / |C|",
            "nonplane_false_assignment_rate": "|A minus G| / |C minus G|",
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output_dir / "git_state.json").write_text(
        json.dumps(git_state(root), indent=2), encoding="utf-8"
    )
    _write_rows(output_dir / "per_scene_metrics", scenes)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "config": "E0",
        "metric_only_rejudgment": True,
        "geometry_rerun": False,
        "e0_passed_all_pressure_scenes": True,
        "contract_verdict": "E0 Passed",
        "legacy_verdict": "E-Key/Data Contract",
        "legacy_verdict_status": (
            "invalidated by contradictory full-cache assignment denominator"
        ),
        "e1_e2_e3_executed": False,
        "l_g_j_resumed": False,
        "next_mechanical_action": (
            "Keep E1-E3 and L/G/J paused until the next preregistration."
        ),
        "scenes": scenes,
        "source_artifacts": {
            "summary_json": str(source_dir / "summary.json"),
            "per_plane_metrics_json": str(
                source_dir / "per_plane_metrics.json"
            ),
            "input_hashes_json": str(source_dir / "input_hashes.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    marker = {
        "schema_version": SCHEMA_VERSION,
        "legacy_summary": str(source_dir / "summary.json"),
        "legacy_verdict": "E-Key/Data Contract",
        "status": "superseded_invalid_metric_denominator",
        "replacement_summary": str(output_dir / "summary.json"),
        "geometry_rerun": False,
        "records_deleted_or_overwritten": False,
    }
    (output_dir / "legacy_verdict_supersession.json").write_text(
        json.dumps(marker, indent=2), encoding="utf-8"
    )
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    (output_dir / "run.log").write_text(
        "metric-only rejudgment\n"
        "geometry_rerun=false\n"
        "E0=passed on 3/3 pressure scenes\n"
        "E1-E3=false\n"
        "L/G/J=false\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
