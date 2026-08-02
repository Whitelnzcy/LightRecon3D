#!/usr/bin/env python3
"""Evaluate Plane-DUSt3R and LightRecon3D baselines on identical point caches."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from adapt_plane_dust3r_to_point_cache import adapt_room
from evaluate_global_plane_baselines import evaluate_arrays


LIGHT_METHODS = ("global_ransac_cc", "learning_guided_ransac_cc")
COMMON_METRICS = (
    "plane_precision",
    "plane_recall",
    "plane_f1",
    "support_matched_iou",
    "support_coverage",
    "normal_angular_error_deg",
    "point_to_plane_residual",
    "segmentation_voi_nats",
    "segmentation_rand_index",
    "segmentation_covering_symmetric",
    "support_partition_pairwise_f1",
    "support_partition_purity_completeness_f1",
)
REFERENCE_METHOD = "plane_dust3r_same_input_common_point_cache"
PAIRED_DIRECTIONS = {
    "plane_f1": "higher",
    "support_matched_iou": "higher",
    "segmentation_voi_nats": "lower",
    "segmentation_rand_index": "higher",
    "segmentation_covering_symmetric": "higher",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_recorded_path(value: str, project_dir: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def plane_f1(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def normalized_metric_row(
    row: dict[str, Any], *, scene_name: str, item_id: str
) -> dict[str, Any]:
    output = dict(row)
    output["scene_name"] = scene_name
    output["item_id"] = item_id
    output["plane_f1"] = plane_f1(
        float(output["plane_precision"]), float(output["plane_recall"])
    )
    return output


def evaluate_plane_prediction(gt_path: Path, prediction_path: Path) -> dict[str, Any]:
    gt = np.load(gt_path, allow_pickle=False)
    prediction = np.load(prediction_path, allow_pickle=False)
    try:
        if prediction["points"].shape != gt["points"].shape or not np.allclose(
            prediction["points"], gt["points"], atol=1e-5
        ):
            raise ValueError("adapted prediction and GT do not use identical points")
        result = evaluate_arrays(
            gt["points"],
            prediction["point_plane_ids"],
            prediction["plane_normals"],
            prediction["plane_offsets"],
            gt["point_plane_ids"],
            gt["plane_normals"],
            match_iou=0.5,
            fragmentation_iou=0.1,
            min_observed_plane_points=64,
        )
        result["method"] = str(prediction["method"].item())
        return result
    finally:
        gt.close()
        prediction.close()


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    output: list[dict[str, Any]] = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        summary: dict[str, Any] = {"method": method, "scenes": len(selected)}
        for metric in COMMON_METRICS:
            values = [
                float(row[metric])
                for row in selected
                if metric in row and np.isfinite(float(row[metric]))
            ]
            summary[f"{metric}_mean"] = mean(values) if values else float("nan")
            summary[f"{metric}_median"] = median(values) if values else float("nan")
        output.append(summary)
    return output


def paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method_scene = {
        (str(row["method"]), str(row["scene_name"])): row for row in rows
    }
    scenes = sorted(
        scene
        for method, scene in by_method_scene
        if method == REFERENCE_METHOD
    )
    output: list[dict[str, Any]] = []
    for baseline in LIGHT_METHODS:
        for metric, direction in PAIRED_DIRECTIONS.items():
            deltas = []
            improvements = []
            for scene in scenes:
                reference = float(
                    by_method_scene[(REFERENCE_METHOD, scene)][metric]
                )
                comparison = float(by_method_scene[(baseline, scene)][metric])
                if not np.isfinite(reference) or not np.isfinite(comparison):
                    continue
                delta = reference - comparison
                deltas.append(delta)
                improvements.append(delta if direction == "higher" else -delta)
            tolerance = 1e-12
            output.append(
                {
                    "reference_method": REFERENCE_METHOD,
                    "baseline_method": baseline,
                    "metric": metric,
                    "direction": direction,
                    "paired_scenes": len(improvements),
                    "raw_delta_mean": mean(deltas) if deltas else float("nan"),
                    "raw_delta_median": median(deltas) if deltas else float("nan"),
                    "improvement_mean": (
                        mean(improvements) if improvements else float("nan")
                    ),
                    "wins": sum(value > tolerance for value in improvements),
                    "ties": sum(abs(value) <= tolerance for value in improvements),
                    "losses": sum(value < -tolerance for value in improvements),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_batch(
    *,
    research_batch_json: Path,
    plane_batch_manifest: Path,
    output_dir: Path,
    project_dir: Path,
    min_points: int = 64,
) -> dict[str, Any]:
    research_batch_json = research_batch_json.resolve()
    plane_batch_manifest = plane_batch_manifest.resolve()
    output_dir = output_dir.resolve()
    project_dir = project_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)

    research = load_json(research_batch_json)
    plane_batch = load_json(plane_batch_manifest)
    if plane_batch.get("status") != "pass":
        raise ValueError("Plane-DUSt3R source batch did not pass")
    if plane_batch.get("native_protocol_claim_allowed") is not False:
        raise ValueError("Plane-DUSt3R batch lacks the non-native protocol guard")
    plane_by_scene = {
        str(item["scene_name"]): item for item in plane_batch.get("items", [])
    }
    runtime_dir = plane_batch_manifest.parent / "runtime"

    common_rows: list[dict[str, Any]] = []
    item_ledger: list[dict[str, Any]] = []
    for item in research.get("items", []):
        item_id = str(item["id"])
        scene_name = str(item["scene_name"])
        started = time.perf_counter()
        ledger: dict[str, Any] = {
            "item_id": item_id,
            "scene_name": scene_name,
            "status": "fail",
            "error": "",
        }
        try:
            if item.get("status") != "pass":
                raise ValueError("source research-practice item did not pass")
            if scene_name not in plane_by_scene:
                raise ValueError("scene missing from Plane-DUSt3R batch")
            artifacts = item["artifacts"]
            gt_path = resolve_recorded_path(
                artifacts["point_aligned_gt"]["path"], project_dir
            )
            cloud_path = resolve_recorded_path(
                artifacts["global_cloud_cache"]["path"], project_dir
            )
            light_metrics_path = resolve_recorded_path(
                artifacts["full_cache_metrics"]["path"], project_dir
            )
            plane_room_dir = Path(plane_by_scene[scene_name]["room_dir"]).resolve()
            scene_dir = output_dir / "items" / item_id
            prediction_path = scene_dir / "plane_dust3r_common_point_cache.npz"
            adapter_manifest_path = scene_dir / "adapter_manifest.json"
            adapter_result = adapt_room(
                global_cloud_npz=cloud_path,
                gt_npz=gt_path,
                plane_room_dir=plane_room_dir,
                runtime_dir=runtime_dir,
                output_npz=prediction_path,
                min_points=min_points,
            )
            adapter_manifest_path.write_text(
                json.dumps(adapter_result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            plane_metrics = normalized_metric_row(
                evaluate_plane_prediction(gt_path, prediction_path),
                scene_name=scene_name,
                item_id=item_id,
            )
            common_rows.append(plane_metrics)

            light_rows = load_json(light_metrics_path)
            selected_methods = {
                str(row["method"]): row
                for row in light_rows
                if str(row.get("method", "")) in LIGHT_METHODS
            }
            missing = sorted(set(LIGHT_METHODS) - set(selected_methods))
            if missing:
                raise ValueError(
                    f"missing LightRecon3D methods: {', '.join(missing)}"
                )
            for method in LIGHT_METHODS:
                common_rows.append(
                    normalized_metric_row(
                        selected_methods[method],
                        scene_name=scene_name,
                        item_id=item_id,
                    )
                )
            ledger.update(
                {
                    "status": "pass",
                    "gt_npz": str(gt_path),
                    "global_cloud_npz": str(cloud_path),
                    "plane_room_dir": str(plane_room_dir),
                    "adapted_prediction_npz": str(prediction_path),
                    "adapter_manifest": str(adapter_manifest_path),
                    "assigned_rate": adapter_result["assigned_rate"],
                    "predicted_planes": adapter_result["planes"],
                }
            )
        except Exception as error:
            ledger["error"] = f"{type(error).__name__}: {error}"
        ledger["runtime_seconds"] = time.perf_counter() - started
        item_ledger.append(ledger)

    aggregate_rows = aggregate(common_rows)
    paired_rows = paired_comparisons(common_rows)
    write_csv(output_dir / "per_scene_common_metrics.csv", common_rows)
    write_csv(output_dir / "aggregate_common_metrics.csv", aggregate_rows)
    write_csv(output_dir / "paired_common_comparisons.csv", paired_rows)
    (output_dir / "per_scene_common_metrics.json").write_text(
        json.dumps(common_rows, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "aggregate_common_metrics.json").write_text(
        json.dumps(aggregate_rows, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "paired_common_comparisons.json").write_text(
        json.dumps(paired_rows, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    passed = sum(item["status"] == "pass" for item in item_ledger)
    result = {
        "schema_version": 1,
        "kind": "plane_dust3r_lightrecon3d_common_point_cache_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed == len(item_ledger) and passed > 0 else "fail",
        "protocol": {
            "comparison_domain": "identical_lightrecon3d_ordered_global_point_cache",
            "join_key": "(alignment_view_index,x,y)",
            "spatial_nearest_neighbor_join": False,
            "match_iou": 0.5,
            "fragmentation_iou": 0.1,
            "min_observed_plane_points": 64,
            "min_predicted_plane_points": min_points,
            "plane_dust3r_source_protocol": plane_batch.get("protocol"),
            "native_paper_protocol_claim_allowed": False,
        },
        "research_batch_json": str(research_batch_json),
        "plane_batch_manifest": str(plane_batch_manifest),
        "runtime_dir": str(runtime_dir),
        "requested_scenes": len(item_ledger),
        "passed_scenes": passed,
        "failed_scenes": len(item_ledger) - passed,
        "methods": [row["method"] for row in aggregate_rows],
        "items": item_ledger,
        "aggregate": aggregate_rows,
        "paired_comparisons": paired_rows,
    }
    (output_dir / "common_evaluation_manifest.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research_batch_json", required=True)
    parser.add_argument("--plane_batch_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_dir", default=".")
    parser.add_argument("--min_points", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_batch(
        research_batch_json=Path(args.research_batch_json),
        plane_batch_manifest=Path(args.plane_batch_manifest),
        output_dir=Path(args.output_dir),
        project_dir=Path(args.project_dir),
        min_points=args.min_points,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed_scenes": result["passed_scenes"],
                "failed_scenes": result["failed_scenes"],
                "aggregate": result["aggregate"],
            },
            indent=2,
            allow_nan=True,
        )
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
