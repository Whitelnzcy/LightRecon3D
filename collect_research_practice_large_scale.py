"""Collect a large research-practice run into one auditable result bundle."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from research_practice_batch import file_sha256


SCHEMA_VERSION = 1


def load_json(path: Path, expected_type: type) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, expected_type):
        raise ValueError(f"unexpected JSON type in {path}")
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_by_name(audit: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next(
        (row for row in audit.get("metric_summary", []) if row.get("metric") == name),
        None,
    )


def artifact_path(item: dict[str, Any], key: str) -> str:
    record = item.get("artifacts", {}).get(key, {})
    return str(record.get("path", "")) if isinstance(record, dict) else ""


def collect_bundle(
    selection_plan_json: Path,
    materialization_json: Path,
    preflight_json: Path,
    batch_execution_json: Path,
    audit_json: Path,
    visualization_manifest_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    selection = load_json(selection_plan_json, dict)
    materialization = load_json(materialization_json, dict)
    preflight = load_json(preflight_json, dict)
    batch = load_json(batch_execution_json, dict)
    audit = load_json(audit_json, dict)
    visualization = load_json(visualization_manifest_json, dict)

    batch_sha256 = file_sha256(batch_execution_json)
    if audit.get("source_batch_execution_sha256") != batch_sha256:
        raise ValueError("audit does not refer to the supplied batch execution")
    if visualization.get("source_batch_execution_sha256") != batch_sha256:
        raise ValueError("visualization does not refer to the supplied batch execution")
    if materialization.get("selection_plan_sha256") != file_sha256(selection_plan_json):
        raise ValueError("materialization does not refer to the supplied selection plan")

    batch_items = {str(row["id"]): row for row in batch.get("items", [])}
    visual_items = {str(row["item_id"]): row for row in visualization.get("scenes", [])}
    scene_rows: list[dict[str, Any]] = []
    for metric in audit.get("per_scene", []):
        item_id = str(metric["item_id"])
        item = batch_items.get(item_id)
        visual = visual_items.get(item_id)
        if item is None or visual is None:
            raise ValueError(f"passed audit item lacks batch/visual output: {item_id}")
        scene_rows.append(
            {
                "item_id": item_id,
                "scene_name": str(metric["scene_name"]),
                "ransac_pairwise_f1": float(metric["ransac_pairwise_f1"]),
                "guided_pairwise_f1": float(metric["guided_pairwise_f1"]),
                "delta_pairwise_f1": float(metric["delta_pairwise_f1"]),
                "ransac_runtime_seconds": float(metric["ransac_runtime_seconds"]),
                "guided_runtime_seconds": float(metric["guided_runtime_seconds"]),
                "global_cloud_cache": artifact_path(item, "global_cloud_cache"),
                "ransac_npz": artifact_path(item, "global_ransac"),
                "guided_npz": artifact_path(item, "learning_guided_ransac"),
                "multiview_png": str(visual["multiview_png"]),
            }
        )

    failures: list[dict[str, str]] = []
    for source, payload in (("materialization", materialization), ("batch", batch)):
        for row in payload.get("items", []):
            if row.get("status") == "pass":
                continue
            failures.append(
                {
                    "source": source,
                    "item_id": str(row.get("id", "")),
                    "scene_name": str(row.get("scene_name", "")),
                    "failure_stage": str(row.get("failure_stage", "")),
                    "error": str(row.get("error", "")),
                }
            )

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "research_practice_large_scale_result_bundle",
        "sources": {
            "selection_plan_json": str(selection_plan_json),
            "materialization_json": str(materialization_json),
            "preflight_json": str(preflight_json),
            "batch_execution_json": str(batch_execution_json),
            "audit_json": str(audit_json),
            "visualization_manifest_json": str(visualization_manifest_json),
        },
        "source_sha256": {
            "selection_plan": file_sha256(selection_plan_json),
            "materialization": file_sha256(materialization_json),
            "preflight": file_sha256(preflight_json),
            "batch_execution": batch_sha256,
            "audit": file_sha256(audit_json),
            "visualization": file_sha256(visualization_manifest_json),
        },
        "selection_summary": selection.get("summary", {}),
        "dataset_inventory": selection.get("dataset_inventory", {}),
        "materialization_summary": materialization.get("summary", {}),
        "preflight_summary": preflight.get("summary", {}),
        "batch_summary": batch.get("summary", {}),
        "gate": audit.get("gate", {}),
        "metric_summary": audit.get("metric_summary", []),
        "contact_sheet_png": str(visualization.get("contact_sheet_png", "")),
        "scenes": scene_rows,
        "failures": failures,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "large_scale_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(
        output_dir / "scene_artifact_index.csv",
        scene_rows,
        [
            "item_id",
            "scene_name",
            "ransac_pairwise_f1",
            "guided_pairwise_f1",
            "delta_pairwise_f1",
            "ransac_runtime_seconds",
            "guided_runtime_seconds",
            "global_cloud_cache",
            "ransac_npz",
            "guided_npz",
            "multiview_png",
        ],
    )
    write_csv(
        output_dir / "failures.csv",
        failures,
        ["source", "item_id", "scene_name", "failure_stage", "error"],
    )
    (output_dir / "large_scale_summary.md").write_text(
        markdown_summary(result), encoding="utf-8"
    )
    return result


def markdown_summary(result: dict[str, Any]) -> str:
    selected = result["selection_summary"].get("selected_scenes", 0)
    batch = result["batch_summary"]
    diagnostics = result.get("gate", {}).get("diagnostics", {})
    lines = [
        "# Research-practice large-scale result",
        "",
        (
            f"Selected {selected} eligible validation scenes. "
            f"The final batch passed {batch.get('passed_items', 0)}/"
            f"{batch.get('items', 0)} scenes; {len(result['failures'])} failure records are retained."
        ),
        "",
        f"Decision: `{result.get('gate', {}).get('decision', '')}`",
        "",
        "| Metric | Ordinary RANSAC | Guided RANSAC | Delta | Valid scenes |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("pairwise_f1", "matched_iou", "overmerge_excess", "runtime_seconds"):
        row = metric_by_name({"metric_summary": result["metric_summary"]}, name)
        if row is None:
            continue
        lines.append(
            f"| {name} | {row['ransac_mean']:.6f} | {row['guided_mean']:.6f} | "
            f"{row['mean_delta_guided_minus_ransac']:+.6f} | {row['valid_scene_pairs']} |"
        )
    lines.extend(
        [
            "",
            (
                f"Guided F1 wins: {diagnostics.get('guided_scene_wins', 0)}/"
                f"{diagnostics.get('view_groups', 0)}; median paired gain: "
                f"{diagnostics.get('median_guided_f1_gain', float('nan')):+.6f}."
            ),
            "",
            f"3D contact sheet: `{result['contact_sheet_png']}`",
            "",
            "Per-scene NPZ and PNG paths are indexed in `scene_artifact_index.csv`. Failed items are preserved in `failures.csv` and are not included in paired metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser("Collect a large-scale experiment result bundle")
    parser.add_argument("--selection_plan_json", required=True)
    parser.add_argument("--materialization_json", required=True)
    parser.add_argument("--preflight_json", required=True)
    parser.add_argument("--batch_execution_json", required=True)
    parser.add_argument("--audit_json", required=True)
    parser.add_argument("--visualization_manifest_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    result = collect_bundle(
        Path(args.selection_plan_json),
        Path(args.materialization_json),
        Path(args.preflight_json),
        Path(args.batch_execution_json),
        Path(args.audit_json),
        Path(args.visualization_manifest_json),
        Path(args.output_dir),
    )
    print(
        json.dumps(
            {
                "passed_scenes": len(result["scenes"]),
                "failures": len(result["failures"]),
                "contact_sheet": result["contact_sheet_png"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
