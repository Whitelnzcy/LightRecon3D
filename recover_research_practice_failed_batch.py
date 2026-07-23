"""Prepare and merge a non-destructive retry of failed large-scale scenes."""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from execute_research_practice_batch import write_batch_outputs
from materialize_research_practice_final_inputs import summary as materialization_summary
from research_practice_batch import DEFAULT_PATTERN, file_sha256, result_summary


SCHEMA_VERSION = 1


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def index_items(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("id", ""))
        if not item_id or item_id in output:
            raise ValueError(f"{label} has an empty or duplicate item ID: {item_id!r}")
        output[item_id] = row
    return output


def prepare_recovery(
    selection_plan_json: Path,
    materialization_json: Path,
    retry_input_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing recovery plan: {output_dir}")
    selection = load_json(selection_plan_json)
    materialization = load_json(materialization_json)
    if materialization.get("selection_plan_sha256") != file_sha256(selection_plan_json):
        raise ValueError("materialization does not refer to the supplied selection plan")
    selected = index_items(selection.get("items", []), "selection plan")
    attempts = index_items(materialization.get("items", []), "materialization")
    failed_ids = [
        item_id for item_id, row in attempts.items() if row.get("status") != "pass"
    ]
    if not failed_ids:
        raise ValueError("the source materialization contains no failed items")

    retry_items: list[dict[str, Any]] = []
    manifest_items: list[dict[str, Any]] = []
    for item_id in failed_ids:
        if item_id not in selected:
            raise ValueError(f"failed materialization item is absent from selection: {item_id}")
        original = selected[item_id]
        retry = copy.deepcopy(original)
        retry["input_dir"] = str(retry_input_root / item_id / "stage2_merge")
        retry["materialization"] = "needs_stage1_stage2"
        retry["recovery_source_input_dir"] = str(original.get("input_dir", ""))
        retry["recovery_source_failure"] = {
            "failure_stage": str(attempts[item_id].get("failure_stage", "")),
            "error": str(attempts[item_id].get("error", "")),
        }
        retry_items.append(retry)
        manifest_item = {
            "id": item_id,
            "input_dir": retry["input_dir"],
            "expected_scene_name": str(retry["scene_name"]),
            "expected_pair_group": str(retry["pair_group"]),
        }
        reusable = retry.get("reusable_cache")
        if isinstance(reusable, dict):
            manifest_item.update(reusable)
        manifest_items.append(manifest_item)

    created = datetime.now(timezone.utc).isoformat()
    retry_plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": "research_practice_final_scene_selection",
        "created_utc": created,
        "configuration": {
            "mode": "failed_item_recovery",
            "retry_input_root": str(retry_input_root),
            "source_selection_plan": str(selection_plan_json),
            "source_selection_plan_sha256": file_sha256(selection_plan_json),
            "source_materialization": str(materialization_json),
            "source_materialization_sha256": file_sha256(materialization_json),
        },
        "dataset_inventory": selection.get("dataset_inventory", {}),
        "summary": {
            "selected_scenes": len(retry_items),
            "unique_scene_names": len({str(row["scene_name"]) for row in retry_items}),
            "unique_pair_groups": len({str(row["pair_group"]) for row in retry_items}),
            "reused_stage2_groups": 0,
            "groups_needing_stage1_stage2": len(retry_items),
            "reused_global_caches": sum(bool(row.get("reusable_cache")) for row in retry_items),
        },
        "warnings": [],
        "items": retry_items,
    }
    retry_manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": "research_practice_failed_scene_recovery",
        "minimum_valid_items": 1,
        "min_views": 5,
        "check_image_files": True,
        "pattern": DEFAULT_PATTERN,
        "items": manifest_items,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / "recovery_selection_plan.json"
    manifest_path = output_dir / "recovery_execute.json"
    plan_path.write_text(json.dumps(retry_plan, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(retry_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result = {
        "failed_items": failed_ids,
        "recovery_selection_plan": str(plan_path),
        "recovery_manifest": str(manifest_path),
        "retry_input_root": str(retry_input_root),
    }
    (output_dir / "recovery_plan_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def replacement_rows(
    original_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    label: str,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    original = index_items(original_rows, f"original {label}")
    recovery = index_items(recovery_rows, f"recovery {label}")
    if not set(recovery).issubset(original):
        raise ValueError(f"recovery {label} contains item IDs absent from the original")
    already_passed = [
        item_id for item_id in recovery if original[item_id].get("status") == "pass"
    ]
    if already_passed:
        raise ValueError(
            f"recovery {label} attempts to replace originally passed items: {already_passed}"
        )
    combined: list[dict[str, Any]] = []
    recovered = 0
    unresolved: list[str] = []
    for item_id, source in original.items():
        attempt = recovery.get(item_id)
        if attempt is not None and attempt.get("status") == "pass":
            row = copy.deepcopy(attempt)
            row["recovered_from_failure"] = {
                "status": source.get("status"),
                "failure_stage": source.get("failure_stage", ""),
                "error": source.get("error", ""),
                "input_dir": source.get("input_dir", ""),
            }
            recovered += 1
        else:
            row = copy.deepcopy(source)
            if attempt is not None:
                row["recovery_attempt"] = copy.deepcopy(attempt)
                if source.get("status") != "pass":
                    unresolved.append(item_id)
        combined.append(row)
    return combined, recovered, unresolved


def merge_recovery(
    selection_plan_json: Path,
    original_materialization_json: Path,
    recovery_materialization_json: Path,
    original_preflight_json: Path,
    recovery_preflight_json: Path,
    original_batch_json: Path,
    recovery_batch_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite merged recovery output: {output_dir}")
    original_materialization = load_json(original_materialization_json)
    recovery_materialization = load_json(recovery_materialization_json)
    original_preflight = load_json(original_preflight_json)
    recovery_preflight = load_json(recovery_preflight_json)
    original_batch = load_json(original_batch_json)
    recovery_batch = load_json(recovery_batch_json)
    selection_sha = file_sha256(selection_plan_json)
    if original_materialization.get("selection_plan_sha256") != selection_sha:
        raise ValueError("original materialization does not match the selection plan")
    original_frozen = original_materialization.get("frozen_inputs", {})
    recovery_frozen = recovery_materialization.get("frozen_inputs", {})
    for name, record in original_frozen.items():
        if recovery_frozen.get(name, {}).get("sha256") != record.get("sha256"):
            raise ValueError(f"recovery changed frozen materialization input: {name}")
    if original_batch.get("weights_sha256") != recovery_batch.get("weights_sha256"):
        raise ValueError("original and recovery batches used different DUSt3R weights")
    if original_batch.get("frozen_config") != recovery_batch.get("frozen_config"):
        raise ValueError("original and recovery batches used different frozen configs")
    if original_batch.get("coordinate_contract") != recovery_batch.get("coordinate_contract"):
        raise ValueError("original and recovery batches used different coordinate contracts")

    material_items, recovered_material, unresolved_material = replacement_rows(
        original_materialization.get("items", []),
        recovery_materialization.get("items", []),
        "materialization",
    )
    preflight_items, _, _ = replacement_rows(
        original_preflight.get("items", []),
        recovery_preflight.get("items", []),
        "preflight",
    )
    batch_items, recovered_batch, unresolved_batch = replacement_rows(
        original_batch.get("items", []), recovery_batch.get("items", []), "batch"
    )

    sources = {
        "selection_plan": {"path": str(selection_plan_json), "sha256": selection_sha},
        "original_materialization": {
            "path": str(original_materialization_json),
            "sha256": file_sha256(original_materialization_json),
        },
        "recovery_materialization": {
            "path": str(recovery_materialization_json),
            "sha256": file_sha256(recovery_materialization_json),
        },
        "original_batch": {
            "path": str(original_batch_json),
            "sha256": file_sha256(original_batch_json),
        },
        "recovery_batch": {
            "path": str(recovery_batch_json),
            "sha256": file_sha256(recovery_batch_json),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)

    combined_materialization = copy.deepcopy(original_materialization)
    combined_materialization["selection_plan"] = str(selection_plan_json)
    combined_materialization["selection_plan_sha256"] = selection_sha
    combined_materialization["recovery"] = sources
    combined_materialization["items"] = material_items
    combined_materialization["summary"] = materialization_summary(material_items)
    combined_materialization_path = output_dir / "combined_materialization.json"
    combined_materialization_path.write_text(
        json.dumps(combined_materialization, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    combined_preflight = copy.deepcopy(original_preflight)
    combined_preflight["recovery"] = sources
    combined_preflight["items"] = preflight_items
    minimum = int(original_preflight.get("summary", {}).get("minimum_valid_items", 1))
    combined_preflight["summary"] = result_summary(preflight_items, minimum)
    combined_preflight_path = output_dir / "combined_preflight.json"
    combined_preflight_path.write_text(
        json.dumps(combined_preflight, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    combined_batch = copy.deepcopy(original_batch)
    combined_batch["batch_name"] = f"{original_batch.get('batch_name', 'batch')}_recovered"
    combined_batch["recovery"] = sources
    combined_batch["items"] = batch_items
    combined_batch_dir = output_dir / "combined_batch"
    combined_batch_dir.mkdir()
    write_batch_outputs(combined_batch, combined_batch_dir)

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "research_practice_failed_scene_recovery_merge",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recovered_materialization_items": recovered_material,
        "recovered_batch_items": recovered_batch,
        "unresolved_materialization_items": unresolved_material,
        "unresolved_batch_items": unresolved_batch,
        "combined_materialization_json": str(combined_materialization_path),
        "combined_preflight_json": str(combined_preflight_path),
        "combined_batch_execution_json": str(combined_batch_dir / "batch_execution.json"),
        "combined_aggregate_metrics_json": str(combined_batch_dir / "aggregate_metrics.json"),
        "combined_batch_summary": combined_batch["summary"],
        "sources": sources,
    }
    (output_dir / "recovery_merge.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser("Recover failed research-practice batch scenes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--selection_plan_json", required=True)
    prepare.add_argument("--materialization_json", required=True)
    prepare.add_argument("--retry_input_root", required=True)
    prepare.add_argument("--output_dir", required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--selection_plan_json", required=True)
    merge.add_argument("--original_materialization_json", required=True)
    merge.add_argument("--recovery_materialization_json", required=True)
    merge.add_argument("--original_preflight_json", required=True)
    merge.add_argument("--recovery_preflight_json", required=True)
    merge.add_argument("--original_batch_json", required=True)
    merge.add_argument("--recovery_batch_json", required=True)
    merge.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_recovery(
            Path(args.selection_plan_json),
            Path(args.materialization_json),
            Path(args.retry_input_root),
            Path(args.output_dir),
        )
    else:
        result = merge_recovery(
            Path(args.selection_plan_json),
            Path(args.original_materialization_json),
            Path(args.recovery_materialization_json),
            Path(args.original_preflight_json),
            Path(args.recovery_preflight_json),
            Path(args.original_batch_json),
            Path(args.recovery_batch_json),
            Path(args.output_dir),
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
