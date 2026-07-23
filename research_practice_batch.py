"""Manifest preflight and auditable summaries for research-practice batches.

This module deliberately performs no GPU work.  It validates the exact Stage2
inputs, image paths, optional frozen artifacts and checksums before the batch
executor is allowed to create expensive global-alignment outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_stage3_view_registry import validate_input_dir


SCHEMA_VERSION = 1
DEFAULT_PATTERN = "*_learned_region_merge_full_pointcloud_editable_planes_data.npz"


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, expected_sha256: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": 0,
        "sha256": "",
        "expected_sha256": str(expected_sha256 or "").lower(),
        "checksum_matches": None,
    }
    if row["exists"]:
        row["bytes"] = int(path.stat().st_size)
        row["sha256"] = file_sha256(path)
        if row["expected_sha256"]:
            row["checksum_matches"] = row["sha256"].lower() == row["expected_sha256"]
    return row


def load_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version={raw.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    items = raw.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a non-empty items list")
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"items[{index}] must be an object")
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            raise ValueError(f"items[{index}] has no id")
        if item_id in seen:
            raise ValueError(f"Duplicate item id: {item_id}")
        seen.add(item_id)
        if not str(item.get("input_dir", "")).strip():
            raise ValueError(f"Item {item_id} has no input_dir")
    return raw


def validate_artifacts(specs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(specs, dict):
        return records, ["artifacts must be an object"], warnings
    for name in sorted(specs):
        spec = specs[name]
        if isinstance(spec, str):
            spec = {"path": spec, "required": True}
        if not isinstance(spec, dict) or not str(spec.get("path", "")).strip():
            errors.append(f"artifact {name} must define a path")
            continue
        required = bool(spec.get("required", True))
        row = file_record(Path(str(spec["path"])), str(spec.get("sha256", "")))
        row.update({"name": name, "required": required})
        records.append(row)
        if not row["exists"]:
            message = f"artifact {name} is missing: {row['path']}"
            (errors if required else warnings).append(message)
        elif row["checksum_matches"] is False:
            errors.append(f"artifact {name} checksum mismatch: {row['path']}")
    return records, errors, warnings


def preflight_item(
    item: dict[str, Any],
    *,
    default_pattern: str,
    default_min_views: int,
    check_image_files: bool,
) -> dict[str, Any]:
    item_id = str(item["id"])
    input_dir = Path(str(item["input_dir"]))
    pattern = str(item.get("pattern", default_pattern))
    min_views = int(item.get("min_views", default_min_views))
    errors: list[str] = []
    warnings: list[str] = []
    input_files: list[dict[str, Any]] = []
    registry: dict[str, Any] = {"summary": {}, "groups": []}

    if not input_dir.is_dir():
        errors.append(f"input_dir is missing: {input_dir}")
    else:
        paths = sorted(input_dir.glob(pattern))
        if not paths:
            errors.append(f"no Stage2 NPZ matched {input_dir / pattern}")
        else:
            input_files = [file_record(path) for path in paths]
            try:
                registry = validate_input_dir(
                    input_dir, pattern, check_files=check_image_files
                )
            except Exception as error:  # preserve a failure row for the batch
                errors.append(f"view-registry validation failed: {error}")

    summary = registry.get("summary", {})
    groups = registry.get("groups", [])
    if registry.get("groups") is not None and len(groups) != 1:
        errors.append(f"expected exactly one scene/pair group, found {len(groups)}")
    if int(summary.get("records_with_errors", 0)):
        errors.append(
            f"{int(summary['records_with_errors'])} Stage2 records have metadata/file errors"
        )
    if groups and int(groups[0].get("unique_views_count", 0)) < min_views:
        errors.append(
            f"only {groups[0].get('unique_views_count', 0)} unique views; require {min_views}"
        )

    scene_name = str(groups[0].get("scene_name", "")) if groups else ""
    pair_group = str(groups[0].get("pair_group", "")) if groups else ""
    group_key = str(groups[0].get("group_key", "")) if groups else ""
    expected_scene = str(item.get("expected_scene_name", "")).strip()
    expected_pair_group = str(item.get("expected_pair_group", "")).strip()
    if expected_scene and scene_name != expected_scene:
        errors.append(f"scene_name {scene_name!r} != expected {expected_scene!r}")
    if expected_pair_group and pair_group != expected_pair_group:
        errors.append(f"pair_group {pair_group!r} != expected {expected_pair_group!r}")

    artifacts, artifact_errors, artifact_warnings = validate_artifacts(
        item.get("artifacts", {})
    )
    errors.extend(artifact_errors)
    warnings.extend(artifact_warnings)
    return {
        "id": item_id,
        "status": "fail" if errors else "pass",
        "input_dir": str(input_dir),
        "pattern": pattern,
        "scene_name": scene_name,
        "pair_group": pair_group,
        "group_key": group_key,
        "records": int(summary.get("records", 0)),
        "unique_views": int(groups[0].get("unique_views_count", 0)) if groups else 0,
        "input_bytes": int(sum(row["bytes"] for row in input_files)),
        "input_files": input_files,
        "artifacts": artifacts,
        "registry_summary": summary,
        "errors": errors,
        "warnings": warnings,
    }


def mark_duplicate_groups(rows: list[dict[str, Any]]) -> None:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["group_key"]:
            by_group.setdefault(row["group_key"], []).append(row)
    for group_key, duplicates in by_group.items():
        if len(duplicates) < 2:
            continue
        ids = ", ".join(row["id"] for row in duplicates)
        for row in duplicates:
            row["errors"].append(f"duplicate group_key shared by items: {ids}: {group_key}")
            row["status"] = "fail"


def result_summary(rows: list[dict[str, Any]], minimum_valid_items: int) -> dict[str, Any]:
    passed = [row for row in rows if row["status"] == "pass"]
    return {
        "items": len(rows),
        "passed_items": len(passed),
        "failed_items": len(rows) - len(passed),
        "minimum_valid_items": int(minimum_valid_items),
        "minimum_met": len(passed) >= int(minimum_valid_items),
        "unique_group_keys": len({row["group_key"] for row in passed if row["group_key"]}),
        "unique_scene_names": len({row["scene_name"] for row in passed if row["scene_name"]}),
        "stage2_records": sum(int(row["records"]) for row in passed),
        "unique_views_per_group": [int(row["unique_views"]) for row in passed],
        "input_bytes": sum(int(row["input_bytes"]) for row in passed),
    }


def csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "status": row["status"],
            "scene_name": row["scene_name"],
            "pair_group": row["pair_group"],
            "records": row["records"],
            "unique_views": row["unique_views"],
            "input_bytes": row["input_bytes"],
            "errors": " | ".join(row["errors"]),
            "warnings": " | ".join(row["warnings"]),
        }
        for row in rows
    ]


def markdown_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        f"# {result['batch_name']} preflight",
        "",
        f"Git SHA: `{result['git_sha'] or 'unknown'}`",
        "",
        (
            f"Passed {summary['passed_items']}/{summary['items']} items; "
            f"{summary['unique_group_keys']} unique view groups from "
            f"{summary['unique_scene_names']} Structured3D scenes."
        ),
        "",
        "| Item | Status | Scene | Views | Records | Errors |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in result["items"]:
        error_text = "<br>".join(row["errors"]).replace("|", "\\|")
        lines.append(
            f"| {row['id']} | {row['status']} | {row['scene_name']} | "
            f"{row['unique_views']} | {row['records']} | {error_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A view group is one retained room/perspective sequence. Multiple view groups may belong to the same Structured3D scene and are not counted as independent scenes.",
            "",
        ]
    )
    return "\n".join(lines)


def run_preflight(
    manifest_path: Path,
    output_dir: Path,
    *,
    git_sha: str = "",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    manifest = load_manifest(manifest_path)
    rows = [
        preflight_item(
            item,
            default_pattern=str(manifest.get("pattern", DEFAULT_PATTERN)),
            default_min_views=int(manifest.get("min_views", 3)),
            check_image_files=bool(manifest.get("check_image_files", True)),
        )
        for item in manifest["items"]
    ]
    mark_duplicate_groups(rows)
    minimum_valid_items = int(manifest.get("minimum_valid_items", len(rows)))
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "research_practice_batch_preflight",
        "batch_name": str(manifest.get("name", manifest_path.stem)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "git_sha": git_sha,
        "python": sys.version,
        "platform": platform.platform(),
        "coordinate_contract": {
            "pixel_order": "xy",
            "support_source": "Stage2 normalized pixel_xy1/pixel_xy2",
            "global_join_key": "(alignment_view_index,x,y)",
            "global_space": "dust3r_aligned_pointmap",
            "guessing_allowed": False,
        },
        "summary": result_summary(rows, minimum_valid_items),
        "items": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "batch_preflight.json"
    csv_path = output_dir / "batch_preflight.csv"
    md_path = output_dir / "batch_preflight.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    flat_rows = csv_rows(rows)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    md_path.write_text(markdown_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser("Preflight a LightRecon3D research-practice batch")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--git_sha", default=os.environ.get("GIT_SHA", ""))
    args = parser.parse_args()
    result = run_preflight(
        Path(args.manifest), Path(args.output_dir), git_sha=str(args.git_sha)
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False), flush=True)
    return 0 if result["summary"]["minimum_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
