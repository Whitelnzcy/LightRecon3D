#!/usr/bin/env python3
"""Audit whether Plane-DUSt3R can be reproduced beside a frozen batch.

This command is deliberately read-only.  It does not clone repositories,
download checkpoints, run inference, or reinterpret Plane-DUSt3R outputs as
LightRecon3D plane labels.  Its job is to make the input/protocol mismatch
explicit before an expensive external-baseline run is started.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REQUIRED_REPO_FILES = (
    "evaluate_planedust3r.py",
    "metric.py",
    "plane_merge_planedust3r.py",
)
REQUIRED_REPO_DIRS = ("MASt3R", "NonCuboidRoom")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def batch_items(batch: dict[str, Any]) -> list[dict[str, str]]:
    raw_items = batch.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("batch JSON must contain a non-empty items list")

    rows: list[dict[str, str]] = []
    seen_scenes: set[str] = set()
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"items[{index}] must be an object")
        if str(raw.get("status", "pass")) != "pass":
            continue
        item_id = str(raw.get("id") or raw.get("item_id") or "").strip()
        scene_name = str(
            raw.get("scene_name") or raw.get("expected_scene_name") or ""
        ).strip()
        pair_group = str(
            raw.get("pair_group") or raw.get("expected_pair_group") or ""
        ).strip()
        if not item_id or not scene_name or not pair_group:
            raise ValueError(
                f"passed items[{index}] needs id, scene_name and pair_group"
            )
        if scene_name in seen_scenes:
            raise ValueError(f"duplicate independent scene ID: {scene_name}")
        seen_scenes.add(scene_name)
        rows.append(
            {"item_id": item_id, "scene_name": scene_name, "pair_group": pair_group}
        )
    if not rows:
        raise ValueError("batch JSON contains no passed items")
    return rows


def image_positions(render_dir: Path) -> list[str]:
    if not render_dir.is_dir():
        return []
    return sorted(
        path.parent.name
        for path in render_dir.glob("*/rgb_rawlight.png")
        if path.is_file()
    )


def path_record(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
    }


def inspect_repository(repo: Path) -> dict[str, Any]:
    missing_files = [name for name in REQUIRED_REPO_FILES if not (repo / name).is_file()]
    missing_dirs = [name for name in REQUIRED_REPO_DIRS if not (repo / name).is_dir()]
    return {
        "path": str(repo),
        "exists": repo.is_dir(),
        "required_files": list(REQUIRED_REPO_FILES),
        "required_directories": list(REQUIRED_REPO_DIRS),
        "missing_files": missing_files,
        "missing_directories": missing_dirs,
        "ready": repo.is_dir() and not missing_files and not missing_dirs,
    }


def inspect_scene(row: dict[str, str]) -> dict[str, Any]:
    pair_group = Path(row["pair_group"])
    input_mode = pair_group.name
    perspective_dir = pair_group.parent
    official_full_dir = perspective_dir / "full"
    source_positions = image_positions(pair_group)
    full_positions = image_positions(official_full_dir)
    shared_positions = sorted(set(source_positions) & set(full_positions))
    return {
        **row,
        "lightrecon_render_mode": input_mode,
        "lightrecon_render_dir": str(pair_group),
        "lightrecon_image_count": len(source_positions),
        "lightrecon_positions": source_positions,
        "official_render_mode": "full",
        "official_render_dir": str(official_full_dir),
        "official_image_count": len(full_positions),
        "official_positions": full_positions,
        "shared_position_count": len(shared_positions),
        "shared_positions": shared_positions,
        "native_scene_ready": 1 <= len(full_positions) <= 5,
        "identical_render_input": pair_group.resolve() == official_full_dir.resolve(),
    }


def markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Plane-DUSt3R compatibility preflight",
        "",
        f"Frozen LightRecon3D scenes: {summary['scenes']}.",
        f"Scenes with 1-5 official `perspective/full` images: "
        f"{summary['native_scene_ready']}/{summary['scenes']}.",
        "",
        f"Native smoke ready: `{str(summary['native_smoke_ready']).lower()}`.",
        f"Native full-batch ready: `{str(summary['native_full_batch_ready']).lower()}`.",
        f"Same-input common-metric ready: "
        f"`{str(summary['common_partition_ready']).lower()}`.",
        "",
        "## Protocol decision",
        "",
        "Plane-DUSt3R must first be reported with its official room-layout "
        "evaluation in a separate table. The frozen LightRecon3D batch uses "
        "`perspective/empty`, while the official evaluator uses "
        "`perspective/full`. Its native output is a structural room layout, "
        "not the unrestricted global point-cloud plane partition used for "
        "VOI/RI/SC in LightRecon3D.",
        "",
        "A shared VOI/RI/SC row is allowed only after an explicit output adapter "
        "produces labels on the same ordered global point cache without "
        "inventing omitted furniture or support labels.",
        "",
        "## Missing prerequisites",
        "",
    ]
    reasons = summary["blocking_reasons"]
    lines.extend([f"- {reason}" for reason in reasons] or ["- None for native protocol."])
    lines.extend(
        [
            "",
            "## Scene inventory",
            "",
            "| Scene | LightRecon mode/images | Official full images | Shared positions | Native ready |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["scenes"]:
        lines.append(
            f"| {row['scene_name']} | {row['lightrecon_render_mode']}/"
            f"{row['lightrecon_image_count']} | {row['official_image_count']} | "
            f"{row['shared_position_count']} | {row['native_scene_ready']} |"
        )
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            f"- Plane-DUSt3R: `{result['plane_checkpoint']['path']}` "
            f"(exists={result['plane_checkpoint']['exists']}, "
            f"bytes={result['plane_checkpoint']['bytes']})",
            f"- NonCuboidRoom: `{result['noncuboid_checkpoint']['path']}` "
            f"(exists={result['noncuboid_checkpoint']['exists']}, "
            f"bytes={result['noncuboid_checkpoint']['bytes']})",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fields = [
        "item_id",
        "scene_name",
        "lightrecon_render_mode",
        "lightrecon_render_dir",
        "lightrecon_image_count",
        "official_render_dir",
        "official_image_count",
        "shared_position_count",
        "native_scene_ready",
        "identical_render_input",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_preflight(
    batch_json: Path,
    repo: Path,
    plane_checkpoint: Path,
    noncuboid_checkpoint: Path,
    output_dir: Path,
    *,
    git_sha: str = "",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)

    rows = [inspect_scene(row) for row in batch_items(load_json(batch_json))]
    repository = inspect_repository(repo)
    plane_record = path_record(plane_checkpoint)
    noncuboid_record = path_record(noncuboid_checkpoint)
    prerequisites_ready = (
        repository["ready"]
        and plane_record["exists"]
        and noncuboid_record["exists"]
    )
    ready_count = sum(bool(row["native_scene_ready"]) for row in rows)

    reasons: list[str] = []
    if not repository["ready"]:
        reasons.append("official repository or required submodules/files are missing")
    if not plane_record["exists"]:
        reasons.append("Plane-DUSt3R checkpoint is missing")
    if not noncuboid_record["exists"]:
        reasons.append("NonCuboidRoom Structured3D checkpoint is missing")
    if ready_count == 0:
        reasons.append("no frozen scene has an official `perspective/full` image group")
    elif ready_count != len(rows):
        reasons.append("only part of the frozen scene list has official full-render inputs")
    reasons.append(
        "common VOI/RI/SC comparison needs a reviewed Plane-DUSt3R output adapter"
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "plane_dust3r_compatibility_preflight",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "source_batch_json": str(batch_json),
        "official_repository": repository,
        "plane_checkpoint": plane_record,
        "noncuboid_checkpoint": noncuboid_record,
        "summary": {
            "scenes": len(rows),
            "native_scene_ready": ready_count,
            "repository_ready": repository["ready"],
            "checkpoints_ready": plane_record["exists"] and noncuboid_record["exists"],
            "native_smoke_ready": prerequisites_ready and ready_count >= 1,
            "native_full_batch_ready": prerequisites_ready and ready_count == len(rows),
            "identical_input_scenes": sum(
                bool(row["identical_render_input"]) for row in rows
            ),
            "common_partition_ready": False,
            "blocking_reasons": reasons,
        },
        "scenes": rows,
    }
    (output_dir / "plane_dust3r_compatibility.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "plane_dust3r_scene_inventory.csv", rows)
    (output_dir / "plane_dust3r_compatibility.md").write_text(
        markdown(result), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_execution_json", required=True)
    parser.add_argument("--official_repo", required=True)
    parser.add_argument("--plane_checkpoint", required=True)
    parser.add_argument("--noncuboid_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--git_sha", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_preflight(
        Path(args.batch_execution_json),
        Path(args.official_repo),
        Path(args.plane_checkpoint),
        Path(args.noncuboid_checkpoint),
        Path(args.output_dir),
        git_sha=args.git_sha,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
