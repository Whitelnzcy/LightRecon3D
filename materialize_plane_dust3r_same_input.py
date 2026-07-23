#!/usr/bin/env python3
"""Build an auditable Plane-DUSt3R dataset view over frozen empty renders.

Plane-DUSt3R's official evaluator looks under ``perspective/full``.  The
LightRecon3D batch instead contains frozen ``perspective/empty`` five-view
groups.  This command creates a separate dataset root whose ``full`` position
directories point to those exact empty-render directories.  The source dataset
is never modified, and the manifest records the protocol deviation explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preflight_plane_dust3r_compatibility import batch_items, load_json


def source_record(row: dict[str, str]) -> dict[str, Any]:
    render_dir = Path(row["pair_group"])
    if render_dir.name != "empty":
        raise ValueError(
            f"{row['item_id']}: expected an empty render group, got {render_dir}"
        )
    if not render_dir.is_dir():
        raise FileNotFoundError(f"{row['item_id']}: missing render group: {render_dir}")

    positions = sorted(
        path.parent
        for path in render_dir.glob("*/rgb_rawlight.png")
        if path.is_file()
    )
    if not 1 <= len(positions) <= 5:
        raise ValueError(
            f"{row['item_id']}: expected 1-5 rgb_rawlight images, found "
            f"{len(positions)} in {render_dir}"
        )
    scene_root = render_dir.parents[3]
    if scene_root.name != row["scene_name"]:
        raise ValueError(
            f"{row['item_id']}: path scene {scene_root.name!r} does not match "
            f"ledger scene {row['scene_name']!r}"
        )
    annotation = scene_root / "annotation_3d.json"
    if not annotation.is_file():
        raise FileNotFoundError(
            f"{row['item_id']}: missing Structured3D annotation: {annotation}"
        )
    return {
        **row,
        "source_scene_root": str(scene_root),
        "source_render_dir": str(render_dir),
        "source_render_mode": "empty",
        "room_id": render_dir.parents[1].name,
        "positions": [path.name for path in positions],
        "source_position_dirs": [str(path) for path in positions],
        "source_annotation": str(annotation),
    }


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    if mode == "symlink":
        os.symlink(source.resolve(), destination, target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "item_id",
        "scene_name",
        "room_id",
        "source_render_mode",
        "declared_render_mode",
        "image_count",
        "positions",
        "source_render_dir",
        "compatibility_render_dir",
        "link_mode",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            value = dict(row)
            value["positions"] = ";".join(row["positions"])
            writer.writerow(value)


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Plane-DUSt3R same-input dataset",
        "",
        f"Materialized {result['summary']['scenes']} frozen scene groups with "
        f"{result['summary']['images']} total images.",
        "",
        f"Dataset root: `{result['dataset_root']}`",
        "",
        "This is not the Plane-DUSt3R native `perspective/full` protocol. Each "
        "compatibility `full` position resolves to the exact frozen "
        "LightRecon3D `perspective/empty` position. It is intended only for a "
        "same-input external-model comparison.",
        "",
        "| Scene | Room | Source | Declared | Images | Positions |",
        "|---|---|---|---|---:|---|",
    ]
    for row in result["items"]:
        lines.append(
            f"| {row['scene_name']} | {row['room_id']} | empty | full | "
            f"{row['image_count']} | {', '.join(row['positions'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def materialize(
    batch_json: Path,
    output_dir: Path,
    *,
    link_mode: str = "symlink",
    maximum_scenes: int = 0,
    git_sha: str = "",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    if link_mode not in {"symlink", "copy"}:
        raise ValueError(f"unsupported link mode: {link_mode}")

    records = [source_record(row) for row in batch_items(load_json(batch_json))]
    if maximum_scenes > 0:
        records = records[:maximum_scenes]
    if not records:
        raise ValueError("no scenes selected")

    dataset_root = output_dir / "dataset"
    dataset_root.mkdir(parents=True)
    output_rows: list[dict[str, Any]] = []
    for record in records:
        scene_out = dataset_root / record["scene_name"]
        scene_out.mkdir()
        annotation_out = scene_out / "annotation_3d.json"
        link_or_copy(Path(record["source_annotation"]), annotation_out, link_mode)

        full_out = (
            scene_out
            / "2D_rendering"
            / record["room_id"]
            / "perspective"
            / "full"
        )
        full_out.mkdir(parents=True)
        compatibility_positions: list[str] = []
        for source_position in record["source_position_dirs"]:
            source = Path(source_position)
            destination = full_out / source.name
            link_or_copy(source, destination, link_mode)
            compatibility_positions.append(str(destination))

        output_rows.append(
            {
                **record,
                "declared_render_mode": "full",
                "image_count": len(record["positions"]),
                "compatibility_render_dir": str(full_out),
                "compatibility_position_dirs": compatibility_positions,
                "link_mode": link_mode,
            }
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "plane_dust3r_same_input_dataset",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "source_batch_json": str(batch_json),
        "dataset_root": str(dataset_root),
        "protocol": {
            "native_plane_dust3r_render_mode": "full",
            "actual_source_render_mode": "empty",
            "comparison_class": "same_input_external_model_reproduction",
            "native_protocol_claim_allowed": False,
        },
        "summary": {
            "scenes": len(output_rows),
            "images": sum(row["image_count"] for row in output_rows),
            "link_mode": link_mode,
        },
        "items": output_rows,
    }
    (output_dir / "same_input_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "same_input_scene_inventory.csv", output_rows)
    (output_dir / "same_input_manifest.md").write_text(
        markdown(result), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_execution_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--link_mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--maximum_scenes", type=int, default=0)
    parser.add_argument("--git_sha", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = materialize(
        Path(args.batch_execution_json),
        Path(args.output_dir),
        link_mode=args.link_mode,
        maximum_scenes=args.maximum_scenes,
        git_sha=args.git_sha,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
