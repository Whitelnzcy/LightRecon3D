"""Prepare a deterministic, scene-unique final research-practice manifest.

This is an inventory/planning step only.  It never runs Stage1, Stage2,
DUSt3R, or any benchmark.  Existing Stage2 groups and archived global caches
are reused only through exact pair-group metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_PATTERN = "*_learned_region_merge_full_pointcloud_editable_planes_data.npz"


def group_dataset_samples(
    samples: Iterable[dict[str, Any]], *, min_pairs: int
) -> list[dict[str, Any]]:
    if min_pairs < 1:
        raise ValueError("min_pairs must be positive")
    grouped: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        scene_name = str(sample.get("scene_name", "")).strip()
        pair_group = str(sample.get("pair_group", "")).strip()
        if not scene_name or not pair_group:
            raise ValueError(f"dataset sample {index} lacks scene_name or pair_group")
        row = grouped.setdefault(
            pair_group,
            {"scene_name": scene_name, "pair_group": pair_group, "indices": []},
        )
        if row["scene_name"] != scene_name:
            raise ValueError(
                f"pair group {pair_group!r} crosses scenes: "
                f"{row['scene_name']!r} and {scene_name!r}"
            )
        row["indices"].append(int(index))
    rows = []
    for pair_group, row in grouped.items():
        if len(row["indices"]) < int(min_pairs):
            continue
        rows.append(
            {
                "scene_name": row["scene_name"],
                "pair_group": pair_group,
                "pair_count": int(len(row["indices"])),
                "selected_indices": row["indices"][: int(min_pairs)],
            }
        )
    return sorted(rows, key=lambda row: (row["scene_name"], row["pair_group"]))


def discover_dataset_groups(
    root_dir: Path,
    *,
    split: str,
    train_ratio: float,
    pair_strategy: str,
    min_pairs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Imported lazily so unit tests for deterministic selection do not require
    # torch/OpenCV or a local Structured3D installation.
    from dataloaders.s3d_dataset import Structured3DDataset

    dataset = Structured3DDataset(
        root_dir=str(root_dir),
        split=split,
        train_ratio=train_ratio,
        image_size=(512, 512),
        input_mode="pair",
        pair_strategy=pair_strategy,
    )
    groups = group_dataset_samples(dataset.samples, min_pairs=min_pairs)
    inventory = {
        "root_dir": str(root_dir),
        "split": split,
        "train_ratio": float(train_ratio),
        "pair_strategy": pair_strategy,
        "dataset_pairs": int(len(dataset.samples)),
        "eligible_groups": int(len(groups)),
        "eligible_scenes": int(len({row['scene_name'] for row in groups})),
        "all_scene_count": int(len(dataset.all_scenes)),
        "split_scene_count": int(len(dataset.scenes)),
        "split_scenes": list(dataset.scenes),
    }
    return groups, inventory


def load_existing_stage2_groups(
    existing_root: Path,
    *,
    selected_groups_tsv: Path | None = None,
    pattern: str = DEFAULT_PATTERN,
    minimum_records: int = 1,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    selected_groups_tsv = selected_groups_tsv or existing_root / "selected_groups.tsv"
    warnings: list[str] = []
    if not selected_groups_tsv.is_file():
        return {}, [f"existing selection TSV is missing: {selected_groups_tsv}"]
    result: dict[str, dict[str, Any]] = {}
    for line_number, text in enumerate(
        selected_groups_tsv.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not text.strip():
            continue
        fields = text.split("\t")
        if len(fields) != 4:
            raise ValueError(
                f"{selected_groups_tsv}:{line_number}: expected four TSV fields"
            )
        ordinal_text, pair_count_text, indices_text, pair_group = fields
        try:
            ordinal = int(ordinal_text)
            pair_count = int(pair_count_text)
            indices = [int(value) for value in indices_text.split(",") if value]
        except ValueError as error:
            raise ValueError(
                f"{selected_groups_tsv}:{line_number}: invalid numeric field"
            ) from error
        group_name = f"group_{ordinal:03d}_pairs_{pair_count}"
        input_dir = existing_root / group_name / "stage2_merge"
        files = sorted(input_dir.glob(pattern)) if input_dir.is_dir() else []
        ready = len(files) >= int(minimum_records)
        row = {
            "pair_group": pair_group,
            "ordinal": ordinal,
            "pair_count": pair_count,
            "selected_indices": indices,
            "group_name": group_name,
            "input_dir": str(input_dir),
            "stage2_records": int(len(files)),
            "ready": bool(ready),
        }
        if pair_group in result:
            raise ValueError(f"duplicate pair group in {selected_groups_tsv}: {pair_group}")
        result[pair_group] = row
        if not ready:
            warnings.append(
                f"existing group is incomplete ({len(files)} records): {input_dir}"
            )
    return result, warnings


def load_reusable_caches(
    batch_execution_json: Path | None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if batch_execution_json is None:
        return {}, []
    if not batch_execution_json.is_file():
        return {}, [f"cache-source batch JSON is missing: {batch_execution_json}"]
    payload = json.loads(batch_execution_json.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for item in payload.get("items", []):
        if item.get("status") != "pass":
            continue
        pair_group = str(item.get("pair_group", "")).strip()
        record = item.get("artifacts", {}).get("global_cloud_cache", {})
        path = str(record.get("path", "")).strip()
        sha256 = str(record.get("sha256", "")).strip().lower()
        if not pair_group or not path or not sha256:
            warnings.append(
                f"passed batch item {item.get('id')} lacks pair-group/cache checksum"
            )
            continue
        if pair_group in result and result[pair_group] != {
            "global_cloud_cache": path,
            "global_cloud_sha256": sha256,
        }:
            raise ValueError(f"conflicting reusable caches for pair group: {pair_group}")
        result[pair_group] = {
            "global_cloud_cache": path,
            "global_cloud_sha256": sha256,
        }
    return result, warnings


def select_unique_scene_groups(
    eligible_groups: list[dict[str, Any]],
    existing_groups: dict[str, dict[str, Any]],
    reusable_caches: dict[str, dict[str, Any]],
    *,
    target_scenes: int,
    expansion_root: Path,
) -> list[dict[str, Any]]:
    if target_scenes < 1:
        raise ValueError("target_scenes must be positive")
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in eligible_groups:
        by_scene.setdefault(str(row["scene_name"]), []).append(row)
    if len(by_scene) < target_scenes:
        raise ValueError(
            f"only {len(by_scene)} eligible independent scenes; require {target_scenes}"
        )

    selected: list[dict[str, Any]] = []
    for scene_name in sorted(by_scene)[:target_scenes]:
        candidates = sorted(
            by_scene[scene_name],
            key=lambda row: (
                0
                if existing_groups.get(row["pair_group"], {}).get("ready")
                else 1,
                row["pair_group"],
            ),
        )
        source = candidates[0]
        item_id = f"final_{len(selected):03d}_{scene_name}"
        existing = existing_groups.get(source["pair_group"])
        if existing and existing["ready"]:
            input_dir = Path(existing["input_dir"])
            materialization = "reuse_existing_stage2"
        else:
            input_dir = expansion_root / item_id / "stage2_merge"
            materialization = "needs_stage1_stage2"
        row = {
            "id": item_id,
            "scene_name": scene_name,
            "pair_group": source["pair_group"],
            "pair_count": int(source["pair_count"]),
            "selected_indices": list(source["selected_indices"]),
            "input_dir": str(input_dir),
            "materialization": materialization,
            "existing_group_name": existing["group_name"] if existing else "",
            "existing_stage2_records": (
                int(existing["stage2_records"]) if existing else 0
            ),
            "reusable_cache": reusable_caches.get(source["pair_group"]),
        }
        selected.append(row)
    return selected


def execution_manifest(
    selected: list[dict[str, Any]], *, pattern: str
) -> dict[str, Any]:
    items = []
    for row in selected:
        item = {
            "id": row["id"],
            "input_dir": row["input_dir"],
            "expected_scene_name": row["scene_name"],
            "expected_pair_group": row["pair_group"],
        }
        if row["reusable_cache"]:
            item.update(row["reusable_cache"])
        items.append(item)
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "research_practice_final_unique_scene_batch",
        "minimum_valid_items": len(items),
        "min_views": 5,
        "check_image_files": True,
        "pattern": pattern,
        "items": items,
    }


def markdown_plan(plan: dict[str, Any]) -> str:
    summary = plan["summary"]
    lines = [
        "# Research-practice final unique-scene selection",
        "",
        (
            f"Selected {summary['selected_scenes']} independent scenes; "
            f"{summary['reused_stage2_groups']} reuse existing Stage2 outputs and "
            f"{summary['groups_needing_stage1_stage2']} require materialization."
        ),
        "",
        "Selection is deterministic and metric-blind: scene names are sorted, one pair group is retained per scene, and a ready existing group is preferred within that scene.",
        "",
        "| Item | Scene | State | Pairs | Cache reuse | Pair group |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in plan["items"]:
        lines.append(
            f"| {row['id']} | {row['scene_name']} | {row['materialization']} | "
            f"{len(row['selected_indices'])} | {bool(row['reusable_cache'])} | "
            f"{row['pair_group']} |"
        )
    lines.extend(
        [
            "",
            "This command performs inventory and manifest generation only. It does not run a neural model, global alignment, RANSAC, or evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_plan(
    output_dir: Path,
    *,
    selected: list[dict[str, Any]],
    dataset_inventory: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    summary = {
        "selected_scenes": int(len(selected)),
        "unique_scene_names": int(len({row['scene_name'] for row in selected})),
        "unique_pair_groups": int(len({row['pair_group'] for row in selected})),
        "reused_stage2_groups": int(
            sum(row["materialization"] == "reuse_existing_stage2" for row in selected)
        ),
        "groups_needing_stage1_stage2": int(
            sum(row["materialization"] == "needs_stage1_stage2" for row in selected)
        ),
        "reused_global_caches": int(sum(bool(row["reusable_cache"]) for row in selected)),
    }
    if summary["selected_scenes"] != summary["unique_scene_names"]:
        raise ValueError("selection contains duplicate scene names")
    if summary["selected_scenes"] != summary["unique_pair_groups"]:
        raise ValueError("selection contains duplicate pair groups")
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": "research_practice_final_scene_selection",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "dataset_inventory": dataset_inventory,
        "summary": summary,
        "warnings": warnings,
        "items": selected,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "selection_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "final_unique_scenes_execute.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "selection_plan.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fields = [
            "id",
            "scene_name",
            "pair_group",
            "pair_count",
            "selected_indices",
            "input_dir",
            "materialization",
            "existing_group_name",
            "existing_stage2_records",
            "reusable_cache",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    (output_dir / "selection_plan.md").write_text(
        markdown_plan(plan), encoding="utf-8"
    )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(
        "Prepare a deterministic scene-unique research-practice manifest"
    )
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--existing_root", required=True)
    parser.add_argument("--expansion_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reuse_batch_execution_json")
    parser.add_argument("--selected_groups_tsv")
    parser.add_argument("--target_scenes", type=int, default=8)
    parser.add_argument("--min_pairs", type=int, default=10)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--pair_strategy", choices=("adjacent", "all"), default="all")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        parser.error(f"refusing to overwrite existing output: {output_dir}")
    groups, inventory = discover_dataset_groups(
        Path(args.root_dir),
        split=args.split,
        train_ratio=args.train_ratio,
        pair_strategy=args.pair_strategy,
        min_pairs=args.min_pairs,
    )
    existing, existing_warnings = load_existing_stage2_groups(
        Path(args.existing_root),
        selected_groups_tsv=(
            Path(args.selected_groups_tsv) if args.selected_groups_tsv else None
        ),
        pattern=args.pattern,
        minimum_records=args.min_pairs,
    )
    caches, cache_warnings = load_reusable_caches(
        Path(args.reuse_batch_execution_json)
        if args.reuse_batch_execution_json
        else None
    )
    selected = select_unique_scene_groups(
        groups,
        existing,
        caches,
        target_scenes=args.target_scenes,
        expansion_root=Path(args.expansion_root),
    )
    manifest = execution_manifest(selected, pattern=args.pattern)
    configuration = {
        "root_dir": args.root_dir,
        "existing_root": args.existing_root,
        "expansion_root": args.expansion_root,
        "target_scenes": args.target_scenes,
        "min_pairs": args.min_pairs,
        "split": args.split,
        "train_ratio": args.train_ratio,
        "pair_strategy": args.pair_strategy,
        "pattern": args.pattern,
        "reuse_batch_execution_json": args.reuse_batch_execution_json or "",
    }
    plan = write_plan(
        output_dir,
        selected=selected,
        dataset_inventory=inventory,
        manifest=manifest,
        warnings=existing_warnings + cache_warnings,
        configuration=configuration,
    )
    print(json.dumps(plan["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
