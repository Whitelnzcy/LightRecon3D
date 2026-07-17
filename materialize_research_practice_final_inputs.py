"""Materialize only missing Stage1/Stage2 inputs from a frozen scene plan."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from execute_research_practice_batch import hardware_record, run_logged_stage
from research_practice_batch import DEFAULT_PATTERN, file_record, file_sha256, preflight_item


SCHEMA_VERSION = 1
Runner = Callable[[str, list[str], Path, Path], dict[str, Any]]


class MaterializationFailure(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def load_selection_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported selection-plan schema version")
    if payload.get("kind") != "research_practice_final_scene_selection":
        raise ValueError(f"unexpected selection-plan kind: {payload.get('kind')!r}")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("selection plan must contain non-empty items")
    ids: set[str] = set()
    scenes: set[str] = set()
    groups: set[str] = set()
    for index, item in enumerate(items):
        item_id = str(item.get("id", "")).strip()
        scene = str(item.get("scene_name", "")).strip()
        group = str(item.get("pair_group", "")).strip()
        input_dir = str(item.get("input_dir", "")).strip()
        state = str(item.get("materialization", ""))
        indices = item.get("selected_indices")
        if not item_id or not scene or not group or not input_dir:
            raise ValueError(f"selection item {index} lacks required metadata")
        if state not in {"reuse_existing_stage2", "needs_stage1_stage2"}:
            raise ValueError(f"selection item {item_id} has invalid state {state!r}")
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"selection item {item_id} has no selected indices")
        if item_id in ids or scene in scenes or group in groups:
            raise ValueError(f"selection plan duplicates id, scene, or group at {item_id}")
        ids.add(item_id)
        scenes.add(scene)
        groups.add(group)
    return payload


def stage1_command(
    python_bin: str,
    project_dir: Path,
    item: dict[str, Any],
    *,
    root_dir: Path,
    weights_path: Path,
    stage1_checkpoint: Path,
    feature_cache_path: Path,
    stage1_dir: Path,
) -> list[str]:
    return [
        python_bin,
        str(project_dir / "export_stage1_pred_support_teacher_npz.py"),
        "--root_dir",
        str(root_dir),
        "--weights_path",
        str(weights_path),
        "--stage1_checkpoint",
        str(stage1_checkpoint),
        "--feature_cache_path",
        str(feature_cache_path),
        "--output_dir",
        str(stage1_dir),
        "--split",
        "val",
        "--indices",
        ",".join(str(int(value)) for value in item["selected_indices"]),
        "--count",
        "0",
        "--train_ratio",
        "0.9",
        "--pair_strategy",
        "all",
        "--max_planes",
        "8",
        "--max_points",
        "4000",
        "--export_second_view_support",
        "--num_workers",
        "0",
    ]


def stage2_command(
    python_bin: str,
    project_dir: Path,
    *,
    stage1_dir: Path,
    stage2_checkpoint: Path,
    stage2_dir: Path,
) -> list[str]:
    return [
        python_bin,
        str(project_dir / "export_stage2_learned_region_merge_editables.py"),
        "--input_dir",
        str(stage1_dir),
        "--checkpoint",
        str(stage2_checkpoint),
        "--output_dir",
        str(stage2_dir),
        "--threshold",
        "0.5",
        "--use_safety_gate",
    ]


def registry_command(
    python_bin: str,
    project_dir: Path,
    *,
    stage2_dir: Path,
    output_json: Path,
) -> list[str]:
    return [
        python_bin,
        str(project_dir / "validate_stage3_view_registry.py"),
        "--input_dir",
        str(stage2_dir),
        "--output_json",
        str(output_json),
        "--pattern",
        DEFAULT_PATTERN,
        "--check_files",
    ]


def run_required(
    row: dict[str, Any],
    stage: str,
    command: list[str],
    project_dir: Path,
    log_path: Path,
    runner: Runner,
) -> None:
    result = runner(stage, command, project_dir, log_path)
    row["stages"].append(result)
    if int(result["return_code"]) != 0:
        raise MaterializationFailure(
            stage, f"stage {stage} exited with {result['return_code']}"
        )


def validate_final_input(item: dict[str, Any]) -> dict[str, Any]:
    return preflight_item(
        {
            "id": item["id"],
            "input_dir": item["input_dir"],
            "expected_scene_name": item["scene_name"],
            "expected_pair_group": item["pair_group"],
        },
        default_pattern=DEFAULT_PATTERN,
        default_min_views=5,
        check_image_files=True,
    )


def materialize_item(
    item: dict[str, Any],
    *,
    project_dir: Path,
    python_bin: str,
    root_dir: Path,
    weights_path: Path,
    stage1_checkpoint: Path,
    feature_cache_path: Path,
    stage2_checkpoint: Path,
    runner: Runner = run_logged_stage,
) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "id": str(item["id"]),
        "scene_name": str(item["scene_name"]),
        "pair_group": str(item["pair_group"]),
        "materialization": str(item["materialization"]),
        "input_dir": str(item["input_dir"]),
        "status": "running",
        "failure_stage": "",
        "error": "",
        "stages": [],
        "preflight": {},
        "artifacts": {},
    }
    try:
        stage2_dir = Path(str(item["input_dir"]))
        if item["materialization"] == "needs_stage1_stage2":
            group_root = stage2_dir.parent
            if group_root.exists():
                raise MaterializationFailure(
                    "refuse_overwrite",
                    f"planned materialization root already exists: {group_root}",
                )
            stage1_dir = group_root / "stage1_teacher"
            logs_dir = group_root / "logs"
            registry_json = group_root / "registry" / "view_registry_summary.json"
            stage1_dir.mkdir(parents=True, exist_ok=False)
            logs_dir.mkdir()
            run_required(
                row,
                "stage1_support",
                stage1_command(
                    python_bin,
                    project_dir,
                    item,
                    root_dir=root_dir,
                    weights_path=weights_path,
                    stage1_checkpoint=stage1_checkpoint,
                    feature_cache_path=feature_cache_path,
                    stage1_dir=stage1_dir,
                ),
                project_dir,
                logs_dir / "stage1_support.log",
                runner,
            )
            run_required(
                row,
                "stage2_region_merge",
                stage2_command(
                    python_bin,
                    project_dir,
                    stage1_dir=stage1_dir,
                    stage2_checkpoint=stage2_checkpoint,
                    stage2_dir=stage2_dir,
                ),
                project_dir,
                logs_dir / "stage2_region_merge.log",
                runner,
            )
            run_required(
                row,
                "view_registry",
                registry_command(
                    python_bin,
                    project_dir,
                    stage2_dir=stage2_dir,
                    output_json=registry_json,
                ),
                project_dir,
                logs_dir / "view_registry.log",
                runner,
            )
            row["artifacts"]["stage1_manifest"] = file_record(
                stage1_dir / "stage1_pred_support_teacher_manifest.json"
            )
            row["artifacts"]["stage2_manifest"] = file_record(
                stage2_dir / "learned_region_merge_manifest.json"
            )
            row["artifacts"]["view_registry"] = file_record(registry_json)

        validation = validate_final_input(item)
        row["preflight"] = validation
        if validation["status"] != "pass":
            raise MaterializationFailure(
                "input_preflight", " | ".join(validation["errors"])
            )
        row["status"] = "pass"
    except MaterializationFailure as error:
        row["status"] = "fail"
        row["failure_stage"] = error.stage
        row["error"] = str(error)
    except Exception as error:
        row["status"] = "fail"
        row["failure_stage"] = row["stages"][-1]["stage"] if row["stages"] else "setup"
        row["error"] = f"{type(error).__name__}: {error}"
    row["runtime_seconds"] = float(time.perf_counter() - started)
    return row


def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [item for item in items if item["status"] == "pass"]
    return {
        "items": int(len(items)),
        "passed_items": int(len(passed)),
        "failed_items": int(len(items) - len(passed)),
        "independent_scenes": int(len({item["scene_name"] for item in passed})),
        "reused_stage2_groups": int(
            sum(
                item["status"] == "pass"
                and item["materialization"] == "reuse_existing_stage2"
                for item in items
            )
        ),
        "materialized_stage2_groups": int(
            sum(
                item["status"] == "pass"
                and item["materialization"] == "needs_stage1_stage2"
                for item in items
            )
        ),
        "runtime_seconds": float(sum(item["runtime_seconds"] for item in items)),
    }


def markdown_report(result: dict[str, Any]) -> str:
    total = result["summary"]
    lines = [
        "# Research-practice final input materialization",
        "",
        (
            f"Passed {total['passed_items']}/{total['items']} groups from "
            f"{total['independent_scenes']} independent scene IDs."
        ),
        "",
        "| Item | Scene | Mode | Status | Failure stage | Runtime (s) |",
        "|---|---|---|---:|---|---:|",
    ]
    for item in result["items"]:
        lines.append(
            f"| {item['id']} | {item['scene_name']} | {item['materialization']} | "
            f"{item['status']} | {item['failure_stage']} | {item['runtime_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Existing Stage2 groups are read-only. Missing groups run the frozen Stage1/Stage2 checkpoints only; DUSt3R global alignment and final baselines are not executed by this step.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    result["summary"] = summary(result["items"])
    (output_dir / "materialization.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = [
        {
            "id": item["id"],
            "scene_name": item["scene_name"],
            "pair_group": item["pair_group"],
            "materialization": item["materialization"],
            "input_dir": item["input_dir"],
            "status": item["status"],
            "failure_stage": item["failure_stage"],
            "error": item["error"],
            "runtime_seconds": item["runtime_seconds"],
        }
        for item in result["items"]
    ]
    csv_path = output_dir / "materialization.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    (output_dir / "materialization.md").write_text(
        markdown_report(result), encoding="utf-8"
    )


def materialize_plan(
    selection_plan_path: Path,
    output_dir: Path,
    *,
    project_dir: Path,
    python_bin: str,
    root_dir: Path,
    weights_path: Path,
    stage1_checkpoint: Path,
    feature_cache_path: Path,
    stage2_checkpoint: Path,
    git_sha: str,
    resume: bool = False,
    runner: Runner = run_logged_stage,
) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    plan = load_selection_plan(selection_plan_path)
    required_files = {
        "weights": weights_path,
        "stage1_checkpoint": stage1_checkpoint,
        "feature_cache": feature_cache_path,
        "stage2_checkpoint": stage2_checkpoint,
    }
    for name, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    selection_sha256 = file_sha256(selection_plan_path)
    frozen_inputs = {
        name: file_record(path) for name, path in required_files.items()
    }
    ledger_path = output_dir / "materialization.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
        if not ledger_path.is_file():
            raise FileNotFoundError(f"resume ledger is missing: {ledger_path}")
        result = json.loads(ledger_path.read_text(encoding="utf-8"))
        if result.get("kind") != "research_practice_final_input_materialization":
            raise ValueError("resume ledger has the wrong kind")
        if result.get("selection_plan_sha256") != selection_sha256:
            raise ValueError("resume selection plan checksum mismatch")
        for name, record in frozen_inputs.items():
            previous = result.get("frozen_inputs", {}).get(name, {})
            if previous.get("sha256") != record.get("sha256"):
                raise ValueError(f"resume frozen-input checksum mismatch: {name}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "research_practice_final_input_materialization",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_plan": str(selection_plan_path),
            "selection_plan_sha256": selection_sha256,
            "git_sha": git_sha,
            "python_bin": python_bin,
            "project_dir": str(project_dir),
            "root_dir": str(root_dir),
            "hardware": hardware_record(),
            "frozen_inputs": frozen_inputs,
            "items": [],
            "summary": {},
        }
        write_outputs(result, output_dir)
    planned_ids = {str(item["id"]) for item in plan["items"]}
    recorded_ids = [str(item.get("id", "")) for item in result["items"]]
    if len(recorded_ids) != len(set(recorded_ids)):
        raise ValueError("resume ledger contains duplicate item IDs")
    if not set(recorded_ids).issubset(planned_ids):
        raise ValueError("resume ledger contains items absent from the selection plan")
    completed = set(recorded_ids)
    for item in plan["items"]:
        if str(item["id"]) in completed:
            print(f"[resume] materialization already recorded: {item['id']}", flush=True)
            continue
        row = materialize_item(
            item,
            project_dir=project_dir,
            python_bin=python_bin,
            root_dir=root_dir,
            weights_path=weights_path,
            stage1_checkpoint=stage1_checkpoint,
            feature_cache_path=feature_cache_path,
            stage2_checkpoint=stage2_checkpoint,
            runner=runner,
        )
        result["items"].append(row)
        write_outputs(result, output_dir)
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "status": row["status"],
                    "failure_stage": row["failure_stage"],
                    "runtime_seconds": row["runtime_seconds"],
                },
                indent=2,
            ),
            flush=True,
        )
    write_outputs(result, output_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        "Materialize missing Stage1/Stage2 groups from a frozen final plan"
    )
    parser.add_argument("--selection_plan", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--feature_cache_path", required=True)
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--git_sha", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = materialize_plan(
        Path(args.selection_plan),
        Path(args.output_dir),
        project_dir=Path(args.project_dir),
        python_bin=str(args.python_bin),
        root_dir=Path(args.root_dir),
        weights_path=Path(args.weights_path),
        stage1_checkpoint=Path(args.stage1_checkpoint),
        feature_cache_path=Path(args.feature_cache_path),
        stage2_checkpoint=Path(args.stage2_checkpoint),
        git_sha=str(args.git_sha),
        resume=bool(args.resume),
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return 0 if result["summary"]["failed_items"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
