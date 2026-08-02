#!/usr/bin/env python3
"""Run an auditable Plane-DUSt3R same-input batch in one model process."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from run_plane_dust3r_same_input_smoke import (
    build_runtime_env,
    file_sha256,
    git_output,
    link_or_copy,
    load_json,
    seed_room_outputs,
    stage_runtime,
    validate_official_worktree_status,
)


FLOAT64_PATTERN = re.compile(
    r"np\.float64\("
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\)"
)
PRECISION_PATTERN = re.compile(
    r"^precision:\s*([-+0-9.eE]+),\s*recall:\s*([-+0-9.eE]+)\s*$"
)
METRIC_NAMES = (
    "layout_iou",
    "pixel_error",
    "edge_error",
    "depth_rmse",
    "scale_aligned_rmse",
)
REQUIRED_ROOM_ARTIFACTS = (
    "dust3r_output.npz",
    "plane_detection.json",
    "node_data.json",
    "layout.png",
    "metric_results.txt",
)


def validate_manifest(manifest: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    protocol = manifest.get("protocol", {})
    if protocol.get("comparison_class") != "same_input_external_model_reproduction":
        raise ValueError("manifest is not a same-input external-model reproduction")
    if protocol.get("native_protocol_claim_allowed") is not False:
        raise ValueError("manifest must explicitly prohibit native-protocol claims")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("same-input manifest has no items")
    dataset_root = Path(str(manifest.get("dataset_root", ""))).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"missing materialized dataset: {dataset_root}")
    scene_names = [str(item.get("scene_name", "")) for item in items]
    if len(scene_names) != len(set(scene_names)) or any(not name for name in scene_names):
        raise ValueError("manifest scene names must be non-empty and unique")
    return dataset_root, items


def select_items(
    items: list[dict[str, Any]], scene_names: list[str]
) -> list[dict[str, Any]]:
    if not scene_names:
        return items
    requested = set(scene_names)
    selected = [item for item in items if item["scene_name"] in requested]
    missing = sorted(requested - {item["scene_name"] for item in selected})
    if missing:
        raise ValueError(f"unknown requested scenes: {', '.join(missing)}")
    return selected


def parse_metric_results(path: Path) -> dict[str, Any]:
    rows: list[list[float]] = []
    precision: float | None = None
    recall: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in FLOAT64_PATTERN.findall(line)]
        if values:
            if len(values) != len(METRIC_NAMES):
                raise ValueError(f"unexpected metric row in {path}: {line}")
            rows.append(values)
            continue
        match = PRECISION_PATTERN.match(line.strip())
        if match:
            precision, recall = map(float, match.groups())
    if not rows or precision is None or recall is None:
        raise ValueError(f"incomplete metric file: {path}")
    return {
        "view_rows": rows,
        "view_count": len(rows),
        "mean": {
            name: mean(row[index] for row in rows)
            for index, name in enumerate(METRIC_NAMES)
        },
        "plane_precision": precision,
        "plane_recall": recall,
    }


def inspect_item(
    prediction_dir: Path, item: dict[str, Any]
) -> dict[str, Any]:
    scene_name = str(item["scene_name"])
    room_id = str(item["room_id"])
    room_dir = prediction_dir / scene_name.removeprefix("scene_") / room_id
    missing = [
        name for name in REQUIRED_ROOM_ARTIFACTS if not (room_dir / name).is_file()
    ]
    result: dict[str, Any] = {
        "item_id": item.get("item_id", ""),
        "scene_name": scene_name,
        "room_id": room_id,
        "expected_views": int(item["image_count"]),
        "room_dir": str(room_dir),
        "status": "missing" if missing else "complete",
        "missing_artifacts": missing,
    }
    if not missing:
        try:
            metrics = parse_metric_results(room_dir / "metric_results.txt")
            result["metrics"] = metrics
            if metrics["view_count"] != result["expected_views"]:
                result["status"] = "invalid"
                result["error"] = (
                    f"metric rows {metrics['view_count']} != expected views "
                    f"{result['expected_views']}"
                )
        except Exception as error:
            result["status"] = "invalid"
            result["error"] = f"{type(error).__name__}: {error}"
    return result


def aggregate_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [item for item in items if item["status"] == "complete"]
    view_rows = [
        row for item in complete for row in item["metrics"]["view_rows"]
    ]
    aggregate: dict[str, Any] = {
        "requested_rooms": len(items),
        "complete_rooms": len(complete),
        "failed_or_incomplete_rooms": len(items) - len(complete),
        "complete_views": len(view_rows),
    }
    if complete:
        aggregate["room_macro_mean"] = {
            name: mean(item["metrics"]["mean"][name] for item in complete)
            for name in METRIC_NAMES
        }
        aggregate["plane_precision_room_macro_mean"] = mean(
            item["metrics"]["plane_precision"] for item in complete
        )
        aggregate["plane_recall_room_macro_mean"] = mean(
            item["metrics"]["plane_recall"] for item in complete
        )
    if view_rows:
        aggregate["view_micro_mean"] = {
            name: mean(row[index] for row in view_rows)
            for index, name in enumerate(METRIC_NAMES)
        }
    return aggregate


def write_item_csv(path: Path, items: list[dict[str, Any]]) -> None:
    fields = [
        "item_id",
        "scene_name",
        "room_id",
        "status",
        "expected_views",
        "view_count",
        *METRIC_NAMES,
        "plane_precision",
        "plane_recall",
        "room_dir",
        "missing_artifacts",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            metrics = item.get("metrics", {})
            row = {
                key: item.get(key, "")
                for key in (
                    "item_id",
                    "scene_name",
                    "room_id",
                    "status",
                    "expected_views",
                    "room_dir",
                    "error",
                )
            }
            row["missing_artifacts"] = ";".join(item["missing_artifacts"])
            row["view_count"] = metrics.get("view_count", "")
            row.update(metrics.get("mean", {}))
            row["plane_precision"] = metrics.get("plane_precision", "")
            row["plane_recall"] = metrics.get("plane_recall", "")
            writer.writerow(row)


def parse_reuse_specifications(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        scene_name, separator, source = value.partition("=")
        if not separator or not scene_name or not source:
            raise ValueError(
                f"reuse specification must be SCENE=ROOM_DIR, got {value!r}"
            )
        if scene_name in parsed:
            raise ValueError(f"duplicate reuse specification for {scene_name}")
        parsed[scene_name] = Path(source).resolve()
    return parsed


def run_batch(
    *,
    manifest_path: Path,
    official_repo: Path,
    expected_commit: str,
    python_bin: Path,
    plane_checkpoint: Path,
    noncuboid_checkpoint: Path,
    output_dir: Path,
    scene_names: list[str] | None = None,
    reuse_room_outputs: dict[str, Path] | None = None,
    project_git_sha: str = "",
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    official_repo = official_repo.resolve()
    python_bin = python_bin.resolve()
    plane_checkpoint = plane_checkpoint.resolve()
    noncuboid_checkpoint = noncuboid_checkpoint.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    for path, label in (
        (python_bin, "Python"),
        (plane_checkpoint, "Plane-DUSt3R checkpoint"),
        (noncuboid_checkpoint, "NonCuboidRoom checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")

    manifest = load_json(manifest_path)
    dataset_root, all_items = validate_manifest(manifest)
    selected = select_items(all_items, scene_names or [])
    selected_names = {str(item["scene_name"]) for item in selected}
    official_commit = git_output(official_repo, "rev-parse", "HEAD")
    if expected_commit and official_commit != expected_commit:
        raise ValueError(
            f"official commit {official_commit} != expected {expected_commit}"
        )
    allowed_untracked = validate_official_worktree_status(
        git_output(official_repo, "status", "--short")
    )

    output_dir.mkdir(parents=True)
    runtime_dir = output_dir / "runtime"
    evaluator = stage_runtime(official_repo, runtime_dir)
    if len(selected) == len(all_items):
        input_root = dataset_root
        input_mode = "direct_manifest_dataset"
    else:
        input_root = output_dir / "input"
        input_root.mkdir()
        input_modes = {
            link_or_copy(
                dataset_root / str(item["scene_name"]),
                input_root / str(item["scene_name"]),
            )
            for item in selected
        }
        input_mode = "+".join(sorted(input_modes))

    prediction_dir = output_dir / "official_output"
    prediction_dir.mkdir()
    reuse_room_outputs = reuse_room_outputs or {}
    unknown_reuse = sorted(set(reuse_room_outputs) - selected_names)
    if unknown_reuse:
        raise ValueError(f"reuse scenes not selected: {', '.join(unknown_reuse)}")
    reuse_ledger: list[dict[str, Any]] = []
    item_by_scene = {str(item["scene_name"]): item for item in selected}
    for scene_name, source_dir in reuse_room_outputs.items():
        copied = seed_room_outputs(
            source_dir,
            prediction_dir,
            scene_name,
            str(item_by_scene[scene_name]["room_id"]),
        )
        reuse_ledger.append({"scene_name": scene_name, "files": copied})

    log_path = output_dir / "batch.log"
    command = [
        str(python_bin),
        str(evaluator),
        "--dust3r_model",
        str(plane_checkpoint),
        "--noncuboid_model",
        str(noncuboid_checkpoint),
        "--root_path",
        str(input_root),
        "--save_path",
        str(prediction_dir),
        "--save_flag",
        "True",
        "--device",
        "cuda",
    ]
    started = time.perf_counter()
    completed: subprocess.CompletedProcess[str] | None = None
    launcher_error = ""
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=runtime_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                env=build_runtime_env(),
            )
    except Exception as error:
        launcher_error = f"{type(error).__name__}: {error}"
    runtime_seconds = time.perf_counter() - started
    return_code = completed.returncode if completed is not None else -1
    item_results = [inspect_item(prediction_dir, item) for item in selected]
    aggregate = aggregate_items(item_results)
    status = (
        "pass"
        if return_code == 0
        and not launcher_error
        and aggregate["complete_rooms"] == len(selected)
        else "fail"
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "plane_dust3r_same_input_batch",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "protocol": "same_input_empty_render_not_native_full_protocol",
        "native_protocol_claim_allowed": False,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "project_git_sha": project_git_sha,
        "official_repo": str(official_repo),
        "official_commit": official_commit,
        "official_allowed_untracked": allowed_untracked,
        "python_bin": str(python_bin),
        "plane_checkpoint": str(plane_checkpoint),
        "plane_checkpoint_sha256": file_sha256(plane_checkpoint),
        "noncuboid_checkpoint": str(noncuboid_checkpoint),
        "noncuboid_checkpoint_sha256": file_sha256(noncuboid_checkpoint),
        "input_root": str(input_root),
        "input_materialization_mode": input_mode,
        "reused_room_outputs": reuse_ledger,
        "command": command,
        "runtime_seconds": runtime_seconds,
        "return_code": return_code,
        "launcher_error": launcher_error,
        "log": str(log_path),
        "prediction_dir": str(prediction_dir),
        "aggregate": aggregate,
        "items": item_results,
    }
    (output_dir / "batch_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_item_csv(output_dir / "batch_metrics.csv", item_results)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same_input_manifest", required=True)
    parser.add_argument("--official_repo", required=True)
    parser.add_argument("--expected_commit", required=True)
    parser.add_argument("--python_bin", required=True)
    parser.add_argument("--plane_checkpoint", required=True)
    parser.add_argument("--noncuboid_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_name", action="append", default=[])
    parser.add_argument(
        "--reuse_room_output",
        action="append",
        default=[],
        metavar="SCENE=ROOM_DIR",
    )
    parser.add_argument("--project_git_sha", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_batch(
        manifest_path=Path(args.same_input_manifest),
        official_repo=Path(args.official_repo),
        expected_commit=args.expected_commit,
        python_bin=Path(args.python_bin),
        plane_checkpoint=Path(args.plane_checkpoint),
        noncuboid_checkpoint=Path(args.noncuboid_checkpoint),
        output_dir=Path(args.output_dir),
        scene_names=args.scene_name,
        reuse_room_outputs=parse_reuse_specifications(args.reuse_room_output),
        project_git_sha=args.project_git_sha,
    )
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
