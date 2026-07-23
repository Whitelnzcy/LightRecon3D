#!/usr/bin/env python3
"""Run one auditable Plane-DUSt3R same-input scene.

The official evaluator currently hardcodes output saving off and silently
continues after a room-level exception.  This wrapper creates an isolated
runtime view of the pinned external repository, patches exactly those two
statements, and writes a success/failure ledger even when inference fails.
The external repository itself is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAVE_PATTERN = re.compile(r"(?m)^([ \t]*)save = False[ \t]*$")
EXCEPT_PATTERN = re.compile(
    r"(?m)^(?P<indent>[ \t]*)except Exception as e:[ \t]*\r?\n"
    r"(?P=indent)[ \t]+continue[ \t]*$"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def patch_evaluator(source: str) -> str:
    patched, save_count = SAVE_PATTERN.subn(r"\1save = bool(args.save_flag)", source)
    if save_count != 1:
        raise ValueError(f"expected one hardcoded save flag, found {save_count}")

    def replace_exception(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}except Exception as e:\n"
            f"{indent}    traceback.print_exc()\n"
            f"{indent}    raise"
        )

    patched, exception_count = EXCEPT_PATTERN.subn(replace_exception, patched)
    if exception_count != 1:
        raise ValueError(
            f"expected one silent room exception, found {exception_count}"
        )
    return patched


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def stage_runtime(official_repo: Path, runtime_dir: Path) -> Path:
    runtime_dir.mkdir()
    evaluator = official_repo / "evaluate_planedust3r.py"
    source = evaluator.read_text(encoding="utf-8")
    patched = patch_evaluator(source)
    patched_path = runtime_dir / evaluator.name
    patched_path.write_text(patched, encoding="utf-8")

    for source_entry in official_repo.iterdir():
        if source_entry.name in {".git", evaluator.name}:
            continue
        os.symlink(
            source_entry.resolve(),
            runtime_dir / source_entry.name,
            target_is_directory=source_entry.is_dir(),
        )
    return patched_path


def select_scene(manifest: dict[str, Any], scene_name: str) -> tuple[Path, dict[str, Any]]:
    protocol = manifest.get("protocol", {})
    if protocol.get("comparison_class") != "same_input_external_model_reproduction":
        raise ValueError("input is not a same-input external-model manifest")
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("same-input manifest has no items list")
    matches = [row for row in items if row.get("scene_name") == scene_name]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest item for {scene_name}, found {len(matches)}")
    dataset_root = Path(str(manifest.get("dataset_root", "")))
    source_scene = dataset_root / scene_name
    if not source_scene.is_dir():
        raise FileNotFoundError(f"missing materialized scene: {source_scene}")
    return source_scene, matches[0]


def run_smoke(
    *,
    manifest_path: Path,
    scene_name: str,
    official_repo: Path,
    expected_commit: str,
    python_bin: Path,
    plane_checkpoint: Path,
    noncuboid_checkpoint: Path,
    output_dir: Path,
    project_git_sha: str = "",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    if not python_bin.is_file():
        raise FileNotFoundError(f"missing Plane-DUSt3R Python: {python_bin}")
    if not plane_checkpoint.is_file():
        raise FileNotFoundError(f"missing Plane-DUSt3R checkpoint: {plane_checkpoint}")
    if not noncuboid_checkpoint.is_file():
        raise FileNotFoundError(f"missing NonCuboidRoom checkpoint: {noncuboid_checkpoint}")
    if not (official_repo / "evaluate_planedust3r.py").is_file():
        raise FileNotFoundError(f"invalid official repository: {official_repo}")

    official_commit = git_output(official_repo, "rev-parse", "HEAD")
    if expected_commit and official_commit != expected_commit:
        raise ValueError(
            f"official commit {official_commit} != expected {expected_commit}"
        )
    working_tree = git_output(official_repo, "status", "--short")
    if working_tree:
        raise ValueError("official repository must have a clean working tree")

    source_scene, source_item = select_scene(load_json(manifest_path), scene_name)
    output_dir.mkdir(parents=True)
    runtime_dir = output_dir / "runtime"
    audited_evaluator = stage_runtime(official_repo, runtime_dir)
    input_root = output_dir / "input"
    input_root.mkdir()
    os.symlink(source_scene.resolve(), input_root / scene_name, target_is_directory=True)
    prediction_dir = output_dir / "official_output"
    prediction_dir.mkdir()
    log_path = output_dir / "smoke.log"

    command = [
        str(python_bin),
        str(audited_evaluator),
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
    error_text = ""
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=runtime_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    except Exception as error:  # ledger must survive launcher errors
        error_text = f"{type(error).__name__}: {error}"
    runtime_seconds = time.perf_counter() - started
    return_code = completed.returncode if completed is not None else -1
    artifacts = sorted(
        str(path.relative_to(output_dir))
        for path in prediction_dir.rglob("*")
        if path.is_file()
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "plane_dust3r_same_input_smoke",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if return_code == 0 and not error_text else "fail",
        "scene_name": scene_name,
        "source_item": source_item,
        "protocol": "same_input_empty_render_not_native_full_protocol",
        "project_git_sha": project_git_sha,
        "official_repo": str(official_repo),
        "official_commit": official_commit,
        "official_evaluator_sha256": file_sha256(
            official_repo / "evaluate_planedust3r.py"
        ),
        "audited_evaluator": str(audited_evaluator),
        "audited_evaluator_sha256": file_sha256(audited_evaluator),
        "python_bin": str(python_bin),
        "plane_checkpoint": str(plane_checkpoint),
        "noncuboid_checkpoint": str(noncuboid_checkpoint),
        "command": command,
        "runtime_seconds": runtime_seconds,
        "return_code": return_code,
        "launcher_error": error_text,
        "log": str(log_path),
        "prediction_dir": str(prediction_dir),
        "prediction_artifacts": artifacts,
    }
    (output_dir / "smoke_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if result["status"] != "pass":
        raise RuntimeError(
            f"Plane-DUSt3R smoke failed with return code {return_code}; see {log_path}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same_input_manifest", required=True)
    parser.add_argument("--scene_name", default="scene_00180")
    parser.add_argument("--official_repo", required=True)
    parser.add_argument("--expected_commit", required=True)
    parser.add_argument("--python_bin", required=True)
    parser.add_argument("--plane_checkpoint", required=True)
    parser.add_argument("--noncuboid_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_git_sha", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_smoke(
        manifest_path=Path(args.same_input_manifest),
        scene_name=args.scene_name,
        official_repo=Path(args.official_repo),
        expected_commit=args.expected_commit,
        python_bin=Path(args.python_bin),
        plane_checkpoint=Path(args.plane_checkpoint),
        noncuboid_checkpoint=Path(args.noncuboid_checkpoint),
        output_dir=Path(args.output_dir),
        project_git_sha=args.project_git_sha,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
