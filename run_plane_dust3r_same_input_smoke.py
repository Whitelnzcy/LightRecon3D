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
import shutil
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
TORCH_IMPORT_PATTERN = re.compile(r"(?m)^import torch[ \t]*$")
NUM_WORKERS_PATTERN = re.compile(r"(?m)^num_workers:[ \t]*[0-9]+[ \t]*$")
ROOM_PATH_PATTERN = re.compile(
    r'(?m)^([ \t]*)room_path = "/"\.join'
    r"\(image_list\[0\]\.split\('/'\)\[:-2\]\)[ \t]*$"
)
IMAGE_ID_PATTERN = re.compile(
    r"(?m)^([ \t]*)img_id = image_path\.split\('/'\)\[-2\][ \t]*$"
)

MMCV_COMPAT_INIT = '''"""Minimal MMCV 1.x compatibility surface for NonCuboidRoom."""

__version__ = "1.2.1-compat"
'''

MMCV_COMPAT_CNN = '''"""Initialization helpers used by NonCuboidRoom's HRNet."""

from torch import nn


def constant_init(module, val, bias=0):
    if getattr(module, "weight", None) is not None:
        nn.init.constant_(module.weight, val)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module,
    a=0,
    mode="fan_out",
    nonlinearity="relu",
    bias=0,
    distribution="normal",
):
    if distribution not in {"uniform", "normal"}:
        raise ValueError(f"unsupported distribution: {distribution}")
    weight = getattr(module, "weight", None)
    if weight is not None:
        initializer = (
            nn.init.kaiming_uniform_
            if distribution == "uniform"
            else nn.init.kaiming_normal_
        )
        initializer(weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)
'''

MMCV_COMPAT_RUNNER = '''"""Checkpoint loader used by NonCuboidRoom's HRNet."""

import torch


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    checkpoint = torch.load(
        filename,
        map_location=map_location,
        weights_only=True,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint state_dict must be a mapping")
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if logger is not None and (incompatible.missing_keys or incompatible.unexpected_keys):
        logger.warning(
            "checkpoint key mismatch: missing=%s unexpected=%s",
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )
    return checkpoint
'''


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
    torch_compat = (
        "import torch\n"
        "if hasattr(torch.serialization, \"add_safe_globals\"):\n"
        "    torch.serialization.add_safe_globals([argparse.Namespace])"
    )
    patched, torch_import_count = TORCH_IMPORT_PATTERN.subn(
        torch_compat,
        patched,
        count=1,
    )
    if torch_import_count != 1:
        raise ValueError(
            f"expected one torch import for safe loading, found {torch_import_count}"
        )
    return patched


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={repo.resolve()}",
            "-C",
            str(repo),
            *args,
        ],
        text=True,
        encoding="utf-8",
    ).strip()


def validate_official_worktree_status(status: str) -> list[str]:
    """Allow only generated bytecode and externally verified checkpoints."""
    allowed_untracked: list[str] = []
    unexpected: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        if not line.startswith("?? "):
            unexpected.append(line)
            continue
        relative_path = line[3:].replace("\\", "/")
        if relative_path.startswith("checkpoints/") or "/__pycache__/" in (
            f"/{relative_path}"
        ):
            allowed_untracked.append(relative_path)
        else:
            unexpected.append(line)
    if unexpected:
        raise ValueError(
            "official repository has unexpected worktree changes: "
            + "; ".join(unexpected)
        )
    return allowed_untracked


def write_mmcv_compat(runtime_dir: Path) -> Path:
    """Write only the MMCV APIs imported by the pinned NonCuboidRoom code."""
    package_dir = runtime_dir / "mmcv"
    cnn_dir = package_dir / "cnn"
    runner_dir = package_dir / "runner"
    cnn_dir.mkdir(parents=True)
    runner_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(MMCV_COMPAT_INIT, encoding="utf-8")
    (cnn_dir / "__init__.py").write_text(MMCV_COMPAT_CNN, encoding="utf-8")
    (runner_dir / "__init__.py").write_text(MMCV_COMPAT_RUNNER, encoding="utf-8")
    return package_dir


def link_or_copy(source: Path, destination: Path) -> str:
    """Prefer a read-only link, with a Windows-safe copy fallback."""
    try:
        os.symlink(
            source.resolve(),
            destination,
            target_is_directory=source.is_dir(),
        )
        return "symlink"
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        return "copy"


def build_runtime_env() -> dict[str, str]:
    """Isolate legacy TensorBoard protobuf behavior to the baseline process."""
    environment = os.environ.copy()
    environment.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    return environment


def patch_noncuboid_config(config_path: Path) -> None:
    """Disable multiprocessing only in the isolated Windows runtime."""
    if os.name != "nt":
        return
    source = config_path.read_text(encoding="utf-8")
    patched, count = NUM_WORKERS_PATTERN.subn("num_workers: 0", source)
    if count != 1:
        raise ValueError(f"expected one num_workers setting, found {count}")
    config_path.write_text(patched, encoding="utf-8")


def patch_metric_paths(metric_path: Path) -> None:
    """Use native path parsing instead of hardcoded POSIX separators."""
    source = metric_path.read_text(encoding="utf-8")
    patched, count = ROOM_PATH_PATTERN.subn(
        r"\1room_path = os.path.dirname(os.path.dirname(image_list[0]))",
        source,
    )
    if count != 3:
        raise ValueError(f"expected three POSIX-only room paths, found {count}")
    patched, image_id_count = IMAGE_ID_PATTERN.subn(
        r"\1img_id = os.path.basename(os.path.dirname(image_path))",
        patched,
    )
    if image_id_count != 3:
        raise ValueError(
            f"expected three POSIX-only image IDs, found {image_id_count}"
        )
    metric_path.write_text(patched, encoding="utf-8")


def seed_dust3r_output(
    source: Path,
    prediction_dir: Path,
    scene_name: str,
    room_id: str,
) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"missing reusable DUSt3R output: {source}")
    scene_number = scene_name.split("_", 1)[-1]
    destination_dir = prediction_dir / scene_number / room_id
    destination_dir.mkdir(parents=True)
    destination = destination_dir / "dust3r_output.npz"
    shutil.copy2(source, destination)
    return destination


def seed_room_outputs(
    source_dir: Path,
    prediction_dir: Path,
    scene_name: str,
    room_id: str,
) -> list[dict[str, Any]]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"missing reusable room output: {source_dir}")
    allowed_names = (
        "dust3r_output.npz",
        "plane_detection.json",
        "node_data.json",
        "layout.png",
    )
    scene_number = scene_name.split("_", 1)[-1]
    destination_dir = prediction_dir / scene_number / room_id
    destination_dir.mkdir(parents=True)
    copied: list[dict[str, Any]] = []
    for name in allowed_names:
        source = source_dir / name
        if not source.is_file():
            continue
        destination = destination_dir / name
        shutil.copy2(source, destination)
        copied.append(
            {
                "name": name,
                "source": str(source),
                "source_sha256": file_sha256(source),
                "destination": str(destination),
            }
        )
    required = {"dust3r_output.npz", "plane_detection.json", "node_data.json"}
    found = {row["name"] for row in copied}
    missing = sorted(required - found)
    if missing:
        raise FileNotFoundError(
            f"reusable room output is incomplete; missing {missing}"
        )
    return copied


def stage_runtime(official_repo: Path, runtime_dir: Path) -> Path:
    runtime_dir.mkdir()
    evaluator = official_repo / "evaluate_planedust3r.py"
    source = evaluator.read_text(encoding="utf-8")
    patched = patch_evaluator(source)
    patched_path = runtime_dir / evaluator.name
    patched_path.write_text(patched, encoding="utf-8")
    write_mmcv_compat(runtime_dir)
    noncuboid_source = official_repo / "NonCuboidRoom"
    noncuboid_runtime = runtime_dir / "NonCuboidRoom"
    shutil.copytree(noncuboid_source, noncuboid_runtime)
    patch_noncuboid_config(noncuboid_runtime / "cfg.yaml")
    metric_runtime = runtime_dir / "metric.py"
    shutil.copy2(official_repo / "metric.py", metric_runtime)
    patch_metric_paths(metric_runtime)

    for source_entry in official_repo.iterdir():
        if source_entry.name in {
            ".git",
            "checkpoints",
            evaluator.name,
            "metric.py",
            "mmcv",
            "NonCuboidRoom",
        }:
            continue
        link_or_copy(source_entry, runtime_dir / source_entry.name)
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
    reuse_dust3r_output: Path | None = None,
    reuse_room_output_dir: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    official_repo = official_repo.resolve()
    python_bin = python_bin.resolve()
    plane_checkpoint = plane_checkpoint.resolve()
    noncuboid_checkpoint = noncuboid_checkpoint.resolve()
    output_dir = output_dir.resolve()
    if reuse_dust3r_output is not None:
        reuse_dust3r_output = reuse_dust3r_output.resolve()
    if reuse_room_output_dir is not None:
        reuse_room_output_dir = reuse_room_output_dir.resolve()
    if reuse_dust3r_output is not None and reuse_room_output_dir is not None:
        raise ValueError(
            "reuse_dust3r_output and reuse_room_output_dir are mutually exclusive"
        )
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
    allowed_official_untracked = validate_official_worktree_status(working_tree)

    source_scene, source_item = select_scene(load_json(manifest_path), scene_name)
    output_dir.mkdir(parents=True)
    runtime_dir = output_dir / "runtime"
    audited_evaluator = stage_runtime(official_repo, runtime_dir)
    input_root = output_dir / "input"
    input_root.mkdir()
    input_mode = link_or_copy(source_scene, input_root / scene_name)
    prediction_dir = output_dir / "official_output"
    prediction_dir.mkdir()
    reused_dust3r_destination: Path | None = None
    if reuse_dust3r_output is not None:
        reused_dust3r_destination = seed_dust3r_output(
            reuse_dust3r_output,
            prediction_dir,
            scene_name,
            str(source_item["room_id"]),
        )
    reused_room_outputs: list[dict[str, Any]] | None = None
    if reuse_room_output_dir is not None:
        reused_room_outputs = seed_room_outputs(
            reuse_room_output_dir,
            prediction_dir,
            scene_name,
            str(source_item["room_id"]),
        )
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
                env=build_runtime_env(),
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
        "input_materialization_mode": input_mode,
        "protocol": "same_input_empty_render_not_native_full_protocol",
        "project_git_sha": project_git_sha,
        "official_repo": str(official_repo),
        "official_commit": official_commit,
        "official_allowed_untracked": allowed_official_untracked,
        "official_evaluator_sha256": file_sha256(
            official_repo / "evaluate_planedust3r.py"
        ),
        "audited_evaluator": str(audited_evaluator),
        "audited_evaluator_sha256": file_sha256(audited_evaluator),
        "python_bin": str(python_bin),
        "plane_checkpoint": str(plane_checkpoint),
        "noncuboid_checkpoint": str(noncuboid_checkpoint),
        "reused_dust3r_output": (
            {
                "source": str(reuse_dust3r_output),
                "source_sha256": file_sha256(reuse_dust3r_output),
                "destination": str(reused_dust3r_destination),
            }
            if reuse_dust3r_output is not None
            and reused_dust3r_destination is not None
            else None
        ),
        "reused_room_outputs": reused_room_outputs,
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
    parser.add_argument("--reuse_dust3r_output", default="")
    parser.add_argument("--reuse_room_output_dir", default="")
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
        reuse_dust3r_output=(
            Path(args.reuse_dust3r_output)
            if args.reuse_dust3r_output
            else None
        ),
        reuse_room_output_dir=(
            Path(args.reuse_room_output_dir)
            if args.reuse_room_output_dir
            else None
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
