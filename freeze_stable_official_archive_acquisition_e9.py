"""Freeze the E9 protocol and create explicit preregistration receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRATION_ID = "E9_PREREG_stable_official_archive_acquisition_20260729_v1"
STATE = "PRE_REGISTERED_NOT_AUTHORIZED"
PROTOCOL_SHA256 = "ee0f73ded7b93389d804c319986e1aa8e7e26330ab60fc50b88363c63cf77e80"
SCENE_RE = re.compile(r"scene_(\d{5})(?!\d)")
RESULT_FILES = (
    "registration.md",
    "registration_status.json",
    "git_state.json",
    "environment.json",
    "pre_acquisition_inventory.json",
    "pre_acquisition_inventory.csv",
    "pre_acquisition_inventory.sha256",
    "inventory_classification_audit.json",
    "scene_scanner_test_results.json",
    "official_text_manifest.json",
    "candidate_universe.json",
    "candidate_order_128.json",
    "candidate_generation_audit.json",
    "remote_object_probe_1.json",
    "remote_object_probe_2.json",
    "remote_object_manifest.json",
    "archive_scene_index.json",
    "scene_to_archive_map.json",
    "disk_preflight.json",
    "archive_download_receipts.json",
    "archive_sha256_manifest.json",
    "raw_scene_manifest.json",
    "sealed_annotation_manifest.json",
    "per_scene_provenance.csv",
    "per_scene_provenance.json",
    "clean_raw_scene_pool.json",
    "input_preparation_source_manifest.json",
    "input_preparation_contract.md",
    "forbidden_call_audit.json",
    "summary.json",
    "run.log",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.rstrip()


def receipt(stage: str) -> dict[str, Any]:
    return {
        "registration_id": REGISTRATION_ID,
        "registration_state": STATE,
        "stage": stage,
        "status": "not_executed",
        "reason": "separate explicit E9 execution authorization required",
        "not_executed": True,
    }


def scan_scene_ids(value: str) -> list[str]:
    return [
        f"scene_{match.group(1)}" for match in SCENE_RE.finditer(value)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--protocol-source", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/stable_official_archive_acquisition_e9_20260729_v1",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    source = Path(args.protocol_source).resolve()
    output = (root / args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite E9 package: {output}")
    if sha256_file(source) != PROTOCOL_SHA256:
        raise ValueError("E9 protocol SHA-256 mismatch")

    output.mkdir(parents=True)
    (output / "registration.md").write_bytes(source.read_bytes())
    timestamp = datetime.now(timezone.utc).isoformat()
    write_json(output / "registration_status.json", {
        "registration_id": REGISTRATION_ID,
        "registration_state": STATE,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_copy_sha256": sha256_file(output / "registration.md"),
        "registered_at_utc": timestamp,
        "execution_authorized": False,
        "inventory_authorized": False,
        "network_probe_authorized": False,
        "download_authorized": False,
        "extraction_authorized": False,
        "D_or_F_directories_created": False,
        "E8_final_verdict_preserved": "Blocked_Official_Metadata_Changed",
    })
    write_json(output / "git_state.json", {
        **receipt("git_state_snapshot"),
        "head_sha": git_output(root, "rev-parse", "HEAD"),
        "branch": git_output(root, "branch", "--show-current"),
        "working_tree_dirty": bool(git_output(root, "status", "--porcelain")),
    })
    write_json(output / "environment.json", {
        **receipt("environment_snapshot"),
        "captured_at_utc": timestamp,
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
    })

    stages = {
        "pre_acquisition_inventory.json": "pre_acquisition_inventory",
        "inventory_classification_audit.json": "inventory_classification",
        "official_text_manifest.json": "official_text_content_freeze",
        "candidate_universe.json": "candidate_universe_generation",
        "candidate_order_128.json": "candidate_order_generation",
        "candidate_generation_audit.json": "candidate_generation_audit",
        "remote_object_probe_1.json": "first_independent_remote_probe",
        "remote_object_probe_2.json": "second_independent_remote_probe",
        "remote_object_manifest.json": "dual_probe_remote_identity_freeze",
        "archive_scene_index.json": "archive_scene_index",
        "scene_to_archive_map.json": "scene_to_archive_map",
        "disk_preflight.json": "disk_preflight",
        "archive_download_receipts.json": "archive_download",
        "archive_sha256_manifest.json": "whole_archive_sha256",
        "raw_scene_manifest.json": "selective_rgb_extraction",
        "sealed_annotation_manifest.json": "sealed_annotation_extraction",
        "per_scene_provenance.json": "per_scene_provenance",
        "clean_raw_scene_pool.json": "clean_raw_scene_pool",
        "input_preparation_source_manifest.json": "input_preparation_freeze",
    }
    for filename, stage in stages.items():
        write_json(output / filename, receipt(stage))

    with (output / "pre_acquisition_inventory.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        csv.writer(handle).writerow([
            "scene_id",
            "source_path",
            "classification",
            "disqualifying",
            "reason",
            "status",
        ])
    (output / "pre_acquisition_inventory.sha256").write_text(
        "NOT_EXECUTED  pre_acquisition_inventory.json\n",
        encoding="ascii",
    )
    with (output / "per_scene_provenance.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        csv.writer(handle).writerow([
            "scene_id",
            "candidate_rank",
            "status",
            "source_archive_name",
        ])

    cases = {
        "scene_00184": ["scene_00184"],
        "scene_20260728": [],
        "scene_001840": [],
        "scene_20260": ["scene_20260"],
    }
    results = {
        value: {
            "expected": expected,
            "actual": scan_scene_ids(value),
        }
        for value, expected in cases.items()
    }
    write_json(output / "scene_scanner_test_results.json", {
        "registration_id": REGISTRATION_ID,
        "regex": r"scene_(\d{5})(?!\d)",
        "status": (
            "passed"
            if all(row["expected"] == row["actual"] for row in results.values())
            else "failed"
        ),
        "synthetic_only": True,
        "real_inventory_executed": False,
        "cases": results,
    })
    (output / "input_preparation_contract.md").write_text(
        "# E9 input-preparation contract\n\n"
        "Status: `not_executed`\n\n"
        "E9 may freeze this definition only after authorized archive download, "
        "selective extraction, and Clean Raw qualification. It may not "
        "materialize DUSt3R or Stage1 inputs. Actual materialization requires "
        "another Registration ID.\n",
        encoding="utf-8",
    )
    forbidden = {
        name: 0
        for name in (
            "GT_semantic_parse_count",
            "GT_plane_count_query_count",
            "GT_visualization_count",
            "GT_quality_metric_count",
            "DUSt3R_input_preparation_call_count",
            "Stage1_input_preparation_call_count",
            "global_cache_created_count",
            "direct_support_created_count",
            "B_inference_call_count",
            "M_inference_call_count",
            "B_training_call_count",
            "M_training_call_count",
            "checkpoint_selection_call_count",
            "threshold_calibration_call_count",
            "formal_evaluation_call_count",
            "GT_matcher_call_count",
            "Plane_metric_call_count",
            "expansion_call_count",
            "refit_call_count",
            "plane_fit_call_count",
            "weighted_plane_fit_call_count",
            "RANSAC_call_count",
            "support_mutation_count",
        )
    }
    write_json(output / "forbidden_call_audit.json", {
        "registration_id": REGISTRATION_ID,
        "registration_state": STATE,
        "status": "passed_preregistration_only",
        **forbidden,
        "forbidden_counter_sum": sum(forbidden.values()),
    })
    summary = {
        "registration_id": REGISTRATION_ID,
        "registration_state": STATE,
        "execution_verdict_issued": False,
        "execution_verdict": None,
        "protocol_frozen": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "required_result_file_count": len(RESULT_FILES),
        "inventory_executed": False,
        "network_probe_executed": False,
        "download_executed": False,
        "extraction_executed": False,
        "input_preparation_executed": False,
        "D_or_F_directories_created": False,
        "forbidden_counter_sum": sum(forbidden.values()),
        "mechanical_next_action": (
            "Stop. Wait for separate explicit authorization to execute E9."
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "run.log").write_text(
        "\n".join([
            f"registration_id={REGISTRATION_ID}",
            f"registration_state={STATE}",
            f"timestamp_utc={timestamp}",
            "action=protocol_freeze_only",
            "inventory_executed=false",
            "network_probe_executed=false",
            "download_executed=false",
            "extraction_executed=false",
            "D_or_F_directory_creation_count=0",
            "execution_verdict_issued=false",
        ]) + "\n",
        encoding="utf-8",
    )
    missing = sorted(set(RESULT_FILES) - {path.name for path in output.iterdir()})
    if missing:
        raise RuntimeError(f"missing E9 preregistration outputs: {missing}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
