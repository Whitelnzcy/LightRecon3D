"""Create the E8 preregistration-only package without executing E8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRATION_ID = "E8_PREREG_official_archive_acquisition_20260728_v1"
REGISTRATION_STATE = "PRE_REGISTERED_NOT_AUTHORIZED"
PROTOCOL_SHA256 = "565324e8ee8ddeb5c527fe463fb5e632b6b0b242a206d431e3e01f70df561799"
SCENE_RE = re.compile(r"scene_(\d{5})(?!\d)")
RESULT_FILES = (
    "registration.md",
    "registration_status.json",
    "git_state.json",
    "environment.json",
    "disk_preflight.json",
    "pre_acquisition_inventory.json",
    "pre_acquisition_inventory.csv",
    "pre_acquisition_inventory.sha256",
    "scene_scanner_test_results.json",
    "official_readme_source.json",
    "official_errata_source.json",
    "official_errata_scene_ids.json",
    "candidate_universe.json",
    "candidate_order_128.json",
    "candidate_generation_audit.json",
    "remote_object_manifest.json",
    "remote_object_manifest.sha256",
    "remote_zip_probe_receipts.json",
    "archive_scene_index.json",
    "scene_to_archive_map.json",
    "archive_member_index_hashes.json",
    "archive_download_receipts.json",
    "archive_sha256_manifest.json",
    "archive_retention_contract.json",
    "raw_scene_manifest.json",
    "sealed_annotation_manifest.json",
    "per_scene_provenance.csv",
    "per_scene_provenance.json",
    "clean_raw_scene_pool.json",
    "input_preparation_source_manifest.json",
    "input_preparation_config.json",
    "input_preparation_weights.json",
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


def write_empty_csv(path: Path, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


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


def receipt(stage: str, reason: str = "separate execution authorization required") -> dict:
    return {
        "registration_id": REGISTRATION_ID,
        "registration_state": REGISTRATION_STATE,
        "stage": stage,
        "status": "not_executed",
        "reason": reason,
        "not_executed": True,
    }


def scan_scene_id(value: str) -> str | None:
    match = SCENE_RE.search(value)
    return f"scene_{match.group(1)}" if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--protocol-source", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/official_archive_acquisition_e8_20260728_v1",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    source = Path(args.protocol_source).resolve()
    output = (root / args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite E8 preregistration: {output}")
    if sha256_file(source) != PROTOCOL_SHA256:
        raise ValueError("E8 protocol source SHA-256 does not match registration")

    # No D/F path is touched and no network operation occurs before or after this.
    output.mkdir(parents=True)
    (output / "registration.md").write_bytes(source.read_bytes())
    timestamp = datetime.now(timezone.utc).isoformat()
    status = {
        "registration_id": REGISTRATION_ID,
        "registration_state": REGISTRATION_STATE,
        "protocol_source_sha256": PROTOCOL_SHA256,
        "protocol_copy_sha256": sha256_file(output / "registration.md"),
        "registered_at_utc": timestamp,
        "execution_authorized": False,
        "network_probe_authorized": False,
        "download_authorized": False,
        "extraction_authorized": False,
        "D_or_F_directories_created": False,
    }
    write_json(output / "registration_status.json", status)
    write_json(
        output / "git_state.json",
        {
            **receipt("git_state_snapshot"),
            "head_sha": git_output(root, "rev-parse", "HEAD"),
            "branch": git_output(root, "branch", "--show-current"),
            "working_tree_dirty": bool(git_output(root, "status", "--porcelain")),
            "note": "snapshot only; E8 execution did not start",
        },
    )
    write_json(
        output / "environment.json",
        {
            **receipt("environment_snapshot"),
            "captured_at_utc": timestamp,
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
    )

    json_receipts = {
        "disk_preflight.json": "disk_preflight",
        "pre_acquisition_inventory.json": "pre_acquisition_inventory",
        "official_readme_source.json": "official_readme_freeze",
        "official_errata_source.json": "official_errata_freeze",
        "official_errata_scene_ids.json": "official_errata_parse",
        "candidate_universe.json": "candidate_universe_generation",
        "candidate_order_128.json": "candidate_order_generation",
        "candidate_generation_audit.json": "candidate_generation_audit",
        "remote_object_manifest.json": "remote_object_freeze",
        "remote_zip_probe_receipts.json": "remote_zip_central_directory_probe",
        "archive_scene_index.json": "archive_scene_index",
        "scene_to_archive_map.json": "scene_to_archive_map",
        "archive_member_index_hashes.json": "archive_member_index_hashes",
        "archive_download_receipts.json": "archive_download",
        "archive_sha256_manifest.json": "archive_sha256",
        "raw_scene_manifest.json": "selective_rgb_extraction",
        "sealed_annotation_manifest.json": "sealed_annotation_extraction",
        "per_scene_provenance.json": "per_scene_provenance",
        "clean_raw_scene_pool.json": "clean_raw_scene_pool",
        "input_preparation_source_manifest.json": "input_preparation_source_freeze",
        "input_preparation_config.json": "input_preparation_config_freeze",
        "input_preparation_weights.json": "input_preparation_weight_freeze",
    }
    for filename, stage in json_receipts.items():
        write_json(output / filename, receipt(stage))

    write_empty_csv(
        output / "pre_acquisition_inventory.csv",
        [
            "scene_id",
            "source_path",
            "source_type",
            "first_observed_by_E8_utc",
            "file_or_directory",
            "historical_manifest_reference",
            "historical_output_reference",
            "inventory_reason",
            "status",
        ],
    )
    (output / "pre_acquisition_inventory.sha256").write_text(
        "NOT_EXECUTED  pre_acquisition_inventory.json\n", encoding="ascii"
    )
    write_empty_csv(
        output / "per_scene_provenance.csv",
        ["scene_id", "candidate_rank", "status", "source_archive_name"],
    )
    (output / "remote_object_manifest.sha256").write_text(
        "NOT_EXECUTED  remote_object_manifest.json\n", encoding="ascii"
    )

    scanner_cases = {
        "scene_00184": "scene_00184",
        "scene_00184_room_20": "scene_00184",
        "scene_20260728": None,
        "scene_20260": "scene_20260",
        "scene_202607": None,
        "xscene_00184": "scene_00184",
        "scene_001840": None,
    }
    scanner_results = {
        key: {"expected": expected, "actual": scan_scene_id(key)}
        for key, expected in scanner_cases.items()
    }
    write_json(
        output / "scene_scanner_test_results.json",
        {
            "registration_id": REGISTRATION_ID,
            "regex": r"scene_(\d{5})(?!\d)",
            "status": (
                "passed"
                if all(row["expected"] == row["actual"] for row in scanner_results.values())
                else "failed"
            ),
            "synthetic_only": True,
            "real_inventory_executed": False,
            "cases": scanner_results,
        },
    )
    write_json(
        output / "archive_retention_contract.json",
        {
            "registration_id": REGISTRATION_ID,
            "status": "frozen_rule_not_executed",
            "completed_official_archives_must_be_retained": True,
            "archive_deletion_stage_exists": False,
            "downloaded_archive_count": 0,
        },
    )
    (output / "input_preparation_contract.md").write_text(
        "# E8 input-preparation contract\n\n"
        "Status: `not_executed`\n\n"
        "E8 may freeze this contract only after archive acquisition, selective "
        "extraction, and Clean_Raw determination. It must not run DUSt3R or "
        "Stage1 input preparation. A separate Registration ID is required for "
        "actual materialization.\n",
        encoding="utf-8",
    )

    counters = {
        "network_probe_call_count": 0,
        "download_call_count": 0,
        "archive_extraction_call_count": 0,
        "D_or_F_directory_creation_count": 0,
        "GT_semantic_parse_count": 0,
        "GT_plane_count_query_count": 0,
        "GT_quality_metric_count": 0,
        "GT_visualization_count": 0,
        "DUSt3R_input_preparation_call_count": 0,
        "Stage1_input_preparation_call_count": 0,
        "global_cache_created_count": 0,
        "direct_support_created_count": 0,
        "B_inference_call_count": 0,
        "M_inference_call_count": 0,
        "B_training_call_count": 0,
        "M_training_call_count": 0,
        "checkpoint_selection_call_count": 0,
        "threshold_calibration_call_count": 0,
        "formal_evaluation_call_count": 0,
        "GT_matcher_call_count": 0,
        "Plane_metric_call_count": 0,
        "expansion_call_count": 0,
        "refit_call_count": 0,
        "plane_fit_call_count": 0,
        "weighted_plane_fit_call_count": 0,
        "RANSAC_call_count": 0,
        "support_mutation_count": 0,
    }
    write_json(
        output / "forbidden_call_audit.json",
        {
            "registration_id": REGISTRATION_ID,
            "registration_state": REGISTRATION_STATE,
            "status": "passed_preregistration_only",
            **counters,
        },
    )
    summary = {
        "registration_id": REGISTRATION_ID,
        "registration_state": REGISTRATION_STATE,
        "execution_verdict_issued": False,
        "execution_verdict": None,
        "protocol_frozen": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "required_result_file_count": len(RESULT_FILES),
        "network_probe_executed": False,
        "inventory_executed": False,
        "disk_preflight_executed": False,
        "download_executed": False,
        "extraction_executed": False,
        "input_preparation_executed": False,
        "forbidden_counter_sum": sum(counters.values()),
        "mechanical_next_action": (
            "Stop. Wait for separate explicit authorization to execute E8. "
            "Do not interpret preregistration receipts as execution results."
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "run.log").write_text(
        "\n".join(
            [
                f"registration_id={REGISTRATION_ID}",
                f"registration_state={REGISTRATION_STATE}",
                f"timestamp_utc={timestamp}",
                "action=protocol_freeze_only",
                "inventory_executed=false",
                "network_probe_executed=false",
                "download_executed=false",
                "extraction_executed=false",
                "D_or_F_directory_creation_count=0",
                "execution_verdict_issued=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    missing = sorted(set(RESULT_FILES) - {path.name for path in output.iterdir()})
    if missing:
        raise RuntimeError(f"Missing preregistration outputs: {missing}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
