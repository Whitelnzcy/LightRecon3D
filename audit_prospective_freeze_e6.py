"""Metadata-only E6 prospective method/split freeze gate.

This command inventories scene paths and required non-GT inputs, applies the
registered conservative legacy quarantine, and stops before method definition,
training, inference, checkpoint loading, or GT evaluation when fewer than 77
clean candidates are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REGISTRATION_ID = "E6_PREREG_prospective_method_and_split_freeze_20260728_v1"
SPLIT_SALT = "LightRecon3D_E6_clean_split_20260728_v1"
REQUIRED_SPLIT_COUNTS = {
    "training": 40,
    "calibration": 10,
    "selection": 10,
    "evaluation": 17,
}
HISTORICAL_SCENES = (
    "scene_00180",
    "scene_00181",
    "scene_00182",
    "scene_00184",
    "scene_00185",
    "scene_00186",
    "scene_00187",
    "scene_00189",
    "scene_00190",
    "scene_00191",
    "scene_00192",
    "scene_00193",
    "scene_00194",
    "scene_00195",
    "scene_00197",
    "scene_00198",
    "scene_00199",
)
PRESSURE_SCENES = ("scene_00184", "scene_00195", "scene_00197")
SCENE_RE = re.compile(r"scene_(\d{5})", re.IGNORECASE)
GLOBAL_CACHE_SUFFIX = "dust3r_global_cloud_cache.npz"
DIRECT_SUPPORT_SUFFIX = (
    "stage3_dust3r_global_fusion_full_pointcloud_editable_planes_data.npz"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.rstrip()


def is_git_tracked(root: Path, path: Path) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(root))],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def canonical_scene_id(value: str) -> str:
    match = SCENE_RE.search(value)
    if match is None:
        raise ValueError(f"No canonical scene ID in {value!r}")
    return f"scene_{match.group(1)}"


def split_sort_key(scene_id: str) -> tuple[str, str]:
    canonical = canonical_scene_id(scene_id)
    digest = sha256_bytes(f"{SPLIT_SALT}|{canonical}".encode("utf-8"))
    return digest, canonical


def deterministic_split(scene_ids: Iterable[str]) -> dict[str, list[str]]:
    unique = sorted({canonical_scene_id(value) for value in scene_ids}, key=split_sort_key)
    if len(unique) < sum(REQUIRED_SPLIT_COUNTS.values()):
        raise ValueError(
            f"Need 77 clean candidates, received {len(unique)}; "
            "the registered split may not be reduced"
        )
    selected = unique[:77]
    return {
        "training": selected[:40],
        "calibration": selected[40:50],
        "selection": selected[50:60],
        "evaluation": selected[60:77],
    }


def quarantine_records(all_scene_ids: Iterable[str]) -> list[dict[str, Any]]:
    historical = set(HISTORICAL_SCENES)
    pressure = set(PRESSURE_SCENES)
    records = []
    for scene_id in sorted({canonical_scene_id(value) for value in all_scene_ids}):
        if scene_id in historical:
            reasons = [
                "member_of_historical_17_scene_set",
                "entered_E0_E5_or_historical_method_analysis",
            ]
            if scene_id in pressure:
                reasons.extend(
                    [
                        "historical_pressure_scene",
                        "entered_old_checkpoint_selection_or_oracle_analysis",
                    ]
                )
            else:
                reasons.append("entered_old_splitter_training_or_historical_analysis")
            evidence_class = "explicit_historical_record"
        else:
            reasons = [
                "absence_from_known_legacy_manifests_does_not_prove_non_use",
                "usage_provenance_unproven_default_quarantine",
            ]
            evidence_class = "conservative_unproven_provenance"
        records.append(
            {
                "scene_id": scene_id,
                "quarantined": True,
                "evidence_class": evidence_class,
                "reasons": reasons,
            }
        )
    return records


def artifact_paths_by_scene(root: Path, suffix: str) -> dict[str, list[str]]:
    records: dict[str, list[str]] = defaultdict(list)
    if not root.is_dir():
        return {}
    for path in root.rglob(f"*{suffix}"):
        if not path.is_file():
            continue
        match = SCENE_RE.search(path.name)
        if match is None:
            match = SCENE_RE.search(str(path))
        if match is not None:
            records[f"scene_{match.group(1)}"].append(str(path.resolve()))
    return {key: sorted(values) for key, values in sorted(records.items())}


def rgb_counts_by_scene(data_root: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if not data_root.is_dir():
        return {}
    for path in data_root.rglob("rgb_rawlight.png"):
        match = SCENE_RE.search(str(path))
        if match is not None:
            counts[f"scene_{match.group(1)}"] += 1
    return dict(sorted(counts.items()))


def build_inventory(
    scene_ids: Iterable[str],
    quarantine: list[dict[str, Any]],
    rgb_counts: dict[str, int],
    global_caches: dict[str, list[str]],
    direct_supports: dict[str, list[str]],
) -> list[dict[str, Any]]:
    quarantine_by_scene = {row["scene_id"]: row for row in quarantine}
    rows = []
    for scene_id in sorted({canonical_scene_id(value) for value in scene_ids}):
        rgb_count = int(rgb_counts.get(scene_id, 0))
        cache_paths = global_caches.get(scene_id, [])
        support_paths = direct_supports.get(scene_id, [])
        input_complete = bool(rgb_count and cache_paths and support_paths)
        quarantined = quarantine_by_scene[scene_id]["quarantined"]
        rows.append(
            {
                "scene_id": scene_id,
                "rgb_rawlight_file_count": rgb_count,
                "global_cache_file_count": len(cache_paths),
                "direct_support_file_count": len(support_paths),
                "global_cache_paths": cache_paths,
                "direct_support_paths": support_paths,
                "required_input_presence_complete": input_complete,
                "in_legacy_quarantine": quarantined,
                "clean_candidate": bool(input_complete and not quarantined),
                "GT_content_loaded": False,
                "quality_metrics_read": False,
                "prediction_results_read": False,
                "semantic_shape_or_count_read": False,
            }
        )
    return rows


def source_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "git_tracked": is_git_tracked(root, path) if path.is_file() else False,
    }


def environment_record() -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "environment_variables": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    }


def empty_split_payload(name: str) -> dict[str, Any]:
    return {
        "registration_id": REGISTRATION_ID,
        "split": name,
        "status": "not_generated_blocked_candidate_gate",
        "required_count": REQUIRED_SPLIT_COUNTS[name],
        "actual_count": 0,
        "scenes": [],
    }


def future_contracts() -> dict[str, dict[str, Any]]:
    return {
        "checkpoint_selection_contract.json": {
            "status": "preregistered_rule_only_no_checkpoint_created",
            "candidate_rule": "one checkpoint after every completed epoch",
            "intermediate_minibatch_checkpoints_allowed": False,
            "primary_metric": "macro partition pair F1 on selection scenes",
            "tie_break": [
                "higher macro Plane F1",
                "higher macro matched IoU",
                "lower fragmentation plus overmerge",
                "earlier epoch",
            ],
            "selection_count": 1,
            "evaluation_content_allowed": False,
        },
        "calibration_contract.json": {
            "status": "preregistered_rule_only_no_calibration_run",
            "order": "after selected checkpoint hash freeze",
            "runs": 1,
            "weight_update_allowed": False,
            "checkpoint_reselection_after_calibration_allowed": False,
            "evaluation_content_allowed": False,
        },
        "prediction_freeze_contract.json": {
            "status": "preregistered_rule_only_no_prediction_created",
            "phase_order": ["P0", "P1", "P2", "P3"],
            "expected_future_prediction_count": 34,
            "prediction_frozen_before_GT_evaluation_required": True,
            "hash_fields": [
                "whole_file",
                "support_assignment",
                "plane_ID",
                "plane_parameter",
                "provenance",
                "view_pixel_global_point_index",
            ],
            "rerun_or_scene_deletion_allowed": False,
        },
        "evaluator_immutability_contract.json": {
            "status": "preregistered_rule_only_no_GT_evaluation_run",
            "allowed": ["read prediction", "read GT", "match", "compute metrics"],
            "hashes_required_unchanged": [
                "support",
                "plane_ID",
                "plane_parameter",
                "provenance",
            ],
            "prediction_mutation_allowed": False,
        },
        "forbidden_call_contract.json": {
            "status": "preregistered_and_zero_in_E6",
            "counters": {
                "expansion_call_count": 0,
                "refit_call_count": 0,
                "plane_fit_call_count": 0,
                "weighted_fit_plane_call_count": 0,
                "ransac_call_count": 0,
                "support_added_point_count": 0,
                "support_removed_point_count": 0,
                "support_reassignment_count": 0,
                "plane_ID_change_count": 0,
                "plane_parameter_change_count": 0,
                "GT_guided_prediction_call_count": 0,
                "training_call_count": 0,
                "checkpoint_load_count": 0,
                "model_inference_call_count": 0,
                "GT_semantic_content_load_count": 0,
                "quality_metric_computation_count": 0,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-root", default="data/Structured3D")
    parser.add_argument("--artifact-root", default="local_outputs")
    parser.add_argument(
        "--output-dir", default="results/prospective_freeze_e6_20260728_v1"
    )
    parser.add_argument(
        "--synthetic-test-results",
        default=None,
        help="Optional JSON produced by the focused synthetic test run.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_root = (root / args.data_root).resolve()
    artifact_root = (root / args.artifact_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite E6 output: {output_dir}")
    output_dir.mkdir(parents=True)

    registration_path = (
        root / "docs/codex_tasks/e6_prospective_method_and_split_freeze_20260728.md"
    )
    scene_ids = sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and SCENE_RE.fullmatch(path.name)
    )
    quarantine = quarantine_records(scene_ids)
    quarantine_set = {row["scene_id"] for row in quarantine}
    rgb_counts = rgb_counts_by_scene(data_root)
    global_caches = artifact_paths_by_scene(artifact_root, GLOBAL_CACHE_SUFFIX)
    direct_supports = artifact_paths_by_scene(artifact_root, DIRECT_SUPPORT_SUFFIX)
    inventory = build_inventory(
        scene_ids, quarantine, rgb_counts, global_caches, direct_supports
    )
    clean_candidates = [row["scene_id"] for row in inventory if row["clean_candidate"]]
    input_complete = [
        row["scene_id"]
        for row in inventory
        if row["required_input_presence_complete"]
    ]

    block_reasons = []
    if len(clean_candidates) < 77:
        block_reasons.append(
            f"clean candidate scene count is {len(clean_candidates)}, below "
            "the registered minimum of 77"
        )
    if set(input_complete).issubset(quarantine_set):
        block_reasons.append(
            f"all {len(input_complete)} scenes with RGB/global-cache/direct-support "
            "input presence are in legacy quarantine"
        )
    unproven_count = sum(
        row["evidence_class"] == "conservative_unproven_provenance"
        for row in quarantine
    )
    if unproven_count:
        block_reasons.append(
            f"{unproven_count} local scenes lack affirmative non-use provenance "
            "and are quarantined by the registered default rule"
        )

    verdict = "Blocked" if block_reasons else "Ready_For_Execution_Preregistration"
    if verdict != "Blocked":
        raise RuntimeError(
            "This implementation only authorizes the observed Blocked metadata "
            "path; method-definition freezing requires a new reviewed implementation"
        )

    (output_dir / "registration.md").write_text(
        registration_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_json(
        output_dir / "legacy_quarantine.json",
        {
            "registration_id": REGISTRATION_ID,
            "policy": (
                "known historical use is explicit quarantine; absence of "
                "affirmative non-use provenance is conservative quarantine"
            ),
            "historical_17": list(HISTORICAL_SCENES),
            "pressure_scenes": list(PRESSURE_SCENES),
            "local_scene_count": len(scene_ids),
            "quarantine_count": len(quarantine),
            "explicit_historical_count": sum(
                row["evidence_class"] == "explicit_historical_record"
                for row in quarantine
            ),
            "unproven_provenance_count": unproven_count,
            "scenes": quarantine,
        },
    )
    write_json(
        output_dir / "candidate_scene_inventory.json",
        {
            "registration_id": REGISTRATION_ID,
            "inspection_scope": (
                "paths and file-presence metadata only; no GT semantic arrays, "
                "predictions, quality metrics, difficulty, or plane counts read"
            ),
            "local_scene_directory_count": len(scene_ids),
            "input_presence_complete_scene_count": len(input_complete),
            "input_presence_complete_scenes": input_complete,
            "clean_candidate_count": len(clean_candidates),
            "clean_candidates": clean_candidates,
            "duplicate_source_audit": "not_applicable_no_clean_candidates",
            "scenes": inventory,
        },
    )

    split_files = {
        "training": "train_scenes.json",
        "calibration": "calibration_scenes.json",
        "selection": "selection_scenes.json",
        "evaluation": "evaluation_scenes.json",
    }
    split_hashes = {}
    for name, filename in split_files.items():
        write_json(output_dir / filename, empty_split_payload(name))
        split_hashes[name] = sha256_file(output_dir / filename)
    write_json(
        output_dir / "split_manifest.json",
        {
            "registration_id": REGISTRATION_ID,
            "status": "Blocked_before_split_generation",
            "salt": SPLIT_SALT,
            "sort_expression": 'SHA256(salt + "|" + canonical_scene_id)',
            "required_total": 77,
            "clean_candidate_count": len(clean_candidates),
            "split_file_sha256": split_hashes,
            "pairwise_disjointness": "not_applicable_empty_blocked_lists",
            "quarantine_disjointness": "not_applicable_empty_blocked_lists",
            "manual_replacement_performed": False,
            "salt_changed": False,
        },
    )

    git_state = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": git_output(root, "branch", "--show-current"),
        "head_sha": git_output(root, "rev-parse", "HEAD"),
        "status_short": git_output(root, "status", "--short").splitlines(),
        "working_tree_clean": not bool(git_output(root, "status", "--porcelain")),
        "independent_clean_worktree_created": False,
        "scientific_source_commit_created": False,
        "reason_no_source_commit": (
            "Blocked at the upstream 77-clean-candidate gate before B/M "
            "method-definition work was authorized"
        ),
    }
    write_json(output_dir / "git_state.json", git_state)
    write_json(output_dir / "environment.json", environment_record())

    test_path = root / "tests/test_audit_prospective_freeze_e6.py"
    source_paths = [Path(__file__).resolve(), registration_path, test_path]
    source_records = [source_record(root, path) for path in source_paths]
    write_json(
        output_dir / "source_manifest.json",
        {
            "registration_id": REGISTRATION_ID,
            "status": "not_frozen_blocked_before_method_definition",
            "git_commit_sha": git_state["head_sha"],
            "clean_committed_worktree": False,
            "B_scientific_sources_frozen": False,
            "M_scientific_sources_frozen": False,
            "evaluator_scientific_sources_frozen": False,
            "audit_only_sources": source_records,
            "scientific_runtime_logic_authorized": False,
        },
    )

    blocked_yaml = (
        f"registration_id: {REGISTRATION_ID}\n"
        "status: not_frozen_blocked_candidate_gate\n"
        "scientific_method_definition: absent\n"
        "formal_inference_authorized: false\n"
        "reason: clean candidate pool is below the registered minimum of 77\n"
    )
    (output_dir / "B_definition_e6.yaml").write_text(blocked_yaml, encoding="utf-8")
    (output_dir / "M_definition_e6.yaml").write_text(blocked_yaml, encoding="utf-8")

    for filename, payload in future_contracts().items():
        write_json(
            output_dir / filename,
            {"registration_id": REGISTRATION_ID, **payload},
        )

    if args.synthetic_test_results:
        synthetic_path = (root / args.synthetic_test_results).resolve()
        synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    else:
        synthetic = {
            "status": "not_supplied",
            "note": "formal package run did not receive a test result path",
        }
    write_json(
        output_dir / "synthetic_test_results.json",
        {"registration_id": REGISTRATION_ID, **synthetic},
    )

    prohibited = {
        "training_call_count": 0,
        "checkpoint_creation_count": 0,
        "checkpoint_load_count": 0,
        "checkpoint_selection_count": 0,
        "calibration_run_count": 0,
        "B_real_scene_inference_count": 0,
        "M_real_scene_inference_count": 0,
        "formal_prediction_file_count": 0,
        "GT_semantic_content_load_count": 0,
        "quality_metric_computation_count": 0,
        "expansion_call_count": 0,
        "refit_call_count": 0,
        "plane_fit_call_count": 0,
        "weighted_fit_plane_call_count": 0,
        "ransac_call_count": 0,
        "support_mutation_count": 0,
        "plane_ID_mutation_count": 0,
        "plane_parameter_mutation_count": 0,
        "provenance_mutation_count": 0,
    }
    summary = {
        "registration_id": REGISTRATION_ID,
        "verdict": verdict,
        "contract_ready": False,
        "formal_training_or_inference_authorized": False,
        "scientific_B_vs_M_comparison_performed": False,
        "H0_H1_tested": False,
        "historical_B_vs_M_effect": "Unverified",
        "local_scene_directory_count": len(scene_ids),
        "legacy_quarantine_count": len(quarantine),
        "explicit_historical_quarantine_count": len(HISTORICAL_SCENES),
        "unproven_provenance_quarantine_count": unproven_count,
        "input_presence_complete_scene_count": len(input_complete),
        "clean_candidate_count": len(clean_candidates),
        "required_clean_candidate_count": 77,
        "split_lists_frozen": False,
        "B_definition_frozen": False,
        "M_definition_frozen": False,
        "clean_source_commit_created": False,
        "block_reasons": block_reasons,
        "prohibited_action_audit": prohibited,
        "mechanical_next_action": (
            "Stop E6. Repair scene-level non-use provenance and materialize "
            "RGB/global-cache/direct-support inputs for at least 77 demonstrably "
            "clean scenes under a new Registration ID. Do not train or infer."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    log_lines = [
        f"registration_id={REGISTRATION_ID}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        "stage=metadata_only_candidate_and_quarantine_gate",
        f"local_scene_directory_count={len(scene_ids)}",
        f"input_presence_complete_scene_count={len(input_complete)}",
        f"legacy_quarantine_count={len(quarantine)}",
        f"clean_candidate_count={len(clean_candidates)}",
        "required_clean_candidate_count=77",
        "GT_semantic_content_load_count=0",
        "model_inference_call_count=0",
        "training_call_count=0",
        *[f"block_reason={reason}" for reason in block_reasons],
        f"verdict={verdict}",
    ]
    (output_dir / "run.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
