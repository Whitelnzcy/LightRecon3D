"""Read-only archival gate for the E4 immutable-support rebaseline.

This audit intentionally stops before metric computation when a registered
historical input hash is unavailable. Computing a hash now records the current
file, but cannot prove that it is byte-identical to the input used by a
historical run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REGISTRATION_ID = "E4_PREREG_immutable_support_rebaseline_20260727_v1"
REQUIRED_INPUT_KEYS = (
    "points",
    "point_plane_ids",
    "plane_ids",
    "plane_normals",
    "plane_offsets",
    "alignment_view_indices",
    "pointmap_pixel_xy",
)
SUPPORT_HASH_KEYS = (
    "alignment_view_indices",
    "pointmap_pixel_xy",
    "point_plane_ids",
)
PARAMETER_HASH_KEYS = (
    "plane_ids",
    "plane_normals",
    "plane_offsets",
)
SCENE_METRICS = (
    "plane_f1",
    "true_positive_plane_count",
    "matched_iou_mean",
    "matched_iou_median",
    "matched_iou_minimum",
    "partition_pairwise_precision",
    "partition_pairwise_recall",
    "partition_pairwise_f1",
    "fragmentation_excess",
    "overmerge_excess",
    "gt_support_assignment_rate",
    "gt_support_matched_coverage",
    "full_cache_label_rate",
    "gt_plane_label_rate",
    "nonplane_false_assignment_rate",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(arrays: dict[str, np.ndarray], keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def scene_from_prediction(path: Path) -> str:
    suffix = "_all_fragment_identity.npz"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected prediction filename: {path.name}")
    return path.name[: -len(suffix)]


def registered_hashes(summary: dict[str, Any]) -> dict[str, Any]:
    """Return only hashes that were embedded in the historical summary.

    Sidecar hashes created after the historical run are deliberately not
    accepted unless the historical summary itself registered their identity.
    """

    value = summary.get("historical_input_hashes")
    return value if isinstance(value, dict) else {}


def audit_method_inputs(
    method: str,
    prediction_dir: Path,
    historical_summary: Path,
    historical_bounded_summary: Path,
    expected_scene_count: int,
) -> dict[str, Any]:
    summary = load_json(historical_summary)
    bounded_summary = load_json(historical_bounded_summary)
    predictions = sorted(prediction_dir.glob("*_all_fragment_identity.npz"))
    expected = registered_hashes(summary)
    referenced_dir = (
        bounded_summary.get("config", {}).get("support_prediction_dir")
    )
    referenced_matches = False
    if referenced_dir:
        referenced_matches = Path(referenced_dir).resolve() == prediction_dir.resolve()

    rows: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    for path in predictions:
        scene = scene_from_prediction(path)
        if scene in seen_scenes:
            raise ValueError(f"Duplicate scene for {method}: {scene}")
        seen_scenes.add(scene)
        with np.load(path, allow_pickle=False) as raw:
            missing = sorted(set(REQUIRED_INPUT_KEYS) - set(raw.files))
            arrays = {key: raw[key] for key in REQUIRED_INPUT_KEYS if key in raw.files}
            embedded_hash_fields = sorted(
                key
                for key in raw.files
                if "hash" in key.lower() or "sha" in key.lower()
            )
        file_hash_before = sha256_file(path)
        file_hash_after = sha256_file(path)
        expected_row = expected.get(scene)
        expected_file_hash = (
            expected_row.get("file_sha256")
            if isinstance(expected_row, dict)
            else None
        )
        support_hash = (
            sha256_arrays(arrays, SUPPORT_HASH_KEYS) if not missing else None
        )
        parameter_hash = (
            sha256_arrays(arrays, PARAMETER_HASH_KEYS) if not missing else None
        )
        rows.append(
            {
                "scene": scene,
                "path": str(path.resolve()),
                "file_size_bytes": path.stat().st_size,
                "file_mtime_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "file_sha256_current": file_hash_before,
                "file_sha256_after_read_only_audit": file_hash_after,
                "read_only_hash_stable": file_hash_before == file_hash_after,
                "support_sha256_current": support_hash,
                "plane_parameter_sha256_current": parameter_hash,
                "historical_expected_file_sha256": expected_file_hash,
                "historical_hash_registered": expected_file_hash is not None,
                "historical_hash_matches": (
                    file_hash_before == expected_file_hash
                    if expected_file_hash is not None
                    else None
                ),
                "missing_required_keys": missing,
                "embedded_hash_fields": embedded_hash_fields,
            }
        )

    return {
        "method": method,
        "prediction_dir": str(prediction_dir.resolve()),
        "historical_summary": str(historical_summary.resolve()),
        "historical_bounded_summary": str(historical_bounded_summary.resolve()),
        "bounded_summary_references_prediction_dir": referenced_matches,
        "expected_scene_count": expected_scene_count,
        "actual_scene_count": len(rows),
        "all_required_keys_present": all(
            not row["missing_required_keys"] for row in rows
        ),
        "all_current_files_read_stable": all(
            row["read_only_hash_stable"] for row in rows
        ),
        "all_historical_hashes_registered": len(rows) == expected_scene_count
        and all(row["historical_hash_registered"] for row in rows),
        "all_historical_hashes_match": len(rows) == expected_scene_count
        and all(row["historical_hash_matches"] is True for row in rows),
        "scenes": rows,
    }


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


def capture_git_state(root: Path) -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": git_output(root, "branch", "--show-current"),
        "head_sha": git_output(root, "rev-parse", "HEAD"),
        "status_short": git_output(root, "status", "--short").splitlines(),
        "diff_stat": git_output(root, "diff", "--stat").splitlines(),
        "recent_commits": git_output(root, "log", "-5", "--oneline").splitlines(),
    }


def historical_scene_rows(summary_path: Path) -> dict[str, dict[str, Any]]:
    summary = load_json(summary_path)
    return {
        row["scene"]: row
        for row in summary.get("scenes", [])
        if isinstance(row, dict) and "scene" in row
    }


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_hash_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        records.append(
            {
                "path": str(path.resolve()),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="results/immutable_support_rebaseline_20260727_v1",
    )
    parser.add_argument(
        "--registration",
        default="docs/codex_tasks/e4_immutable_support_rebaseline_20260727.md",
    )
    parser.add_argument(
        "--b-prediction-dir",
        default=(
            "local_outputs/"
            "all_fragment_identity_topology_logistic_17scene_20260725_v1/"
            "predictions"
        ),
    )
    parser.add_argument(
        "--m-prediction-dir",
        default=(
            "local_outputs/all_fragment_identity_split_val3_20260726_v2/"
            "predictions"
        ),
    )
    parser.add_argument(
        "--b-summary",
        default=(
            "local_outputs/"
            "all_fragment_identity_topology_logistic_17scene_20260725_v1/"
            "summary.json"
        ),
    )
    parser.add_argument(
        "--m-summary",
        default=(
            "local_outputs/all_fragment_identity_split_val3_20260726_v2/"
            "summary.json"
        ),
    )
    parser.add_argument(
        "--b-bounded-summary",
        default=(
            "local_outputs/"
            "all_fragment_identity_topology_logistic_bounded_common_"
            "17scene_20260725_v1/summary.json"
        ),
    )
    parser.add_argument(
        "--m-bounded-summary",
        default=(
            "local_outputs/"
            "all_fragment_identity_split_val3_bounded_common_20260726_v1/"
            "summary.json"
        ),
    )
    parser.add_argument("--expected-scene-count", type=int, default=17)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite formal E4 output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    registration_path = (root / args.registration).resolve()
    b_summary = (root / args.b_summary).resolve()
    m_summary = (root / args.m_summary).resolve()
    b_bounded_summary = (root / args.b_bounded_summary).resolve()
    m_bounded_summary = (root / args.m_bounded_summary).resolve()

    methods = {
        "B": audit_method_inputs(
            "B",
            (root / args.b_prediction_dir).resolve(),
            b_summary,
            b_bounded_summary,
            args.expected_scene_count,
        ),
        "M": audit_method_inputs(
            "M",
            (root / args.m_prediction_dir).resolve(),
            m_summary,
            m_bounded_summary,
            args.expected_scene_count,
        ),
    }

    block_reasons: list[str] = []
    for method, audit in methods.items():
        if audit["actual_scene_count"] != args.expected_scene_count:
            block_reasons.append(
                f"{method}: expected {args.expected_scene_count} scenes, "
                f"found {audit['actual_scene_count']}"
            )
        if not audit["bounded_summary_references_prediction_dir"]:
            block_reasons.append(
                f"{method}: bounded historical summary does not reference "
                "this prediction directory"
            )
        if not audit["all_required_keys_present"]:
            block_reasons.append(f"{method}: required pre-expansion arrays missing")
        if not audit["all_historical_hashes_registered"]:
            block_reasons.append(
                f"{method}: no registered historical hash for every scene input"
            )
        elif not audit["all_historical_hashes_match"]:
            block_reasons.append(f"{method}: current input hash mismatch")

    verdict = "Blocked" if block_reasons else "ReadyForImmutableEvaluation"
    evaluation_started = False
    if verdict != "Blocked":
        block_reasons.append(
            "Archival gate passed, but this audit-only command does not implement "
            "the separately reviewable immutable metric run"
        )
        verdict = "Blocked"

    input_hashes = {
        "registration_id": REGISTRATION_ID,
        "hash_definition": {
            "file": "SHA-256 of exact NPZ bytes",
            "support": list(SUPPORT_HASH_KEYS),
            "plane_parameters": list(PARAMETER_HASH_KEYS),
        },
        "important_distinction": (
            "Current hashes were first registered by E4. They are not historical "
            "expected hashes and cannot establish identity with the prior run."
        ),
        "methods": methods,
    }
    write_json(output_dir / "input_hashes.json", input_hashes)
    write_json(output_dir / "git_state.json", capture_git_state(root))

    source_paths = [
        Path(__file__).resolve(),
        registration_path,
        root / "evaluate_all_fragment_identity_end_to_end.py",
        root / "evaluate_all_fragment_common_cache.py",
        root / "evaluate_support_record_partitions.py",
        root / "audit_support_bounded_evaluator_contract.py",
        root / "apply_intra_fragment_splitter.py",
        b_summary,
        m_summary,
        b_bounded_summary,
        m_bounded_summary,
    ]
    write_json(
        output_dir / "source_hashes.json",
        {
            "registration_id": REGISTRATION_ID,
            "files": source_hash_records(source_paths),
        },
    )

    immutability = {
        "registration_id": REGISTRATION_ID,
        "verdict": verdict,
        "evaluation_started": evaluation_started,
        "prediction_output_created": False,
        "metric_comparison_performed": False,
        "expansion_call_count": 0,
        "refit_call_count": 0,
        "model_inference_call_count": 0,
        "training_call_count": 0,
        "gt_guided_prediction_call_count": 0,
        "input_files_read_only_hash_stable": all(
            method["all_current_files_read_stable"] for method in methods.values()
        ),
        "output_support_hash_equals_input_support_hash": None,
        "output_plane_parameter_hash_equals_input_plane_parameter_hash": None,
        "not_applicable_reason": (
            "The historical-hash validity gate failed before evaluator output "
            "creation. No output support or parameters exist to compare."
        ),
        "block_reasons": block_reasons,
    }
    write_json(output_dir / "immutability_audit.json", immutability)

    all_scene_names = sorted(
        {
            row["scene"]
            for method in methods.values()
            for row in method["scenes"]
        }
    )
    metric_rows = []
    for scene in all_scene_names:
        for method in ("B", "M"):
            row = {
                "scene": scene,
                "method": method,
                "status": "Blocked",
                "block_reason": (
                    "Historical expected input hash was not registered; "
                    "immutable metrics were not computed."
                ),
                "weighting": "not_computed",
            }
            row.update({metric: "" for metric in SCENE_METRICS})
            metric_rows.append(row)
    write_csv(
        output_dir / "per_scene_metrics.csv",
        ["scene", "method", "status", "block_reason", "weighting", *SCENE_METRICS],
        metric_rows,
    )
    write_csv(
        output_dir / "per_plane_metrics.csv",
        [
            "scene",
            "method",
            "plane_id",
            "status",
            "matched_gt_plane_id",
            "matched_iou",
            "point_count",
        ],
        [],
    )

    historical_b = historical_scene_rows(b_bounded_summary)
    historical_m = historical_scene_rows(m_bounded_summary)
    comparison_rows = []
    for scene in all_scene_names:
        b_row = historical_b.get(scene, {})
        m_row = historical_m.get(scene, {})
        b_f1 = b_row.get("plane_f1")
        m_f1 = m_row.get("plane_f1")
        comparison_rows.append(
            {
                "scene": scene,
                "historical_b_plane_f1": b_f1,
                "historical_m_plane_f1": m_f1,
                "historical_method_effect": (
                    m_f1 - b_f1
                    if isinstance(b_f1, (int, float))
                    and isinstance(m_f1, (int, float))
                    else ""
                ),
                "immutable_b_plane_f1": "",
                "immutable_m_plane_f1": "",
                "immutable_method_effect": "",
                "effect_retention_ratio": "",
                "effect_sign_agreement": "",
                "status": "Blocked",
            }
        )
    write_csv(
        output_dir / "historical_vs_immutable.csv",
        [
            "scene",
            "historical_b_plane_f1",
            "historical_m_plane_f1",
            "historical_method_effect",
            "immutable_b_plane_f1",
            "immutable_m_plane_f1",
            "immutable_method_effect",
            "effect_retention_ratio",
            "effect_sign_agreement",
            "status",
        ],
        comparison_rows,
    )

    registration_text = registration_path.read_text(encoding="utf-8")
    (output_dir / "registration.md").write_text(
        registration_text, encoding="utf-8"
    )
    summary = {
        "registration_id": REGISTRATION_ID,
        "verdict": verdict,
        "contract_valid": False,
        "scientific_effect_interpretation_allowed": False,
        "block_reasons": block_reasons,
        "historical_reference": {
            "B_mean_plane_f1": 0.7087,
            "M_mean_plane_f1": 0.7247,
            "method_effect": 0.0160,
        },
        "current_artifact_audit": {
            method: {
                "scene_count": audit["actual_scene_count"],
                "historical_bounded_summary_references_inputs": audit[
                    "bounded_summary_references_prediction_dir"
                ],
                "required_arrays_present": audit["all_required_keys_present"],
                "historical_hashes_registered": audit[
                    "all_historical_hashes_registered"
                ],
                "historical_hashes_match": audit["all_historical_hashes_match"],
            }
            for method, audit in methods.items()
        },
        "mechanical_route": (
            "Stop method comparison. Restore only traceable archived "
            "pre-expansion artifacts with contemporaneously registered hashes; "
            "do not rerun inference to recreate historical inputs."
        ),
        "prohibited_actions_observed": False,
        "outputs": [
            "registration.md",
            "git_state.json",
            "input_hashes.json",
            "source_hashes.json",
            "immutability_audit.json",
            "per_scene_metrics.csv",
            "per_plane_metrics.csv",
            "historical_vs_immutable.csv",
            "summary.json",
            "run.log",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    log_lines = [
        f"registration_id={REGISTRATION_ID}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        "mode=read_only_archival_gate",
        f"B_scene_count={methods['B']['actual_scene_count']}",
        f"M_scene_count={methods['M']['actual_scene_count']}",
        "B_historical_hashes_registered="
        f"{methods['B']['all_historical_hashes_registered']}",
        "M_historical_hashes_registered="
        f"{methods['M']['all_historical_hashes_registered']}",
        f"evaluation_started={evaluation_started}",
        "expansion_call_count=0",
        "refit_call_count=0",
        "model_inference_call_count=0",
        f"verdict={verdict}",
    ]
    (output_dir / "run.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 2 if verdict == "Blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
