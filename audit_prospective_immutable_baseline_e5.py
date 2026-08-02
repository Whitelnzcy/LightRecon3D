"""Pre-inference method-definition gate for E5.

This command never runs B/M inference or loads GT arrays.  It creates the
registered Blocked artifact package when the historical method cannot be
defined without violating the prospective preregistration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import scipy
import torch


REGISTRATION_ID = "E5_PREREG_prospective_immutable_baseline_rebuild_20260727_v1"
PRESSURE_SCENES = {
    "final_003_scene_00184",
    "final_013_scene_00195",
    "final_014_scene_00197",
}
METRIC_FIELDS = (
    "plane_f1",
    "true_positive_plane_count",
    "predicted_plane_count",
    "gt_plane_count",
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, fields: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


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


def environment_record() -> dict[str, Any]:
    env_keys = (
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TORCH_DETERMINISTIC",
    )
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "environment_variables": {key: os.environ.get(key) for key in env_keys},
    }


def checkpoint_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "loadable": False,
        "model_config": None,
        "state_parameter_count": None,
        "error": None,
    }
    if not path.is_file():
        record["error"] = "checkpoint missing"
        return record
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model"]
        record["model_config"] = checkpoint.get("model_config")
        record["state_parameter_count"] = int(
            sum(int(value.numel()) for value in state.values())
        )
        record["loadable"] = True
    except Exception as exc:  # pragma: no cover - diagnostic path
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def source_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "git_tracked": is_git_tracked(root, path) if path.is_file() else False,
        "mtime_utc": (
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            if path.is_file()
            else None
        ),
    }


def scene_manifest(batch: dict[str, Any], split_manifest: dict[str, Any]) -> dict:
    split_by_scene = {
        row["scene"]: row.get("split") for row in split_manifest["scenes"]
    }
    scenes = []
    for order, item in enumerate(batch["items"]):
        scene = item["id"]
        scenes.append(
            {
                "order": order,
                "scene": scene,
                "scene_name": item["scene_name"],
                "pair_group": item["pair_group"],
                "pressure_scene": scene in PRESSURE_SCENES,
                "splitter_training_manifest_split": split_by_scene.get(scene),
            }
        )
    return {
        "registration_id": REGISTRATION_ID,
        "scene_count": len(scenes),
        "pressure_scenes": sorted(PRESSURE_SCENES),
        "scene_order": [row["scene"] for row in scenes],
        "scenes": scenes,
    }


def image_records_from_cache(cache_path: Path) -> list[dict[str, Any]]:
    with np.load(cache_path, allow_pickle=False) as raw:
        registry = json.loads(str(raw["dust3r_view_registry_json"]))
    records = []
    for row in registry:
        image_path = Path(row["image_path"]).resolve()
        records.append(
            {
                "alignment_view_index": int(row["alignment_view_index"]),
                "points_hw": row["points_hw"],
                "path": str(image_path),
                "exists": image_path.is_file(),
                "sha256": sha256_file(image_path) if image_path.is_file() else None,
            }
        )
    return records


def input_hash_audit(root: Path, batch: dict[str, Any]) -> dict[str, Any]:
    scenes = []
    for item in batch["items"]:
        artifacts = item["artifacts"]
        cache_path = resolve_path(root, artifacts["global_cloud_cache"]["path"])
        support_path = resolve_path(root, artifacts["direct_support_records"]["path"])
        cache_hash = sha256_file(cache_path) if cache_path.is_file() else None
        support_hash = sha256_file(support_path) if support_path.is_file() else None
        images = image_records_from_cache(cache_path) if cache_path.is_file() else []
        scenes.append(
            {
                "scene": item["id"],
                "common_input_for_B_and_M": True,
                "global_cache": {
                    "path": str(cache_path),
                    "exists": cache_path.is_file(),
                    "sha256_current": cache_hash,
                    "sha256_recorded_in_batch": artifacts["global_cloud_cache"].get(
                        "sha256"
                    ),
                    "recorded_hash_matches": cache_hash
                    == artifacts["global_cloud_cache"].get("sha256"),
                },
                "direct_support": {
                    "path": str(support_path),
                    "exists": support_path.is_file(),
                    "sha256_current": support_hash,
                    "sha256_recorded_in_batch": artifacts[
                        "direct_support_records"
                    ].get("sha256"),
                    "recorded_hash_matches": support_hash
                    == artifacts["direct_support_records"].get("sha256"),
                },
                "view_order": [
                    row["alignment_view_index"] for row in images
                ],
                "images": images,
                "coordinate_contract": batch["coordinate_contract"],
            }
        )
    return {
        "registration_id": REGISTRATION_ID,
        "status": "prospective common inputs audited before any new inference",
        "GT_arrays_loaded": False,
        "scene_count": len(scenes),
        "all_common_input_files_exist": all(
            row["global_cache"]["exists"] and row["direct_support"]["exists"]
            for row in scenes
        ),
        "all_recorded_batch_hashes_match": all(
            row["global_cache"]["recorded_hash_matches"]
            and row["direct_support"]["recorded_hash_matches"]
            for row in scenes
        ),
        "all_images_exist": all(
            image["exists"] for row in scenes for image in row["images"]
        ),
        "scenes": scenes,
    }


def method_definition_audit(root: Path) -> dict[str, Any]:
    b_summary_path = root / (
        "local_outputs/all_fragment_identity_topology_logistic_"
        "17scene_20260725_v1/summary.json"
    )
    m_summary_path = root / (
        "local_outputs/all_fragment_identity_split_val3_20260726_v2/summary.json"
    )
    split_summary_path = root / (
        "local_outputs/intra_fragment_split_support_val3_20260726_v3/summary.json"
    )
    checkpoint_summary_path = root / (
        "local_outputs/intra_fragment_splitter_global_pairs_20260726_v1/"
        "summary.json"
    )
    checkpoint_path = root / (
        "local_outputs/intra_fragment_splitter_global_pairs_20260726_v1/best.pt"
    )
    current_apply = root / "apply_intra_fragment_splitter.py"
    archived_apply = root / (
        "local_outputs/web_collab_context_20260727_v1/source/"
        "apply_intra_fragment_splitter.py"
    )
    identity_source = root / "evaluate_all_fragment_identity_end_to_end.py"

    b_summary = load_json(b_summary_path)
    m_summary = load_json(m_summary_path)
    split_summary = load_json(split_summary_path)
    checkpoint_summary = load_json(checkpoint_summary_path)
    checkpoint = checkpoint_record(checkpoint_path)

    b_fixed_config = {
        "feature_group": b_summary.get("feature_group"),
        "model_kind": b_summary.get("model_kind"),
        "hard_negative_weight": b_summary.get("hard_negative_weight"),
        "clustering": b_summary.get("clustering"),
        **b_summary["config"],
    }
    m_config = {
        "split": split_summary["config"],
        "identity": {
            "feature_group": m_summary.get("feature_group"),
            "model_kind": m_summary.get("model_kind"),
            "hard_negative_weight": m_summary.get("hard_negative_weight"),
            "clustering": m_summary.get("clustering"),
            **m_summary["config"],
        },
    }
    historical_split_scenes = sorted(
        row["scene"] for row in split_summary.get("scenes", [])
    )
    checkpoint_val_scenes = sorted(PRESSURE_SCENES)
    checkpoint_selected_by_validation = (
        checkpoint_summary.get("best_step") is not None
        and checkpoint_summary.get("best_selection_score") is not None
    )

    current_source = source_record(root, current_apply)
    archived_source = source_record(root, archived_apply)
    identity_record = source_record(root, identity_source)
    historical_run_time = datetime.fromtimestamp(
        split_summary_path.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    archive_is_post_run = (
        archived_apply.stat().st_mtime > split_summary_path.stat().st_mtime
        if archived_apply.is_file()
        else None
    )

    block_reasons = []
    if not checkpoint["loadable"]:
        block_reasons.append("M checkpoint is missing or unreadable")
    if not is_git_tracked(root, current_apply):
        block_reasons.append(
            "M inference source was never committed; no authoritative Git "
            "version identifies the historical method"
        )
    if (
        current_source["sha256"]
        and archived_source["sha256"]
        and current_source["sha256"] != archived_source["sha256"]
    ):
        block_reasons.append(
            "M current and archived inference sources differ; the archived "
            "copy postdates the historical run and the exact run source is absent"
        )
    if checkpoint_selected_by_validation and checkpoint_val_scenes:
        block_reasons.append(
            "M checkpoint best-step selection used the three registered pressure "
            "validation scenes, which E5 permits only for final evaluation"
        )
    if set(historical_split_scenes) != PRESSURE_SCENES:
        block_reasons.append(
            "Historical M split-scene set is not the registered pressure set"
        )
    else:
        block_reasons.append(
            "Historical M ran learned split on only 3/17 scenes; E5 requires "
            "17xM inference, while the other 14 scenes trained the splitter"
        )
    if not is_git_tracked(root, identity_source):
        block_reasons.append(
            "B/M identity inference source was never committed and its "
            "historical summary registered no source hash"
        )

    return {
        "registration_id": REGISTRATION_ID,
        "status": "Blocked",
        "formal_inference_authorized": False,
        "block_reasons": block_reasons,
        "B": {
            "definition": "topology-logistic LOSO identity",
            "summary_path": str(b_summary_path),
            "fixed_config": b_fixed_config,
            "fixed_config_sha256": canonical_hash(b_fixed_config),
            "checkpoint": None,
            "identity_source": identity_record,
            "authoritative_historical_source_proven": False,
        },
        "M": {
            "definition": "old 27,730-parameter learned split/remerge + topology logistic",
            "summary_path": str(m_summary_path),
            "split_summary_path": str(split_summary_path),
            "checkpoint_summary_path": str(checkpoint_summary_path),
            "checkpoint": checkpoint,
            "config": m_config,
            "config_sha256": canonical_hash(m_config),
            "historical_split_scenes": historical_split_scenes,
            "historical_split_scene_count": len(historical_split_scenes),
            "splitter_training_scenes": 14,
            "checkpoint_selection_validation_scenes": checkpoint_val_scenes,
            "checkpoint_selected_by_validation_metric": (
                checkpoint_selected_by_validation
            ),
            "historical_split_run_time_utc": historical_run_time,
            "current_inference_source": current_source,
            "archived_inference_source": archived_source,
            "archived_source_postdates_historical_run": archive_is_post_run,
            "authoritative_historical_source_proven": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="results/prospective_immutable_baseline_e5_20260727_v1",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = resolve_path(root, args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite E5 output: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "predictions_B").mkdir()
    (output_dir / "predictions_M").mkdir()

    registration_path = (
        root / "docs/codex_tasks/e5_prospective_immutable_baseline_rebuild_20260727.md"
    )
    literature_path = (
        root / "docs/literature/LITERATURE_GATE_immutable_evaluation_e5_20260727.md"
    )
    batch_path = (
        root / "local_outputs/local_17scene_benchmark_20260718_v2/batch_execution.json"
    )
    split_manifest_path = (
        root / "local_outputs/intra_fragment_split_manifest_17scene_20260725_v2/"
        "manifest.json"
    )
    batch = load_json(batch_path)
    split_manifest = load_json(split_manifest_path)
    method_audit = method_definition_audit(root)
    scenes = scene_manifest(batch, split_manifest)
    inputs = input_hash_audit(root, batch)

    if len(scenes["scenes"]) != 17 or not PRESSURE_SCENES.issubset(
        set(scenes["scene_order"])
    ):
        method_audit["block_reasons"].append(
            "The frozen scene list is not exactly the registered 17-scene set"
        )

    (output_dir / "registration.md").write_text(
        registration_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output_dir / "literature_gate.md").write_text(
        literature_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_json(output_dir / "scene_manifest.json", scenes)
    write_json(output_dir / "input_hashes.json", inputs)

    git_state = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": git_output(root, "branch", "--show-current"),
        "head_sha": git_output(root, "rev-parse", "HEAD"),
        "status_short": git_output(root, "status", "--short").splitlines(),
        "diff_stat": git_output(root, "diff", "--stat").splitlines(),
        "recent_commits": git_output(root, "log", "-5", "--oneline").splitlines(),
    }
    write_json(output_dir / "git_state.json", git_state)
    write_json(output_dir / "environment.json", environment_record())

    source_paths = [
        Path(__file__).resolve(),
        registration_path,
        literature_path,
        root / "apply_intra_fragment_splitter.py",
        root / "train_intra_fragment_splitter.py",
        root / "evaluate_all_fragment_identity_end_to_end.py",
        root / "evaluate_support_record_partitions.py",
        root / "audit_support_bounded_evaluator_contract.py",
        root / (
            "local_outputs/web_collab_context_20260727_v1/source/"
            "apply_intra_fragment_splitter.py"
        ),
        batch_path,
        split_manifest_path,
    ]
    source_hashes = {
        "registration_id": REGISTRATION_ID,
        "files": [source_record(root, path) for path in source_paths],
    }
    write_json(output_dir / "source_hashes.json", source_hashes)

    forbidden_calls = {
        "expansion_call_count": 0,
        "refit_call_count": 0,
        "plane_fit_call_count": 0,
        "weighted_fit_plane_call_count": 0,
        "ransac_call_count": 0,
        "support_reassignment_count": 0,
        "support_added_point_count": 0,
        "support_removed_point_count": 0,
        "plane_ID_change_count": 0,
        "plane_parameter_change_count": 0,
        "model_inference_call_count": 0,
        "training_call_count": 0,
        "GT_evaluation_load_count": 0,
    }
    write_json(
        output_dir / "forbidden_call_audit.json",
        {
            "registration_id": REGISTRATION_ID,
            "status": "passed before Blocked stop",
            **forbidden_calls,
        },
    )
    write_json(
        output_dir / "immutability_audit.json",
        {
            "registration_id": REGISTRATION_ID,
            "status": "not_applicable_after_method_definition_block",
            "formal_inference_authorized": False,
            "prediction_files_created": 0,
            "prediction_output_hash_checks": [],
            "prediction_frozen_before_GT_evaluation": None,
            "GT_arrays_loaded": False,
            **forbidden_calls,
        },
    )
    write_json(
        output_dir / "prediction_manifest.json",
        {
            "registration_id": REGISTRATION_ID,
            "status": "Blocked_before_prediction",
            "expected_prediction_count": 34,
            "actual_prediction_count": 0,
            "prediction_frozen_before_GT_evaluation": None,
            "predictions_B": [],
            "predictions_M": [],
        },
    )

    metric_rows = []
    for scene in scenes["scene_order"]:
        for method in ("B", "M"):
            row = {
                "scene": scene,
                "method": method,
                "status": "Blocked",
                "weighting": "not_computed",
            }
            row.update({field: "" for field in METRIC_FIELDS})
            metric_rows.append(row)
    write_csv(
        output_dir / "per_scene_metrics.csv",
        ["scene", "method", "status", "weighting", *METRIC_FIELDS],
        metric_rows,
    )
    write_json(
        output_dir / "per_scene_metrics.json",
        {"status": "Blocked", "rows": metric_rows},
    )
    plane_fields = [
        "scene",
        "method",
        "plane_id",
        "status",
        "matched_gt_plane_id",
        "matched_iou",
        "point_count",
    ]
    write_csv(output_dir / "per_plane_metrics.csv", plane_fields, [])
    write_json(
        output_dir / "per_plane_metrics.json",
        {"status": "Blocked", "rows": []},
    )
    effect_rows = [
        {
            "scene": scene,
            "delta_f1": "",
            "delta_iou": "",
            "delta_pair_f1": "",
            "status": "Blocked",
        }
        for scene in scenes["scene_order"]
    ]
    write_csv(
        output_dir / "method_effect.csv",
        ["scene", "delta_f1", "delta_iou", "delta_pair_f1", "status"],
        effect_rows,
    )
    write_json(
        output_dir / "method_effect.json",
        {"status": "Blocked", "rows": effect_rows},
    )
    write_json(
        output_dir / "statistical_test.json",
        {
            "status": "not_run",
            "reason": "Blocked before predictions",
            "registered_test": "one-sided exact paired sign test excluding exact zeros",
        },
    )

    prerun_manifest = {
        "registration_id": REGISTRATION_ID,
        "status": "Blocked_at_method_definition_gate",
        "formal_inference_authorized": False,
        "git_sha": git_state["head_sha"],
        "working_tree_dirty": bool(git_state["status_short"]),
        "source_hashes_sha256": sha256_file(output_dir / "source_hashes.json"),
        "input_hashes_sha256": sha256_file(output_dir / "input_hashes.json"),
        "scene_manifest_sha256": sha256_file(output_dir / "scene_manifest.json"),
        "B_config_sha256": method_audit["B"]["fixed_config_sha256"],
        "M_config_sha256": method_audit["M"]["config_sha256"],
        "M_checkpoint_sha256": method_audit["M"]["checkpoint"]["sha256"],
        "random_seed": 20260725,
        "deterministic_settings": environment_record(),
        "forbidden_calls": list(forbidden_calls),
        "method_definition_audit": method_audit,
        "block_reasons": method_audit["block_reasons"],
    }
    write_json(output_dir / "prerun_manifest.json", prerun_manifest)
    manifest_hash = sha256_file(output_dir / "prerun_manifest.json")
    (output_dir / "prerun_manifest.sha256").write_text(
        manifest_hash + "  prerun_manifest.json\n", encoding="ascii"
    )

    summary = {
        "registration_id": REGISTRATION_ID,
        "verdict": "Blocked",
        "contract_valid": False,
        "formal_inference_authorized": False,
        "scientific_effect_interpretation_allowed": False,
        "block_reasons": method_audit["block_reasons"],
        "method_definition_audit": method_audit,
        "scene_count": len(scenes["scenes"]),
        "common_input_hash_audit_passed": (
            inputs["all_common_input_files_exist"]
            and inputs["all_recorded_batch_hashes_match"]
            and inputs["all_images_exist"]
        ),
        "prediction_count": {"B": 0, "M": 0, "total": 0},
        "GT_arrays_loaded": False,
        "scientific_verdict": "Blocked",
        "mechanical_next_action": (
            "Stop. Do not substitute historical predictions or guess the old "
            "method. A new preregistration must define a leakage-free checkpoint, "
            "an authoritative committed inference implementation, and a valid "
            "17-scene evaluation split before prospective inference."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    log = [
        f"registration_id={REGISTRATION_ID}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        "stage=pre_inference_method_definition_gate",
        "formal_inference_authorized=false",
        "predictions_B=0",
        "predictions_M=0",
        "GT_arrays_loaded=false",
        *[f"block_reason={reason}" for reason in method_audit["block_reasons"]],
        "verdict=Blocked",
    ]
    (output_dir / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
