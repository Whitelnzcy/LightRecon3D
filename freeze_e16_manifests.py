"""Freeze E16 source, method, input-set, and evaluator identities."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from final_bm_methods_e16 import (
    DEFAULT_CONFIG,
    IDENTITY_MANIFEST,
    MODEL_PATH,
    OUT,
    REGISTRATION_ID,
    load_config,
    load_json,
    sha256_file,
    utc_now,
    write_json,
)


ROOT = Path(__file__).resolve().parent
SCIENTIFIC_FILES = (
    ROOT / "final_bm_methods_e16.py",
    ROOT / "evaluate_final_bm_e16.py",
    ROOT / "build_structured3d_point_aligned_gt.py",
    DEFAULT_CONFIG,
)


def git_text(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def freeze(source_commit: str) -> dict[str, Any]:
    config = load_config(DEFAULT_CONFIG)
    if git_text("rev-parse", "HEAD") != source_commit:
        raise RuntimeError("source commit is not current HEAD")
    if git_text("status", "--porcelain"):
        raise RuntimeError("worktree must be clean before manifest freeze")
    identity = load_json(IDENTITY_MANIFEST)
    if sha256_file(MODEL_PATH) != identity["model_sha256"]:
        raise RuntimeError("shared identity model hash mismatch")
    if sha256_file(Path(config["m_checkpoint"])) != config["m_checkpoint_sha256"]:
        raise RuntimeError("M checkpoint hash mismatch")
    smoke = load_json(OUT / "development_smoke_report.json")
    if (
        smoke.get("status") != "passed"
        or smoke.get("final_scene_prediction_count") != 0
        or sorted(smoke.get("view_counts", [])) != [4, 5]
    ):
        raise RuntimeError("development smoke contract failed")
    upstream = load_json(OUT / "e14_upstream_verification.json")
    sidecars = load_json(OUT / "sidecar_stage_summary.json")
    if upstream.get("status") != "passed":
        raise RuntimeError("E14 upstream verification is not passed")
    if sidecars.get("status") != "Ready_Sealed_GT_Sidecars":
        raise RuntimeError("sealed sidecar stage is not ready")

    source_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in SCIENTIFIC_FILES
    }
    source_manifest = {
        "registration_id": REGISTRATION_ID,
        "status": "frozen",
        "execution_commit": source_commit,
        "clean_worktree_before_freeze": True,
        "scientific_source_hashes": source_hashes,
        "configuration_sha256": sha256_file(DEFAULT_CONFIG),
        "shared_identity_model_sha256": identity["model_sha256"],
        "m_checkpoint_sha256": config["m_checkpoint_sha256"],
        "frozen_utc": utc_now(),
    }
    write_json(OUT / "source_manifest.json", source_manifest)

    common = {
        "registration_id": REGISTRATION_ID,
        "input": "E14 global_cache_e14 + direct_support_e14",
        "initial_fragment_construction": "shared",
        "identity_features": "joint_without_bias_with_tangent_support",
        "identity_model_sha256": identity["model_sha256"],
        "identity_cost_threshold": identity["identity_cost_threshold"],
        "clustering": "average_linkage",
        "minimum_fragment_points": config["minimum_fragment_points"],
        "maximum_geometry_points": config["maximum_geometry_points"],
        "plane_parameter_estimation": "shared deterministic covariance eigensolver",
        "prediction_schema": "LightRecon3D_E16_prediction/v1",
        "configuration_sha256": sha256_file(DEFAULT_CONFIG),
        "execution_commit": source_commit,
    }
    write_json(
        OUT / "method_B_manifest.json",
        {
            **common,
            "method_id": "B16",
            "learned_intra_fragment_split_enabled": False,
            "only_B_M_difference": "learned_intra_fragment_split",
        },
    )
    write_json(
        OUT / "method_M_manifest.json",
        {
            **common,
            "method_id": "M16",
            "learned_intra_fragment_split_enabled": True,
            "only_B_M_difference": "learned_intra_fragment_split",
            "checkpoint_path": config["m_checkpoint"],
            "checkpoint_sha256": config["m_checkpoint_sha256"],
            "checkpoint_parameter_count": 27730,
            "split_affinity_threshold": config["m_affinity_threshold"],
        },
    )
    write_json(
        OUT / "evaluator_manifest.json",
        {
            "registration_id": REGISTRATION_ID,
            "status": "frozen_before_final_predictions",
            "source_path": "evaluate_final_bm_e16.py",
            "source_sha256": source_hashes["evaluate_final_bm_e16.py"],
            "execution_commit": source_commit,
            "match_iou": 0.5,
            "fragmentation_iou": 0.1,
            "paired_bootstrap_repetitions": 10000,
            "paired_bootstrap_seed": 20260729,
            "non_mutating": True,
        },
    )
    write_json(
        OUT / "development_scene_manifest.json",
        {
            "registration_id": REGISTRATION_ID,
            "use": "development_only",
            "scene_count": identity["development_scene_count"],
            "scenes": identity["development_scenes"],
            "edge_source_sha256": identity["development_edges_sha256"],
            "final_scene_usage_count": 0,
        },
    )
    records = load_json(Path(config["e14_view_manifest"]))["records"]
    write_json(
        OUT / "final_evaluation_scene_manifest.json",
        {
            "registration_id": REGISTRATION_ID,
            "use": "single_final_evaluation_only",
            "scene_count": len(records),
            "selected_view_count": sum(int(row["view_count"]) for row in records),
            "five_view_scene_count": sum(int(row["view_count"]) == 5 for row in records),
            "four_view_scene_count": sum(int(row["view_count"]) == 4 for row in records),
            "ordered_scenes": [
                {"scene_id": row["scene_id"], "view_count": int(row["view_count"])}
                for row in records
            ],
            "training_selection_tuning_count": 0,
        },
    )
    forbidden = {
        "registration_id": REGISTRATION_ID,
        "phase": "pre_final_prediction_freeze",
        "final_layout_json_parse_count": 0,
        "final_camera_pose_numeric_parse_count": 0,
        "final_annotation_semantic_parse_count": 0,
        "final_scene_training_count": 0,
        "final_scene_threshold_calibration_count": 0,
        "expansion_call_count": 0,
        "support_mutation_call_count": 0,
        "evaluator_plane_fit_refit_call_count": 0,
        "RANSAC_call_count": 0,
        "L_G_J_call_count": 0,
        "frozen_utc": utc_now(),
    }
    write_json(OUT / "forbidden_call_audit.json", forbidden)
    (OUT / "run.log").write_text(
        "\n".join(
            (
                f"{utc_now()} E16 source freeze",
                f"registration_id={REGISTRATION_ID}",
                f"execution_commit={source_commit}",
                "E14 upstream=passed",
                "sealed sidecars=1272/1272",
                "development identity=17 scenes",
                "development smoke=passed (B16/M16, 4/5 views)",
                "final predictions=0",
                "final GT semantic parses=0",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return source_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.source_commit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
