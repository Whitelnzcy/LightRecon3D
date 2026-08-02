"""Freeze the E7 pre-acquisition inventory and stop if source access is absent.

The command reads file-system metadata and text references only. It never
parses GT semantics, downloads data, loads a model/checkpoint, materializes
inputs, or computes quality metrics.
"""

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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REGISTRATION_ID = "E7_PREREG_clean_scene_acquisition_and_materialization_20260728_v1"
SALT = "LightRecon3D_E7_clean_acquisition_20260728_v1"
OFFICIAL_WEBSITE = "https://structured3d-dataset.org/"
OFFICIAL_REPOSITORY = "https://github.com/bertjiazheng/Structured3D"
OFFICIAL_ERRATA_BLOB_SHA = "859568c5df2af217fe6688265bc6db8058dc91d5"
HISTORICAL_17 = (
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
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
ARCHIVE_SUFFIXES = {".zip", ".7z", ".rar", ".tar", ".gz", ".tgz"}
SCENE_RE = re.compile(r"scene_(\d{5})", re.IGNORECASE)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def canonical_scene_id(value: str) -> str:
    match = SCENE_RE.search(value)
    if match is None:
        raise ValueError(f"No canonical scene ID in {value!r}")
    return f"scene_{match.group(1)}"


def candidate_sort_key(scene_id: str) -> tuple[str, str]:
    canonical = canonical_scene_id(scene_id)
    return sha256_bytes(f"{SALT}|{canonical}".encode("utf-8")), canonical


def parse_errata_scene_ids(path: Path) -> list[str]:
    return sorted(set(SCENE_RE.findall(path.read_text(encoding="utf-8"))))


def metadata_tree_record(scene_dir: Path) -> tuple[str, int, int, dict[str, int]]:
    entries = []
    total_bytes = 0
    file_count = 0
    suffixes: Counter[str] = Counter()
    for path in sorted(scene_dir.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(scene_dir).as_posix()
        stat = path.stat()
        if path.is_file():
            size = int(stat.st_size)
            total_bytes += size
            file_count += 1
            suffixes[path.suffix.lower() or "<no_extension>"] += 1
            kind = "file"
        elif path.is_dir():
            size = 0
            kind = "directory"
        else:
            size = int(stat.st_size)
            kind = "other"
        entries.append(
            {
                "path": relative,
                "kind": kind,
                "size": size,
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return canonical_json_hash(entries), file_count, total_bytes, dict(sorted(suffixes.items()))


def iter_reference_files(root: Path, excluded: set[Path]) -> Iterable[Path]:
    scan_roots = [
        root / "configs",
        root / "docs",
        root / "results",
        root / "local_outputs",
    ]
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            scan_roots.append(path)
    seen: set[Path] = set()
    for scan_root in scan_roots:
        if scan_root.is_file():
            paths = [scan_root]
        elif scan_root.is_dir():
            paths = scan_root.rglob("*")
        else:
            continue
        for path in paths:
            if (
                not path.is_file()
                or path in excluded
                or path in seen
                or path.suffix.lower() not in TEXT_SUFFIXES
                or path.stat().st_size > 10 * 1024 * 1024
            ):
                continue
            seen.add(path)
            yield path


def scan_scene_references(root: Path, excluded: set[Path]) -> dict[str, list[str]]:
    references: dict[str, set[str]] = defaultdict(set)
    for path in iter_reference_files(root, excluded):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        source = str(path.resolve())
        for match in SCENE_RE.finditer(text):
            references[f"scene_{match.group(1)}"].add(source)
        for match in SCENE_RE.finditer(str(path)):
            references[f"scene_{match.group(1)}"].add(source)
    return {key: sorted(values) for key, values in sorted(references.items())}


def build_pre_acquisition_inventory(
    root: Path,
    data_root: Path,
    timestamp: str,
    excluded: set[Path],
) -> dict[str, Any]:
    local_dirs = {
        path.name: path.resolve()
        for path in data_root.iterdir()
        if path.is_dir() and SCENE_RE.fullmatch(path.name)
    }
    references = scan_scene_references(root, excluded)
    all_ids = sorted(set(local_dirs) | set(references))
    records = []
    for scene_id in all_ids:
        scene_dir = local_dirs.get(scene_id)
        if scene_dir is not None:
            tree_hash, file_count, total_bytes, file_types = metadata_tree_record(scene_dir)
            absolute_path = str(scene_dir)
        else:
            tree_hash = None
            file_count = 0
            total_bytes = 0
            file_types = {}
            absolute_path = None
        sources = []
        if scene_dir is not None:
            sources.append("repository_data_directory")
        if references.get(scene_id):
            sources.append("config_log_manifest_or_document_reference")
        records.append(
            {
                "scene_id": scene_id,
                "absolute_path": absolute_path,
                "data_root": str(data_root),
                "existing_file_types": file_types,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "directory_tree_hash_kind": (
                    "SHA256(canonical relative path/type/size/mtime metadata)"
                    if scene_dir is not None
                    else None
                ),
                "directory_tree_hash": tree_hash,
                "inventory_timestamp_utc": timestamp,
                "inventory_source": sources,
                "reference_paths": references.get(scene_id, []),
                "GT_semantic_content_parsed": False,
            }
        )
    local_200 = sorted(local_dirs)
    return {
        "registration_id": REGISTRATION_ID,
        "inventory_timestamp_utc": timestamp,
        "inspection_contract": (
            "filesystem metadata and text/path references only; no GT semantic "
            "content, method predictions, or quality metrics read"
        ),
        "data_root": str(data_root),
        "explicit_historical_17": list(HISTORICAL_17),
        "current_local_200": local_200,
        "unknown_provenance_183": sorted(set(local_200) - set(HISTORICAL_17)),
        "all_scene_ids_present_before_E7": all_ids,
        "local_scene_directory_count": len(local_200),
        "all_preexisting_scene_id_count": len(all_ids),
        "records": records,
    }


def find_official_archives(search_roots: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path not in seen
                and "structured3d" in path.name.lower()
                and path.suffix.lower() in ARCHIVE_SUFFIXES
            ):
                seen.add(path)
                records.append(
                    {
                        "path": str(path.resolve()),
                        "size": int(path.stat().st_size),
                        "source_authority_proven": False,
                    }
                )
    return sorted(records, key=lambda row: row["path"])


def environment_access_hint_names() -> list[str]:
    return sorted(
        key
        for key in os.environ
        if re.search(
            r"STRUCTURED|DATASET|GDRIVE|GOOGLE.*DRIVE|DOWNLOAD.*URL",
            key,
            re.IGNORECASE,
        )
    )


def blocked_receipt(status: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "registration_id": REGISTRATION_ID,
        "status": status,
        "reason": reason,
        **extra,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-root", default="data/Structured3D")
    parser.add_argument(
        "--output-dir", default="results/clean_scene_acquisition_e7_20260728_v1"
    )
    parser.add_argument(
        "--download-search-root",
        action="append",
        default=[],
        help="Additional read-only root to inspect for an official archive.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_root = (root / args.data_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite E7 output: {output_dir}")
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    task_path = root / "docs/codex_tasks/e7_prospective_clean_scene_acquisition_20260728.md"
    errata_path = root / "docs/codex_tasks/e7_official_structured3d_errata_snapshot.txt"
    audit_path = Path(__file__).resolve()
    timestamp = datetime.now(timezone.utc).isoformat()

    # Inventory is completed before the E7 output directory is created.
    excluded = {task_path.resolve(), errata_path.resolve(), audit_path}
    inventory = build_pre_acquisition_inventory(root, data_root, timestamp, excluded)
    inventory_hash = canonical_json_hash(inventory)

    preexisting = set(inventory["all_scene_ids_present_before_E7"])
    invalid_ids = {f"scene_{value}" for value in parse_errata_scene_ids(errata_path)}
    official_universe = {f"scene_{index:05d}" for index in range(200, 3000)}
    candidate_universe = sorted(official_universe - preexisting - invalid_ids)
    ordered = sorted(candidate_universe, key=candidate_sort_key)
    frozen_128 = ordered[:128]

    archive_search_roots = [root, *[Path(value).resolve() for value in args.download_search_root]]
    archives = find_official_archives(archive_search_roots)
    env_hints = environment_access_hint_names()
    authorized_url_present = False
    source_access_available = bool(
        authorized_url_present
        or any(row["source_authority_proven"] for row in archives)
    )

    output_dir.mkdir(parents=True)
    (output_dir / "registration.md").write_text(
        task_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_json(output_dir / "pre_acquisition_inventory.json", inventory)

    official_source = {
        "registration_id": REGISTRATION_ID,
        "official_website": OFFICIAL_WEBSITE,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_dataset_size_scene_count": 3500,
        "official_split": {
            "training": ["scene_00000", "scene_02999"],
            "validation": ["scene_03000", "scene_03249"],
            "test": ["scene_03250", "scene_03499"],
        },
        "E7_allowed_source_range": ["scene_00200", "scene_02999"],
        "terms_of_use_acceptance_required": True,
        "public_page_download_instruction": (
            "submit the official agreement form; access is provided after receipt"
        ),
        "authorized_download_url_present_in_execution_context": authorized_url_present,
        "official_archive_candidates_found": archives,
        "environment_access_hint_names": env_hints,
        "official_errata": {
            "url": (
                "https://github.com/bertjiazheng/Structured3D/"
                "blob/master/metadata/errata.txt"
            ),
            "github_blob_sha": OFFICIAL_ERRATA_BLOB_SHA,
            "snapshot_path": str(errata_path),
            "snapshot_sha256": sha256_file(errata_path),
            "unique_scene_id_count": len(invalid_ids),
        },
        "source_access_available": source_access_available,
        "source_access_status": (
            "available" if source_access_available else "Blocked_Source_Access"
        ),
        "third_party_source_used": False,
    }
    write_json(output_dir / "official_source_manifest.json", official_source)

    candidate_rows = [
        {
            "order": index + 1,
            "scene_id": scene_id,
            "sort_sha256": candidate_sort_key(scene_id)[0],
            "present_before_E7": False,
            "official_invalid_case": False,
        }
        for index, scene_id in enumerate(frozen_128)
    ]
    write_json(
        output_dir / "candidate_order_128.json",
        {
            "registration_id": REGISTRATION_ID,
            "status": "frozen_before_any_download",
            "salt": SALT,
            "official_candidate_range": ["scene_00200", "scene_02999"],
            "preexisting_scene_id_count": len(preexisting),
            "official_invalid_scene_id_count": len(invalid_ids),
            "candidate_universe_count": len(candidate_universe),
            "frozen_count": len(frozen_128),
            "candidates": candidate_rows,
        },
    )

    git_state = {
        "captured_at_utc": timestamp,
        "branch": git_output(root, "branch", "--show-current"),
        "head_sha": git_output(root, "rev-parse", "HEAD"),
        "status_short": git_output(root, "status", "--short").splitlines(),
        "working_tree_dirty": bool(git_output(root, "status", "--porcelain")),
    }
    write_json(output_dir / "git_state.json", git_state)
    write_json(
        output_dir / "environment.json",
        {
            "captured_at_utc": timestamp,
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "environment_access_hint_names": env_hints,
        },
    )

    acquisition_plan = {
        "registration_id": REGISTRATION_ID,
        "status": "frozen_before_any_download",
        "salt": SALT,
        "official_source": OFFICIAL_WEBSITE,
        "official_repository": OFFICIAL_REPOSITORY,
        "candidate_universe_definition": (
            "scene_00200-scene_02999 minus all pre-E7 IDs minus all unique "
            "scene IDs in official errata blob"
        ),
        "candidate_universe_count": len(candidate_universe),
        "frozen_candidate_count": len(frozen_128),
        "frozen_candidate_order": frozen_128,
        "pre_acquisition_inventory_canonical_sha256": inventory_hash,
        "pre_acquisition_inventory_file_sha256": sha256_file(
            output_dir / "pre_acquisition_inventory.json"
        ),
        "official_source_manifest_sha256": sha256_file(
            output_dir / "official_source_manifest.json"
        ),
        "candidate_order_file_sha256": sha256_file(
            output_dir / "candidate_order_128.json"
        ),
        "generated_at_utc": timestamp,
        "git_sha": git_state["head_sha"],
        "working_tree_dirty": git_state["working_tree_dirty"],
        "download_started_before_plan_freeze": False,
        "candidate_replacement_allowed": False,
        "backup_candidate_addition_allowed": False,
    }
    write_json(output_dir / "acquisition_plan.json", acquisition_plan)
    plan_hash = sha256_file(output_dir / "acquisition_plan.json")
    (output_dir / "acquisition_plan.sha256").write_text(
        plan_hash + "  acquisition_plan.json\n", encoding="ascii"
    )

    verdict = (
        "Blocked_Insufficient_Source_Universe"
        if len(candidate_universe) < 128
        else (
            "Blocked_Source_Access"
            if not source_access_available
            else "Invalid"
        )
    )
    if verdict == "Invalid":
        raise RuntimeError(
            "Source access appears available, but this source-access audit "
            "implementation is not authorized to download. Review under a new run."
        )
    reason = (
        "official Structured3D dataset access is not available in the execution "
        "context; the public official page requires agreement and subsequent access"
        if verdict == "Blocked_Source_Access"
        else "official candidate universe contains fewer than 128 IDs"
    )

    receipts = {
        "download_receipts.json": blocked_receipt(
            "not_started", reason, download_attempt_count=0, receipts=[]
        ),
        "archive_hashes.json": blocked_receipt(
            "not_available", reason, official_checksum_unavailable=None, archives=[]
        ),
        "raw_scene_manifest.json": blocked_receipt(
            "not_acquired", reason, first_seen_scene_count=0, scenes=[]
        ),
        "sealed_annotation_audit.json": blocked_receipt(
            "not_acquired",
            reason,
            annotation_file_hash_count=0,
            GT_semantic_parse_count=0,
            GT_visualization_count=0,
            GT_plane_count_query_count=0,
        ),
        "input_preparation_manifest.json": blocked_receipt(
            "not_frozen_source_access_block",
            reason,
            DUSt3R_weight_loaded=False,
            Stage1_weight_loaded=False,
            input_preparation_authorized=False,
        ),
        "input_preparation_call_audit.json": blocked_receipt(
            "not_run",
            reason,
            input_preparation_DUSt3R_call_count=0,
            input_preparation_Stage1_call_count=0,
        ),
        "materialized_input_manifest.json": blocked_receipt(
            "not_materialized",
            reason,
            global_cache_count=0,
            direct_support_count=0,
            scenes=[],
        ),
        "clean_scene_pool.json": blocked_receipt(
            "not_created",
            reason,
            required_clean_scene_count=77,
            clean_scene_count=0,
            scenes=[],
        ),
    }
    for filename, payload in receipts.items():
        write_json(output_dir / filename, payload)

    provenance_rows = [
        {
            "order": row["order"],
            "scene_id": row["scene_id"],
            "status": "Not_Acquired_Blocked_Source_Access",
            "source_archive": "",
            "first_seen_timestamp_utc": "",
            "raw_scene_directory": "",
            "raw_tree_sha256": "",
            "global_cache_sha256": "",
            "direct_support_sha256": "",
            "GT_semantic_parsed": False,
        }
        for row in candidate_rows
    ]
    write_csv(
        output_dir / "per_scene_provenance.csv",
        [
            "order",
            "scene_id",
            "status",
            "source_archive",
            "first_seen_timestamp_utc",
            "raw_scene_directory",
            "raw_tree_sha256",
            "global_cache_sha256",
            "direct_support_sha256",
            "GT_semantic_parsed",
        ],
        provenance_rows,
    )
    write_json(
        output_dir / "per_scene_provenance.json",
        blocked_receipt("not_acquired", reason, rows=provenance_rows),
    )

    forbidden = {
        "third_party_dataset_access_count": 0,
        "preexisting_183_scene_use_count": 0,
        "historical_17_scene_use_count": 0,
        "download_attempt_count": 0,
        "archive_extract_count": 0,
        "GT_semantic_load_count": 0,
        "GT_visualization_count": 0,
        "GT_plane_count_query_count": 0,
        "B_definition_call_count": 0,
        "M_definition_call_count": 0,
        "B_inference_call_count": 0,
        "M_inference_call_count": 0,
        "training_call_count": 0,
        "checkpoint_creation_or_selection_count": 0,
        "evaluation_call_count": 0,
        "quality_metric_computation_count": 0,
        "expansion_call_count": 0,
        "refit_call_count": 0,
        "ransac_call_count": 0,
        "input_preparation_DUSt3R_call_count": 0,
        "input_preparation_Stage1_call_count": 0,
    }
    write_json(
        output_dir / "forbidden_call_audit.json",
        {
            "registration_id": REGISTRATION_ID,
            "status": "passed_at_source_access_stop",
            **forbidden,
        },
    )

    summary = {
        "registration_id": REGISTRATION_ID,
        "verdict": verdict,
        "E6_verdict": "Blocked_unchanged",
        "candidate_plan_frozen_before_download": True,
        "acquisition_plan_sha256": plan_hash,
        "candidate_universe_count": len(candidate_universe),
        "frozen_candidate_count": len(frozen_128),
        "download_started": False,
        "raw_scene_count_acquired": 0,
        "materialized_scene_count": 0,
        "clean_scene_count": 0,
        "required_clean_scene_count": 77,
        "official_source_access_available": source_access_available,
        "block_reason": reason,
        "GT_semantic_parse_count": 0,
        "quality_metric_computation_count": 0,
        "forbidden_action_counter_sum": sum(forbidden.values()),
        "historical_B_vs_M_effect": "Unverified",
        "mechanical_next_action": (
            "Stop E7. The user must independently accept the official "
            "Structured3D Terms of Use and obtain an authorized official "
            "download link. A new Registration ID must re-inventory before "
            "download; do not reuse or amend E7."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "run.log").write_text(
        "\n".join(
            [
                f"registration_id={REGISTRATION_ID}",
                f"timestamp_utc={timestamp}",
                "stage=pre_acquisition_inventory_and_source_access_gate",
                f"preexisting_scene_id_count={len(preexisting)}",
                f"official_invalid_scene_id_count={len(invalid_ids)}",
                f"candidate_universe_count={len(candidate_universe)}",
                f"frozen_candidate_count={len(frozen_128)}",
                f"acquisition_plan_sha256={plan_hash}",
                "download_attempt_count=0",
                "GT_semantic_load_count=0",
                "B_inference_call_count=0",
                "M_inference_call_count=0",
                "training_call_count=0",
                "evaluation_call_count=0",
                f"verdict={verdict}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
