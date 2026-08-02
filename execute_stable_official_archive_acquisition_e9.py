"""Execute the frozen E9 stable official archive acquisition protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from execute_official_archive_acquisition_e8 import (
    ANNOTATION_NAME,
    BASE_URL,
    BOUNDARY_PERSPECTIVE,
    FORMAL_PERSPECTIVE,
    GIB,
    E8Stop,
    PERSPECTIVE_NAMES,
    TEXT_SUFFIXES,
    atomic_write,
    canonical_json,
    extract_members,
    get_range,
    local_member_listing,
    parse_central_directory,
    remove_incomplete_staging,
    scene_ids,
    sha256_bytes,
    sha256_file,
    tree_sha256,
    utc_now,
    write_csv,
    write_json,
)


REGISTRATION_ID = "E9_PREREG_stable_official_archive_acquisition_20260729_v1"
PROTOCOL_SHA256 = "ee0f73ded7b93389d804c319986e1aa8e7e26330ab60fc50b88363c63cf77e80"
SALT = "LightRecon3D_E7_clean_acquisition_20260728_v1"
README_URL = f"{BASE_URL}/README.txt"
ERRATA_URL = (
    "https://raw.githubusercontent.com/bertjiazheng/Structured3D/"
    "master/metadata/errata.txt"
)
ALLOWED_VERDICTS = {
    "Ready_Clean_Raw_Pool",
    "Paused_Resumable",
    "Paused_Low_Disk",
    "Blocked_Official_Text_Content_Changed",
    "Blocked_Remote_Object_Unstable",
    "Blocked_Archive_Mapping",
    "Blocked_Insufficient_Candidate_Universe",
    "Blocked_Insufficient_D_Space",
    "Blocked_Insufficient_F_Space",
    "Blocked_Insufficient_E_Space",
    "Blocked_Insufficient_Clean_Raw_Scenes",
    "Blocked_Input_Preparation_Freeze",
    "Invalid",
}
PLAN_MARKERS = (
    "clean_scene_acquisition_e7",
    "official_archive_acquisition_e8",
    "prospective_clean_scene_acquisition_e7",
    "e7_official_structured3d",
    "e8_official_archive",
)
HISTORICAL_MARKERS = (
    "local_outputs",
    "global_cache",
    "direct_support",
    "prediction",
    "checkpoint",
    "oracle",
    "evaluation",
    "rebaseline",
    "baseline",
    "training",
    "train_",
    "badcase",
)


class E9Stop(RuntimeError):
    def __init__(self, verdict: str, reason: str):
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"unregistered E9 verdict: {verdict}")
        super().__init__(reason)
        self.verdict = verdict
        self.reason = reason


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


def classify_reference(path: Path, scan_root_type: str) -> tuple[str, str]:
    normalized = str(path).replace("\\", "/").lower()
    if any(marker in normalized for marker in PLAN_MARKERS):
        return (
            "acquisition_plan_only",
            "E7/E8 registration, candidate plan, blocked receipt, or return",
        )
    if any(marker in normalized for marker in HISTORICAL_MARKERS):
        return (
            "historical_scientific_use",
            "historical model input, output, prediction, training, or evaluation evidence",
        )
    if scan_root_type.endswith("_known_structured3d"):
        return (
            "physical_data_presence",
            "pre-existing file or directory in a known Structured3D data root",
        )
    if "structured3d" in normalized and path.suffix.lower() in {
        ".png", ".jpg", ".jpeg", ".webp", ".npz", ".npy", ".json",
    }:
        return (
            "physical_data_presence",
            "pre-existing Structured3D data-like file",
        )
    return (
        "unclassified_reference",
        "reference could not be proven to be acquisition-plan-only",
    )


def build_inventory(root: Path) -> tuple[list[dict], set[str], dict]:
    observed = utc_now()
    scan_roots: list[tuple[Path, str]] = [(root, "repository")]
    for drive in ("D:/", "F:/"):
        base = Path(drive)
        if not base.exists():
            continue
        for child in base.iterdir():
            if "structured3d" in child.name.lower():
                scan_roots.append(
                    (child, f"{drive[0]}_known_structured3d")
                )
    records: dict[tuple[str, str, str, str], dict] = {}

    def add(
        scene: str, source: Path, root_type: str, observation: str
    ) -> None:
        classification, reason = classify_reference(source, root_type)
        key = (scene, str(source), observation, classification)
        records[key] = {
            "scene_id": scene,
            "source_path": str(source),
            "source_root_type": root_type,
            "observation": observation,
            "classification": classification,
            "disqualifying": classification != "acquisition_plan_only",
            "reason": reason,
            "first_observed_by_E9_utc": observed,
            "status": "pre_existing_reference",
        }

    for scan_root, root_type in scan_roots:
        if not scan_root.exists():
            continue
        for current, directories, files in os.walk(scan_root):
            current_path = Path(current)
            directories[:] = [name for name in directories if name != ".git"]
            for name in directories + files:
                item = current_path / name
                for scene in scene_ids(str(item)):
                    add(scene, item, root_type, "path_or_filename")
            for name in files:
                item = current_path / name
                try:
                    size = item.stat().st_size
                except OSError:
                    continue
                if (
                    item.suffix.lower() not in TEXT_SUFFIXES
                    or size > 64 * 1024**2
                ):
                    continue
                try:
                    text = item.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for scene in sorted(set(scene_ids(text))):
                    add(scene, item, root_type, "text_content")
    rows = sorted(
        records.values(),
        key=lambda row: (
            row["scene_id"],
            row["source_path"],
            row["classification"],
            row["observation"],
        ),
    )
    classes_by_scene: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        classes_by_scene[row["scene_id"]].add(row["classification"])
    disqualifying = {
        scene
        for scene, classes in classes_by_scene.items()
        if classes - {"acquisition_plan_only"}
    }
    audit = {
        "status": "completed",
        "classification_counts": dict(
            sorted(Counter(row["classification"] for row in rows).items())
        ),
        "unique_scene_count": len(classes_by_scene),
        "disqualifying_scene_count": len(disqualifying),
        "acquisition_plan_only_scene_count": sum(
            classes == {"acquisition_plan_only"}
            for classes in classes_by_scene.values()
        ),
        "rules": {
            "acquisition_plan_only_is_not_disqualifying": True,
            "unclassified_reference_is_disqualifying": True,
            "physical_data_presence_is_disqualifying": True,
            "historical_scientific_use_is_disqualifying": True,
        },
        "scene_classifications": {
            scene: sorted(classes)
            for scene, classes in sorted(classes_by_scene.items())
        },
    }
    return rows, disqualifying, audit


def response_metadata(response: requests.Response) -> dict[str, Any]:
    return {
        "final_url": response.url,
        "http_status": response.status_code,
        "content_length_header": response.headers.get("Content-Length"),
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "content_type": response.headers.get("Content-Type"),
        "http_protocol_version": getattr(response.raw, "version", None),
        "server": response.headers.get("Server"),
        "date": response.headers.get("Date"),
        "probe_method": response.request.method,
    }


def get_text_response(
    session: requests.Session, role: str, url: str
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            response = session.get(
                url, allow_redirects=True, timeout=(30, 120)
            )
            if response.status_code == 200:
                return response
            last_error = requests.RequestException(
                f"HTTP {response.status_code}"
            )
            response.close()
        except requests.RequestException as exc:
            last_error = exc
        if attempt < 5:
            time.sleep(2**attempt)
    raise E9Stop(
        "Paused_Resumable",
        f"{role} retrieval interrupted after retries: {last_error}",
    )


def fetch_text(
    session: requests.Session, role: str, url: str, output: Path
) -> tuple[bytes, dict]:
    response = get_text_response(session, role, url)
    body = response.content
    atomic_write(output, body)
    manifest = {
        "role": role,
        "requested_url": url,
        **response_metadata(response),
        "response_body_sha256": sha256_bytes(body),
        "bytes_received": len(body),
        "retrieval_utc": utc_now(),
        "snapshot_path": str(output),
        "status": "frozen",
    }
    return body, manifest


def range_0_total(
    session: requests.Session, url: str
) -> tuple[int, requests.Response, bytes]:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            response = session.get(
                url,
                headers={"Range": "bytes=0-0"},
                stream=True,
                allow_redirects=True,
                timeout=(30, 180),
            )
            body = response.content
            content_range = response.headers.get("Content-Range", "")
            if response.status_code == 206 and content_range.startswith(
                "bytes 0-0/"
            ):
                return int(content_range.rsplit("/", 1)[-1]), response, body
            length = response.headers.get("Content-Length")
            if response.status_code == 200 and length and length.isdigit():
                # A full-body 200 for a 0-0 request is unsafe for large objects.
                response.close()
                raise E9Stop(
                    "Blocked_Remote_Object_Unstable",
                    f"server ignored fixed Range for {url}",
                )
            response.close()
            raise requests.RequestException(
                f"invalid 0-0 response HTTP={response.status_code} "
                f"Content-Range={content_range!r}"
            )
        except E9Stop:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(2**attempt)
    raise E9Stop(
        "Paused_Resumable",
        f"0-0 Range probe interrupted after retries: {last_error}",
    )


def canonical_members(archive_name: str, members: list[dict]) -> list[dict]:
    rows = [
        {
            "archive_name": archive_name,
            "normalized_member_path": member["path"].replace("\\", "/"),
            "CRC32": member["crc32"].lower(),
            "compressed_size": int(member["compressed_size"]),
            "uncompressed_size": int(member["uncompressed_size"]),
            "compression_method": int(member["compression_method"]),
            "local_header_offset": int(member["local_header_offset"]),
        }
        for member in members
    ]
    return sorted(
        rows,
        key=lambda row: (
            row["archive_name"],
            row["normalized_member_path"],
            row["local_header_offset"],
        ),
    )


def remote_zip_index_e9(
    session: requests.Session, identity: dict
) -> tuple[list[dict], dict, bytes]:
    total = int(identity["content_length"])
    tail_start = max(0, total - 65577)
    tail = get_range(session, identity, tail_start, total - 1)
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0 or eocd_at + 22 > len(tail):
        raise E9Stop("Blocked_Archive_Mapping", "ZIP EOCD not found")
    (
        _signature,
        disk_no,
        cd_disk,
        _disk_entries,
        total_entries,
        cd_size,
        cd_offset,
        _comment_length,
    ) = struct.unpack_from("<4s4H2IH", tail, eocd_at)
    if disk_no != 0 or cd_disk != 0:
        raise E9Stop("Blocked_Archive_Mapping", "multi-disk ZIP unsupported")
    zip64 = (
        total_entries == 0xFFFF
        or cd_size == 0xFFFFFFFF
        or cd_offset == 0xFFFFFFFF
    )
    if zip64:
        locator_absolute = tail_start + eocd_at - 20
        locator = get_range(
            session, identity, locator_absolute, locator_absolute + 19
        )
        if locator[:4] != b"PK\x06\x07":
            raise E9Stop(
                "Blocked_Archive_Mapping", "ZIP64 locator not found"
            )
        _sig, _disk, zip64_offset, _disks = struct.unpack(
            "<4sIQI", locator
        )
        header = get_range(
            session, identity, zip64_offset, zip64_offset + 55
        )
        if header[:4] != b"PK\x06\x06":
            raise E9Stop(
                "Blocked_Archive_Mapping", "ZIP64 EOCD not found"
            )
        (
            _sig,
            _size,
            _made,
            _needed,
            zip_disk,
            zip_cd_disk,
            _disk_entries,
            total_entries,
            cd_size,
            cd_offset,
        ) = struct.unpack("<4sQ2H2I4Q", header)
        if zip_disk != 0 or zip_cd_disk != 0:
            raise E9Stop(
                "Blocked_Archive_Mapping", "multi-disk ZIP64 unsupported"
            )
    central = get_range(
        session,
        identity,
        int(cd_offset),
        int(cd_offset + cd_size - 1),
    )
    try:
        members = parse_central_directory(central, int(total_entries))
    except (ValueError, UnicodeError, struct.error) as exc:
        raise E9Stop("Blocked_Archive_Mapping", str(exc)) from exc
    return members, {
        "zip64": zip64,
        "entry_count": len(members),
        "central_directory_offset": int(cd_offset),
        "central_directory_size": int(cd_size),
        "central_directory_bytes_sha256": sha256_bytes(central),
    }, tail


def probe_archive(
    archive_name: str, url: str
) -> tuple[dict, list[dict]]:
    session = requests.Session()
    session.headers["User-Agent"] = "LightRecon3D-E9-stable-acquisition/1.0"
    total, response, zero_byte = range_0_total(session, url)
    metadata = response_metadata(response)
    response.close()
    identity_for_ranges = {"requested_url": url, "content_length": total}
    head_end = min(total - 1, 65535)
    head = get_range(session, identity_for_ranges, 0, head_end)
    members, zip_receipt, tail = remote_zip_index_e9(
        session, identity_for_ranges
    )
    tail_start = max(0, total - 65577)
    canonical = canonical_members(archive_name, members)
    etag = metadata["etag"]
    strong_etag = (
        etag if etag and not etag.strip().lower().startswith("w/") else None
    )
    weak_etag = (
        etag if etag and etag.strip().lower().startswith("w/") else None
    )
    receipt = {
        "archive_name": archive_name,
        "requested_url": url,
        "final_url": metadata["final_url"],
        "total_length": total,
        "strong_etag": strong_etag,
        "weak_etag": weak_etag,
        "last_modified": metadata["last_modified"],
        "content_length_header_present": (
            metadata["content_length_header"] is not None
        ),
        "content_type": metadata["content_type"],
        "server": metadata["server"],
        "date": metadata["date"],
        "zero_range_sha256": sha256_bytes(zero_byte),
        "fixed_head_range": [0, head_end],
        "fixed_head_range_sha256": sha256_bytes(head),
        "fixed_tail_range": [tail_start, total - 1],
        "fixed_tail_range_sha256": sha256_bytes(tail),
        "central_directory_canonical_sha256": sha256_bytes(
            canonical_json(canonical)
        ),
        "central_directory_entry_count": len(canonical),
        "zip64": zip_receipt["zip64"],
        "probe_utc": utc_now(),
        "status": "completed",
    }
    return receipt, members


def hard_identity(row: dict) -> dict:
    return {
        key: row.get(key)
        for key in (
            "final_url",
            "total_length",
            "strong_etag",
            "fixed_head_range_sha256",
            "fixed_tail_range_sha256",
            "central_directory_canonical_sha256",
        )
    }


def ensure_probe_stability(
    first: dict[str, dict], second: dict[str, dict]
) -> None:
    errors = []
    for name in sorted(first):
        if hard_identity(first[name]) != hard_identity(second[name]):
            errors.append({
                "archive_name": name,
                "probe_1": hard_identity(first[name]),
                "probe_2": hard_identity(second[name]),
            })
    if errors:
        raise E9Stop(
            "Blocked_Remote_Object_Unstable",
            f"{len(errors)} remote objects failed dual-probe hard identity",
        )


def validate_current_probe(name: str, frozen: dict) -> None:
    current, _members = probe_archive(name, frozen["requested_url"])
    if hard_identity(current) != hard_identity(frozen):
        raise E9Stop(
            "Blocked_Remote_Object_Unstable",
            f"remote hard identity changed before/resuming {name}",
        )


def download_one(
    name: str,
    frozen: dict,
    part: Path,
    final: Path,
    quarantine: Path,
    log,
) -> dict:
    total = int(frozen["total_length"])
    if final.exists():
        if final.stat().st_size != total:
            raise E9Stop("Invalid", f"existing final size mismatch: {final}")
        return {
            "status": "reused_complete",
            "final_path": str(final),
            "bytes": total,
            "interruptions": 0,
        }
    interruptions = 0
    for connection_attempt in range(1, 7):
        validate_current_probe(name, frozen)
        existing = part.stat().st_size if part.exists() else 0
        if existing > total:
            raise E9Stop("Invalid", f"oversized partial file: {part}")
        headers = (
            {"Range": f"bytes={existing}-{total - 1}"} if existing else {}
        )
        session = requests.Session()
        session.headers["User-Agent"] = (
            "LightRecon3D-E9-stable-acquisition/1.0"
        )
        try:
            response = session.get(
                frozen["requested_url"],
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 180),
            )
        except requests.RequestException as exc:
            interruptions += 1
            if connection_attempt == 6:
                raise E9Stop(
                    "Paused_Resumable",
                    f"download connection repeatedly failed for {name}: {exc}",
                )
            continue
        if existing:
            expected_start = f"bytes {existing}-"
            content_range = response.headers.get("Content-Range", "")
            if response.status_code == 200:
                response.close()
                quarantine.mkdir(parents=True, exist_ok=True)
                quarantined = (
                    quarantine
                    / f"{part.name}.{int(time.time())}.range_ignored"
                )
                os.replace(part, quarantined)
                log(
                    f"partial_quarantined archive={name} "
                    f"path={quarantined}"
                )
                continue
            if (
                response.status_code != 206
                or not content_range.startswith(expected_start)
                or not content_range.endswith(f"/{total}")
            ):
                response.close()
                raise E9Stop(
                    "Blocked_Remote_Object_Unstable",
                    f"invalid resume response for {name}",
                )
        elif response.status_code != 200:
            response.close()
            interruptions += 1
            continue
        part.parent.mkdir(parents=True, exist_ok=True)
        written = existing
        last_report = time.monotonic()
        try:
            with part.open("ab" if existing else "wb") as handle:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                    if shutil.disk_usage("D:/").free < 40 * GIB:
                        raise E9Stop(
                            "Paused_Low_Disk",
                            "D free space fell below 40 GiB during download",
                        )
                    if time.monotonic() - last_report >= 30:
                        log(
                            f"download_progress archive={name} "
                            f"bytes={written}/{total}"
                        )
                        last_report = time.monotonic()
        except E9Stop:
            response.close()
            raise
        except (requests.RequestException, OSError) as exc:
            response.close()
            interruptions += 1
            log(
                f"download_interrupted archive={name} "
                f"bytes={written}/{total} reason={exc}"
            )
            if connection_attempt == 6:
                raise E9Stop(
                    "Paused_Resumable",
                    f"download repeatedly interrupted for {name}",
                )
            continue
        response.close()
        if written == total:
            os.replace(part, final)
            return {
                "status": "downloaded",
                "final_path": str(final),
                "bytes": written,
                "interruptions": interruptions,
                "completed_utc": utc_now(),
            }
    raise E9Stop("Paused_Resumable", f"download retries exhausted: {name}")


def local_canonical_hash(name: str, path: Path) -> str:
    return sha256_bytes(
        canonical_json(canonical_members(name, local_member_listing(path)))
    )


def freeze_input_preparation(root: Path) -> tuple[dict, str, bool]:
    source_names = [
        "execute_stable_official_archive_acquisition_e9.py",
        "execute_official_archive_acquisition_e8.py",
        "run_plane_dust3r_same_input_smoke.py",
        "export_stage3_scene_plane_fusion.py",
        "train_stage1_plane_masks.py",
        "dataloaders/s3d_dataset.py",
    ]
    sources = []
    missing_sources = []
    for name in source_names:
        path = root / name
        if path.is_file():
            sources.append({
                "path": name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
        else:
            missing_sources.append(name)
    weight_names = [
        (
            "DUSt3R",
            "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        ),
        (
            "Stage1",
            "local_outputs/stage1_large_80m_train2048_sharded_v1/best.pt",
        ),
    ]
    weights = []
    missing_weights = []
    for role, name in weight_names:
        path = root / name
        if path.is_file():
            weights.append({
                "role": role,
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
        else:
            missing_weights.append({"role": role, "path": name})
    manifest = {
        "status": (
            "frozen"
            if not missing_sources and not missing_weights
            else "incomplete"
        ),
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "clean_working_tree": not bool(
            git_output(root, "status", "--porcelain")
        ),
        "sources": sources,
        "missing_sources": missing_sources,
        "weights": weights,
        "missing_weights": missing_weights,
        "scene_order": "exact clean_raw_scene_pool.json order",
        "view_selection_rule": (
            "stable lexical member-path order; no semantic, quality, or "
            "model-result selection"
        ),
        "resize_crop_rule": (
            "exact functions frozen by the hashed DUSt3R/Stage1 source files"
        ),
        "deterministic": True,
        "random_seed": 0,
        "output_schema": {
            "global_cache": (
                "aligned XYZ, RGB, confidence, alignment view index, "
                "pointmap (x,y), registry, alignment loss"
            ),
            "direct_support": (
                "schema version, source view/pixel provenance, bounded support "
                "points, colors, plane IDs and plane parameters"
            ),
        },
        "exception_handling": (
            "fail explicitly without silently substituting view or coordinate "
            "conventions"
        ),
        "materialization_authorized": False,
    }
    contract = (
        "# E9 input-preparation contract\n\n"
        "E9 freezes but does not run input preparation. Future execution must "
        "consume scenes in exactly `clean_raw_scene_pool.json` order and use "
        "the source, weights, configuration, view selection and resize/crop "
        "definitions recorded in the adjacent manifest. DUSt3R and Stage1 "
        "materialization require a new Registration ID.\n"
    )
    return (
        manifest,
        contract,
        not missing_sources and not missing_weights,
    )


class Runner:
    def __init__(self, root: Path, result: Path, d_root: Path, f_root: Path):
        self.root = root
        self.result = result
        self.d_root = d_root
        self.f_root = f_root
        self.log_path = result / "run.log"
        self.forbidden = {
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
                "L_route_call_count",
                "G_route_call_count",
                "J_route_call_count",
            )
        }

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def set_status(self, **fields: Any) -> None:
        path = self.result / "registration_status.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update(fields)
        write_json(path, value)

    def finalize(
        self, verdict: str, reason: str, **extra: Any
    ) -> None:
        if verdict not in ALLOWED_VERDICTS:
            verdict = "Invalid"
            reason = f"attempted unregistered verdict: {reason}"
        write_json(self.result / "forbidden_call_audit.json", {
            "registration_id": REGISTRATION_ID,
            "status": (
                "passed" if sum(self.forbidden.values()) == 0 else "failed"
            ),
            **self.forbidden,
            "forbidden_counter_sum": sum(self.forbidden.values()),
        })
        write_json(self.result / "summary.json", {
            "registration_id": REGISTRATION_ID,
            "registration_state": "EXECUTED",
            "execution_verdict_issued": True,
            "execution_verdict": verdict,
            "reason": reason,
            "forbidden_counter_sum": sum(self.forbidden.values()),
            **extra,
        })
        self.set_status(
            registration_state="EXECUTED",
            execution_authorized=True,
            authorization_scope="frozen_E9_protocol_only",
            protocol_modification_authorized=False,
            execution_verdict=verdict,
            execution_updated_utc=utc_now(),
        )
        self.log(f"verdict={verdict} reason={reason}")

    def run(self) -> None:
        registration = self.result / "registration.md"
        if sha256_file(registration) != PROTOCOL_SHA256:
            raise E9Stop("Invalid", "registration.md SHA-256 changed")
        self.set_status(
            registration_state="EXECUTING",
            execution_authorized=True,
            inventory_authorized=True,
            network_probe_authorized=True,
            download_authorized=True,
            extraction_authorized=True,
            authorization_scope="frozen_E9_protocol_only",
            protocol_modification_authorized=False,
            authorization_received_local_date="2026-07-29",
        )
        self.log("execution_authorized=true scope=frozen_E9_protocol_only")

        inventory_path = self.result / "pre_acquisition_inventory.json"
        existing_inventory = json.loads(
            inventory_path.read_text(encoding="utf-8")
        )
        if existing_inventory.get("status") == "completed":
            rows = existing_inventory["records"]
            disqualifying = set(
                existing_inventory["disqualifying_scene_ids"]
            )
            inventory_audit = json.loads(
                (
                    self.result / "inventory_classification_audit.json"
                ).read_text(encoding="utf-8")
            )
            self.log(
                f"inventory_reused records={len(rows)} "
                f"disqualifying={len(disqualifying)}"
            )
        else:
            rows, disqualifying, inventory_audit = build_inventory(self.root)
            write_json(inventory_path, {
                "registration_id": REGISTRATION_ID,
                "status": "completed",
                "scanner_regex": r"scene_(\d{5})(?!\d)",
                "record_count": len(rows),
                "disqualifying_scene_count": len(disqualifying),
                "disqualifying_scene_ids": sorted(disqualifying),
                "records": rows,
                "completed_utc": utc_now(),
            })
            write_json(
                self.result / "inventory_classification_audit.json",
                inventory_audit,
            )
            write_csv(
                self.result / "pre_acquisition_inventory.csv",
                rows,
                [
                    "scene_id",
                    "source_path",
                    "source_root_type",
                    "observation",
                    "classification",
                    "disqualifying",
                    "reason",
                    "first_observed_by_E9_utc",
                    "status",
                ],
            )
            self.log(
                f"inventory_completed records={len(rows)} "
                f"disqualifying={len(disqualifying)}"
            )
        inventory_hash = sha256_file(inventory_path)
        atomic_write(
            self.result / "pre_acquisition_inventory.sha256",
            f"{inventory_hash}  pre_acquisition_inventory.json\n".encode(
                "ascii"
            ),
        )

        text_path = self.result / "official_text_manifest.json"
        existing_text = json.loads(text_path.read_text(encoding="utf-8"))
        text_session = requests.Session()
        text_session.headers["User-Agent"] = (
            "LightRecon3D-E9-stable-acquisition/1.0"
        )
        if existing_text.get("status") == "frozen":
            readme = (self.result / "official_readme.bin").read_bytes()
            errata = (self.result / "official_errata.bin").read_bytes()
            text_manifest = existing_text
            for role, url, prior in (
                ("readme", README_URL, text_manifest["readme"]),
                ("errata", ERRATA_URL, text_manifest["errata"]),
            ):
                response = get_text_response(
                    text_session, f"{role} content recheck", url
                )
                if (
                    response.url != prior["final_url"]
                    or sha256_bytes(response.content)
                    != prior["response_body_sha256"]
                ):
                    raise E9Stop(
                        "Blocked_Official_Text_Content_Changed",
                        f"{role} final URL or response body changed",
                    )
            self.log("official_text_manifest_reused_and_verified")
        else:
            readme, readme_manifest = fetch_text(
                text_session,
                "readme",
                README_URL,
                self.result / "official_readme.bin",
            )
            errata, errata_manifest = fetch_text(
                text_session,
                "errata",
                ERRATA_URL,
                self.result / "official_errata.bin",
            )
            e8_readme = (
                self.root
                / "results/official_archive_acquisition_e8_20260728_v1/"
                "official_readme_source.json"
            )
            e8_errata = (
                self.root
                / "results/official_archive_acquisition_e8_20260728_v1/"
                "official_errata_source.json"
            )
            readme_manifest["content_equal_to_E8"] = (
                e8_readme.is_file()
                and json.loads(e8_readme.read_text(encoding="utf-8")).get(
                    "sha256"
                )
                == readme_manifest["response_body_sha256"]
            )
            errata_manifest["content_equal_to_E8"] = (
                e8_errata.is_file()
                and json.loads(e8_errata.read_text(encoding="utf-8")).get(
                    "sha256"
                )
                == errata_manifest["response_body_sha256"]
            )
            text_manifest = {
                "status": "frozen",
                "readme": readme_manifest,
                "errata": errata_manifest,
                "frozen_utc": utc_now(),
            }
            write_json(text_path, text_manifest)
            self.log("official_text_content_frozen")

        errata_ids = set(
            scene_ids(errata.decode("utf-8", errors="replace"))
        )
        candidate_path = self.result / "candidate_order_128.json"
        existing_candidates = json.loads(
            candidate_path.read_text(encoding="utf-8")
        )
        if existing_candidates.get("status") == "frozen":
            candidates = existing_candidates["candidates"]
            universe = json.loads(
                (self.result / "candidate_universe.json").read_text(
                    encoding="utf-8"
                )
            )["scene_ids"]
            self.log(f"candidates_reused count={len(candidates)}")
        else:
            universe = [
                f"scene_{value:05d}"
                for value in range(200, 3000)
                if f"scene_{value:05d}" not in disqualifying
                and f"scene_{value:05d}" not in errata_ids
            ]
            if len(universe) < 128:
                raise E9Stop(
                    "Blocked_Insufficient_Candidate_Universe",
                    f"candidate universe contains {len(universe)} scenes",
                )
            ranked = sorted(
                (
                    hashlib.sha256(
                        f"{SALT}|{scene}".encode("utf-8")
                    ).hexdigest(),
                    scene,
                )
                for scene in universe
            )
            candidates = [
                {"rank": rank, "scene_id": scene, "candidate_hash": digest}
                for rank, (digest, scene) in enumerate(ranked[:128], 1)
            ]
            write_json(self.result / "candidate_universe.json", {
                "status": "generated",
                "scene_ids": universe,
                "count": len(universe),
            })
            write_json(candidate_path, {
                "status": "frozen",
                "salt": SALT,
                "count": len(candidates),
                "candidates": candidates,
            })
            write_json(
                self.result / "candidate_generation_audit.json",
                {
                    "status": "passed",
                    "inventory_sha256": inventory_hash,
                    "errata_response_body_sha256": text_manifest["errata"][
                        "response_body_sha256"
                    ],
                    "salt": SALT,
                    "candidate_universe_count": len(universe),
                    "implementation_source_sha256": sha256_file(
                        Path(__file__)
                    ),
                    "read_E7_candidate_file": False,
                    "read_E8_candidate_file_as_input": False,
                    "generated_utc": utc_now(),
                },
            )
            self.log(f"candidates_frozen count={len(candidates)}")

        # Mandatory post-candidate text content recheck.
        for role, url, prior in (
            ("readme", README_URL, text_manifest["readme"]),
            ("errata", ERRATA_URL, text_manifest["errata"]),
        ):
            response = get_text_response(
                text_session, f"{role} post-candidate recheck", url
            )
            if (
                response.url != prior["final_url"]
                or sha256_bytes(response.content)
                != prior["response_body_sha256"]
            ):
                raise E9Stop(
                    "Blocked_Official_Text_Content_Changed",
                    f"{role} content changed after candidate freeze",
                )
        text_manifest["post_candidate_recheck"] = {
            "status": "passed",
            "checked_utc": utc_now(),
            "optional_header_changes_do_not_block": True,
        }
        write_json(text_path, text_manifest)

        archive_names = list(PERSPECTIVE_NAMES) + [ANNOTATION_NAME]
        probe_results: list[dict[str, dict]] = []
        probe_members: dict[str, list[dict]] = {}
        for probe_number in (1, 2):
            probe_path = (
                self.result / f"remote_object_probe_{probe_number}.json"
            )
            member_dir = (
                self.result / f"remote_member_indexes_probe_{probe_number}"
            )
            member_dir.mkdir(exist_ok=True)
            existing_probe = json.loads(
                probe_path.read_text(encoding="utf-8")
            )
            rows_by_name = {
                row["archive_name"]: row
                for row in existing_probe.get("objects", [])
            }
            for name in archive_names:
                member_path = member_dir / f"{name}.members.json"
                if name in rows_by_name and member_path.is_file():
                    members = json.loads(
                        member_path.read_text(encoding="utf-8")
                    )["members"]
                    probe_members[name] = members
                    self.log(
                        f"remote_probe_checkpoint_reused "
                        f"probe={probe_number} archive={name}"
                    )
                else:
                    receipt, members = probe_archive(
                        name, f"{BASE_URL}/{name}"
                    )
                    rows_by_name[name] = receipt
                    probe_members[name] = members
                    write_json(member_path, {
                        "status": "frozen_probe_member_index",
                        "probe_number": probe_number,
                        "archive_name": name,
                        "canonical_listing_sha256": receipt[
                            "central_directory_canonical_sha256"
                        ],
                        "members": members,
                    })
                    write_json(probe_path, {
                        "status": "in_progress",
                        "probe_number": probe_number,
                        "objects": [
                            rows_by_name[item]
                            for item in archive_names
                            if item in rows_by_name
                        ],
                    })
                    self.log(
                        f"remote_probe probe={probe_number} archive={name} "
                        f"bytes={receipt['total_length']} "
                        f"entries={receipt['central_directory_entry_count']}"
                    )
            write_json(probe_path, {
                "status": "completed",
                "probe_number": probe_number,
                "objects": [
                    rows_by_name[name] for name in archive_names
                ],
            })
            probe_results.append(rows_by_name)
        ensure_probe_stability(probe_results[0], probe_results[1])
        frozen_remote = probe_results[1]
        write_json(self.result / "remote_object_manifest.json", {
            "status": "frozen",
            "dual_probe_hard_identity_match": True,
            "objects": [
                frozen_remote[name] for name in archive_names
            ],
            "frozen_utc": utc_now(),
        })
        self.log("dual_remote_probe_stability_passed")

        scene_archives: dict[str, set[str]] = defaultdict(set)
        member_archives: dict[str, set[str]] = defaultdict(set)
        scene_index: dict[str, dict] = {}
        for name in archive_names:
            scenes: set[str] = set()
            for member in probe_members[name]:
                normalized = member["path"].replace("\\", "/")
                member_archives[normalized].add(name)
                for scene in scene_ids(normalized):
                    scenes.add(scene)
                    if name != ANNOTATION_NAME:
                        scene_archives[scene].add(name)
            scene_index[name] = {
                "scene_count": len(scenes),
                "scene_ids": sorted(scenes),
                "member_count": len(probe_members[name]),
                "canonical_listing_sha256": frozen_remote[name][
                    "central_directory_canonical_sha256"
                ],
            }
        write_json(self.result / "archive_scene_index.json", {
            "status": "frozen",
            "archives": scene_index,
        })
        formal = set(FORMAL_PERSPECTIVE)
        boundary = set(BOUNDARY_PERSPECTIVE)
        candidate_map: dict[str, str] = {}
        mapping_errors: list[dict] = []
        for candidate in candidates:
            scene = candidate["scene_id"]
            owners = scene_archives.get(scene, set())
            formal_owners = owners & formal
            boundary_owners = owners & boundary
            if len(formal_owners) != 1 or boundary_owners:
                mapping_errors.append({
                    "scene_id": scene,
                    "archives": sorted(owners),
                    "reason": "not uniquely present in formal 01-14 only",
                })
            else:
                candidate_map[scene] = next(iter(formal_owners))
        duplicates = [
            path
            for path, owners in member_archives.items()
            if len(owners) > 1
        ]
        if duplicates:
            mapping_errors.append({
                "reason": "member path conflict across archives",
                "count": len(duplicates),
                "examples": sorted(duplicates)[:20],
            })
        write_json(self.result / "scene_to_archive_map.json", {
            "status": "passed" if not mapping_errors else "failed",
            "mapping": candidate_map,
            "errors": mapping_errors,
            "formula_mapping_used": False,
        })
        if mapping_errors:
            raise E9Stop(
                "Blocked_Archive_Mapping",
                f"archive mapping produced {len(mapping_errors)} errors",
            )

        selected = set(candidate_map)
        required_names = list(FORMAL_PERSPECTIVE) + [ANNOTATION_NAME]
        total_archive_bytes = sum(
            frozen_remote[name]["total_length"] for name in required_names
        )
        raw_bytes = sum(
            member["uncompressed_size"]
            for name in FORMAL_PERSPECTIVE
            for member in probe_members[name]
            if selected.intersection(scene_ids(member["path"]))
            and not member["is_directory"]
        )
        annotation_bytes = sum(
            member["uncompressed_size"]
            for member in probe_members[ANNOTATION_NAME]
            if selected.intersection(scene_ids(member["path"]))
            and not member["is_directory"]
        )
        d_free = shutil.disk_usage("D:/").free
        e_free = shutil.disk_usage(str(self.root)).free
        f_free = shutil.disk_usage("F:/").free
        disk = {
            "status": "passed",
            "measured_utc": utc_now(),
            "total_archive_download_bytes": total_archive_bytes,
            "predicted_raw_rgb_uncompressed_bytes": raw_bytes,
            "predicted_annotation_uncompressed_bytes": annotation_bytes,
            "predicted_total_extraction_bytes": raw_bytes + annotation_bytes,
            "D_free_bytes": d_free,
            "D_required_bytes": total_archive_bytes + 40 * GIB,
            "F_free_bytes": f_free,
            "F_required_bytes": raw_bytes + annotation_bytes + 64 * GIB,
            "E_free_bytes": e_free,
            "E_required_bytes": 2 * GIB,
        }
        if e_free < 2 * GIB:
            disk["status"] = "blocked"
            write_json(self.result / "disk_preflight.json", disk)
            raise E9Stop(
                "Blocked_Insufficient_E_Space", "E disk gate failed"
            )
        if d_free < total_archive_bytes + 40 * GIB:
            disk["status"] = "blocked"
            write_json(self.result / "disk_preflight.json", disk)
            raise E9Stop(
                "Blocked_Insufficient_D_Space", "D disk gate failed"
            )
        if f_free < raw_bytes + annotation_bytes + 64 * GIB:
            disk["status"] = "blocked"
            write_json(self.result / "disk_preflight.json", disk)
            raise E9Stop(
                "Blocked_Insufficient_F_Space", "F disk gate failed"
            )
        write_json(self.result / "disk_preflight.json", disk)
        self.log(
            f"disk_preflight_passed D={d_free} E={e_free} F={f_free}"
        )

        for path in (
            self.d_root / "perspective_empty",
            self.d_root / "annotations",
            self.d_root / "partial",
            self.d_root / "quarantine",
            self.f_root / "raw_rgb",
            self.f_root / "sealed_annotations",
            self.f_root / "extraction_staging",
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.set_status(D_or_F_directories_created=True)

        download_path = self.result / "archive_download_receipts.json"
        existing_downloads = json.loads(
            download_path.read_text(encoding="utf-8")
        )
        downloads = (
            existing_downloads.get("archives", {})
            if existing_downloads.get("status") in {"in_progress", "completed"}
            else {}
        )
        hash_path = self.result / "archive_sha256_manifest.json"
        existing_hashes = json.loads(hash_path.read_text(encoding="utf-8"))
        archive_hashes = (
            existing_hashes.get("archives", {})
            if existing_hashes.get("status") in {"in_progress", "completed"}
            else {}
        )
        for name in required_names:
            if shutil.disk_usage("D:/").free < 40 * GIB:
                raise E9Stop(
                    "Paused_Low_Disk",
                    "D free space below 40 GiB before new archive",
                )
            final = (
                self.d_root / "annotations" / name
                if name == ANNOTATION_NAME
                else self.d_root / "perspective_empty" / name
            )
            part = self.d_root / "partial" / f"{name}.part"
            downloads[name] = download_one(
                name,
                frozen_remote[name],
                part,
                final,
                self.d_root / "quarantine",
                self.log,
            )
            digest = sha256_file(final)
            with zipfile.ZipFile(final) as archive:
                bad_member = archive.testzip()
            if bad_member is not None:
                raise E9Stop(
                    "Invalid", f"ZIP CRC test failed for {name}: {bad_member}"
                )
            local_cd = local_canonical_hash(name, final)
            if local_cd != frozen_remote[name][
                "central_directory_canonical_sha256"
            ]:
                raise E9Stop(
                    "Invalid",
                    f"local central-directory hash mismatch for {name}",
                )
            if final.stat().st_size != frozen_remote[name]["total_length"]:
                raise E9Stop("Invalid", f"final size mismatch for {name}")
            os.chmod(final, stat.S_IREAD)
            archive_hashes[name] = {
                "path": str(final),
                "whole_file_sha256": digest,
                "file_byte_count": final.stat().st_size,
                "local_central_directory_sha256": local_cd,
                "remote_central_directory_sha256": frozen_remote[name][
                    "central_directory_canonical_sha256"
                ],
                "zip_test_passed": True,
                "read_only": not bool(final.stat().st_mode & stat.S_IWRITE),
            }
            downloads[name].update({
                "status": "complete_verified",
                "whole_file_sha256": digest,
            })
            write_json(download_path, {
                "status": "in_progress",
                "archives": downloads,
            })
            write_json(hash_path, {
                "status": "in_progress",
                "archives": archive_hashes,
            })
            self.log(f"archive_complete archive={name} sha256={digest}")
        write_json(download_path, {
            "status": "completed",
            "archives": downloads,
        })
        write_json(hash_path, {
            "status": "completed",
            "archives": archive_hashes,
        })

        raw_path = self.result / "raw_scene_manifest.json"
        annotation_path = self.result / "sealed_annotation_manifest.json"
        provenance_path = self.result / "per_scene_provenance.json"
        existing_raw = json.loads(raw_path.read_text(encoding="utf-8"))
        existing_annotations = json.loads(
            annotation_path.read_text(encoding="utf-8")
        )
        existing_provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        raw_manifest = existing_raw.get("scenes", {})
        annotation_manifest = existing_annotations.get("scenes", {})
        provenance = existing_provenance.get("scenes", [])
        completed_scenes = {
            row["scene_id"] for row in provenance
            if row.get("clean_raw_status") == "Clean_Raw"
        }
        handles = {
            name: zipfile.ZipFile(Path(archive_hashes[name]["path"]))
            for name in required_names
        }
        try:
            infos: dict[str, dict[str, list[zipfile.ZipInfo]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for name, archive in handles.items():
                for info in archive.infolist():
                    for scene in scene_ids(info.filename):
                        if scene in selected:
                            infos[scene][name].append(info)
            for candidate in candidates:
                scene = candidate["scene_id"]
                if scene in completed_scenes:
                    continue
                if shutil.disk_usage("F:/").free < 64 * GIB:
                    raise E9Stop(
                        "Paused_Low_Disk",
                        "F free space below 64 GiB before new scene",
                    )
                archive_name = candidate_map[scene]
                staging_root = self.f_root / "extraction_staging"
                stage = staging_root / f"{scene}.tmp"
                remove_incomplete_staging(stage, staging_root)
                raw_target = self.f_root / "raw_rgb" / scene
                ann_target = self.f_root / "sealed_annotations" / scene
                if raw_target.exists() or ann_target.exists():
                    raise E9Stop(
                        "Invalid",
                        f"untracked existing extraction target for {scene}",
                    )
                stage.mkdir(parents=True)
                started = utc_now()
                try:
                    raw_rows = extract_members(
                        handles[archive_name],
                        infos[scene].get(archive_name, []),
                        scene,
                        stage / "raw_rgb",
                        verify_images=True,
                    )
                    annotation_rows = extract_members(
                        handles[ANNOTATION_NAME],
                        infos[scene].get(ANNOTATION_NAME, []),
                        scene,
                        stage / "sealed_annotations",
                        verify_images=False,
                    )
                    if not raw_rows or not annotation_rows:
                        raise ValueError(
                            "required RGB or annotation members missing"
                        )
                    os.replace(stage / "raw_rgb", raw_target)
                    os.replace(stage / "sealed_annotations", ann_target)
                    stage.rmdir()
                    raw_tree = tree_sha256(raw_target)
                    annotation_tree = tree_sha256(ann_target)
                    completed = utc_now()
                    raw_manifest[scene] = {
                        "status": "complete",
                        "root": str(raw_target),
                        "member_count": len(raw_rows),
                        "tree_sha256": raw_tree,
                        "members": raw_rows,
                    }
                    annotation_manifest[scene] = {
                        "status": "sealed",
                        "root": str(ann_target),
                        "member_count": len(annotation_rows),
                        "tree_sha256": annotation_tree,
                        "members": annotation_rows,
                        "GT_semantic_parse_count": 0,
                    }
                    all_rows = raw_rows + annotation_rows
                    provenance.append({
                        "scene_id": scene,
                        "candidate_rank": candidate["rank"],
                        "candidate_hash": candidate["candidate_hash"],
                        "source_archive_name": archive_name,
                        "source_archive_URL": frozen_remote[archive_name][
                            "requested_url"
                        ],
                        "source_archive_SHA256": archive_hashes[archive_name][
                            "whole_file_sha256"
                        ],
                        "source_member_listing": [
                            row["member_path"] for row in all_rows
                        ],
                        "member_CRC32": [
                            row["crc32"] for row in all_rows
                        ],
                        "compressed_bytes": sum(
                            row["compressed_bytes"] for row in all_rows
                        ),
                        "uncompressed_bytes": sum(
                            row["uncompressed_bytes"] for row in all_rows
                        ),
                        "first_extraction_start_utc": started,
                        "first_complete_utc": completed,
                        "raw_rgb_tree_SHA256": raw_tree,
                        "sealed_annotation_tree_SHA256": annotation_tree,
                        "pre_acquisition_inventory_SHA256": inventory_hash,
                        "remote_object_manifest_SHA256": sha256_file(
                            self.result / "remote_object_manifest.json"
                        ),
                        "archive_scene_index_SHA256": sha256_file(
                            self.result / "archive_scene_index.json"
                        ),
                        "not_in_disqualifying_inventory": (
                            scene not in disqualifying
                        ),
                        "not_in_errata": scene not in errata_ids,
                        "GT_semantics_read": False,
                        "clean_raw_status": "Clean_Raw",
                    })
                    write_json(raw_path, {
                        "status": "in_progress",
                        "scenes": raw_manifest,
                    })
                    write_json(annotation_path, {
                        "status": "in_progress",
                        "GT_semantic_parse_count": 0,
                        "scenes": annotation_manifest,
                    })
                    write_json(provenance_path, {
                        "status": "in_progress",
                        "scenes": provenance,
                    })
                    self.log(
                        f"scene_extracted scene={scene} "
                        f"raw_members={len(raw_rows)} "
                        f"annotation_members={len(annotation_rows)}"
                    )
                except Exception as exc:
                    remove_incomplete_staging(stage, staging_root)
                    raw_manifest[scene] = {
                        "status": "ineligible",
                        "reason": str(exc),
                    }
                    annotation_manifest[scene] = {
                        "status": "not_complete",
                        "reason": str(exc),
                        "GT_semantic_parse_count": 0,
                    }
                    self.log(f"scene_ineligible scene={scene} reason={exc}")
        finally:
            for archive in handles.values():
                archive.close()
        write_json(raw_path, {
            "status": "completed",
            "scenes": raw_manifest,
        })
        write_json(annotation_path, {
            "status": "completed",
            "GT_semantic_parse_count": 0,
            "scenes": annotation_manifest,
        })
        write_json(provenance_path, {
            "status": "completed",
            "scenes": provenance,
        })
        write_csv(
            self.result / "per_scene_provenance.csv",
            provenance,
            [
                "scene_id",
                "candidate_rank",
                "candidate_hash",
                "source_archive_name",
                "source_archive_URL",
                "source_archive_SHA256",
                "compressed_bytes",
                "uncompressed_bytes",
                "first_extraction_start_utc",
                "first_complete_utc",
                "raw_rgb_tree_SHA256",
                "sealed_annotation_tree_SHA256",
                "pre_acquisition_inventory_SHA256",
                "remote_object_manifest_SHA256",
                "archive_scene_index_SHA256",
                "not_in_disqualifying_inventory",
                "not_in_errata",
                "GT_semantics_read",
                "clean_raw_status",
            ],
        )
        clean = [
            candidate["scene_id"]
            for candidate in candidates
            if any(
                row["scene_id"] == candidate["scene_id"]
                and row.get("clean_raw_status") == "Clean_Raw"
                for row in provenance
            )
        ]
        write_json(self.result / "clean_raw_scene_pool.json", {
            "status": "frozen",
            "qualification": "Clean_Raw",
            "count": len(clean),
            "scene_ids": clean,
            "order_rule": "frozen E9 candidate order",
        })

        manifest, contract, prep_complete = freeze_input_preparation(self.root)
        write_json(
            self.result / "input_preparation_source_manifest.json",
            manifest,
        )
        atomic_write(
            self.result / "input_preparation_contract.md",
            contract.encode("utf-8"),
        )
        if len(clean) < 77:
            raise E9Stop(
                "Blocked_Insufficient_Clean_Raw_Scenes",
                f"only {len(clean)} Clean_Raw scenes qualified",
            )
        if not prep_complete:
            raise E9Stop(
                "Blocked_Input_Preparation_Freeze",
                "required input-preparation source or weight is unavailable",
            )
        self.finalize(
            "Ready_Clean_Raw_Pool",
            f"{len(clean)} Clean Raw scenes acquired",
            clean_raw_scene_count=len(clean),
            complete_archive_count=len(archive_hashes),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--result-dir",
        default="results/stable_official_archive_acquisition_e9_20260729_v1",
    )
    parser.add_argument(
        "--d-root", default="D:/Structured3D_official_archives_e9"
    )
    parser.add_argument("--f-root", default="F:/Structured3D_clean_e9")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = (root / args.result_dir).resolve()
    runner = Runner(
        root,
        result,
        Path(args.d_root).resolve(),
        Path(args.f_root).resolve(),
    )
    try:
        runner.run()
    except E9Stop as stop:
        runner.finalize(stop.verdict, stop.reason)
        return 2
    except E8Stop as stop:
        runner.finalize("Paused_Resumable", str(stop))
        return 2
    except Exception as exc:
        runner.finalize("Paused_Resumable", f"unhandled interruption: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
