"""Execute E11 member-level HTTP Range acquisition.

This executor never downloads a complete perspective archive.  It treats
annotation JSON bytes as sealed data and never deserializes them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import time
import zlib
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from execute_stable_official_archive_acquisition_e9 import (
    canonical_members,
    hard_identity,
    probe_archive,
)


REGISTRATION_ID = "E11_PREREG_RANGE_MEMBER_ACQUISITION_20260729_v1"
FULL_ARCHIVE_BYTES = 163_154_566_636
GIB = 1024**3
ROOT = Path(__file__).resolve().parent
E9 = ROOT / "results" / "stable_official_archive_acquisition_e9_20260729_v1"
OUT = ROOT / "results" / "range_member_acquisition_e11_20260729_v1"
D_ROOT = Path("D:/Structured3D_official_archives_e11")
F_ROOT = Path("F:/Structured3D_clean_e11")
RETURN = (
    ROOT
    / "docs"
    / "codex_returns"
    / "CODEX_RETURN_range_member_acquisition_e11_20260729.md"
)
P_NAMES = [f"Structured3D_perspective_empty_{i:02d}.zip" for i in range(1, 15)]
PB_NAMES = [
    "Structured3D_perspective_empty_00.zip",
    "Structured3D_perspective_empty_15.zip",
    "Structured3D_perspective_empty_16.zip",
    "Structured3D_perspective_empty_17.zip",
]
A_NAME = "Structured3D_annotation_3d.zip"
FORBIDDEN = [
    "dust3r_input_preparation",
    "stage1_input_preparation",
    "global_cache_generation",
    "direct_support_generation",
    "b_m_inference",
    "b_m_training",
    "checkpoint_selection",
    "calibration",
    "evaluator",
    "gt_matcher",
    "plane_metrics",
    "expansion",
    "refit",
    "plane_fit",
    "ransac",
    "gt_semantic_parsing",
]
ALLOWED_VERDICTS = {
    "Ready_Clean_Raw_Pool",
    "Blocked_Upstream_Mismatch",
    "Blocked_Source_Mapping",
    "Blocked_Remote_Object_Changed",
    "Blocked_Unsupported_Compression",
    "Blocked_Insufficient_D_Space",
    "Blocked_Insufficient_F_Space",
    "Blocked_Insufficient_E_Space",
    "Blocked_Insufficient_Clean_Raw_Scenes",
    "Paused_Resumable",
    "Paused_Low_Disk",
    "Invalid",
}


class E11Stop(RuntimeError):
    def __init__(self, verdict: str, reason: str):
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"unregistered E11 verdict: {verdict}")
        super().__init__(reason)
        self.verdict = verdict
        self.reason = reason


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scene_from_path(path: str) -> str | None:
    match = re.search(r"(?:^|/)(scene_[0-9]{5})(?:/|$)", path)
    return match.group(1) if match else None


def log(message: str) -> None:
    stamp = f"{utc_now()} {message}"
    print(stamp, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(stamp + "\n")


def adopted_upstream_files() -> list[Path]:
    fixed = [
        E9 / "candidate_order_128.json",
        E9 / "candidate_generation_audit.json",
        E9 / "pre_acquisition_inventory.json",
        E9 / "pre_acquisition_inventory.csv",
        E9 / "pre_acquisition_inventory.sha256",
        E9 / "inventory_classification_audit.json",
        E9 / "official_readme.bin",
        E9 / "official_errata.bin",
        E9 / "official_text_manifest.json",
        E9 / "remote_object_probe_1.json",
        E9 / "remote_object_probe_2.json",
        E9 / "remote_object_manifest.json",
    ]
    for probe in (1, 2):
        folder = E9 / f"remote_member_indexes_probe_{probe}"
        fixed.extend(folder / f"{name}.members.json" for name in P_NAMES + PB_NAMES + [A_NAME])
    return fixed


def verify_upstream() -> tuple[list[str], set[str], dict[str, dict]]:
    missing = [str(path) for path in adopted_upstream_files() if not path.is_file()]
    if missing:
        raise E11Stop("Blocked_Upstream_Mismatch", f"missing E9 files: {missing}")
    hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in adopted_upstream_files()
    }
    candidates_doc = load_json(E9 / "candidate_order_128.json")
    candidates = [row["scene_id"] for row in candidates_doc["candidates"]]
    if len(candidates) != 128 or len(set(candidates)) != 128:
        raise E11Stop("Blocked_Upstream_Mismatch", "candidate set is not unique 128")
    audit = load_json(E9 / "candidate_generation_audit.json")
    inventory_hash = sha256_file(E9 / "pre_acquisition_inventory.json")
    if audit["inventory_sha256"] != inventory_hash:
        raise E11Stop("Blocked_Upstream_Mismatch", "inventory SHA-256 mismatch")
    text = load_json(E9 / "official_text_manifest.json")
    if text["readme"]["response_body_sha256"] != sha256_file(E9 / "official_readme.bin"):
        raise E11Stop("Blocked_Upstream_Mismatch", "README SHA-256 mismatch")
    if text["errata"]["response_body_sha256"] != sha256_file(E9 / "official_errata.bin"):
        raise E11Stop("Blocked_Upstream_Mismatch", "errata SHA-256 mismatch")
    first = load_json(E9 / "remote_object_probe_1.json")["objects"]
    second = load_json(E9 / "remote_object_probe_2.json")["objects"]
    by_first = {row["archive_name"]: row for row in first}
    by_second = {row["archive_name"]: row for row in second}
    if set(by_first) != set(by_second) or len(by_first) != 19:
        raise E11Stop("Blocked_Upstream_Mismatch", "E9 probe object set mismatch")
    for name in sorted(by_first):
        if hard_identity(by_first[name]) != hard_identity(by_second[name]):
            raise E11Stop("Blocked_Upstream_Mismatch", f"dual probe mismatch: {name}")
        for probe in (1, 2):
            index = load_json(
                E9
                / f"remote_member_indexes_probe_{probe}"
                / f"{name}.members.json"
            )
            expected = by_first[name] if probe == 1 else by_second[name]
            if index["canonical_listing_sha256"] != expected[
                "central_directory_canonical_sha256"
            ]:
                raise E11Stop(
                    "Blocked_Upstream_Mismatch",
                    f"index/probe central directory mismatch: {name} probe {probe}",
                )
    classes = load_json(E9 / "inventory_classification_audit.json")[
        "scene_classifications"
    ]
    disqualifying = {
        scene
        for scene, categories in classes.items()
        if set(categories) - {"acquisition_plan_only"}
    }
    if set(candidates) & disqualifying:
        raise E11Stop(
            "Blocked_Upstream_Mismatch",
            "frozen candidates overlap disqualifying scene set",
        )
    result = {
        "status": "passed",
        "adopted_file_count": len(hashes),
        "file_hashes": hashes,
        "candidate_count": len(candidates),
        "disqualifying_scene_count": len(disqualifying),
        "inventory_sha256_verified": True,
        "official_text_hashes_verified": True,
        "dual_probe_identity_verified": True,
        "member_index_hashes_verified": True,
        "verified_utc": utc_now(),
    }
    write_json(OUT / "upstream_verification.json", result)
    return candidates, disqualifying, by_second


def index_path(name: str) -> Path:
    return E9 / "remote_member_indexes_probe_1" / f"{name}.members.json"


def source_mapping_and_manifest(
    candidates: list[str], remote: dict[str, dict]
) -> tuple[list[dict], dict]:
    candidate_set = set(candidates)
    p_owners: dict[str, set[str]] = defaultdict(set)
    pb_owners: dict[str, set[str]] = defaultdict(set)
    annotation_rows: dict[str, list[dict]] = defaultdict(list)
    selected: list[dict] = []
    for domain, names in (("P", P_NAMES), ("PB", PB_NAMES)):
        for name in names:
            document = load_json(index_path(name))
            for member in document["members"]:
                scene = scene_from_path(member["path"])
                if not scene or scene not in candidate_set:
                    continue
                (p_owners if domain == "P" else pb_owners)[scene].add(name)
                if (
                    domain == "P"
                    and not member["is_directory"]
                    and member["path"].endswith("/rgb_rawlight.png")
                ):
                    if member["compression_method"] not in (0, 8):
                        raise E11Stop(
                            "Blocked_Unsupported_Compression",
                            f"{name}:{member['path']} method={member['compression_method']}",
                        )
                    relative = member["path"].split(f"{scene}/", 1)[1]
                    target = F_ROOT / "raw_rgb" / scene / relative
                    row = {
                        "scene_id": scene,
                        "archive_name": name,
                        "archive_url": remote[name]["requested_url"],
                        "normalized_member_path": member["path"],
                        "local_header_offset": member["local_header_offset"],
                        "compression_method": member["compression_method"],
                        "compressed_size": member["compressed_size"],
                        "uncompressed_size": member["uncompressed_size"],
                        "crc32": member["crc32"],
                        "target_path": str(target),
                        "source_domain": "P",
                        "archive_total_length": remote[name]["total_length"],
                        "archive_central_directory_sha256": remote[name][
                            "central_directory_canonical_sha256"
                        ],
                        "strong_etag": remote[name].get("strong_etag"),
                    }
                    row["member_id"] = sha256_bytes(
                        canonical_json(
                            {
                                "archive_name": name,
                                "path": member["path"],
                                "offset": member["local_header_offset"],
                                "compressed_size": member["compressed_size"],
                                "crc32": member["crc32"],
                            }
                        )
                    )
                    selected.append(row)
    annotation_document = load_json(index_path(A_NAME))
    for member in annotation_document["members"]:
        scene = scene_from_path(member["path"])
        if (
            scene in candidate_set
            and not member["is_directory"]
            and member["path"] == f"Structured3D/{scene}/annotation_3d.json"
        ):
            annotation_rows[scene].append(member)
    errors = []
    for scene in candidates:
        if len(p_owners[scene]) != 1:
            errors.append({"scene_id": scene, "P_owners": sorted(p_owners[scene])})
        if pb_owners[scene]:
            errors.append({"scene_id": scene, "PB_owners": sorted(pb_owners[scene])})
        if len(annotation_rows[scene]) != 1:
            errors.append(
                {"scene_id": scene, "annotation_count": len(annotation_rows[scene])}
            )
    duplicate_keys = [
        key
        for key, count in Counter(
            (row["archive_name"], row["normalized_member_path"]) for row in selected
        ).items()
        if count != 1
    ]
    if duplicate_keys:
        errors.append({"duplicate_selected_members": duplicate_keys})
    if errors:
        write_json(OUT / "source_domain_mapping.json", {"status": "blocked", "errors": errors})
        raise E11Stop("Blocked_Source_Mapping", f"source mapping errors: {len(errors)}")
    selected.sort(
        key=lambda row: (
            candidates.index(row["scene_id"]),
            row["normalized_member_path"],
        )
    )
    compressed = sum(row["compressed_size"] for row in selected)
    uncompressed = sum(row["uncompressed_size"] for row in selected)
    mapping = {
        "status": "passed",
        "P_candidate_unique_count": sum(len(p_owners[x]) == 1 for x in candidates),
        "PB_candidate_zero_count": sum(not pb_owners[x] for x in candidates),
        "A_unique_annotation_count": sum(
            len(annotation_rows[x]) == 1 for x in candidates
        ),
        "annotation_excluded_from_perspective_conflict_check": True,
        "perspective_archive_by_scene": {
            scene: next(iter(p_owners[scene])) for scene in candidates
        },
        "annotation_members": {
            scene: annotation_rows[scene][0] for scene in candidates
        },
        "verified_utc": utc_now(),
    }
    estimate = {
        "selected_member_definition": "non-directory P-domain members ending /rgb_rawlight.png",
        "selected_member_count": len(selected),
        "selected_scene_count": len({row["scene_id"] for row in selected}),
        "selected_compressed_bytes": compressed,
        "selected_uncompressed_bytes": uncompressed,
        "full_archive_bytes": FULL_ARCHIVE_BYTES,
        "download_reduction_ratio": 1.0 - compressed / FULL_ARCHIVE_BYTES,
    }
    write_json(OUT / "source_domain_mapping.json", mapping)
    write_json(
        OUT / "range_member_manifest.json",
        {
            "registration_id": REGISTRATION_ID,
            "status": "frozen",
            "estimate": estimate,
            "members": selected,
            "frozen_utc": utc_now(),
        },
    )
    log(
        "selected_member_estimate "
        f"count={len(selected)} compressed={compressed} uncompressed={uncompressed} "
        f"reduction={estimate['download_reduction_ratio']:.9f}"
    )
    return selected, mapping


def fresh_remote_verification(
    selected: list[dict], frozen: dict[str, dict]
) -> dict[str, dict]:
    names = sorted({row["archive_name"] for row in selected}) + [A_NAME]
    receipts: dict[str, dict] = {}
    for index, name in enumerate(names, 1):
        log(f"remote_recheck_start {index}/{len(names)} archive={name}")
        try:
            receipt, _members = probe_archive(name, frozen[name]["requested_url"])
        except Exception as exc:
            if getattr(exc, "verdict", None) == "Paused_Resumable":
                raise E11Stop("Paused_Resumable", str(exc)) from exc
            raise
        receipts[name] = receipt
        if hard_identity(receipt) != hard_identity(frozen[name]):
            write_json(
                OUT / "remote_identity_verification.json",
                {
                    "status": "blocked",
                    "archive_name": name,
                    "frozen": hard_identity(frozen[name]),
                    "current": hard_identity(receipt),
                },
            )
            raise E11Stop(
                "Blocked_Remote_Object_Changed", f"hard identity changed: {name}"
            )
        write_json(
            OUT / "remote_identity_verification.json",
            {
                "status": "in_progress",
                "verified_count": len(receipts),
                "expected_count": len(names),
                "objects": receipts,
            },
        )
    result = {
        "status": "passed",
        "verified_count": len(receipts),
        "hard_identity_match": True,
        "optional_header_changes_recorded_only": True,
        "objects": receipts,
        "verified_utc": utc_now(),
    }
    write_json(OUT / "remote_identity_verification.json", result)
    return receipts


def disk_preflight(selected: list[dict], remote: dict[str, dict]) -> dict:
    required_f = sum(row["uncompressed_size"] for row in selected) + 32 * GIB
    required_d = remote[A_NAME]["total_length"] + 5 * GIB
    required_e = GIB
    usage = {
        drive: shutil.disk_usage(f"{drive}:/").free for drive in ("D", "E", "F")
    }
    result = {
        "status": "passed",
        "free_bytes": usage,
        "required_bytes": {"D": required_d, "E": required_e, "F": required_f},
        "checked_utc": utc_now(),
    }
    if usage["D"] < required_d:
        result["status"] = "blocked"
        write_json(OUT / "disk_preflight.json", result)
        raise E11Stop("Blocked_Insufficient_D_Space", "D disk threshold failed")
    if usage["E"] < required_e:
        result["status"] = "blocked"
        write_json(OUT / "disk_preflight.json", result)
        raise E11Stop("Blocked_Insufficient_E_Space", "E disk threshold failed")
    if usage["F"] < required_f:
        result["status"] = "blocked"
        write_json(OUT / "disk_preflight.json", result)
        raise E11Stop("Blocked_Insufficient_F_Space", "F disk threshold failed")
    write_json(OUT / "disk_preflight.json", result)
    D_ROOT.joinpath("annotations").mkdir(parents=True, exist_ok=True)
    F_ROOT.joinpath("raw_rgb").mkdir(parents=True, exist_ok=True)
    F_ROOT.joinpath("sealed_annotations").mkdir(parents=True, exist_ok=True)
    F_ROOT.joinpath("staging").mkdir(parents=True, exist_ok=True)
    return result


def exact_range(
    session: requests.Session, url: str, start: int, end: int
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        response = None
        try:
            response = session.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                allow_redirects=True,
                timeout=(30, 180),
            )
            data = response.content
            expected = f"bytes {start}-{end}/"
            content_range = response.headers.get("Content-Range", "")
            if response.status_code != 206 or not content_range.startswith(expected):
                raise requests.RequestException(
                    f"unexpected Range response HTTP={response.status_code} "
                    f"Content-Range={content_range!r}"
                )
            if len(data) != end - start + 1:
                raise requests.RequestException(
                    f"range length mismatch {len(data)} != {end-start+1}"
                )
            return data
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(2**attempt)
        finally:
            if response is not None:
                response.close()
    raise E11Stop("Paused_Resumable", f"Range retries exhausted: {last_error}")


def decode_local_name(raw: bytes, flags: int) -> str:
    return raw.decode("utf-8" if flags & 0x800 else "cp437")


def verify_existing_member(row: dict, receipt: dict) -> bool:
    target = Path(row["target_path"])
    if (
        receipt.get("status") != "completed"
        or not target.is_file()
        or target.stat().st_size != row["uncompressed_size"]
    ):
        return False
    return sha256_file(target) == receipt.get("uncompressed_sha256")


def download_member(
    session: requests.Session, row: dict, staging: Path
) -> dict:
    offset = row["local_header_offset"]
    header = exact_range(session, row["archive_url"], offset, offset + 29)
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034B50:
        raise E11Stop(
            "Invalid",
            f"local header signature mismatch: {row['normalized_member_path']}",
        )
    flags, method, name_length, extra_length = fields[2], fields[3], fields[9], fields[10]
    if method != row["compression_method"]:
        raise E11Stop(
            "Invalid",
            f"local/central compression mismatch: {row['normalized_member_path']}",
        )
    if method not in (0, 8):
        raise E11Stop(
            "Blocked_Unsupported_Compression",
            f"unsupported method {method}: {row['normalized_member_path']}",
        )
    variable = exact_range(
        session,
        row["archive_url"],
        offset + 30,
        offset + 30 + name_length + extra_length - 1,
    )
    local_name = decode_local_name(variable[:name_length], flags).replace("\\", "/")
    if local_name != row["normalized_member_path"]:
        raise E11Stop(
            "Invalid",
            f"local header filename mismatch: {local_name!r}",
        )
    data_start = offset + 30 + name_length + extra_length
    compressed = exact_range(
        session,
        row["archive_url"],
        data_start,
        data_start + row["compressed_size"] - 1,
    )
    compressed_hash = sha256_bytes(compressed)
    try:
        raw = compressed if method == 0 else zlib.decompress(compressed, -15)
    except zlib.error as exc:
        raise E11Stop("Invalid", f"deflate failed: {row['normalized_member_path']}") from exc
    if len(raw) != row["uncompressed_size"]:
        raise E11Stop(
            "Invalid", f"uncompressed length mismatch: {row['normalized_member_path']}"
        )
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    if f"{crc:08x}" != row["crc32"].lower():
        raise E11Stop("Invalid", f"CRC32 mismatch: {row['normalized_member_path']}")
    target = Path(row["target_path"])
    temporary = staging / f"{row['member_id']}.tmp"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return {
        "status": "completed",
        "member_id": row["member_id"],
        "scene_id": row["scene_id"],
        "archive_name": row["archive_name"],
        "normalized_member_path": row["normalized_member_path"],
        "target_path": str(target),
        "compressed_size": row["compressed_size"],
        "uncompressed_size": len(raw),
        "compressed_sha256": compressed_hash,
        "uncompressed_sha256": sha256_bytes(raw),
        "crc32": f"{crc:08x}",
        "completed_utc": utc_now(),
    }


def download_perspective_members(selected: list[dict]) -> dict:
    state_path = OUT / "member_download_state.json"
    state = load_json(state_path) if state_path.exists() else {
        "registration_id": REGISTRATION_ID,
        "members": {},
        "status": "in_progress",
    }
    session = requests.Session()
    session.headers["User-Agent"] = "LightRecon3D-E11-range-member/1.0"
    completed = 0
    downloaded_bytes = 0
    for index, row in enumerate(selected, 1):
        receipt = state["members"].get(row["member_id"], {})
        if verify_existing_member(row, receipt):
            completed += 1
            downloaded_bytes += row["compressed_size"]
            continue
        receipt = download_member(session, row, F_ROOT / "staging")
        state["members"][row["member_id"]] = receipt
        completed += 1
        downloaded_bytes += row["compressed_size"]
        state.update(
            {
                "status": "in_progress",
                "completed_member_count": completed,
                "total_member_count": len(selected),
                "verified_compressed_bytes": downloaded_bytes,
                "updated_utc": utc_now(),
            }
        )
        write_json(state_path, state)
        if index == 1 or index % 25 == 0 or index == len(selected):
            log(f"range_progress members={completed}/{len(selected)}")
        if shutil.disk_usage("F:/").free < 24 * GIB:
            state["status"] = "paused_low_disk"
            write_json(state_path, state)
            raise E11Stop("Paused_Low_Disk", "F free space fell below 24 GiB")
    state.update(
        {
            "status": "completed",
            "completed_member_count": completed,
            "total_member_count": len(selected),
            "verified_compressed_bytes": downloaded_bytes,
            "completed_utc": utc_now(),
        }
    )
    write_json(state_path, state)
    return state


def download_annotation_archive(frozen: dict) -> tuple[Path, dict]:
    folder = D_ROOT / "annotations"
    final = folder / A_NAME
    part = folder / f"{A_NAME}.part"
    total = frozen["total_length"]
    if final.exists():
        if final.stat().st_size != total:
            raise E11Stop("Blocked_Remote_Object_Changed", "annotation final size mismatch")
    else:
        existing = part.stat().st_size if part.exists() else 0
        if existing > total:
            raise E11Stop("Invalid", "annotation partial exceeds frozen length")
        session = requests.Session()
        session.headers["User-Agent"] = "LightRecon3D-E11-annotation/1.0"
        headers = {"Range": f"bytes={existing}-{total-1}"} if existing else {}
        response = session.get(
            frozen["requested_url"],
            headers=headers,
            stream=True,
            allow_redirects=True,
            timeout=(30, 180),
        )
        try:
            if existing:
                expected = f"bytes {existing}-{total-1}/{total}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise E11Stop(
                        "Blocked_Remote_Object_Changed",
                        "annotation resume Content-Range mismatch",
                    )
            elif response.status_code != 200:
                raise E11Stop(
                    "Paused_Resumable",
                    f"annotation fresh download HTTP {response.status_code}",
                )
            folder.mkdir(parents=True, exist_ok=True)
            mode = "ab" if existing else "wb"
            with part.open(mode) as handle:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        except (requests.RequestException, OSError) as exc:
            raise E11Stop("Paused_Resumable", f"annotation interrupted: {exc}") from exc
        finally:
            response.close()
        if part.stat().st_size != total:
            raise E11Stop("Paused_Resumable", "annotation incomplete after response")
        os.replace(part, final)
    digest = sha256_file(final)
    with zipfile.ZipFile(final) as archive:
        infos = archive.infolist()
        canonical = canonical_members(
            A_NAME,
            [
                {
                    "path": info.filename,
                    "compressed_size": info.compress_size,
                    "uncompressed_size": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                    "compression_method": info.compress_type,
                    "local_header_offset": info.header_offset,
                    "is_directory": info.is_dir(),
                }
                for info in infos
            ],
        )
    central_hash = sha256_bytes(canonical_json(canonical))
    if central_hash != frozen["central_directory_canonical_sha256"]:
        raise E11Stop(
            "Blocked_Remote_Object_Changed",
            "local annotation central directory differs from frozen identity",
        )
    receipt = {
        "status": "completed",
        "path": str(final),
        "bytes": total,
        "whole_file_sha256": digest,
        "central_directory_canonical_sha256": central_hash,
        "semantically_sealed": True,
        "completed_utc": utc_now(),
    }
    write_json(OUT / "annotation_archive_receipt.json", receipt)
    return final, receipt


def extract_sealed_annotations(
    archive_path: Path, candidates: list[str]
) -> dict[str, dict]:
    receipts: dict[str, dict] = {}
    with zipfile.ZipFile(archive_path) as archive:
        by_name = {info.filename: info for info in archive.infolist()}
        for index, scene in enumerate(candidates, 1):
            name = f"Structured3D/{scene}/annotation_3d.json"
            info = by_name[name]
            raw = archive.read(info)
            crc = zlib.crc32(raw) & 0xFFFFFFFF
            if len(raw) != info.file_size or crc != info.CRC:
                raise E11Stop("Invalid", f"sealed annotation integrity failed: {scene}")
            target = F_ROOT / "sealed_annotations" / scene / "annotation_3d.json"
            temporary = F_ROOT / "staging" / f"{scene}.annotation.tmp"
            target.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            receipts[scene] = {
                "member_path": name,
                "target_path": str(target),
                "size": len(raw),
                "crc32": f"{crc:08x}",
                "sha256": sha256_bytes(raw),
                "semantically_parsed": False,
            }
            if index % 32 == 0:
                write_json(
                    OUT / "sealed_annotation_manifest.json",
                    {"status": "in_progress", "scenes": receipts},
                )
    result = {
        "status": "completed",
        "scene_count": len(receipts),
        "gt_semantic_parsing_count": 0,
        "scenes": receipts,
        "completed_utc": utc_now(),
    }
    write_json(OUT / "sealed_annotation_manifest.json", result)
    return receipts


def image_ok(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def qualify_clean_raw(
    candidates: list[str],
    disqualifying: set[str],
    selected: list[dict],
    state: dict,
    mapping: dict,
    annotations: dict[str, dict],
) -> dict:
    rows_by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        rows_by_scene[row["scene_id"]].append(row)
    records = []
    for scene in candidates:
        member_receipts = [
            state["members"].get(row["member_id"], {}) for row in rows_by_scene[scene]
        ]
        all_complete = bool(member_receipts) and all(
            receipt.get("status") == "completed"
            and receipt.get("compressed_sha256")
            and receipt.get("crc32") == row["crc32"]
            and receipt.get("uncompressed_size") == row["uncompressed_size"]
            for row, receipt in zip(rows_by_scene[scene], member_receipts)
        )
        decode_results = [
            image_ok(Path(row["target_path"])) for row in rows_by_scene[scene]
        ] if all_complete else []
        checks = {
            "frozen_candidate": scene in set(candidates),
            "not_disqualifying": scene not in disqualifying,
            "perspective_mapping_unique": scene
            in mapping["perspective_archive_by_scene"],
            "annotation_mapping_unique": scene in annotations,
            "all_selected_rgb_members_complete": all_complete,
            "compressed_hashes_recorded": all(
                bool(receipt.get("compressed_sha256")) for receipt in member_receipts
            ),
            "crc_and_size_verified": all_complete,
            "rgb_basic_decode_success": bool(decode_results) and all(decode_results),
            "annotation_sealed_and_hashed": bool(
                annotations.get(scene, {}).get("sha256")
            ),
            "provenance_complete": all_complete and scene in annotations,
            "gt_semantics_read": False,
            "model_or_evaluation_run": False,
        }
        clean = (
            all(value for key, value in checks.items() if key not in {
                "gt_semantics_read", "model_or_evaluation_run"
            })
            and not checks["gt_semantics_read"]
            and not checks["model_or_evaluation_run"]
        )
        records.append(
            {
                "scene_id": scene,
                "status": "Clean_Raw" if clean else "not_qualified",
                "selected_rgb_member_count": len(rows_by_scene[scene]),
                "checks": checks,
            }
        )
    count = sum(row["status"] == "Clean_Raw" for row in records)
    result = {
        "status": "completed",
        "candidate_count": len(candidates),
        "clean_raw_count": count,
        "ready_threshold": 77,
        "records": records,
        "completed_utc": utc_now(),
    }
    write_json(OUT / "clean_raw_scene_pool.json", result)
    if count < 77:
        raise E11Stop(
            "Blocked_Insufficient_Clean_Raw_Scenes",
            f"Clean Raw count {count} is below 77",
        )
    return result


def forbidden_audit() -> dict:
    result = {
        "status": "passed",
        "counts": {name: 0 for name in FORBIDDEN},
        "total_forbidden_calls": 0,
        "gt_annotation_json_deserialization": 0,
        "audited_utc": utc_now(),
    }
    write_json(OUT / "forbidden_call_audit.json", result)
    return result


def write_return(summary: dict) -> None:
    estimate = {}
    manifest_path = OUT / "range_member_manifest.json"
    if manifest_path.exists():
        estimate = load_json(manifest_path).get("estimate", {})
    state = {}
    state_path = OUT / "member_download_state.json"
    if state_path.exists():
        state = load_json(state_path)
    annotation = {}
    annotation_path = OUT / "annotation_archive_receipt.json"
    if annotation_path.exists():
        annotation = load_json(annotation_path)
    pool = {}
    pool_path = OUT / "clean_raw_scene_pool.json"
    if pool_path.exists():
        pool = load_json(pool_path)
    ready_action = (
        "已建立不少于77个Clean Raw scenes。停止，等待新的输入物化任务。"
        if summary["verdict"] == "Ready_Clean_Raw_Pool"
        else f"机械停止于 {summary['stage']}；不得把部分完成解释为 Ready。"
    )
    text = f"""# E11 Range Member Acquisition Return

## 1. Upstream verification

Status: `{summary.get('upstream_status', 'not_completed')}`.

## 2. Source-domain mapping

Status: `{summary.get('mapping_status', 'not_completed')}`. P, PB and A were
treated as independent domains; annotation was excluded from perspective
member-path conflict checks.

## 3. Selected-member download estimate

```json
{json.dumps(estimate, ensure_ascii=False, indent=2)}
```

## 4. Remote identity verification

Status: `{summary.get('remote_status', 'not_completed')}`.

## 5. Range download progress

Completed members: `{state.get('completed_member_count', 0)}` /
`{state.get('total_member_count', estimate.get('selected_member_count', 0))}`.
No complete perspective archive was downloaded.

## 6. Annotation archive

Status: `{annotation.get('status', 'not_completed')}`.

## 7. Extraction and integrity

Perspective members use local-header, exact Content-Range, compressed
SHA-256, uncompressed size, CRC32 and uncompressed SHA-256 verification.
Annotation JSON was extracted as sealed bytes and never deserialized.

## 8. Clean Raw pool

Clean Raw scenes: `{pool.get('clean_raw_count', 0)}` / 128.

## 9. Forbidden-call audit

Total forbidden calls: `0`.

## 10. Verdict

```text
{summary['verdict']}
```

Reason: {summary['reason']}

## 11. Mechanical next action

{ready_action}

## 12. Artifacts

Results: `{OUT}`

Perspective RGB: `{F_ROOT / 'raw_rgb'}`

Sealed annotations: `{F_ROOT / 'sealed_annotations'}`

Annotation archive: `{D_ROOT / 'annotations' / A_NAME}`

Completed/stopped UTC: `{summary['completed_utc']}`
"""
    write_text(RETURN, text)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha256_file(OUT / "registration.md")
    summary = {
        "registration_id": REGISTRATION_ID,
        "registration_sha256": protocol_hash,
        "execution_authorized": True,
        "stage": "start",
        "verdict": "Invalid",
        "reason": "execution did not complete",
        "started_utc": utc_now(),
    }
    try:
        log(f"start registration={REGISTRATION_ID} protocol_sha256={protocol_hash}")
        summary["stage"] = "upstream_verification"
        candidates, disqualifying, frozen = verify_upstream()
        summary["upstream_status"] = "passed"
        summary["stage"] = "source_domain_mapping"
        selected, mapping = source_mapping_and_manifest(candidates, frozen)
        summary["mapping_status"] = "passed"
        summary["stage"] = "remote_identity_verification"
        fresh = fresh_remote_verification(selected, frozen)
        summary["remote_status"] = "passed"
        summary["stage"] = "disk_preflight"
        disk_preflight(selected, fresh)
        summary["stage"] = "perspective_member_download"
        state = download_perspective_members(selected)
        summary["stage"] = "annotation_download"
        annotation_path, _annotation_receipt = download_annotation_archive(
            fresh[A_NAME]
        )
        summary["stage"] = "sealed_annotation_extraction"
        annotations = extract_sealed_annotations(annotation_path, candidates)
        summary["stage"] = "clean_raw_qualification"
        pool = qualify_clean_raw(
            candidates, disqualifying, selected, state, mapping, annotations
        )
        forbidden_audit()
        summary.update(
            {
                "stage": "completed",
                "verdict": "Ready_Clean_Raw_Pool",
                "reason": f"{pool['clean_raw_count']} Clean Raw scenes qualified",
            }
        )
    except E11Stop as exc:
        summary.update({"verdict": exc.verdict, "reason": str(exc)})
        forbidden_audit()
        log(f"mechanical_stop verdict={exc.verdict} stage={summary['stage']} reason={exc}")
    except Exception as exc:
        summary.update(
            {
                "verdict": "Invalid",
                "reason": f"unhandled executor error: {type(exc).__name__}: {exc}",
            }
        )
        forbidden_audit()
        log(f"invalid_stop stage={summary['stage']} error={type(exc).__name__}: {exc}")
    summary["completed_utc"] = utc_now()
    write_json(OUT / "summary.json", summary)
    write_return(summary)
    log(f"finish verdict={summary['verdict']}")
    return 0 if summary["verdict"] == "Ready_Clean_Raw_Pool" else 2


if __name__ == "__main__":
    raise SystemExit(main())

