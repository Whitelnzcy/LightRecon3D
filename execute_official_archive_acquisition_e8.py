"""Execute the frozen E8 official Structured3D archive acquisition contract.

This program is deliberately limited to inventory, immutable source/ZIP
metadata acquisition, resumable archive download, selective extraction, and
freezing (not running) the future input-preparation definition.
"""

from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import requests
from PIL import Image


REGISTRATION_ID = "E8_PREREG_official_archive_acquisition_20260728_v1"
PROTOCOL_SHA256 = "565324e8ee8ddeb5c527fe463fb5e632b6b0b242a206d431e3e01f70df561799"
SCENE_RE = re.compile(r"scene_(\d{5})(?!\d)")
SALT = "LightRecon3D_E7_clean_acquisition_20260728_v1"
BASE_URL = "https://zju-kjl-jointlab-azure.kujiale.com/Structured3D"
README_URL = f"{BASE_URL}/README.txt"
ERRATA_URL = (
    "https://raw.githubusercontent.com/bertjiazheng/Structured3D/"
    "master/metadata/errata.txt"
)
PERSPECTIVE_NAMES = tuple(
    f"Structured3D_perspective_empty_{index:02d}.zip" for index in range(18)
)
FORMAL_PERSPECTIVE = PERSPECTIVE_NAMES[1:15]
BOUNDARY_PERSPECTIVE = (
    PERSPECTIVE_NAMES[0],
    PERSPECTIVE_NAMES[15],
    PERSPECTIVE_NAMES[16],
    PERSPECTIVE_NAMES[17],
)
ANNOTATION_NAME = "Structured3D_annotation_3d.zip"
GIB = 1024**3
TEXT_SUFFIXES = {
    ".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".py", ".ps1", ".sh", ".bat", ".log",
}


class E8Stop(RuntimeError):
    def __init__(self, verdict: str, reason: str):
        super().__init__(reason)
        self.verdict = verdict
        self.reason = reason


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        (
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


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


def scene_ids(value: str) -> list[str]:
    return [f"scene_{match.group(1)}" for match in SCENE_RE.finditer(value)]


def safe_member_relative(member: str, scene_id: str) -> Path:
    normalized = member.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe ZIP member path: {member!r}")
    matches = scene_ids(normalized)
    if scene_id not in matches:
        raise ValueError(f"member does not belong to {scene_id}: {member!r}")
    scene_index = pure.parts.index(scene_id)
    remainder = pure.parts[scene_index + 1 :]
    if not remainder:
        raise ValueError(f"member has no relative payload path: {member!r}")
    return Path(*remainder)


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item for item in root.rglob("*") if item.is_file())):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def inventory_paths(root: Path) -> tuple[list[dict[str, Any]], set[str]]:
    observed = utc_now()
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    scan_roots: list[tuple[Path, str]] = [
        (root, "repository"),
    ]
    for drive in ("D:/", "F:/"):
        base = Path(drive)
        if not base.exists():
            continue
        try:
            for child in base.iterdir():
                if "structured3d" in child.name.lower():
                    scan_roots.append((child, f"{drive[0]}_known_structured3d"))
        except OSError:
            continue

    def add(scene_id: str, source: Path, source_type: str, reason: str) -> None:
        key = (scene_id, str(source), source_type)
        records[key] = {
            "scene_id": scene_id,
            "source_path": str(source),
            "source_type": source_type,
            "first_observed_by_E8_utc": observed,
            "file_or_directory": (
                "directory" if source.is_dir() else "file"
            ),
            "historical_manifest_reference": (
                str(source)
                if "result" in {part.lower() for part in source.parts}
                else None
            ),
            "historical_output_reference": (
                str(source)
                if "local_outputs" in {part.lower() for part in source.parts}
                else None
            ),
            "inventory_reason": reason,
            "status": "pre_existing",
        }

    for scan_root, label in scan_roots:
        if not scan_root.exists():
            continue
        for current, directories, files in os.walk(scan_root):
            current_path = Path(current)
            # Never inspect Git object contents; path names are sufficient.
            directories[:] = [name for name in directories if name != ".git"]
            for name in directories + files:
                item = current_path / name
                for scene_id in scene_ids(str(item)):
                    add(scene_id, item, label, "path_or_filename_match")
            for name in files:
                item = current_path / name
                try:
                    size = item.stat().st_size
                except OSError:
                    continue
                if item.suffix.lower() not in TEXT_SUFFIXES or size > 64 * 1024**2:
                    continue
                try:
                    text = item.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for scene_id in sorted(set(scene_ids(text))):
                    add(scene_id, item, label, "text_content_match")
    rows = sorted(records.values(), key=lambda row: (
        row["scene_id"], row["source_path"], row["source_type"]
    ))
    return rows, {row["scene_id"] for row in rows}


def response_identity(response: requests.Response, requested_url: str) -> dict:
    headers = response.headers
    length = headers.get("Content-Length")
    if response.status_code == 206 and headers.get("Content-Range"):
        length = headers["Content-Range"].rsplit("/", 1)[-1]
    return {
        "requested_url": requested_url,
        "final_url": response.url,
        "http_status": response.status_code,
        "accept_ranges": headers.get("Accept-Ranges"),
        "content_length": int(length) if length and length.isdigit() else None,
        "etag": headers.get("ETag"),
        "last_modified": headers.get("Last-Modified"),
        "content_type": headers.get("Content-Type"),
        "probe_utc": utc_now(),
    }


def probe_object(session: requests.Session, url: str) -> dict:
    response = session.head(url, allow_redirects=True, timeout=(30, 120))
    identity = response_identity(response, url)
    if response.status_code >= 400 or identity["content_length"] is None:
        response.close()
        response = session.get(
            url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            allow_redirects=True,
            timeout=(30, 120),
        )
        identity = response_identity(response, url)
    response.close()
    if identity["http_status"] not in (200, 206):
        raise E8Stop(
            "Blocked_Remote_Object_Changed",
            f"remote probe failed for {url}: HTTP {identity['http_status']}",
        )
    if identity["content_length"] is None:
        raise E8Stop(
            "Blocked_Remote_Object_Changed",
            f"remote length unavailable for {url}",
        )
    return identity


def get_range(
    session: requests.Session, identity: dict, start: int, end: int
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        response = None
        try:
            response = session.get(
                identity["requested_url"],
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                allow_redirects=True,
                timeout=(30, 180),
            )
            data = response.content
            content_range = response.headers.get("Content-Range")
            if response.status_code != 206 or not content_range:
                raise E8Stop(
                    "Blocked_Archive_Mapping",
                    f"range request rejected for {identity['requested_url']}: "
                    f"HTTP {response.status_code}",
                )
            expected_prefix = f"bytes {start}-{end}/"
            if not content_range.startswith(expected_prefix):
                raise E8Stop(
                    "Blocked_Archive_Mapping",
                    f"unexpected Content-Range {content_range!r}",
                )
            if len(data) != end - start + 1:
                raise requests.RequestException(
                    "range response byte count mismatch"
                )
            return data
        except E8Stop:
            raise
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(2**attempt)
        finally:
            if response is not None:
                response.close()
    raise E8Stop(
        "Paused_Resumable",
        f"same-source Range read interrupted after retries: {last_error}",
    )


def _zip64_values(extra: bytes, needs: list[bool]) -> list[int | None]:
    cursor = 0
    payload = b""
    while cursor + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        field = extra[cursor : cursor + field_size]
        cursor += field_size
        if field_id == 0x0001:
            payload = field
            break
    values: list[int | None] = []
    cursor = 0
    for needed in needs:
        if not needed:
            values.append(None)
            continue
        if cursor + 8 > len(payload):
            raise ValueError("truncated ZIP64 extended field")
        values.append(struct.unpack_from("<Q", payload, cursor)[0])
        cursor += 8
    return values


def parse_central_directory(data: bytes, expected_entries: int) -> list[dict]:
    members: list[dict] = []
    cursor = 0
    while cursor < len(data):
        if cursor + 46 > len(data) or data[cursor : cursor + 4] != b"PK\x01\x02":
            raise ValueError(f"invalid central directory at byte {cursor}")
        values = struct.unpack_from("<4s6H3I5H2I", data, cursor)
        (
            _signature, _made, _needed, flags, method, _mtime, _mdate, crc,
            compressed, uncompressed, name_len, extra_len, comment_len,
            disk_start, _internal, _external, local_offset,
        ) = values
        payload_start = cursor + 46
        name_raw = data[payload_start : payload_start + name_len]
        extra = data[
            payload_start + name_len : payload_start + name_len + extra_len
        ]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_raw.decode(encoding, errors="strict")
        needs = [
            uncompressed == 0xFFFFFFFF,
            compressed == 0xFFFFFFFF,
            local_offset == 0xFFFFFFFF,
            disk_start == 0xFFFF,
        ]
        zip64 = _zip64_values(extra, needs)
        if needs[0]:
            uncompressed = int(zip64[0])
        if needs[1]:
            compressed = int(zip64[1])
        if needs[2]:
            local_offset = int(zip64[2])
        members.append({
            "path": name,
            "compressed_size": compressed,
            "uncompressed_size": uncompressed,
            "crc32": f"{crc:08x}",
            "compression_method": method,
            "local_header_offset": local_offset,
            "is_directory": name.endswith("/"),
        })
        cursor = payload_start + name_len + extra_len + comment_len
    if len(members) != expected_entries:
        raise ValueError(
            f"central directory entry count mismatch: "
            f"{len(members)} != {expected_entries}"
        )
    return members


def remote_zip_index(
    session: requests.Session, identity: dict
) -> tuple[list[dict], dict]:
    total = identity["content_length"]
    tail_start = max(0, total - (65535 + 22 + 20))
    tail = get_range(session, identity, tail_start, total - 1)
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0 or eocd_at + 22 > len(tail):
        raise E8Stop("Blocked_Archive_Mapping", "ZIP EOCD not found")
    (
        _sig, disk_no, cd_disk, disk_entries, total_entries, cd_size, cd_offset,
        _comment_len,
    ) = struct.unpack_from("<4s4H2IH", tail, eocd_at)
    if disk_no != 0 or cd_disk != 0:
        raise E8Stop("Blocked_Archive_Mapping", "multi-disk ZIP unsupported")
    zip64 = (
        total_entries == 0xFFFF
        or cd_size == 0xFFFFFFFF
        or cd_offset == 0xFFFFFFFF
    )
    if zip64:
        locator_absolute = tail_start + eocd_at - 20
        if locator_absolute < 0:
            raise E8Stop("Blocked_Archive_Mapping", "ZIP64 locator unavailable")
        locator = get_range(
            session, identity, locator_absolute, locator_absolute + 19
        )
        if locator[:4] != b"PK\x06\x07":
            raise E8Stop("Blocked_Archive_Mapping", "ZIP64 locator not found")
        _lsig, _disk, zip64_offset, _disks = struct.unpack("<4sIQI", locator)
        header = get_range(
            session, identity, zip64_offset, zip64_offset + 55
        )
        if header[:4] != b"PK\x06\x06":
            raise E8Stop("Blocked_Archive_Mapping", "ZIP64 EOCD not found")
        (
            _zsig, _zsize, _made, _needed, zdisk, zcd_disk, _disk_entries,
            total_entries, cd_size, cd_offset,
        ) = struct.unpack("<4sQ2H2I4Q", header)
        if zdisk != 0 or zcd_disk != 0:
            raise E8Stop("Blocked_Archive_Mapping", "multi-disk ZIP64 unsupported")
    cd_data = get_range(
        session, identity, int(cd_offset), int(cd_offset + cd_size - 1)
    )
    try:
        members = parse_central_directory(cd_data, int(total_entries))
    except (ValueError, UnicodeError, struct.error) as exc:
        raise E8Stop("Blocked_Archive_Mapping", str(exc)) from exc
    listing_hash = sha256_bytes(canonical_json(members))
    receipt = {
        "status": "passed",
        "zip64": zip64,
        "entry_count": len(members),
        "central_directory_offset": int(cd_offset),
        "central_directory_size": int(cd_size),
        "central_directory_bytes_sha256": sha256_bytes(cd_data),
        "member_listing_sha256": listing_hash,
    }
    return members, receipt


def assert_identity_matches(frozen: dict, current: dict) -> None:
    for field in ("final_url", "content_length", "etag", "last_modified"):
        if current.get(field) != frozen.get(field):
            raise E8Stop(
                "Blocked_Remote_Object_Changed",
                f"remote {field} changed for {frozen['requested_url']}",
            )


def download_archive(
    session: requests.Session,
    frozen: dict,
    part_path: Path,
    final_path: Path,
    quarantine: Path,
    log,
) -> dict:
    current = probe_object(session, frozen["requested_url"])
    assert_identity_matches(frozen, current)
    total = frozen["content_length"]
    if final_path.exists():
        if final_path.stat().st_size != total:
            raise E8Stop(
                "Blocked_Remote_Object_Changed",
                f"existing final archive size mismatch: {final_path}",
            )
        return {
            "status": "reused_complete",
            "path": str(final_path),
            "bytes": total,
            "completed_utc": utc_now(),
        }
    existing = part_path.stat().st_size if part_path.exists() else 0
    if existing > total:
        quarantine.mkdir(parents=True, exist_ok=True)
        moved = quarantine / f"{part_path.name}.{int(time.time())}.oversize"
        os.replace(part_path, moved)
        existing = 0
    headers = {"Range": f"bytes={existing}-{total - 1}"} if existing else {}
    response = session.get(
        frozen["requested_url"],
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=(30, 180),
    )
    if existing:
        content_range = response.headers.get("Content-Range", "")
        expected = f"bytes {existing}-"
        if response.status_code == 200:
            response.close()
            quarantine.mkdir(parents=True, exist_ok=True)
            moved = quarantine / f"{part_path.name}.{int(time.time())}.range_ignored"
            os.replace(part_path, moved)
            raise E8Stop(
                "Paused_Resumable",
                f"server ignored Range; quarantined partial {part_path.name}",
            )
        if response.status_code != 206 or not content_range.startswith(expected):
            response.close()
            raise E8Stop(
                "Blocked_Remote_Object_Changed",
                f"invalid resume response for {part_path.name}",
            )
        if not content_range.endswith(f"/{total}"):
            response.close()
            raise E8Stop(
                "Blocked_Remote_Object_Changed",
                f"resume total changed for {part_path.name}",
            )
    elif response.status_code != 200:
        response.close()
        raise E8Stop(
            "Paused_Resumable",
            f"fresh download HTTP {response.status_code}: {part_path.name}",
        )
    part_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if existing else "wb"
    written = existing
    last_report = time.monotonic()
    try:
        with part_path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if time.monotonic() - last_report >= 30:
                    log(
                        f"download_progress archive={part_path.name} "
                        f"bytes={written}/{total}"
                    )
                    last_report = time.monotonic()
    except (requests.RequestException, OSError) as exc:
        raise E8Stop(
            "Paused_Resumable",
            f"download interrupted for {part_path.name}: {exc}",
        ) from exc
    finally:
        response.close()
    if written != total:
        raise E8Stop(
            "Paused_Resumable",
            f"download size mismatch for {part_path.name}: {written}/{total}",
        )
    return {
        "status": "downloaded",
        "path": str(part_path),
        "bytes": written,
        "resumed_from_bytes": existing,
        "completed_utc": utc_now(),
    }


def local_member_listing(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        return [
            {
                "path": info.filename,
                "compressed_size": info.compress_size,
                "uncompressed_size": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "compression_method": info.compress_type,
                "local_header_offset": info.header_offset,
                "is_directory": info.is_dir(),
            }
            for info in archive.infolist()
        ]


def image_decode_ok(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def remove_incomplete_staging(path: Path, staging_root: Path) -> None:
    resolved = path.resolve()
    resolved_root = staging_root.resolve()
    if resolved.parent != resolved_root or not resolved.name.endswith(".tmp"):
        raise ValueError(f"refusing unsafe staging removal: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def extract_members(
    archive: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
    scene_id: str,
    destination: Path,
    verify_images: bool,
) -> list[dict]:
    seen: set[Path] = set()
    manifest: list[dict] = []
    destination.mkdir(parents=True, exist_ok=False)
    for info in infos:
        if info.is_dir():
            continue
        relative = safe_member_relative(info.filename, scene_id)
        if relative in seen:
            raise ValueError(f"duplicate extraction member: {relative}")
        seen.add(relative)
        output = destination / relative
        resolved = output.resolve()
        if destination.resolve() not in resolved.parents:
            raise ValueError(f"extraction escaped destination: {info.filename}")
        output.parent.mkdir(parents=True, exist_ok=True)
        crc = 0
        digest = hashlib.sha256()
        size = 0
        with archive.open(info) as source, output.open("xb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
                crc = binascii.crc32(chunk, crc)
                digest.update(chunk)
                size += len(chunk)
        if size != info.file_size or (crc & 0xFFFFFFFF) != info.CRC:
            raise ValueError(f"size/CRC mismatch: {info.filename}")
        decode_checked = (
            verify_images
            and output.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if decode_checked and not image_decode_ok(output):
            raise ValueError(f"basic image decode failed: {info.filename}")
        manifest.append({
            "member_path": info.filename,
            "relative_output_path": relative.as_posix(),
            "crc32": f"{info.CRC:08x}",
            "compressed_bytes": info.compress_size,
            "uncompressed_bytes": info.file_size,
            "sha256": digest.hexdigest(),
            "basic_image_decode_checked": decode_checked,
        })
    return manifest


def source_manifest(root: Path) -> tuple[dict, dict, dict, str, bool]:
    source_names = [
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
        "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "local_outputs/stage1_large_80m_train2048_sharded_v1/best.pt",
    ]
    weights = []
    missing_weights = []
    for name in weight_names:
        path = root / name
        if path.is_file():
            weights.append({
                "role": "DUSt3R" if "dust3r/checkpoints" in name else "Stage1",
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
        else:
            missing_weights.append(name)
    source = {
        "status": "frozen" if not missing_sources else "incomplete",
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "git_branch": git_output(root, "branch", "--show-current"),
        "clean_working_tree": not bool(git_output(root, "status", "--porcelain")),
        "sources": sources,
        "missing_sources": missing_sources,
    }
    weight = {
        "status": "frozen" if not missing_weights else "incomplete",
        "weights": weights,
        "missing_weights": missing_weights,
    }
    config = {
        "status": "frozen",
        "scene_order_source": "clean_raw_scene_pool.json order",
        "view_selection_rule": (
            "stable lexical member-path order within each Clean_Raw scene; "
            "no semantic or quality selection"
        ),
        "image_resize_crop_rule": (
            "must use the exact preprocessing implementation frozen in "
            "run_plane_dust3r_same_input_smoke.py and "
            "export_stage3_scene_plane_fusion.py"
        ),
        "random_seed": 0,
        "deterministic": True,
        "materialization_authorized": False,
        "global_cache_created_count": 0,
        "direct_support_created_count": 0,
        "exception_rule": "fail scene explicitly; do not silently substitute views",
    }
    contract = (
        "# E8 input-preparation contract\n\n"
        "E8 freezes but does not run input preparation. Future materialization "
        "requires a new Registration ID and must consume scenes in exactly the "
        "order stored in `clean_raw_scene_pool.json`.\n\n"
        "The source, weights, configuration, view ordering, resize/crop "
        "implementation and output schema are the objects recorded in the "
        "three adjacent manifests. No DUSt3R, Stage1, cache or direct-support "
        "materialization occurred in E8.\n"
    )
    complete = not missing_sources and not missing_weights
    return source, config, weight, contract, complete


class Runner:
    def __init__(self, root: Path, result: Path, d_root: Path, f_root: Path):
        self.root = root
        self.result = result
        self.d_root = d_root
        self.f_root = f_root
        self.log_path = result / "run.log"
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "LightRecon3D-E8-official-acquisition/1.0"
        )
        self.forbidden = {
            name: 0 for name in (
                "GT_semantic_parse_count", "GT_plane_count_query_count",
                "GT_quality_metric_count", "GT_visualization_count",
                "DUSt3R_input_preparation_call_count",
                "Stage1_input_preparation_call_count",
                "global_cache_created_count", "direct_support_created_count",
                "B_inference_call_count", "M_inference_call_count",
                "B_training_call_count", "M_training_call_count",
                "checkpoint_selection_call_count",
                "threshold_calibration_call_count",
                "formal_evaluation_call_count", "GT_matcher_call_count",
                "Plane_metric_call_count", "expansion_call_count",
                "refit_call_count", "plane_fit_call_count",
                "weighted_plane_fit_call_count", "RANSAC_call_count",
                "support_mutation_count",
            )
        }

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def finalize(self, verdict: str, reason: str, **extra: Any) -> None:
        write_json(self.result / "forbidden_call_audit.json", {
            "registration_id": REGISTRATION_ID,
            "status": "passed" if sum(self.forbidden.values()) == 0 else "failed",
            **self.forbidden,
            "forbidden_counter_sum": sum(self.forbidden.values()),
        })
        summary = {
            "registration_id": REGISTRATION_ID,
            "registration_state": "EXECUTED",
            "execution_verdict_issued": True,
            "execution_verdict": verdict,
            "reason": reason,
            "forbidden_counter_sum": sum(self.forbidden.values()),
            **extra,
        }
        write_json(self.result / "summary.json", summary)
        status_path = self.result / "registration_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update({
            "registration_state": "EXECUTED",
            "execution_authorized": True,
            "authorization_received_local_date": "2026-07-29",
            "execution_verdict": verdict,
            "execution_completed_utc": utc_now(),
        })
        write_json(status_path, status)
        self.log(f"verdict={verdict} reason={reason}")

    def run(self) -> None:
        if sha256_file(self.result / "registration.md") != PROTOCOL_SHA256:
            raise E8Stop("Invalid", "frozen registration hash mismatch")
        self.log("execution_authorized=true")

        inventory_path = self.result / "pre_acquisition_inventory.json"
        existing_inventory = json.loads(
            inventory_path.read_text(encoding="utf-8")
        )
        if existing_inventory.get("status") == "completed":
            inventory = existing_inventory
            rows = inventory["records"]
            preexisting = set(inventory["canonical_scene_ids"])
            self.log(
                f"inventory_reused records={len(rows)} "
                f"scenes={len(preexisting)}"
            )
        else:
            rows, preexisting = inventory_paths(self.root)
            inventory = {
                "registration_id": REGISTRATION_ID,
                "status": "completed",
                "scanner_regex": r"scene_(\d{5})(?!\d)",
                "record_count": len(rows),
                "unique_scene_count": len(preexisting),
                "canonical_scene_ids": sorted(preexisting),
                "records": rows,
                "completed_utc": utc_now(),
            }
            write_json(inventory_path, inventory)
            write_csv(
                self.result / "pre_acquisition_inventory.csv",
                rows,
                [
                    "scene_id", "source_path", "source_type",
                    "first_observed_by_E8_utc", "file_or_directory",
                    "historical_manifest_reference",
                    "historical_output_reference",
                    "inventory_reason", "status",
                ],
            )
            self.log(
                f"inventory_completed records={len(rows)} "
                f"scenes={len(preexisting)}"
            )
        inventory_hash = sha256_file(
            inventory_path
        )
        inventory_hash_receipt = (
            self.result / "pre_acquisition_inventory.sha256"
        )
        if not inventory_hash_receipt.read_text(
            encoding="ascii"
        ).startswith(inventory_hash):
            atomic_write(
                inventory_hash_receipt,
                (
                    f"{inventory_hash}  pre_acquisition_inventory.json\n"
                ).encode("ascii"),
            )

        # Freeze small official sources before candidate generation.
        source_rows: dict[str, dict] = {}
        source_bytes: dict[str, bytes] = {}
        for role, url, filename in (
            ("readme", README_URL, "official_readme_snapshot.txt"),
            ("errata", ERRATA_URL, "official_errata_snapshot.txt"),
        ):
            manifest_path = self.result / f"official_{role}_source.json"
            snapshot_path = self.result / filename
            existing_source = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            response = self.session.get(url, allow_redirects=True, timeout=(30, 120))
            if response.status_code != 200:
                raise E8Stop(
                    "Blocked_Official_Metadata_Changed",
                    f"{role} retrieval HTTP {response.status_code}",
                )
            content = response.content
            if existing_source.get("status") == "frozen":
                current_identity = response_identity(response, url)
                if (
                    sha256_bytes(content) != existing_source["sha256"]
                    or not snapshot_path.is_file()
                    or sha256_file(snapshot_path) != existing_source["sha256"]
                ):
                    raise E8Stop(
                        "Blocked_Official_Metadata_Changed",
                        f"frozen {role} content changed",
                    )
                for field in (
                    "final_url", "content_length", "etag", "last_modified"
                ):
                    if current_identity.get(field) != existing_source.get(field):
                        raise E8Stop(
                            "Blocked_Official_Metadata_Changed",
                            f"frozen {role} identity field changed: {field}",
                        )
                identity = existing_source
                self.log(f"official_source_reused role={role}")
            else:
                atomic_write(snapshot_path, content)
                identity = response_identity(response, url)
                identity.update({
                    "role": role,
                    "sha256": sha256_bytes(content),
                    "bytes_received": len(content),
                    "snapshot_path": str(snapshot_path),
                    "status": "frozen",
                })
                write_json(manifest_path, identity)
            source_rows[role] = identity
            source_bytes[role] = content
        errata_text = source_bytes["errata"].decode("utf-8", errors="replace")
        errata_ids = sorted(set(scene_ids(errata_text)))
        errata_manifest_path = self.result / "official_errata_scene_ids.json"
        existing_errata = json.loads(
            errata_manifest_path.read_text(encoding="utf-8")
        )
        if existing_errata.get("status") == "frozen":
            if existing_errata["scene_ids"] != errata_ids:
                raise E8Stop(
                    "Blocked_Official_Metadata_Changed",
                    "frozen errata scene set changed",
                )
        else:
            write_json(errata_manifest_path, {
                "status": "frozen",
                "scanner_regex": r"scene_(\d{5})(?!\d)",
                "source_sha256": source_rows["errata"]["sha256"],
                "scene_count": len(errata_ids),
                "scene_ids": errata_ids,
            })
            atomic_write(
                self.result / "official_errata.sha256",
                (
                    f"{source_rows['errata']['sha256']}  "
                    "official_errata_snapshot.txt\n"
                ).encode("ascii"),
            )

        candidate_path = self.result / "candidate_order_128.json"
        existing_candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
        if existing_candidates.get("status") == "frozen":
            if existing_candidates.get("salt") != SALT:
                raise E8Stop("Invalid", "frozen candidate salt changed")
            candidates = existing_candidates["candidates"]
            universe = json.loads(
                (self.result / "candidate_universe.json").read_text(encoding="utf-8")
            )["scene_ids"]
            self.log(f"candidates_reused count={len(candidates)}")
        else:
            universe = [
                f"scene_{value:05d}"
                for value in range(200, 3000)
                if f"scene_{value:05d}" not in preexisting
                and f"scene_{value:05d}" not in set(errata_ids)
            ]
            if len(universe) < 128:
                raise E8Stop(
                    "Blocked_Insufficient_Candidate_Universe",
                    f"candidate universe contains only {len(universe)} scenes",
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
                "range": ["scene_00200", "scene_02999"],
                "preexisting_total_count": len(preexisting),
                "errata_count": len(errata_ids),
                "candidate_universe_count": len(universe),
                "scene_ids": universe,
            })
            write_json(candidate_path, {
                "status": "frozen",
                "salt": SALT,
                "count": len(candidates),
                "candidates": candidates,
            })
            generator_source_hash = sha256_file(Path(__file__))
            write_json(self.result / "candidate_generation_audit.json", {
                "status": "passed",
                "inventory_sha256": inventory_hash,
                "errata_sha256": source_rows["errata"]["sha256"],
                "salt": SALT,
                "candidate_universe_count": len(universe),
                "ordering_implementation_source": str(Path(__file__).resolve()),
                "ordering_implementation_source_sha256": generator_source_hash,
                "generated_utc": utc_now(),
                "read_E7_candidate_file": False,
            })
            self.log(f"candidates_frozen count={len(candidates)}")

        names = list(PERSPECTIVE_NAMES) + [ANNOTATION_NAME]
        remote_path = self.result / "remote_object_manifest.json"
        existing_remote = json.loads(remote_path.read_text(encoding="utf-8"))
        if existing_remote.get("status") == "frozen":
            remote = {
                row["archive_name"]: row for row in existing_remote["objects"]
            }
            for name in names:
                current = probe_object(self.session, remote[name]["requested_url"])
                assert_identity_matches(remote[name], current)
            self.log(f"remote_object_manifest_reused count={len(remote)}")
        else:
            remote = {}
            for name in names:
                identity = probe_object(self.session, f"{BASE_URL}/{name}")
                if name in FORMAL_PERSPECTIVE:
                    target = self.d_root / "perspective_empty" / name
                    role = "formal_perspective"
                elif name == ANNOTATION_NAME:
                    target = self.d_root / "annotations" / name
                    role = "annotation"
                else:
                    target = None
                    role = "boundary_mapping_probe"
                identity.update({
                    "archive_name": name,
                    "role": role,
                    "D_target_path": str(target) if target else None,
                })
                remote[name] = identity
                self.log(
                    f"remote_object_frozen archive={name} "
                    f"bytes={identity['content_length']}"
                )
            remote_manifest = {
                "status": "frozen",
                "frozen_utc": utc_now(),
                "objects": [remote[name] for name in names],
            }
            write_json(remote_path, remote_manifest)
        remote_hash = sha256_file(self.result / "remote_object_manifest.json")
        atomic_write(
            self.result / "remote_object_manifest.sha256",
            f"{remote_hash}  remote_object_manifest.json\n".encode("ascii"),
        )

        indexes: dict[str, list[dict]] = {}
        probe_receipts: dict[str, dict] = {}
        member_hashes: dict[str, str] = {}
        scene_archives: dict[str, set[str]] = defaultdict(set)
        member_archives: dict[str, set[str]] = defaultdict(set)
        for name in PERSPECTIVE_NAMES:
            members, receipt = remote_zip_index(self.session, remote[name])
            indexes[name] = members
            probe_receipts[name] = receipt
            member_hashes[name] = receipt["member_listing_sha256"]
            for member in members:
                member_archives[member["path"]].add(name)
                for scene in scene_ids(member["path"]):
                    scene_archives[scene].add(name)
            self.log(
                f"remote_zip_index archive={name} entries={len(members)} "
                f"zip64={receipt['zip64']}"
            )
        annotation_members, annotation_receipt = remote_zip_index(
            self.session, remote[ANNOTATION_NAME]
        )
        indexes[ANNOTATION_NAME] = annotation_members
        probe_receipts[ANNOTATION_NAME] = annotation_receipt
        member_hashes[ANNOTATION_NAME] = annotation_receipt[
            "member_listing_sha256"
        ]
        write_json(self.result / "remote_zip_probe_receipts.json", {
            "status": "passed",
            "archives": probe_receipts,
        })
        write_json(self.result / "archive_member_index_hashes.json", {
            "status": "frozen",
            "hash_algorithm": "sha256 canonical JSON member listing",
            "archives": member_hashes,
        })
        scene_index = {
            name: {
                "scene_count": len(
                    {scene for member in indexes[name] for scene in scene_ids(member["path"])}
                ),
                "scene_ids": sorted(
                    {scene for member in indexes[name] for scene in scene_ids(member["path"])}
                ),
                "member_count": len(indexes[name]),
                "member_listing_sha256": member_hashes[name],
            }
            for name in names
        }
        write_json(self.result / "archive_scene_index.json", {
            "status": "frozen",
            "scanner_regex": r"scene_(\d{5})(?!\d)",
            "archives": scene_index,
        })
        candidate_map: dict[str, str] = {}
        mapping_errors: list[dict] = []
        formal_set = set(FORMAL_PERSPECTIVE)
        boundary_set = set(BOUNDARY_PERSPECTIVE)
        for candidate in candidates:
            scene = candidate["scene_id"]
            found = scene_archives.get(scene, set())
            formal_found = found & formal_set
            boundary_found = found & boundary_set
            if len(formal_found) != 1 or boundary_found:
                mapping_errors.append({
                    "scene_id": scene,
                    "archives": sorted(found),
                    "reason": "candidate not uniquely mapped to formal archive",
                })
            else:
                candidate_map[scene] = next(iter(formal_found))
        duplicate_members = sorted(
            member for member, owners in member_archives.items()
            if len(owners) > 1
        )
        if duplicate_members:
            mapping_errors.append({
                "reason": "duplicate member path across archives",
                "count": len(duplicate_members),
                "examples": duplicate_members[:20],
            })
        write_json(self.result / "scene_to_archive_map.json", {
            "status": "passed" if not mapping_errors else "failed",
            "mapping": candidate_map,
            "errors": mapping_errors,
            "formula_mapping_used": False,
        })
        if mapping_errors:
            raise E8Stop(
                "Blocked_Archive_Mapping",
                f"archive mapping produced {len(mapping_errors)} error groups",
            )

        # Disk gate is evaluated before creating D/F acquisition directories.
        formal_bytes = sum(
            remote[name]["content_length"]
            for name in list(FORMAL_PERSPECTIVE) + [ANNOTATION_NAME]
        )
        selected = set(candidate_map)
        raw_uncompressed = sum(
            member["uncompressed_size"]
            for name in FORMAL_PERSPECTIVE
            for member in indexes[name]
            if selected.intersection(scene_ids(member["path"]))
            and not member["is_directory"]
        )
        annotation_uncompressed = sum(
            member["uncompressed_size"]
            for member in annotation_members
            if selected.intersection(scene_ids(member["path"]))
            and not member["is_directory"]
        )
        d_free = shutil.disk_usage("D:/").free
        e_free = shutil.disk_usage(str(self.root)).free
        f_free = shutil.disk_usage("F:/").free
        disk = {
            "status": "passed",
            "measured_utc": utc_now(),
            "gib_definition_bytes": GIB,
            "total_archive_download_bytes": formal_bytes,
            "predicted_raw_rgb_uncompressed_bytes": raw_uncompressed,
            "predicted_annotation_uncompressed_bytes": annotation_uncompressed,
            "predicted_raw_uncompressed_bytes": (
                raw_uncompressed + annotation_uncompressed
            ),
            "D_free_bytes": d_free,
            "D_required_bytes": formal_bytes + 40 * GIB,
            "F_free_bytes": f_free,
            "F_required_bytes": (
                raw_uncompressed + annotation_uncompressed + 64 * GIB
            ),
            "E_free_bytes": e_free,
            "E_required_bytes": 2 * GIB,
        }
        if e_free < 2 * GIB:
            disk["status"] = "blocked"
            write_json(self.result / "disk_preflight.json", disk)
            raise E8Stop("Blocked_Insufficient_E_Space", "E disk gate failed")
        if d_free < formal_bytes + 40 * GIB:
            disk["status"] = "blocked"
            write_json(self.result / "disk_preflight.json", disk)
            raise E8Stop("Blocked_Insufficient_D_Space", "D disk gate failed")
        if f_free < raw_uncompressed + annotation_uncompressed + 64 * GIB:
            disk["status"] = "blocked"
            write_json(self.result / "disk_preflight.json", disk)
            raise E8Stop("Blocked_Insufficient_F_Space", "F disk gate failed")
        write_json(self.result / "disk_preflight.json", disk)
        self.log(
            f"disk_preflight passed D_free={d_free} F_free={f_free} "
            f"E_free={e_free}"
        )

        for path in (
            self.d_root / "perspective_empty",
            self.d_root / "annotations",
            self.d_root / "partial",
            self.d_root / "quarantine",
            self.f_root / "raw_rgb",
            self.f_root / "sealed_annotations",
            self.f_root / "extraction_staging",
            self.f_root / "future_global_cache",
            self.f_root / "future_direct_support",
        ):
            path.mkdir(parents=True, exist_ok=True)

        downloads: dict[str, dict] = {}
        archive_hashes: dict[str, dict] = {}
        for name in list(FORMAL_PERSPECTIVE) + [ANNOTATION_NAME]:
            if shutil.disk_usage("D:/").free < 40 * GIB:
                raise E8Stop(
                    "Paused_Low_Disk",
                    "D free space fell below 40 GiB before new archive",
                )
            final = (
                self.d_root / "annotations" / name
                if name == ANNOTATION_NAME
                else self.d_root / "perspective_empty" / name
            )
            part = self.d_root / "partial" / f"{name}.part"
            downloads[name] = download_archive(
                self.session,
                remote[name],
                part,
                final,
                self.d_root / "quarantine",
                self.log,
            )
            candidate_path = final if final.exists() else part
            digest = sha256_file(candidate_path)
            local_listing = local_member_listing(candidate_path)
            local_listing_hash = sha256_bytes(canonical_json(local_listing))
            if local_listing_hash != member_hashes[name]:
                raise E8Stop(
                    "Invalid",
                    f"local/remote member index mismatch for {name}",
                )
            if candidate_path.stat().st_size != remote[name]["content_length"]:
                raise E8Stop("Invalid", f"archive size mismatch for {name}")
            if candidate_path == part:
                os.replace(part, final)
            os.chmod(final, stat.S_IREAD)
            archive_hashes[name] = {
                "path": str(final),
                "sha256": digest,
                "bytes": final.stat().st_size,
                "remote_member_listing_sha256": member_hashes[name],
                "local_member_listing_sha256": local_listing_hash,
                "official_checksum_available": False,
                "read_only": not bool(final.stat().st_mode & stat.S_IWRITE),
            }
            downloads[name].update({
                "status": "complete_verified",
                "final_path": str(final),
                "sha256": digest,
                "member_index_match": True,
            })
            write_json(self.result / "archive_download_receipts.json", {
                "status": "in_progress",
                "archives": downloads,
            })
            write_json(self.result / "archive_sha256_manifest.json", {
                "status": "in_progress",
                "archives": archive_hashes,
            })
            self.log(f"archive_complete archive={name} sha256={digest}")
        write_json(self.result / "archive_download_receipts.json", {
            "status": "completed",
            "archives": downloads,
        })
        write_json(self.result / "archive_sha256_manifest.json", {
            "status": "completed",
            "archives": archive_hashes,
        })
        write_json(self.result / "archive_retention_contract.json", {
            "status": "active",
            "completed_official_archives_must_be_retained": True,
            "archive_deletion_stage_exists": False,
            "completed_archive_count": len(archive_hashes),
            "paths": [row["path"] for row in archive_hashes.values()],
        })

        raw_manifest: dict[str, Any] = {}
        annotation_manifest: dict[str, Any] = {}
        provenance: list[dict] = []
        zip_handles = {
            name: zipfile.ZipFile(Path(archive_hashes[name]["path"]))
            for name in archive_hashes
        }
        try:
            info_by_scene: dict[str, dict[str, list[zipfile.ZipInfo]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for name, archive in zip_handles.items():
                for info in archive.infolist():
                    for scene in scene_ids(info.filename):
                        if scene in selected:
                            info_by_scene[scene][name].append(info)
            for candidate in candidates:
                scene = candidate["scene_id"]
                if shutil.disk_usage("F:/").free < 64 * GIB:
                    raise E8Stop(
                        "Paused_Low_Disk",
                        "F free space fell below 64 GiB before new scene",
                    )
                archive_name = candidate_map[scene]
                raw_infos = info_by_scene[scene].get(archive_name, [])
                ann_infos = info_by_scene[scene].get(ANNOTATION_NAME, [])
                staging_root = self.f_root / "extraction_staging"
                stage = staging_root / f"{scene}.tmp"
                remove_incomplete_staging(stage, staging_root)
                stage.mkdir(parents=True)
                started = utc_now()
                try:
                    raw_rows = extract_members(
                        zip_handles[archive_name],
                        raw_infos,
                        scene,
                        stage / "raw_rgb",
                        verify_images=True,
                    )
                    ann_rows = extract_members(
                        zip_handles[ANNOTATION_NAME],
                        ann_infos,
                        scene,
                        stage / "sealed_annotations",
                        verify_images=False,
                    )
                    if not raw_rows or not ann_rows:
                        raise ValueError("required RGB or annotation members missing")
                    raw_target = self.f_root / "raw_rgb" / scene
                    ann_target = self.f_root / "sealed_annotations" / scene
                    if raw_target.exists() or ann_target.exists():
                        raise ValueError("refusing to overwrite existing scene")
                    os.replace(stage / "raw_rgb", raw_target)
                    os.replace(stage / "sealed_annotations", ann_target)
                    stage.rmdir()
                    completed = utc_now()
                    raw_tree = tree_sha256(raw_target)
                    ann_tree = tree_sha256(ann_target)
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
                        "member_count": len(ann_rows),
                        "tree_sha256": ann_tree,
                        "members": ann_rows,
                        "semantic_parse_count": 0,
                    }
                    all_rows = raw_rows + ann_rows
                    provenance.append({
                        "scene_id": scene,
                        "candidate_rank": candidate["rank"],
                        "candidate_hash": candidate["candidate_hash"],
                        "source_archive_name": archive_name,
                        "source_archive_URL": remote[archive_name]["requested_url"],
                        "source_archive_SHA256": archive_hashes[archive_name]["sha256"],
                        "source_member_listing": [row["member_path"] for row in all_rows],
                        "member_CRC32": [row["crc32"] for row in all_rows],
                        "compressed_bytes": sum(row["compressed_bytes"] for row in all_rows),
                        "uncompressed_bytes": sum(row["uncompressed_bytes"] for row in all_rows),
                        "first_extraction_start_utc": started,
                        "first_complete_utc": completed,
                        "raw_rgb_tree_SHA256": raw_tree,
                        "sealed_annotation_tree_SHA256": ann_tree,
                        "pre_acquisition_inventory_SHA256": inventory_hash,
                        "remote_object_manifest_SHA256": remote_hash,
                        "archive_scene_index_SHA256": sha256_file(
                            self.result / "archive_scene_index.json"
                        ),
                        "not_in_pre_acquisition_inventory": scene not in preexisting,
                        "in_frozen_candidate_order": True,
                        "GT_semantics_read": False,
                        "clean_raw_status": "Clean_Raw",
                    })
                    self.log(
                        f"scene_extracted scene={scene} raw_members={len(raw_rows)} "
                        f"annotation_members={len(ann_rows)}"
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
                        "semantic_parse_count": 0,
                    }
                    self.log(f"scene_ineligible scene={scene} reason={exc}")
        finally:
            for archive in zip_handles.values():
                archive.close()

        write_json(self.result / "raw_scene_manifest.json", {
            "status": "completed",
            "scenes": raw_manifest,
        })
        write_json(self.result / "sealed_annotation_manifest.json", {
            "status": "completed",
            "semantic_parse_count": 0,
            "scenes": annotation_manifest,
        })
        write_json(self.result / "per_scene_provenance.json", {
            "status": "completed",
            "scenes": provenance,
        })
        provenance_fields = [
            "scene_id", "candidate_rank", "candidate_hash",
            "source_archive_name", "source_archive_URL",
            "source_archive_SHA256", "compressed_bytes",
            "uncompressed_bytes", "first_extraction_start_utc",
            "first_complete_utc", "raw_rgb_tree_SHA256",
            "sealed_annotation_tree_SHA256",
            "pre_acquisition_inventory_SHA256",
            "remote_object_manifest_SHA256",
            "archive_scene_index_SHA256",
            "not_in_pre_acquisition_inventory",
            "in_frozen_candidate_order", "GT_semantics_read",
            "clean_raw_status",
        ]
        write_csv(
            self.result / "per_scene_provenance.csv",
            provenance,
            provenance_fields,
        )
        clean = [row["scene_id"] for row in provenance]
        write_json(self.result / "clean_raw_scene_pool.json", {
            "status": "frozen",
            "qualification": "Clean_Raw",
            "count": len(clean),
            "scene_ids": clean,
            "order_rule": "frozen candidate order",
        })

        source, config, weights, contract, prep_complete = source_manifest(self.root)
        write_json(self.result / "input_preparation_source_manifest.json", source)
        write_json(self.result / "input_preparation_config.json", config)
        write_json(self.result / "input_preparation_weights.json", weights)
        atomic_write(
            self.result / "input_preparation_contract.md",
            contract.encode("utf-8"),
        )
        if len(clean) < 77:
            raise E8Stop(
                "Blocked_Insufficient_Clean_Raw_Scenes",
                f"only {len(clean)} of 128 candidates qualified",
            )
        if not prep_complete:
            raise E8Stop(
                "Blocked_Input_Preparation_Freeze",
                "future input-preparation source or required weights are missing",
            )
        self.finalize(
            "Ready_Clean_Raw_Pool",
            f"{len(clean)} Clean_Raw scenes acquired",
            clean_raw_scene_count=len(clean),
            archive_count=len(archive_hashes),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--result-dir",
        default="results/official_archive_acquisition_e8_20260728_v1",
    )
    parser.add_argument(
        "--d-root", default="D:/Structured3D_official_archives_e8"
    )
    parser.add_argument("--f-root", default="F:/Structured3D_clean_e8")
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
    except E8Stop as stop:
        runner.finalize(stop.verdict, stop.reason)
        return 2
    except Exception as exc:
        runner.finalize("Paused_Resumable", f"unhandled interruption: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
