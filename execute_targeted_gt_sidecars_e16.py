"""Acquire only E14-selected Structured3D GT sidecars by HTTP Range.

This program deliberately treats layout and camera-pose members as sealed
bytes. It never deserializes JSON or parses camera numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


REGISTRATION_ID = "E16_PREREG_targeted_GT_sidecar_and_final_BM_20260730_v1"
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "e16_targeted_gt_sidecars.json"
OUT = ROOT / "results" / "targeted_GT_sidecar_and_final_BM_e16_20260730_v1"
ALLOWED_STOP = {
    "Blocked_E14_Upstream_Mismatch",
    "Blocked_Missing_GT_Sidecar_Members",
    "Blocked_Remote_Object_Changed",
    "Paused_Resumable",
    "Invalid",
}


class E16Stop(RuntimeError):
    def __init__(self, verdict: str, reason: str):
        if verdict not in ALLOWED_STOP:
            raise ValueError(f"unregistered E16 stop: {verdict}")
        super().__init__(reason)
        self.verdict = verdict
        self.reason = reason


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    if config.get("registration_id") != REGISTRATION_ID:
        raise E16Stop("Invalid", "configuration registration ID mismatch")
    return config


def upstream_path(config: dict[str, Any], key: str) -> Path:
    return Path(config["upstream_repository"]) / config[key]


def output_npz(root: Path, scene_id: str, filename: str) -> Path:
    direct = root / scene_id / filename
    if direct.is_file():
        return direct
    matches = list((root / scene_id).rglob(filename))
    if len(matches) != 1:
        raise E16Stop(
            "Blocked_E14_Upstream_Mismatch",
            f"{scene_id}: expected one {filename}, found {len(matches)}",
        )
    return matches[0]


def verify_e14(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    view_manifest_path = upstream_path(config, "e14_view_manifest")
    state_path = upstream_path(config, "e14_execution_state")
    manifest = load_json(view_manifest_path)
    state = load_json(state_path)
    records = manifest.get("records", [])
    state_records = state.get("records", [])
    expected = {
        "scene_count": config["expected_scene_count"],
        "five_view_scene_count": config["expected_five_view_scenes"],
        "four_view_scene_count": config["expected_four_view_scenes"],
    }
    observed = {
        "scene_count": len(records),
        "five_view_scene_count": sum(row.get("view_count") == 5 for row in records),
        "four_view_scene_count": sum(row.get("view_count") == 4 for row in records),
    }
    if observed != expected or len(state_records) != len(records):
        raise E16Stop(
            "Blocked_E14_Upstream_Mismatch",
            f"E14 scene/view counts differ: expected={expected} observed={observed}",
        )
    selected_views = sum(int(row["view_count"]) for row in records)
    if selected_views != config["expected_selected_views"]:
        raise E16Stop(
            "Blocked_E14_Upstream_Mismatch",
            f"selected views {selected_views} != {config['expected_selected_views']}",
        )
    state_by_scene = {row["scene_id"]: row for row in state_records}
    if list(state_by_scene) != [row["scene_id"] for row in records]:
        raise E16Stop("Blocked_E14_Upstream_Mismatch", "E14 scene order differs")
    global_root = Path(config["global_cache_root"])
    support_root = Path(config["direct_support_root"])
    rgb_verified = global_verified = support_verified = 0
    for row in records:
        scene = row["scene_id"]
        frozen = state_by_scene[scene]
        paths = [Path(value) for value in row["absolute_image_paths"]]
        hashes = row["input_file_hashes"]
        if len(paths) != row["view_count"] or len(hashes) != row["view_count"]:
            raise E16Stop("Blocked_E14_Upstream_Mismatch", f"{scene}: RGB list mismatch")
        for path, expected_hash in zip(paths, hashes, strict=True):
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise E16Stop(
                    "Blocked_E14_Upstream_Mismatch", f"{scene}: RGB hash mismatch {path}"
                )
            rgb_verified += 1
        global_path = output_npz(global_root, scene, "global_cache_e14.npz")
        support_path = output_npz(support_root, scene, "direct_support_e14.npz")
        if sha256_file(global_path) != frozen["global_cache_sha256"]:
            raise E16Stop(
                "Blocked_E14_Upstream_Mismatch", f"{scene}: global-cache hash mismatch"
            )
        if sha256_file(support_path) != frozen["direct_support_sha256"]:
            raise E16Stop(
                "Blocked_E14_Upstream_Mismatch", f"{scene}: direct-support hash mismatch"
            )
        global_verified += 1
        support_verified += 1
    audit = {
        "status": "passed",
        "scene_count": len(records),
        "five_view_scene_count": observed["five_view_scene_count"],
        "four_view_scene_count": observed["four_view_scene_count"],
        "selected_view_count": selected_views,
        "rgb_hashes_verified": rgb_verified,
        "global_cache_hashes_verified": global_verified,
        "direct_support_hashes_verified": support_verified,
        "view_manifest_sha256": sha256_file(view_manifest_path),
        "execution_state_sha256": sha256_file(state_path),
        "verified_utc": utc_now(),
    }
    write_json(OUT / "e14_upstream_verification.json", audit)
    return records, audit


def normalize_member(path: str) -> str:
    return path.replace("\\", "/")


def map_sidecars(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    e9 = upstream_path(config, "e9_result_directory")
    source = load_json(upstream_path(config, "e11_source_mapping"))
    remote_doc = load_json(e9 / "remote_object_manifest.json")
    remote = {row["archive_name"]: row for row in remote_doc["objects"]}
    archive_by_scene = source["perspective_archive_by_scene"]
    index_cache: dict[str, tuple[dict[str, list[dict]], str, str]] = {}
    selected: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for record in records:
        scene = record["scene_id"]
        archive = archive_by_scene.get(scene)
        if not archive or archive not in remote:
            errors.append({"scene_id": scene, "error": "missing archive owner"})
            continue
        if archive not in index_cache:
            first_path = e9 / "remote_member_indexes_probe_1" / f"{archive}.members.json"
            second_path = e9 / "remote_member_indexes_probe_2" / f"{archive}.members.json"
            first = load_json(first_path)
            second = load_json(second_path)
            first_rows = [row for row in first["members"] if not row["is_directory"]]
            second_rows = [row for row in second["members"] if not row["is_directory"]]
            if canonical_json(first_rows) != canonical_json(second_rows):
                raise E16Stop(
                    "Blocked_Missing_GT_Sidecar_Members",
                    f"dual frozen indexes differ: {archive}",
                )
            by_path: dict[str, list[dict]] = {}
            for member in first_rows:
                by_path.setdefault(normalize_member(member["path"]), []).append(member)
            index_cache[archive] = (
                by_path,
                sha256_file(first_path),
                sha256_file(second_path),
            )
        by_path, probe_1_hash, probe_2_hash = index_cache[archive]
        for rgb_relative in record["relative_image_paths"]:
            rgb_member = normalize_member(f"Structured3D/{scene}/{rgb_relative}")
            rgb_parent = rgb_member.rsplit("/", 1)[0]
            if len(by_path.get(rgb_member, [])) != 1:
                errors.append(
                    {"scene_id": scene, "member": rgb_member, "error": "RGB not unique"}
                )
                continue
            for sidecar_name in config["sidecar_names"]:
                target_member = f"{rgb_parent}/{sidecar_name}"
                matches = by_path.get(target_member, [])
                if len(matches) != 1:
                    errors.append(
                        {
                            "scene_id": scene,
                            "member": target_member,
                            "count": len(matches),
                        }
                    )
                    continue
                member = matches[0]
                relative = target_member.split(f"Structured3D/{scene}/", 1)[1]
                target_path = Path(config["sealed_sidecar_root"]) / scene / relative
                identity = {
                    "archive_name": archive,
                    "path": target_member,
                    "local_header_offset": member["local_header_offset"],
                    "compressed_size": member["compressed_size"],
                    "crc32": member["crc32"],
                }
                selected.append(
                    {
                        "member_id": sha256_bytes(canonical_json(identity)),
                        "scene_id": scene,
                        "view_id": rgb_relative.rsplit("/", 2)[-2],
                        "rgb_relative_path": rgb_relative,
                        "sidecar_name": sidecar_name,
                        "archive_name": archive,
                        "archive_url": remote[archive]["requested_url"],
                        "archive_total_length": remote[archive]["total_length"],
                        "archive_central_directory_sha256": remote[archive][
                            "central_directory_canonical_sha256"
                        ],
                        "probe_1_index_sha256": probe_1_hash,
                        "probe_2_index_sha256": probe_2_hash,
                        "normalized_member_path": target_member,
                        "normalized_parent_path": rgb_parent,
                        "local_header_offset": member["local_header_offset"],
                        "compression_method": member["compression_method"],
                        "compressed_size": member["compressed_size"],
                        "uncompressed_size": member["uncompressed_size"],
                        "crc32": member["crc32"],
                        "target_path": str(target_path),
                    }
                )
    duplicate_count = sum(
        count - 1
        for count in Counter(
            (row["archive_name"], row["normalized_member_path"]) for row in selected
        ).values()
        if count > 1
    )
    counts = Counter(row["sidecar_name"] for row in selected)
    if (
        errors
        or duplicate_count
        or len(selected) != config["expected_sidecar_members"]
        or counts["layout.json"] != config["expected_selected_views"]
        or counts["camera_pose.txt"] != config["expected_selected_views"]
    ):
        write_json(
            OUT / "sidecar_mapping_audit.json",
            {
                "status": "blocked",
                "errors": errors,
                "duplicates": duplicate_count,
                "counts": counts,
            },
        )
        raise E16Stop(
            "Blocked_Missing_GT_Sidecar_Members",
            f"sidecar mapping failed errors={len(errors)} duplicates={duplicate_count}",
        )
    mapping = {
        "registration_id": REGISTRATION_ID,
        "status": "frozen",
        "selected_view_count": config["expected_selected_views"],
        "layout_mapping_count": counts["layout.json"],
        "camera_mapping_count": counts["camera_pose.txt"],
        "total_member_count": len(selected),
        "duplicate_count": duplicate_count,
        "missing_count": 0,
        "archive_count": len({row["archive_name"] for row in selected}),
        "compressed_bytes": sum(row["compressed_size"] for row in selected),
        "uncompressed_bytes": sum(row["uncompressed_size"] for row in selected),
        "members": selected,
        "frozen_utc": utc_now(),
    }
    write_json(OUT / "sidecar_member_manifest.json", mapping)
    return selected


def exact_range(
    session: requests.Session,
    url: str,
    start: int,
    end: int,
    total: int,
    config: dict[str, Any],
) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, int(config["range_retry_count"]) + 1):
        response = None
        try:
            response = session.get(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
                stream=True,
                allow_redirects=True,
                timeout=(
                    config["range_connect_timeout_seconds"],
                    config["range_read_timeout_seconds"],
                ),
            )
            expected_range = f"bytes {start}-{end}/{total}"
            if response.status_code != 206:
                raise requests.RequestException(f"HTTP {response.status_code}")
            if response.headers.get("Content-Range") != expected_range:
                raise requests.RequestException(
                    f"Content-Range {response.headers.get('Content-Range')!r} "
                    f"!= {expected_range!r}"
                )
            expected_length = end - start + 1
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected_length:
                raise requests.RequestException(
                    f"Content-Length {content_length} != {expected_length}"
                )
            # Never call response.content here. If a server ignores Range, that
            # convenience property can materialize the complete multi-GiB ZIP
            # before the caller gets a chance to reject it.
            chunks: list[bytes] = []
            received = 0
            for chunk in response.iter_content(chunk_size=min(65536, expected_length)):
                if not chunk:
                    continue
                received += len(chunk)
                if received > expected_length:
                    raise requests.RequestException("Range returned excess bytes")
                chunks.append(chunk)
            data = b"".join(chunks)
            if received != expected_length:
                raise requests.RequestException("Range byte length mismatch")
            return data, dict(response.headers)
        except requests.RequestException as exc:
            last_error = exc
            if attempt < int(config["range_retry_count"]):
                time.sleep(2**attempt)
        finally:
            if response is not None:
                response.close()
    raise E16Stop("Paused_Resumable", f"Range retries exhausted: {last_error}")


def verify_remote_identities(
    config: dict[str, Any], selected: list[dict[str, Any]]
) -> dict[str, Any]:
    e9 = upstream_path(config, "e9_result_directory")
    frozen_doc = load_json(e9 / "remote_object_manifest.json")
    frozen = {row["archive_name"]: row for row in frozen_doc["objects"]}
    checkpoint_path = OUT / "remote_identity_verification.json"
    checkpoint = load_json(checkpoint_path) if checkpoint_path.is_file() else {}
    verified = checkpoint.get("objects", {})
    names = sorted({row["archive_name"] for row in selected})
    with requests.Session() as session:
        for name in names:
            row = frozen[name]
            previous = verified.get(name)
            if previous and previous.get("status") == "matched":
                continue
            head_start, head_end = row["fixed_head_range"]
            tail_start, tail_end = row["fixed_tail_range"]
            head, head_headers = exact_range(
                session,
                row["requested_url"],
                head_start,
                head_end,
                row["total_length"],
                config,
            )
            tail, tail_headers = exact_range(
                session,
                row["requested_url"],
                tail_start,
                tail_end,
                row["total_length"],
                config,
            )
            observed = {
                "status": "matched",
                "archive_name": name,
                "total_length": row["total_length"],
                "head_sha256": sha256_bytes(head),
                "tail_sha256": sha256_bytes(tail),
                "strong_etag": head_headers.get("ETag") or tail_headers.get("ETag"),
                "last_modified": head_headers.get("Last-Modified")
                or tail_headers.get("Last-Modified"),
                "verified_utc": utc_now(),
            }
            if (
                observed["head_sha256"] != row["fixed_head_range_sha256"]
                or observed["tail_sha256"] != row["fixed_tail_range_sha256"]
                or (
                    row.get("strong_etag")
                    and observed["strong_etag"] != row["strong_etag"]
                )
                or (
                    row.get("last_modified")
                    and observed["last_modified"] != row["last_modified"]
                )
            ):
                write_json(
                    checkpoint_path,
                    {"status": "blocked", "archive_name": name, "observed": observed},
                )
                raise E16Stop(
                    "Blocked_Remote_Object_Changed", f"remote identity changed: {name}"
                )
            verified[name] = observed
            write_json(
                checkpoint_path,
                {
                    "status": "in_progress",
                    "verified_count": len(verified),
                    "expected_count": len(names),
                    "objects": verified,
                },
            )
    result = {
        "status": "passed",
        "verified_count": len(verified),
        "expected_count": len(names),
        "objects": verified,
        "verified_utc": utc_now(),
    }
    write_json(checkpoint_path, result)
    return result


def decode_local_name(raw: bytes, flags: int) -> str:
    return raw.decode("utf-8" if flags & 0x800 else "cp437")


def existing_member_valid(row: dict[str, Any], receipt: dict[str, Any]) -> bool:
    target = Path(row["target_path"])
    return (
        receipt.get("status") == "completed"
        and target.is_file()
        and target.stat().st_size == row["uncompressed_size"]
        and sha256_file(target) == receipt.get("uncompressed_sha256")
    )


def download_member(
    session: requests.Session, row: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    offset = row["local_header_offset"]
    total = row["archive_total_length"]
    header, _ = exact_range(
        session, row["archive_url"], offset, offset + 29, total, config
    )
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034B50:
        raise E16Stop("Invalid", f"local header signature mismatch: {row['member_id']}")
    flags, method = fields[2], fields[3]
    name_length, extra_length = fields[9], fields[10]
    if method != row["compression_method"] or method not in (0, 8):
        raise E16Stop("Invalid", f"compression mismatch: {row['member_id']}")
    variable_and_compressed, _ = exact_range(
        session,
        row["archive_url"],
        offset + 30,
        offset
        + 30
        + name_length
        + extra_length
        + row["compressed_size"]
        - 1,
        total,
        config,
    )
    variable_length = name_length + extra_length
    variable = variable_and_compressed[:variable_length]
    local_name = decode_local_name(variable[:name_length], flags).replace("\\", "/")
    if local_name != row["normalized_member_path"]:
        raise E16Stop("Invalid", f"local filename mismatch: {row['member_id']}")
    compressed = variable_and_compressed[variable_length:]
    if len(compressed) != row["compressed_size"]:
        raise E16Stop("Invalid", f"compressed size mismatch: {row['member_id']}")
    try:
        raw = compressed if method == 0 else zlib.decompress(compressed, -15)
    except zlib.error as exc:
        raise E16Stop("Invalid", f"deflate failed: {row['member_id']}") from exc
    crc32 = f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}"
    if len(raw) != row["uncompressed_size"] or crc32 != row["crc32"].lower():
        raise E16Stop("Invalid", f"size/CRC mismatch: {row['member_id']}")
    target = Path(row["target_path"])
    staging = Path(config["sealed_sidecar_root"]) / ".staging"
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
        "view_id": row["view_id"],
        "sidecar_name": row["sidecar_name"],
        "archive_name": row["archive_name"],
        "normalized_member_path": row["normalized_member_path"],
        "target_path": str(target),
        "compressed_size": len(compressed),
        "uncompressed_size": len(raw),
        "crc32": crc32,
        "compressed_sha256": sha256_bytes(compressed),
        "uncompressed_sha256": sha256_bytes(raw),
        "completed_utc": utc_now(),
    }


def acquire(config: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
    state_path = OUT / "sidecar_download_state.json"
    state = (
        load_json(state_path)
        if state_path.is_file()
        else {"registration_id": REGISTRATION_ID, "members": {}}
    )
    members = state.setdefault("members", {})
    pending = [
        row
        for row in selected
        if not existing_member_valid(row, members.get(row["member_id"], {}))
    ]

    def fetch(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        with requests.Session() as session:
            return row, download_member(session, row, config)

    newly_completed = 0
    with ThreadPoolExecutor(max_workers=int(config["download_workers"])) as pool:
        futures = [pool.submit(fetch, row) for row in pending]
        for future in as_completed(futures):
            row, receipt = future.result()
            members[row["member_id"]] = receipt
            newly_completed += 1
            if newly_completed % 25 == 0 or newly_completed == len(pending):
                state.update(
                    {
                        "status": "in_progress",
                        "completed_member_count": sum(
                            existing_member_valid(item, members.get(item["member_id"], {}))
                            for item in selected
                        ),
                        "expected_member_count": len(selected),
                        "updated_utc": utc_now(),
                    }
                )
                write_json(state_path, state)
    completed = [
        members[row["member_id"]]
        for row in selected
        if existing_member_valid(row, members.get(row["member_id"], {}))
    ]
    if len(completed) != len(selected):
        raise E16Stop("Paused_Resumable", "not all sidecar members are complete")
    state.update(
        {
            "status": "completed",
            "completed_member_count": len(completed),
            "expected_member_count": len(selected),
            "compressed_bytes": sum(row["compressed_size"] for row in completed),
            "uncompressed_bytes": sum(row["uncompressed_size"] for row in completed),
            "final_layout_json_parse_count": 0,
            "final_camera_pose_numeric_parse_count": 0,
            "final_annotation_semantic_parse_count": 0,
            "completed_utc": utc_now(),
        }
    )
    write_json(state_path, state)
    write_json(
        OUT / "sealed_sidecar_integrity_manifest.json",
        {
            "registration_id": REGISTRATION_ID,
            "status": "Ready_Sealed_GT_Sidecars",
            "member_count": len(completed),
            "layout_count": sum(row["sidecar_name"] == "layout.json" for row in completed),
            "camera_pose_count": sum(
                row["sidecar_name"] == "camera_pose.txt" for row in completed
            ),
            "receipts": completed,
            "sealed_audit": {
                "layout_json_parse_count": 0,
                "camera_pose_numeric_parse_count": 0,
                "annotation_semantic_parse_count": 0,
            },
            "completed_utc": utc_now(),
        },
    )
    return state


def write_stop(stop: E16Stop, phase: str) -> None:
    write_json(
        OUT / "sidecar_stage_summary.json",
        {
            "registration_id": REGISTRATION_ID,
            "status": stop.verdict,
            "phase": phase,
            "reason": str(stop),
            "final_layout_json_parse_count": 0,
            "final_camera_pose_numeric_parse_count": 0,
            "final_annotation_semantic_parse_count": 0,
            "updated_utc": utc_now(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase", choices=("map", "acquire"), default="map"
    )
    args = parser.parse_args()
    phase = "configuration"
    try:
        config = load_config(args.config)
        OUT.mkdir(parents=True, exist_ok=True)
        phase = "e14_upstream_verification"
        records, upstream = verify_e14(config)
        phase = "sidecar_mapping"
        selected = map_sidecars(config, records)
        if args.phase == "map":
            write_json(
                OUT / "sidecar_stage_summary.json",
                {
                    "registration_id": REGISTRATION_ID,
                    "status": "Ready_For_Targeted_Range",
                    "phase": phase,
                    "e14_upstream": upstream,
                    "sidecar_member_count": len(selected),
                    "final_layout_json_parse_count": 0,
                    "final_camera_pose_numeric_parse_count": 0,
                    "final_annotation_semantic_parse_count": 0,
                    "updated_utc": utc_now(),
                },
            )
            return 0
        phase = "remote_identity_verification"
        verify_remote_identities(config, selected)
        phase = "targeted_range_acquisition"
        state = acquire(config, selected)
        write_json(
            OUT / "sidecar_stage_summary.json",
            {
                "registration_id": REGISTRATION_ID,
                "status": "Ready_Sealed_GT_Sidecars",
                "phase": phase,
                "completed_member_count": state["completed_member_count"],
                "final_layout_json_parse_count": 0,
                "final_camera_pose_numeric_parse_count": 0,
                "final_annotation_semantic_parse_count": 0,
                "updated_utc": utc_now(),
            },
        )
        return 0
    except E16Stop as stop:
        write_stop(stop, phase)
        print(f"{stop.verdict}: {stop}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
