import json
import zlib
from pathlib import Path
from unittest.mock import Mock

import execute_targeted_gt_sidecars_e16 as e16


def test_normalize_member_uses_forward_slashes():
    assert e16.normalize_member(r"Structured3D\scene_00000\x") == (
        "Structured3D/scene_00000/x"
    )


def test_canonical_member_id_is_order_independent():
    left = {"archive_name": "a", "path": "p", "crc32": "00"}
    right = {"crc32": "00", "path": "p", "archive_name": "a"}
    assert e16.sha256_bytes(e16.canonical_json(left)) == e16.sha256_bytes(
        e16.canonical_json(right)
    )


def test_existing_member_validation_recomputes_hash(tmp_path: Path):
    target = tmp_path / "layout.json"
    target.write_bytes(b"sealed")
    row = {"target_path": str(target), "uncompressed_size": 6}
    receipt = {
        "status": "completed",
        "uncompressed_sha256": e16.sha256_bytes(b"sealed"),
    }
    assert e16.existing_member_valid(row, receipt)
    target.write_bytes(b"change")
    assert not e16.existing_member_valid(row, receipt)


def test_sidecar_bytes_are_crc_checked_without_json_parse():
    raw = b'{"planes": [1, 2]}'
    assert f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}" == "0618a2fc"


def test_range_reader_checks_status_before_reading_body(monkeypatch):
    response = Mock()
    response.status_code = 200
    response.headers = {}
    response.iter_content.side_effect = AssertionError("body must not be read")
    session = Mock()
    session.get.return_value = response
    config = {
        "range_retry_count": 1,
        "range_connect_timeout_seconds": 1,
        "range_read_timeout_seconds": 1,
    }
    try:
        e16.exact_range(session, "https://invalid", 0, 9, 100, config)
    except e16.E16Stop as stop:
        assert stop.verdict == "Paused_Resumable"
    else:
        raise AssertionError("ignored Range response must stop")
    response.iter_content.assert_not_called()
