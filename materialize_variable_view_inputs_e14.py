"""E14 variable-view clean input materialization.

This entrypoint consumes only sealed E11 RGB provenance. It produces one
method-independent DUSt3R global cache and one raw Stage1 pixel-support file
per scene. It never opens annotation content and never fits planes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np


REGISTRATION_ID = "E14_PREREG_variable_view_clean_input_materialization_20260729_v1"
EXPECTED_DUST3R_SHA256 = (
    "9f194a32e34fc124ed10f64115b8e848b7da87c2d1ed131d34b4806721c19dbd"
)
EXPECTED_SCENE_COUNT = 128
EXPECTED_RGB_COUNT = 3265
EXPECTED_FOUR_VIEW_SCENES = {
    "scene_02465",
    "scene_01698",
    "scene_02008",
    "scene_02208",
}
GLOBAL_CACHE_NAME = "global_cache_e14.npz"
DIRECT_SUPPORT_NAME = "direct_support_e14.npz"


class E14Blocked(RuntimeError):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").casefold()


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    data = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_value(project_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={project_dir}", "-C", str(project_dir), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def git_state(project_dir: Path) -> dict[str, Any]:
    return {
        "commit": git_value(project_dir, "rev-parse", "HEAD"),
        "branch": git_value(project_dir, "branch", "--show-current"),
        "status_porcelain": git_value(project_dir, "status", "--porcelain"),
    }


def _room_for_rgb(scene_root: Path, path: Path) -> str:
    relative = path.relative_to(scene_root)
    parts = relative.parts
    try:
        rendering_index = parts.index("2D_rendering")
    except ValueError as error:
        raise ValueError(f"RGB path lacks 2D_rendering component: {relative}") from error
    if rendering_index + 1 >= len(parts):
        raise ValueError(f"RGB path lacks room below 2D_rendering: {relative}")
    return Path(*parts[: rendering_index + 2]).as_posix()


def select_scene_views(scene_root: Path) -> dict[str, Any]:
    candidates = sorted(
        scene_root.rglob("rgb_rawlight.png"),
        key=lambda path: normalized_path(path.relative_to(scene_root)),
    )
    groups: dict[str, list[Path]] = {}
    for path in candidates:
        room = _room_for_rgb(scene_root, path)
        groups.setdefault(room, []).append(path)
    if not groups:
        raise E14Blocked(
            "Blocked_Insufficient_Views", f"{scene_root.name}: no RGB images"
        )
    ranked = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), normalized_path(item[0])),
    )
    selected_room, room_paths = ranked[0]
    room_paths = sorted(
        room_paths, key=lambda path: normalized_path(path.relative_to(scene_root))
    )
    if len(room_paths) < 4:
        raise E14Blocked(
            "Blocked_Insufficient_Views",
            f"{scene_root.name}: largest room has {len(room_paths)} views",
        )
    selected = room_paths[:5]
    return {
        "scene_id": scene_root.name,
        "selected_room": selected_room,
        "view_count": len(selected),
        "view_ids": [str(index) for index in range(len(selected))],
        "relative_image_paths": [
            path.relative_to(scene_root).as_posix() for path in selected
        ],
        "absolute_image_paths": [str(path.resolve()) for path in selected],
        "room_group_counts": {
            room: len(paths)
            for room, paths in sorted(
                groups.items(), key=lambda item: normalized_path(item[0])
            )
        },
        "all_rgb_count": len(candidates),
    }


def freeze_view_selection(
    scene_ids: list[str], raw_root: Path, rgb_hashes: dict[str, str]
) -> dict[str, Any]:
    records = []
    for scene_id in scene_ids:
        record = select_scene_views(raw_root / scene_id)
        record["input_file_hashes"] = [
            rgb_hashes[normalized_path(path)]
            for path in record["absolute_image_paths"]
        ]
        records.append(record)
    four = [row["scene_id"] for row in records if row["view_count"] == 4]
    five = [row["scene_id"] for row in records if row["view_count"] == 5]
    other = [row["scene_id"] for row in records if row["view_count"] not in {4, 5}]
    if (
        len(five) != 124
        or len(four) != 4
        or set(four) != EXPECTED_FOUR_VIEW_SCENES
        or other
    ):
        raise E14Blocked(
            "Blocked_View_Selection_Mismatch",
            f"expected 124x5 and registered 4x4; got five={len(five)}, "
            f"four={four}, other={other}",
        )
    return {
        "registration_id": REGISTRATION_ID,
        "frozen_utc": utc_now(),
        "scene_count": len(records),
        "five_view_scene_count": len(five),
        "four_view_scene_count": len(four),
        "four_view_scenes": four,
        "records": records,
    }


def _hash_record(record: tuple[str, str]) -> tuple[str, str, int]:
    path, expected = record
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(target)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: {actual} != {expected}")
    return normalized_path(target.resolve()), actual, target.stat().st_size


def verify_e11(
    e11_results: Path, raw_root: Path, workers: int = 8
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    pool = load_json(e11_results / "clean_raw_scene_pool.json")
    records = pool.get("records", [])
    scene_ids = [str(row["scene_id"]) for row in records]
    if (
        len(scene_ids) != EXPECTED_SCENE_COUNT
        or len(set(scene_ids)) != EXPECTED_SCENE_COUNT
        or any(row.get("status") != "Clean_Raw" for row in records)
    ):
        raise E14Blocked(
            "Blocked_E11_Input_Mismatch", "E11 Clean Raw pool is not 128 unique scenes"
        )

    state = load_json(e11_results / "member_download_state.json")
    members = list(state.get("members", {}).values())
    if len(members) != EXPECTED_RGB_COUNT:
        raise E14Blocked(
            "Blocked_E11_Input_Mismatch",
            f"E11 member count {len(members)} != {EXPECTED_RGB_COUNT}",
        )
    expected_scene_set = set(scene_ids)
    if {str(row["scene_id"]) for row in members} != expected_scene_set:
        raise E14Blocked(
            "Blocked_E11_Input_Mismatch", "RGB provenance scene set differs from pool"
        )
    work = [
        (str(row["target_path"]), str(row["uncompressed_sha256"]))
        for row in members
    ]
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            verified = list(executor.map(_hash_record, work))
    except (OSError, ValueError) as error:
        raise E14Blocked("Blocked_E11_Input_Mismatch", str(error)) from error
    rgb_hashes = {path: digest for path, digest, _ in verified}

    sealed = load_json(e11_results / "sealed_annotation_manifest.json")
    sealed_scenes = sealed.get("scenes", {})
    if set(sealed_scenes) != expected_scene_set:
        raise E14Blocked(
            "Blocked_E11_Input_Mismatch", "sealed annotation scene set differs"
        )
    annotation_bytes = 0
    for scene_id in scene_ids:
        row = sealed_scenes[scene_id]
        path = Path(str(row["target_path"]))
        if not path.is_file() or path.stat().st_size != int(row["size"]):
            raise E14Blocked(
                "Blocked_E11_Input_Mismatch",
                f"sealed annotation existence/size mismatch: {scene_id}",
            )
        # Byte hashing is explicitly allowed; content is never decoded.
        if file_sha256(path) != str(row["sha256"]):
            raise E14Blocked(
                "Blocked_E11_Input_Mismatch",
                f"sealed annotation byte hash mismatch: {scene_id}",
            )
        annotation_bytes += path.stat().st_size

    return scene_ids, rgb_hashes, {
        "registration_id": REGISTRATION_ID,
        "verified_utc": utc_now(),
        "scene_count": len(scene_ids),
        "rgb_file_count": len(verified),
        "rgb_total_bytes": sum(size for _, _, size in verified),
        "rgb_hash_mismatch_count": 0,
        "sealed_annotation_count": len(sealed_scenes),
        "sealed_annotation_total_bytes": annotation_bytes,
        "sealed_annotation_hash_mismatch_count": 0,
        "gt_semantic_parse_count": 0,
    }


def checkpoint_inventory(
    checkpoint_search_root: Path, e11_completed_utc: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    priority_suffix = Path(
        "stage1_large_80m_train2048_sharded_v1"
    ) / "best.pt"
    candidates = sorted(
        {
            path.resolve()
            for name in ("best.pt", "last.pt")
            for path in checkpoint_search_root.rglob(name)
        },
        key=normalized_path,
    )
    rows = []
    selected = None
    for path in candidates:
        priority = normalized_path(path).endswith(normalized_path(priority_suffix))
        row: dict[str, Any] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "modified_utc": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).isoformat(),
            "priority_1_path_identity": priority,
            "compatible": False,
        }
        if priority:
            try:
                payload = torch.load(
                    path, map_location="cpu", weights_only=False, mmap=True
                )
                args = payload.get("args", {})
                if not isinstance(args, dict):
                    args = vars(args)
                head = payload.get("head")
                row.update(
                    {
                        "readable": True,
                        "has_head_state": isinstance(head, dict),
                        "epoch": int(payload.get("epoch", -1)),
                        "documented_training_scene_count": 2048,
                        "completed_optimizer_step": int(
                            payload.get("global_step", payload.get("step", -1))
                        ),
                        "saved_args": {
                            key: args.get(key)
                            for key in (
                                "hidden_dim",
                                "num_queries",
                                "decoder_layers",
                                "decoder_heads",
                                "output_size",
                            )
                        },
                    }
                )
                row["compatible"] = bool(row["has_head_state"])
                del payload
                gc.collect()
            except Exception as error:  # inventory must record unreadable candidates
                row.update({"readable": False, "error": repr(error)})
        rows.append(row)
        if priority and row["compatible"] and selected is None:
            selected = row
    inventory = {
        "registration_id": REGISTRATION_ID,
        "generated_utc": utc_now(),
        "search_root": str(checkpoint_search_root.resolve()),
        "e11_completed_utc": e11_completed_utc,
        "candidate_count": len(rows),
        "records": rows,
    }
    if selected is None:
        raise E14Blocked(
            "Blocked_No_Compatible_Stage1_Checkpoint",
            "priority checkpoint was not found with a readable head state",
        )
    selected_payload = {
        "registration_id": REGISTRATION_ID,
        "selection_rule": "priority_1_proven_stage1_large_80m_train2048_sharded_v1_best",
        "selected_path": selected["path"],
        "sha256": file_sha256(selected["path"]),
        "bytes": selected["bytes"],
        "epoch": selected.get("epoch"),
        "documented_training_scene_count": 2048,
        "effectiveness_metric_used": False,
        "e11_output_used": False,
    }
    return inventory, selected_payload


def configure_determinism(config: dict[str, Any]) -> None:
    import random
    import torch

    seed = int(config["determinism"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(
        config["determinism"]["cudnn_benchmark"]
    )
    torch.backends.cudnn.deterministic = bool(
        config["determinism"]["cudnn_deterministic"]
    )
    torch.use_deterministic_algorithms(
        bool(config["determinism"]["torch_deterministic_algorithms"])
    )


def load_models(dust3r_weight: Path, stage1_checkpoint: Path, config: dict[str, Any]):
    import torch

    if file_sha256(dust3r_weight) != EXPECTED_DUST3R_SHA256:
        raise E14Blocked(
            "Blocked_DUSt3R_Weight_Mismatch", "DUSt3R whole-file SHA-256 mismatch"
        )
    configure_determinism(config)
    from models.build_backbone import build_dust3r_backbone
    from export_stage1_pred_support_teacher_npz import load_stage1_head

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_dust3r_backbone(str(dust3r_weight), device=device)
    model.eval()
    head, saved_args, checkpoint = load_stage1_head(
        str(stage1_checkpoint), {}, device
    )
    return model, head, saved_args, checkpoint, device


def pair_indices(view_count: int) -> np.ndarray:
    if view_count not in {4, 5}:
        raise ValueError(f"view_count must be 4 or 5, got {view_count}")
    return np.asarray(
        [(i, j) for i in range(view_count) for j in range(i + 1, view_count)],
        dtype=np.int16,
    )


def stage1_to_global_index_map(
    original_hw: tuple[int, int],
    global_hw: tuple[int, int],
    view_offset: int,
    mask_hw: tuple[int, int] = (128, 128),
    image_size: int = 512,
) -> np.ndarray:
    """Map Stage1 mask pixel centers into the frozen DUSt3R crop.

    Stage1 uses a direct square resize. DUSt3R resizes the original long edge
    to ``image_size`` and center-crops to patch-size multiples. The returned
    flattened indices point into the concatenated global cache; -1 denotes a
    Stage1 center outside the DUSt3R crop.
    """

    original_h, original_w = map(int, original_hw)
    global_h, global_w = map(int, global_hw)
    scale = image_size / max(original_w, original_h)
    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))
    crop_left = (resized_w - global_w) / 2.0
    crop_top = (resized_h - global_h) / 2.0
    mask_h, mask_w = map(int, mask_hw)
    yy, xx = np.indices((mask_h, mask_w), dtype=np.float64)
    original_x = (xx + 0.5) * original_w / mask_w
    original_y = (yy + 0.5) * original_h / mask_h
    gx = np.floor(original_x * scale - crop_left).astype(np.int64)
    gy = np.floor(original_y * scale - crop_top).astype(np.int64)
    valid = (gx >= 0) & (gx < global_w) & (gy >= 0) & (gy < global_h)
    indices = np.full((mask_h, mask_w), -1, dtype=np.int64)
    indices[valid] = int(view_offset) + gy[valid] * global_w + gx[valid]
    return indices


def _global_cache_arrays(global_views: dict[str, dict[str, Any]]):
    points, colors, confidence, view_indices, pixel_xy = [], [], [], [], []
    registry = []
    offset = 0
    for _, view in sorted(
        global_views.items(), key=lambda item: item[1]["alignment_view_index"]
    ):
        xyz = np.asarray(view["points"], dtype=np.float32)
        rgb = np.asarray(view["colors"], dtype=np.uint8)
        conf = np.asarray(view["conf"], dtype=np.float32)
        if xyz.shape[:2] != rgb.shape[:2] or xyz.shape[:2] != conf.shape[:2]:
            raise ValueError("global cache point/color/confidence shape mismatch")
        height, width = xyz.shape[:2]
        yy, xx = np.indices((height, width), dtype=np.int32)
        count = height * width
        points.append(xyz.reshape(-1, 3))
        colors.append(rgb.reshape(-1, 3))
        confidence.append(conf.reshape(-1))
        view_indices.append(
            np.full(count, int(view["alignment_view_index"]), dtype=np.int16)
        )
        pixel_xy.append(np.stack((xx, yy), axis=-1).reshape(-1, 2))
        registry.append(
            {
                "alignment_view_index": int(view["alignment_view_index"]),
                "image_path": str(view["image_path"]),
                "points_hw": [height, width],
                "global_point_offset": offset,
                "global_point_count": count,
            }
        )
        offset += count
    return {
        "points": np.concatenate(points),
        "colors": np.concatenate(colors),
        "confidence": np.concatenate(confidence),
        "view_indices": np.concatenate(view_indices),
        "pixel_xy": np.concatenate(pixel_xy),
        "registry": registry,
    }


def write_global_cache(
    path: Path,
    global_views: dict[str, dict[str, Any]],
    record: dict[str, Any],
    alignment_loss: float,
    dust3r_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    arrays = _global_cache_arrays(global_views)
    registry_json = json.dumps(arrays["registry"], separators=(",", ":"))
    np.savez_compressed(
        path,
        schema_name=np.asarray("lightrecon3d_e14_global_cache_v1"),
        schema_version=np.asarray(1, dtype=np.int16),
        scene_id=np.asarray(record["scene_id"]),
        view_count=np.asarray(record["view_count"], dtype=np.int16),
        view_ids=np.asarray(record["view_ids"]),
        selected_room=np.asarray(record["selected_room"]),
        relative_image_paths=np.asarray(record["relative_image_paths"]),
        input_file_hashes=np.asarray(record["input_file_hashes"]),
        points=arrays["points"],
        colors=arrays["colors"],
        confidence=arrays["confidence"],
        view_indices=arrays["view_indices"],
        pixel_xy=arrays["pixel_xy"],
        pixel_coordinate_order=np.asarray("xy"),
        pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
        dust3r_view_registry_json=np.asarray(registry_json),
        dust3r_global_alignment_loss=np.asarray(alignment_loss, dtype=np.float32),
        dust3r_weight_sha256=np.asarray(dust3r_hash),
        config_sha256=np.asarray(config_hash),
    )
    return arrays


def run_global_alignment(
    paths: list[str], model: Any, device: Any, config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], float, Any]:
    from export_stage3_scene_plane_fusion import run_dust3r_global_alignment

    dust = config["dust3r"]
    args = SimpleNamespace(
        image_size=int(dust["image_size"]),
        scene_graph=str(dust["scene_graph"]),
        batch_size=int(dust["batch_size"]),
        niter=int(dust["alignment_niter"]),
        schedule=str(dust["alignment_schedule"]),
        lr=float(dust["alignment_lr"]),
    )
    return run_dust3r_global_alignment(paths, model, device, args)


def _load_stage1_images(paths: list[str], image_size: int, device: Any):
    from export_custom_image_stage1_support_npz import load_rgb_tensor

    images, original_hws = [], []
    for path in paths:
        image, original_hw = load_rgb_tensor(path, image_size)
        images.append(image.to(device))
        original_hws.append(original_hw)
    return images, np.stack(original_hws).astype(np.int32)


def run_direct_support(
    path: Path,
    record: dict[str, Any],
    global_arrays: dict[str, Any],
    model: Any,
    head: Any,
    saved_args: Any,
    checkpoint: dict[str, Any],
    device: Any,
    stage1_hash: str,
    config_hash: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from export_stage1_pred_support_teacher_npz import predict_stage1
    from train_stage1_plane_masks import build_views

    image_size = int(config["stage1"]["image_size"])
    mask_hw = (
        int(config["stage1"]["mask_height"]),
        int(config["stage1"]["mask_width"]),
    )
    paths = list(record["absolute_image_paths"])
    images, original_hws = _load_stage1_images(paths, image_size, device)
    pairs = pair_indices(len(paths))
    source_views, partner_views, orientations = [], [], []
    labels, confidences, margins, existences = [], [], [], []
    with torch.no_grad():
        for pair_number, (first, second) in enumerate(pairs.tolist()):
            batch = {
                "img": images[first][None],
                "img1": images[first][None],
                "img2": images[second][None],
            }
            view1, view2 = build_views(batch, f"e14_pair_{pair_number:03d}")
            result1, result2 = model(view1, view2)
            for source, partner, orientation, result in (
                (first, second, 1, result1),
                (second, first, 2, result2),
            ):
                prediction, confidence, margin, existence = predict_stage1(
                    head,
                    saved_args,
                    result,
                    images[source][None],
                    mask_hw,
                )
                source_views.append(source)
                partner_views.append(partner)
                orientations.append(orientation)
                labels.append(
                    prediction.detach().cpu().numpy().astype(np.int16, copy=False)
                )
                confidences.append(
                    confidence.detach().cpu().numpy().astype(np.float16, copy=False)
                )
                margins.append(
                    margin.detach().cpu().numpy().astype(np.float16, copy=False)
                )
                existences.append(
                    existence.detach().cpu().numpy().astype(np.float16, copy=False)
                )
            del result1, result2
    global_index_maps = []
    registry = global_arrays["registry"]
    for view_index, original_hw in enumerate(original_hws):
        global_h, global_w = registry[view_index]["points_hw"]
        global_index_maps.append(
            stage1_to_global_index_map(
                tuple(original_hw),
                (global_h, global_w),
                int(registry[view_index]["global_point_offset"]),
                mask_hw=mask_hw,
                image_size=image_size,
            )
        )
    source_views_array = np.asarray(source_views, dtype=np.int16)
    provenance_global = np.stack(global_index_maps)
    np.savez_compressed(
        path,
        schema_name=np.asarray("lightrecon3d_e14_direct_support_v1"),
        schema_version=np.asarray(1, dtype=np.int16),
        scene_id=np.asarray(record["scene_id"]),
        view_count=np.asarray(record["view_count"], dtype=np.int16),
        view_ids=np.asarray(record["view_ids"]),
        selected_room=np.asarray(record["selected_room"]),
        relative_image_paths=np.asarray(record["relative_image_paths"]),
        input_file_hashes=np.asarray(record["input_file_hashes"]),
        original_hws=original_hws,
        stage1_input_hw=np.asarray([image_size, image_size], dtype=np.int32),
        support_mask_hw=np.asarray(mask_hw, dtype=np.int32),
        pair_indices=pairs,
        support_source_view_indices=source_views_array,
        support_partner_view_indices=np.asarray(partner_views, dtype=np.int16),
        support_pair_orientations=np.asarray(orientations, dtype=np.int8),
        support_plane_labels=np.stack(labels),
        support_confidence=np.stack(confidences),
        support_margin=np.stack(margins),
        query_existence=np.stack(existences),
        view_mask_pixel_to_global_point_index=provenance_global,
        pixel_coordinate_order=np.asarray("xy"),
        provenance_convention=np.asarray(
            "stage1_mask_pixel_center_to_dust3r_crop_nearest_floor"
        ),
        stage1_checkpoint_sha256=np.asarray(stage1_hash),
        stage1_checkpoint_epoch=np.asarray(
            int(checkpoint.get("epoch", -1)), dtype=np.int32
        ),
        config_sha256=np.asarray(config_hash),
    )
    return {
        "pair_count": len(pairs),
        "support_record_count": len(source_views),
        "global_index_valid_count": int((provenance_global >= 0).sum()),
        "global_index_invalid_count": int((provenance_global < 0).sum()),
    }


def array_hashes(path: Path, keys: Iterable[str]) -> dict[str, str]:
    result = {}
    with np.load(path, allow_pickle=False) as payload:
        for key in keys:
            value = np.ascontiguousarray(payload[key])
            result[key] = hashlib.sha256(value.tobytes()).hexdigest()
    return result


def validate_scene_outputs(
    global_path: Path,
    support_path: Path,
    expected_record: dict[str, Any],
) -> dict[str, Any]:
    with np.load(global_path, allow_pickle=False) as global_payload:
        if str(global_payload["schema_name"].item()) != "lightrecon3d_e14_global_cache_v1":
            raise ValueError("global cache schema mismatch")
        if int(global_payload["view_count"].item()) != expected_record["view_count"]:
            raise ValueError("global cache view_count mismatch")
        points = global_payload["points"]
        colors = global_payload["colors"]
        confidence = global_payload["confidence"]
        view_indices = global_payload["view_indices"]
        pixel_xy = global_payload["pixel_xy"]
        if not (
            points.shape == colors.shape
            and points.ndim == 2
            and points.shape[1] == 3
            and confidence.shape == (len(points),)
            and view_indices.shape == (len(points),)
            and pixel_xy.shape == (len(points), 2)
        ):
            raise ValueError("global cache array shape mismatch")
        if not np.isfinite(points).all() or not np.isfinite(confidence).all():
            raise ValueError("global cache contains NaN/Inf")
    with np.load(support_path, allow_pickle=False) as support_payload:
        if str(support_payload["schema_name"].item()) != "lightrecon3d_e14_direct_support_v1":
            raise ValueError("direct support schema mismatch")
        view_count = int(support_payload["view_count"].item())
        if view_count != expected_record["view_count"]:
            raise ValueError("direct support view_count mismatch")
        pairs = support_payload["pair_indices"]
        sources = support_payload["support_source_view_indices"]
        labels = support_payload["support_plane_labels"]
        confidence = support_payload["support_confidence"]
        margin = support_payload["support_margin"]
        provenance = support_payload["view_mask_pixel_to_global_point_index"]
        expected_pairs = view_count * (view_count - 1) // 2
        if pairs.shape != (expected_pairs, 2) or len(sources) != expected_pairs * 2:
            raise ValueError("direct support pair/source length mismatch")
        if labels.shape != confidence.shape or labels.shape != margin.shape:
            raise ValueError("direct support dense array shape mismatch")
        if labels.shape[0] != len(sources) or provenance.shape[0] != view_count:
            raise ValueError("direct support provenance length mismatch")
        if not np.isfinite(confidence).all() or not np.isfinite(margin).all():
            raise ValueError("direct support contains NaN/Inf")
        if sources.min() < 0 or sources.max() >= view_count:
            raise ValueError("direct support source view out of range")
    return {
        "global_cache_sha256": file_sha256(global_path),
        "global_cache_bytes": global_path.stat().st_size,
        "global_cache_array_sha256": array_hashes(
            global_path,
            ("points", "confidence", "view_indices", "pixel_xy"),
        ),
        "direct_support_sha256": file_sha256(support_path),
        "direct_support_bytes": support_path.stat().st_size,
        "direct_support_array_sha256": array_hashes(
            support_path,
            (
                "pair_indices",
                "support_source_view_indices",
                "support_plane_labels",
                "view_mask_pixel_to_global_point_index",
            ),
        ),
    }


def materialize_one(
    record: dict[str, Any],
    *,
    global_root: Path,
    support_root: Path,
    staging_root: Path,
    model: Any,
    head: Any,
    saved_args: Any,
    checkpoint: dict[str, Any],
    device: Any,
    dust3r_hash: str,
    stage1_hash: str,
    config_hash: str,
    config: dict[str, Any],
    execution_commit: str,
    technical_retry_count: int = 0,
) -> dict[str, Any]:
    scene_id = record["scene_id"]
    final_global_dir = global_root / scene_id
    final_support_dir = support_root / scene_id
    final_global_path = final_global_dir / GLOBAL_CACHE_NAME
    final_support_path = final_support_dir / DIRECT_SUPPORT_NAME
    if final_global_path.is_file() and final_support_path.is_file():
        integrity = validate_scene_outputs(
            final_global_path, final_support_path, record
        )
        return {
            "scene_id": scene_id,
            "status": "resumed_verified",
            "view_count": record["view_count"],
            "technical_retry_count": technical_retry_count,
            **integrity,
        }
    if final_global_dir.exists() or final_support_dir.exists():
        raise RuntimeError(f"incomplete final output exists for {scene_id}")

    scene_stage = staging_root / f"{scene_id}.tmp"
    if scene_stage.exists():
        shutil.rmtree(scene_stage)
    scene_stage.mkdir(parents=True)
    started_utc = utc_now()
    started = time.perf_counter()
    global_path = scene_stage / GLOBAL_CACHE_NAME
    support_path = scene_stage / DIRECT_SUPPORT_NAME
    global_views, alignment_loss, scene = run_global_alignment(
        record["absolute_image_paths"], model, device, config
    )
    global_arrays = write_global_cache(
        global_path,
        global_views,
        record,
        alignment_loss,
        dust3r_hash,
        config_hash,
    )
    del scene, global_views
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    support_stats = run_direct_support(
        support_path,
        record,
        global_arrays,
        model,
        head,
        saved_args,
        checkpoint,
        device,
        stage1_hash,
        config_hash,
        config,
    )
    integrity = validate_scene_outputs(global_path, support_path, record)
    final_global_dir.parent.mkdir(parents=True, exist_ok=True)
    final_support_dir.parent.mkdir(parents=True, exist_ok=True)
    global_stage_dir = scene_stage / "global"
    support_stage_dir = scene_stage / "support"
    global_stage_dir.mkdir()
    support_stage_dir.mkdir()
    os.replace(global_path, global_stage_dir / GLOBAL_CACHE_NAME)
    os.replace(support_path, support_stage_dir / DIRECT_SUPPORT_NAME)
    os.replace(global_stage_dir, final_global_dir)
    os.replace(support_stage_dir, final_support_dir)
    scene_stage.rmdir()
    return {
        "scene_id": scene_id,
        "status": "completed",
        "view_count": record["view_count"],
        "selected_room": record["selected_room"],
        "relative_image_paths": record["relative_image_paths"],
        "input_image_hashes": record["input_file_hashes"],
        "dust3r_weight_hash": dust3r_hash,
        "stage1_checkpoint_hash": stage1_hash,
        "config_hash": config_hash,
        "execution_commit": execution_commit,
        "technical_retry_count": technical_retry_count,
        "start_utc": started_utc,
        "end_utc": utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "alignment_loss": alignment_loss,
        **support_stats,
        **integrity,
    }


def prepare(args: argparse.Namespace) -> None:
    project_dir = Path(__file__).resolve().parent
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    if config.get("registration_id") != REGISTRATION_ID:
        raise ValueError("configuration registration mismatch")
    e11_results = Path(args.e11_results)
    pool = load_json(e11_results / "clean_raw_scene_pool.json")
    scene_ids, rgb_hashes, upstream = verify_e11(
        e11_results, Path(args.raw_root), workers=args.hash_workers
    )
    views = freeze_view_selection(scene_ids, Path(args.raw_root), rgb_hashes)
    inventory, selected = checkpoint_inventory(
        Path(args.checkpoint_search_root), str(pool["completed_utc"])
    )
    dust3r_hash = file_sha256(args.dust3r_weight)
    if dust3r_hash != EXPECTED_DUST3R_SHA256:
        raise E14Blocked(
            "Blocked_DUSt3R_Weight_Mismatch",
            f"{dust3r_hash} != {EXPECTED_DUST3R_SHA256}",
        )
    state = git_state(project_dir)
    source_paths = [
        project_dir / "materialize_variable_view_inputs_e14.py",
        project_dir / "configs" / "e14_variable_view_materialization.json",
        project_dir / "export_stage3_scene_plane_fusion.py",
        project_dir / "export_stage1_pred_support_teacher_npz.py",
        project_dir / "models" / "plane_mask_head.py",
        project_dir / "dust3r" / "dust3r" / "utils" / "image.py",
    ]
    manifest = {
        "registration_id": REGISTRATION_ID,
        "prepared_utc": utc_now(),
        "project_dir": str(project_dir),
        "git": state,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "source_sha256": {str(path): file_sha256(path) for path in source_paths},
        "dust3r_weight_path": str(Path(args.dust3r_weight).resolve()),
        "dust3r_weight_sha256": dust3r_hash,
        "stage1_checkpoint_path": selected["selected_path"],
        "stage1_checkpoint_sha256": selected["sha256"],
        "view_selection_manifest_sha256": json_sha256(views),
        "upstream_verification_sha256": json_sha256(upstream),
        "gt_semantic_parse_count": 0,
        "forbidden_call_count": 0,
    }
    write_json(artifact_dir / "e11_upstream_verification.json", upstream)
    write_json(artifact_dir / "view_selection_manifest_e14.json", views)
    write_json(artifact_dir / "stage1_checkpoint_inventory_e14.json", inventory)
    write_json(artifact_dir / "selected_stage1_checkpoint_e14.json", selected)
    write_json(artifact_dir / "input_preparation_manifest_e14.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def _load_frozen_manifest(args: argparse.Namespace):
    artifact_dir = Path(args.artifact_dir)
    manifest = load_json(artifact_dir / "input_preparation_manifest_e14.json")
    views = load_json(artifact_dir / "view_selection_manifest_e14.json")
    config = load_json(manifest["config_path"])
    return artifact_dir, manifest, views, config


def smoke(args: argparse.Namespace) -> None:
    artifact_dir, manifest, _, config = _load_frozen_manifest(args)
    historical_root = Path(args.historical_scene_root)
    historical = select_scene_views(historical_root)
    if historical["view_count"] != 5:
        raise RuntimeError("historical smoke room must provide five views")
    historical["input_file_hashes"] = [
        file_sha256(path) for path in historical["absolute_image_paths"]
    ]
    model, head, saved_args, checkpoint, device = load_models(
        Path(manifest["dust3r_weight_path"]),
        Path(manifest["stage1_checkpoint_path"]),
        config,
    )
    smoke_root = artifact_dir / "historical_smoke"
    global_root = smoke_root / "global"
    support_root = smoke_root / "support"
    staging_root = smoke_root / "staging"
    records = []
    for view_count in (5, 4):
        record = dict(historical)
        record["scene_id"] = f"scene_00180_smoke_{view_count}view"
        record["view_count"] = view_count
        record["view_ids"] = record["view_ids"][:view_count]
        record["relative_image_paths"] = record["relative_image_paths"][:view_count]
        record["absolute_image_paths"] = record["absolute_image_paths"][:view_count]
        record["input_file_hashes"] = record["input_file_hashes"][:view_count]
        row = materialize_one(
            record,
            global_root=global_root,
            support_root=support_root,
            staging_root=staging_root,
            model=model,
            head=head,
            saved_args=saved_args,
            checkpoint=checkpoint,
            device=device,
            dust3r_hash=manifest["dust3r_weight_sha256"],
            stage1_hash=manifest["stage1_checkpoint_sha256"],
            config_hash=manifest["config_sha256"],
            config=config,
            execution_commit=git_state(Path(__file__).resolve().parent)["commit"],
        )
        records.append(row)
    write_json(
        artifact_dir / "four_five_view_smoke_test.json",
        {
            "registration_id": REGISTRATION_ID,
            "completed_utc": utc_now(),
            "records": records,
            "passed": len(records) == 2
            and {row["view_count"] for row in records} == {4, 5},
        },
    )


def run(args: argparse.Namespace) -> None:
    artifact_dir, manifest, views, config = _load_frozen_manifest(args)
    project_dir = Path(__file__).resolve().parent
    state = git_state(project_dir)
    if state["status_porcelain"]:
        raise E14Blocked(
            "Blocked_Materialization_Implementation",
            "execution worktree is not clean",
        )
    if file_sha256(manifest["config_path"]) != manifest["config_sha256"]:
        raise E14Blocked("Invalid", "configuration changed after freeze")
    for path, expected in manifest["source_sha256"].items():
        if file_sha256(path) != expected:
            raise E14Blocked("Invalid", f"source changed after freeze: {path}")
    if file_sha256(manifest["dust3r_weight_path"]) != manifest["dust3r_weight_sha256"]:
        raise E14Blocked("Blocked_DUSt3R_Weight_Mismatch", "weight changed")
    if file_sha256(manifest["stage1_checkpoint_path"]) != manifest["stage1_checkpoint_sha256"]:
        raise E14Blocked("Invalid", "Stage1 checkpoint changed after freeze")
    smoke_report = load_json(artifact_dir / "four_five_view_smoke_test.json")
    if not smoke_report.get("passed"):
        raise E14Blocked(
            "Blocked_Materialization_Implementation", "4/5-view smoke did not pass"
        )
    model, head, saved_args, checkpoint, device = load_models(
        Path(manifest["dust3r_weight_path"]),
        Path(manifest["stage1_checkpoint_path"]),
        config,
    )
    execution_commit = state["commit"]
    state_path = artifact_dir / "execution_state_e14.json"
    execution = (
        load_json(state_path)
        if state_path.is_file()
        else {
            "registration_id": REGISTRATION_ID,
            "started_utc": utc_now(),
            "execution_commit": execution_commit,
            "records": [],
        }
    )
    by_scene = {row["scene_id"]: row for row in execution["records"]}
    for index, record in enumerate(views["records"]):
        scene_id = record["scene_id"]
        if scene_id in by_scene and by_scene[scene_id].get("status") in {
            "completed",
            "resumed_verified",
        }:
            continue
        retry = 0
        while True:
            try:
                row = materialize_one(
                    record,
                    global_root=Path(args.global_root),
                    support_root=Path(args.support_root),
                    staging_root=Path(args.staging_root),
                    model=model,
                    head=head,
                    saved_args=saved_args,
                    checkpoint=checkpoint,
                    device=device,
                    dust3r_hash=manifest["dust3r_weight_sha256"],
                    stage1_hash=manifest["stage1_checkpoint_sha256"],
                    config_hash=manifest["config_sha256"],
                    config=config,
                    execution_commit=execution_commit,
                    technical_retry_count=retry,
                )
                by_scene[scene_id] = row
                execution["records"] = [
                    by_scene[item["scene_id"]]
                    for item in views["records"]
                    if item["scene_id"] in by_scene
                ]
                execution["updated_utc"] = utc_now()
                write_json(state_path, execution)
                print(
                    json.dumps(
                        {
                            "scene_index": index,
                            "scene_id": scene_id,
                            "view_count": row["view_count"],
                            "status": row["status"],
                            "duration_seconds": row.get("duration_seconds"),
                        }
                    ),
                    flush=True,
                )
                break
            except (OSError, RuntimeError) as error:
                if isinstance(error, E14Blocked):
                    raise
                if retry >= 2:
                    execution["verdict"] = "Blocked_Incomplete_Input_Materialization"
                    execution["blocked_scene"] = scene_id
                    execution["error"] = repr(error)
                    execution["updated_utc"] = utc_now()
                    write_json(state_path, execution)
                    raise E14Blocked(
                        "Blocked_Incomplete_Input_Materialization",
                        f"{scene_id} failed after two technical retries: {error}",
                    ) from error
                retry += 1
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
    completed = list(by_scene.values())
    four = sum(row["view_count"] == 4 for row in completed)
    five = sum(row["view_count"] == 5 for row in completed)
    verdict = (
        "Ready_Clean_Input_Pool"
        if len(completed) == 128 and four == 4 and five == 124
        else "Blocked_Incomplete_Input_Materialization"
    )
    execution.update(
        {
            "completed_utc": utc_now(),
            "global_cache_completed": len(completed),
            "direct_support_completed": len(completed),
            "provenance_completed": len(completed),
            "four_view_scene_count": four,
            "five_view_scene_count": five,
            "gt_semantic_parse_count": 0,
            "forbidden_call_count": 0,
            "verdict": verdict,
        }
    )
    write_json(state_path, execution)
    print(json.dumps({key: execution[key] for key in (
        "verdict", "global_cache_completed", "direct_support_completed",
        "provenance_completed", "four_view_scene_count",
        "five_view_scene_count", "gt_semantic_parse_count",
        "forbidden_call_count",
    )}, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--e11-results", required=True)
    prepare_parser.add_argument("--raw-root", required=True)
    prepare_parser.add_argument("--checkpoint-search-root", required=True)
    prepare_parser.add_argument("--dust3r-weight", required=True)
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--artifact-dir", required=True)
    prepare_parser.add_argument("--hash-workers", type=int, default=8)
    prepare_parser.set_defaults(function=prepare)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--artifact-dir", required=True)
    smoke_parser.add_argument("--historical-scene-root", required=True)
    smoke_parser.set_defaults(function=smoke)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--artifact-dir", required=True)
    run_parser.add_argument("--global-root", required=True)
    run_parser.add_argument("--support-root", required=True)
    run_parser.add_argument("--staging-root", required=True)
    run_parser.set_defaults(function=run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.function(args)
    except E14Blocked as error:
        print(
            json.dumps(
                {"verdict": error.verdict, "error": str(error)}, indent=2
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
