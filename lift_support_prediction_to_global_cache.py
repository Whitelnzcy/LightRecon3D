"""Lift a sparse Stage3 support prediction onto an identical global cache.

The only permitted join is the saved DUSt3R pointmap provenance tuple
``(alignment_view_index, x, y)``.  XYZ nearest-neighbour joins are deliberately
not supported because they can hide duplicate, conflicting or moved samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from global_plane_baselines import load_global_cache, save_result


KEY_BITS = 21
KEY_LIMIT = 1 << KEY_BITS


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_text(value):
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f"expected one scalar string, got shape {value.shape}")
    return str(value.reshape(()).item())


def pack_registry_keys(view_indices, pixel_xy, name):
    view_indices = np.asarray(view_indices, dtype=np.int64).reshape(-1)
    pixel_xy = np.asarray(pixel_xy, dtype=np.int64)
    if pixel_xy.shape != (len(view_indices), 2):
        raise ValueError(
            f"{name}: view indices and pixel coordinates must have shapes "
            f"(N,) and (N,2), got {view_indices.shape} and {pixel_xy.shape}"
        )
    if (
        np.any(view_indices < 0)
        or np.any(view_indices >= KEY_LIMIT)
        or np.any(pixel_xy < 0)
        or np.any(pixel_xy >= KEY_LIMIT)
    ):
        raise ValueError(
            f"{name}: view indices and (x,y) pixels must be in [0, {KEY_LIMIT})"
        )
    return (
        (view_indices << (2 * KEY_BITS))
        | (pixel_xy[:, 0] << KEY_BITS)
        | pixel_xy[:, 1]
    )


def resolve_support_assignments(
    cache_view_indices,
    cache_pixel_xy,
    support_view_indices,
    support_pixel_xy,
    support_labels,
    conflict_policy="drop",
):
    """Return source plane labels in cache order plus explicit join diagnostics."""

    if conflict_policy not in {"drop", "error"}:
        raise ValueError(f"unsupported conflict policy: {conflict_policy}")
    support_labels = np.asarray(support_labels, dtype=np.int32).reshape(-1)
    support_view_indices = np.asarray(support_view_indices).reshape(-1)
    support_pixel_xy = np.asarray(support_pixel_xy)
    if len(support_labels) != len(support_view_indices):
        raise ValueError("support labels and view indices have different lengths")
    if support_pixel_xy.shape != (len(support_labels), 2):
        raise ValueError(
            "support labels and pixel coordinates must have shapes (N,) and "
            f"(N,2), got {support_labels.shape} and {support_pixel_xy.shape}"
        )

    cache_keys = pack_registry_keys(
        cache_view_indices, cache_pixel_xy, "global cache"
    )
    _, cache_counts = np.unique(cache_keys, return_counts=True)
    duplicate_cache_keys = int((cache_counts > 1).sum())
    if duplicate_cache_keys:
        raise ValueError(
            "global cache registry is not one-to-one: "
            f"{duplicate_cache_keys} duplicate (view,x,y) keys"
        )

    positive = support_labels >= 0
    positive_labels = support_labels[positive]
    positive_keys = pack_registry_keys(
        support_view_indices[positive],
        support_pixel_xy[positive],
        "positive support",
    )
    order = np.argsort(positive_keys, kind="stable")
    sorted_keys = positive_keys[order]
    sorted_labels = positive_labels[order]
    if len(sorted_keys):
        starts = np.flatnonzero(
            np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
        )
        ends = np.r_[starts[1:], len(sorted_keys)]
    else:
        starts = np.zeros((0,), dtype=np.int64)
        ends = np.zeros((0,), dtype=np.int64)

    resolved_keys = []
    resolved_labels = []
    conflicting_key_count = 0
    conflicting_record_count = 0
    for start, end in zip(starts, ends):
        labels = np.unique(sorted_labels[start:end])
        if len(labels) == 1:
            resolved_keys.append(int(sorted_keys[start]))
            resolved_labels.append(int(labels[0]))
        else:
            conflicting_key_count += 1
            conflicting_record_count += int(end - start)
    if conflicting_key_count and conflict_policy == "error":
        raise ValueError(
            f"{conflicting_key_count} support pixels have conflicting plane labels"
        )

    resolved_keys = np.asarray(resolved_keys, dtype=np.int64)
    resolved_labels = np.asarray(resolved_labels, dtype=np.int32)
    cache_order = np.argsort(cache_keys, kind="stable")
    sorted_cache_keys = cache_keys[cache_order]
    positions = np.searchsorted(sorted_cache_keys, resolved_keys)
    matched = positions < len(sorted_cache_keys)
    if matched.any():
        matched_positions = positions[matched]
        matched[matched] = (
            sorted_cache_keys[matched_positions] == resolved_keys[matched]
        )

    assignment = np.full(len(cache_keys), -1, dtype=np.int32)
    if matched.any():
        cache_indices = cache_order[positions[matched]]
        assignment[cache_indices] = resolved_labels[matched]

    diagnostics = {
        "coordinate_order": "xy",
        "coordinate_space": "dust3r_aligned_pointmap",
        "join_key": "(alignment_view_index,x,y)",
        "conflict_policy": conflict_policy,
        "cache_points": int(len(cache_keys)),
        "support_records": int(len(support_labels)),
        "positive_support_records": int(positive.sum()),
        "ignored_unassigned_records": int((~positive).sum()),
        "unique_positive_support_keys": int(len(starts)),
        "duplicate_positive_support_records": int(len(positive_keys) - len(starts)),
        "conflicting_support_keys": int(conflicting_key_count),
        "conflicting_support_records": int(conflicting_record_count),
        "resolved_support_keys": int(len(resolved_keys)),
        "matched_support_keys": int(matched.sum()),
        "unmatched_support_keys": int((~matched).sum()),
        "matched_cache_points": int((assignment >= 0).sum()),
    }
    return assignment, diagnostics


def load_support_prediction(path):
    required = {
        "point_plane_ids",
        "alignment_view_indices",
        "pointmap_pixel_xy",
    }
    with np.load(path, allow_pickle=False) as raw:
        missing = sorted(required - set(raw.files))
        if missing:
            raise ValueError(f"support prediction missing fields: {missing}")
        if "pointmap_pixel_coordinate_order" in raw:
            order = scalar_text(raw["pointmap_pixel_coordinate_order"])
            if order != "xy":
                raise ValueError(
                    f"support prediction coordinate order must be xy, got {order!r}"
                )
        if "pointmap_pixel_coordinate_space" in raw:
            space = scalar_text(raw["pointmap_pixel_coordinate_space"])
            if space != "dust3r_aligned_pointmap":
                raise ValueError(
                    "support prediction coordinate space must be "
                    f"dust3r_aligned_pointmap, got {space!r}"
                )
        return {
            "labels": raw["point_plane_ids"].astype(np.int32),
            "view_indices": raw["alignment_view_indices"].astype(np.int32),
            "pixel_xy": raw["pointmap_pixel_xy"].astype(np.int32),
        }


def main():
    parser = argparse.ArgumentParser(
        "Lift Stage3 plane support labels onto the identical DUSt3R cache"
    )
    parser.add_argument("--global_cloud_npz", required=True)
    parser.add_argument("--support_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_key", default="scene")
    parser.add_argument("--method", default="stage2_manual_merge_support")
    parser.add_argument("--min_conf", type=float, default=1.0)
    parser.add_argument(
        "--conflict_policy", choices=("drop", "error"), default="drop"
    )
    parser.add_argument("--min_points_per_plane", type=int, default=3)
    args = parser.parse_args()
    if args.min_points_per_plane < 3:
        parser.error("--min_points_per_plane must be at least 3")

    output_dir = Path(args.output_dir)
    cache = load_global_cache(args.global_cloud_npz, args.min_conf)
    support = load_support_prediction(args.support_npz)
    started = time.perf_counter()
    source_assignment, diagnostics = resolve_support_assignments(
        cache["view_indices"],
        cache["pixel_xy"],
        support["view_indices"],
        support["pixel_xy"],
        support["labels"],
        args.conflict_policy,
    )
    supports = []
    source_plane_ids = []
    dropped_small_planes = []
    for source_plane_id in sorted(
        int(value) for value in np.unique(source_assignment) if value >= 0
    ):
        indices = np.flatnonzero(source_assignment == source_plane_id)
        if len(indices) < args.min_points_per_plane:
            dropped_small_planes.append(
                {"source_plane_id": source_plane_id, "points": int(len(indices))}
            )
            continue
        source_plane_ids.append(source_plane_id)
        supports.append(indices)
    runtime = time.perf_counter() - started

    diagnostics["output_planes"] = int(len(supports))
    diagnostics["output_assigned_points"] = int(sum(map(len, supports)))
    diagnostics["source_plane_ids_by_output_id"] = source_plane_ids
    diagnostics["dropped_small_planes"] = dropped_small_planes
    config = {
        "global_cloud_npz": str(Path(args.global_cloud_npz)),
        "support_npz": str(Path(args.support_npz)),
        "min_conf": args.min_conf,
        "conflict_policy": args.conflict_policy,
        "min_points_per_plane": args.min_points_per_plane,
        "source_plane_ids_by_output_id": source_plane_ids,
        "join_diagnostics": diagnostics,
    }
    row = save_result(
        output_dir,
        args.scene_key,
        cache,
        supports,
        args.method,
        runtime,
        config,
    )
    manifest = {
        **row,
        "global_cloud_npz": str(Path(args.global_cloud_npz)),
        "global_cloud_sha256": file_sha256(args.global_cloud_npz),
        "support_npz": str(Path(args.support_npz)),
        "support_npz_sha256": file_sha256(args.support_npz),
        "min_conf": args.min_conf,
        "join_diagnostics": diagnostics,
    }
    manifest_path = output_dir / "support_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
