"""Build Structured3D plane labels on an exact DUSt3R global-cloud registry.

This writer preserves the cached DUSt3R XYZ samples and attaches visible
Structured3D layout-plane identities at the same ``(view_index, x, y)``
locations.  Fitted GT plane parameters therefore live in the DUSt3R global
frame; they evaluate support/identity, not absolute metric reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from global_plane_baselines import global_cache_keep_mask


SCHEMA_VERSION = 1
PLANE_COLORS = np.asarray(
    [
        [230, 57, 70], [29, 53, 87], [69, 123, 157], [42, 157, 143],
        [233, 196, 106], [244, 162, 97], [154, 95, 156], [118, 200, 147],
    ],
    dtype=np.uint8,
)


def path_key(path):
    return os.path.normpath(str(path).strip()).replace("\\", "/")


def parse_path_prefix_maps(values):
    mappings = []
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"Bad path prefix map {value!r}; expected SOURCE=DESTINATION")
        source, destination = value.split("=", 1)
        source = path_key(source).rstrip("/")
        destination = str(destination).strip().rstrip("/\\")
        if not source or not destination:
            raise ValueError("Path prefix source and destination must be non-empty")
        mappings.append((source, destination))
    return tuple(sorted(mappings, key=lambda row: len(row[0]), reverse=True))


def remap_path(path, mappings):
    normalized = path_key(path)
    for source, destination in mappings:
        if normalized == source:
            return destination
        if normalized.startswith(source + "/"):
            return destination + normalized[len(source):]
    return str(path)


def dust3r_resize_crop_transform(original_hw, image_size=512, patch_size=16, square_ok=False):
    """Reproduce the geometry of vendored DUSt3R ``load_images``."""

    original_h, original_w = map(int, original_hw)
    if original_h <= 0 or original_w <= 0:
        raise ValueError(f"Invalid original image shape {original_hw}")
    if int(image_size) == 224:
        long_edge = round(image_size * max(original_w / original_h, original_h / original_w))
    else:
        long_edge = int(image_size)
    scale = long_edge / max(original_w, original_h)
    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))
    center_x, center_y = resized_w // 2, resized_h // 2
    if int(image_size) == 224:
        half_w = half_h = min(center_x, center_y)
    else:
        half_w = ((2 * center_x) // int(patch_size)) * int(patch_size) / 2
        half_h = ((2 * center_y) // int(patch_size)) * int(patch_size) / 2
        if not square_ok and resized_w == resized_h:
            half_h = 3 * half_w / 4
    left = int(round(center_x - half_w))
    top = int(round(center_y - half_h))
    right = int(round(center_x + half_w))
    bottom = int(round(center_y + half_h))
    return {
        "original_hw": [original_h, original_w],
        "resized_hw": [resized_h, resized_w],
        "crop_xyxy": [left, top, right, bottom],
        "pointmap_hw": [bottom - top, right - left],
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "square_ok": bool(square_ok),
    }


def render_layout_plane_ids(layout, original_hw):
    height, width = map(int, original_hw)
    mask = np.full((height, width), -1, dtype=np.int32)
    junctions = layout.get("junctions", [])
    for plane in layout.get("planes", []):
        if plane.get("ID") is None:
            continue
        plane_id = int(plane["ID"])
        for polygon_indices in plane.get("visible_mask", []):
            points = [
                junctions[int(index)]["coordinate"]
                for index in polygon_indices
                if 0 <= int(index) < len(junctions)
            ]
            if len(points) >= 3:
                polygon = np.asarray(points, dtype=np.int32)
                cv2.fillPoly(mask, [polygon], color=plane_id)
    return mask


def preprocess_plane_mask(mask, transform, boundary_ignore_radius=1):
    resized_h, resized_w = transform["resized_hw"]
    left, top, right, bottom = transform["crop_xyxy"]
    resized = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
    cropped = resized[top:bottom, left:right].copy()
    if list(cropped.shape) != transform["pointmap_hw"]:
        raise RuntimeError(
            f"Preprocessed mask shape {cropped.shape} != {transform['pointmap_hw']}"
        )
    radius = int(boundary_ignore_radius)
    if radius > 0:
        boundary = np.zeros(cropped.shape, dtype=np.uint8)
        boundary[1:] |= cropped[1:] != cropped[:-1]
        boundary[:-1] |= cropped[:-1] != cropped[1:]
        boundary[:, 1:] |= cropped[:, 1:] != cropped[:, :-1]
        boundary[:, :-1] |= cropped[:, :-1] != cropped[:, 1:]
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        boundary = cv2.dilate(boundary, kernel)
        cropped[boundary.astype(bool)] = -1
    return cropped


def fit_plane(points):
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0:
        normal = -normal
    return normal.astype(np.float32), -float(normal @ center)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_colored_ply(path, points, colors, labels, max_points=100000):
    display = np.asarray(colors, dtype=np.uint8).copy()
    for plane_id in np.unique(labels):
        if plane_id >= 0:
            display[labels == plane_id] = PLANE_COLORS[int(plane_id) % len(PLANE_COLORS)]
    indices = np.linspace(
        0, len(points) - 1, min(len(points), int(max_points)), dtype=np.int64
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write(f"ply\nformat ascii 1.0\nelement vertex {len(indices)}\n")
        handle.write(
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        np.savetxt(
            handle,
            np.column_stack((points[indices], display[indices])),
            fmt="%.7g %.7g %.7g %d %d %d",
        )


def build_point_aligned_gt(
    global_cloud_npz,
    output_npz,
    *,
    min_conf=1.0,
    min_plane_points=64,
    image_size=512,
    patch_size=16,
    square_ok=False,
    boundary_ignore_radius=1,
    path_prefix_maps=(),
    output_ply=None,
    max_display_points=100000,
):
    raw = np.load(global_cloud_npz, allow_pickle=False)
    required = {
        "points", "colors", "confidence", "view_indices", "pixel_xy",
        "dust3r_view_registry_json", "pixel_coordinate_order", "pixel_coordinate_space",
    }
    missing = sorted(required - set(raw.files))
    if missing:
        raise ValueError(f"Global cloud cache missing fields: {missing}")
    if str(raw["pixel_coordinate_order"].item()) != "xy":
        raise ValueError("Global cloud cache pixel order must be xy")
    if str(raw["pixel_coordinate_space"].item()) != "dust3r_aligned_pointmap":
        raise ValueError("Global cloud cache pixels must be DUSt3R aligned-pointmap pixels")

    source_points = raw["points"].astype(np.float32)
    keep = global_cache_keep_mask(source_points, raw["confidence"], min_conf)
    source_indices = np.flatnonzero(keep)
    points = source_points[keep]
    colors = raw["colors"][keep].astype(np.uint8)
    confidence = raw["confidence"][keep].astype(np.float32)
    view_indices = raw["view_indices"][keep].astype(np.int32)
    pixel_xy = raw["pixel_xy"][keep].astype(np.int32)
    raw_labels = np.full(len(points), -1, dtype=np.int32)

    registry = json.loads(str(raw["dust3r_view_registry_json"].item()))
    records = {int(row["alignment_view_index"]): row for row in registry}
    transforms = []
    layout_paths = []
    for view_index in sorted(int(value) for value in np.unique(view_indices)):
        if view_index not in records:
            raise ValueError(f"View {view_index} is missing from the cache registry")
        image_path = Path(remap_path(records[view_index]["image_path"], path_prefix_maps))
        layout_path = image_path.with_name("layout.json")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read registry RGB image: {image_path}")
        if not layout_path.is_file():
            raise FileNotFoundError(f"Cannot find Structured3D layout: {layout_path}")
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        original_hw = image.shape[:2]
        transform = dust3r_resize_crop_transform(
            original_hw, image_size=image_size, patch_size=patch_size, square_ok=square_ok
        )
        expected_hw = list(map(int, records[view_index]["points_hw"]))
        if transform["pointmap_hw"] != expected_hw:
            raise ValueError(
                f"View {view_index} transform gives {transform['pointmap_hw']}, "
                f"but cache registry records {expected_hw}"
            )
        processed_mask = preprocess_plane_mask(
            render_layout_plane_ids(layout, original_hw),
            transform,
            boundary_ignore_radius=boundary_ignore_radius,
        )
        mask = view_indices == view_index
        xy = pixel_xy[mask]
        if not (
            (xy[:, 0] >= 0).all()
            and (xy[:, 0] < expected_hw[1]).all()
            and (xy[:, 1] >= 0).all()
            and (xy[:, 1] < expected_hw[0]).all()
        ):
            raise ValueError(f"Cache pixels are out of range for view {view_index}")
        raw_labels[mask] = processed_mask[xy[:, 1], xy[:, 0]]
        transforms.append({
            "alignment_view_index": view_index,
            "image_path": str(image_path),
            "layout_path": str(layout_path),
            **transform,
        })
        layout_paths.append(str(layout_path))

    source_plane_ids = []
    point_plane_ids = np.full(len(points), -1, dtype=np.int32)
    normals, offsets, counts = [], [], []
    for source_id in sorted(int(value) for value in np.unique(raw_labels) if value >= 0):
        mask = raw_labels == source_id
        if int(mask.sum()) < int(min_plane_points):
            continue
        plane_id = len(source_plane_ids)
        point_plane_ids[mask] = plane_id
        normal, offset = fit_plane(points[mask])
        source_plane_ids.append(source_id)
        normals.append(normal)
        offsets.append(offset)
        counts.append(int(mask.sum()))
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    offsets = np.asarray(offsets, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int32)

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    checksum = file_sha256(global_cloud_npz)
    np.savez_compressed(
        output_npz,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        points=points,
        colors=colors,
        confidence=confidence,
        view_indices=view_indices,
        pixel_xy=pixel_xy,
        source_cache_indices=source_indices.astype(np.int64),
        point_plane_ids=point_plane_ids,
        raw_structured3d_plane_ids=raw_labels,
        plane_ids=np.arange(len(normals), dtype=np.int32),
        source_gt_plane_ids=np.asarray(source_plane_ids, dtype=np.int32),
        plane_normals=normals,
        plane_offsets=offsets,
        plane_inlier_counts=counts,
        pixel_coordinate_order=np.asarray("xy"),
        pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
        geometry_interpretation=np.asarray("support_labels_in_dust3r_global_frame"),
        source_global_cloud_npz=np.asarray(str(global_cloud_npz)),
        source_global_cloud_sha256=np.asarray(checksum),
        min_conf=np.asarray(float(min_conf), dtype=np.float32),
        min_plane_points=np.asarray(int(min_plane_points), dtype=np.int32),
        view_transforms_json=np.asarray(json.dumps(transforms)),
        source_layout_paths_json=np.asarray(json.dumps(layout_paths)),
        scene_key=np.asarray(str(raw["scene_key"].item()) if "scene_key" in raw else "scene"),
    )
    if output_ply is not None:
        write_colored_ply(
            Path(output_ply), points, colors, point_plane_ids, max_points=max_display_points
        )
    return {
        "output_npz": str(output_npz),
        "output_ply": str(output_ply) if output_ply is not None else "",
        "source_global_cloud_npz": str(global_cloud_npz),
        "source_global_cloud_sha256": checksum,
        "points": int(len(points)),
        "labeled_points": int((point_plane_ids >= 0).sum()),
        "planes": int(len(normals)),
        "source_gt_plane_ids": source_plane_ids,
        "views": int(len(transforms)),
        "min_conf": float(min_conf),
        "boundary_ignore_radius": int(boundary_ignore_radius),
    }


def main():
    parser = argparse.ArgumentParser("Build point-aligned Structured3D plane GT")
    parser.add_argument("--global_cloud_npz", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_ply", default="")
    parser.add_argument("--output_manifest", default="")
    parser.add_argument("--min_conf", type=float, default=1.0)
    parser.add_argument("--min_plane_points", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--square_ok", action="store_true")
    parser.add_argument("--boundary_ignore_radius", type=int, default=1)
    parser.add_argument("--max_display_points", type=int, default=100000)
    parser.add_argument("--path_prefix_map", action="append", default=[])
    args = parser.parse_args()
    try:
        mappings = parse_path_prefix_maps(args.path_prefix_map)
    except ValueError as error:
        parser.error(str(error))
    row = build_point_aligned_gt(
        args.global_cloud_npz,
        args.output_npz,
        min_conf=args.min_conf,
        min_plane_points=args.min_plane_points,
        image_size=args.image_size,
        patch_size=args.patch_size,
        square_ok=args.square_ok,
        boundary_ignore_radius=args.boundary_ignore_radius,
        path_prefix_maps=mappings,
        output_ply=args.output_ply or None,
        max_display_points=args.max_display_points,
    )
    manifest = Path(args.output_manifest) if args.output_manifest else Path(args.output_npz).with_suffix(".json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps({**row, "manifest": str(manifest)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
