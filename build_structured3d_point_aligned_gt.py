"""Build Structured3D plane labels and metric rays on a DUSt3R cache registry.

This writer preserves the cached DUSt3R XYZ samples and attaches visible
Structured3D layout-plane identities at the same ``(view_index, x, y)``
locations.  Fitted plane parameters in ``plane_normals`` still live in the
DUSt3R global frame for identity evaluation.  When ``camera_pose.txt`` and the
camera-coordinate plane equations in ``layout.json`` are available, the same
pixels are also intersected with their GT structural planes and transformed to
Structured3D world coordinates in metres.
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


SCHEMA_VERSION = 2
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


def parse_camera_pose(path):
    """Parse Structured3D's eye, view direction, up and half-FOV record."""

    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if values.shape != (12,):
        raise ValueError(f"Expected 12 camera-pose values in {path}, got {values.shape}")
    position = values[:3]
    forward = values[3:6]
    up_hint = values[6:9]
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-12:
        raise ValueError(f"Camera view direction is degenerate in {path}")
    forward = forward / forward_norm
    up_hint = up_hint - forward * float(up_hint @ forward)
    up_norm = float(np.linalg.norm(up_hint))
    if up_norm <= 1e-12:
        raise ValueError(f"Camera up direction is degenerate in {path}")
    up_hint /= up_norm
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    camera_to_world_basis = np.column_stack((right, up, forward))
    determinant = float(np.linalg.det(camera_to_world_basis))
    # Structured3D's perspective camera coordinates are left-handed
    # (+x right, +y up, +z view direction), while its world frame is
    # right-handed.  The camera-to-world basis is therefore an improper
    # orthogonal matrix with determinant -1.
    if abs(abs(determinant) - 1.0) > 1e-6:
        raise ValueError(f"Camera basis is not orthogonal in {path}")
    return {
        "position_mm": position,
        "camera_to_world_basis": camera_to_world_basis,
        "camera_basis_determinant": determinant,
        "half_fov_rad": values[9:11],
        "trailing_value": float(values[11]),
    }


def pointmap_pixels_to_original_xy(pixel_xy, transform):
    """Invert DUSt3R's PIL resize and integer crop using pixel centres."""

    pixel_xy = np.asarray(pixel_xy, dtype=np.float64)
    if pixel_xy.ndim != 2 or pixel_xy.shape[1] != 2:
        raise ValueError(f"pixel_xy must have shape (N,2), got {pixel_xy.shape}")
    original_h, original_w = map(float, transform["original_hw"])
    resized_h, resized_w = map(float, transform["resized_hw"])
    left, top, _, _ = map(float, transform["crop_xyxy"])
    resized_x = pixel_xy[:, 0] + left
    resized_y = pixel_xy[:, 1] + top
    original_x = (resized_x + 0.5) * original_w / resized_w - 0.5
    original_y = (resized_y + 0.5) * original_h / resized_h - 0.5
    return np.column_stack((original_x, original_y))


def camera_rays_from_original_xy(original_xy, original_hw, half_fov_rad):
    """Return Structured3D camera rays with +x right, +y up and +z forward."""

    original_xy = np.asarray(original_xy, dtype=np.float64)
    height, width = map(int, original_hw)
    if height <= 1 or width <= 1:
        raise ValueError(f"Original image is too small for calibrated rays: {original_hw}")
    center_x = 0.5 * (width - 1)
    center_y = 0.5 * (height - 1)
    x = (original_xy[:, 0] - center_x) / center_x * np.tan(float(half_fov_rad[0]))
    y = -(original_xy[:, 1] - center_y) / center_y * np.tan(float(half_fov_rad[1]))
    return np.column_stack((x, y, np.ones(len(original_xy), dtype=np.float64)))


def intersect_camera_rays_with_plane(rays, normal, offset, epsilon=1e-9):
    """Intersect rays with Structured3D's camera plane ``n.x + d = 0``."""

    rays = np.asarray(rays, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    denominator = rays @ normal
    distance = np.full(len(rays), np.nan, dtype=np.float64)
    valid = np.abs(denominator) > float(epsilon)
    distance[valid] = -float(offset) / denominator[valid]
    valid &= np.isfinite(distance) & (distance > 0.0)
    points = rays * distance[:, None]
    points[~valid] = np.nan
    return points, valid


def camera_plane_to_world(normal, offset, camera_pose):
    """Transform ``n_c.x_c + d_c = 0`` to Structured3D world coordinates."""

    rotation = camera_pose["camera_to_world_basis"]
    position = camera_pose["position_mm"]
    world_normal = rotation @ np.asarray(normal, dtype=np.float64).reshape(3)
    world_normal /= max(float(np.linalg.norm(world_normal)), 1e-12)
    world_offset = float(offset) - float(world_normal @ position)
    return world_normal, world_offset


def find_scene_annotation(image_path):
    for parent in (Path(image_path), *Path(image_path).parents):
        candidate = parent / "annotation_3d.json"
        if candidate.is_file():
            return candidate
    return None


def plane_consistency_row(layout_plane, world_plane, camera_pose):
    normal, offset = camera_plane_to_world(
        layout_plane["normal"], layout_plane["offset"], camera_pose
    )
    expected_normal = np.asarray(world_plane["normal"], dtype=np.float64)
    expected_normal /= max(float(np.linalg.norm(expected_normal)), 1e-12)
    expected_offset = float(world_plane["offset"])
    if float(normal @ expected_normal) < 0:
        normal = -normal
        offset = -offset
    cosine = np.clip(float(normal @ expected_normal), -1.0, 1.0)
    return {
        "plane_id": int(layout_plane["ID"]),
        "normal_error_deg": float(np.degrees(np.arccos(cosine))),
        "offset_error_mm": abs(float(offset) - expected_offset),
    }


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
    output_metric_ply=None,
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
    metric_points_camera_mm = np.full((len(points), 3), np.nan, dtype=np.float64)
    metric_points_world_mm = np.full((len(points), 3), np.nan, dtype=np.float64)
    metric_valid = np.zeros(len(points), dtype=bool)

    registry = json.loads(str(raw["dust3r_view_registry_json"].item()))
    scene_key = str(raw["scene_key"].item()) if "scene_key" in raw else "scene"
    raw.close()
    records = {int(row["alignment_view_index"]): row for row in registry}
    transforms = []
    layout_paths = []
    camera_view_indices = []
    camera_to_world_m = []
    camera_half_fov_rad = []
    plane_consistency = []
    annotation_path = None
    annotation_planes = None
    for view_index in sorted(int(value) for value in np.unique(view_indices)):
        if view_index not in records:
            raise ValueError(f"View {view_index} is missing from the cache registry")
        image_path = Path(remap_path(records[view_index]["image_path"], path_prefix_maps))
        layout_path = image_path.with_name("layout.json")
        camera_pose_path = image_path.with_name("camera_pose.txt")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read registry RGB image: {image_path}")
        if not layout_path.is_file():
            raise FileNotFoundError(f"Cannot find Structured3D layout: {layout_path}")
        if not camera_pose_path.is_file():
            raise FileNotFoundError(f"Cannot find Structured3D camera pose: {camera_pose_path}")
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        camera_pose = parse_camera_pose(camera_pose_path)
        if annotation_path is None:
            annotation_path = find_scene_annotation(image_path)
            if annotation_path is not None:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                annotation_planes = {
                    int(row["ID"]): row for row in annotation.get("planes", [])
                }
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
        original_xy = pointmap_pixels_to_original_xy(xy, transform)
        rays = camera_rays_from_original_xy(
            original_xy, original_hw, camera_pose["half_fov_rad"]
        )
        local_indices = np.flatnonzero(mask)
        layout_by_id = {
            int(plane["ID"]): plane
            for plane in layout.get("planes", [])
            if plane.get("ID") is not None
        }
        for plane_id in sorted(int(value) for value in np.unique(raw_labels[mask]) if value >= 0):
            if plane_id not in layout_by_id:
                raise ValueError(f"View {view_index} has mask plane {plane_id} without an equation")
            selected_local = raw_labels[mask] == plane_id
            camera_points, valid_intersection = intersect_camera_rays_with_plane(
                rays[selected_local],
                layout_by_id[plane_id]["normal"],
                layout_by_id[plane_id]["offset"],
            )
            selected_global = local_indices[selected_local]
            metric_points_camera_mm[selected_global] = camera_points
            world_points = (
                camera_points @ camera_pose["camera_to_world_basis"].T
                + camera_pose["position_mm"]
            )
            metric_points_world_mm[selected_global] = world_points
            metric_valid[selected_global] = valid_intersection
        if annotation_planes is not None:
            for plane in layout.get("planes", []):
                plane_id = plane.get("ID")
                if plane_id is not None and int(plane_id) in annotation_planes:
                    row = plane_consistency_row(
                        plane, annotation_planes[int(plane_id)], camera_pose
                    )
                    row["alignment_view_index"] = view_index
                    plane_consistency.append(row)
        camera_matrix = np.eye(4, dtype=np.float64)
        camera_matrix[:3, :3] = camera_pose["camera_to_world_basis"]
        camera_matrix[:3, 3] = camera_pose["position_mm"] * 0.001
        camera_view_indices.append(view_index)
        camera_to_world_m.append(camera_matrix)
        camera_half_fov_rad.append(camera_pose["half_fov_rad"])
        transforms.append({
            "alignment_view_index": view_index,
            "image_path": str(image_path),
            "layout_path": str(layout_path),
            "camera_pose_path": str(camera_pose_path),
            "pixel_inverse_mapping": "pillow_half_pixel_after_integer_crop",
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
    metric_valid &= point_plane_ids >= 0

    world_plane_normals, world_plane_offsets_m = [], []
    if annotation_planes is None:
        raise FileNotFoundError(
            f"Cannot find scene annotation_3d.json above registry image {registry[0]['image_path']}"
        )
    for source_id in source_plane_ids:
        if source_id not in annotation_planes:
            raise ValueError(f"Structured3D world annotation has no plane ID {source_id}")
        plane = annotation_planes[source_id]
        normal = np.asarray(plane["normal"], dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        world_plane_normals.append(normal)
        world_plane_offsets_m.append(float(plane["offset"]) * 0.001)
    world_plane_normals = np.asarray(world_plane_normals, dtype=np.float32).reshape(-1, 3)
    world_plane_offsets_m = np.asarray(world_plane_offsets_m, dtype=np.float32)

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
        metric_valid_mask=metric_valid,
        metric_points_camera_m=(metric_points_camera_mm * 0.001).astype(np.float32),
        metric_points_world_m=(metric_points_world_mm * 0.001).astype(np.float32),
        plane_ids=np.arange(len(normals), dtype=np.int32),
        source_gt_plane_ids=np.asarray(source_plane_ids, dtype=np.int32),
        plane_normals=normals,
        plane_offsets=offsets,
        plane_inlier_counts=counts,
        structured3d_world_plane_normals=world_plane_normals,
        structured3d_world_plane_offsets_m=world_plane_offsets_m,
        camera_view_indices=np.asarray(camera_view_indices, dtype=np.int32),
        camera_to_world_m=np.asarray(camera_to_world_m, dtype=np.float64),
        camera_half_fov_rad=np.asarray(camera_half_fov_rad, dtype=np.float64),
        pixel_coordinate_order=np.asarray("xy"),
        pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
        geometry_interpretation=np.asarray(
            "identity_in_dust3r_frame_plus_metric_structural_rays_in_structured3d_world"
        ),
        metric_length_unit=np.asarray("metre"),
        structured3d_plane_equation=np.asarray("normal_dot_point_plus_offset_equals_zero"),
        source_global_cloud_npz=np.asarray(str(global_cloud_npz)),
        source_global_cloud_sha256=np.asarray(checksum),
        min_conf=np.asarray(float(min_conf), dtype=np.float32),
        min_plane_points=np.asarray(int(min_plane_points), dtype=np.int32),
        view_transforms_json=np.asarray(json.dumps(transforms)),
        source_layout_paths_json=np.asarray(json.dumps(layout_paths)),
        source_annotation_3d_path=np.asarray(str(annotation_path)),
        plane_consistency_json=np.asarray(json.dumps(plane_consistency)),
        scene_key=np.asarray(scene_key),
    )
    if output_ply is not None:
        write_colored_ply(
            Path(output_ply), points, colors, point_plane_ids, max_points=max_display_points
        )
    if output_metric_ply is not None:
        write_colored_ply(
            Path(output_metric_ply),
            (metric_points_world_mm[metric_valid] * 0.001).astype(np.float32),
            colors[metric_valid],
            point_plane_ids[metric_valid],
            max_points=max_display_points,
        )
    return {
        "output_npz": str(output_npz),
        "output_ply": str(output_ply) if output_ply is not None else "",
        "output_metric_ply": (
            str(output_metric_ply) if output_metric_ply is not None else ""
        ),
        "source_global_cloud_npz": str(global_cloud_npz),
        "source_global_cloud_sha256": checksum,
        "points": int(len(points)),
        "labeled_points": int((point_plane_ids >= 0).sum()),
        "metric_points": int(metric_valid.sum()),
        "planes": int(len(normals)),
        "source_gt_plane_ids": source_plane_ids,
        "views": int(len(transforms)),
        "min_conf": float(min_conf),
        "boundary_ignore_radius": int(boundary_ignore_radius),
        "source_annotation_3d_path": str(annotation_path),
        "max_plane_normal_consistency_error_deg": float(
            max((row["normal_error_deg"] for row in plane_consistency), default=0.0)
        ),
        "max_plane_offset_consistency_error_mm": float(
            max((row["offset_error_mm"] for row in plane_consistency), default=0.0)
        ),
    }


def main():
    parser = argparse.ArgumentParser("Build point-aligned Structured3D plane GT")
    parser.add_argument("--global_cloud_npz", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_ply", default="")
    parser.add_argument("--output_metric_ply", default="")
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
        output_metric_ply=args.output_metric_ply or None,
        max_display_points=args.max_display_points,
    )
    manifest = Path(args.output_manifest) if args.output_manifest else Path(args.output_npz).with_suffix(".json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps({**row, "manifest": str(manifest)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
