#!/usr/bin/env python3
"""Map Plane-DUSt3R room planes onto LightRecon3D's ordered point cache.

The two methods reconstruct different 3D coordinate frames.  Same-input
comparison therefore joins predictions by the stable key
``(alignment_view_index, x, y)`` and refits every predicted plane on the
LightRecon3D cache points.  No nearest-neighbour spatial correspondence is
invented.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


POINTMAP_HW = (288, 512)
RENDER_HW = (360, 640)
ORIGINAL_HW = (720, 1280)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_official_reconstruction(runtime_dir: Path) -> Callable[..., Any]:
    """Load only the pinned runtime's already-audited reconstruction function."""
    runtime_dir = runtime_dir.resolve()
    metric_path = runtime_dir / "metric.py"
    if not metric_path.is_file():
        raise FileNotFoundError(f"missing audited Plane-DUSt3R metric.py: {metric_path}")
    sys.path.insert(0, str(runtime_dir))
    try:
        specification = importlib.util.spec_from_file_location(
            "_plane_dust3r_audited_metric", metric_path
        )
        if specification is None or specification.loader is None:
            raise ImportError(f"cannot load {metric_path}")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
    finally:
        sys.path.remove(str(runtime_dir))
    return module.reconstruction


def closest_reference_pose(poses: np.ndarray) -> int:
    identity = np.eye(3, dtype=np.float64)
    best_index = 0
    best_rotation = float("inf")
    best_translation = float("inf")
    for index, pose in enumerate(poses):
        rotation_error = float(np.linalg.norm(pose[:3, :3] - identity))
        translation_error = float(np.linalg.norm(pose[:3, 3]))
        if rotation_error < best_rotation or (
            rotation_error == best_rotation
            and translation_error < best_translation
        ):
            best_index = index
            best_rotation = rotation_error
            best_translation = translation_error
    return best_index


def transformed_plane(parameter: Any, transform: np.ndarray) -> np.ndarray:
    value = np.asarray(parameter, dtype=np.float64)
    if value.shape != (4,):
        raise ValueError(f"plane parameter must have shape (4,), got {value.shape}")
    return value @ np.linalg.inv(transform)


def render_global_label_maps(
    node_info: dict[str, Any],
    plane_detection: dict[str, Any],
    poses: np.ndarray,
    reconstruction: Callable[..., Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reproduce official layout polygons while retaining global plane IDs."""
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be Nx4x4, got {poses.shape}")
    view_count = len(poses)
    expected_keys = {str(index) for index in range(view_count)}
    if set(plane_detection) != expected_keys:
        raise ValueError("plane_detection keys do not match ordered views")
    if set(node_info.get("planes", {})) != expected_keys:
        raise ValueError("node_info plane keys do not match ordered views")

    walls = node_info.get("global_plane_info", [])
    for index, wall in enumerate(walls):
        if int(wall.get("index", -1)) != index:
            raise ValueError("global_plane_info must be ordered by its index")
    floor = node_info.get("floor_pparam", [])
    ceiling = node_info.get("ceiling_pparam", [])
    floor_id = len(walls) if len(floor) else None
    ceiling_id = len(walls) + int(floor_id is not None) if len(ceiling) else None

    reference_index = closest_reference_pose(poses)
    reference_pose = poses[reference_index]
    intrinsic = np.array(
        [[762.0, 0.0, 640.0], [0.0, 762.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    label_maps: dict[str, np.ndarray] = {}
    per_view: list[dict[str, Any]] = []
    for view_index in range(view_count):
        key = str(view_index)
        transform = (
            np.linalg.inv(poses[view_index]) @ reference_pose
            if view_index != reference_index
            else np.eye(4, dtype=np.float64)
        )
        current_floor = transformed_plane(floor, transform) if len(floor) else []
        current_ceiling = (
            transformed_plane(ceiling, transform) if len(ceiling) else []
        )
        wall_ids = [int(value) for value in node_info["planes"][key]]
        if any(value < 0 or value >= len(walls) for value in wall_ids):
            raise ValueError(f"view {key} references an invalid global wall")
        current_walls = [
            transformed_plane(walls[wall_id]["pparam"], transform)
            for wall_id in wall_ids
        ]
        relationships = node_info["wall_relationship"][view_index]
        upper, lower, wall_output = reconstruction(
            current_walls,
            relationships,
            plane_detection[key]["line"],
            current_ceiling,
            current_floor,
            intrinsic,
        )
        upper = np.asarray(upper, dtype=np.int32)
        lower = np.asarray(lower, dtype=np.int32)
        if len(upper) != len(wall_output) + 1 or len(lower) != len(upper):
            raise ValueError(f"view {key} returned inconsistent layout boundaries")

        segmentation = np.full(RENDER_HW, -1, dtype=np.int32)
        if ceiling_id is not None:
            minimum_y = min(int(np.min(upper[:, 1])) - 10, -1)
            polygon = np.array(
                [[[upper[0, 0], minimum_y], *upper, [upper[-1, 0], minimum_y]]],
                dtype=np.int32,
            )
            cv2.fillPoly(segmentation, polygon, int(ceiling_id))
        if floor_id is not None:
            maximum_y = max(int(np.max(lower[:, 1])) + 10, RENDER_HW[0] + 1)
            polygon = np.array(
                [[[lower[0, 0], maximum_y], *lower, [lower[-1, 0], maximum_y]]],
                dtype=np.int32,
            )
            cv2.fillPoly(segmentation, polygon, int(floor_id))

        wall_cursor = 0
        rendered_wall_ids: list[int] = []
        for boundary_index, parameter in enumerate(wall_output):
            if parameter is None:
                continue
            if wall_cursor >= len(wall_ids):
                raise ValueError(f"view {key} rendered more walls than declared")
            wall_id = wall_ids[wall_cursor]
            wall_cursor += 1
            polygon = np.array(
                [[
                    upper[boundary_index],
                    lower[boundary_index],
                    lower[boundary_index + 1],
                    upper[boundary_index + 1],
                ]],
                dtype=np.int32,
            )
            cv2.fillPoly(segmentation, polygon, int(wall_id))
            rendered_wall_ids.append(wall_id)
        if wall_cursor != len(wall_ids):
            raise ValueError(
                f"view {key} rendered {wall_cursor} of {len(wall_ids)} walls"
            )
        pointmap = cv2.resize(
            segmentation,
            (POINTMAP_HW[1], POINTMAP_HW[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
        label_maps[key] = pointmap
        per_view.append(
            {
                "view_index": view_index,
                "rendered_wall_ids": rendered_wall_ids,
                "assigned_pixels": int((pointmap >= 0).sum()),
            }
        )

    plane_count = len(walls) + int(floor_id is not None) + int(ceiling_id is not None)
    return label_maps, {
        "reference_pose_index": reference_index,
        "wall_plane_count": len(walls),
        "floor_plane_id": floor_id,
        "ceiling_plane_id": ceiling_id,
        "predicted_plane_count_before_min_points": plane_count,
        "views": per_view,
    }


def registry_position_map(global_cloud: np.lib.npyio.NpzFile) -> dict[int, str]:
    registry = json.loads(str(global_cloud["dust3r_view_registry_json"].item()))
    result: dict[int, str] = {}
    for row in registry:
        view_index = int(row["alignment_view_index"])
        height, width = map(int, row["points_hw"])
        if (height, width) != POINTMAP_HW:
            raise ValueError(
                f"view {view_index} pointmap {(height, width)} != {POINTMAP_HW}"
            )
        position = Path(str(row["image_path"])).parent.name
        if position in result.values():
            raise ValueError(f"duplicate position ID in registry: {position}")
        result[view_index] = position
    return result


def sample_labels_on_gt_cache(
    gt: np.lib.npyio.NpzFile,
    position_by_view: dict[int, str],
    label_maps: dict[str, np.ndarray],
) -> np.ndarray:
    view_indices = gt["view_indices"].astype(np.int32)
    pixel_xy = gt["pixel_xy"].astype(np.int32)
    labels = np.full(len(view_indices), -1, dtype=np.int32)
    for view_index in sorted(int(value) for value in np.unique(view_indices)):
        if view_index not in position_by_view:
            raise ValueError(f"GT view {view_index} is absent from source registry")
        position = position_by_view[view_index]
        if position not in label_maps:
            raise ValueError(f"position {position} is absent from Plane-DUSt3R labels")
        mask = view_indices == view_index
        xy = pixel_xy[mask]
        height, width = label_maps[position].shape
        if (
            (xy[:, 0] < 0).any()
            or (xy[:, 0] >= width).any()
            or (xy[:, 1] < 0).any()
            or (xy[:, 1] >= height).any()
        ):
            raise ValueError(f"GT pixels for view {view_index} are out of range")
        labels[mask] = label_maps[position][xy[:, 1], xy[:, 0]]
    return labels


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    normal = vectors[-1]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return normal.astype(np.float32), float(-np.dot(normal, center))


def refit_predictions(
    points: np.ndarray, labels: np.ndarray, min_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if min_points < 1:
        raise ValueError("min_points must be positive")
    labels = labels.copy()
    candidate_ids = sorted(int(value) for value in np.unique(labels) if value >= 0)
    for plane_id in candidate_ids:
        if int((labels == plane_id).sum()) < min_points:
            labels[labels == plane_id] = -1
    retained = sorted(int(value) for value in np.unique(labels) if value >= 0)
    array_size = max(retained, default=-1) + 1
    normals = np.zeros((array_size, 3), dtype=np.float32)
    offsets = np.zeros(array_size, dtype=np.float32)
    counts = np.zeros(array_size, dtype=np.int32)
    for plane_id in retained:
        support = labels == plane_id
        normals[plane_id], offsets[plane_id] = fit_plane(points[support])
        counts[plane_id] = int(support.sum())
    return labels, normals, offsets, counts


def adapt_room(
    *,
    global_cloud_npz: Path,
    gt_npz: Path,
    plane_room_dir: Path,
    runtime_dir: Path,
    output_npz: Path,
    min_points: int = 64,
) -> dict[str, Any]:
    global_cloud_npz = global_cloud_npz.resolve()
    gt_npz = gt_npz.resolve()
    plane_room_dir = plane_room_dir.resolve()
    runtime_dir = runtime_dir.resolve()
    output_npz = output_npz.resolve()
    global_cloud = np.load(global_cloud_npz, allow_pickle=False)
    gt = np.load(gt_npz, allow_pickle=False)
    try:
        if str(gt["pixel_coordinate_order"].item()) != "xy":
            raise ValueError("GT pixel coordinate order must be xy")
        if str(gt["pixel_coordinate_space"].item()) != "dust3r_aligned_pointmap":
            raise ValueError("GT pixels must be in DUSt3R aligned-pointmap space")
        source_indices = gt["source_cache_indices"].astype(np.int64)
        if not np.array_equal(
            global_cloud["view_indices"][source_indices],
            gt["view_indices"],
        ) or not np.array_equal(
            global_cloud["pixel_xy"][source_indices],
            gt["pixel_xy"],
        ):
            raise ValueError("GT source indices do not preserve the ordered cache join")
        position_by_view = registry_position_map(global_cloud)
        node_info = load_json(plane_room_dir / "node_data.json")
        plane_detection = load_json(plane_room_dir / "plane_detection.json")
        dust3r = np.load(plane_room_dir / "dust3r_output.npz", allow_pickle=False)
        try:
            poses = dust3r["poses"].astype(np.float64)
        finally:
            dust3r.close()
        reconstruction = load_official_reconstruction(runtime_dir)
        label_maps, render_manifest = render_global_label_maps(
            node_info, plane_detection, poses, reconstruction
        )
        labels = sample_labels_on_gt_cache(gt, position_by_view, label_maps)
        labels, normals, offsets, counts = refit_predictions(
            gt["points"].astype(np.float32), labels, min_points
        )
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            schema_version=np.asarray(1, dtype=np.int32),
            method=np.asarray("plane_dust3r_same_input_common_point_cache"),
            points=gt["points"].astype(np.float32),
            point_plane_ids=labels,
            plane_normals=normals,
            plane_offsets=offsets,
            plane_inlier_counts=counts,
            view_indices=gt["view_indices"].astype(np.int32),
            pixel_xy=gt["pixel_xy"].astype(np.int32),
            pixel_coordinate_order=np.asarray("xy"),
            pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
            join_key=np.asarray("(alignment_view_index,x,y)"),
            spatial_nearest_neighbor_join=np.asarray(False),
            source_global_cloud_npz=np.asarray(str(global_cloud_npz)),
            source_gt_npz=np.asarray(str(gt_npz)),
            source_plane_room_dir=np.asarray(str(plane_room_dir)),
            min_plane_points=np.asarray(min_points, dtype=np.int32),
            render_manifest_json=np.asarray(json.dumps(render_manifest)),
        )
        return {
            "output_npz": str(output_npz),
            "points": int(len(labels)),
            "assigned_points": int((labels >= 0).sum()),
            "assigned_rate": float((labels >= 0).mean()),
            "planes": int(len(np.unique(labels[labels >= 0]))),
            "plane_inlier_counts": [
                int(counts[index]) for index in range(len(counts)) if counts[index] > 0
            ],
            "join_key": "(alignment_view_index,x,y)",
            "spatial_nearest_neighbor_join": False,
            "render_manifest": render_manifest,
        }
    finally:
        global_cloud.close()
        gt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global_cloud_npz", required=True)
    parser.add_argument("--gt_npz", required=True)
    parser.add_argument("--plane_room_dir", required=True)
    parser.add_argument("--runtime_dir", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_manifest", default="")
    parser.add_argument("--min_points", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = adapt_room(
        global_cloud_npz=Path(args.global_cloud_npz),
        gt_npz=Path(args.gt_npz),
        plane_room_dir=Path(args.plane_room_dir),
        runtime_dir=Path(args.runtime_dir),
        output_npz=Path(args.output_npz),
        min_points=args.min_points,
    )
    manifest_path = (
        Path(args.output_manifest)
        if args.output_manifest
        else Path(args.output_npz).with_suffix(".json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**result, "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
