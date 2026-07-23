"""Extract lightweight 2D/3D structural lines from a frozen DUSt3R cache.

The exporter operates in the saved aligned-pointmap image space. Every line
endpoint and every lifted 3D sample therefore has an explicit
``(alignment_view_index, x, y)`` provenance. It never joins geometry by nearest
XYZ and it never changes the cached pointmaps or camera variables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from export_stage3_scene_plane_fusion import load_global_views_from_cache


SCHEMA_VERSION = 1
ASSOCIATION_NAMES = {
    0: "unassigned",
    1: "within_plane",
    2: "plane_boundary",
    3: "single_plane_edge",
}
ASSOCIATION_RGB = {
    0: (160, 160, 160),
    1: (255, 196, 64),
    2: (239, 71, 111),
    3: (17, 138, 178),
}


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def scalar_text(raw, key, default=""):
    if key not in raw:
        return default
    value = raw[key]
    if np.ndim(value) == 0:
        return str(value.item())
    if len(value) == 0:
        return default
    return str(value.reshape(-1)[0])


def load_plane_label_records(path):
    """Load plane labels with explicit pointmap provenance from common schemas."""

    with np.load(path, allow_pickle=False) as raw:
        if "point_plane_ids" not in raw:
            raise ValueError("plane prediction is missing point_plane_ids")
        view_key = next(
            (key for key in ("source_views", "alignment_view_indices", "view_indices") if key in raw),
            None,
        )
        pixel_key = next(
            (key for key in ("pixel_xy", "pointmap_pixel_xy") if key in raw),
            None,
        )
        if view_key is None or pixel_key is None:
            raise ValueError(
                "plane prediction requires source_views/alignment_view_indices "
                "and pixel_xy/pointmap_pixel_xy"
            )
        labels = raw["point_plane_ids"].astype(np.int32).reshape(-1)
        views = raw[view_key].astype(np.int32).reshape(-1)
        pixels = raw[pixel_key].astype(np.int32)
        if pixels.shape != (len(labels), 2) or len(views) != len(labels):
            raise ValueError("plane prediction provenance arrays have inconsistent shapes")
        order = scalar_text(
            raw,
            "pointmap_pixel_coordinate_order",
            scalar_text(raw, "pixel_coordinate_order", "xy"),
        )
        space = scalar_text(
            raw,
            "pointmap_pixel_coordinate_space",
            scalar_text(raw, "pixel_coordinate_space", "dust3r_aligned_pointmap"),
        )
        if order != "xy":
            raise ValueError(f"plane prediction coordinate order must be xy, got {order!r}")
        if space != "dust3r_aligned_pointmap":
            raise ValueError(
                "plane prediction coordinate space must be "
                f"dust3r_aligned_pointmap, got {space!r}"
            )
    return {"labels": labels, "view_indices": views, "pixel_xy": pixels}


def build_plane_label_maps(global_views, records=None):
    """Rasterize exact plane records and explicitly drop duplicate conflicts."""

    by_index = {
        int(view["alignment_view_index"]): view for view in global_views.values()
    }
    maps = {
        index: np.full(np.asarray(view["points"]).shape[:2], -1, dtype=np.int32)
        for index, view in by_index.items()
    }
    diagnostics = {
        "records": 0,
        "positive_records": 0,
        "matched_positive_records": 0,
        "out_of_range_records": 0,
        "duplicate_positive_records": 0,
        "conflicting_keys": 0,
        "assigned_keys": 0,
    }
    if records is None:
        return maps, diagnostics

    labels = np.asarray(records["labels"], dtype=np.int32).reshape(-1)
    views = np.asarray(records["view_indices"], dtype=np.int32).reshape(-1)
    pixels = np.asarray(records["pixel_xy"], dtype=np.int32)
    diagnostics["records"] = int(len(labels))
    positive = labels >= 0
    diagnostics["positive_records"] = int(positive.sum())

    for view_index, label_map in maps.items():
        mask = positive & (views == view_index)
        if not mask.any():
            continue
        xy = pixels[mask]
        local_labels = labels[mask]
        height, width = label_map.shape
        valid = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < height)
        )
        diagnostics["out_of_range_records"] += int((~valid).sum())
        if not valid.any():
            continue
        xy = xy[valid]
        local_labels = local_labels[valid]
        diagnostics["matched_positive_records"] += int(len(local_labels))
        linear = xy[:, 1].astype(np.int64) * width + xy[:, 0]
        order = np.argsort(linear, kind="stable")
        linear = linear[order]
        local_labels = local_labels[order]
        starts = np.r_[0, np.flatnonzero(np.diff(linear)) + 1]
        counts = np.diff(np.r_[starts, len(linear)])
        unique_linear = linear[starts]
        minimum = np.minimum.reduceat(local_labels, starts)
        maximum = np.maximum.reduceat(local_labels, starts)
        consistent = minimum == maximum
        diagnostics["duplicate_positive_records"] += int((counts - 1).sum())
        diagnostics["conflicting_keys"] += int((~consistent).sum())
        label_map.reshape(-1)[unique_linear[consistent]] = minimum[consistent]
        diagnostics["assigned_keys"] += int(consistent.sum())

    unknown_views = positive & ~np.isin(views, np.asarray(sorted(maps), np.int32))
    diagnostics["out_of_range_records"] += int(unknown_views.sum())
    return maps, diagnostics


def detect_lines_lsd(image_rgb, min_length_px=24.0, max_lines=256):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required for structural-line extraction") from error

    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape (H,W,3), got {image.shape}")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(gray)[0]
    if detected is None:
        return np.empty((0, 4), dtype=np.float32)
    lines = np.asarray(detected, dtype=np.float32).reshape(-1, 4)
    lengths = np.linalg.norm(lines[:, 2:4] - lines[:, 0:2], axis=1)
    keep = lengths >= float(min_length_px)
    lines = lines[keep]
    lengths = lengths[keep]
    if not len(lines):
        return np.empty((0, 4), dtype=np.float32)
    order = np.argsort(-lengths, kind="stable")
    return lines[order[: int(max_lines)]].astype(np.float32)


def sample_line_pixels(line_xyxy, height, width, step_px=2.0):
    line = np.asarray(line_xyxy, dtype=np.float64).reshape(4)
    length = float(np.linalg.norm(line[2:4] - line[0:2]))
    count = max(2, int(math.ceil(length / max(float(step_px), 0.25))) + 1)
    x = np.linspace(line[0], line[2], count)
    y = np.linspace(line[1], line[3], count)
    xy = np.rint(np.column_stack((x, y))).astype(np.int32)
    xy[:, 0] = np.clip(xy[:, 0], 0, int(width) - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, int(height) - 1)
    if len(xy) > 1:
        keep = np.r_[True, np.any(xy[1:] != xy[:-1], axis=1)]
        xy = xy[keep]
    return xy


def dominant_nonnegative(values):
    values = np.asarray(values, dtype=np.int32).reshape(-1)
    values = values[values >= 0]
    if not len(values):
        return -1, 0.0
    labels, counts = np.unique(values, return_counts=True)
    best = int(np.argmax(counts))
    return int(labels[best]), float(counts[best] / len(values))


def associate_line_with_planes(line_xyxy, plane_label_map, step_px=2.0, side_offset_px=2.0):
    label_map = np.asarray(plane_label_map, dtype=np.int32)
    height, width = label_map.shape
    line = np.asarray(line_xyxy, dtype=np.float64).reshape(4)
    direction = line[2:4] - line[0:2]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return -1, -1, 0.0, 0.0, 0
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64) / norm
    center = sample_line_pixels(line, height, width, step_px).astype(np.float64)

    def side_values(sign):
        xy = np.rint(center + sign * float(side_offset_px) * normal).astype(np.int32)
        valid = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < height)
        )
        if not valid.any():
            return np.empty((0,), dtype=np.int32)
        xy = xy[valid]
        return label_map[xy[:, 1], xy[:, 0]]

    left, left_fraction = dominant_nonnegative(side_values(1.0))
    right, right_fraction = dominant_nonnegative(side_values(-1.0))
    if left >= 0 and right >= 0:
        code = 1 if left == right else 2
    elif left >= 0 or right >= 0:
        code = 3
    else:
        code = 0
    return left, right, left_fraction, right_fraction, code


def _split_valid_run(sample_positions, points, gap_factor):
    if not len(sample_positions):
        return np.empty((0,), dtype=np.int64)
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    adjacent = np.diff(sample_positions) == 1
    positive = distances[adjacent & np.isfinite(distances) & (distances > 1e-8)]
    gap_threshold = math.inf
    if len(positive):
        gap_threshold = float(np.median(positive) * float(gap_factor))
    runs = []
    start = 0
    for index in range(1, len(points)):
        broken_pixel_run = sample_positions[index] != sample_positions[index - 1] + 1
        broken_geometry = distances[index - 1] > gap_threshold
        if broken_pixel_run or broken_geometry:
            runs.append(np.arange(start, index, dtype=np.int64))
            start = index
    runs.append(np.arange(start, len(points), dtype=np.int64))
    return max(runs, key=lambda item: (len(item), -int(item[0])))


def lift_line_to_3d(
    line_xyxy,
    points,
    confidence,
    colors=None,
    min_conf=1.0,
    step_px=2.0,
    min_valid_samples=6,
    max_gap_factor=8.0,
):
    pointmap = np.asarray(points, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    if pointmap.ndim != 3 or pointmap.shape[2] != 3:
        raise ValueError(f"points must have shape (H,W,3), got {pointmap.shape}")
    if conf.shape != pointmap.shape[:2]:
        raise ValueError("confidence shape must match pointmap height/width")
    height, width = conf.shape
    xy = sample_line_pixels(line_xyxy, height, width, step_px)
    xyz = pointmap[xy[:, 1], xy[:, 0]]
    values = conf[xy[:, 1], xy[:, 0]]
    valid = (
        np.isfinite(xyz).all(axis=1)
        & (np.max(np.abs(xyz), axis=1) < 1e5)
        & np.isfinite(values)
        & (values >= float(min_conf))
    )
    positions = np.flatnonzero(valid)
    if len(positions) < int(min_valid_samples):
        return None
    valid_xyz = xyz[valid]
    run = _split_valid_run(positions, valid_xyz, max_gap_factor)
    if len(run) < int(min_valid_samples):
        return None
    run_xyz = valid_xyz[run]
    run_positions = positions[run]
    center = run_xyz.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(run_xyz - center, full_matrices=False)
    direction = vh[0].astype(np.float32)
    dominant = int(np.argmax(np.abs(direction)))
    if direction[dominant] < 0:
        direction = -direction
    projection = (run_xyz - center) @ direction
    lower, upper = np.quantile(projection, [0.02, 0.98])
    endpoint_3d = np.stack((center + lower * direction, center + upper * direction)).astype(np.float32)
    length_3d = float(np.linalg.norm(endpoint_3d[1] - endpoint_3d[0]))
    if not np.isfinite(length_3d) or length_3d <= 1e-8:
        return None
    reconstructed = center + projection[:, None] * direction[None]
    residual = np.linalg.norm(run_xyz - reconstructed, axis=1)
    line_ratio = float(
        singular_values[0] / max(float(singular_values[1]) if len(singular_values) > 1 else 0.0, 1e-8)
    )
    mean_color = np.asarray([220, 220, 220], dtype=np.uint8)
    if colors is not None:
        color_map = np.asarray(colors, dtype=np.uint8)
        if color_map.shape != pointmap.shape:
            raise ValueError("colors must have shape (H,W,3) matching points")
        run_xy = xy[run_positions]
        mean_color = np.rint(color_map[run_xy[:, 1], run_xy[:, 0]].mean(axis=0)).astype(np.uint8)
    return {
        "endpoints_3d": endpoint_3d,
        "direction_3d": direction,
        "length_3d": length_3d,
        "sample_count": int(len(xy)),
        "valid_sample_count": int(len(positions)),
        "retained_sample_count": int(len(run_xyz)),
        "mean_confidence": float(values[run_positions].mean()),
        "linearity_ratio": line_ratio,
        "fit_residual_mean": float(residual.mean()),
        "fit_residual_p95": float(np.quantile(residual, 0.95)),
        "mean_color": mean_color,
    }


def write_edge_ply(path, endpoints, colors):
    endpoints = np.asarray(endpoints, dtype=np.float32).reshape(-1, 2, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(endpoints) != len(colors):
        raise ValueError("line endpoints and colors must have matching counts")
    vertices = endpoints.reshape(-1, 3)
    vertex_colors = np.repeat(colors, 2, axis=0)
    path = Path(path)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element edge {len(endpoints)}\n")
        handle.write("property int vertex1\nproperty int vertex2\nend_header\n")
        for point, color in zip(vertices, vertex_colors):
            handle.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for index in range(len(endpoints)):
            handle.write(f"{2 * index} {2 * index + 1}\n")


def draw_overlay(image_rgb, rows, output_path):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required for line overlays") from error
    canvas = np.asarray(image_rgb, dtype=np.uint8).copy()
    for row in rows:
        x1, y1, x2, y2 = row["endpoints_2d"]
        rgb = ASSOCIATION_RGB[int(row["association_code"])]
        cv2.line(
            canvas,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            tuple(map(int, rgb)),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write line overlay: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        "Lightweight structural-line extraction on a frozen DUSt3R cache"
    )
    parser.add_argument("--global_cloud_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--plane_prediction_npz")
    parser.add_argument("--min_conf", type=float, default=1.0)
    parser.add_argument("--min_length_px", type=float, default=24.0)
    parser.add_argument("--max_lines_per_view", type=int, default=256)
    parser.add_argument("--sample_step_px", type=float, default=2.0)
    parser.add_argument("--min_valid_samples", type=int, default=6)
    parser.add_argument("--plane_side_offset_px", type=float, default=2.0)
    parser.add_argument("--max_3d_gap_factor", type=float, default=8.0)
    parser.add_argument(
        "--association_filter",
        choices=("all", "plane_related", "boundary"),
        default="all",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_length_px <= 0 or args.sample_step_px <= 0:
        raise ValueError("line length and sampling step must be positive")
    if args.max_lines_per_view < 1 or args.min_valid_samples < 2:
        raise ValueError("max lines and minimum valid samples are too small")
    if args.max_3d_gap_factor <= 1:
        raise ValueError("--max_3d_gap_factor must be greater than one")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir()

    started = time.perf_counter()
    global_views, alignment_loss, scene_key = load_global_views_from_cache(
        args.global_cloud_npz
    )
    plane_records = None
    if args.plane_prediction_npz:
        plane_records = load_plane_label_records(args.plane_prediction_npz)
    plane_maps, join_diagnostics = build_plane_label_maps(global_views, plane_records)

    rows = []
    view_rows = []
    for _, view in sorted(
        global_views.items(), key=lambda item: int(item[1]["alignment_view_index"])
    ):
        view_index = int(view["alignment_view_index"])
        image = np.asarray(view["colors"], dtype=np.uint8)
        pointmap = np.asarray(view["points"], dtype=np.float32)
        confidence = np.asarray(view["conf"], dtype=np.float32)
        detected = detect_lines_lsd(
            image, args.min_length_px, args.max_lines_per_view
        )
        local_rows = []
        rejected_3d = 0
        rejected_association = 0
        for line in detected:
            left, right, left_fraction, right_fraction, association_code = (
                associate_line_with_planes(
                    line,
                    plane_maps[view_index],
                    args.sample_step_px,
                    args.plane_side_offset_px,
                )
            )
            if args.association_filter == "plane_related" and association_code == 0:
                rejected_association += 1
                continue
            if args.association_filter == "boundary" and association_code != 2:
                rejected_association += 1
                continue
            lifted = lift_line_to_3d(
                line,
                pointmap,
                confidence,
                image,
                args.min_conf,
                args.sample_step_px,
                args.min_valid_samples,
                args.max_3d_gap_factor,
            )
            if lifted is None:
                rejected_3d += 1
                continue
            length_2d = float(np.linalg.norm(line[2:4] - line[0:2]))
            row = {
                "line_id": int(len(rows)),
                "alignment_view_index": view_index,
                "image_path": str(view["image_path"]),
                "endpoints_2d": [float(value) for value in line],
                "pixel_coordinate_order": "xy",
                "pixel_coordinate_space": "dust3r_aligned_pointmap",
                "length_2d_pixels": length_2d,
                "left_plane_id": int(left),
                "right_plane_id": int(right),
                "left_plane_fraction": float(left_fraction),
                "right_plane_fraction": float(right_fraction),
                "association_code": int(association_code),
                "association": ASSOCIATION_NAMES[int(association_code)],
                "endpoints_3d": lifted["endpoints_3d"].tolist(),
                "direction_3d": lifted["direction_3d"].tolist(),
                "length_3d": lifted["length_3d"],
                "sample_count": lifted["sample_count"],
                "valid_sample_count": lifted["valid_sample_count"],
                "retained_sample_count": lifted["retained_sample_count"],
                "mean_confidence": lifted["mean_confidence"],
                "linearity_ratio": lifted["linearity_ratio"],
                "fit_residual_mean": lifted["fit_residual_mean"],
                "fit_residual_p95": lifted["fit_residual_p95"],
                "mean_color": lifted["mean_color"].tolist(),
            }
            rows.append(row)
            local_rows.append(row)
        overlay_path = overlay_dir / f"view_{view_index:03d}_lines.png"
        draw_overlay(image, local_rows, overlay_path)
        view_rows.append(
            {
                "alignment_view_index": view_index,
                "image_path": str(view["image_path"]),
                "pointmap_hw": list(map(int, pointmap.shape[:2])),
                "detected_2d_lines": int(len(detected)),
                "output_3d_lines": int(len(local_rows)),
                "rejected_3d": int(rejected_3d),
                "rejected_association": int(rejected_association),
                "overlay": str(overlay_path),
            }
        )

    endpoints_2d = np.asarray(
        [row["endpoints_2d"] for row in rows], dtype=np.float32
    ).reshape(-1, 2, 2)
    endpoints_3d = np.asarray(
        [row["endpoints_3d"] for row in rows], dtype=np.float32
    ).reshape(-1, 2, 3)
    line_colors = np.asarray(
        [ASSOCIATION_RGB[row["association_code"]] for row in rows], dtype=np.uint8
    ).reshape(-1, 3)
    config = vars(args).copy()
    npz_path = output_dir / "structural_lines.npz"
    np.savez_compressed(
        npz_path,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        scene_key=np.asarray(scene_key),
        line_ids=np.arange(len(rows), dtype=np.int32),
        alignment_view_indices=np.asarray(
            [row["alignment_view_index"] for row in rows], dtype=np.int32
        ),
        line_pixel_xy=endpoints_2d,
        line_endpoints_3d=endpoints_3d,
        line_directions_3d=np.asarray(
            [row["direction_3d"] for row in rows], dtype=np.float32
        ).reshape(-1, 3),
        line_lengths_2d=np.asarray(
            [row["length_2d_pixels"] for row in rows], dtype=np.float32
        ),
        line_lengths_3d=np.asarray(
            [row["length_3d"] for row in rows], dtype=np.float32
        ),
        line_left_plane_ids=np.asarray(
            [row["left_plane_id"] for row in rows], dtype=np.int32
        ),
        line_right_plane_ids=np.asarray(
            [row["right_plane_id"] for row in rows], dtype=np.int32
        ),
        line_association_codes=np.asarray(
            [row["association_code"] for row in rows], dtype=np.int8
        ),
        line_mean_confidence=np.asarray(
            [row["mean_confidence"] for row in rows], dtype=np.float32
        ),
        line_fit_residual_mean=np.asarray(
            [row["fit_residual_mean"] for row in rows], dtype=np.float32
        ),
        pixel_coordinate_order=np.asarray("xy"),
        pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
        geometry_coordinate_space=np.asarray("dust3r_global_alignment"),
        config_json=np.asarray(json.dumps(config, sort_keys=True)),
    )
    ply_path = output_dir / "structural_lines.ply"
    write_edge_ply(ply_path, endpoints_3d, line_colors)
    rows_path = output_dir / "structural_lines.json"
    rows_path.write_text(
        json.dumps({"scene_key": scene_key, "lines": rows}, indent=2),
        encoding="utf-8",
    )

    runtime = time.perf_counter() - started
    association_counts = {
        name: int(sum(row["association"] == name for row in rows))
        for name in ASSOCIATION_NAMES.values()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": "lsd_exact_pointmap_lift",
        "scene_key": scene_key,
        "source_global_cloud_npz": str(Path(args.global_cloud_npz)),
        "source_global_cloud_sha256": file_sha256(args.global_cloud_npz),
        "source_plane_prediction_npz": str(Path(args.plane_prediction_npz))
        if args.plane_prediction_npz
        else None,
        "source_plane_prediction_sha256": file_sha256(args.plane_prediction_npz)
        if args.plane_prediction_npz
        else None,
        "dust3r_global_alignment_loss": alignment_loss,
        "coordinate_order": "xy",
        "coordinate_space": "dust3r_aligned_pointmap",
        "geometry_coordinate_space": "dust3r_global_alignment",
        "views": view_rows,
        "line_count": int(len(rows)),
        "association_counts": association_counts,
        "plane_join_diagnostics": join_diagnostics,
        "runtime_seconds": runtime,
        "config": config,
        "npz": str(npz_path),
        "ply": str(ply_path),
        "json": str(rows_path),
    }
    manifest_path = output_dir / "structural_lines_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
