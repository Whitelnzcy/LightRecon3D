import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from export_stage2_geometry_refit_editables import plane_rows, write_ascii_ply, write_params
from make_full_pointcloud_edit_comparison import HTML_TEMPLATE, PLANE_COLORS, deterministic_sample
from train_stage2_region_merge_net import fit_plane_np


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def scalar_string(raw, key, default=""):
    if key not in raw:
        return default
    value = raw[key]
    if np.ndim(value) == 0:
        return str(value.item())
    if len(value) == 0:
        return default
    return str(value[0])


def path_key(path):
    text = str(path).strip()
    if not text:
        return ""
    return os.path.normpath(text).replace("\\", "/")


def group_key(raw, fallback, mode):
    scene = scalar_string(raw, "scene_name", fallback)
    pair_group = scalar_string(raw, "pair_group", "")
    rgb_path1 = scalar_string(raw, "rgb_path1", "")
    if mode == "scene":
        return scene
    if mode == "pair_group":
        return f"{scene}|{pair_group}" if pair_group else scene
    if mode == "reference_view":
        return f"{scene}|{rgb_path1}" if rgb_path1 else fallback
    return fallback


def plane_angle_deg(normal_a, normal_b):
    dot = float(abs(np.dot(normal_a, normal_b)))
    dot = min(max(dot, 0.0), 1.0)
    return float(np.degrees(np.arccos(dot)))


def mutual_residual(plane_a, plane_b):
    points_a = plane_a["points"]
    points_b = plane_b["points"]
    normal_a = plane_a["normal"]
    normal_b = plane_b["normal"]
    offset_a = plane_a["offset"]
    offset_b = plane_b["offset"]
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf")
    residual_ab = np.abs(points_a @ normal_b + offset_b)
    residual_ba = np.abs(points_b @ normal_a + offset_a)
    return float(0.5 * (np.mean(residual_ab) + np.mean(residual_ba)))


def normalize_rgb(img):
    try:
        import torch
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
    except ImportError:
        pass
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32)
    if img.max() > 2.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def setup_dust3r_imports():
    project_root = os.path.dirname(os.path.abspath(__file__))
    dust3r_root = os.path.join(project_root, "dust3r")
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)
    if dust3r_root in sys.path:
        sys.path.remove(dust3r_root)
    sys.path.insert(1, dust3r_root)


def collect_group_images(files, include_second_view=True):
    paths = []
    seen = set()
    for path in files:
        raw = np.load(path)
        keys = ["rgb_path1"]
        if include_second_view:
            keys.append("rgb_path2")
        for key in keys:
            value = scalar_string(raw, key, "")
            norm = path_key(value)
            if norm and norm not in seen:
                seen.add(norm)
                paths.append(value)
    return paths


def run_dust3r_global_alignment(image_paths, model, device, args):
    setup_dust3r_imports()
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    if len(image_paths) < 2:
        raise RuntimeError(f"DUSt3R global alignment needs at least 2 images, got {len(image_paths)}")

    imgs = load_images(image_paths, size=args.image_size, verbose=True)
    pairs = make_pairs(imgs, scene_graph=args.scene_graph, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=args.batch_size, verbose=True)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=args.niter, schedule=args.schedule, lr=args.lr)

    pts3d = scene.get_pts3d()
    confs = scene.get_conf()
    views = {}
    for index, image_path in enumerate(image_paths):
        pts = pts3d[index]
        conf = confs[index]
        try:
            import torch
            if torch.is_tensor(pts):
                pts = pts.detach().cpu().numpy()
            if torch.is_tensor(conf):
                conf = conf.detach().cpu().numpy()
        except ImportError:
            pass
        img = normalize_rgb(scene.imgs[index])
        views[path_key(image_path)] = {
            "alignment_view_index": int(index),
            "image_path": image_path,
            "points": np.asarray(pts, dtype=np.float32),
            "conf": np.asarray(conf, dtype=np.float32),
            "colors": (img * 255.0).clip(0, 255).astype(np.uint8),
        }
    return views, float(loss)


def pixel_array_for_role(raw, role, point_count):
    key = f"pixel_xy{role}"
    fallback_key = "pixel_xy" if role == 1 else key
    if key in raw:
        pixel_xy = raw[key].astype(np.float32)
    elif fallback_key in raw:
        pixel_xy = raw[fallback_key].astype(np.float32)
    else:
        raise RuntimeError(f"Stage2 NPZ has no {key}; cannot project support from view{role}")
    if pixel_xy.ndim != 2 or pixel_xy.shape[1] != 2:
        raise RuntimeError(f"Bad {key} shape: {pixel_xy.shape}")
    if len(pixel_xy) not in (0, point_count):
        raise RuntimeError(f"{key} length {len(pixel_xy)} is incompatible with point count {point_count}")
    return pixel_xy


def sample_global_view(view, pixel_xy, args):
    pts_hw = view["points"]
    conf_hw = view["conf"]
    color_hw = view["colors"]
    h, w = pts_hw.shape[:2]
    x = np.rint((pixel_xy[:, 0] + 1.0) * 0.5 * (w - 1)).astype(np.int64)
    y = np.rint((pixel_xy[:, 1] + 1.0) * 0.5 * (h - 1)).astype(np.int64)
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    points = pts_hw[y, x].astype(np.float32)
    colors = color_hw[y, x].astype(np.uint8)
    conf = conf_hw[y, x].astype(np.float32)
    valid = np.isfinite(points).all(axis=1)
    valid &= np.max(np.abs(points), axis=1) < 1e5
    valid &= conf >= float(args.global_min_conf)
    return points, colors, conf, valid


def map_stage2_points_to_global(raw, global_views, args):
    point_count = int(len(raw["point_plane_ids"]))
    if "support_source_view" in raw:
        source_view = np.asarray(raw["support_source_view"]).reshape(-1).astype(np.int8)
        if len(source_view) != point_count:
            raise RuntimeError(f"support_source_view length {len(source_view)} != point count {point_count}")
    else:
        source_view = np.ones((point_count,), dtype=np.int8)

    points = np.zeros((point_count, 3), dtype=np.float32)
    colors = np.zeros((point_count, 3), dtype=np.uint8)
    conf = np.zeros((point_count,), dtype=np.float32)
    keep = np.zeros((point_count,), dtype=bool)
    stats = {
        "point_count": point_count,
        "source_counts": {},
        "kept_counts": {},
        "registered_view_indices": {},
        "missing_view_roles": [],
    }

    for role in (1, 2):
        role_mask = source_view == role
        role_count = int(role_mask.sum())
        stats["source_counts"][str(role)] = role_count
        if role_count == 0:
            stats["kept_counts"][str(role)] = 0
            continue

        rgb_path = scalar_string(raw, f"rgb_path{role}", "")
        view = global_views.get(path_key(rgb_path))
        if view is None:
            stats["missing_view_roles"].append(role)
            raise RuntimeError(f"No DUSt3R global view for rgb_path{role}={rgb_path!r}")

        pixel_xy = pixel_array_for_role(raw, role, point_count)
        if len(pixel_xy) == 0:
            raise RuntimeError(f"support_source_view references view{role}, but pixel_xy{role} is empty")

        sampled_points, sampled_colors, sampled_conf, sampled_keep = sample_global_view(view, pixel_xy[role_mask], args)
        points[role_mask] = sampled_points
        colors[role_mask] = sampled_colors
        conf[role_mask] = sampled_conf
        keep[role_mask] = sampled_keep
        stats["kept_counts"][str(role)] = int(sampled_keep.sum())
        stats["registered_view_indices"][str(role)] = int(view["alignment_view_index"])

    invalid_roles = sorted(set(int(x) for x in source_view.tolist() if int(x) not in (1, 2)))
    if invalid_roles:
        raise RuntimeError(f"support_source_view has unsupported roles: {invalid_roles}")

    stats["kept_total"] = int(keep.sum())
    stats["dropped_total"] = int(point_count - keep.sum())
    stats["mean_kept_conf"] = float(conf[keep].mean()) if int(keep.sum()) else 0.0
    return points[keep], colors[keep], keep, stats


def load_scene_inputs(files, min_points, args, global_views=None):
    scene_points = []
    scene_colors = []
    scene_local_assignment = []
    scene_source_views = []
    scene_file_indices = []
    planes = []
    point_offset = 0
    skipped = []
    mapping_records = []
    for file_index, path in enumerate(files):
        raw = np.load(path)
        assignment = raw["point_plane_ids"].astype(np.int32)
        mapping_stats = None
        if global_views is None:
            points = raw["points"].astype(np.float32)
            colors_key = "original_colors" if "original_colors" in raw else "colors"
            colors = raw[colors_key].astype(np.uint8)
            keep = np.isfinite(points).all(axis=1)
            keep &= np.max(np.abs(points), axis=1) < 1e5
        else:
            points, colors, keep, mapping_stats = map_stage2_points_to_global(raw, global_views, args)
        if int(keep.sum()) == 0:
            skipped.append({"file": str(path), "reason": "no_valid_points"})
            continue
        if len(assignment) != len(keep):
            raise RuntimeError(f"Assignment length mismatch for {path}: {len(assignment)} vs {len(keep)}")
        assignment = assignment[keep]
        if "support_source_view" in raw:
            source_view = np.asarray(raw["support_source_view"]).reshape(-1).astype(np.int8)
            if len(source_view) != len(keep):
                raise RuntimeError(f"support_source_view length mismatch for {path}: {len(source_view)} vs {len(keep)}")
            source_view = source_view[keep]
        else:
            source_view = np.ones((len(assignment),), dtype=np.int8)
        scene_source_views.append(source_view)
        scene_file_indices.append(np.full((len(assignment),), file_index, dtype=np.int32))
        if mapping_stats is not None:
            mapping_stats = dict(mapping_stats)
            mapping_stats["file"] = str(path)
            mapping_stats["kept_after_assignment_filter"] = int(len(assignment))
            mapping_records.append(mapping_stats)
        normals = raw["plane_normals"].astype(np.float32)
        offsets = raw["plane_offsets"].astype(np.float32)
        scene_points.append(points)
        scene_colors.append(colors)
        local_assignment = np.full_like(assignment, -1, dtype=np.int32)
        for plane_id in sorted(int(x) for x in np.unique(assignment) if int(x) >= 0):
            mask = assignment == plane_id
            count = int(mask.sum())
            if count < min_points or plane_id >= len(normals):
                continue
            local_index = len(planes)
            local_assignment[mask] = local_index
            pts = points[mask]
            normal, offset, _ = fit_plane_np(pts)
            planes.append(
                {
                    "local_index": local_index,
                    "file_index": file_index,
                    "source_file": str(path),
                    "source_plane_id": int(plane_id),
                    "global_point_indices": np.nonzero(mask)[0].astype(np.int64) + point_offset,
                    "points": pts,
                    "normal": normal.astype(np.float32),
                    "offset": float(offset),
                    "source_local_normal": normals[plane_id].astype(np.float32).tolist(),
                    "source_local_offset": float(offsets[plane_id]),
                    "count": count,
                    "centroid": pts.mean(axis=0),
                }
            )
        scene_local_assignment.append(local_assignment)
        point_offset += len(points)
    if not scene_points:
        return None
    return {
        "points": np.concatenate(scene_points, axis=0),
        "colors": np.concatenate(scene_colors, axis=0),
        "source_views": np.concatenate(scene_source_views, axis=0),
        "point_file_indices": np.concatenate(scene_file_indices, axis=0),
        "local_assignments": scene_local_assignment,
        "planes": planes,
        "skipped": skipped,
        "mapping_records": mapping_records,
    }


def should_merge(plane_a, plane_b, args):
    angle = plane_angle_deg(plane_a["normal"], plane_b["normal"])
    offset = abs(abs(float(plane_a["offset"])) - abs(float(plane_b["offset"])))
    residual = mutual_residual(plane_a, plane_b)
    centroid_distance = float(np.linalg.norm(plane_a["centroid"] - plane_b["centroid"]))
    return (
        angle <= args.max_angle_deg
        and offset <= args.max_offset
        and residual <= args.max_mutual_residual
        and centroid_distance <= args.max_centroid_distance
    ), {
        "angle_deg": angle,
        "offset_abs": offset,
        "mutual_residual": residual,
        "centroid_distance": centroid_distance,
    }


def safe_filename(text):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)[-120:] or "scene"


def plane_basis(normal):
    normal = np.asarray(normal, dtype=np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1e-8)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(normal, helper).astype(np.float32)
    u /= max(float(np.linalg.norm(u)), 1e-8)
    v = np.cross(normal, u).astype(np.float32)
    v /= max(float(np.linalg.norm(v)), 1e-8)
    return u, v


def plane_bbox_area(points, normal):
    if len(points) < 3:
        return 0.0
    u, v = plane_basis(normal)
    centered = points - points.mean(axis=0, keepdims=True)
    uv = np.stack((centered @ u, centered @ v), axis=1)
    span = uv.max(axis=0) - uv.min(axis=0)
    return float(max(span[0], 0.0) * max(span[1], 0.0))



def plane_uv(points, normal):
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32)
    u, v = plane_basis(normal)
    center = points.mean(axis=0).astype(np.float32)
    centered = points - center[None]
    uv = np.stack((centered @ u, centered @ v), axis=1).astype(np.float32)
    return uv, center, u, v


def textured_plane_mesh(points, colors, assignment, normals, offsets, grid_resolution=48, min_cell_points=2):
    vertices = []
    faces = []
    vertex_key_to_index = {}
    plane_mesh_rows = []

    for plane_id in range(len(normals)):
        mask = assignment == plane_id
        pts = points[mask]
        rgb = colors[mask]
        if len(pts) < max(16, min_cell_points):
            plane_mesh_rows.append({"plane_id": int(plane_id), "vertices": 0, "faces": 0, "skipped": "too_few_points"})
            continue

        uv, center, u, v = plane_uv(pts, normals[plane_id])
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        span = uv_max - uv_min
        if float(span.min()) <= 1e-5:
            plane_mesh_rows.append({"plane_id": int(plane_id), "vertices": 0, "faces": 0, "skipped": "degenerate_extent"})
            continue

        bins_u = int(np.clip(grid_resolution, 8, 128))
        bins_v = int(np.clip(round(grid_resolution * float(span[1] / max(span[0], 1e-6))), 8, 128))
        cell = span / np.asarray([bins_u, bins_v], dtype=np.float32)
        ij = np.floor((uv - uv_min[None]) / cell[None]).astype(np.int32)
        ij[:, 0] = np.clip(ij[:, 0], 0, bins_u - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, bins_v - 1)

        count_grid = np.zeros((bins_u, bins_v), dtype=np.int32)
        color_grid = np.zeros((bins_u, bins_v, 3), dtype=np.float32)
        for (iu, iv), color in zip(ij, rgb):
            count_grid[iu, iv] += 1
            color_grid[iu, iv] += color.astype(np.float32)
        occupied = count_grid >= int(min_cell_points)
        color_grid[occupied] /= count_grid[occupied, None]

        plane_vertex_start = len(vertices)
        plane_face_start = len(faces)

        def vertex_index(iu, iv):
            key = (plane_id, int(iu), int(iv))
            if key in vertex_key_to_index:
                return vertex_key_to_index[key]
            uv_corner = uv_min + np.asarray([iu, iv], dtype=np.float32) * cell
            xyz = center + uv_corner[0] * u + uv_corner[1] * v
            # Use the nearest occupied cell color around this corner for a texture-like vertex color.
            nearby = []
            for du in (-1, 0):
                for dv in (-1, 0):
                    cu, cv = iu + du, iv + dv
                    if 0 <= cu < bins_u and 0 <= cv < bins_v and occupied[cu, cv]:
                        nearby.append(color_grid[cu, cv])
            if nearby:
                color = np.mean(np.stack(nearby, axis=0), axis=0)
            else:
                color = np.asarray(PLANE_COLORS[plane_id % len(PLANE_COLORS)], dtype=np.float32)
            index = len(vertices)
            vertices.append((xyz.astype(np.float32), np.clip(color, 0, 255).astype(np.uint8)))
            vertex_key_to_index[key] = index
            return index

        for iu in range(bins_u):
            for iv in range(bins_v):
                if not occupied[iu, iv]:
                    continue
                v00 = vertex_index(iu, iv)
                v10 = vertex_index(iu + 1, iv)
                v11 = vertex_index(iu + 1, iv + 1)
                v01 = vertex_index(iu, iv + 1)
                faces.append((v00, v10, v11))
                faces.append((v00, v11, v01))

        plane_mesh_rows.append(
            {
                "plane_id": int(plane_id),
                "vertices": int(len(vertices) - plane_vertex_start),
                "faces": int(len(faces) - plane_face_start),
                "grid_u": int(bins_u),
                "grid_v": int(bins_v),
                "occupied_cells": int(occupied.sum()),
            }
        )

    return vertices, faces, plane_mesh_rows


def write_textured_plane_mesh_ply(path, points, colors, assignment, normals, offsets, grid_resolution=48, min_cell_points=2):
    vertices, faces, rows = textured_plane_mesh(
        points,
        colors,
        assignment,
        normals,
        offsets,
        grid_resolution=grid_resolution,
        min_cell_points=min_cell_points,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for xyz, color in vertices:
            f.write(
                f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    return rows

def plane_quality_rows(points, assignment, normals, offsets, source_views, point_file_indices):
    rows = []
    for plane_id in range(len(normals)):
        mask = assignment == plane_id
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "plane_id": int(plane_id),
                    "inlier_count": 0,
                    "source_view_count": 0,
                    "source_pair_count": 0,
                    "residual_mean": 0.0,
                    "residual_p50": 0.0,
                    "residual_p95": 0.0,
                    "plane_area_bbox": 0.0,
                    "support_source_counts": {},
                }
            )
            continue
        pts = points[mask]
        residual = np.abs(pts @ normals[plane_id] + offsets[plane_id]).astype(np.float32)
        source_values, source_counts = np.unique(source_views[mask], return_counts=True)
        support_counts = {str(int(value)): int(num) for value, num in zip(source_values, source_counts)}
        rows.append(
            {
                "plane_id": int(plane_id),
                "inlier_count": count,
                "source_view_count": int(len(source_values)),
                "source_pair_count": int(len(np.unique(point_file_indices[mask]))),
                "residual_mean": float(residual.mean()) if len(residual) else 0.0,
                "residual_p50": float(np.quantile(residual, 0.50)) if len(residual) else 0.0,
                "residual_p95": float(np.quantile(residual, 0.95)) if len(residual) else 0.0,
                "plane_area_bbox": plane_bbox_area(pts, normals[plane_id]),
                "support_source_counts": support_counts,
            }
        )
    return rows


def aggregate_plane_quality(plane_quality):
    if not plane_quality:
        return {
            "plane_count": 0,
            "bad_plane_count": 0,
            "mean_residual_mean": 0.0,
            "mean_residual_p95": 0.0,
            "avg_source_views_per_plane": 0.0,
            "multi_view_plane_rate": 0.0,
        }
    residual_mean = np.asarray([row["residual_mean"] for row in plane_quality], dtype=np.float32)
    residual_p95 = np.asarray([row["residual_p95"] for row in plane_quality], dtype=np.float32)
    source_views = np.asarray([row["source_view_count"] for row in plane_quality], dtype=np.float32)
    bad = (residual_p95 > 0.08) | (source_views < 2)
    return {
        "plane_count": int(len(plane_quality)),
        "bad_plane_count": int(bad.sum()),
        "mean_residual_mean": float(residual_mean.mean()),
        "mean_residual_p95": float(residual_p95.mean()),
        "avg_source_views_per_plane": float(source_views.mean()),
        "multi_view_plane_rate": float((source_views >= 2).mean()),
    }



def quality_grade(summary, loss):
    bad = int(summary.get("bad_plane_count", 0))
    planes = max(int(summary.get("plane_count", 0)), 1)
    multi = float(summary.get("multi_view_plane_rate", 0.0))
    p95 = float(summary.get("mean_residual_p95", 0.0))
    score = 100.0
    score -= 35.0 * bad / planes
    score -= 25.0 * max(0.0, 0.6 - multi) / 0.6
    score -= 25.0 * min(max(p95 / 0.12, 0.0), 1.0)
    if loss is not None and np.isfinite(loss):
        score -= 15.0 * min(max(float(loss) / 0.05, 0.0), 1.0)
    score = max(0.0, min(100.0, score))
    if score >= 80.0:
        label = "good"
    elif score >= 55.0:
        label = "mixed"
    else:
        label = "weak"
    return {"score": float(score), "label": label}


def write_stage3_report_html(
    path,
    points,
    colors,
    assignment,
    planes,
    edit_delta,
    edit_plane,
    max_display_points,
    source_npz,
    scene_key,
    fusion_mode,
    global_alignment_loss,
    quality_summary,
    plane_quality,
    global_view_registry,
    mapping_records,
    image_paths,
):
    if not planes:
        raise ValueError(f"No active planes in {source_npz}")
    if edit_plane == "largest":
        edit_plane_id = max(planes, key=lambda row: row["assigned_point_count"])["id"]
    else:
        edit_plane_id = int(edit_plane)
    if edit_plane_id < 0 or edit_plane_id >= len(planes):
        raise ValueError(f"Invalid edit plane {edit_plane_id}; active planes={len(planes)}")

    normal = np.asarray(planes[edit_plane_id]["normal"], dtype=np.float32)
    moved_mask = assignment == edit_plane_id
    edited_points = points.copy()
    edited_points[moved_mask] = edited_points[moved_mask] - float(edit_delta) * normal

    sample_idx = deterministic_sample(len(points), moved_mask, max_display_points)
    sample_colors = []
    for plane_id, color in zip(assignment[sample_idx], colors[sample_idx]):
        if int(plane_id) >= 0:
            palette = PLANE_COLORS[int(plane_id) % len(PLANE_COLORS)]
            sample_colors.append(f"rgb({palette[0]},{palette[1]},{palette[2]})")
        else:
            sample_colors.append(f"rgb({int(color[0])},{int(color[1])},{int(color[2])})")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    scale = 0.74 / max(span, 1e-6)
    offset_before = float(planes[edit_plane_id]["offset"])
    grade = quality_grade(quality_summary, global_alignment_loss)

    html_data = {
        "input_npz": str(source_npz),
        "total_points": int(len(points)),
        "display_points": int(len(sample_idx)),
        "moved_points": int(np.sum(moved_mask)),
        "center": [float(x) for x in center],
        "scale": scale,
        "planes": planes,
        "edit": {
            "plane_id": int(edit_plane_id),
            "delta": float(edit_delta),
            "offset_before": offset_before,
            "offset_after": offset_before + float(edit_delta),
        },
        "before_points": points[sample_idx].round(5).tolist(),
        "after_points": edited_points[sample_idx].round(5).tolist(),
        "sample_moved": moved_mask[sample_idx].astype(bool).tolist(),
        "sample_colors": sample_colors,
        "stage3": {
            "scene_key": str(scene_key),
            "fusion_mode": str(fusion_mode),
            "global_alignment_loss": None if global_alignment_loss is None else float(global_alignment_loss),
            "quality_grade": grade,
            "quality_summary": quality_summary,
            "plane_quality": plane_quality,
            "view_registry": global_view_registry,
            "mapping_records": mapping_records,
            "image_paths": image_paths,
        },
    }

    extra_css = """
.scoreBox { margin:12px 0 10px; border:1px solid #d7dce5; border-radius:8px; padding:12px; background:#fbfcff; }
.scoreTop { display:flex; align-items:baseline; justify-content:space-between; gap:10px; }
.score { font-size:32px; font-weight:800; letter-spacing:0; }
.badge { border-radius:999px; padding:4px 9px; font-size:12px; font-weight:800; text-transform:uppercase; }
.badge.good { background:#e8f5e9; color:#166534; }
.badge.mixed { background:#fff7ed; color:#9a3412; }
.badge.weak { background:#fef2f2; color:#991b1b; }
.kv { display:grid; grid-template-columns:1fr auto; gap:5px 10px; margin-top:8px; font-size:12px; color:#4b5563; }
.kv b { color:#111827; }
.table { width:100%; border-collapse:collapse; font-size:12px; }
.table th, .table td { border-bottom:1px solid #e5e7eb; padding:6px 4px; text-align:right; }
.table th:first-child, .table td:first-child { text-align:left; }
.table .risk { color:#b91c1c; font-weight:800; }
.views { display:grid; gap:6px; }
.viewRow { border:1px solid #e5e7eb; border-radius:6px; padding:7px; background:#fafafa; font-size:12px; overflow-wrap:anywhere; }
.path { color:#4b5563; margin-top:3px; }
"""
    extra_panel = """
    <section id=\"stage3Report\"></section>
"""
    extra_js = """
function pct(x) { return `${(100 * Number(x || 0)).toFixed(1)}%`; }
function shortPath(p) { const parts = String(p || '').split(/[\\/]/); return parts.slice(-3).join('/'); }
function renderStage3Report() {
  const s = DATA.stage3 || {};
  const q = s.quality_summary || {};
  const grade = s.quality_grade || {score:0, label:'weak'};
  const loss = s.global_alignment_loss == null ? 'n/a' : Number(s.global_alignment_loss).toFixed(5);
  const rows = (s.plane_quality || []).map(p => {
    const risky = Number(p.residual_p95 || 0) > 0.08 || Number(p.source_view_count || 0) < 2;
    return `<tr>
      <td class=\"${risky ? 'risk' : ''}\">Plane ${p.plane_id}${risky ? ' !' : ''}</td>
      <td>${Number(p.inlier_count || 0).toLocaleString()}</td>
      <td>${p.source_view_count || 0}</td>
      <td>${Number(p.residual_mean || 0).toFixed(4)}</td>
      <td>${Number(p.residual_p95 || 0).toFixed(4)}</td>
      <td>${Number(p.plane_area_bbox || 0).toFixed(3)}</td>
    </tr>`;
  }).join('');
  const views = (s.view_registry || []).map(v => `<div class=\"viewRow\"><b>View ${v.alignment_view_index}</b> ${shortPath(v.image_path)}<div class=\"path\">${v.image_path}</div></div>`).join('');
  stage3Report.innerHTML = `
    <h2>Stage3 Quality</h2>
    <div class=\"scoreBox\">
      <div class=\"scoreTop\"><div><div class=\"small\">Demo quality score</div><div class=\"score\">${Number(grade.score || 0).toFixed(0)}</div></div><span class=\"badge ${grade.label}\">${grade.label}</span></div>
      <div class=\"kv\">
        <span>global alignment loss</span><b>${loss}</b>
        <span>bad planes</span><b>${q.bad_plane_count || 0} / ${q.plane_count || 0}</b>
        <span>multi-view plane rate</span><b>${pct(q.multi_view_plane_rate)}</b>
        <span>mean residual p95</span><b>${Number(q.mean_residual_p95 || 0).toFixed(4)}</b>
      </div>
    </div>
    <h2>Plane Diagnostics</h2>
    <table class=\"table\"><thead><tr><th>plane</th><th>pts</th><th>views</th><th>res mean</th><th>res p95</th><th>area</th></tr></thead><tbody>${rows}</tbody></table>
    <h2>Registered Views</h2>
    <div class=\"views\">${views || '<div class=\"small\">No view registry stored.</div>'}</div>
    <div class=\"hint\">A useful result should have low alignment loss, low residual p95, and major planes supported by multiple views. Red rows are weak planes: high residual or single-view support.</div>
  `;
}
"""
    template = HTML_TEMPLATE.replace("Full Pointcloud Plane Edit Comparison", "Stage3 Global Plane Report")
    template = template.replace("Editable Plane Parameters", "Stage3 Global Plane Report")
    template = template.replace("Before: DUSt3R full point cloud", "Before: global point cloud + plane colors")
    template = template.replace("After: plane offset edit", "After: editable plane offset preview")
    template = template.replace("</style>", extra_css + "\n</style>")
    template = template.replace('<h2>Plane Equations</h2>', extra_panel + '\n    <h2>Plane Equations</h2>')
    template = template.replace("function renderSide() {", extra_js + "\nfunction renderSide() {")
    template = template.replace("renderSide();", "renderSide();\nrenderStage3Report();")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("__DATA__", json.dumps(html_data, separators=(",", ":"))), encoding="utf-8")

def fuse_scene(scene_key, files, output_dir, args, model=None, device=None):
    global_views = None
    global_alignment_loss = None
    image_paths = []
    global_view_registry = []
    if args.fusion_mode == "dust3r_global":
        image_paths = collect_group_images(files, include_second_view=args.include_second_view)
        global_views, global_alignment_loss = run_dust3r_global_alignment(image_paths, model, device, args)
        global_view_registry = [
            {
                "alignment_view_index": int(view["alignment_view_index"]),
                "image_path": view["image_path"],
                "canonical_path": key,
                "points_hw": list(map(int, view["points"].shape[:2])),
            }
            for key, view in sorted(global_views.items(), key=lambda item: item[1]["alignment_view_index"])
        ]

    loaded = load_scene_inputs(files, args.min_points, args, global_views=global_views)
    if loaded is None:
        return None
    points = loaded["points"]
    colors = loaded["colors"]
    source_views = loaded["source_views"]
    point_file_indices = loaded["point_file_indices"]
    planes = loaded["planes"]
    uf = UnionFind(len(planes))
    merge_records = []
    for i, plane_a in enumerate(planes):
        for j in range(i + 1, len(planes)):
            plane_b = planes[j]
            ok, stats = should_merge(plane_a, plane_b, args)
            if ok:
                uf.union(i, j)
            merge_records.append(
                {
                    "plane_a": i,
                    "plane_b": j,
                    "source_a": [plane_a["source_file"], plane_a["source_plane_id"]],
                    "source_b": [plane_b["source_file"], plane_b["source_plane_id"]],
                    "merged": bool(ok),
                    **stats,
                }
            )

    roots = {}
    for i in range(len(planes)):
        root = uf.find(i)
        roots.setdefault(root, len(roots))

    assignment = np.full((len(points),), -1, dtype=np.int32)
    source_groups = [[] for _ in range(len(roots))]
    for i, plane in enumerate(planes):
        global_id = roots[uf.find(i)]
        assignment[plane["global_point_indices"]] = global_id
        source_groups[global_id].append(
            {
                "source_file": plane["source_file"],
                "source_plane_id": int(plane["source_plane_id"]),
                "points": int(plane["count"]),
            }
        )

    normals = []
    offsets = []
    counts = []
    for global_id in range(len(source_groups)):
        mask = assignment == global_id
        normal, offset, _ = fit_plane_np(points[mask])
        normals.append(normal)
        offsets.append(float(offset))
        counts.append(int(mask.sum()))
    normals = np.stack(normals, axis=0).astype(np.float32) if normals else np.zeros((0, 3), dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int32)
    planes_out = plane_rows(normals, offsets, assignment)
    plane_quality = plane_quality_rows(points, assignment, normals, offsets, source_views, point_file_indices)
    quality_summary = aggregate_plane_quality(plane_quality)

    safe_name = safe_filename(scene_key)
    suffix = "stage3_dust3r_global_fusion" if args.fusion_mode == "dust3r_global" else "stage3_scene_fusion"
    npz_path = output_dir / f"{safe_name}_{suffix}_full_pointcloud_editable_planes_data.npz"
    html_path = output_dir / f"{safe_name}_{suffix}_edit.html"
    ply_path = output_dir / f"{safe_name}_{suffix}.ply"
    mesh_ply_path = output_dir / f"{safe_name}_{suffix}_textured_planes.ply"
    json_path = output_dir / f"{safe_name}_{suffix}_plane_params.json"
    txt_path = output_dir / f"{safe_name}_{suffix}_plane_params.txt"

    np.savez_compressed(
        npz_path,
        points=points.astype(np.float32),
        colors=colors.astype(np.uint8),
        original_colors=colors.astype(np.uint8),
        point_plane_ids=assignment.astype(np.int32),
        plane_ids=np.arange(len(normals), dtype=np.int32),
        plane_normals=normals,
        plane_offsets=offsets,
        plane_inlier_counts=counts,
        source_views=source_views.astype(np.int8),
        point_file_indices=point_file_indices.astype(np.int32),
        plane_quality_json=np.asarray(json.dumps(plane_quality), dtype=object),
        quality_summary_json=np.asarray(json.dumps(quality_summary), dtype=object),
        scene_key=np.asarray(scene_key),
        fusion_mode=np.asarray(args.fusion_mode),
        dust3r_image_paths=np.asarray(image_paths, dtype=object),
        dust3r_global_alignment_loss=np.asarray(global_alignment_loss if global_alignment_loss is not None else np.nan, dtype=np.float32),
        dust3r_view_registry_json=np.asarray(json.dumps(global_view_registry), dtype=object),
        source_groups_json=np.asarray(json.dumps(source_groups), dtype=object),
        merge_records_json=np.asarray(json.dumps(merge_records), dtype=object),
        mapping_records_json=np.asarray(json.dumps(loaded.get("mapping_records", [])), dtype=object),
        skipped_inputs_json=np.asarray(json.dumps(loaded.get("skipped", [])), dtype=object),
    )

    display_colors = colors.copy()
    for plane_id in range(len(planes_out)):
        display_colors[assignment == plane_id] = np.asarray(
            PLANE_COLORS[plane_id % len(PLANE_COLORS)],
            dtype=np.uint8,
        )
    write_ascii_ply(ply_path, points, display_colors)
    mesh_rows = write_textured_plane_mesh_ply(mesh_ply_path, points, colors, assignment, normals, offsets, grid_resolution=args.mesh_grid_resolution)
    write_stage3_report_html(
        html_path,
        points,
        colors,
        assignment,
        planes_out,
        args.edit_delta,
        args.edit_plane,
        args.max_display_points,
        npz_path,
        scene_key,
        args.fusion_mode,
        global_alignment_loss,
        quality_summary,
        plane_quality,
        global_view_registry,
        loaded.get("mapping_records", []),
        image_paths,
    )
    write_params(json_path, txt_path, planes_out, np.arange(len(planes_out), dtype=np.int32), npz_path)
    return {
        "scene_key": scene_key,
        "fusion_mode": args.fusion_mode,
        "input_files": [str(path) for path in files],
        "dust3r_image_paths": image_paths,
        "dust3r_view_registry": global_view_registry,
        "dust3r_global_alignment_loss": global_alignment_loss,
        "mapping_records": loaded.get("mapping_records", []),
        "plane_quality": plane_quality,
        "quality_summary": quality_summary,
        "local_planes": int(len(planes)),
        "global_planes": int(len(planes_out)),
        "merged_pairs": int(sum(1 for row in merge_records if row["merged"])),
        "points": int(len(points)),
        "npz": str(npz_path),
        "html": str(html_path),
        "ply": str(ply_path),
        "mesh_ply": str(mesh_ply_path),
        "mesh_summary": mesh_rows,
        "json": str(json_path),
        "txt": str(txt_path),
    }


def main():
    parser = argparse.ArgumentParser("Fuse Stage2 plane primitives into scene-level editable planes")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*_full_pointcloud_editable_planes_data.npz")
    parser.add_argument("--group_by", default="reference_view", choices=("sample", "scene", "pair_group", "reference_view"))
    parser.add_argument("--fusion_mode", default="local", choices=("local", "dust3r_global"))
    parser.add_argument("--weights_path", default="")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--scene_graph", default="complete")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--schedule", default="cosine")
    parser.add_argument("--global_min_conf", type=float, default=0.0)
    parser.add_argument("--include_second_view", action="store_true")
    parser.add_argument("--min_group_size", type=int, default=1)
    parser.add_argument("--min_points", type=int, default=64)
    parser.add_argument("--max_angle_deg", type=float, default=8.0)
    parser.add_argument("--max_offset", type=float, default=0.06)
    parser.add_argument("--max_mutual_residual", type=float, default=0.06)
    parser.add_argument("--max_centroid_distance", type=float, default=2.5)
    parser.add_argument("--edit_plane", default="largest")
    parser.add_argument("--edit_delta", type=float, default=0.25)
    parser.add_argument("--max_display_points", type=int, default=32000)
    parser.add_argument("--mesh_grid_resolution", type=int, default=48)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No files matched {input_dir / args.pattern}")

    model = None
    device = None
    if args.fusion_mode == "dust3r_global":
        if not args.weights_path:
            raise RuntimeError("--weights_path is required when --fusion_mode dust3r_global")
        setup_dust3r_imports()
        import torch
        from dust3r.model import load_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(args.weights_path, device=device)
        model.eval()

    groups = {}
    for path in files:
        raw = np.load(path)
        key = group_key(raw, path.stem, args.group_by)
        groups.setdefault(key, []).append(path)

    rows = []
    for key, group_files in sorted(groups.items()):
        if len(group_files) < args.min_group_size:
            continue
        row = fuse_scene(key, group_files, output_dir, args, model=model, device=device)
        if row is not None:
            rows.append(row)

    manifest = output_dir / "stage3_scene_fusion_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "scenes": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
