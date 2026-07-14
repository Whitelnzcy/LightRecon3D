"""Fair plane baselines on one cached DUSt3R globally aligned point cloud.

The cache is deliberately method-agnostic: every method receives exactly the
same XYZ samples, colours, confidence values, view indices and pixel indices.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PLANE_COLORS = np.asarray([[230, 57, 70], [29, 53, 87], [69, 123, 157], [42, 157, 143],
                           [233, 196, 106], [244, 162, 97], [154, 95, 156], [118, 200, 147]], np.uint8)


CACHE_SCHEMA_VERSION = 1
OUTPUT_SCHEMA_VERSION = 1


def global_cache_keep_mask(points, confidence, min_conf):
    """Shared finite/confidence filter for every identical-cache method."""

    points = np.asarray(points, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if points.shape != (len(confidence), 3):
        raise ValueError("points and confidence must have matching (N,3)/(N,) shapes")
    return (
        np.isfinite(points).all(axis=1)
        & (np.max(np.abs(points), axis=1) < 1e5)
        & (confidence >= float(min_conf))
    )


def fit_plane(points):
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1].astype(np.float32)
    offset = -float(normal @ center)
    return normal, offset, np.abs(points @ normal + offset).astype(np.float32)


def euclidean_components(points, radius, min_points):
    """Return index arrays for radius-connected components (no sklearn needed)."""
    points = np.asarray(points, dtype=np.float32)
    if not len(points):
        return []
    cells = np.floor(points / radius).astype(np.int64)
    buckets = {}
    for index, cell in enumerate(cells):
        buckets.setdefault(tuple(cell), []).append(index)
    seen = np.zeros(len(points), dtype=bool)
    result = []
    for seed in range(len(points)):
        if seen[seed]:
            continue
        stack = [seed]
        seen[seed] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            cell = cells[current]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for neighbor in buckets.get(tuple(cell + (dx, dy, dz)), ()):
                            if not seen[neighbor] and np.linalg.norm(points[current] - points[neighbor]) <= radius:
                                seen[neighbor] = True
                                stack.append(neighbor)
        if len(component) >= min_points:
            result.append(np.asarray(component, dtype=np.int64))
    return sorted(result, key=len, reverse=True)


def voxel_components(points, voxel_size, min_points):
    """Scalable occupancy components for large planar inlier sets."""

    points = np.asarray(points, dtype=np.float32)
    if not len(points):
        return []
    cells = np.floor(points / float(voxel_size)).astype(np.int64)
    buckets = {}
    for index, cell in enumerate(cells):
        buckets.setdefault(tuple(cell), []).append(index)
    unseen = set(buckets)
    result = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        stack = [seed]
        component = []
        while stack:
            cell = stack.pop()
            component.extend(buckets[cell])
            cx, cy, cz = cell
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor = (cx + dx, cy + dy, cz + dz)
                        if neighbor in unseen:
                            unseen.remove(neighbor)
                            stack.append(neighbor)
        if len(component) >= int(min_points):
            result.append(np.asarray(component, dtype=np.int64))
    return sorted(result, key=len, reverse=True)


def sequential_plane_ransac(points, distance_threshold=0.03, iterations=1000,
                            min_inliers=200, cluster_radius=0.08,
                            min_component_points=100, max_planes=64, seed=0,
                            hypothesis_max_points=50000,
                            component_exact_max_points=20000):
    points = np.asarray(points, dtype=np.float32)
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(points), dtype=np.int64)
    supports = []
    while len(remaining) >= min_inliers and len(supports) < max_planes:
        xyz = points[remaining]
        if len(xyz) > int(hypothesis_max_points):
            score_indices = rng.choice(
                len(xyz), int(hypothesis_max_points), replace=False
            )
            score_xyz = xyz[score_indices]
        else:
            score_xyz = xyz
        best_model = None
        best_count = -1
        for _ in range(iterations):
            sample = score_xyz[rng.choice(len(score_xyz), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = float(np.linalg.norm(normal))
            if norm < 1e-8:
                continue
            normal /= norm
            offset = -float(normal @ sample[0])
            count = int(
                (np.abs(score_xyz @ normal + offset) <= distance_threshold).sum()
            )
            if count > best_count:
                best_count = count
                best_model = (normal.copy(), offset)
        if best_model is None:
            break
        normal, offset = best_model
        candidate = np.abs(xyz @ normal + offset) <= distance_threshold
        if int(candidate.sum()) < min_inliers:
            break
        candidate_global = remaining[candidate]
        normal, offset, _ = fit_plane(points[candidate_global])
        refined = np.abs(xyz @ normal + offset) <= distance_threshold
        candidate_global = remaining[refined]
        if len(candidate_global) <= int(component_exact_max_points):
            components = euclidean_components(
                points[candidate_global], cluster_radius, min_component_points
            )
        else:
            components = voxel_components(
                points[candidate_global], cluster_radius, min_component_points
            )
        accepted = []
        for component in components:
            indices = candidate_global[component]
            if len(indices) >= min_component_points:
                supports.append(indices)
                accepted.append(indices)
                if len(supports) >= max_planes:
                    break
        # Remove every geometric inlier, including tiny components, so the same
        # dominant infinite plane cannot be rediscovered indefinitely.
        remaining = remaining[~refined]
        if not accepted and len(remaining) < min_inliers:
            break
    return supports


def supports_to_primitives(points, supports):
    assignment = np.full(len(points), -1, dtype=np.int32)
    normals, offsets, counts = [], [], []
    for indices in supports:
        normal, offset, _ = fit_plane(points[indices])
        plane_id = len(normals)
        assignment[indices] = plane_id
        normals.append(normal)
        offsets.append(offset)
        counts.append(len(indices))
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    return assignment, normals, np.asarray(offsets, np.float32), np.asarray(counts, np.int32)


def load_global_cache(path, min_conf):
    raw = np.load(path, allow_pickle=False)
    required = {"points", "colors", "confidence", "view_indices", "pixel_xy"}
    missing = sorted(required - set(raw.files))
    if missing:
        raise ValueError(f"global cache missing fields: {missing}")
    points = raw["points"].astype(np.float32)
    keep = global_cache_keep_mask(points, raw["confidence"], min_conf)
    return {key: raw[key][keep] for key in required}


def save_result(output_dir, scene_key, cache, supports, method, runtime, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    points, colors = cache["points"], cache["colors"].astype(np.uint8)
    assignment, normals, offsets, counts = supports_to_primitives(points, supports)
    stem = f"{scene_key}_{method}_full_pointcloud_editable_planes_data"
    npz_path = output_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path, schema_version=np.asarray(OUTPUT_SCHEMA_VERSION),
        points=points, colors=colors, original_colors=colors,
        point_plane_ids=assignment, plane_ids=np.arange(len(normals), dtype=np.int32),
        plane_normals=normals, plane_offsets=offsets, plane_inlier_counts=counts,
        source_views=cache["view_indices"].astype(np.int32),
        pixel_xy=cache["pixel_xy"].astype(np.int32), method=np.asarray(method),
        runtime_seconds=np.asarray(runtime, np.float64),
        config_json=np.asarray(json.dumps(config, sort_keys=True)), scene_key=np.asarray(scene_key),
    )
    display = colors.copy()
    for plane_id in range(len(normals)):
        display[assignment == plane_id] = np.asarray(PLANE_COLORS[plane_id % len(PLANE_COLORS)], np.uint8)
    ply_path = output_dir / f"{scene_key}_{method}.ply"
    with ply_path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(points))
        handle.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(handle, np.column_stack((points, display)), fmt="%.7g %.7g %.7g %d %d %d")
    params_json = output_dir / f"{scene_key}_{method}_plane_params.json"
    params_txt = output_dir / f"{scene_key}_{method}_plane_params.txt"
    rows = [{"id": i, "normal": normals[i].tolist(), "offset": float(offsets[i]),
             "assigned_point_count": int(counts[i])} for i in range(len(normals))]
    params_json.write_text(json.dumps({"source_npz": str(npz_path), "planes": rows}, indent=2), encoding="utf-8")
    params_txt.write_text("\n".join(f"plane {r['id']}: n={r['normal']} d={r['offset']:.8f} points={r['assigned_point_count']}" for r in rows), encoding="utf-8")
    return {"method": method, "planes": len(normals), "assigned_points": int((assignment >= 0).sum()),
            "runtime_seconds": runtime, "npz": str(npz_path), "ply": str(ply_path),
            "json": str(params_json), "txt": str(params_txt)}


def main():
    parser = argparse.ArgumentParser("Sequential RANSAC on a shared DUSt3R global cloud cache")
    parser.add_argument("--global_cloud_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_key", default="scene")
    parser.add_argument("--min_conf", type=float, default=1.0)
    parser.add_argument("--distance_threshold", type=float, default=0.03)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--min_inliers", type=int, default=200)
    parser.add_argument("--cluster_radius", type=float, default=0.08)
    parser.add_argument("--min_component_points", type=int, default=100)
    parser.add_argument("--max_planes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hypothesis_max_points", type=int, default=50000)
    parser.add_argument("--component_exact_max_points", type=int, default=20000)
    args = parser.parse_args()
    config = vars(args).copy()
    cache = load_global_cache(args.global_cloud_npz, args.min_conf)
    started = time.perf_counter()
    supports = sequential_plane_ransac(
        cache["points"], args.distance_threshold, args.iterations, args.min_inliers,
        args.cluster_radius, args.min_component_points, args.max_planes, args.seed,
        args.hypothesis_max_points, args.component_exact_max_points)
    runtime = time.perf_counter() - started
    row = save_result(Path(args.output_dir), args.scene_key, cache, supports,
                      "global_ransac_cc", runtime, config)
    manifest = Path(args.output_dir) / "global_plane_baseline_manifest.json"
    manifest.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
