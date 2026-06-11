import argparse
import json
from pathlib import Path

import numpy as np


PLANE_COLORS = np.asarray(
    [
        [229, 57, 53],
        [30, 136, 229],
        [67, 160, 71],
        [251, 140, 0],
        [142, 36, 170],
        [0, 150, 136],
        [117, 117, 117],
        [216, 27, 96],
    ],
    dtype=np.uint8,
)
BACKGROUND = np.asarray([160, 160, 160], dtype=np.uint8)


def fit_plane(points, reference_normal=None):
    centroid = points.mean(axis=0).astype(np.float32)
    centered = points - centroid[None, :]
    _, _, vh = np.linalg.svd(centered.astype(np.float32), full_matrices=False)
    normal = vh[-1].astype(np.float32)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-6)
    if reference_normal is not None and float(np.dot(normal, reference_normal)) < 0.0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    return normal.astype(np.float32), float(offset)


def export_one(path, output_dir, max_planes, min_points):
    raw = np.load(path)
    points = raw["points"].astype(np.float32)
    original_colors = raw["original_colors"].astype(np.uint8) if "original_colors" in raw else raw["colors"].astype(np.uint8)
    point_plane_ids = raw["point_plane_ids"].astype(np.int32)
    plane_ids = raw["plane_ids"].astype(np.int32) if "plane_ids" in raw else np.arange(len(raw["plane_normals"]), dtype=np.int32)
    plane_normals_in = raw["plane_normals"].astype(np.float32) if "plane_normals" in raw else np.zeros((len(plane_ids), 3), dtype=np.float32)

    selected_plane_ids = []
    for pid in plane_ids[:max_planes]:
        if int((point_plane_ids == int(pid)).sum()) >= min_points:
            selected_plane_ids.append(int(pid))

    assignment = np.full(len(points), -1, dtype=np.int32)
    normals = []
    offsets = []
    planes = []
    for out_id, source_pid in enumerate(selected_plane_ids):
        mask = point_plane_ids == source_pid
        reference = plane_normals_in[out_id] if out_id < len(plane_normals_in) else None
        normal, offset = fit_plane(points[mask], reference)
        assignment[mask] = out_id
        normals.append(normal)
        offsets.append(offset)
        dist = np.abs(points[mask] @ normal + offset)
        planes.append(
            {
                "id": out_id,
                "source_plane_id": int(source_pid),
                "normal": [float(x) for x in normal],
                "offset": float(offset),
                "assigned_point_count": int(mask.sum()),
                "assigned_ratio": float(mask.mean()),
                "mean_abs_distance": float(dist.mean()),
                "median_abs_distance": float(np.median(dist)),
                "p95_abs_distance": float(np.percentile(dist, 95)),
                "active": True,
            }
        )

    if normals:
        normals = np.stack(normals).astype(np.float32)
        offsets = np.asarray(offsets, dtype=np.float32)
    else:
        normals = np.zeros((0, 3), dtype=np.float32)
        offsets = np.zeros((0,), dtype=np.float32)

    colors = np.tile(BACKGROUND[None, :], (len(points), 1))
    valid = assignment >= 0
    colors[valid] = PLANE_COLORS[assignment[valid] % len(PLANE_COLORS)]

    stem = Path(path).name.replace("_full_pointcloud_editable_planes_data.npz", "")
    npz_path = output_dir / f"{stem}_teacher_support_refit_assignment.npz"
    json_path = output_dir / f"{stem}_teacher_support_refit.json"
    summary = {
        "input_npz": str(path),
        "method": "teacher_support_svd_refit",
        "num_points_used": int(len(points)),
        "num_planes": int(len(normals)),
        "active_planes": int(len(normals)),
        "background_point_count": int((assignment < 0).sum()),
        "background_ratio": float((assignment < 0).mean()),
        "planes": planes,
    }
    np.savez_compressed(
        npz_path,
        points=points,
        colors=colors.astype(np.uint8),
        original_colors=original_colors,
        assignment=assignment,
        plane_normals=normals,
        plane_offsets=offsets,
        active_planes=np.ones(len(normals), dtype=np.bool_),
    )
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return npz_path, json_path, summary


def main():
    parser = argparse.ArgumentParser("Export support-first teacher baseline with SVD refit")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_planes", type=int, default=8)
    parser.add_argument("--min_points", type=int, default=300)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(input_dir.glob("*_full_pointcloud_editable_planes_data.npz")):
        _, _, summary = export_one(path, output_dir, args.max_planes, args.min_points)
        rows.append(summary)
    (output_dir / "teacher_support_refit_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(output_dir / "teacher_support_refit_summary.json")
    print(f"samples={len(rows)}")


if __name__ == "__main__":
    main()
