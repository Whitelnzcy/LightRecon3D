import argparse
import json
from pathlib import Path

import numpy as np

from make_full_pointcloud_edit_comparison import HTML_TEMPLATE, PLANE_COLORS, deterministic_sample


def write_ascii_ply(path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def compact_active_planes(raw):
    assignment = raw["point_plane_ids"].astype(np.int32)
    normals = raw["plane_normals"].astype(np.float32)
    offsets = raw["plane_offsets"].astype(np.float32)
    if "active_planes" in raw:
        active = raw["active_planes"].astype(bool)
    else:
        active = np.ones(len(normals), dtype=bool)

    active_ids = []
    for plane_id in range(len(normals)):
        if active[plane_id] and np.count_nonzero(assignment == plane_id) > 0:
            active_ids.append(int(plane_id))

    compact = np.full_like(assignment, -1, dtype=np.int32)
    for new_id, old_id in enumerate(active_ids):
        compact[assignment == old_id] = new_id

    return {
        "assignment": compact,
        "source_plane_ids": np.asarray(active_ids, dtype=np.int32),
        "normals": normals[active_ids] if active_ids else np.zeros((0, 3), dtype=np.float32),
        "offsets": offsets[active_ids] if active_ids else np.zeros((0,), dtype=np.float32),
    }


def plane_rows(normals, offsets, assignment):
    rows = []
    for plane_id, (normal, offset) in enumerate(zip(normals, offsets)):
        count = int(np.count_nonzero(assignment == plane_id))
        rows.append(
            {
                "id": int(plane_id),
                "normal": [float(x) for x in normal],
                "offset": float(offset),
                "inlier_count": count,
                "assigned_point_count": count,
                "color": PLANE_COLORS[plane_id % len(PLANE_COLORS)],
            }
        )
    return rows


def write_params(path_json, path_txt, planes, source_plane_ids, source_npz):
    payload = {
        "source_npz": str(source_npz),
        "num_planes": len(planes),
        "source_plane_ids": [int(x) for x in source_plane_ids],
        "planes": planes,
    }
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = []
    for plane in planes:
        n = plane["normal"]
        d = plane["offset"]
        lines.append(
            f"Plane {plane['id']} source={source_plane_ids[plane['id']]} "
            f"count={plane['assigned_point_count']} "
            f"{n[0]:.8f}*x + {n[1]:.8f}*y + {n[2]:.8f}*z + {d:.8f} = 0"
        )
    path_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_html(path, points, colors, assignment, planes, edit_delta, edit_plane, max_display_points, source_npz):
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
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        HTML_TEMPLATE.replace("__DATA__", json.dumps(html_data, separators=(",", ":"))),
        encoding="utf-8",
    )


def export_one(input_npz, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(input_npz)
    points = raw["points"].astype(np.float32)
    colors_key = "original_colors" if "original_colors" in raw else "colors"
    original_colors = raw[colors_key].astype(np.uint8)
    compact = compact_active_planes(raw)
    assignment = compact["assignment"]
    planes = plane_rows(compact["normals"], compact["offsets"], assignment)

    display_colors = original_colors.copy()
    valid = assignment >= 0
    for plane_id in range(len(planes)):
        display_colors[assignment == plane_id] = np.asarray(
            PLANE_COLORS[plane_id % len(PLANE_COLORS)],
            dtype=np.uint8,
        )

    stem = input_npz.name.replace("_full_pointcloud_editable_planes_data.npz", "")
    stem = stem.replace("_bounded_support_head_assignment.npz", "")
    compatible_npz = output_dir / f"{stem}_stage2_geometry_refit_full_pointcloud_editable_planes_data.npz"
    html_path = output_dir / f"{stem}_stage2_geometry_refit_edit.html"
    ply_path = output_dir / f"{stem}_stage2_geometry_refit.ply"
    json_path = output_dir / f"{stem}_stage2_geometry_refit_plane_params.json"
    txt_path = output_dir / f"{stem}_stage2_geometry_refit_plane_params.txt"

    np.savez_compressed(
        compatible_npz,
        points=points,
        colors=original_colors,
        original_colors=original_colors,
        point_plane_ids=assignment.astype(np.int32),
        plane_ids=np.arange(len(planes), dtype=np.int32),
        plane_normals=compact["normals"].astype(np.float32),
        plane_offsets=compact["offsets"].astype(np.float32),
        plane_inlier_counts=np.asarray([p["assigned_point_count"] for p in planes], dtype=np.int32),
        source_plane_ids=compact["source_plane_ids"].astype(np.int32),
    )
    write_html(
        html_path,
        points,
        original_colors,
        assignment,
        planes,
        args.edit_delta,
        args.edit_plane,
        args.max_display_points,
        input_npz,
    )
    write_ascii_ply(ply_path, points, display_colors)
    write_params(json_path, txt_path, planes, compact["source_plane_ids"], input_npz)
    return {
        "input": str(input_npz),
        "compatible_npz": str(compatible_npz),
        "html": str(html_path),
        "ply": str(ply_path),
        "json": str(json_path),
        "txt": str(txt_path),
        "points": int(len(points)),
        "planes": len(planes),
    }


def main():
    parser = argparse.ArgumentParser("Export Stage2 geometry/refit NPZ files to editable HTML/PLY")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*_full_pointcloud_editable_planes_data.npz")
    parser.add_argument("--edit_plane", default="largest")
    parser.add_argument("--edit_delta", type=float, default=0.25)
    parser.add_argument("--max_display_points", type=int, default=28000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No files matched {input_dir / args.pattern}")
    rows = [export_one(path, output_dir, args) for path in files]
    manifest = output_dir / "stage2_geometry_refit_editables_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(manifest)
    print(f"exported={len(rows)}")


if __name__ == "__main__":
    main()
