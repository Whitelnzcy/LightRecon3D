import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_case(path):
    raw = np.load(path)
    points = raw["points"].astype(np.float32)
    colors = raw["colors"].astype(np.uint8)
    point_plane_ids = raw["point_plane_ids"].astype(np.int32)
    plane_ids = raw["plane_ids"].astype(np.int32)
    plane_normals = raw["plane_normals"].astype(np.float32)
    plane_offsets = raw["plane_offsets"].astype(np.float32)
    plane_inlier_counts = raw["plane_inlier_counts"].astype(np.int32)

    planes = []
    for plane_id, normal, offset, inlier_count in zip(
        plane_ids, plane_normals, plane_offsets, plane_inlier_counts
    ):
        assigned_count = int(np.sum(point_plane_ids == int(plane_id)))
        planes.append(
            {
                "id": int(plane_id),
                "normal": [float(x) for x in normal],
                "offset": float(offset),
                "inlier_count": int(inlier_count),
                "assigned_point_count": assigned_count,
            }
        )

    return {
        "path": Path(path),
        "points": points,
        "colors": colors,
        "point_plane_ids": point_plane_ids,
        "planes": planes,
    }


def select_edit_plane(planes, edit_plane):
    if not planes:
        return None
    if edit_plane == "largest":
        return max(planes, key=lambda p: p["assigned_point_count"])["id"]
    return int(edit_plane)


def write_ply(path, points, colors):
    with Path(path).open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def export_edit(case, output_dir, edit_plane_id, edit_delta):
    plane = next((p for p in case["planes"] if p["id"] == edit_plane_id), None)
    if plane is None:
        raise ValueError(f"Plane {edit_plane_id} not found in {case['path']}")

    edited_points = case["points"].copy()
    mask = case["point_plane_ids"] == edit_plane_id
    normal = np.asarray(plane["normal"], dtype=np.float32)
    edited_points[mask] = edited_points[mask] - float(edit_delta) * normal

    stem = case["path"].name.replace("_full_pointcloud_editable_planes_data.npz", "")
    ply_path = output_dir / f"{stem}_edit_plane{edit_plane_id}_d{edit_delta:+.3f}_full_points.ply"
    report_path = output_dir / f"{stem}_edit_plane{edit_plane_id}_d{edit_delta:+.3f}_report.json"
    write_ply(ply_path, edited_points, case["colors"])
    report = {
        "input_npz": str(case["path"]),
        "output_ply": str(ply_path),
        "total_points": int(len(case["points"])),
        "edit": {"plane_id": int(edit_plane_id), "delta": float(edit_delta)},
        "moved_points": int(np.sum(mask)),
        "plane_before": plane,
        "plane_after": {
            **plane,
            "offset": float(plane["offset"] + float(edit_delta)),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return ply_path, report_path, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*_full_pointcloud_editable_planes_data.npz")
    parser.add_argument("--edit_plane", default="largest", help="'largest' or a numeric plane id")
    parser.add_argument("--edit_delta", type=float, default=0.25)
    parser.add_argument("--export_edits", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(input_dir.glob(args.pattern))
    if not npz_paths:
        raise FileNotFoundError(f"No files matched {args.pattern!r} under {input_dir}")

    cases = []
    for path in npz_paths:
        case = load_case(path)
        edit_plane_id = select_edit_plane(case["planes"], args.edit_plane)
        edit_report = None
        if args.export_edits and edit_plane_id is not None:
            _, _, edit_report = export_edit(case, output_dir, edit_plane_id, args.edit_delta)
        assigned_total = int(np.sum(case["point_plane_ids"] >= 0))
        cases.append(
            {
                "input_npz": str(path),
                "total_points": int(len(case["points"])),
                "plane_count": int(len(case["planes"])),
                "assigned_points": assigned_total,
                "unassigned_points": int(len(case["points"]) - assigned_total),
                "planes": case["planes"],
                "edit_report": edit_report,
            }
        )

    json_path = output_dir / "full_pointcloud_plane_edit_summary.json"
    csv_path = output_dir / "full_pointcloud_plane_edit_summary.csv"
    md_path = output_dir / "full_pointcloud_plane_edit_summary.md"

    json_path.write_text(json.dumps({"cases": cases}, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "input_npz",
                "total_points",
                "plane_count",
                "assigned_points",
                "unassigned_points",
                "plane_id",
                "nx",
                "ny",
                "nz",
                "d",
                "inlier_count",
                "assigned_point_count",
            ]
        )
        for case in cases:
            for plane in case["planes"]:
                writer.writerow(
                    [
                        case["input_npz"],
                        case["total_points"],
                        case["plane_count"],
                        case["assigned_points"],
                        case["unassigned_points"],
                        plane["id"],
                        *plane["normal"],
                        plane["offset"],
                        plane["inlier_count"],
                        plane["assigned_point_count"],
                    ]
                )

    lines = ["# Full Pointcloud Editable Plane Summary", ""]
    for i, case in enumerate(cases):
        lines.append(f"## Case {i}: {Path(case['input_npz']).name}")
        lines.append("")
        lines.append(f"- full points: {case['total_points']}")
        lines.append(f"- major planes: {case['plane_count']}")
        lines.append(f"- assigned points: {case['assigned_points']}")
        lines.append(f"- unassigned points: {case['unassigned_points']}")
        if case["edit_report"]:
            edit = case["edit_report"]["edit"]
            lines.append(
                f"- exported edit: plane {edit['plane_id']} delta {edit['delta']:+.3f}, "
                f"moved {case['edit_report']['moved_points']} points"
            )
            lines.append(f"- edited PLY: {case['edit_report']['output_ply']}")
        lines.append("")
        lines.append("| plane | equation | inliers | assigned points |")
        lines.append("| --- | --- | ---: | ---: |")
        for plane in case["planes"]:
            nx, ny, nz = plane["normal"]
            lines.append(
                f"| {plane['id']} | {nx:.6f}*x + {ny:.6f}*y + {nz:.6f}*z + "
                f"{plane['offset']:.6f} = 0 | {plane['inlier_count']} | "
                f"{plane['assigned_point_count']} |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json_path)
    print(csv_path)
    print(md_path)
    print(f"cases={len(cases)}")


if __name__ == "__main__":
    main()
