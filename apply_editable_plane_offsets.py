import argparse
import json
import re
from pathlib import Path


def extract_data(input_html):
    text = Path(input_html).read_text(encoding="utf-8")
    match = re.search(r"const DATA = (.*?);\n(?:DATA\.planeById|const bar)", text, re.S)
    if not match:
        raise RuntimeError(f"Cannot find DATA block in {input_html}")
    data = json.loads(match.group(1))
    data["planeById"] = {int(p["id"]): p for p in data["planes"]}
    return data


def parse_edits(edit_args):
    edits = {}
    for item in edit_args:
        if ":" not in item:
            raise ValueError(f"Invalid --edit {item!r}; expected PLANE_ID:DELTA")
        plane_id, delta = item.split(":", 1)
        edits[int(plane_id)] = float(delta)
    return edits


def edited_point(point, plane_id, data, edits):
    if plane_id < 0 or plane_id not in edits:
        return point
    plane = data["planeById"][plane_id]
    normal = plane["normal"]
    delta = edits[plane_id]
    return [point[i] - delta * normal[i] for i in range(3)]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_html", required=True)
    parser.add_argument("--output_ply", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--edit",
        action="append",
        default=[],
        help="Plane edit in PLANE_ID:DELTA form. Example: --edit 2:0.25",
    )
    args = parser.parse_args()

    data = extract_data(args.input_html)
    edits = parse_edits(args.edit)
    if not edits:
        raise ValueError("At least one --edit PLANE_ID:DELTA is required")

    points = []
    moved_counts = {pid: 0 for pid in edits}
    for point, plane_id in zip(data["points"], data["point_plane_ids"]):
        plane_id = int(plane_id)
        if plane_id in edits:
            moved_counts[plane_id] += 1
        points.append(edited_point(point, plane_id, data, edits))

    Path(args.output_ply).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    write_ply(args.output_ply, points, data["colors"])

    report = {
        "input_html": args.input_html,
        "output_ply": args.output_ply,
        "edits": edits,
        "moved_counts": moved_counts,
        "total_points": len(points),
        "planes": [
            {
                "id": int(p["id"]),
                "equation_before": "nx*x + ny*y + nz*z + d = 0",
                "normal": p["normal"],
                "offset_before": p["offset"],
                "offset_after": p["offset"] + edits.get(int(p["id"]), 0.0),
                "delta": edits.get(int(p["id"]), 0.0),
                "inlier_count": p.get("inlier_count"),
            }
            for p in data["planes"]
        ],
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output_ply)
    print(args.output_json)
    print(f"total_points={len(points)} moved_counts={moved_counts}")


if __name__ == "__main__":
    main()
