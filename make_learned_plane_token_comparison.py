import argparse
import json
from pathlib import Path

import numpy as np

from make_full_pointcloud_edit_comparison import HTML_TEMPLATE


def load_learned(path):
    raw = np.load(path)
    return {
        "points": raw["points"].astype(np.float32),
        "colors": raw["colors"].astype(np.uint8),
        "original_colors": raw["original_colors"].astype(np.uint8),
        "assignment": raw["assignment"].astype(np.int32),
        "plane_normals": raw["plane_normals"].astype(np.float32),
        "plane_offsets": raw["plane_offsets"].astype(np.float32),
    }


def rgb_string(color):
    return f"rgb({int(color[0])},{int(color[1])},{int(color[2])})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned_npz", required=True)
    parser.add_argument("--learned_json", required=True)
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--edit_plane", default="largest")
    parser.add_argument("--edit_delta", type=float, default=0.25)
    parser.add_argument("--max_display_points", type=int, default=28000)
    args = parser.parse_args()

    data = load_learned(args.learned_npz)
    summary = json.loads(Path(args.learned_json).read_text(encoding="utf-8"))
    points = data["points"]
    assignment = data["assignment"]
    plane_normals = data["plane_normals"]
    plane_offsets = data["plane_offsets"]

    counts = np.bincount(assignment, minlength=len(plane_normals))
    if args.edit_plane == "largest":
        edit_plane = int(np.argmax(counts))
    else:
        edit_plane = int(args.edit_plane)
    moved_mask = assignment == edit_plane
    edited_points = points.copy()
    edited_points[moved_mask] = edited_points[moved_mask] - float(args.edit_delta) * plane_normals[edit_plane]

    total = len(points)
    base_idx = np.linspace(0, total - 1, min(total, args.max_display_points), dtype=np.int64)
    moved_idx = np.flatnonzero(moved_mask)
    if len(moved_idx) > 0:
        moved_keep = moved_idx[
            np.linspace(0, len(moved_idx) - 1, min(len(moved_idx), args.max_display_points // 2), dtype=np.int64)
        ]
        display_idx = np.unique(np.concatenate([base_idx, moved_keep]))
    else:
        display_idx = base_idx
    if len(display_idx) > args.max_display_points:
        display_idx = display_idx[
            np.linspace(0, len(display_idx) - 1, args.max_display_points, dtype=np.int64)
        ]

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    scale = 0.74 / max(span, 1e-6)

    planes = []
    for i, (n, d) in enumerate(zip(plane_normals, plane_offsets)):
        plane_summary = summary["planes"][i] if i < len(summary.get("planes", [])) else {}
        color = data["colors"][np.argmax(assignment == i)] if np.any(assignment == i) else np.array([128, 128, 128])
        planes.append(
            {
                "id": int(i),
                "normal": [float(x) for x in n],
                "offset": float(d),
                "inlier_count": int(counts[i]),
                "assigned_point_count": int(counts[i]),
                "color": [int(x) for x in color],
                "mean_abs_distance_normalized": plane_summary.get("mean_abs_distance_normalized"),
            }
        )

    html_data = {
        "input_npz": args.learned_npz,
        "total_points": int(total),
        "display_points": int(len(display_idx)),
        "moved_points": int(np.sum(moved_mask)),
        "center": [float(x) for x in center],
        "scale": scale,
        "planes": planes,
        "edit": {
            "plane_id": int(edit_plane),
            "delta": float(args.edit_delta),
            "offset_before": float(plane_offsets[edit_plane]),
            "offset_after": float(plane_offsets[edit_plane] + float(args.edit_delta)),
        },
        "before_points": points[display_idx].round(5).tolist(),
        "after_points": edited_points[display_idx].round(5).tolist(),
        "sample_moved": moved_mask[display_idx].astype(bool).tolist(),
        "sample_colors": [rgb_string(c) for c in data["colors"][display_idx]],
    }
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(html_data, separators=(",", ":")))
    html = html.replace("Before: DUSt3R full point cloud", "Before: learned plane-token assignment")
    html = html.replace("After: plane offset edit", "After: learned plane-token offset edit")
    output = Path(args.output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(output)
    print(
        f"total_points={total} display_points={len(display_idx)} "
        f"edit_plane={edit_plane} moved_points={int(np.sum(moved_mask))}"
    )


if __name__ == "__main__":
    main()
