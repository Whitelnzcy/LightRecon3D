#!/usr/bin/env python3
"""Render separate 3D common-cache views for GT and arbitrary predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from visualize_common_point_cache_partitions import prediction_to_gt_color_ids
from visualize_research_practice_final_3d import (
    BACKGROUND,
    PLANE_PALETTE,
    UNASSIGNED,
    CameraView,
    automatic_views,
    canonical_frame,
    load_font,
    prepare_projection,
    stable_sample_indices,
    visible_raster_samples,
    zbuffer_render,
)


def parse_prediction(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"prediction must be NAME=NPZ, got {value!r}")
    return name, Path(path)


def matched_plane_colors(
    labels: np.ndarray, gt_labels: np.ndarray
) -> np.ndarray:
    mapping = prediction_to_gt_color_ids(labels, gt_labels)
    output = np.broadcast_to(UNASSIGNED, (len(labels), 3)).copy()
    for plane_id, color_id in mapping.items():
        output[labels == plane_id] = PLANE_PALETTE[
            color_id % len(PLANE_PALETTE)
        ]
    return output


def headed_panel(
    render: Image.Image, title: str, subtitle: str, width: int
) -> Image.Image:
    header_height = 66
    panel = Image.new("RGB", (width, render.height + header_height), "white")
    panel.paste(render, (0, header_height))
    draw = ImageDraw.Draw(panel)
    draw.text((14, 9), title, fill=(25, 52, 83), font=load_font(18, True))
    draw.text((14, 37), subtitle, fill=(78, 91, 107), font=load_font(12))
    draw.rectangle(
        (0, header_height, width - 1, panel.height - 1),
        outline=(196, 205, 216),
        width=2,
    )
    return panel


def metrics_index(path: Path | None, scene_name: str) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["method"]): row
        for row in rows
        if str(row.get("scene_name")) == scene_name
    }


def render_scene(
    *,
    scene_name: str,
    gt_npz: Path,
    predictions: list[tuple[str, Path]],
    output_dir: Path,
    metrics_json: Path | None = None,
    width: int = 760,
    height: int = 620,
    max_points: int = 80_000,
    point_radius: int = 1,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    gt = np.load(gt_npz, allow_pickle=False)
    archives = [(name, np.load(path, allow_pickle=False), path) for name, path in predictions]
    try:
        points = gt["points"].astype(np.float32)
        original_colors = gt["colors"].astype(np.uint8)
        gt_labels = gt["point_plane_ids"].astype(np.int32)
        for name, archive, path in archives:
            if archive["points"].shape != points.shape or not np.allclose(
                archive["points"], points, atol=1e-5
            ):
                raise ValueError(f"{name} does not use the identical 3D cache: {path}")
        sample_indices = stable_sample_indices(len(points), max_points)
        center, basis = canonical_frame(points)
        views = automatic_views(points, center, basis)
        metric_rows = metrics_index(metrics_json, scene_name)
        gt_colors = matched_plane_colors(gt_labels, gt_labels)
        render_sets: list[tuple[str, str, np.ndarray, np.ndarray]] = [
            ("rgb_cloud", "RGB 3D global cloud", original_colors, gt_labels),
            ("ground_truth", "Structured3D GT planes", gt_colors, gt_labels),
        ]
        for name, archive, _ in archives:
            labels = archive["point_plane_ids"].astype(np.int32)
            render_sets.append(
                (name, name.replace("_", " "), matched_plane_colors(labels, gt_labels), labels)
            )

        files: list[str] = []
        for view_index, view in enumerate(views):
            projection = prepare_projection(
                points,
                sample_indices,
                center,
                basis,
                view,
                width,
                height,
            )
            raster = visible_raster_samples(projection, point_radius)
            for file_name, display_name, colors, labels in render_sets:
                background = (
                    np.asarray([22, 29, 38], dtype=np.uint8)
                    if file_name == "rgb_cloud"
                    else BACKGROUND
                )
                image = zbuffer_render(projection, colors, raster, background)
                assigned = float((labels >= 0).mean())
                metric_text = ""
                if file_name not in {"rgb_cloud", "ground_truth"}:
                    archive = next(row[1] for row in archives if row[0] == file_name)
                    method = (
                        str(archive["method"].item())
                        if "method" in archive.files
                        else file_name
                    )
                    row = metric_rows.get(method)
                    if row is not None:
                        metric_text = (
                            f" | 3D plane F1 {float(row['plane_f1']):.3f}"
                            f" | IoU {float(row['support_matched_iou']):.3f}"
                        )
                panel = headed_panel(
                    image,
                    f"{scene_name} | {display_name} | 3D common cloud",
                    (
                        f"view {view.name} ({view.yaw_deg:g}, {view.pitch_deg:g})"
                        f" | assigned {assigned:.1%}{metric_text}"
                    ),
                    width,
                )
                path = output_dir / f"view_{view_index:02d}_{file_name}.png"
                panel.save(path, optimize=True)
                files.append(str(path))
        result = {
            "scene_name": scene_name,
            "points": int(len(points)),
            "rendered_sample": int(len(sample_indices)),
            "views": [view.__dict__ for view in views],
            "files": files,
            "identical_3d_cache": True,
        }
        (output_dir / "visualization_manifest.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return result
    finally:
        gt.close()
        for _, archive, _ in archives:
            archive.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--gt_npz", required=True)
    parser.add_argument("--prediction", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics_json", default="")
    parser.add_argument("--width", type=int, default=760)
    parser.add_argument("--height", type=int, default=620)
    parser.add_argument("--max_points", type=int, default=80000)
    parser.add_argument("--point_radius", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = render_scene(
        scene_name=args.scene_name,
        gt_npz=Path(args.gt_npz),
        predictions=[parse_prediction(value) for value in args.prediction],
        output_dir=Path(args.output_dir),
        metrics_json=Path(args.metrics_json) if args.metrics_json else None,
        width=args.width,
        height=args.height,
        max_points=args.max_points,
        point_radius=args.point_radius,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
