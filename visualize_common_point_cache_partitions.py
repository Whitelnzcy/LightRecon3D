#!/usr/bin/env python3
"""Render separate, headed images for common-point-cache plane partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_global_plane_baselines import iou_matrix, linear_sum_assignment


COLORS = np.asarray(
    [
        [0, 114, 178],
        [230, 159, 0],
        [0, 158, 115],
        [204, 121, 167],
        [86, 180, 233],
        [213, 94, 0],
        [240, 228, 66],
        [0, 0, 0],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
    ],
    dtype=np.uint8,
)


def prediction_to_gt_color_ids(
    prediction: np.ndarray, gt: np.ndarray
) -> dict[int, int]:
    pred_ids = np.asarray(
        sorted(int(value) for value in np.unique(prediction) if value >= 0)
    )
    gt_ids = np.asarray(sorted(int(value) for value in np.unique(gt) if value >= 0))
    matrix = iou_matrix(prediction, gt, pred_ids, gt_ids)
    rows, columns = (
        linear_sum_assignment(1.0 - matrix)
        if matrix.size
        else (np.array([], dtype=int), np.array([], dtype=int))
    )
    mapping = {
        int(pred_ids[row]): int(gt_ids[column])
        for row, column in zip(rows, columns)
    }
    next_id = max([*gt_ids.tolist(), -1]) + 1
    for pred_id in pred_ids:
        if int(pred_id) not in mapping:
            mapping[int(pred_id)] = next_id
            next_id += 1
    return mapping


def dense_view_labels(
    labels_archive: np.lib.npyio.NpzFile,
    coordinate_archive: np.lib.npyio.NpzFile,
    view_index: int,
    height: int,
    width: int,
) -> np.ndarray:
    result = np.full((height, width), -1, dtype=np.int32)
    if len(labels_archive["point_plane_ids"]) != len(
        coordinate_archive["view_indices"]
    ):
        raise ValueError("labels and coordinate cache do not have identical order")
    mask = coordinate_archive["view_indices"] == view_index
    xy = coordinate_archive["pixel_xy"][mask].astype(np.int32)
    result[xy[:, 1], xy[:, 0]] = labels_archive["point_plane_ids"][mask].astype(
        np.int32
    )
    return result


def colorize(labels: np.ndarray, mapping: dict[int, int] | None = None) -> Image.Image:
    output = np.full((*labels.shape, 3), 245, dtype=np.uint8)
    for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
        color_id = mapping.get(label, label) if mapping is not None else label
        output[labels == label] = COLORS[color_id % len(COLORS)]
    return Image.fromarray(output, mode="RGB")


def with_header(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    header_height = 62
    canvas = Image.new("RGB", (image.width, image.height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=13)
    draw.text((12, 8), title, fill=(25, 50, 82), font=font)
    draw.text((12, 36), subtitle, fill=(70, 85, 105), font=small)
    return canvas


def parse_prediction(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"prediction must be NAME=NPZ, got {value!r}")
    return name, Path(path)


def render(
    *,
    gt_npz: Path,
    global_cloud_npz: Path,
    predictions: list[tuple[str, Path]],
    output_dir: Path,
    scene_name: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    gt = np.load(gt_npz, allow_pickle=False)
    cloud = np.load(global_cloud_npz, allow_pickle=False)
    prediction_archives = [
        (name, np.load(path, allow_pickle=False), path)
        for name, path in predictions
    ]
    try:
        registry = {
            int(row["alignment_view_index"]): row
            for row in json.loads(str(cloud["dust3r_view_registry_json"].item()))
        }
        files: list[str] = []
        for view_index in sorted(registry):
            height, width = map(int, registry[view_index]["points_hw"])
            gt_labels = dense_view_labels(gt, gt, view_index, height, width)
            rgb_bgr = cv2.imread(str(registry[view_index]["image_path"]))
            if rgb_bgr is None:
                raise FileNotFoundError(registry[view_index]["image_path"])
            rgb = Image.fromarray(
                cv2.cvtColor(
                    cv2.resize(rgb_bgr, (width, height), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )
            )
            rgb_path = output_dir / f"view_{view_index:02d}_rgb.png"
            with_header(
                rgb,
                f"{scene_name} | RGB input | view {view_index}",
                "same ordered image used by every method",
            ).save(rgb_path)
            files.append(str(rgb_path))

            gt_path = output_dir / f"view_{view_index:02d}_ground_truth.png"
            with_header(
                colorize(gt_labels),
                f"{scene_name} | Structured3D GT | view {view_index}",
                "colors are GT structural plane identities",
            ).save(gt_path)
            files.append(str(gt_path))

            for name, archive, _ in prediction_archives:
                pred_labels = dense_view_labels(
                    archive, gt, view_index, height, width
                )
                mapping = prediction_to_gt_color_ids(
                    archive["point_plane_ids"], gt["point_plane_ids"]
                )
                path = output_dir / f"view_{view_index:02d}_{name}.png"
                with_header(
                    colorize(pred_labels, mapping),
                    f"{scene_name} | {name} | view {view_index}",
                    "Hungarian-matched colors; white pixels are unassigned",
                ).save(path)
                files.append(str(path))
        return {"scene_name": scene_name, "views": len(registry), "files": files}
    finally:
        gt.close()
        cloud.close()
        for _, archive, _ in prediction_archives:
            archive.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt_npz", required=True)
    parser.add_argument("--global_cloud_npz", required=True)
    parser.add_argument("--prediction", action="append", default=[], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_name", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = render(
        gt_npz=Path(args.gt_npz),
        global_cloud_npz=Path(args.global_cloud_npz),
        predictions=[parse_prediction(value) for value in args.prediction],
        output_dir=Path(args.output_dir),
        scene_name=args.scene_name,
    )
    manifest = Path(args.output_dir) / "visualization_manifest.json"
    manifest.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
