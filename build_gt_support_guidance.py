"""Adapt point-aligned GT labels into the guided-RANSAC support schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_text(value: np.ndarray) -> str:
    item = np.asarray(value).reshape(()).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def build_gt_support(gt_npz: Path, output_npz: Path) -> dict[str, object]:
    if output_npz.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_npz}")
    with np.load(gt_npz, allow_pickle=False) as source:
        required = {"point_plane_ids", "view_indices", "pixel_xy"}
        missing = sorted(required - set(source.files))
        if missing:
            raise ValueError(f"point-aligned GT is missing fields: {missing}")
        labels = np.asarray(source["point_plane_ids"], dtype=np.int32).reshape(-1)
        views = np.asarray(source["view_indices"], dtype=np.int32).reshape(-1)
        pixels = np.asarray(source["pixel_xy"], dtype=np.int32)
        if len(labels) != len(views) or pixels.shape != (len(labels), 2):
            raise ValueError("GT labels, view indices, and pixel coordinates disagree")
        if "pixel_coordinate_order" in source.files:
            order = scalar_text(source["pixel_coordinate_order"])
            if order != "xy":
                raise ValueError(f"GT pixel coordinate order must be xy, got {order!r}")
        if "pixel_coordinate_space" in source.files:
            space = scalar_text(source["pixel_coordinate_space"])
            if space != "dust3r_aligned_pointmap":
                raise ValueError(
                    "GT pixel coordinate space must be dust3r_aligned_pointmap, "
                    f"got {space!r}"
                )

    keys = (views.astype(np.int64) << 40) | (pixels[:, 0].astype(np.int64) << 20) | pixels[:, 1]
    if len(np.unique(keys)) != len(keys):
        raise ValueError("point-aligned GT registry contains duplicate (view,x,y) keys")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        point_plane_ids=labels,
        alignment_view_indices=views,
        pointmap_pixel_xy=pixels,
        pointmap_pixel_coordinate_order=np.asarray("xy"),
        pointmap_pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
        support_source=np.asarray("point_aligned_gt"),
        source_gt_npz=np.asarray(str(gt_npz)),
        source_gt_sha256=np.asarray(file_sha256(gt_npz)),
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "point_aligned_gt_support_guidance",
        "source_gt_npz": str(gt_npz),
        "source_gt_sha256": file_sha256(gt_npz),
        "output_npz": str(output_npz),
        "output_sha256": file_sha256(output_npz),
        "records": int(len(labels)),
        "assigned_records": int(np.count_nonzero(labels >= 0)),
        "plane_labels": int(len(np.unique(labels[labels >= 0]))),
        "coordinate_order": "xy",
        "coordinate_space": "dust3r_aligned_pointmap",
    }
    output_npz.with_suffix(".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt_npz", required=True)
    parser.add_argument("--output_npz", required=True)
    args = parser.parse_args()
    result = build_gt_support(Path(args.gt_npz), Path(args.output_npz))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
