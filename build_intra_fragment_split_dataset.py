"""Build a scene-held-out manifest for intra-fragment plane splitting.

The manifest keeps existing Stage3 support and point-aligned GT archives
immutable. GT labels are joined only through the saved provenance tuple
``(alignment_view_index, x, y)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_bias_aware_fragment_association import (
    _artifact_path,
    _scene_name,
    exact_gt_join,
)


OUTPUT_SCHEMA_VERSION = 1


def summarize_fragments(
    fragment_ids: np.ndarray,
    gt_labels: np.ndarray,
    *,
    min_fragment_points: int,
    min_labeled_rate: float,
    impurity_threshold: float,
) -> list[dict]:
    fragment_ids = np.asarray(fragment_ids, dtype=np.int64)
    gt_labels = np.asarray(gt_labels, dtype=np.int64)
    if fragment_ids.shape != gt_labels.shape:
        raise ValueError("fragment_ids and gt_labels must have identical shapes")
    records = []
    for fragment_id in sorted(
        int(value) for value in np.unique(fragment_ids) if value >= 0
    ):
        mask = fragment_ids == fragment_id
        count = int(mask.sum())
        local_gt = gt_labels[mask]
        labeled = local_gt[local_gt >= 0]
        if len(labeled):
            labels, counts = np.unique(labeled, return_counts=True)
            dominant_index = int(np.argmax(counts))
            dominant_label = int(labels[dominant_index])
            purity = float(counts[dominant_index] / len(labeled))
            gt_plane_count = int(len(labels))
        else:
            dominant_label = -1
            purity = 0.0
            gt_plane_count = 0
        labeled_rate = float(len(labeled) / max(count, 1))
        eligible = (
            count >= int(min_fragment_points)
            and labeled_rate >= float(min_labeled_rate)
        )
        records.append(
            {
                "fragment_id": fragment_id,
                "point_count": count,
                "labeled_point_count": int(len(labeled)),
                "labeled_rate": labeled_rate,
                "dominant_gt_plane": dominant_label,
                "gt_plane_count": gt_plane_count,
                "gt_purity": purity,
                "eligible": bool(eligible),
                "hard_impure": bool(
                    eligible
                    and gt_plane_count >= 2
                    and purity < float(impurity_threshold)
                ),
            }
        )
    return records


def load_scene_labels(
    support_path: Path,
    gt_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(support_path, allow_pickle=True) as support:
        required = {
            "point_plane_ids",
            "alignment_view_indices",
            "pointmap_pixel_xy",
        }
        missing = sorted(required - set(support.files))
        if missing:
            raise ValueError(f"{support_path}: missing fields {missing}")
        fragment_ids = support["point_plane_ids"].astype(np.int64)
        support_views = support["alignment_view_indices"].astype(np.int64)
        support_pixels = support["pointmap_pixel_xy"].astype(np.int64)
    with np.load(gt_path, allow_pickle=False) as gt:
        gt_labels = exact_gt_join(
            support_views,
            support_pixels,
            gt["view_indices"].astype(np.int64),
            gt["pixel_xy"].astype(np.int64),
            gt["point_plane_ids"].astype(np.int64),
        )
    return fragment_ids, gt_labels, support_views


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the intra-fragment point-affinity training manifest."
    )
    parser.add_argument("--batch_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_scenes", default="")
    parser.add_argument("--min_fragment_points", type=int, default=128)
    parser.add_argument("--min_labeled_rate", type=float, default=0.8)
    parser.add_argument("--impurity_threshold", type=float, default=0.8)
    args = parser.parse_args()

    batch_path = Path(args.batch_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    root = batch_path
    while root.parent != root and not (root / ".git").exists():
        root = root.parent
    if not (root / ".git").exists():
        raise ValueError("Could not infer repository root from batch_json")
    val_scenes = {
        value.strip() for value in args.val_scenes.split(",") if value.strip()
    }
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    scenes = []
    for item in batch["items"]:
        gt_path = _artifact_path(root, item, "point_aligned_gt")
        support_path = _artifact_path(root, item, "direct_support_records")
        scene = _scene_name(item, gt_path)
        fragments, gt_labels, support_views = load_scene_labels(
            support_path, gt_path
        )
        records = summarize_fragments(
            fragments,
            gt_labels,
            min_fragment_points=args.min_fragment_points,
            min_labeled_rate=args.min_labeled_rate,
            impurity_threshold=args.impurity_threshold,
        )
        for record in records:
            mask = fragments == int(record["fragment_id"])
            record["within_view_mixed"] = bool(
                any(
                    len(np.unique(gt_labels[mask & (support_views == view)][
                        gt_labels[mask & (support_views == view)] >= 0
                    ]))
                    >= 2
                    for view in np.unique(support_views[mask])
                )
            )
        eligible = [record for record in records if record["eligible"]]
        hard = [record for record in records if record["hard_impure"]]
        scenes.append(
            {
                "scene": scene,
                "split": "val" if scene in val_scenes else "train",
                "support_npz": str(support_path),
                "gt_npz": str(gt_path),
                "point_count": int(len(fragments)),
                "gt_labeled_point_rate": float((gt_labels >= 0).mean()),
                "fragment_count": int(len(records)),
                "eligible_fragment_count": int(len(eligible)),
                "hard_impure_fragment_count": int(len(hard)),
                "within_view_mixed_hard_count": int(
                    sum(record["within_view_mixed"] for record in hard)
                ),
                "eligible_point_count": int(
                    sum(record["point_count"] for record in eligible)
                ),
                "hard_impure_point_count": int(
                    sum(record["point_count"] for record in hard)
                ),
                "fragments": records,
            }
        )
    if val_scenes - {scene["scene"] for scene in scenes}:
        raise ValueError(
            "Unknown val scenes: "
            + ", ".join(sorted(val_scenes - {row["scene"] for row in scenes}))
        )
    split_summary = {}
    for split in ("train", "val"):
        rows = [scene for scene in scenes if scene["split"] == split]
        split_summary[split] = {
            "scene_count": len(rows),
            "eligible_fragment_count": int(
                sum(row["eligible_fragment_count"] for row in rows)
            ),
            "hard_impure_fragment_count": int(
                sum(row["hard_impure_fragment_count"] for row in rows)
            ),
            "within_view_mixed_hard_count": int(
                sum(row["within_view_mixed_hard_count"] for row in rows)
            ),
            "eligible_point_count": int(
                sum(row["eligible_point_count"] for row in rows)
            ),
            "hard_impure_point_count": int(
                sum(row["hard_impure_point_count"] for row in rows)
            ),
        }
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "task": "intra_fragment_point_affinity_and_boundary",
        "coordinate_contract": {
            "pixel_order": "(x,y)",
            "pixel_space": "dust3r_aligned_pointmap",
            "join_key": "(alignment_view_index,x,y)",
            "xyz_space": "DUSt3R globally aligned pointmap coordinates",
        },
        "feature_contract": [
            "fragment-centered normalized XYZ",
            "RGB in [0,1]",
            "pointmap pixel XY normalized by explicit 512 input size",
            "alignment view index",
            "signed residual to the current fragment plane",
        ],
        "label_contract": {
            "point_instance": "Structured3D point-aligned plane identity",
            "edge_affinity": "same GT plane for local point pair",
            "boundary": "has a local neighbor with a different valid GT plane",
            "gt_at_inference": False,
        },
        "split_summary": split_summary,
        "scenes": scenes,
        "config": {
            "batch_json": str(batch_path),
            "val_scenes": sorted(val_scenes),
            "min_fragment_points": int(args.min_fragment_points),
            "min_labeled_rate": float(args.min_labeled_rate),
            "impurity_threshold": float(args.impurity_threshold),
        },
        "outputs": {
            "manifest_json": str(output_dir / "manifest.json"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "split_summary": split_summary,
                "manifest_json": result["outputs"]["manifest_json"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
