import argparse
import csv
import json
from pathlib import Path


def load_rows(root):
    rows = []
    for manifest in sorted(Path(root).glob("*/stage3_global_visual_v1/stage3_scene_fusion_manifest.json")):
        group_root = manifest.parents[1]
        payload = json.load(open(manifest, encoding="utf-8"))
        for row in payload:
            quality = row.get("quality_summary", {}) or {}
            mapping = row.get("mapping_records", []) or []
            source1 = sum(int(rec.get("source_counts", {}).get("1", 0)) for rec in mapping)
            source2 = sum(int(rec.get("source_counts", {}).get("2", 0)) for rec in mapping)
            dropped = sum(int(rec.get("dropped_total", 0)) for rec in mapping)
            rows.append(
                {
                    "group_root": str(group_root),
                    "scene_key": row.get("scene_key", ""),
                    "views": len(row.get("dust3r_view_registry", []) or []),
                    "loss": row.get("dust3r_global_alignment_loss"),
                    "points": row.get("points"),
                    "local_planes": row.get("local_planes"),
                    "global_planes": row.get("global_planes"),
                    "merged_pairs": row.get("merged_pairs"),
                    "source1_points": source1,
                    "source2_points": source2,
                    "dropped_points": dropped,
                    "plane_count": quality.get("plane_count"),
                    "bad_plane_count": quality.get("bad_plane_count"),
                    "mean_residual_mean": quality.get("mean_residual_mean"),
                    "mean_residual_p95": quality.get("mean_residual_p95"),
                    "avg_source_views_per_plane": quality.get("avg_source_views_per_plane"),
                    "multi_view_plane_rate": quality.get("multi_view_plane_rate"),
                    "html": row.get("html", ""),
                    "npz": row.get("npz", ""),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser("Summarize Stage3 batch val-group manifests")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    rows = load_rows(args.root)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = [
        "group_root",
        "scene_key",
        "views",
        "loss",
        "points",
        "local_planes",
        "global_planes",
        "merged_pairs",
        "source1_points",
        "source2_points",
        "dropped_points",
        "plane_count",
        "bad_plane_count",
        "mean_residual_mean",
        "mean_residual_p95",
        "avg_source_views_per_plane",
        "multi_view_plane_rate",
        "html",
        "npz",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"groups": len(rows), "csv": str(output_csv), "json": str(output_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()