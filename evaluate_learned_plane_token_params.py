import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_case(json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    npz_path = Path(json_path.replace("_tokens.json", "_tokens_assignment.npz"))
    if not npz_path.exists():
        # Handles multisample file names too.
        npz_path = Path(str(json_path).replace(".json", "_assignment.npz"))
    raw = np.load(npz_path)
    return data, raw, npz_path


def residual_stats(points, assignment, normal, offset, plane_id, thresholds, trim_ratio):
    mask = assignment == plane_id
    count = int(mask.sum())
    if count == 0:
        return {
            "assigned_points": 0,
            "assigned_ratio": 0.0,
            "mean_residual": None,
            "median_residual": None,
            "p90_residual": None,
            "trimmed_mean_residual": None,
            **{f"inlier_ratio_{str(t).replace('.', '')}": None for t in thresholds},
        }
    residual = np.abs(points[mask] @ normal + offset)
    residual_sorted = np.sort(residual)
    keep = max(1, int(round(len(residual_sorted) * trim_ratio)))
    out = {
        "assigned_points": count,
        "assigned_ratio": float(count / len(points)),
        "mean_residual": float(np.mean(residual)),
        "median_residual": float(np.median(residual)),
        "p90_residual": float(np.percentile(residual, 90)),
        "trimmed_mean_residual": float(np.mean(residual_sorted[:keep])),
    }
    for threshold in thresholds:
        key = f"inlier_ratio_{str(threshold).replace('.', '')}"
        out[key] = float(np.mean(residual <= threshold))
    return out


def normal_relation_stats(normals):
    rows = []
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            dot = float(abs(np.dot(normals[i], normals[j])))
            dot = max(-1.0, min(1.0, dot))
            angle = float(np.degrees(np.arccos(dot)))
            if angle <= 10:
                relation = "parallel"
            elif abs(angle - 90) <= 10:
                relation = "orthogonal"
            else:
                relation = "other"
            rows.append({"i": i, "j": j, "angle_deg": angle, "relation": relation})
    return rows


def evaluate_file(json_path, thresholds, trim_ratio):
    data, raw, npz_path = load_case(str(json_path))
    points = raw["points"].astype(np.float32)
    assignment = raw["assignment"].astype(np.int32)
    normals = raw["plane_normals"].astype(np.float32)
    offsets = raw["plane_offsets"].astype(np.float32)
    sample = Path(json_path).name
    sample = sample.replace("_multisample_learned_plane_tokens.json", "")
    sample = sample.replace("_amortized_plane_tokens.json", "")
    sample = sample.replace("_learned_plane_tokens.json", "")
    rows = []
    for plane_id, (normal, offset) in enumerate(zip(normals, offsets)):
        stats = residual_stats(points, assignment, normal, float(offset), plane_id, thresholds, trim_ratio)
        plane_meta = data.get("planes", [{} for _ in range(len(normals))])[plane_id]
        rows.append(
            {
                "sample": sample,
                "plane_id": plane_id,
                "normal": [float(x) for x in normal],
                "offset": float(offset),
                "meta_assigned_points": plane_meta.get("assigned_point_count"),
                "meta_assigned_ratio": plane_meta.get("assigned_ratio"),
                **stats,
            }
        )
    relations = normal_relation_stats(normals)
    return rows, relations, npz_path


def fmt(x, n=4):
    if x is None:
        return ""
    return f"{float(x):.{n}f}"


def main():
    parser = argparse.ArgumentParser("Evaluate learned plane-token equations against assigned point clouds")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*_multisample_learned_plane_tokens.json")
    parser.add_argument("--trim_ratio", type=float, default=0.8)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.02, 0.05])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern} in {input_dir}")

    plane_rows = []
    relation_rows = []
    for path in files:
        rows, relations, npz_path = evaluate_file(path, args.thresholds, args.trim_ratio)
        plane_rows.extend(rows)
        sample = rows[0]["sample"] if rows else path.stem
        for rel in relations:
            relation_rows.append({"sample": sample, **rel})

    csv_path = output_dir / "learned_plane_token_param_eval.csv"
    fieldnames = list(plane_rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plane_rows)

    relation_csv_path = output_dir / "learned_plane_token_normal_relations.csv"
    with relation_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "i", "j", "angle_deg", "relation"])
        writer.writeheader()
        writer.writerows(relation_rows)

    by_sample = {}
    for row in plane_rows:
        by_sample.setdefault(row["sample"], []).append(row)

    sample_summary = []
    for sample, rows in by_sample.items():
        assigned = [r["assigned_ratio"] for r in rows if r["assigned_ratio"] is not None]
        trimmed = [r["trimmed_mean_residual"] for r in rows if r["trimmed_mean_residual"] is not None]
        inlier005_key = "inlier_ratio_005"
        inlier005 = [r[inlier005_key] for r in rows if r.get(inlier005_key) is not None]
        sample_summary.append(
            {
                "sample": sample,
                "num_planes": len(rows),
                "max_assigned_ratio": max(assigned) if assigned else None,
                "min_assigned_ratio": min(assigned) if assigned else None,
                "coverage_spread": (max(assigned) - min(assigned)) if assigned else None,
                "mean_trimmed_residual": float(np.mean(trimmed)) if trimmed else None,
                "mean_inlier_ratio_005": float(np.mean(inlier005)) if inlier005 else None,
            }
        )
    sample_summary = sorted(
        sample_summary,
        key=lambda r: (
            r["mean_trimmed_residual"] if r["mean_trimmed_residual"] is not None else 1e9,
            r["coverage_spread"] if r["coverage_spread"] is not None else 1e9,
        ),
    )

    md = [
        "# Learned Plane Token Parameter Evaluation",
        "",
        f"Files evaluated: {len(files)}",
        f"Trim ratio: {args.trim_ratio}",
        "",
        "## Sample Summary",
        "",
        "| sample | planes | max assigned | min assigned | spread | mean trimmed residual | mean inlier@0.05 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sample_summary:
        md.append(
            f"| {row['sample']} | {row['num_planes']} | {fmt(row['max_assigned_ratio'])} | "
            f"{fmt(row['min_assigned_ratio'])} | {fmt(row['coverage_spread'])} | "
            f"{fmt(row['mean_trimmed_residual'], 5)} | {fmt(row['mean_inlier_ratio_005'])} |"
        )
    md.extend(
        [
            "",
            "## Plane-Level Rows",
            "",
            "| sample | plane | assigned | trimmed residual | median | p90 | inlier@0.02 | inlier@0.05 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(plane_rows, key=lambda r: (r["sample"], r["plane_id"])):
        md.append(
            f"| {row['sample']} | {row['plane_id']} | {fmt(row['assigned_ratio'])} | "
            f"{fmt(row['trimmed_mean_residual'], 5)} | {fmt(row['median_residual'], 5)} | "
            f"{fmt(row['p90_residual'], 5)} | {fmt(row.get('inlier_ratio_002'))} | "
            f"{fmt(row.get('inlier_ratio_005'))} |"
        )

    md_path = output_dir / "learned_plane_token_param_eval_summary.md"
    json_path = output_dir / "learned_plane_token_param_eval_summary.json"
    md_path.write_text("\n".join(md), encoding="utf-8")
    json_path.write_text(
        json.dumps({"sample_summary": sample_summary, "planes": plane_rows, "relations": relation_rows}, indent=2),
        encoding="utf-8",
    )
    print(csv_path)
    print(relation_csv_path)
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
