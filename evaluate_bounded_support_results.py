import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_json_rows(directory):
    rows = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name.endswith("summary.json") or path.name.endswith("manifest.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and "planes" in payload:
            rows.append(payload)
    return rows


def summarize_assignments(directory):
    accuracies = []
    boundary_accuracies = []
    plane_ious = []
    for path in sorted(Path(directory).glob("*_assignment.npz")):
        try:
            with np.load(path) as payload:
                labels = np.asarray(payload["patch_labels"], dtype=np.int64)
                predictions = np.asarray(payload["patch_assignment"], dtype=np.int64)
                boundary_weight = np.asarray(
                    payload.get("patch_boundary_ce_weight", np.ones_like(labels)),
                    dtype=np.float32,
                )
        except (KeyError, OSError, ValueError):
            continue

        valid = labels >= 0
        if valid.any():
            accuracies.append(float(np.mean(predictions[valid] == labels[valid])))

        boundary = valid & (boundary_weight > 1.0001)
        if boundary.any():
            boundary_accuracies.append(
                float(np.mean(predictions[boundary] == labels[boundary]))
            )

        for plane_id in np.unique(labels[valid]):
            target = labels == plane_id
            predicted = predictions == plane_id
            union = np.count_nonzero(target | predicted)
            if union:
                plane_ious.append(float(np.count_nonzero(target & predicted) / union))

    return {
        "assignment_samples": len(accuracies),
        "mean_patch_accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
        "mean_boundary_accuracy": (
            float(np.mean(boundary_accuracies))
            if boundary_accuracies
            else float("nan")
        ),
        "mean_plane_iou": float(np.mean(plane_ious)) if plane_ious else float("nan"),
    }


def summarize(method, directory):
    samples = load_json_rows(directory)
    residuals = []
    assigned_points = []
    active_planes = []
    background_ratios = []
    total_planes = 0
    for sample in samples:
        planes = sample.get("planes", [])
        total_planes += len(planes)
        active = [plane for plane in planes if plane.get("active", True)]
        active_planes.append(len(active))
        background_ratios.append(float(sample.get("background_ratio", 0.0)))
        for plane in active:
            residual = plane.get(
                "mean_abs_distance_normalized",
                plane.get("mean_abs_distance"),
            )
            if residual is not None and np.isfinite(float(residual)):
                residuals.append(float(residual))
            count = plane.get("assigned_point_count")
            if count is not None:
                assigned_points.append(float(count))
            elif plane.get("assigned_ratio") is not None and sample.get("num_points_used"):
                assigned_points.append(
                    float(plane["assigned_ratio"]) * float(sample["num_points_used"])
                )

    summary = {
        "method": method,
        "num_samples": len(samples),
        "num_planes": total_planes,
        "active_planes": int(sum(active_planes)),
        "mean_active_planes": float(np.mean(active_planes)) if active_planes else 0.0,
        "mean_assigned_points": float(np.mean(assigned_points)) if assigned_points else 0.0,
        "mean_background_ratio": float(np.mean(background_ratios)) if background_ratios else 0.0,
        "mean_plane_residual": float(np.mean(residuals)) if residuals else float("nan"),
        "median_plane_residual": float(np.median(residuals)) if residuals else float("nan"),
        "p90_plane_residual": float(np.quantile(residuals, 0.9)) if residuals else float("nan"),
    }
    summary.update(summarize_assignments(directory))
    return summary


def format_number(value, digits=5):
    if isinstance(value, float) and not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser("Compare bounded-support experiment JSON outputs")
    parser.add_argument(
        "--method",
        action="append",
        nargs=2,
        metavar=("NAME", "DIRECTORY"),
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize(name, directory) for name, directory in args.method]
    fields = [
        "method",
        "num_samples",
        "num_planes",
        "active_planes",
        "mean_active_planes",
        "mean_assigned_points",
        "mean_background_ratio",
        "mean_plane_residual",
        "median_plane_residual",
        "p90_plane_residual",
        "assignment_samples",
        "mean_patch_accuracy",
        "mean_boundary_accuracy",
        "mean_plane_iou",
    ]
    csv_path = output_dir / "bounded_support_eval_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| method | samples | active planes | mean residual | median residual | p90 residual | background ratio | patch acc. | boundary acc. | plane IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['num_samples']} | "
            f"{format_number(row['mean_active_planes'], 2)} | "
            f"{format_number(row['mean_plane_residual'])} | "
            f"{format_number(row['median_plane_residual'])} | "
            f"{format_number(row['p90_plane_residual'])} | "
            f"{format_number(row['mean_background_ratio'], 4)} | "
            f"{format_number(row['mean_patch_accuracy'], 4)} | "
            f"{format_number(row['mean_boundary_accuracy'], 4)} | "
            f"{format_number(row['mean_plane_iou'], 4)} |"
        )
    lines.extend(
        [
            "",
            "### Current interpretation",
            "",
            "v4 moves SVD refitting from offline post-processing into a differentiable "
            "geometry layer. Soft support weights define weighted plane covariance; "
            "the smallest-eigenvalue eigenvector and anchored offset define the plane. "
            "Plane residual therefore backpropagates into the bounded-support head.",
            "",
            "Strict decoding uses teacher-side safeguards and must not be interpreted "
            "as raw network quality. Use the raw v4 row for model comparison and the "
            "strict row only as the deployable guarded output.",
        ]
    )
    md_path = output_dir / "bounded_support_eval_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
