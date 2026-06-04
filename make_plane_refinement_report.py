import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def pct(value):
    return f"{value * 100.0:.2f}%"


def summarize(eval_rows, binding_rows):
    for row in eval_rows:
        row["_base"] = to_float(row, "base_mean_abs_distance")
        row["_refined"] = to_float(row, "refined_mean_abs_distance")
        row["_delta"] = to_float(row, "delta")
        row["_accepted"] = row.get("accepted") == "True"
        row["_confidence"] = to_float(row, "confidence")
        row["_base_bound"] = to_float(row, "base_bound_points")
        row["_refined_bound"] = to_float(row, "refined_bound_points")

    for row in binding_rows:
        row["_base_count"] = to_float(row, "base_count")
        row["_refined_count"] = to_float(row, "refined_count")
        row["_added"] = to_float(row, "added_count")
        row["_removed"] = to_float(row, "removed_count")
        row["_base_residual"] = to_float(row, "base_residual")
        row["_refined_residual"] = to_float(row, "refined_residual")
        row["_residual_delta"] = row["_base_residual"] - row["_refined_residual"]
        row["_change"] = row["_added"] + row["_removed"]

    accepted = [r for r in eval_rows if r["_accepted"]]
    improved = [r for r in eval_rows if r["_delta"] > 0]
    base_mean = sum(r["_base"] for r in eval_rows) / max(len(eval_rows), 1)
    refined_mean = sum(r["_refined"] for r in eval_rows) / max(len(eval_rows), 1)

    binding_base = sum(r["_base_residual"] for r in binding_rows) / max(len(binding_rows), 1)
    binding_refined = sum(r["_refined_residual"] for r in binding_rows) / max(len(binding_rows), 1)
    binding_added = sum(r["_added"] for r in binding_rows) / max(len(binding_rows), 1)
    binding_removed = sum(r["_removed"] for r in binding_rows) / max(len(binding_rows), 1)

    return {
        "planes": len(eval_rows),
        "accepted_planes": len(accepted),
        "improved_planes": len(improved),
        "base_mean": base_mean,
        "refined_mean": refined_mean,
        "mean_delta": base_mean - refined_mean,
        "relative_improvement": (base_mean - refined_mean) / base_mean if base_mean else 0.0,
        "mean_confidence": sum(r["_confidence"] for r in eval_rows) / max(len(eval_rows), 1),
        "binding_samples": len(binding_rows),
        "binding_base": binding_base,
        "binding_refined": binding_refined,
        "binding_delta": binding_base - binding_refined,
        "binding_relative_improvement": (binding_base - binding_refined) / binding_base if binding_base else 0.0,
        "binding_added": binding_added,
        "binding_removed": binding_removed,
    }


def row_link(output_dir, sample, suffix):
    path = output_dir / f"{sample}{suffix}"
    return path.name if path.exists() else ""


def make_markdown(summary, eval_rows, binding_rows, refined_dir, binding_dir):
    best_param = sorted(eval_rows, key=lambda r: r["_delta"], reverse=True)[:8]
    binding_change = sorted(binding_rows, key=lambda r: r["_change"], reverse=True)[:8]
    binding_improve = sorted(binding_rows, key=lambda r: r["_residual_delta"], reverse=True)[:8]

    lines = [
        "# Plane Proposal Refinement Report",
        "",
        "## One-line Result",
        (
            "The current learned head is useful as a proposal-conditioned refiner: "
            f"mean plane residual improves from {fmt(summary['base_mean'], 6)} to {fmt(summary['refined_mean'], 6)} "
            f"({pct(summary['relative_improvement'])}), with {summary['accepted_planes']}/{summary['planes']} planes accepted."
        ),
        "",
        "## Metrics",
        f"- Evaluated planes: {summary['planes']}",
        f"- Accepted refined planes: {summary['accepted_planes']}",
        f"- Improved planes: {summary['improved_planes']}",
        f"- Mean confidence: {fmt(summary['mean_confidence'], 4)}",
        f"- Largest-plane binding residual: {fmt(summary['binding_base'], 6)} -> {fmt(summary['binding_refined'], 6)} ({pct(summary['binding_relative_improvement'])})",
        f"- Mean largest-plane added / removed points: {fmt(summary['binding_added'], 1)} / {fmt(summary['binding_removed'], 1)}",
        "",
        "## Best Parameter Refinement Cases",
        "| sample | plane | base residual | refined residual | delta | confidence | edit html |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in best_param:
        sample = row["sample"]
        link = row_link(refined_dir, sample, "_refined_plane_edit_comparison.html")
        lines.append(
            f"| {sample} | {row['plane_id']} | {fmt(row['_base'], 6)} | {fmt(row['_refined'], 6)} | "
            f"{fmt(row['_delta'], 6)} | {fmt(row['_confidence'], 3)} | {link} |"
        )

    lines.extend(
        [
            "",
            "## Most Visible Binding Changes",
            "| sample | plane | base bound | refined bound | added | removed | binding html |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in binding_change:
        sample = row["sample"]
        link = row_link(binding_dir, sample, "_binding_refinement_comparison.html")
        lines.append(
            f"| {sample} | {row['plane_id']} | {int(row['_base_count'])} | {int(row['_refined_count'])} | "
            f"{int(row['_added'])} | {int(row['_removed'])} | {link} |"
        )

    lines.extend(
        [
            "",
            "## Best Binding Residual Improvements",
            "| sample | plane | base residual | refined residual | delta | binding html |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in binding_improve:
        sample = row["sample"]
        link = row_link(binding_dir, sample, "_binding_refinement_comparison.html")
        lines.append(
            f"| {sample} | {row['plane_id']} | {fmt(row['_base_residual'], 6)} | "
            f"{fmt(row['_refined_residual'], 6)} | {fmt(row['_residual_delta'], 6)} | {link} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation For Report",
            "- RANSAC gives coarse structural proposals; the learned head refines the equation and predicts whether the proposal should be accepted.",
            "- The current head is conservative: it mostly removes outlier support points instead of aggressively adding new support points.",
            "- This is enough to claim a first-stage learnable editable primitive refiner, but the next improvement should strengthen learned point-to-plane assignment.",
            "- Recommended figures: val_000027 for add/remove behavior, val_000034 for visible support cleaning, val_000039 for residual improvement, val_000037 for the commonly inspected sample.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown_text):
    rows = []
    in_table = False
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            rows.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            rows.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("- "):
            rows.append(f"<p class='bullet'>{line[2:]}</p>")
        elif line.startswith("|") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "th" if not in_table else "td"
            if not in_table:
                rows.append("<table>")
                in_table = True
            rows.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
        elif line.startswith("|---"):
            continue
        else:
            if in_table:
                rows.append("</table>")
                in_table = False
            if line:
                rows.append(f"<p>{line}</p>")
    if in_table:
        rows.append("</table>")
    body = "\n".join(rows)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Plane Refinement Report</title>
<style>
body {{ font-family: Inter, Segoe UI, Arial, sans-serif; margin: 32px; color: #172033; background: #f7f8fb; }}
h1 {{ margin: 0 0 18px; }}
h2 {{ margin: 28px 0 10px; font-size: 18px; }}
p {{ max-width: 980px; line-height: 1.55; }}
.bullet {{ margin: 5px 0; }}
table {{ border-collapse: collapse; width: 100%; background: white; margin: 10px 0 20px; font-size: 13px; }}
th, td {{ border: 1px solid #dce1e8; padding: 7px 8px; text-align: left; }}
th {{ background: #eef2f7; }}
td:nth-child(n+2), th:nth-child(n+2) {{ text-align: right; }}
td:last-child, th:last-child {{ text-align: left; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser("Create a report-ready summary for plane proposal refinement.")
    parser.add_argument("--refined_dir", required=True)
    parser.add_argument("--binding_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    refined_dir = Path(args.refined_dir)
    binding_dir = Path(args.binding_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = read_csv(refined_dir / "plane_proposal_refinement_eval.csv")
    binding_rows = read_csv(binding_dir / "binding_refinement_summary.csv")
    summary = summarize(eval_rows, binding_rows)
    markdown = make_markdown(summary, eval_rows, binding_rows, refined_dir, binding_dir)
    (output_dir / "plane_refinement_report.md").write_text(markdown, encoding="utf-8")
    (output_dir / "plane_refinement_report.html").write_text(markdown_to_html(markdown), encoding="utf-8")
    (output_dir / "plane_refinement_report_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(output_dir / "plane_refinement_report.html")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
