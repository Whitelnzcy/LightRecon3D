import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        "Build clean/weak/reject cache-index splits from Structured3D GT audit CSV"
    )
    parser.add_argument(
        "--pair_audit_csv",
        default="local_outputs/gt_audit_structured3d/pair_audit_summary.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="local_outputs/gt_quality_splits_v1",
    )
    parser.add_argument("--reject_largest_threshold", type=float, default=0.92)
    parser.add_argument("--weak_largest_threshold", type=float, default=0.80)
    parser.add_argument("--reject_overlap_threshold", type=float, default=0.05)
    parser.add_argument("--weak_overlap_threshold", type=float, default=0.01)
    parser.add_argument("--weak_min_visible_threshold", type=int, default=2)
    parser.add_argument("--weak_visible_gap_threshold", type=int, default=4)
    return parser.parse_args()


def _float(row, key, default=0.0):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else default


def _int(row, key, default=0):
    value = row.get(key, "")
    return int(float(value)) if value not in ("", None) else default


def classify_pair(row, args):
    view1_overlap = _float(row, "view1_overlap_ratio")
    view2_overlap = _float(row, "view2_overlap_ratio")
    view1_covered = _int(row, "view1_fully_covered")
    view2_covered = _int(row, "view2_fully_covered")
    view1_largest = _float(row, "view1_legacy_largest")
    view2_largest = _float(row, "view2_legacy_largest")
    view1_visible = _int(row, "view1_legacy_visible")
    view2_visible = _int(row, "view2_legacy_visible")

    max_overlap = max(view1_overlap, view2_overlap)
    max_covered = max(view1_covered, view2_covered)
    max_largest = max(view1_largest, view2_largest)
    min_visible = min(view1_visible, view2_visible)
    visible_gap = abs(view1_visible - view2_visible)

    reasons = []
    if max_covered > 0:
        reasons.append("fully_covered_plane")
    if max_overlap >= args.reject_overlap_threshold:
        reasons.append("large_polygon_overlap")
    if max_largest >= args.reject_largest_threshold and min_visible <= args.weak_min_visible_threshold:
        reasons.append("extreme_single_plane_dominance")

    if reasons:
        return "reject", reasons

    if max_largest >= args.weak_largest_threshold:
        reasons.append("single_plane_dominance")
    if min_visible <= args.weak_min_visible_threshold:
        reasons.append("few_visible_planes")
    if max_overlap >= args.weak_overlap_threshold:
        reasons.append("moderate_polygon_overlap")
    if visible_gap >= args.weak_visible_gap_threshold:
        reasons.append("large_view_visible_count_gap")

    if reasons:
        return "weak", reasons
    return "clean", ["passed_gt_quality_checks"]


def write_index_list(path, rows):
    text = "\n".join(str(row["cache_idx"]) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_path = Path(args.pair_audit_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            quality, reasons = classify_pair(row, args)
            enriched = dict(row)
            enriched["cache_idx"] = int(row["cache_idx"])
            enriched["quality"] = quality
            enriched["quality_reasons"] = ";".join(reasons)
            rows.append(enriched)

    buckets = {
        "clean": [row for row in rows if row["quality"] == "clean"],
        "weak": [row for row in rows if row["quality"] == "weak"],
        "reject": [row for row in rows if row["quality"] == "reject"],
    }
    buckets["clean_plus_weak"] = buckets["clean"] + buckets["weak"]

    write_csv(output_dir / "gt_quality_pairs.csv", rows)
    for name, bucket in buckets.items():
        write_index_list(output_dir / f"{name}_cache_indices.txt", bucket)

    reason_counts = {}
    for row in rows:
        for reason in row["quality_reasons"].split(";"):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    summary = {
        "pair_audit_csv": str(input_path),
        "thresholds": {
            "reject_largest_threshold": args.reject_largest_threshold,
            "weak_largest_threshold": args.weak_largest_threshold,
            "reject_overlap_threshold": args.reject_overlap_threshold,
            "weak_overlap_threshold": args.weak_overlap_threshold,
            "weak_min_visible_threshold": args.weak_min_visible_threshold,
            "weak_visible_gap_threshold": args.weak_visible_gap_threshold,
        },
        "pair_count": len(rows),
        "clean_count": len(buckets["clean"]),
        "weak_count": len(buckets["weak"]),
        "reject_count": len(buckets["reject"]),
        "clean_plus_weak_count": len(buckets["clean_plus_weak"]),
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = [
        "# GT Quality Splits",
        "",
        "This is a cache-index filter generated from the Structured3D GT audit. It does not modify GT files.",
        "",
        f"- pair_count: {summary['pair_count']}",
        f"- clean_count: {summary['clean_count']}",
        f"- weak_count: {summary['weak_count']}",
        f"- reject_count: {summary['reject_count']}",
        f"- clean_plus_weak_count: {summary['clean_plus_weak_count']}",
        "",
        "## Reason Counts",
        "",
    ]
    for reason, count in summary["reason_counts"].items():
        report.append(f"- {reason}: {count}")
    report.extend(
        [
            "",
            "## Usage",
            "",
            "Use `clean_cache_indices.txt` for precision-first training.",
            "Use `clean_plus_weak_cache_indices.txt` when the clean subset is too small.",
            "Keep `reject_cache_indices.txt` out of supervised mask training unless it is used only for qualitative failure analysis.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
