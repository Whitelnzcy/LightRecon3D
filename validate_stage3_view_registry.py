import argparse
import json
import os
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = (
    "schema_version",
    "scene_name",
    "pair_group",
    "rgb_path1",
    "rgb_path2",
    "view_id1",
    "view_id2",
    "pixel_xy1",
    "pixel_xy2",
    "point_plane_ids",
)
OPTIONAL_VIEW_FIELDS = (
    "json_path",
    "original_hw",
    "stage1_input_hw",
    "stage1_mask_hw",
)


def scalar_string(raw, key, default=""):
    if key not in raw:
        return default
    value = raw[key]
    if np.ndim(value) == 0:
        return str(value.item())
    if len(value) == 0:
        return default
    return str(value[0])


def scalar_int(raw, key, default=0):
    if key not in raw:
        return default
    value = raw[key]
    try:
        if np.ndim(value) == 0:
            return int(value.item())
        if len(value) == 0:
            return default
        return int(value[0])
    except (TypeError, ValueError):
        return default


def array_hw(raw, key):
    if key not in raw:
        return []
    value = np.asarray(raw[key]).reshape(-1)
    if len(value) < 2:
        return []
    return [int(value[0]), int(value[1])]


def canonical_path(path):
    text = str(path).strip()
    if not text:
        return ""
    return os.path.normcase(os.path.normpath(text)).replace("\\", "/")


def json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def view_record(raw, role, alignment_view_index, canonical):
    return {
        "canonical_path": canonical,
        "source_path": scalar_string(raw, f"rgb_path{role}"),
        "json_path": scalar_string(raw, f"json_path{role}"),
        "source_view_id": scalar_string(raw, f"view_id{role}"),
        "alignment_view_index": int(alignment_view_index),
        "original_hw": array_hw(raw, f"original_hw{role}"),
        "stage1_input_hw": array_hw(raw, f"stage1_input_hw{role}"),
        "stage1_mask_hw": array_hw(raw, f"stage1_mask_hw{role}"),
    }


def same_view_metadata(a, b):
    keys = ("source_view_id", "original_hw", "stage1_input_hw", "stage1_mask_hw")
    return all(a.get(key) == b.get(key) for key in keys if a.get(key) and b.get(key))


def validate_pixel_array(raw, key, expected_len, allow_empty=False):
    errors = []
    if key not in raw:
        errors.append(f"missing {key}")
        return errors
    xy = np.asarray(raw[key])
    if allow_empty and len(xy) == 0:
        return errors
    if xy.ndim != 2 or xy.shape[1] != 2:
        errors.append(f"{key} bad shape {xy.shape}")
        return errors
    if len(xy) != expected_len:
        errors.append(f"{key} length {len(xy)} != point_plane_ids length {expected_len}")
    if len(xy):
        finite = np.isfinite(xy).all(axis=1)
        in_range = (xy >= -1.0).all(axis=1) & (xy <= 1.0).all(axis=1)
        bad = int((~(finite & in_range)).sum())
        if bad:
            errors.append(f"{key} has {bad} invalid/out-of-range rows")
    return errors


def validate_record(path, raw, check_files=False):
    missing = [key for key in REQUIRED_FIELDS if key not in raw]
    errors = []
    if missing:
        errors.extend(f"missing {key}" for key in missing)

    point_count = int(len(raw["point_plane_ids"])) if "point_plane_ids" in raw else 0
    if "pixel_xy1" in raw:
        errors.extend(validate_pixel_array(raw, "pixel_xy1", point_count))
    elif "pixel_xy" in raw:
        errors.extend(validate_pixel_array(raw, "pixel_xy", point_count))
    if "pixel_xy2" in raw:
        errors.extend(validate_pixel_array(raw, "pixel_xy2", point_count, allow_empty=True))

    if "support_source_view" in raw:
        source = np.asarray(raw["support_source_view"]).reshape(-1)
        if len(source) != point_count:
            errors.append(f"support_source_view length {len(source)} != point_plane_ids length {point_count}")
        elif len(source):
            invalid = int((~np.isin(source, [1, 2])).sum())
            if invalid:
                errors.append(f"support_source_view has {invalid} invalid rows")

    missing_files = []
    if check_files:
        for key in ("rgb_path1", "rgb_path2"):
            value = scalar_string(raw, key)
            if value and not Path(value).exists():
                missing_files.append(value)
                errors.append(f"missing image file {key}={value}")

    return {
        "path": str(path),
        "schema_version": scalar_int(raw, "schema_version", 1),
        "scene_name": scalar_string(raw, "scene_name"),
        "pair_group": scalar_string(raw, "pair_group"),
        "point_count": point_count,
        "missing_fields": missing,
        "missing_files": missing_files,
        "errors": errors,
    }


def build_view_registry(records):
    groups = {}
    summary = {
        "records": 0,
        "valid_records": 0,
        "groups": 0,
        "unique_views": 0,
        "groups_with_3plus_views": 0,
        "missing_required_records": 0,
        "records_with_errors": 0,
        "duplicate_view_conflict_count": 0,
        "unmapped_support_records": 0,
    }

    for record in records:
        raw = record["raw"]
        validation = record["validation"]
        summary["records"] += 1
        if validation["missing_fields"]:
            summary["missing_required_records"] += 1
        if validation["errors"]:
            summary["records_with_errors"] += 1
        else:
            summary["valid_records"] += 1

        scene = validation["scene_name"] or Path(record["path"]).stem
        pair_group = validation["pair_group"]
        group_key = f"{scene}|{pair_group}" if pair_group else scene
        group = groups.setdefault(
            group_key,
            {
                "group_key": group_key,
                "scene_name": scene,
                "pair_group": pair_group,
                "records": [],
                "view_registry": [],
                "view_by_path": {},
                "duplicate_view_conflicts": [],
                "unmapped_support_records": [],
            },
        )
        group["records"].append(validation)

        for role in (1, 2):
            source_path = scalar_string(raw, f"rgb_path{role}")
            key = canonical_path(source_path)
            if not key:
                continue
            existing_index = group["view_by_path"].get(key)
            if existing_index is None:
                index = len(group["view_registry"])
                group["view_by_path"][key] = index
                group["view_registry"].append(view_record(raw, role, index, key))
            else:
                candidate = view_record(raw, role, existing_index, key)
                existing = group["view_registry"][existing_index]
                if not same_view_metadata(existing, candidate):
                    group["duplicate_view_conflicts"].append(
                        {"path": key, "existing": existing, "candidate": candidate, "record": record["path"]}
                    )

        source = np.asarray(raw["support_source_view"]).reshape(-1) if "support_source_view" in raw else None
        if source is not None and len(source):
            bad_roles = sorted(set(int(x) for x in source.tolist() if int(x) not in (1, 2)))
            missing_roles = []
            for role in sorted(set(int(x) for x in source.tolist() if int(x) in (1, 2))):
                if not canonical_path(scalar_string(raw, f"rgb_path{role}")):
                    missing_roles.append(role)
            if bad_roles or missing_roles:
                group["unmapped_support_records"].append(
                    {"record": record["path"], "bad_roles": bad_roles, "missing_roles": missing_roles}
                )

    group_rows = []
    for group in groups.values():
        conflicts = group.pop("duplicate_view_conflicts")
        unmapped = group.pop("unmapped_support_records")
        group.pop("view_by_path")
        group["records_count"] = len(group["records"])
        group["unique_views_count"] = len(group["view_registry"])
        group["duplicate_view_conflicts"] = conflicts
        group["unmapped_support_records"] = unmapped
        summary["unique_views"] += group["unique_views_count"]
        summary["duplicate_view_conflict_count"] += len(conflicts)
        summary["unmapped_support_records"] += len(unmapped)
        if group["unique_views_count"] >= 3:
            summary["groups_with_3plus_views"] += 1
        group_rows.append(group)

    summary["groups"] = len(group_rows)
    summary["metadata_complete_rate"] = summary["valid_records"] / max(summary["records"], 1)
    summary["groups_with_3plus_views_rate"] = summary["groups_with_3plus_views"] / max(summary["groups"], 1)
    return {"summary": summary, "groups": sorted(group_rows, key=lambda row: row["group_key"])}


def validate_input_dir(input_dir, pattern, check_files=False):
    paths = sorted(Path(input_dir).glob(pattern))
    if not paths:
        raise RuntimeError(f"No files matched {Path(input_dir) / pattern}")
    records = []
    for path in paths:
        raw = np.load(path, allow_pickle=True)
        records.append({"path": str(path), "raw": raw, "validation": validate_record(path, raw, check_files=check_files)})
    return build_view_registry(records)


def main():
    parser = argparse.ArgumentParser("Validate Stage2 metadata and build a Stage3 view-registry dry run")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--pattern", default="*_learned_region_merge_full_pointcloud_editable_planes_data.npz")
    parser.add_argument("--check_files", action="store_true")
    args = parser.parse_args()

    result = validate_input_dir(args.input_dir, args.pattern, check_files=args.check_files)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=json_ready), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
