import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from export_stage1_pred_support_teacher_npz import (
    apply_safe_geometry_merges,
    compute_teacher_iou,
    finite_points,
    image_to_uint8,
    load_stage1_head,
    plane_boundary,
    remap_and_fit_planes,
    resize_label,
)
from train_stage1_clean_baseline import apply_query_class_scores


def parse_args():
    parser = argparse.ArgumentParser("Export Stage1 predicted support teacher NPZ from cached features")
    parser.add_argument("--feature_cache_path", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--count", type=int, default=-1)
    parser.add_argument("--max_planes", type=int, default=8)
    parser.add_argument("--min_plane_points", type=int, default=128)
    parser.add_argument("--min_core_points", type=int, default=64)
    parser.add_argument("--core_confidence", type=float, default=0.55)
    parser.add_argument("--core_margin", type=float, default=0.12)
    parser.add_argument("--max_points", type=int, default=24000)
    parser.add_argument("--enable_safe_geometry_merge", action="store_true")
    parser.add_argument("--geometry_merge_angle_deg", type=float, default=5.0)
    parser.add_argument("--geometry_merge_offset", type=float, default=0.03)
    parser.add_argument("--geometry_merge_residual", type=float, default=0.04)
    parser.add_argument("--geometry_merge_max_boundary_rgb_edge", type=float, default=0.05)
    parser.add_argument("--geometry_merge_min_area_ratio", type=float, default=0.05)
    parser.add_argument("--geometry_merge_adjacency_radius", type=int, default=2)
    return parser.parse_args()


def predict_from_cache(head, saved_args, entry, view_suffix, device):
    features = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in entry[f"features{view_suffix}"].items()
    }
    rgb = entry[f"rgb{view_suffix}"].to(device=device, dtype=torch.float32)
    geometry = None
    if getattr(saved_args, "use_geometry", False):
        geometry = entry[f"geometry{view_suffix}"].to(device=device, dtype=torch.float32)
    output = head(features, rgb=rgb, geometry=geometry)
    mask_logits = apply_query_class_scores(
        output["mask_logits"],
        output["existence_logits"],
        saved_args,
    )[0]
    background_logits = output["background_logits"][0, 0]
    class_logits = torch.cat((mask_logits, background_logits[None]), dim=0)
    probs = torch.softmax(class_logits, dim=0)
    prediction = probs.argmax(dim=0)
    top2 = torch.topk(probs, k=2, dim=0).values
    confidence = top2[0]
    margin = top2[0] - top2[1]
    num_queries = mask_logits.shape[0]
    prediction = torch.where(
        prediction >= num_queries,
        torch.full_like(prediction, -1),
        prediction,
    )
    return prediction, confidence, margin, output["existence_logits"][0].sigmoid()


def export_entry(entry, sample_idx, view_suffix, head, saved_args, checkpoint, args, output_dir, device):
    rgb = entry[f"rgb{view_suffix}"].to(device=device, dtype=torch.float32)
    geometry = entry[f"geometry{view_suffix}"].to(device=device, dtype=torch.float32)
    point_map = geometry[0, :3].permute(1, 2, 0).contiguous()
    target_hw = point_map.shape[:2]
    valid = finite_points(point_map)
    prediction, confidence, margin, existence = predict_from_cache(
        head,
        saved_args,
        entry,
        view_suffix,
        device,
    )
    labels = resize_label(entry[f"gt_plane{view_suffix}"].to(device), target_hw)[0]
    labels = torch.where((labels > 0) & (labels != 255), labels, torch.full_like(labels, -1))

    candidates = remap_and_fit_planes(prediction, confidence, margin, point_map, valid, args)
    if not candidates:
        return None
    merge_records = []
    if args.enable_safe_geometry_merge:
        prediction, merge_records = apply_safe_geometry_merges(
            prediction,
            candidates,
            point_map,
            valid,
            rgb,
            args,
        )
        candidates = remap_and_fit_planes(prediction, confidence, margin, point_map, valid, args)
        if not candidates:
            return None

    remap = {row["query_id"]: i for i, row in enumerate(candidates)}
    source_gt_ids = []
    for row in candidates:
        full_mask = (prediction == row["query_id"]) & valid
        gt_values = labels[full_mask]
        gt_values = gt_values[gt_values >= 0]
        if len(gt_values):
            source_gt_ids.append(int(torch.argmax(torch.bincount(gt_values))))
        else:
            source_gt_ids.append(-1)

    colors = image_to_uint8(rgb, target_hw)
    flat_points = point_map.reshape(-1, 3)
    flat_colors = colors.reshape(-1, 3)
    flat_valid = valid.reshape(-1)
    flat_prediction = prediction.reshape(-1)
    flat_confidence = confidence.reshape(-1)
    flat_margin = margin.reshape(-1)
    flat_labels = labels.reshape(-1)
    flat_boundary = plane_boundary(prediction).reshape(-1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, target_hw[0], device=device),
        torch.linspace(-1.0, 1.0, target_hw[1], device=device),
        indexing="ij",
    )
    flat_pixel_xy = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    valid_indices = torch.nonzero(flat_valid).flatten()
    if args.max_points > 0 and len(valid_indices) > args.max_points:
        sample_positions = torch.linspace(
            0,
            len(valid_indices) - 1,
            steps=args.max_points,
            device=device,
        ).long()
        valid_indices = valid_indices[sample_positions]

    sampled_prediction = flat_prediction[valid_indices]
    point_plane_ids = torch.full_like(sampled_prediction, -1, dtype=torch.int32)
    for query_id, output_id in remap.items():
        point_plane_ids[sampled_prediction == int(query_id)] = int(output_id)

    sampled_gt = flat_labels[valid_indices].cpu().numpy().astype(np.int32)
    ious, leakage = compute_teacher_iou(
        point_plane_ids.cpu().numpy().astype(np.int32),
        sampled_gt,
        source_gt_ids,
    )
    normals = torch.stack([row["normal"] for row in candidates]).cpu().numpy().astype(np.float32)
    offsets = np.asarray([float(row["offset"]) for row in candidates], dtype=np.float32)
    counts = np.asarray([int((point_plane_ids == i).sum()) for i in range(len(candidates))], dtype=np.int32)
    stem = f"{args.split}_{sample_idx:06d}_view{view_suffix}_stage1_teacher_full_pointcloud_editable_planes_data"
    output_path = output_dir / f"{stem}.npz"
    np.savez_compressed(
        output_path,
        points=flat_points[valid_indices].cpu().numpy().astype(np.float32),
        colors=flat_colors[valid_indices].cpu().numpy().astype(np.uint8),
        original_colors=flat_colors[valid_indices].cpu().numpy().astype(np.uint8),
        pixel_xy=flat_pixel_xy[valid_indices].cpu().numpy().astype(np.float32),
        point_plane_ids=point_plane_ids.cpu().numpy().astype(np.int32),
        gt_point_plane_ids=sampled_gt,
        plane_normals=normals,
        plane_offsets=offsets,
        plane_inlier_counts=counts,
        plane_ids=np.arange(len(candidates), dtype=np.int32),
        source_gt_plane_ids=np.asarray(source_gt_ids, dtype=np.int32),
        line_prob=flat_boundary[valid_indices].cpu().numpy().astype(np.float32),
        point_confidence=flat_confidence[valid_indices].cpu().numpy().astype(np.float32),
        point_margin=flat_margin[valid_indices].cpu().numpy().astype(np.float32),
        query_ids=np.asarray([row["query_id"] for row in candidates], dtype=np.int32),
        query_existence=existence.cpu().numpy().astype(np.float32),
        full_mask_counts=np.asarray([row["full_count"] for row in candidates], dtype=np.int32),
        core_mask_counts=np.asarray([row["core_count"] for row in candidates], dtype=np.int32),
        fit_mask_counts=np.asarray([row["fit_count"] for row in candidates], dtype=np.int32),
    )
    return {
        "sample_idx": int(sample_idx),
        "view": int(view_suffix),
        "output": str(output_path),
        "points": int(len(valid_indices)),
        "planes": len(candidates),
        "background_points": int((point_plane_ids < 0).sum()),
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "stage1_epoch": int(checkpoint.get("epoch", -1)),
        "geometry_source": "cached_geometry_channels_0_1_2",
        "use_for_final_plane_params": False,
        "safe_geometry_merge_enabled": bool(args.enable_safe_geometry_merge),
        "safe_geometry_merge_pairs": merge_records,
        "mean_teacher_iou": float(np.mean(ious)) if ious else 0.0,
        "leakage": float(leakage),
        "mean_fit_residual": float(np.mean([row["mean_residual"] for row in candidates])),
        "max_fit_residual": float(np.max([row["mean_residual"] for row in candidates])),
    }


@torch.no_grad()
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.feature_cache_path, map_location="cpu", weights_only=False)
    entries = cache[args.split]
    cache_config = cache.get("config", {})
    indices = cache_config.get("selected_val_indices" if args.split == "val" else "selected_train_indices")
    if indices is None:
        indices = list(range(len(entries)))
    start = max(args.start_idx, 0)
    end = len(entries) if args.count < 0 else min(len(entries), start + args.count)
    head, saved_args, checkpoint = load_stage1_head(args.stage1_checkpoint, cache_config, device)

    manifest = []
    for local_idx in range(start, end):
        entry = entries[local_idx]
        sample_idx = int(indices[local_idx])
        for view_suffix in ("1", "2"):
            row = export_entry(
                entry,
                sample_idx,
                view_suffix,
                head,
                saved_args,
                checkpoint,
                args,
                output_dir,
                device,
            )
            if row is not None:
                manifest.append(row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
        if (local_idx - start + 1) % 16 == 0:
            print(f"[cache export] {local_idx - start + 1}/{end - start}", flush=True)

    manifest_path = output_dir / "stage1_pred_support_teacher_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
