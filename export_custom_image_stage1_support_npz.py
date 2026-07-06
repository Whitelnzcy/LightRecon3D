import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from export_stage1_pred_support_teacher_npz import (
    STAGE2_SCHEMA_VERSION,
    apply_duplicate_geometry_merges,
    apply_safe_geometry_merges,
    finite_points,
    image_to_uint8,
    load_stage1_head,
    plane_boundary,
    predict_stage1,
    remap_and_fit_planes,
)
from models.build_backbone import build_dust3r_backbone
from train_stage1_plane_masks import build_views, point_map_from_result


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_image_paths(image_dir, images):
    if images:
        paths = [Path(path) for path in images.split(",") if path.strip()]
    else:
        root = Path(image_dir)
        paths = [
            path
            for path in sorted(root.iterdir())
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    paths = [path.expanduser().resolve() for path in paths]
    if len(paths) < 2:
        raise RuntimeError("Custom image demo needs at least two RGB images")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing images: {missing}")
    return paths


def load_rgb_tensor(path, image_size):
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = image_rgb.shape[:2]
    image_rgb = cv2.resize(
        image_rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    return tensor, np.asarray([original_h, original_w], dtype=np.int32)


def build_pairs(paths, pair_strategy):
    if pair_strategy == "adjacent":
        return [(i, i + 1) for i in range(len(paths) - 1)]
    if pair_strategy == "all":
        return [(i, j) for i in range(len(paths)) for j in range(i + 1, len(paths))]
    raise ValueError(f"Unsupported pair_strategy={pair_strategy}")


def sample_view_support(
    point_map,
    image,
    prediction,
    confidence,
    margin,
    candidates,
    plane_offset,
    source_view,
    max_points,
):
    target_hw = point_map.shape[:2]
    valid = finite_points(point_map)
    remap = {row["query_id"]: plane_offset + i for i, row in enumerate(candidates)}

    colors = image_to_uint8(image[None], target_hw)
    flat_points = point_map.reshape(-1, 3)
    flat_colors = colors.reshape(-1, 3)
    flat_valid = valid.reshape(-1)
    flat_prediction = prediction.reshape(-1)
    flat_confidence = confidence.reshape(-1)
    flat_margin = margin.reshape(-1)
    flat_boundary = plane_boundary(prediction).reshape(-1)

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, target_hw[0], device=point_map.device),
        torch.linspace(-1.0, 1.0, target_hw[1], device=point_map.device),
        indexing="ij",
    )
    flat_pixel_xy = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    valid_indices = torch.nonzero(flat_valid).flatten()
    if max_points > 0 and len(valid_indices) > max_points:
        sample_positions = torch.linspace(
            0,
            len(valid_indices) - 1,
            steps=max_points,
            device=point_map.device,
        ).long()
        valid_indices = valid_indices[sample_positions]

    sampled_prediction = flat_prediction[valid_indices]
    point_plane_ids = torch.full_like(sampled_prediction, -1, dtype=torch.int32)
    for query_id, output_id in remap.items():
        point_plane_ids[sampled_prediction == int(query_id)] = int(output_id)

    count = int(len(valid_indices))
    pixel_xy = flat_pixel_xy[valid_indices].detach().cpu().numpy().astype(np.float32)
    pixel_xy1 = np.zeros((count, 2), dtype=np.float32)
    pixel_xy2 = np.zeros((count, 2), dtype=np.float32)
    if source_view == 1:
        pixel_xy1 = pixel_xy.copy()
    elif source_view == 2:
        pixel_xy2 = pixel_xy.copy()
    else:
        raise ValueError(f"source_view must be 1 or 2, got {source_view}")

    return {
        "points": flat_points[valid_indices].detach().cpu().numpy().astype(np.float32),
        "colors": flat_colors[valid_indices].detach().cpu().numpy().astype(np.uint8),
        "point_plane_ids": point_plane_ids.detach().cpu().numpy().astype(np.int32),
        "line_prob": flat_boundary[valid_indices].detach().cpu().numpy().astype(np.float32),
        "point_confidence": flat_confidence[valid_indices].detach().cpu().numpy().astype(np.float32),
        "point_margin": flat_margin[valid_indices].detach().cpu().numpy().astype(np.float32),
        "pixel_xy": pixel_xy,
        "pixel_xy1": pixel_xy1,
        "pixel_xy2": pixel_xy2,
        "support_source_view": np.full((count,), source_view, dtype=np.int8),
        "stage1_mask_hw": np.asarray(target_hw, dtype=np.int32),
    }


def predict_view(head, saved_args, result, image, args):
    point_map = point_map_from_result(result)[0]
    target_hw = point_map.shape[:2]
    valid = finite_points(point_map)
    prediction, confidence, margin, existence = predict_stage1(
        head,
        saved_args,
        result,
        image[None],
        target_hw,
    )
    candidates = remap_and_fit_planes(prediction, confidence, margin, point_map, valid, args)
    merge_records = []
    duplicate_merge_records = []
    if candidates and args.enable_safe_geometry_merge:
        prediction, merge_records = apply_safe_geometry_merges(
            prediction,
            candidates,
            point_map,
            valid,
            image[None],
            args,
        )
        candidates = remap_and_fit_planes(prediction, confidence, margin, point_map, valid, args)
    if candidates and args.enable_duplicate_geometry_merge:
        prediction, duplicate_merge_records = apply_duplicate_geometry_merges(
            prediction,
            candidates,
            point_map,
            valid,
            args,
        )
        candidates = remap_and_fit_planes(prediction, confidence, margin, point_map, valid, args)
    return point_map, prediction, confidence, margin, existence, candidates, merge_records, duplicate_merge_records


def concatenate_arrays(rows, key):
    values = [row[key] for row in rows if len(row[key]) > 0]
    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(values, axis=0)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Export LightRecon3D Stage1 support for ordinary custom RGB images")
    parser.add_argument("--image_dir", default="")
    parser.add_argument("--images", default="", help="Comma-separated image paths. Overrides --image_dir.")
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_cache_path", default="")
    parser.add_argument("--scene_name", default="custom_scene")
    parser.add_argument("--pair_group", default="")
    parser.add_argument("--pair_strategy", default="all", choices=("adjacent", "all"))
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_planes", type=int, default=8)
    parser.add_argument("--min_plane_points", type=int, default=128)
    parser.add_argument("--min_core_points", type=int, default=64)
    parser.add_argument("--core_confidence", type=float, default=0.55)
    parser.add_argument("--core_margin", type=float, default=0.12)
    parser.add_argument("--enable_safe_geometry_merge", action="store_true")
    parser.add_argument("--geometry_merge_angle_deg", type=float, default=5.0)
    parser.add_argument("--geometry_merge_offset", type=float, default=0.03)
    parser.add_argument("--geometry_merge_residual", type=float, default=0.04)
    parser.add_argument("--geometry_merge_max_boundary_rgb_edge", type=float, default=0.05)
    parser.add_argument("--geometry_merge_min_area_ratio", type=float, default=0.05)
    parser.add_argument("--geometry_merge_adjacency_radius", type=int, default=2)
    parser.add_argument("--enable_duplicate_geometry_merge", action="store_true")
    parser.add_argument("--duplicate_merge_angle_deg", type=float, default=3.0)
    parser.add_argument("--duplicate_merge_offset", type=float, default=0.02)
    parser.add_argument("--duplicate_merge_residual", type=float, default=0.025)
    parser.add_argument("--duplicate_merge_min_area_ratio", type=float, default=0.015)
    parser.add_argument("--max_points", type=int, default=4000)
    args = parser.parse_args()

    paths = parse_image_paths(args.image_dir, args.images)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_group = args.pair_group or str(Path(args.image_dir or paths[0].parent).expanduser().resolve())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_config = {}
    if args.feature_cache_path:
        cache_config = torch.load(args.feature_cache_path, map_location="cpu", weights_only=False).get("config", {})
    backbone = build_dust3r_backbone(args.weights_path, device=device)
    backbone.eval()
    head, saved_args, checkpoint = load_stage1_head(args.stage1_checkpoint, cache_config, device)

    images = []
    original_hws = []
    for path in paths:
        image, original_hw = load_rgb_tensor(path, args.image_size)
        images.append(image.to(device))
        original_hws.append(original_hw)

    manifest = []
    for pair_idx, (idx1, idx2) in enumerate(build_pairs(paths, args.pair_strategy)):
        img1, img2 = images[idx1], images[idx2]
        batch = {
            "img": img1[None],
            "img1": img1[None],
            "img2": img2[None],
        }
        view1, view2 = build_views(batch, f"custom_stage1_pair_{pair_idx:06d}")
        result1, result2 = backbone(view1, view2)

        view1_pred = predict_view(head, saved_args, result1, img1, args)
        view2_pred = predict_view(head, saved_args, result2, img2, args)
        (
            point_map1,
            prediction1,
            confidence1,
            margin1,
            existence1,
            candidates1,
            merge_records1,
            duplicate_records1,
        ) = view1_pred
        (
            point_map2,
            prediction2,
            confidence2,
            margin2,
            existence2,
            candidates2,
            merge_records2,
            duplicate_records2,
        ) = view2_pred

        all_candidates = list(candidates1) + list(candidates2)
        if not all_candidates:
            print(f"skip pair={pair_idx}: no predicted planes")
            continue

        support_rows = []
        if candidates1:
            support_rows.append(
                sample_view_support(
                    point_map1,
                    img1,
                    prediction1,
                    confidence1,
                    margin1,
                    candidates1,
                    plane_offset=0,
                    source_view=1,
                    max_points=args.max_points,
                )
            )
        if candidates2:
            support_rows.append(
                sample_view_support(
                    point_map2,
                    img2,
                    prediction2,
                    confidence2,
                    margin2,
                    candidates2,
                    plane_offset=len(candidates1),
                    source_view=2,
                    max_points=args.max_points,
                )
            )

        combined_points = concatenate_arrays(support_rows, "points")
        combined_colors = concatenate_arrays(support_rows, "colors")
        combined_point_plane_ids = concatenate_arrays(support_rows, "point_plane_ids").astype(np.int32)
        combined_line_prob = concatenate_arrays(support_rows, "line_prob").astype(np.float32)
        combined_confidence = concatenate_arrays(support_rows, "point_confidence").astype(np.float32)
        combined_margin = concatenate_arrays(support_rows, "point_margin").astype(np.float32)
        combined_pixel_xy = concatenate_arrays(support_rows, "pixel_xy").astype(np.float32)
        pixel_xy1 = concatenate_arrays(support_rows, "pixel_xy1").astype(np.float32)
        pixel_xy2 = concatenate_arrays(support_rows, "pixel_xy2").astype(np.float32)
        support_source_view = concatenate_arrays(support_rows, "support_source_view").astype(np.int8)

        normals = torch.stack([row["normal"] for row in all_candidates]).cpu().numpy().astype(np.float32)
        offsets = np.asarray([float(row["offset"]) for row in all_candidates], dtype=np.float32)
        counts = np.asarray(
            [int((combined_point_plane_ids == i).sum()) for i in range(len(all_candidates))],
            dtype=np.int32,
        )
        gt_assignment = np.full_like(combined_point_plane_ids, -1, dtype=np.int32)
        stem = f"custom_{pair_idx:06d}_{idx1:03d}_{idx2:03d}_stage1_teacher_full_pointcloud_editable_planes_data"
        output_path = output_dir / f"{stem}.npz"
        stage1_mask_hw = np.asarray([args.image_size, args.image_size], dtype=np.int32)
        np.savez_compressed(
            output_path,
            schema_version=np.asarray(STAGE2_SCHEMA_VERSION, dtype=np.int32),
            points=combined_points,
            colors=combined_colors,
            original_colors=combined_colors,
            scene_name=np.asarray(args.scene_name),
            pair_group=np.asarray(pair_group),
            rgb_path1=np.asarray(str(paths[idx1])),
            rgb_path2=np.asarray(str(paths[idx2])),
            json_path1=np.asarray(""),
            json_path2=np.asarray(""),
            view_id1=np.asarray(str(idx1)),
            view_id2=np.asarray(str(idx2)),
            original_hw1=original_hws[idx1],
            original_hw2=original_hws[idx2],
            stage1_input_hw1=stage1_mask_hw,
            stage1_input_hw2=stage1_mask_hw,
            stage1_mask_hw1=stage1_mask_hw,
            stage1_mask_hw2=stage1_mask_hw,
            pixel_coordinate_space=np.asarray("stage1_pointmap_normalized"),
            pixel_coordinate_order=np.asarray("xy"),
            pixel_coordinate_range=np.asarray("[-1,1]"),
            pixel_coordinate_view=np.asarray("mixed_source_view"),
            sample_idx=np.asarray(int(pair_idx), dtype=np.int32),
            pixel_xy=combined_pixel_xy,
            pixel_xy1=pixel_xy1,
            pixel_xy2=pixel_xy2,
            support_source_view=support_source_view,
            point_plane_ids=combined_point_plane_ids,
            gt_point_plane_ids=gt_assignment,
            plane_normals=normals,
            plane_offsets=offsets,
            plane_inlier_counts=counts,
            plane_ids=np.arange(len(all_candidates), dtype=np.int32),
            source_gt_plane_ids=np.full((len(all_candidates),), -1, dtype=np.int32),
            line_prob=combined_line_prob,
            point_confidence=combined_confidence,
            point_margin=combined_margin,
            query_ids=np.asarray([row["query_id"] for row in all_candidates], dtype=np.int32),
            query_existence=existence1.cpu().numpy().astype(np.float32),
            query_existence2=existence2.cpu().numpy().astype(np.float32),
            full_mask_counts=np.asarray([row["full_count"] for row in all_candidates], dtype=np.int32),
            core_mask_counts=np.asarray([row["core_count"] for row in all_candidates], dtype=np.int32),
            fit_mask_counts=np.asarray([row["fit_count"] for row in all_candidates], dtype=np.int32),
        )
        row = {
            "pair_idx": int(pair_idx),
            "view_index1": int(idx1),
            "view_index2": int(idx2),
            "rgb_path1": str(paths[idx1]),
            "rgb_path2": str(paths[idx2]),
            "output": str(output_path),
            "points": int(len(combined_point_plane_ids)),
            "planes": int(len(all_candidates)),
            "background_points": int((combined_point_plane_ids < 0).sum()),
            "stage1_checkpoint": str(args.stage1_checkpoint),
            "stage1_epoch": int(checkpoint.get("epoch", -1)),
            "safe_geometry_merge_pairs": [{**row, "view": 1} for row in merge_records1]
            + [{**row, "view": 2} for row in merge_records2],
            "duplicate_geometry_merge_pairs": [{**row, "view": 1} for row in duplicate_records1]
            + [{**row, "view": 2} for row in duplicate_records2],
        }
        manifest.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)

    manifest_path = output_dir / "custom_stage1_pred_support_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
