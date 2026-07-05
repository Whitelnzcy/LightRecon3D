import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from cache_stage1_multiscale_symref import geometry_from_result
from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone
from models.multiscale_plane_mask_head import MultiScalePlaneMaskHead
from train_stage1_clean_pair_baseline import move_batch, parse_indices
from train_stage1_plane_masks import build_views, feature_maps_from_result, point_map_from_result


STAGE_NAMES = ("encoder", "shallow", "middle", "deep")
STAGE2_SCHEMA_VERSION = 2


def apply_query_class_scores(mask_logits, existence_logits, args):
    weight = float(getattr(args, "class_score_weight", 0.0))
    if weight <= 0.0:
        return mask_logits
    if mask_logits.dim() == 4:
        return mask_logits + weight * existence_logits[:, :, None, None]
    if mask_logits.dim() == 3:
        return mask_logits + weight * existence_logits[:, None, None]
    raise ValueError(f"Unsupported mask logits shape: {tuple(mask_logits.shape)}")


def batch_value(batch, key, default=""):
    value = batch.get(key, default)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        if value.numel() == 1:
            return value.detach().cpu().reshape(-1)[0].item()
        return value.detach().cpu()[0].numpy()
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def batch_string(batch, key):
    return str(batch_value(batch, key, ""))


def batch_hw(batch, key):
    value = batch_value(batch, key, np.asarray([0, 0], dtype=np.int32))
    return np.asarray(value, dtype=np.int32).reshape(-1)[:2]


def resize_label(label, target_hw):
    if label.shape[-2:] == target_hw:
        return label.long()
    return F.interpolate(label[:, None].float(), size=target_hw, mode="nearest")[:, 0].long()


def finite_points(points):
    return torch.isfinite(points).all(dim=-1) & (points.abs().amax(dim=-1) < 1e4)


def plane_boundary(labels):
    valid = labels >= 0
    boundary = torch.zeros_like(labels, dtype=torch.float32)
    horizontal = (labels[:, 1:] != labels[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    vertical = (labels[1:, :] != labels[:-1, :]) & valid[1:, :] & valid[:-1, :]
    boundary[:, 1:][horizontal] = 1.0
    boundary[:, :-1][horizontal] = 1.0
    boundary[1:, :][vertical] = 1.0
    boundary[:-1, :][vertical] = 1.0
    return F.max_pool2d(boundary[None, None], kernel_size=3, stride=1, padding=1)[0, 0]


def image_to_uint8(image, target_hw):
    image = F.interpolate(image, size=target_hw, mode="bilinear", align_corners=False)[0]
    if float(image.min()) < -0.05:
        image = (image + 1.0) * 0.5
    return (image.clamp(0.0, 1.0).permute(1, 2, 0) * 255.0).byte()


def fit_plane(points):
    centroid = points.mean(dim=0)
    centered = points - centroid
    _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
    normal = vh[-1]
    normal = normal / normal.norm().clamp_min(1e-8)
    dominant = torch.argmax(torch.abs(normal))
    if normal[dominant] < 0:
        normal = -normal
    offset = -torch.dot(normal, centroid)
    residual = torch.abs(points @ normal + offset)
    return normal, offset, residual


def plane_angle_deg(normal_a, normal_b):
    dot = torch.abs(torch.sum(normal_a * normal_b)).clamp(0.0, 1.0)
    return float(torch.rad2deg(torch.acos(dot)))


def mutual_plane_residual(points_a, plane_a, points_b, plane_b):
    normal_a, offset_a = plane_a["normal"], plane_a["offset"]
    normal_b, offset_b = plane_b["normal"], plane_b["offset"]
    residual_ab = torch.abs(points_a @ normal_b + offset_b).mean()
    residual_ba = torch.abs(points_b @ normal_a + offset_a).mean()
    return float(0.5 * (residual_ab + residual_ba))


def adjacent_band(mask_a, mask_b, radius):
    kernel = radius * 2 + 1
    a = mask_a.float()[None, None]
    b = mask_b.float()[None, None]
    dilated_a = F.max_pool2d(a, kernel_size=kernel, stride=1, padding=radius)[0, 0] > 0
    dilated_b = F.max_pool2d(b, kernel_size=kernel, stride=1, padding=radius)[0, 0] > 0
    return (dilated_a & mask_b) | (dilated_b & mask_a)


def rgb_edge_map_from_image(image, target_hw):
    rgb = F.interpolate(image, size=target_hw, mode="bilinear", align_corners=False)[0]
    if float(rgb.min()) < -0.05:
        rgb = (rgb + 1.0) * 0.5
    gray = rgb.clamp(0.0, 1.0).mean(dim=0)
    edge = torch.zeros_like(gray)
    edge[:, 1:] = torch.maximum(edge[:, 1:], torch.abs(gray[:, 1:] - gray[:, :-1]))
    edge[:, :-1] = torch.maximum(edge[:, :-1], torch.abs(gray[:, 1:] - gray[:, :-1]))
    edge[1:, :] = torch.maximum(edge[1:, :], torch.abs(gray[1:, :] - gray[:-1, :]))
    edge[:-1, :] = torch.maximum(edge[:-1, :], torch.abs(gray[1:, :] - gray[:-1, :]))
    return F.max_pool2d(edge[None, None], kernel_size=3, stride=1, padding=1)[0, 0]


def merge_query_ids(prediction, merge_edges):
    if not merge_edges:
        return prediction, {}
    query_ids = sorted({int(q) for edge in merge_edges for q in edge})
    parent = {q: q for q in query_ids}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in merge_edges:
        root_a, root_b = find(int(a)), find(int(b))
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    remap = {q: find(q) for q in query_ids}
    corrected = prediction.clone()
    for old_q, new_q in remap.items():
        corrected[prediction == int(old_q)] = int(new_q)
    return corrected, remap


def apply_safe_geometry_merges(prediction, candidates, point_map, valid, image, args):
    if not args.enable_safe_geometry_merge:
        return prediction, []
    rgb_edge = rgb_edge_map_from_image(image, prediction.shape[-2:]).to(prediction.device)
    query_points = {}
    query_masks = {}
    candidate_by_query = {}
    for row in candidates:
        query_id = int(row["query_id"])
        mask = (prediction == query_id) & valid
        if int(mask.sum()) < args.min_plane_points:
            continue
        query_masks[query_id] = mask
        query_points[query_id] = point_map[mask]
        candidate_by_query[query_id] = row

    merge_edges = []
    query_ids = sorted(query_masks)
    for i, query_a in enumerate(query_ids):
        for query_b in query_ids[i + 1 :]:
            mask_a, mask_b = query_masks[query_a], query_masks[query_b]
            band = adjacent_band(mask_a, mask_b, args.geometry_merge_adjacency_radius)
            if int(band.sum()) == 0:
                continue
            area_a, area_b = int(mask_a.sum()), int(mask_b.sum())
            area_ratio = min(area_a, area_b) / max(max(area_a, area_b), 1)
            plane_a = candidate_by_query[query_a]
            plane_b = candidate_by_query[query_b]
            angle = plane_angle_deg(plane_a["normal"], plane_b["normal"])
            offset = abs(abs(float(plane_a["offset"])) - abs(float(plane_b["offset"])))
            residual = mutual_plane_residual(
                query_points[query_a],
                plane_a,
                query_points[query_b],
                plane_b,
            )
            boundary_rgb = float(rgb_edge[band].mean()) if int(band.sum()) else 0.0
            if (
                angle <= args.geometry_merge_angle_deg
                and offset <= args.geometry_merge_offset
                and residual <= args.geometry_merge_residual
                and boundary_rgb <= args.geometry_merge_max_boundary_rgb_edge
                and area_ratio >= args.geometry_merge_min_area_ratio
            ):
                merge_edges.append((query_a, query_b))
    corrected, remap = merge_query_ids(prediction, merge_edges)
    return corrected, [
        {
            "type": "adjacent_safe",
            "query_a": int(a),
            "query_b": int(b),
            "merged_to": int(remap.get(int(a), int(a))),
        }
        for a, b in merge_edges
    ]


def apply_duplicate_geometry_merges(prediction, candidates, point_map, valid, args):
    if not args.enable_duplicate_geometry_merge:
        return prediction, []
    query_points = {}
    query_masks = {}
    candidate_by_query = {}
    for row in candidates:
        query_id = int(row["query_id"])
        mask = (prediction == query_id) & valid
        if int(mask.sum()) < args.min_plane_points:
            continue
        query_masks[query_id] = mask
        query_points[query_id] = point_map[mask]
        candidate_by_query[query_id] = row

    merge_edges = []
    merge_rows = []
    query_ids = sorted(query_masks)
    for i, query_a in enumerate(query_ids):
        for query_b in query_ids[i + 1 :]:
            mask_a, mask_b = query_masks[query_a], query_masks[query_b]
            area_a, area_b = int(mask_a.sum()), int(mask_b.sum())
            area_ratio = min(area_a, area_b) / max(max(area_a, area_b), 1)
            if area_ratio < args.duplicate_merge_min_area_ratio:
                continue
            plane_a = candidate_by_query[query_a]
            plane_b = candidate_by_query[query_b]
            angle = plane_angle_deg(plane_a["normal"], plane_b["normal"])
            offset = abs(abs(float(plane_a["offset"])) - abs(float(plane_b["offset"])))
            residual = mutual_plane_residual(
                query_points[query_a],
                plane_a,
                query_points[query_b],
                plane_b,
            )
            if (
                angle <= args.duplicate_merge_angle_deg
                and offset <= args.duplicate_merge_offset
                and residual <= args.duplicate_merge_residual
            ):
                merge_edges.append((query_a, query_b))
                merge_rows.append(
                    {
                        "type": "near_duplicate",
                        "query_a": int(query_a),
                        "query_b": int(query_b),
                        "angle_deg": float(angle),
                        "offset_abs_diff": float(offset),
                        "mutual_residual": float(residual),
                        "area_ratio": float(area_ratio),
                    }
                )
    corrected, remap = merge_query_ids(prediction, merge_edges)
    for row in merge_rows:
        row["merged_to"] = int(remap.get(int(row["query_a"]), int(row["query_a"])))
    return corrected, merge_rows


def build_dataset(args, cache_config=None):
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        input_mode="pair",
        pair_strategy=args.pair_strategy,
        pair_max_view_id_gap=args.pair_max_view_id_gap,
    )
    indices = parse_indices(args.indices)
    if indices is None and cache_config:
        key = "selected_train_indices" if args.split == "train" else "selected_val_indices"
        indices = cache_config.get(key)
    if indices is None:
        end = min(len(dataset), args.start_idx + args.count)
        indices = list(range(args.start_idx, end))
    elif args.count > 0:
        indices = indices[args.start_idx : args.start_idx + args.count]
    indices = [int(index) for index in indices if 0 <= int(index) < len(dataset)]
    if not indices:
        raise RuntimeError("No dataset indices selected for export")
    return dataset, indices


def load_stage1_head(checkpoint_path, cache_config, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = argparse.Namespace(**checkpoint["args"])
    if not hasattr(saved_args, "use_geometry"):
        saved_args.use_geometry = False
    if not hasattr(saved_args, "use_masked_query_refine"):
        saved_args.use_masked_query_refine = False
    if not hasattr(saved_args, "decoder_ffn_multiplier"):
        saved_args.decoder_ffn_multiplier = 4
    if not hasattr(saved_args, "fuse_refine_blocks"):
        saved_args.fuse_refine_blocks = 1
    if not hasattr(saved_args, "pixel_refine_blocks"):
        saved_args.pixel_refine_blocks = 1
    input_dims = tuple(
        int(value)
        for value in checkpoint.get(
            "input_dims",
            cache_config.get("input_dims", (1024, 768, 768, 768)),
        )
    )
    head = MultiScalePlaneMaskHead(
        input_dims=input_dims,
        hidden_dim=saved_args.hidden_dim,
        num_queries=saved_args.num_queries,
        num_decoder_layers=saved_args.decoder_layers,
        num_heads=saved_args.decoder_heads,
        output_size=saved_args.output_size,
        use_rgb_skip=not saved_args.disable_rgb_skip,
        use_geometry=saved_args.use_geometry,
        geometry_dim=int(cache_config.get("geometry_channels", 9)),
        use_masked_query_refine=saved_args.use_masked_query_refine,
        decoder_ffn_multiplier=saved_args.decoder_ffn_multiplier,
        fuse_refine_blocks=saved_args.fuse_refine_blocks,
        pixel_refine_blocks=saved_args.pixel_refine_blocks,
    ).to(device)
    head.load_state_dict(checkpoint["head"], strict=True)
    head.eval()
    return head, saved_args, checkpoint


def predict_stage1(head, saved_args, result, image, target_hw):
    features = feature_maps_from_result(result, image)
    features = {key: value.float() for key, value in features.items()}
    rgb = F.interpolate(image, size=(128, 128), mode="bilinear", align_corners=False)
    geometry = None
    if getattr(saved_args, "use_geometry", False):
        geometry = geometry_from_result(result, (128, 128)).to(device=image.device, dtype=torch.float32)
    output = head(features, rgb=rgb, geometry=geometry)
    mask_logits = F.interpolate(
        apply_query_class_scores(output["mask_logits"], output["existence_logits"], saved_args),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )[0]
    background_logits = F.interpolate(
        output["background_logits"],
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
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


def remap_and_fit_planes(prediction, confidence, margin, point_map, valid, args):
    candidates = []
    for query_id in torch.unique(prediction):
        query_id = int(query_id)
        if query_id < 0:
            continue
        full_mask = (prediction == query_id) & valid
        full_count = int(full_mask.sum())
        if full_count < args.min_plane_points:
            continue
        boundary = plane_boundary(prediction)
        core_mask = (
            full_mask
            & (confidence >= args.core_confidence)
            & (margin >= args.core_margin)
            & (boundary <= 0.0)
        )
        fit_mask = core_mask if int(core_mask.sum()) >= args.min_core_points else full_mask
        normal, offset, residual = fit_plane(point_map[fit_mask])
        candidates.append(
            {
                "query_id": query_id,
                "full_count": full_count,
                "core_count": int(core_mask.sum()),
                "fit_count": int(fit_mask.sum()),
                "normal": normal,
                "offset": offset,
                "mean_residual": float(residual.mean()),
                "p95_residual": float(torch.quantile(residual, 0.95)),
            }
        )
    candidates.sort(key=lambda row: row["full_count"], reverse=True)
    return candidates[: args.max_planes]


def compute_teacher_iou(point_plane_ids, gt_ids, plane_source_gt_ids):
    rows = []
    for plane_id, gt_id in enumerate(plane_source_gt_ids):
        pred = point_plane_ids == plane_id
        gt = gt_ids == int(gt_id)
        union = np.count_nonzero(pred | gt)
        rows.append(float(np.count_nonzero(pred & gt) / union) if union else 0.0)
    pred_fg = point_plane_ids >= 0
    if pred_fg.any():
        mapped_gt = np.full_like(point_plane_ids, -9999, dtype=np.int32)
        for plane_id, gt_id in enumerate(plane_source_gt_ids):
            mapped_gt[point_plane_ids == plane_id] = int(gt_id)
        leakage = float(np.mean(mapped_gt[pred_fg] != gt_ids[pred_fg]))
    else:
        leakage = 0.0
    return rows, leakage


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Export Stage1 predicted bounded-plane support for Stage2")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_cache_path", default="")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--indices", default="")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--pair_strategy", default="adjacent", choices=("adjacent", "all"))
    parser.add_argument("--pair_max_view_id_gap", type=int, default=None)
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
    parser.add_argument("--max_points", type=int, default=24000)
    parser.add_argument("--export_second_view_support", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_config = {}
    if args.feature_cache_path:
        cache_config = torch.load(args.feature_cache_path, map_location="cpu", weights_only=False).get("config", {})
    dataset, indices = build_dataset(args, cache_config=cache_config)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    backbone = build_dust3r_backbone(args.weights_path, device=device)
    backbone.eval()
    head, saved_args, checkpoint = load_stage1_head(args.stage1_checkpoint, cache_config, device)

    manifest = []
    for ordinal, batch in enumerate(loader):
        sample_idx = indices[ordinal]
        batch = move_batch(batch, device)
        view1, view2 = build_views(batch, f"stage1_pred_support_{args.split}_{sample_idx}")
        result1, result2 = backbone(view1, view2)
        point_map = point_map_from_result(result1)[0]
        target_hw = point_map.shape[:2]
        valid = finite_points(point_map)
        prediction, confidence, margin, existence = predict_stage1(
            head,
            saved_args,
            result1,
            view1["img"],
            target_hw,
        )
        labels = resize_label(batch["gt_plane1"], target_hw)[0]
        labels = torch.where((labels > 0) & (labels != 255), labels, torch.full_like(labels, -1))
        candidates = remap_and_fit_planes(prediction, confidence, margin, point_map, valid, args)
        if not candidates:
            print(f"skip sample={sample_idx}: no predicted planes")
            continue
        merge_records = []
        if args.enable_safe_geometry_merge:
            prediction, merge_records = apply_safe_geometry_merges(
                prediction,
                candidates,
                point_map,
                valid,
                view1["img"],
                args,
            )
            candidates = remap_and_fit_planes(prediction, confidence, margin, point_map, valid, args)
            if not candidates:
                print(f"skip sample={sample_idx}: no predicted planes after geometry merge")
                continue
        duplicate_merge_records = []
        if args.enable_duplicate_geometry_merge:
            prediction, duplicate_merge_records = apply_duplicate_geometry_merges(
                prediction,
                candidates,
                point_map,
                valid,
                args,
            )
            candidates = remap_and_fit_planes(prediction, confidence, margin, point_map, valid, args)
            if not candidates:
                print(f"skip sample={sample_idx}: no predicted planes after duplicate geometry merge")
                continue

        remap = {row["query_id"]: i for i, row in enumerate(candidates)}
        source_gt_ids = []
        full_masks = []
        core_masks = []
        for row in candidates:
            full_mask = (prediction == row["query_id"]) & valid
            boundary = plane_boundary(prediction)
            core_mask = (
                full_mask
                & (confidence >= args.core_confidence)
                & (margin >= args.core_margin)
                & (boundary <= 0.0)
            )
            full_masks.append(full_mask)
            core_masks.append(core_mask)
            gt_values = labels[full_mask]
            gt_values = gt_values[gt_values >= 0]
            if len(gt_values):
                counts = torch.bincount(gt_values)
                source_gt_ids.append(int(torch.argmax(counts)))
            else:
                source_gt_ids.append(-1)

        colors = image_to_uint8(batch["img1"], target_hw)
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
        stem = f"{args.split}_{sample_idx:06d}_stage1_teacher_full_pointcloud_editable_planes_data"
        output_path = output_dir / f"{stem}.npz"
        sampled_pixel_xy = flat_pixel_xy[valid_indices].cpu().numpy().astype(np.float32)
        sampled_count = int(len(valid_indices))
        stage1_mask_hw = np.asarray(target_hw, dtype=np.int32)

        combined_points = flat_points[valid_indices].cpu().numpy().astype(np.float32)
        combined_colors = flat_colors[valid_indices].cpu().numpy().astype(np.uint8)
        combined_point_plane_ids = point_plane_ids.cpu().numpy().astype(np.int32)
        combined_gt = sampled_gt
        combined_line_prob = flat_boundary[valid_indices].cpu().numpy().astype(np.float32)
        combined_confidence = flat_confidence[valid_indices].cpu().numpy().astype(np.float32)
        combined_margin = flat_margin[valid_indices].cpu().numpy().astype(np.float32)
        combined_pixel_xy = sampled_pixel_xy
        pixel_xy1 = sampled_pixel_xy.copy()
        pixel_xy2 = np.zeros_like(sampled_pixel_xy, dtype=np.float32)
        support_source_view = np.ones((sampled_count,), dtype=np.int8)
        all_candidates = list(candidates)
        all_source_gt_ids = list(source_gt_ids)
        query_existence1 = existence.cpu().numpy().astype(np.float32)
        query_existence2 = np.zeros_like(query_existence1, dtype=np.float32)

        if args.export_second_view_support and "gt_plane2" in batch:
            point_map2 = point_map_from_result(result2)[0]
            target_hw2 = point_map2.shape[:2]
            valid2 = finite_points(point_map2)
            prediction2, confidence2, margin2, existence2 = predict_stage1(
                head,
                saved_args,
                result2,
                view2["img"],
                target_hw2,
            )
            labels2 = resize_label(batch["gt_plane2"], target_hw2)[0]
            labels2 = torch.where((labels2 > 0) & (labels2 != 255), labels2, torch.full_like(labels2, -1))
            candidates2 = remap_and_fit_planes(prediction2, confidence2, margin2, point_map2, valid2, args)
            merge_records2 = []
            duplicate_merge_records2 = []
            if candidates2:
                if args.enable_safe_geometry_merge:
                    prediction2, merge_records2 = apply_safe_geometry_merges(
                        prediction2,
                        candidates2,
                        point_map2,
                        valid2,
                        view2["img"],
                        args,
                    )
                    candidates2 = remap_and_fit_planes(prediction2, confidence2, margin2, point_map2, valid2, args)
                if candidates2 and args.enable_duplicate_geometry_merge:
                    prediction2, duplicate_merge_records2 = apply_duplicate_geometry_merges(
                        prediction2,
                        candidates2,
                        point_map2,
                        valid2,
                        args,
                    )
                    candidates2 = remap_and_fit_planes(prediction2, confidence2, margin2, point_map2, valid2, args)
            if candidates2:
                plane_offset = len(all_candidates)
                remap2 = {row["query_id"]: plane_offset + i for i, row in enumerate(candidates2)}
                source_gt_ids2 = []
                for row in candidates2:
                    full_mask2 = (prediction2 == row["query_id"]) & valid2
                    gt_values2 = labels2[full_mask2]
                    gt_values2 = gt_values2[gt_values2 >= 0]
                    if len(gt_values2):
                        counts2 = torch.bincount(gt_values2)
                        source_gt_ids2.append(int(torch.argmax(counts2)))
                    else:
                        source_gt_ids2.append(-1)

                colors2 = image_to_uint8(batch["img2"], target_hw2)
                flat_points2 = point_map2.reshape(-1, 3)
                flat_colors2 = colors2.reshape(-1, 3)
                flat_valid2 = valid2.reshape(-1)
                flat_prediction2 = prediction2.reshape(-1)
                flat_confidence2 = confidence2.reshape(-1)
                flat_margin2 = margin2.reshape(-1)
                flat_labels2 = labels2.reshape(-1)
                flat_boundary2 = plane_boundary(prediction2).reshape(-1)
                yy2, xx2 = torch.meshgrid(
                    torch.linspace(-1.0, 1.0, target_hw2[0], device=device),
                    torch.linspace(-1.0, 1.0, target_hw2[1], device=device),
                    indexing="ij",
                )
                flat_pixel_xy2 = torch.stack((xx2, yy2), dim=-1).reshape(-1, 2)
                valid_indices2 = torch.nonzero(flat_valid2).flatten()
                if args.max_points > 0 and len(valid_indices2) > args.max_points:
                    sample_positions2 = torch.linspace(
                        0,
                        len(valid_indices2) - 1,
                        steps=args.max_points,
                        device=device,
                    ).long()
                    valid_indices2 = valid_indices2[sample_positions2]

                sampled_prediction2 = flat_prediction2[valid_indices2]
                point_plane_ids2 = torch.full_like(sampled_prediction2, -1, dtype=torch.int32)
                for query_id, output_id in remap2.items():
                    point_plane_ids2[sampled_prediction2 == int(query_id)] = int(output_id)

                sampled_pixel_xy2 = flat_pixel_xy2[valid_indices2].cpu().numpy().astype(np.float32)
                sampled_count2 = int(len(valid_indices2))
                combined_points = np.concatenate(
                    [combined_points, flat_points2[valid_indices2].cpu().numpy().astype(np.float32)],
                    axis=0,
                )
                combined_colors = np.concatenate(
                    [combined_colors, flat_colors2[valid_indices2].cpu().numpy().astype(np.uint8)],
                    axis=0,
                )
                combined_point_plane_ids = np.concatenate(
                    [combined_point_plane_ids, point_plane_ids2.cpu().numpy().astype(np.int32)],
                    axis=0,
                )
                combined_gt = np.concatenate(
                    [combined_gt, flat_labels2[valid_indices2].cpu().numpy().astype(np.int32)],
                    axis=0,
                )
                combined_line_prob = np.concatenate(
                    [combined_line_prob, flat_boundary2[valid_indices2].cpu().numpy().astype(np.float32)],
                    axis=0,
                )
                combined_confidence = np.concatenate(
                    [combined_confidence, flat_confidence2[valid_indices2].cpu().numpy().astype(np.float32)],
                    axis=0,
                )
                combined_margin = np.concatenate(
                    [combined_margin, flat_margin2[valid_indices2].cpu().numpy().astype(np.float32)],
                    axis=0,
                )
                combined_pixel_xy = np.concatenate([combined_pixel_xy, sampled_pixel_xy2], axis=0)
                pixel_xy1 = np.concatenate([pixel_xy1, np.zeros_like(sampled_pixel_xy2, dtype=np.float32)], axis=0)
                pixel_xy2 = np.concatenate([pixel_xy2, sampled_pixel_xy2], axis=0)
                support_source_view = np.concatenate(
                    [support_source_view, np.full((sampled_count2,), 2, dtype=np.int8)],
                    axis=0,
                )
                all_candidates.extend(candidates2)
                all_source_gt_ids.extend(source_gt_ids2)
                query_existence2 = existence2.cpu().numpy().astype(np.float32)
                duplicate_merge_records.extend(
                    [{**row, "view": 2} for row in duplicate_merge_records2]
                )
                merge_records.extend([{**row, "view": 2} for row in merge_records2])

        candidates = all_candidates
        source_gt_ids = all_source_gt_ids
        normals = torch.stack([row["normal"] for row in candidates]).cpu().numpy().astype(np.float32)
        offsets = np.asarray([float(row["offset"]) for row in candidates], dtype=np.float32)
        counts = np.asarray([int((combined_point_plane_ids == i).sum()) for i in range(len(candidates))], dtype=np.int32)
        ious, leakage = compute_teacher_iou(combined_point_plane_ids, combined_gt, source_gt_ids)
        sampled_count = int(len(combined_point_plane_ids))
        pixel_coordinate_view = "mixed_source_view" if np.any(support_source_view == 2) else "view1"
        np.savez_compressed(
            output_path,
            schema_version=np.asarray(STAGE2_SCHEMA_VERSION, dtype=np.int32),
            points=combined_points,
            colors=combined_colors,
            original_colors=combined_colors,
            scene_name=np.asarray(batch_string(batch, "scene_name")),
            pair_group=np.asarray(batch_string(batch, "pair_group")),
            rgb_path1=np.asarray(batch_string(batch, "rgb_path1")),
            rgb_path2=np.asarray(batch_string(batch, "rgb_path2")),
            json_path1=np.asarray(batch_string(batch, "json_path1")),
            json_path2=np.asarray(batch_string(batch, "json_path2")),
            view_id1=np.asarray(batch_string(batch, "view_id1")),
            view_id2=np.asarray(batch_string(batch, "view_id2")),
            original_hw1=batch_hw(batch, "original_hw1"),
            original_hw2=batch_hw(batch, "original_hw2"),
            stage1_input_hw1=batch_hw(batch, "stage1_input_hw1"),
            stage1_input_hw2=batch_hw(batch, "stage1_input_hw2"),
            stage1_mask_hw1=stage1_mask_hw,
            stage1_mask_hw2=stage1_mask_hw,
            pixel_coordinate_space=np.asarray("stage1_pointmap_normalized"),
            pixel_coordinate_order=np.asarray("xy"),
            pixel_coordinate_range=np.asarray("[-1,1]"),
            pixel_coordinate_view=np.asarray(pixel_coordinate_view),
            sample_idx=np.asarray(int(sample_idx), dtype=np.int32),
            pixel_xy=combined_pixel_xy,
            pixel_xy1=pixel_xy1,
            pixel_xy2=pixel_xy2,
            support_source_view=support_source_view,
            point_plane_ids=combined_point_plane_ids,
            gt_point_plane_ids=combined_gt,
            plane_normals=normals,
            plane_offsets=offsets,
            plane_inlier_counts=counts,
            plane_ids=np.arange(len(candidates), dtype=np.int32),
            source_gt_plane_ids=np.asarray(source_gt_ids, dtype=np.int32),
            line_prob=combined_line_prob,
            point_confidence=combined_confidence,
            point_margin=combined_margin,
            query_ids=np.asarray([row["query_id"] for row in candidates], dtype=np.int32),
            query_existence=query_existence1,
            query_existence2=query_existence2,
            full_mask_counts=np.asarray([row["full_count"] for row in candidates], dtype=np.int32),
            core_mask_counts=np.asarray([row["core_count"] for row in candidates], dtype=np.int32),
            fit_mask_counts=np.asarray([row["fit_count"] for row in candidates], dtype=np.int32),
        )
        row = {
            "sample_idx": int(sample_idx),
            "output": str(output_path),
            "points": int(len(combined_point_plane_ids)),
            "planes": len(candidates),
            "background_points": int((combined_point_plane_ids < 0).sum()),
            "stage1_checkpoint": str(args.stage1_checkpoint),
            "stage1_epoch": int(checkpoint.get("epoch", -1)),
            "safe_geometry_merge_enabled": bool(args.enable_safe_geometry_merge),
            "safe_geometry_merge_pairs": merge_records,
            "duplicate_geometry_merge_enabled": bool(args.enable_duplicate_geometry_merge),
            "duplicate_geometry_merge_pairs": duplicate_merge_records,
            "mean_teacher_iou": float(np.mean(ious)) if ious else 0.0,
            "leakage": float(leakage),
            "plane_fits": [
                {
                    "plane_id": i,
                    "query_id": int(candidate["query_id"]),
                    "source_gt_plane_id": int(source_gt_ids[i]),
                    "full_support_points": int(candidate["full_count"]),
                    "core_support_points": int(candidate["core_count"]),
                    "fit_support_points": int(candidate["fit_count"]),
                    "mean_residual": float(candidate["mean_residual"]),
                    "p95_residual": float(candidate["p95_residual"]),
                    "sampled_iou_to_major_gt": float(ious[i]) if i < len(ious) else 0.0,
                }
                for i, candidate in enumerate(candidates)
            ],
        }
        manifest.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)

    manifest_path = output_dir / "stage1_pred_support_teacher_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()

