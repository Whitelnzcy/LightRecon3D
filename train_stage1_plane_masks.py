import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataloaders.s3d_dataset import Structured3DDataset
from models.differentiable_plane_fit import (
    differentiable_weighted_plane_fit,
    point_to_plane_distance,
)
from models.plane_mask_head import PlaneMaskHead


SCALES = (32, 64, 128)


def parse_args():
    parser = argparse.ArgumentParser("Train coarse-to-fine Stage1 plane masks")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--input_mode", default="pair", choices=("pair", "single"))
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--small_train_size", type=int, default=128)
    parser.add_argument("--small_val_size", type=int, default=32)
    parser.add_argument("--hard_case_indices", default="")
    parser.add_argument("--hard_case_repeat", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_queries", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--feature_indices", type=int, nargs=4, default=(0, 6, 9, 12))
    parser.add_argument("--feature_dims", type=int, nargs=4, default=(1024, 768, 768, 768))
    parser.add_argument("--disable_rgb_edge", action="store_true")
    parser.add_argument("--refinement_margin", type=float, default=0.55)
    parser.add_argument("--resume_joint", action="store_true")
    parser.add_argument("--eval_before_train", action="store_true")
    parser.add_argument("--refine_lr", type=float, default=2e-4)
    parser.add_argument("--coarse_lr", type=float, default=2e-5)
    parser.add_argument("--refine_only_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--min_plane_pixels", type=int, default=4)
    parser.add_argument("--match_bce_weight", type=float, default=1.0)
    parser.add_argument("--match_dice_weight", type=float, default=2.0)
    parser.add_argument("--scale_weights", type=float, nargs=3, default=(1.0, 0.7, 1.0))
    parser.add_argument("--mask_focal_weight", type=float, default=1.0)
    parser.add_argument("--mask_tversky_weight", type=float, default=2.0)
    parser.add_argument("--partition_weight", type=float, default=1.0)
    parser.add_argument("--existence_weight", type=float, default=1.0)
    parser.add_argument("--small_existence_max_weight", type=float, default=3.0)
    parser.add_argument("--plane_count_recall_weight", type=float, default=0.25)
    parser.add_argument("--plane_count_over_weight", type=float, default=0.15)
    parser.add_argument("--aux_scale_match_weight", type=float, default=0.25)
    parser.add_argument("--aux_existence_target", type=float, default=0.7)
    parser.add_argument("--aux_existence_min_iou", type=float, default=0.05)
    parser.add_argument("--boundary_loss_weight", type=float, default=1.0)
    parser.add_argument("--boundary_head_weight", type=float, default=0.5)
    parser.add_argument("--structural_boundary_head_weight", type=float, default=0.5)
    parser.add_argument("--boundary_head_pos_weight_max", type=float, default=12.0)
    parser.add_argument("--decorative_line_weight", type=float, default=0.05)
    parser.add_argument("--structural_line_negative_weight", type=float, default=4.0)
    parser.add_argument("--boundary_pair_weight", type=float, default=0.5)
    parser.add_argument("--boundary_pair_same_margin", type=float, default=0.05)
    parser.add_argument("--separation_weight", type=float, default=0.5)
    parser.add_argument("--smoothness_weight", type=float, default=0.05)
    parser.add_argument("--boundary_band", type=int, default=5)
    parser.add_argument("--separation_margin", type=float, default=1.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--tversky_fp_weight", type=float, default=0.7)
    parser.add_argument("--tversky_fn_weight", type=float, default=0.3)
    parser.add_argument("--small_plane_max_weight", type=float, default=3.0)
    parser.add_argument("--existence_threshold", type=float, default=0.5)
    parser.add_argument("--full_mask_threshold", type=float, default=0.5)
    parser.add_argument("--core_mask_threshold", type=float, default=0.75)
    parser.add_argument("--core_margin_threshold", type=float, default=0.35)
    parser.add_argument("--plane_fit_warmup_epochs", type=int, default=1)
    parser.add_argument("--plane_normal_weight", type=float, default=0.1)
    parser.add_argument("--plane_offset_weight", type=float, default=0.1)
    parser.add_argument("--plane_residual_weight", type=float, default=0.05)
    parser.add_argument("--plane_fit_min_mass", type=float, default=32.0)
    parser.add_argument("--plane_fit_trim_quantile", type=float, default=0.85)
    parser.add_argument("--plane_fit_cov_jitter", type=float, default=1e-5)
    parser.add_argument("--plane_fit_temperature", type=float, default=0.25)
    parser.add_argument("--log_every", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260612)
    # Legacy flags remain accepted so older launch scripts fail gracefully into
    # the equivalent new loss terms.
    parser.add_argument("--lr", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mask_bce_weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mask_dice_weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--boundary_weight", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.lr is not None:
        args.refine_lr = args.lr
        args.coarse_lr = args.lr
    if args.mask_bce_weight is not None:
        args.mask_focal_weight = args.mask_bce_weight
    if args.mask_dice_weight is not None:
        args.mask_tversky_weight = args.mask_dice_weight
    if args.boundary_weight is not None:
        args.boundary_loss_weight = args.boundary_weight
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _read_hard_indices(path):
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("indices", payload.get("hard_cases", []))
    return [int(value) for value in payload]


def make_loader(args, split):
    dataset_kwargs = {
        "root_dir": args.root_dir,
        "split": split,
        "train_ratio": args.train_ratio,
        "image_size": (args.image_size, args.image_size),
    }
    try:
        dataset = Structured3DDataset(input_mode=args.input_mode, **dataset_kwargs)
    except TypeError as error:
        if "input_mode" not in str(error):
            raise
        dataset = Structured3DDataset(**dataset_kwargs)
        dataset.lightrecon_input_mode = "single"

    limit = args.small_train_size if split == "train" else args.small_val_size
    count = min(limit, len(dataset)) if limit > 0 else len(dataset)
    indices = list(range(count))
    if split == "train" and args.hard_case_indices:
        hard_indices = [
            index for index in _read_hard_indices(args.hard_case_indices)
            if 0 <= index < count
        ]
        indices.extend(hard_indices * max(args.hard_case_repeat, 0))
    dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=True,
    )


def build_views(batch, prefix):
    image1 = batch.get("img1", batch["img"])
    image2 = batch.get("img2", batch["img"])
    batch_size = image1.shape[0]
    shape1 = torch.tensor(image1.shape[-2:], device=image1.device)[None].repeat(batch_size, 1)
    shape2 = torch.tensor(image2.shape[-2:], device=image2.device)[None].repeat(batch_size, 1)
    return (
        {"img": image1, "true_shape": shape1, "instance": [f"{prefix}_{i}_1" for i in range(batch_size)]},
        {"img": image2, "true_shape": shape2, "instance": [f"{prefix}_{i}_2" for i in range(batch_size)]},
    )


def _tokens_to_map(features, image, patch_size=16, name="features"):
    if features.ndim != 3:
        raise ValueError(f"{name} must be [B,S,C], got {tuple(features.shape)}")
    batch_size, tokens, channels = features.shape
    height = image.shape[-2] // patch_size
    width = image.shape[-1] // patch_size
    if tokens != height * width:
        raise ValueError(f"{name} token count {tokens} does not match {height}x{width}")
    return features.transpose(1, 2).reshape(batch_size, channels, height, width)


def feature_maps_from_result(result, image, feature_indices=(0, 6, 9, 12), patch_size=16):
    all_features = result.get("dec_features_all")
    if all_features is None:
        raise KeyError(
            "DUSt3R result lacks dec_features_all. Use the LightRecon3D "
            "backbone with multi-layer feature export enabled."
        )
    if len(feature_indices) != 4:
        raise ValueError("feature_indices must contain four decoder stages")
    resolved = []
    for index in feature_indices:
        actual = index if index >= 0 else len(all_features) + index
        if actual < 0 or actual >= len(all_features):
            raise IndexError(
                f"Feature index {index} is invalid for {len(all_features)} decoder outputs"
            )
        resolved.append(
            _tokens_to_map(
                all_features[actual],
                image,
                patch_size=patch_size,
                name=f"dec_features_all[{actual}]",
            )
        )
    spatial_shapes = {tuple(feature.shape[-2:]) for feature in resolved}
    if len(spatial_shapes) != 1:
        raise ValueError(f"Decoder feature maps must share one token grid, got {spatial_shapes}")
    return {
        "encoder": resolved[0],
        "shallow": resolved[1],
        "middle": resolved[2],
        "deep": resolved[3],
    }


def feature_map_from_result(result, image, patch_size=16):
    """Backward-compatible last-layer feature accessor."""
    features = result["dec_features"]
    if isinstance(features, (list, tuple)):
        features = features[-1]
    return _tokens_to_map(features, image, patch_size=patch_size)


def point_map_from_result(result):
    for key in ("pts3d", "pts3d_in_other_view", "pointmap", "pred_pts3d"):
        if key in result:
            points = result[key]
            break
    else:
        raise KeyError(f"Cannot find point map in result keys: {list(result.keys())}")
    if points.ndim != 4:
        raise ValueError(f"Point map must be rank 4, got {tuple(points.shape)}")
    if points.shape[-1] == 3:
        return points
    if points.shape[1] == 3:
        return points.permute(0, 2, 3, 1).contiguous()
    raise ValueError(f"Cannot interpret point-map shape {tuple(points.shape)}")


def resize_point_map(point_map, target_hw):
    return F.interpolate(
        point_map.permute(0, 3, 1, 2),
        size=target_hw,
        mode="nearest",
    ).permute(0, 2, 3, 1).contiguous()


def _resize_labels(labels, target_hw):
    return F.interpolate(
        labels[:, None].float(),
        size=target_hw,
        mode="nearest",
    )[:, 0].long()


def _resize_lines(lines, target_hw):
    return F.interpolate(
        lines.float(),
        size=target_hw,
        mode="nearest",
    )[:, 0].float()


def select_plane_ids(labels, target_hw, max_planes, min_pixels):
    coarse_labels = _resize_labels(labels, target_hw)
    batch_ids = []
    for label_map in coarse_labels:
        candidates = []
        for plane_id_tensor in torch.unique(label_map):
            plane_id = int(plane_id_tensor)
            if plane_id <= 0 or plane_id == 255:
                continue
            count = int((label_map == plane_id).sum())
            if count >= min_pixels:
                candidates.append((count, plane_id))
        candidates.sort(key=lambda item: item[0], reverse=True)
        batch_ids.append([plane_id for _, plane_id in candidates[:max_planes]])
    return coarse_labels, batch_ids


def masks_for_plane_ids(labels, target_hw, batch_plane_ids):
    resized = _resize_labels(labels, target_hw)
    masks = []
    for label_map, plane_ids in zip(resized, batch_plane_ids):
        masks.append([(label_map == plane_id).float() for plane_id in plane_ids])
    return resized, masks


def instance_masks(labels, target_hw, max_planes, min_pixels):
    resized, plane_ids = select_plane_ids(labels, target_hw, max_planes, min_pixels)
    _, masks = masks_for_plane_ids(labels, target_hw, plane_ids)
    return resized, masks


def dice_cost(pred_prob, targets, eps=1e-6):
    intersection = torch.einsum("qhw,thw->qt", pred_prob, targets)
    denominator = pred_prob.sum(dim=(1, 2))[:, None] + targets.sum(dim=(1, 2))[None]
    return 1.0 - (2.0 * intersection + eps) / (denominator + eps)


def match_queries(mask_logits, targets, args):
    if targets.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    pred_prob = mask_logits.sigmoid()
    query_count = mask_logits.shape[0]
    target_count = targets.shape[0]
    logits_flat = mask_logits.flatten(1)[:, None].expand(query_count, target_count, -1)
    targets_flat = targets.flatten(1)[None].expand(query_count, target_count, -1)
    bce = F.binary_cross_entropy_with_logits(
        logits_flat,
        targets_flat,
        reduction="none",
    ).mean(dim=-1)
    cost = args.match_bce_weight * bce + args.match_dice_weight * dice_cost(pred_prob, targets)
    cost_np = cost.detach().cpu().numpy()
    states = {0: (0.0, ())}
    for target_id in range(target_count):
        next_states = {}
        for used_mask, (running_cost, assignment) in states.items():
            for query_id in range(query_count):
                bit = 1 << query_id
                if used_mask & bit:
                    continue
                new_mask = used_mask | bit
                new_cost = running_cost + float(cost_np[query_id, target_id])
                previous = next_states.get(new_mask)
                if previous is None or new_cost < previous[0]:
                    next_states[new_mask] = (new_cost, assignment + (query_id,))
        states = next_states
    _, best_assignment = min(states.values(), key=lambda item: item[0])
    return np.asarray(best_assignment, dtype=np.int64), np.arange(target_count, dtype=np.int64)


def boundary_map(label_map):
    boundary = torch.zeros_like(label_map, dtype=torch.bool)
    horizontal = label_map[:, 1:] != label_map[:, :-1]
    vertical = label_map[1:, :] != label_map[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def typed_boundary_maps(label_map):
    plane_plane = torch.zeros_like(label_map, dtype=torch.bool)
    plane_background = torch.zeros_like(label_map, dtype=torch.bool)

    def add_pair(left, right, left_slice, right_slice):
        different = left != right
        left_plane = (left > 0) & (left != 255)
        right_plane = (right > 0) & (right != 255)
        pp = different & left_plane & right_plane
        pb = different & (left_plane ^ right_plane)
        plane_plane[left_slice] |= pp
        plane_plane[right_slice] |= pp
        plane_background[left_slice] |= pb
        plane_background[right_slice] |= pb

    add_pair(
        label_map[:, :-1],
        label_map[:, 1:],
        (slice(None), slice(None, -1)),
        (slice(None), slice(1, None)),
    )
    add_pair(
        label_map[:-1, :],
        label_map[1:, :],
        (slice(None, -1), slice(None)),
        (slice(1, None), slice(None)),
    )
    return plane_plane, plane_background


def dilate_mask(mask, band_size):
    radius = max(int(band_size) // 2, 0)
    if radius == 0:
        return mask
    kernel = radius * 2 + 1
    return F.max_pool2d(
        mask[None, None].float(),
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )[0, 0] > 0


def boundary_target_from_labels_and_lines(labels, lines, band_size):
    target, _ = boundary_supervision_from_labels_and_lines(
        labels,
        lines,
        band_size,
        decorative_line_weight=0.0,
    )
    return target


def structural_boundary_target_from_labels(labels, band_size):
    pp_boundary, _ = typed_boundary_maps(labels)
    target = pp_boundary
    if band_size > 1:
        target = dilate_mask(target, band_size)
    return target.float()


def structural_boundary_supervision_from_labels_and_lines(
    labels,
    lines,
    band_size,
    hard_negative_weight=4.0,
):
    pp_boundary, _ = typed_boundary_maps(labels)
    target = pp_boundary
    if band_size > 1:
        target = dilate_mask(target, band_size)

    line_boundary = lines > 0.5
    nearby_structural = dilate_mask(pp_boundary, max(int(band_size) * 2, 1))
    line_hard_negative = line_boundary & ~nearby_structural
    weight = torch.ones_like(target, dtype=torch.float32)
    weight[line_hard_negative] = float(hard_negative_weight)
    return target.float(), weight


def boundary_supervision_from_labels_and_lines(
    labels,
    lines,
    band_size,
    decorative_line_weight=0.05,
):
    pp_boundary, pb_boundary = typed_boundary_maps(labels)
    plane_boundary = pp_boundary | pb_boundary
    line_boundary = lines > 0.5
    structural_band = dilate_mask(plane_boundary, max(int(band_size), 1))
    structural_line = line_boundary & structural_band
    decorative_line = line_boundary & ~structural_band
    target = plane_boundary | structural_line
    if band_size > 1:
        target = dilate_mask(target, band_size)
    weight = torch.ones_like(target, dtype=torch.float32)
    if decorative_line_weight > 0:
        decorative_target = decorative_line
        if band_size > 1:
            decorative_target = dilate_mask(decorative_target, band_size)
        target = target | decorative_target
        decorative_only = decorative_target & ~dilate_mask(plane_boundary | structural_line, band_size)
        weight[decorative_only] = float(decorative_line_weight)
    return target.float(), weight


def boundary_head_loss(logits, target, args, weight=None):
    target = target.to(dtype=logits.dtype, device=logits.device)
    if weight is not None:
        weight = weight.to(dtype=logits.dtype, device=logits.device)
    positive = target.sum()
    negative = target.numel() - positive
    pos_weight = (
        negative / positive.clamp_min(1.0)
    ).clamp(max=float(args.boundary_head_pos_weight_max))
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        weight=weight,
    )
    probability = logits.sigmoid()
    prediction = probability > 0.5
    target_bool = target > 0.5
    true_positive = (prediction & target_bool).sum().float()
    precision = true_positive / prediction.sum().clamp_min(1).float()
    recall = true_positive / target_bool.sum().clamp_min(1).float()
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)
    return bce, {
        "boundary_head_precision": precision,
        "boundary_head_recall": recall,
        "boundary_head_f1": f1,
    }


def _stack_masks(mask_list, reference):
    if mask_list:
        return torch.stack(mask_list)
    return reference.new_zeros((0, *reference.shape[-2:]))


def _area_weights(targets, max_weight):
    if targets.shape[0] == 0:
        return targets.new_zeros((0,))
    areas = targets.sum(dim=(1, 2)).clamp_min(1.0)
    weights = torch.sqrt(areas.max() / areas).clamp(max=max_weight)
    return weights / weights.mean().clamp_min(1e-6)


def focal_bce_loss(logits, targets, gamma):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = logits.sigmoid()
    pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    return ((1.0 - pt).pow(gamma) * bce).mean(dim=(1, 2))


def tversky_loss(logits, targets, fp_weight, fn_weight):
    probabilities = logits.sigmoid()
    true_positive = (probabilities * targets).sum(dim=(1, 2))
    false_positive = (probabilities * (1.0 - targets)).sum(dim=(1, 2))
    false_negative = ((1.0 - probabilities) * targets).sum(dim=(1, 2))
    score = (true_positive + 1e-6) / (
        true_positive
        + fp_weight * false_positive
        + fn_weight * false_negative
        + 1e-6
    )
    return 1.0 - score


def make_class_target(labels, targets, query_ids, target_ids, num_queries):
    class_target = torch.full(
        labels.shape,
        num_queries,
        device=labels.device,
        dtype=torch.long,
    )
    for query_id, target_id in zip(query_ids, target_ids):
        class_target[targets[target_id] > 0.5] = int(query_id)
    return class_target


def interior_smoothness(class_logits, class_target, boundary_probability=None):
    probabilities = class_logits.softmax(dim=0)
    total = class_logits.sum() * 0.0
    count = 0
    same_h = class_target[:, 1:] == class_target[:, :-1]
    same_v = class_target[1:, :] == class_target[:-1, :]
    if same_h.any():
        diff_h = (probabilities[:, :, 1:] - probabilities[:, :, :-1]).abs().mean(dim=0)
        if boundary_probability is not None:
            boundary_h = torch.maximum(
                boundary_probability[:, 1:],
                boundary_probability[:, :-1],
            ).detach()
            weight_h = (1.0 - boundary_h).clamp_min(0.05)
            total = total + (diff_h[same_h] * weight_h[same_h]).sum() / (
                weight_h[same_h].sum().clamp_min(1e-6)
            )
        else:
            total = total + diff_h[same_h].mean()
        count += 1
    if same_v.any():
        diff_v = (probabilities[:, 1:, :] - probabilities[:, :-1, :]).abs().mean(dim=0)
        if boundary_probability is not None:
            boundary_v = torch.maximum(
                boundary_probability[1:, :],
                boundary_probability[:-1, :],
            ).detach()
            weight_v = (1.0 - boundary_v).clamp_min(0.05)
            total = total + (diff_v[same_v] * weight_v[same_v]).sum() / (
                weight_v[same_v].sum().clamp_min(1e-6)
            )
        else:
            total = total + diff_v[same_v].mean()
        count += 1
    return total / max(count, 1)


def boundary_pair_separation(
    class_logits,
    class_target,
    num_queries,
    margin,
    boundary_probability=None,
):
    probabilities = class_logits.softmax(dim=0)[:num_queries]
    total = class_logits.sum() * 0.0
    count = 0

    def pair_loss(left_target, right_target, left_prob, right_prob, pair_boundary):
        valid = (
            (left_target < num_queries)
            & (right_target < num_queries)
            & (left_target != right_target)
        )
        if not valid.any():
            return None
        same_query_probability = (left_prob[:, valid] * right_prob[:, valid]).sum(dim=0)
        raw = F.relu(same_query_probability - float(margin))
        if pair_boundary is None:
            return raw.mean()
        weight = (1.0 + pair_boundary[valid].detach()).clamp_min(1.0)
        return (raw * weight).sum() / weight.sum().clamp_min(1e-6)

    boundary_h = None
    boundary_v = None
    if boundary_probability is not None:
        boundary_h = torch.maximum(
            boundary_probability[:, :-1],
            boundary_probability[:, 1:],
        )
        boundary_v = torch.maximum(
            boundary_probability[:-1, :],
            boundary_probability[1:, :],
        )

    horizontal = pair_loss(
        class_target[:, :-1],
        class_target[:, 1:],
        probabilities[:, :, :-1],
        probabilities[:, :, 1:],
        boundary_h,
    )
    if horizontal is not None:
        total = total + horizontal
        count += 1
    vertical = pair_loss(
        class_target[:-1, :],
        class_target[1:, :],
        probabilities[:, :-1, :],
        probabilities[:, 1:, :],
        boundary_v,
    )
    if vertical is not None:
        total = total + vertical
        count += 1
    return total / max(count, 1)


def scale_loss(
    mask_logits,
    background_logits,
    boundary_logits,
    structural_boundary_logits,
    labels,
    targets,
    query_ids,
    target_ids,
    args,
):
    zero = mask_logits.sum() * 0.0
    query_ids_t = torch.as_tensor(query_ids, device=mask_logits.device, dtype=torch.long)
    target_ids_t = torch.as_tensor(target_ids, device=mask_logits.device, dtype=torch.long)
    if len(query_ids):
        matched_logits = mask_logits[query_ids_t]
        matched_targets = targets[target_ids_t]
        area_weights = _area_weights(matched_targets, args.small_plane_max_weight)
        focal = (
            focal_bce_loss(matched_logits, matched_targets, args.focal_gamma)
            * area_weights
        ).mean()
        tversky = (
            tversky_loss(
                matched_logits,
                matched_targets,
                args.tversky_fp_weight,
                args.tversky_fn_weight,
            )
            * area_weights
        ).mean()
    else:
        focal = zero
        tversky = zero

    class_logits = torch.cat((mask_logits, background_logits), dim=0)
    class_target = make_class_target(
        labels,
        targets,
        query_ids,
        target_ids,
        args.num_queries,
    )
    partition = F.cross_entropy(class_logits[None], class_target[None])

    pp_boundary, pb_boundary = typed_boundary_maps(labels)
    pp_band = dilate_mask(pp_boundary, args.boundary_band)
    pb_band = dilate_mask(pb_boundary, args.boundary_band)
    boundary_pixels = pp_band | pb_band
    pixel_ce = F.cross_entropy(
        class_logits[None],
        class_target[None],
        reduction="none",
    )[0]
    boundary_loss = pixel_ce[boundary_pixels].mean() if boundary_pixels.any() else zero

    valid_pp = pp_band & (class_target < args.num_queries)
    if valid_pp.any():
        correct = class_logits.gather(0, class_target.clamp_max(args.num_queries)[None])[0]
        query_logits = mask_logits.clone()
        one_hot = F.one_hot(
            class_target.clamp_max(args.num_queries - 1),
            num_classes=args.num_queries,
        ).permute(2, 0, 1).bool()
        query_logits = query_logits.masked_fill(one_hot, -1e4)
        strongest_wrong = query_logits.max(dim=0).values
        separation = F.relu(
            args.separation_margin - correct[valid_pp] + strongest_wrong[valid_pp]
        ).mean()
    else:
        separation = zero
    if structural_boundary_logits is not None:
        boundary_probability = structural_boundary_logits.sigmoid()
    elif boundary_logits is not None:
        boundary_probability = boundary_logits.sigmoid()
    else:
        boundary_probability = None
    smoothness = interior_smoothness(
        class_logits,
        class_target,
        boundary_probability=boundary_probability,
    )
    boundary_pair = boundary_pair_separation(
        class_logits,
        class_target,
        args.num_queries,
        args.boundary_pair_same_margin,
        boundary_probability=boundary_probability,
    )

    loss = (
        args.mask_focal_weight * focal
        + args.mask_tversky_weight * tversky
        + args.partition_weight * partition
        + args.boundary_loss_weight * boundary_loss
        + args.boundary_pair_weight * boundary_pair
        + args.separation_weight * separation
        + args.smoothness_weight * smoothness
    )
    predicted = class_logits.argmax(dim=0)
    valid = class_target < args.num_queries
    accuracy = (
        (predicted[valid] == class_target[valid]).float().mean()
        if valid.any()
        else zero
    )
    pp_valid = pp_band & valid
    pb_valid = pb_band & valid
    pp_accuracy = (
        (predicted[pp_valid] == class_target[pp_valid]).float().mean()
        if pp_valid.any()
        else zero
    )
    pb_accuracy = (
        (predicted[pb_valid] == class_target[pb_valid]).float().mean()
        if pb_valid.any()
        else zero
    )
    matched_ious = []
    for query_id, target_id in zip(query_ids, target_ids):
        pred_mask = predicted == int(query_id)
        target_mask = targets[target_id] > 0.5
        union = (pred_mask | target_mask).sum()
        if union > 0:
            matched_ious.append((pred_mask & target_mask).sum().float() / union)
    mean_iou = torch.stack(matched_ious).mean() if matched_ious else zero
    return loss, {
        "focal": focal,
        "tversky": tversky,
        "partition": partition,
        "boundary_loss": boundary_loss,
        "boundary_pair": boundary_pair,
        "separation": separation,
        "smoothness": smoothness,
        "mask_accuracy": accuracy,
        "plane_plane_boundary_accuracy": pp_accuracy,
        "plane_background_boundary_accuracy": pb_accuracy,
        "mean_iou": mean_iou,
    }


def auxiliary_scale_matching_loss(mask_logits, targets, args):
    zero = mask_logits.sum() * 0.0
    if targets.shape[0] == 0:
        return zero, {"aux_match_iou": zero}
    query_ids, target_ids = match_queries(mask_logits, targets, args)
    if len(query_ids) == 0:
        return zero, {"aux_match_iou": zero}
    query_ids_t = torch.as_tensor(query_ids, device=mask_logits.device, dtype=torch.long)
    target_ids_t = torch.as_tensor(target_ids, device=mask_logits.device, dtype=torch.long)
    matched_logits = mask_logits[query_ids_t]
    matched_targets = targets[target_ids_t]
    area_weights = _area_weights(matched_targets, args.small_plane_max_weight)
    focal = (
        focal_bce_loss(matched_logits, matched_targets, args.focal_gamma)
        * area_weights
    ).mean()
    tversky = (
        tversky_loss(
            matched_logits,
            matched_targets,
            args.tversky_fp_weight,
            args.tversky_fn_weight,
        )
        * area_weights
    ).mean()
    predicted = mask_logits.argmax(dim=0)
    ious = []
    for query_id, target_id in zip(query_ids, target_ids):
        pred_mask = predicted == int(query_id)
        target_mask = targets[target_id] > 0.5
        union = (pred_mask | target_mask).sum()
        if union > 0:
            ious.append((pred_mask & target_mask).sum().float() / union)
    mean_iou = torch.stack(ious).mean() if ious else zero
    return focal + tversky, {"aux_match_iou": mean_iou}


def matched_query_ious(mask_logits, targets, query_ids, target_ids):
    if len(query_ids) == 0:
        return []
    predicted = mask_logits.argmax(dim=0)
    ious = []
    for query_id, target_id in zip(query_ids, target_ids):
        pred_mask = predicted == int(query_id)
        target_mask = targets[target_id] > 0.5
        union = (pred_mask | target_mask).sum()
        if union > 0:
            iou = (pred_mask & target_mask).sum().float() / union
        else:
            iou = mask_logits.sum() * 0.0
        ious.append(iou)
    return ious


def prediction_masks(
    mask_logits,
    background_logits,
    existence_logits,
    existence_threshold=0.5,
    full_threshold=0.5,
    core_threshold=0.75,
    core_margin_threshold=0.35,
):
    probabilities = mask_logits.sigmoid()
    class_probabilities = torch.cat(
        (probabilities, background_logits.sigmoid()),
        dim=0,
    )
    active = existence_logits.sigmoid() > existence_threshold
    inactive = ~active
    class_probabilities[:-1][inactive] = 0.0
    top2 = class_probabilities.topk(k=2, dim=0)
    predicted = top2.indices[0]
    confidence = top2.values[0]
    margin = top2.values[0] - top2.values[1]
    predicted[predicted == mask_logits.shape[0]] = -1
    full_masks = []
    core_masks = []
    for query_id in range(mask_logits.shape[0]):
        assigned = predicted == query_id
        full_masks.append(assigned & (probabilities[query_id] >= full_threshold))
        core_masks.append(
            assigned
            & (probabilities[query_id] >= core_threshold)
            & (margin >= core_margin_threshold)
        )
    return {
        "assignment": predicted,
        "full_masks": torch.stack(full_masks),
        "core_masks": torch.stack(core_masks),
        "confidence": confidence,
        "margin": margin,
        "active": active,
    }


def plane_fit_supervision(
    mask_logits,
    background_logits,
    point_map,
    targets,
    query_ids,
    target_ids,
    args,
):
    zero = mask_logits.sum() * 0.0
    empty_stats = {
        "plane_fit_valid": 0.0,
        "plane_normal_angle_deg": 0.0,
        "plane_offset_abs_error": 0.0,
        "plane_pred_residual": 0.0,
        "plane_teacher_residual": 0.0,
        "plane_fit_mass": 0.0,
        "plane_fit_eigengap": 0.0,
    }
    if len(query_ids) == 0:
        return zero, empty_stats

    query_ids_t = torch.as_tensor(query_ids, device=mask_logits.device, dtype=torch.long)
    target_ids_t = torch.as_tensor(target_ids, device=mask_logits.device, dtype=torch.long)
    temperature = max(float(args.plane_fit_temperature), 1e-3)
    class_probabilities = (
        torch.cat((mask_logits, background_logits), dim=0) / temperature
    ).softmax(dim=0)
    pred_weights = class_probabilities[query_ids_t].flatten(1).transpose(0, 1)
    teacher_weights = targets[target_ids_t].flatten(1).transpose(0, 1)
    points = point_map.reshape(-1, 3).float()
    valid_points = torch.isfinite(points).all(dim=-1) & (points.abs().amax(dim=-1) < 1e4)

    with torch.no_grad():
        initial_normals, initial_offsets, _, _, initial_mass = (
            differentiable_weighted_plane_fit(
                points,
                teacher_weights,
                valid_mask=valid_points,
                cov_jitter=args.plane_fit_cov_jitter,
            )
        )
        initial_distances = point_to_plane_distance(
            torch.where(valid_points[:, None], points, torch.zeros_like(points)),
            initial_normals,
            initial_offsets,
        )
        trimmed_teacher = teacher_weights.clone()
        trim_quantile = float(args.plane_fit_trim_quantile)
        if trim_quantile < 1.0:
            for plane_index in range(trimmed_teacher.shape[1]):
                selected = (teacher_weights[:, plane_index] > 0.5) & valid_points
                if int(selected.sum()) < 3:
                    trimmed_teacher[:, plane_index] = 0.0
                    continue
                threshold = torch.quantile(
                    initial_distances[selected, plane_index],
                    trim_quantile,
                )
                trimmed_teacher[:, plane_index] *= (
                    initial_distances[:, plane_index] <= threshold
                ).to(trimmed_teacher.dtype)
        teacher_normals, teacher_offsets, _, teacher_eigenvalues, teacher_mass = (
            differentiable_weighted_plane_fit(
                points,
                trimmed_teacher,
                valid_mask=valid_points,
                cov_jitter=args.plane_fit_cov_jitter,
            )
        )

    pred_normals, pred_offsets, _, pred_eigenvalues, pred_mass = (
        differentiable_weighted_plane_fit(
            points,
            pred_weights,
            valid_mask=valid_points,
            cov_jitter=args.plane_fit_cov_jitter,
        )
    )
    valid_planes = (
        (initial_mass >= args.plane_fit_min_mass)
        & (teacher_mass >= args.plane_fit_min_mass)
        & (pred_mass >= args.plane_fit_min_mass)
    )
    if not valid_planes.any():
        return zero, empty_stats

    dot = (pred_normals * teacher_normals).sum(dim=-1)
    normal_loss_per_plane = 1.0 - dot.abs().clamp(max=1.0)
    orientation = torch.where(dot.detach() < 0.0, -1.0, 1.0)
    aligned_offsets = pred_offsets * orientation
    offset_loss_per_plane = F.smooth_l1_loss(
        aligned_offsets,
        teacher_offsets,
        reduction="none",
        beta=0.02,
    )
    safe_points = torch.where(valid_points[:, None], points, torch.zeros_like(points))
    pred_distances = point_to_plane_distance(safe_points, pred_normals, pred_offsets)
    teacher_distances = point_to_plane_distance(
        safe_points,
        teacher_normals,
        teacher_offsets,
    )
    valid_float = valid_points[:, None].to(pred_weights.dtype)
    pred_residual_per_plane = (
        pred_distances * pred_weights * valid_float
    ).sum(dim=0) / pred_mass.clamp_min(1e-6)
    teacher_residual_per_plane = (
        teacher_distances * trimmed_teacher * valid_float
    ).sum(dim=0) / teacher_mass.clamp_min(1e-6)

    normal_loss = normal_loss_per_plane[valid_planes].mean()
    offset_loss = offset_loss_per_plane[valid_planes].mean()
    residual_loss = pred_residual_per_plane[valid_planes].mean()
    loss = (
        args.plane_normal_weight * normal_loss
        + args.plane_offset_weight * offset_loss
        + args.plane_residual_weight * residual_loss
    )
    normal_angle = torch.rad2deg(
        torch.acos(dot.abs().clamp(min=0.0, max=1.0))
    )
    eigengap = pred_eigenvalues[:, 1] - pred_eigenvalues[:, 0]
    stats = {
        "plane_fit_valid": float(valid_planes.sum().detach()),
        "plane_normal_angle_deg": float(normal_angle[valid_planes].mean().detach()),
        "plane_offset_abs_error": float(
            (aligned_offsets - teacher_offsets).abs()[valid_planes].mean().detach()
        ),
        "plane_pred_residual": float(
            pred_residual_per_plane[valid_planes].mean().detach()
        ),
        "plane_teacher_residual": float(
            teacher_residual_per_plane[valid_planes].mean().detach()
        ),
        "plane_fit_mass": float(pred_mass[valid_planes].mean().detach()),
        "plane_fit_eigengap": float(eigengap[valid_planes].mean().detach()),
    }
    return loss, stats


def sample_loss(
    output,
    raw_labels,
    raw_lines,
    plane_ids,
    point_map,
    args,
    plane_fit_scale=1.0,
):
    total = output["mask_logits"].sum() * 0.0
    stats = defaultdict(float)
    batch_size = output["mask_logits"].shape[0]
    scale_weights = dict(zip(SCALES, args.scale_weights))

    for batch_id in range(batch_size):
        labels_by_scale = {}
        lines_by_scale = {}
        targets_by_scale = {}
        for scale in SCALES:
            logits = output[f"mask_logits_{scale}"]
            labels, masks = masks_for_plane_ids(
                raw_labels[batch_id : batch_id + 1],
                logits.shape[-2:],
                [plane_ids[batch_id]],
            )
            labels_by_scale[scale] = labels[0]
            lines_by_scale[scale] = _resize_lines(
                raw_lines[batch_id : batch_id + 1],
                logits.shape[-2:],
            )[0]
            targets_by_scale[scale] = _stack_masks(masks[0], logits[batch_id])

        matches_by_scale = {}
        for scale in SCALES:
            matches_by_scale[scale] = match_queries(
                output[f"mask_logits_{scale}"][batch_id],
                targets_by_scale[scale],
                args,
            )
        query_ids, target_ids = matches_by_scale[32]
        existence_target = torch.zeros_like(output["existence_logits"][batch_id])
        existence_weight = torch.ones_like(output["existence_logits"][batch_id])
        if len(query_ids):
            query_ids_t = torch.as_tensor(query_ids, device=existence_target.device)
            target_ids_t = torch.as_tensor(target_ids, device=existence_target.device)
            existence_target[query_ids_t] = 1.0
            target_areas = targets_by_scale[32][target_ids_t].sum(dim=(1, 2)).clamp_min(1.0)
            target_weights = torch.sqrt(target_areas.max() / target_areas).clamp(
                max=float(args.small_existence_max_weight)
            )
            existence_weight[query_ids_t] = target_weights
        aux_existence_queries = 0
        for scale in (64, 128):
            aux_query_ids, aux_target_ids = matches_by_scale[scale]
            aux_ious = matched_query_ious(
                output[f"mask_logits_{scale}"][batch_id],
                targets_by_scale[scale],
                aux_query_ids,
                aux_target_ids,
            )
            if not aux_ious:
                continue
            aux_query_ids_t = torch.as_tensor(
                aux_query_ids,
                device=existence_target.device,
                dtype=torch.long,
            )
            aux_target_ids_t = torch.as_tensor(
                aux_target_ids,
                device=existence_target.device,
                dtype=torch.long,
            )
            aux_keep = torch.stack(aux_ious).detach() >= float(args.aux_existence_min_iou)
            if not bool(aux_keep.any()):
                continue
            kept_query_ids = aux_query_ids_t[aux_keep]
            kept_target_ids = aux_target_ids_t[aux_keep]
            old_targets = existence_target[kept_query_ids].clone()
            weak_target = existence_target.new_full(
                kept_query_ids.shape,
                float(args.aux_existence_target),
            )
            existence_target[kept_query_ids] = torch.maximum(
                existence_target[kept_query_ids],
                weak_target,
            )
            target_areas = targets_by_scale[scale][kept_target_ids].sum(dim=(1, 2)).clamp_min(1.0)
            target_weights = torch.sqrt(target_areas.max() / target_areas).clamp(
                max=float(args.small_existence_max_weight)
            )
            existence_weight[kept_query_ids] = torch.maximum(
                existence_weight[kept_query_ids],
                target_weights,
            )
            aux_existence_queries += int((existence_target[kept_query_ids] > old_targets).sum().item())
        existence_raw = F.binary_cross_entropy_with_logits(
            output["existence_logits"][batch_id],
            existence_target,
            reduction="none",
        )
        existence = (existence_raw * existence_weight).sum() / existence_weight.sum().clamp_min(1e-6)
        predicted_count_soft = output["existence_logits"][batch_id].sigmoid().sum()
        target_count = output["existence_logits"][batch_id].new_tensor(
            float(len(plane_ids[batch_id]))
        )
        count_under = F.relu(target_count - predicted_count_soft).pow(2)
        count_over = F.relu(predicted_count_soft - target_count).pow(2)
        count_recall = (
            args.plane_count_recall_weight * count_under
            + args.plane_count_over_weight * count_over
        )
        sample_total = args.existence_weight * existence
        sample_total = sample_total + count_recall
        stats["existence"] += float(existence.detach())
        stats["plane_count_loss"] += float(count_recall.detach())
        stats["pred_planes_soft"] += float(predicted_count_soft.detach())
        stats["aux_existence_queries"] += float(aux_existence_queries)

        for scale in SCALES:
            scale_value, scale_stats = scale_loss(
                output[f"mask_logits_{scale}"][batch_id],
                output[f"background_logits_{scale}"][batch_id],
                output[f"boundary_logits_{scale}"][batch_id, 0],
                output[f"structural_boundary_logits_{scale}"][batch_id, 0],
                labels_by_scale[scale],
                targets_by_scale[scale],
                query_ids,
                target_ids,
                args,
            )
            sample_total = sample_total + scale_weights[scale] * scale_value
            for key, value in scale_stats.items():
                stats[f"{key}_{scale}"] += float(value.detach())

            if scale in (64, 128) and args.aux_scale_match_weight > 0:
                aux_value, aux_stats = auxiliary_scale_matching_loss(
                    output[f"mask_logits_{scale}"][batch_id],
                    targets_by_scale[scale],
                    args,
                )
                sample_total = sample_total + args.aux_scale_match_weight * aux_value
                stats[f"aux_match_loss_{scale}"] += float(aux_value.detach())
                stats[f"aux_match_iou_{scale}"] += float(
                    aux_stats["aux_match_iou"].detach()
                )

            boundary_target, boundary_weight = boundary_supervision_from_labels_and_lines(
                labels_by_scale[scale],
                lines_by_scale[scale],
                args.boundary_band,
                decorative_line_weight=args.decorative_line_weight,
            )
            boundary_value, boundary_stats = boundary_head_loss(
                output[f"boundary_logits_{scale}"][batch_id, 0],
                boundary_target,
                args,
                weight=boundary_weight,
            )
            sample_total = sample_total + args.boundary_head_weight * boundary_value
            stats[f"boundary_head_loss_{scale}"] += float(boundary_value.detach())
            for key, value in boundary_stats.items():
                stats[f"{key}_{scale}"] += float(value.detach())

            structural_target, structural_weight = (
                structural_boundary_supervision_from_labels_and_lines(
                    labels_by_scale[scale],
                    lines_by_scale[scale],
                    args.boundary_band,
                    hard_negative_weight=args.structural_line_negative_weight,
                )
            )
            structural_value, structural_stats = boundary_head_loss(
                output[f"structural_boundary_logits_{scale}"][batch_id, 0],
                structural_target,
                args,
                weight=structural_weight,
            )
            sample_total = (
                sample_total
                + args.structural_boundary_head_weight * structural_value
            )
            stats[f"structural_boundary_head_loss_{scale}"] += float(
                structural_value.detach()
            )
            for key, value in structural_stats.items():
                stats[f"structural_{key}_{scale}"] += float(value.detach())

        fit_point_map = resize_point_map(
            point_map[batch_id : batch_id + 1],
            output["mask_logits_128"].shape[-2:],
        )[0]
        fit_loss, fit_stats = plane_fit_supervision(
            output["mask_logits_128"][batch_id],
            output["background_logits_128"][batch_id],
            fit_point_map,
            targets_by_scale[128],
            query_ids,
            target_ids,
            args,
        )
        sample_total = sample_total + float(plane_fit_scale) * fit_loss
        stats["plane_fit_loss"] += float(fit_loss.detach())
        stats["plane_fit_scale"] += float(plane_fit_scale)
        for key, value in fit_stats.items():
            stats[key] += value

        for scale in (64, 128):
            gate = output[f"refinement_gate{scale}"][batch_id, 0]
            boundary = dilate_mask(
                boundary_map(labels_by_scale[scale]),
                args.boundary_band,
            )
            interior = ~boundary
            stats[f"gate_boundary_{scale}"] += (
                float(gate[boundary].mean().detach()) if boundary.any() else 0.0
            )
            stats[f"gate_interior_{scale}"] += (
                float(gate[interior].mean().detach()) if interior.any() else 0.0
            )

        stats["gt_planes"] += float(len(plane_ids[batch_id]))
        stats["pred_planes"] += float(
            (output["existence_logits"][batch_id].sigmoid() > args.existence_threshold).sum()
        )
        total = total + sample_total

    stats["alpha64"] = float(output["refinement_alpha64"].detach())
    stats["alpha128"] = float(output["refinement_alpha128"].detach())
    return total / batch_size, {
        key: value / batch_size
        if key not in ("alpha64", "alpha128")
        else value
        for key, value in stats.items()
    }


def _grad_norm(parameters):
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().norm().pow(2))
    return squared ** 0.5


def set_training_phase(head, optimizer, epoch, args):
    refine_only = bool(args.init_checkpoint) and epoch <= args.refine_only_epochs
    for parameter in head.coarse_parameters():
        parameter.requires_grad = not refine_only
    optimizer.param_groups[0]["lr"] = 0.0 if refine_only else args.coarse_lr
    optimizer.param_groups[1]["lr"] = args.refine_lr
    return "refinement_only" if refine_only else "joint"


def run_epoch(backbone, head, loader, optimizer, device, args, train, epoch=0):
    head.train(train)
    if train and not any(parameter.requires_grad for parameter in head.coarse_parameters()):
        for module in head.coarse_modules():
            module.eval()
    backbone.eval()
    totals = defaultdict(float)
    steps = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        view1, view2 = build_views(batch, "stage1_multiscale")
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            result1, result2 = backbone(view1, view2)
            features1 = feature_maps_from_result(
                result1,
                view1["img"],
                args.feature_indices,
            )
            features2 = feature_maps_from_result(
                result2,
                view2["img"],
                args.feature_indices,
            )
            points1 = point_map_from_result(result1)
            points2 = point_map_from_result(result2)
        views = [
            (
                features1,
                view1["img"],
                batch.get("gt_plane1", batch["gt_plane"]),
                batch.get("gt_line1", batch["gt_line"]),
                points1,
            )
        ]
        if args.input_mode == "pair" and "gt_plane2" in batch:
            views.append((features2, view2["img"], batch["gt_plane2"], batch["gt_line2"], points2))

        losses = []
        rows = []
        plane_fit_scale = (
            0.0
            if train and epoch <= args.plane_fit_warmup_epochs
            else 1.0
        )
        for features, image, gt_plane, gt_line, point_map in views:
            output = head(
                features["deep"],
                middle_feature=features["middle"],
                shallow_feature=features["shallow"],
                encoder_feature=features["encoder"],
                image=image,
            )
            _, plane_ids = select_plane_ids(
                gt_plane,
                output["mask_logits_32"].shape[-2:],
                args.num_queries,
                args.min_plane_pixels,
            )
            loss, row = sample_loss(
                output,
                gt_plane,
                gt_line,
                plane_ids,
                point_map,
                args,
                plane_fit_scale=plane_fit_scale,
            )
            losses.append(loss)
            rows.append(row)
        loss = torch.stack(losses).mean()
        if train:
            loss.backward()
            coarse_grad = _grad_norm(head.coarse_parameters())
            refine_grad = _grad_norm(head.refinement_parameters())
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()
        else:
            coarse_grad = 0.0
            refine_grad = 0.0

        totals["loss"] += float(loss.detach())
        totals["coarse_grad_norm"] += coarse_grad
        totals["refine_grad_norm"] += refine_grad
        for row in rows:
            for key, value in row.items():
                totals[key] += value / len(rows)
        steps += 1
        if step == 1 or step % args.log_every == 0 or step == len(loader):
            print(
                f"[{'Train' if train else 'Val'}] {step}/{len(loader)} "
                f"loss={float(loss.detach()):.4f} "
                f"iou32={np.mean([row['mean_iou_32'] for row in rows]):.3f} "
                f"iou64={np.mean([row['mean_iou_64'] for row in rows]):.3f} "
                f"iou128={np.mean([row['mean_iou_128'] for row in rows]):.3f} "
                f"pp128={np.mean([row['plane_plane_boundary_accuracy_128'] for row in rows]):.3f} "
                f"bf128={np.mean([row['boundary_head_f1_128'] for row in rows]):.3f} "
                f"angle={np.mean([row['plane_normal_angle_deg'] for row in rows]):.2f} "
                f"offset={np.mean([row['plane_offset_abs_error'] for row in rows]):.4f} "
                f"alpha=({float(head.alpha64.detach()):.4f},"
                f"{float(head.alpha128.detach()):.4f})",
                flush=True,
            )
    return {key: value / max(steps, 1) for key, value in totals.items()}


def verify_zero_refinement(head, device, args):
    head.eval()
    batch_size = 1
    height = width = args.image_size // 16
    with torch.no_grad():
        output = head(
            torch.randn(batch_size, args.feature_dims[3], height, width, device=device),
            middle_feature=torch.randn(
                batch_size,
                args.feature_dims[2],
                height,
                width,
                device=device,
            ),
            shallow_feature=torch.randn(
                batch_size,
                args.feature_dims[1],
                height,
                width,
                device=device,
            ),
            encoder_feature=torch.randn(
                batch_size,
                args.feature_dims[0],
                height,
                width,
                device=device,
            ),
            image=torch.randn(
                batch_size,
                3,
                args.image_size,
                args.image_size,
                device=device,
            ),
        )
        expected64 = F.interpolate(
            torch.cat(
                (output["mask_logits_32"], output["background_logits_32"]),
                dim=1,
            ),
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        expected128 = F.interpolate(
            expected64,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        actual128 = torch.cat(
            (output["mask_logits_128"], output["background_logits_128"]),
            dim=1,
        )
        max_error = float((actual128 - expected128).abs().max())
    if max_error >= 1e-5:
        raise RuntimeError(f"Zero-refinement equivalence failed: max_error={max_error}")
    print(f"Zero-refinement equivalence passed: max_error={max_error:.3e}", flush=True)


def load_multiscale_state_dict_allow_boundary_init(head, state_dict):
    incompatible = head.load_state_dict(state_dict, strict=False)
    allowed_missing = (
        "boundary_head32.",
        "boundary_head64.",
        "boundary_head128.",
        "structural_boundary_head32.",
        "structural_boundary_head64.",
        "structural_boundary_head128.",
    )
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing)
    ]
    if unexpected or missing:
        raise RuntimeError(
            "Incompatible multiscale checkpoint: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return incompatible


def main():
    args = parse_args()
    from models.build_backbone import build_dust3r_backbone

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(args, "train")
    val_loader = make_loader(args, "val")
    backbone = build_dust3r_backbone(args.weights_path, device=device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    head = PlaneMaskHead(
        feature_dim=args.feature_dims[3],
        encoder_feature_dim=args.feature_dims[0],
        shallow_feature_dim=args.feature_dims[1],
        middle_feature_dim=args.feature_dims[2],
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.decoder_layers,
        num_heads=args.decoder_heads,
        use_rgb_edge=not args.disable_rgb_edge,
        refinement_margin=args.refinement_margin,
    ).to(device)
    verify_coarse_equivalence = True
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        source_state = checkpoint["head"]
        if "alpha64" in source_state and "alpha128" in source_state:
            incompatible = load_multiscale_state_dict_allow_boundary_init(
                head,
                source_state,
            )
            verify_coarse_equivalence = False
            if args.resume_joint:
                args.refine_only_epochs = 0
            print(f"Resumed multiscale mask head from {args.init_checkpoint}", flush=True)
            if incompatible.missing_keys:
                print(
                    "Initialized missing boundary head weights: "
                    f"{list(incompatible.missing_keys)}",
                    flush=True,
                )
        else:
            head.load_coarse_state_dict(source_state)
            print(f"Initialized coarse mask head from {args.init_checkpoint}", flush=True)
    else:
        args.refine_only_epochs = 0

    if verify_coarse_equivalence:
        verify_zero_refinement(head, device, args)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(head.coarse_parameters()), "lr": args.coarse_lr},
            {"params": list(head.refinement_parameters()), "lr": args.refine_lr},
        ],
        weight_decay=args.weight_decay,
    )

    history = []
    best_iou = -1.0
    best_geometry_error = float("inf")
    if args.eval_before_train:
        initial_val = run_epoch(
            backbone,
            head,
            val_loader,
            None,
            device,
            args,
            train=False,
            epoch=0,
        )
        initial_row = {
            "epoch": 0,
            "phase": "initial_eval",
            "val": initial_val,
        }
        history.append(initial_row)
        best_iou = initial_val["mean_iou_128"]
        print(json.dumps(initial_row), flush=True)
        torch.save(
            {
                "model_version": "stage1_plane_masks_multiscale_svd_v1",
                "head": head.state_dict(),
                "args": vars(args),
                "epoch": 0,
                "phase": "initial_eval",
                "val_stats": initial_val,
            },
            os.path.join(args.save_dir, "best.pt"),
        )
        if initial_val.get("plane_fit_valid", 0.0) > 0:
            best_geometry_error = (
                initial_val["plane_normal_angle_deg"]
                + 50.0 * initial_val["plane_offset_abs_error"]
                + 50.0 * initial_val["plane_pred_residual"]
            )
            torch.save(
                {
                    "model_version": "stage1_plane_masks_multiscale_svd_v1",
                    "head": head.state_dict(),
                    "args": vars(args),
                    "epoch": 0,
                    "phase": "initial_eval",
                    "val_stats": initial_val,
                    "geometry_error": best_geometry_error,
                },
                os.path.join(args.save_dir, "best_geometry.pt"),
            )
    for epoch in range(1, args.num_epochs + 1):
        phase = set_training_phase(head, optimizer, epoch, args)
        print(f"Epoch {epoch}: phase={phase}", flush=True)
        train_stats = run_epoch(
            backbone,
            head,
            train_loader,
            optimizer,
            device,
            args,
            train=True,
            epoch=epoch,
        )
        val_stats = run_epoch(
            backbone,
            head,
            val_loader,
            None,
            device,
            args,
            train=False,
            epoch=epoch,
        )
        row = {"epoch": epoch, "phase": phase, "train": train_stats, "val": val_stats}
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "model_version": "stage1_plane_masks_multiscale_svd_v1",
            "head": head.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "phase": phase,
            "train_stats": train_stats,
            "val_stats": val_stats,
        }
        torch.save(checkpoint, os.path.join(args.save_dir, "latest.pt"))
        if val_stats["mean_iou_128"] > best_iou:
            best_iou = val_stats["mean_iou_128"]
            torch.save(checkpoint, os.path.join(args.save_dir, "best.pt"))
        if val_stats.get("plane_fit_valid", 0.0) > 0:
            geometry_error = (
                val_stats["plane_normal_angle_deg"]
                + 50.0 * val_stats["plane_offset_abs_error"]
                + 50.0 * val_stats["plane_pred_residual"]
            )
            if geometry_error < best_geometry_error:
                best_geometry_error = geometry_error
                geometry_checkpoint = dict(checkpoint)
                geometry_checkpoint["geometry_error"] = geometry_error
                torch.save(
                    geometry_checkpoint,
                    os.path.join(args.save_dir, "best_geometry.pt"),
                )
        with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)


if __name__ == "__main__":
    main()
