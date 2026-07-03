import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")

if DUST3R_REPO_ROOT not in sys.path:
    sys.path.insert(0, DUST3R_REPO_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel
from models.multiscale_plane_mask_head import MultiScalePlaneMaskHead
from train_stage1_plane_masks import masks_for_plane_ids, match_queries, select_plane_ids


STAGE_NAMES = ("encoder", "shallow", "middle", "deep")
QUERY_COLORS = np.asarray(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 212],
        [0, 128, 128],
        [220, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
    ],
    dtype=np.float32,
) / 255.0

ERROR_COLORS = np.asarray(
    [
        [0.10, 0.10, 0.10],  # correct background
        [0.20, 0.80, 0.20],  # correct plane assignment
        [0.90, 0.15, 0.15],  # wrong plane
        [1.00, 0.80, 0.10],  # plane predicted as background
        [0.85, 0.15, 0.85],  # background predicted as plane
    ],
    dtype=np.float32,
)


def class_target_from_matches(labels, targets, query_ids, target_ids, num_queries):
    class_target = torch.full(
        labels.shape,
        num_queries,
        device=labels.device,
        dtype=torch.long,
    )
    for query_id, target_id in zip(query_ids, target_ids):
        class_target[targets[target_id] > 0.5] = int(query_id)
    return class_target


def apply_query_class_scores(mask_logits, existence_logits, args):
    weight = float(getattr(args, "class_score_weight", 0.0))
    if weight <= 0.0:
        return mask_logits
    return mask_logits + weight * existence_logits[:, None, None]


def parse_args():
    parser = argparse.ArgumentParser("Visualize LightRecon3D predictions")
    parser.add_argument(
        "--model_type",
        default="multiscale_mask",
        choices=("embedding", "multiscale_mask"),
    )

    # Legacy embedding-mode arguments.
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/data/zhucy23u/datasets/Structured3D",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="/data/zhucy23u/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
    )
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--input_mode", default="pair", choices=("pair", "single"))
    parser.add_argument("--pair_strategy", default="adjacent", choices=("adjacent", "all"))
    parser.add_argument("--pair_max_view_id_gap", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)
    parser.add_argument("--line_threshold", type=float, default=0.3)
    parser.add_argument("--allow_partial_load", action="store_true")
    parser.add_argument(
        "--save_path",
        type=str,
        default="/data/zhucy23u/logs/vis_plane_embedding.png",
    )

    # Multiscale mask-mode arguments.
    parser.add_argument("--feature_cache_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--sample_indices", type=str, default="")
    parser.add_argument(
        "--select_mode",
        default="all",
        choices=(
            "all",
            "manual",
            "best_iou",
            "worst_iou",
            "highest_leakage",
            "largest_view_gap",
            "random",
        ),
    )
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--random_seed", type=int, default=20260617)
    return parser.parse_args()


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_views_from_batch(batch, prefix="vis"):
    img1 = batch.get("img1", batch["img"])
    img2 = batch.get("img2", batch["img"])
    batch_size = img1.shape[0]
    true_shape1 = torch.tensor(
        img1.shape[-2:], device=img1.device, dtype=torch.long
    )[None].repeat(batch_size, 1)
    true_shape2 = torch.tensor(
        img2.shape[-2:], device=img2.device, dtype=torch.long
    )[None].repeat(batch_size, 1)
    return (
        {
            "img": img1,
            "true_shape": true_shape1,
            "instance": [f"{prefix}_{index}_view1" for index in range(batch_size)],
        },
        {
            "img": img2,
            "true_shape": true_shape2,
            "instance": [f"{prefix}_{index}_view2" for index in range(batch_size)],
        },
    )


def safe_load_checkpoint(path, device="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_state_dict_for_visualization(model, checkpoint, allow_partial=False):
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    if not allow_partial:
        model.load_state_dict(state_dict, strict=True)
        return [], [], []

    model_dict = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return missing, unexpected, skipped


def tensor_image_to_numpy(image_tensor):
    image = image_tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def embedding_pca_to_rgb(embedding, eps=1e-6):
    if embedding.ndim != 3:
        raise ValueError(f"Expected embedding [C,H,W], got {embedding.shape}")
    channels, height, width = embedding.shape
    values = embedding.detach().float().permute(1, 2, 0).reshape(-1, channels)
    finite = torch.isfinite(values).all(dim=1)
    if finite.sum() < 10:
        return torch.zeros((height, width, 3)).numpy()

    valid = values[finite]
    centered = valid - valid.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (centered.shape[0] + eps)
    _, eigenvectors = torch.linalg.eigh(covariance)
    component_count = min(3, channels)
    projected_valid = centered @ eigenvectors[:, -component_count:]
    if component_count < 3:
        projected_valid = torch.cat(
            (
                projected_valid,
                torch.zeros(
                    projected_valid.shape[0],
                    3 - component_count,
                    device=projected_valid.device,
                    dtype=projected_valid.dtype,
                ),
            ),
            dim=1,
        )

    projected = torch.zeros((values.shape[0], 3), device=values.device)
    projected[finite] = projected_valid
    projected = projected.reshape(height, width, 3)
    output = torch.zeros_like(projected)
    for channel_index in range(3):
        channel = projected[..., channel_index]
        finite_channel = channel[torch.isfinite(channel)]
        if finite_channel.numel() == 0:
            continue
        low = torch.quantile(finite_channel, 0.01)
        high = torch.quantile(finite_channel, 0.99)
        output[..., channel_index] = (channel - low) / (high - low + eps)
    return output.clamp(0.0, 1.0).cpu().numpy()


@torch.no_grad()
def run_embedding_visualization(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        input_mode=args.input_mode,
        pair_strategy=args.pair_strategy,
        pair_max_view_id_gap=args.pair_max_view_id_gap,
    )
    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        raise IndexError(
            f"sample_idx={args.sample_idx} out of range, dataset size={len(dataset)}"
        )

    sample = dataset[args.sample_idx]
    batch = {
        "img": sample["img"].unsqueeze(0),
        "gt_line": sample["gt_line"].unsqueeze(0),
        "gt_plane": sample["gt_plane"].unsqueeze(0),
    }
    if "img2" in sample:
        batch.update(
            {
                "img1": sample["img1"].unsqueeze(0),
                "img2": sample["img2"].unsqueeze(0),
                "gt_line1": sample["gt_line1"].unsqueeze(0),
                "gt_line2": sample["gt_line2"].unsqueeze(0),
                "gt_plane1": sample["gt_plane1"].unsqueeze(0),
                "gt_plane2": sample["gt_plane2"].unsqueeze(0),
            }
        )
    batch = move_batch_to_device(batch, device)
    view1, view2 = build_views_from_batch(batch, prefix=args.split)

    backbone = build_dust3r_backbone(args.weights_path, device=device)
    model = LightReconModel(
        dust3r_backbone=backbone,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    ).to(device)
    checkpoint = safe_load_checkpoint(args.ckpt_path, device)
    missing, unexpected, skipped = load_state_dict_for_visualization(
        model, checkpoint, allow_partial=args.allow_partial_load
    )
    if skipped:
        print(f"[Warning] skipped mismatched keys: {skipped[:10]}")
    if missing:
        print(f"[Warning] missing keys: {missing[:10]}")
    if unexpected:
        print(f"[Warning] unexpected keys: {unexpected[:10]}")

    model.eval()
    result1, result2 = model(view1, view2)

    def collect(view_index, result):
        if view_index == 1:
            image_key = "img1" if "img1" in batch else "img"
            line_key = "gt_line1" if "gt_line1" in batch else "gt_line"
            plane_key = "gt_plane1" if "gt_plane1" in batch else "gt_plane"
        else:
            image_key = "img2" if "img2" in batch else "img"
            line_key = "gt_line2" if "gt_line2" in batch else "gt_line"
            plane_key = "gt_plane2" if "gt_plane2" in batch else "gt_plane"
        line = batch[line_key][0]
        if line.ndim == 3:
            line = line[0]
        return (
            tensor_image_to_numpy(batch[image_key][0]),
            line.detach().cpu().numpy(),
            torch.sigmoid(result["pred_line"][0, 0]).detach().cpu().numpy(),
            batch[plane_key][0].detach().cpu().numpy(),
            embedding_pca_to_rgb(result["pred_plane"][0]),
        )

    rows = (collect(1, result1), collect(2, result2))
    titles = (
        "RGB",
        "GT Line",
        "Pred Line Prob",
        "GT Plane Instance",
        "Pred Plane Embedding PCA",
    )
    figure, axes = plt.subplots(2, 5, figsize=(22, 9))
    for row_index, outputs in enumerate(rows):
        for column_index, (title, image) in enumerate(zip(titles, outputs)):
            axis = axes[row_index, column_index]
            axis.set_title(f"View {row_index + 1} {title}")
            if title in ("GT Line", "Pred Line Prob"):
                axis.imshow(image, cmap="gray")
            elif title == "GT Plane Instance":
                axis.imshow(image, cmap="jet")
            else:
                axis.imshow(image)
            axis.axis("off")
    figure.tight_layout()
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Visualization saved to: {save_path}")


def flatten_cached_samples(cache_samples):
    flattened = []
    for cache_item in cache_samples:
        batch_size = int(cache_item["rgb1"].shape[0])
        for batch_index in range(batch_size):
            flattened.append(
                {
                    "features1": {
                        stage: cache_item["features1"][stage][batch_index : batch_index + 1]
                        for stage in STAGE_NAMES
                    },
                    "features2": {
                        stage: cache_item["features2"][stage][batch_index : batch_index + 1]
                        for stage in STAGE_NAMES
                    },
                    "rgb1": cache_item["rgb1"][batch_index : batch_index + 1],
                    "rgb2": cache_item["rgb2"][batch_index : batch_index + 1],
                    "gt_plane1": cache_item["gt_plane1"][batch_index : batch_index + 1],
                    "gt_plane2": cache_item["gt_plane2"][batch_index : batch_index + 1],
                    **(
                        {
                            "geometry1": cache_item["geometry1"][
                                batch_index : batch_index + 1
                            ],
                            "geometry2": cache_item["geometry2"][
                                batch_index : batch_index + 1
                            ],
                        }
                        if "geometry1" in cache_item and "geometry2" in cache_item
                        else {}
                    ),
                }
            )
    return flattened


def parse_manual_indices(text, fallback):
    if not text.strip():
        return [int(fallback)]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def build_multiscale_head(checkpoint, cache_config, device):
    if checkpoint.get("model_type") != "MultiScalePlaneMaskHead":
        raise ValueError(
            "Checkpoint is not a MultiScalePlaneMaskHead checkpoint: "
            f"model_type={checkpoint.get('model_type')}"
        )
    config = checkpoint.get("args", {})
    input_dims = tuple(
        int(value)
        for value in checkpoint.get(
            "input_dims",
            cache_config.get("input_dims", (1024, 768, 768, 768)),
        )
    )
    head = MultiScalePlaneMaskHead(
        input_dims=input_dims,
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_queries=int(config.get("num_queries", 8)),
        num_decoder_layers=int(config.get("decoder_layers", 3)),
        num_heads=int(config.get("decoder_heads", 8)),
        output_size=int(config.get("output_size", 128)),
        use_rgb_skip=not bool(config.get("disable_rgb_skip", False)),
        use_geometry=bool(config.get("use_geometry", False)),
        geometry_dim=int(cache_config.get("geometry_channels", 9)),
        use_masked_query_refine=bool(config.get("use_masked_query_refine", False)),
        decoder_ffn_multiplier=int(config.get("decoder_ffn_multiplier", 4)),
        fuse_refine_blocks=int(config.get("fuse_refine_blocks", 1)),
        pixel_refine_blocks=int(config.get("pixel_refine_blocks", 1)),
    ).to(device)
    head.load_state_dict(checkpoint["head"], strict=True)
    head.eval()
    return head, config


def output_slice(output, index):
    return {
        "mask_logits": output["mask_logits"][index],
        "background_logits": output["background_logits"][index],
        "existence_logits": output["existence_logits"][index],
    }


def analyze_view(output, labels, config):
    num_queries = int(config.get("num_queries", 8))
    min_plane_pixels = int(config.get("min_plane_pixels", 4))
    match_args = argparse.Namespace(
        match_bce_weight=float(config.get("match_bce_weight", 1.0)),
        match_dice_weight=float(config.get("match_dice_weight", 2.0)),
        match_existence_weight=float(config.get("match_existence_weight", 0.0)),
    )
    score_args = argparse.Namespace(
        class_score_weight=float(config.get("class_score_weight", 0.0))
    )
    target_hw = output["mask_logits"].shape[-2:]
    _, plane_ids = select_plane_ids(
        labels[None], target_hw, num_queries, min_plane_pixels
    )
    resized_labels, masks = masks_for_plane_ids(
        labels[None], target_hw, plane_ids
    )
    targets = (
        torch.stack(masks[0])
        if masks[0]
        else output["mask_logits"].new_zeros((0, *target_hw))
    )
    try:
        query_ids, target_ids = match_queries(
            output["mask_logits"],
            targets,
            match_args,
            existence_logits=output["existence_logits"],
        )
    except TypeError:
        query_ids, target_ids = match_queries(
            output["mask_logits"],
            targets,
            match_args,
        )

    class_logits = torch.cat(
        (
            apply_query_class_scores(
                output["mask_logits"],
                output["existence_logits"],
                score_args,
            ),
            output["background_logits"],
        ),
        dim=0,
    )
    predicted = class_logits.argmax(dim=0)
    gt_query = class_target_from_matches(
        resized_labels[0], targets, query_ids, target_ids, num_queries
    )

    ious = []
    for query_id, target_id in zip(query_ids, target_ids):
        ground_truth = targets[int(target_id)] > 0.5
        prediction = predicted == int(query_id)
        union = (ground_truth | prediction).sum().clamp_min(1)
        ious.append(float(((ground_truth & prediction).sum() / union).detach().cpu()))

    valid_plane = gt_query < num_queries
    background = gt_query == num_queries
    correct_plane = valid_plane & (predicted == gt_query)
    wrong_plane = valid_plane & (predicted < num_queries) & (predicted != gt_query)
    plane_miss = valid_plane & (predicted == num_queries)
    background_error = background & (predicted < num_queries)

    error_codes = torch.zeros_like(predicted, dtype=torch.long)
    error_codes[correct_plane] = 1
    error_codes[wrong_plane] = 2
    error_codes[plane_miss] = 3
    error_codes[background_error] = 4

    pred_region_count = sum(
        int((predicted == query_index).any()) for query_index in range(num_queries)
    )
    return {
        "metrics": {
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "leakage_rate": float(
                wrong_plane.sum() / valid_plane.sum().clamp_min(1)
            ),
            "plane_miss_rate": float(
                plane_miss.sum() / valid_plane.sum().clamp_min(1)
            ),
            "background_error_rate": float(
                background_error.sum() / background.sum().clamp_min(1)
            ),
            "gt_plane_count": int(len(target_ids)),
            "pred_region_count": int(pred_region_count),
        },
        "predicted": predicted.detach().cpu().numpy().astype(np.int16),
        "gt_query": gt_query.detach().cpu().numpy().astype(np.int16),
        "error_codes": error_codes.detach().cpu().numpy().astype(np.uint8),
    }


def load_cache_metadata(cache_config, split, pair_count):
    metadata = [{} for _ in range(pair_count)]
    selected_key = f"selected_{split}_indices"
    selected_indices = cache_config.get(selected_key)
    root_dir = cache_config.get("root_dir")
    if not root_dir or not selected_indices or len(selected_indices) != pair_count:
        return metadata

    try:
        dataset = Structured3DDataset(
            root_dir=root_dir,
            split=split,
            train_ratio=float(cache_config.get("train_ratio", 0.9)),
            image_size=(512, 512),
            input_mode="pair",
            pair_strategy=cache_config.get("pair_strategy", "adjacent"),
            pair_max_view_id_gap=cache_config.get("pair_max_view_id_gap"),
        )
        for cache_index, dataset_index in enumerate(selected_indices):
            sample = dataset.samples[int(dataset_index)]
            view1 = sample["view1"]
            view2 = sample["view2"]
            metadata[cache_index] = {
                "dataset_index": int(dataset_index),
                "scene_name": sample.get("scene_name", ""),
                "pair_group": str(sample.get("pair_group", "")),
                "view1_id": str(view1.get("view_id", "")),
                "view2_id": str(view2.get("view_id", "")),
            }
    except Exception as error:
        print(f"[Warning] Could not reconstruct cache metadata: {error}")
    return metadata


@torch.no_grad()
def collect_multiscale_records(head, samples, metadata, config, device, batch_size):
    metric_rows = []
    visualization_rows = []
    for start in range(0, len(samples), batch_size):
        items = samples[start : start + batch_size]
        pair_batch = len(items)
        features1 = {
            stage: torch.cat([item["features1"][stage] for item in items], dim=0)
            .to(device=device, dtype=torch.float32)
            for stage in STAGE_NAMES
        }
        features2 = {
            stage: torch.cat([item["features2"][stage] for item in items], dim=0)
            .to(device=device, dtype=torch.float32)
            for stage in STAGE_NAMES
        }
        features = {
            stage: torch.cat((features1[stage], features2[stage]), dim=0)
            for stage in STAGE_NAMES
        }
        rgb1 = torch.cat([item["rgb1"] for item in items], dim=0).to(
            device=device, dtype=torch.float32
        )
        rgb2 = torch.cat([item["rgb2"] for item in items], dim=0).to(
            device=device, dtype=torch.float32
        )
        if bool(config.get("use_geometry", False)):
            if "geometry1" not in items[0] or "geometry2" not in items[0]:
                raise RuntimeError(
                    "This checkpoint requires geometry1/geometry2 in the feature cache"
                )
            geometry1 = torch.cat([item["geometry1"] for item in items], dim=0).to(
                device=device, dtype=torch.float32
            )
            geometry2 = torch.cat([item["geometry2"] for item in items], dim=0).to(
                device=device, dtype=torch.float32
            )
            geometry = torch.cat((geometry1, geometry2), dim=0)
        else:
            geometry = None
        output = head(features, torch.cat((rgb1, rgb2), dim=0), geometry=geometry)

        for local_index, item in enumerate(items):
            sample_index = start + local_index
            view1 = analyze_view(
                output_slice(output, local_index), item["gt_plane1"][0].to(device), config
            )
            view2 = analyze_view(
                output_slice(output, pair_batch + local_index),
                item["gt_plane2"][0].to(device),
                config,
            )
            metrics1 = view1["metrics"]
            metrics2 = view2["metrics"]
            row = {
                "sample_idx": sample_index,
                **metadata[sample_index],
                "view1_mean_iou": metrics1["mean_iou"],
                "view2_mean_iou": metrics2["mean_iou"],
                "mean_iou": 0.5 * (metrics1["mean_iou"] + metrics2["mean_iou"]),
                "view_gap": abs(metrics1["mean_iou"] - metrics2["mean_iou"]),
                "view1_leakage": metrics1["leakage_rate"],
                "view2_leakage": metrics2["leakage_rate"],
                "leakage": 0.5
                * (metrics1["leakage_rate"] + metrics2["leakage_rate"]),
                "view1_plane_miss": metrics1["plane_miss_rate"],
                "view2_plane_miss": metrics2["plane_miss_rate"],
                "plane_miss": 0.5
                * (metrics1["plane_miss_rate"] + metrics2["plane_miss_rate"]),
                "view1_background_error": metrics1["background_error_rate"],
                "view2_background_error": metrics2["background_error_rate"],
                "background_error": 0.5
                * (
                    metrics1["background_error_rate"]
                    + metrics2["background_error_rate"]
                ),
                "view1_gt_plane_count": metrics1["gt_plane_count"],
                "view2_gt_plane_count": metrics2["gt_plane_count"],
                "view1_pred_region_count": metrics1["pred_region_count"],
                "view2_pred_region_count": metrics2["pred_region_count"],
            }
            metric_rows.append(row)
            visualization_rows.append(
                {
                    "metrics": row,
                    "rgb1": tensor_image_to_numpy(item["rgb1"][0]),
                    "rgb2": tensor_image_to_numpy(item["rgb2"][0]),
                    "view1": view1,
                    "view2": view2,
                }
            )

        processed = min(start + pair_batch, len(samples))
        if start == 0 or processed == len(samples) or processed % 32 == 0:
            print(f"[Visualization inference] {processed}/{len(samples)}", flush=True)
    return metric_rows, visualization_rows


def colorize_class_map(class_map, num_queries):
    output = np.zeros((*class_map.shape, 3), dtype=np.float32)
    for query_index in range(num_queries):
        output[class_map == query_index] = QUERY_COLORS[
            query_index % len(QUERY_COLORS)
        ]
    return output


def boundary_map(class_map):
    boundary = np.zeros(class_map.shape, dtype=bool)
    horizontal = class_map[:, 1:] != class_map[:, :-1]
    vertical = class_map[1:, :] != class_map[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def build_overlay(rgb, predicted, gt_query, num_queries):
    prediction_rgb = colorize_class_map(predicted, num_queries)
    plane_pixels = predicted < num_queries
    overlay = rgb.copy()
    overlay[plane_pixels] = (
        0.55 * overlay[plane_pixels] + 0.45 * prediction_rgb[plane_pixels]
    )
    predicted_boundary = boundary_map(predicted)
    gt_boundary = boundary_map(gt_query)
    overlay[predicted_boundary] = np.asarray([1.0, 1.0, 1.0])
    overlay[gt_boundary] = np.asarray([0.0, 1.0, 1.0])
    return np.clip(overlay, 0.0, 1.0)


def render_case(case, output_path, checkpoint_epoch, num_queries, split):
    metrics = case["metrics"]
    figure, axes = plt.subplots(2, 5, figsize=(22, 9))
    error_legend = [
        Patch(facecolor=ERROR_COLORS[1], label="Correct plane"),
        Patch(facecolor=ERROR_COLORS[2], label="Wrong plane"),
        Patch(facecolor=ERROR_COLORS[3], label="Plane miss"),
        Patch(facecolor=ERROR_COLORS[4], label="Background → plane"),
    ]
    boundary_legend = [
        Line2D([0], [0], color="white", linewidth=2, label="Pred boundary"),
        Line2D([0], [0], color="cyan", linewidth=2, label="GT boundary"),
    ]

    for row_index, (rgb, view_data, prefix) in enumerate(
        (
            (case["rgb1"], case["view1"], "view1"),
            (case["rgb2"], case["view2"], "view2"),
        )
    ):
        predicted = view_data["predicted"]
        gt_query = view_data["gt_query"]
        gt_rgb = colorize_class_map(gt_query, num_queries)
        pred_rgb = colorize_class_map(predicted, num_queries)
        error_rgb = ERROR_COLORS[view_data["error_codes"]]
        overlay = build_overlay(rgb, predicted, gt_query, num_queries)
        images = (rgb, gt_rgb, pred_rgb, error_rgb, overlay)
        titles = (
            "RGB",
            "GT plane (matched colors)",
            "Pred plane (raw argmax)",
            "Error map",
            "Prediction overlay",
        )
        for column_index, (image, title) in enumerate(zip(images, titles)):
            axis = axes[row_index, column_index]
            axis.imshow(image)
            if column_index == 0:
                axis.set_title(
                    f"{prefix.upper()} {title}\n"
                    f"IoU={metrics[prefix + '_mean_iou']:.3f}  "
                    f"leak={metrics[prefix + '_leakage']:.3f}  "
                    f"miss={metrics[prefix + '_plane_miss']:.3f}"
                )
            else:
                axis.set_title(title)
            axis.axis("off")

    axes[0, 3].legend(
        handles=error_legend,
        loc="lower left",
        fontsize=8,
        framealpha=0.75,
    )
    axes[0, 4].legend(
        handles=boundary_legend,
        loc="lower left",
        fontsize=8,
        framealpha=0.75,
    )
    scene_text = metrics.get("scene_name", "")
    view_text = ""
    if metrics.get("view1_id", "") or metrics.get("view2_id", ""):
        view_text = f" views={metrics.get('view1_id', '')}/{metrics.get('view2_id', '')}"
    figure.suptitle(
        f"split={split} cache_idx={metrics['sample_idx']} scene={scene_text}{view_text}  "
        f"pair IoU={metrics['mean_iou']:.4f} gap={metrics['view_gap']:.4f}  "
        f"leak={metrics['leakage']:.4f} bgerr={metrics['background_error']:.4f}\n"
        f"GT planes={metrics['view1_gt_plane_count']}/{metrics['view2_gt_plane_count']}  "
        f"raw pred regions={metrics['view1_pred_region_count']}/"
        f"{metrics['view2_pred_region_count']}  checkpoint epoch={checkpoint_epoch}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_metric_files(rows, output_dir):
    json_path = output_dir / "all_metrics.json"
    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_path = output_dir / "all_metrics.csv"
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def save_metrics_overview(rows, output_path):
    mean_iou = np.asarray([row["mean_iou"] for row in rows], dtype=np.float64)
    leakage = np.asarray([row["leakage"] for row in rows], dtype=np.float64)
    view1 = np.asarray([row["view1_mean_iou"] for row in rows], dtype=np.float64)
    view2 = np.asarray([row["view2_mean_iou"] for row in rows], dtype=np.float64)
    view_gap = np.asarray([row["view_gap"] for row in rows], dtype=np.float64)

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(mean_iou, bins=20)
    axes[0, 0].axvline(mean_iou.mean(), linestyle="--", label=f"mean={mean_iou.mean():.3f}")
    axes[0, 0].set_title("Pair mean IoU distribution")
    axes[0, 0].set_xlabel("Mean IoU")
    axes[0, 0].legend()

    axes[0, 1].hist(leakage, bins=20)
    axes[0, 1].axvline(leakage.mean(), linestyle="--", label=f"mean={leakage.mean():.3f}")
    axes[0, 1].set_title("Leakage distribution")
    axes[0, 1].set_xlabel("Leakage rate")
    axes[0, 1].legend()

    axes[1, 0].scatter(view1, view2, alpha=0.7)
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel("View1 mean IoU")
    axes[1, 0].set_ylabel("View2 mean IoU")
    axes[1, 0].set_title("View1 vs View2")

    axes[1, 1].hist(view_gap, bins=20)
    axes[1, 1].axvline(view_gap.mean(), linestyle="--", label=f"mean={view_gap.mean():.3f}")
    axes[1, 1].set_title("View-gap distribution")
    axes[1, 1].set_xlabel("Absolute IoU gap")
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def print_distribution_summary(rows):
    mean_iou = np.asarray([row["mean_iou"] for row in rows], dtype=np.float64)
    leakage = np.asarray([row["leakage"] for row in rows], dtype=np.float64)
    view_gap = np.asarray([row["view_gap"] for row in rows], dtype=np.float64)
    quantiles = np.quantile(mean_iou, [0.10, 0.25, 0.50, 0.75, 0.90])
    below = int((mean_iou < 0.5).sum())
    above = int((mean_iou > 0.85).sum())
    print("=" * 80)
    print(f"pair count             : {len(rows)}")
    print(f"mean / median IoU      : {mean_iou.mean():.6f} / {np.median(mean_iou):.6f}")
    print(
        "IoU q10/q25/q50/q75/q90: "
        + " / ".join(f"{value:.6f}" for value in quantiles)
    )
    print(f"IoU < 0.50             : {below} ({below / max(len(rows), 1):.2%})")
    print(f"IoU > 0.85             : {above} ({above / max(len(rows), 1):.2%})")
    print(f"leakage mean / median  : {leakage.mean():.6f} / {np.median(leakage):.6f}")
    print(f"view gap mean / median : {view_gap.mean():.6f} / {np.median(view_gap):.6f}")
    print("=" * 80)


def select_case_indices(rows, mode, count, random_seed, manual_indices=None):
    count = min(max(int(count), 1), len(rows))
    if mode == "manual":
        selected = []
        for index in manual_indices or []:
            if index < 0 or index >= len(rows):
                raise IndexError(
                    f"sample index {index} is out of range for {len(rows)} cached pairs"
                )
            selected.append(index)
        return selected
    if mode == "best_iou":
        return [row["sample_idx"] for row in sorted(rows, key=lambda row: row["mean_iou"], reverse=True)[:count]]
    if mode == "worst_iou":
        return [row["sample_idx"] for row in sorted(rows, key=lambda row: row["mean_iou"])[:count]]
    if mode == "highest_leakage":
        return [row["sample_idx"] for row in sorted(rows, key=lambda row: row["leakage"], reverse=True)[:count]]
    if mode == "largest_view_gap":
        return [row["sample_idx"] for row in sorted(rows, key=lambda row: row["view_gap"], reverse=True)[:count]]
    if mode == "random":
        return random.Random(random_seed).sample(range(len(rows)), count)
    raise ValueError(f"Unsupported selection mode: {mode}")


def save_selected_cases(
    rows,
    visualization_rows,
    output_dir,
    mode,
    count,
    random_seed,
    checkpoint_epoch,
    num_queries,
    split,
    manual_indices=None,
):
    selected = select_case_indices(
        rows, mode, count, random_seed, manual_indices=manual_indices
    )
    directory_names = {
        "best_iou": "best_iou",
        "worst_iou": "worst_iou",
        "highest_leakage": "highest_leakage",
        "largest_view_gap": "largest_view_gap",
        "random": "random",
        "manual": "manual",
    }
    target_dir = output_dir / directory_names[mode]
    target_dir.mkdir(parents=True, exist_ok=True)
    for rank, sample_index in enumerate(selected, start=1):
        metric = rows[sample_index]
        file_path = target_dir / (
            f"rank_{rank:02d}_idx_{sample_index:06d}_iou_{metric['mean_iou']:.4f}.png"
        )
        render_case(
            visualization_rows[sample_index],
            file_path,
            checkpoint_epoch,
            num_queries,
            split,
        )
    return target_dir, selected


@torch.no_grad()
def run_multiscale_mask_visualization(args):
    if not args.feature_cache_path:
        raise ValueError("--feature_cache_path is required for multiscale_mask mode")
    if not args.output_dir:
        raise ValueError("--output_dir is required for multiscale_mask mode")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = safe_load_checkpoint(args.ckpt_path, "cpu")
    cache_path = Path(args.feature_cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")
    try:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(cache_path, map_location="cpu")
    if args.split not in cache:
        raise KeyError(f"Cache has no split '{args.split}'. Keys: {list(cache.keys())}")

    cache_config = cache.get("config", {})
    head, config = build_multiscale_head(checkpoint, cache_config, device)
    samples = flatten_cached_samples(cache[args.split])
    if not samples:
        raise RuntimeError(f"Cache split '{args.split}' contains no pairs")
    metadata = load_cache_metadata(cache_config, args.split, len(samples))
    rows, visualization_rows = collect_multiscale_records(
        head,
        samples,
        metadata,
        config,
        device,
        max(int(args.batch_size), 1),
    )

    json_path, csv_path = save_metric_files(rows, output_dir)
    overview_path = output_dir / "metrics_overview.png"
    save_metrics_overview(rows, overview_path)
    print_distribution_summary(rows)

    checkpoint_epoch = checkpoint.get("epoch", "unknown")
    num_queries = int(config.get("num_queries", 8))
    generated = []
    if args.select_mode == "all":
        modes = (
            "best_iou",
            "worst_iou",
            "highest_leakage",
            "largest_view_gap",
        )
        for mode in modes:
            generated.append(
                save_selected_cases(
                    rows,
                    visualization_rows,
                    output_dir,
                    mode,
                    args.num_samples,
                    args.random_seed,
                    checkpoint_epoch,
                    num_queries,
                    args.split,
                )
            )
    else:
        manual_indices = None
        if args.select_mode == "manual":
            manual_indices = parse_manual_indices(args.sample_indices, args.sample_idx)
        generated.append(
            save_selected_cases(
                rows,
                visualization_rows,
                output_dir,
                args.select_mode,
                args.num_samples,
                args.random_seed,
                checkpoint_epoch,
                num_queries,
                args.split,
                manual_indices=manual_indices,
            )
        )

    print(f"Saved metrics JSON : {json_path}")
    print(f"Saved metrics CSV  : {csv_path}")
    print(f"Saved overview     : {overview_path}")
    for directory, selected in generated:
        print(f"Saved {len(selected)} cases to: {directory}")


def main():
    args = parse_args()
    if args.model_type == "embedding":
        run_embedding_visualization(args)
    else:
        run_multiscale_mask_visualization(args)


if __name__ == "__main__":
    main()
