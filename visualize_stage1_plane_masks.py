import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataloaders.s3d_dataset import Structured3DDataset
from models.plane_mask_head import PlaneMaskHead
from train_stage1_plane_masks import (
    build_views,
    feature_maps_from_result,
    masks_for_plane_ids,
    match_queries,
    prediction_masks,
    select_plane_ids,
    typed_boundary_maps,
)


COLORS = np.asarray(
    [
        [230, 57, 53],
        [33, 150, 243],
        [67, 160, 71],
        [255, 143, 0],
        [142, 36, 170],
        [0, 137, 123],
        [244, 81, 30],
        [117, 117, 117],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser("Visualize coarse-to-fine Stage1 plane masks")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample_indices", type=int, nargs="+", required=True)
    parser.add_argument("--existence_threshold", type=float, default=None)
    parser.add_argument("--full_mask_threshold", type=float, default=None)
    parser.add_argument("--core_mask_threshold", type=float, default=None)
    parser.add_argument("--core_margin_threshold", type=float, default=None)
    return parser.parse_args()


def colorize(ids):
    output = np.full((*ids.shape, 3), 245, dtype=np.uint8)
    for plane_id in np.unique(ids):
        if plane_id < 0:
            continue
        output[ids == plane_id] = COLORS[int(plane_id) % len(COLORS)]
    return output


def image_uint8(image):
    image = image.detach().cpu()
    if float(image.min()) < -0.05:
        image = (image + 1.0) * 0.5
    return (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def keep_largest_component(mask):
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    if count <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return torch.from_numpy(labels == largest).to(mask.device)


def assignment_from_output(output, scale, thresholds):
    masks = prediction_masks(
        output[f"mask_logits_{scale}"][0],
        output[f"background_logits_{scale}"][0],
        output["existence_logits"][0],
        **thresholds,
    )
    masks["core_masks"] = torch.stack(
        [keep_largest_component(mask) for mask in masks["core_masks"]]
    )
    return masks


def make_gt_query_ids(targets, query_ids, target_ids, shape, device):
    gt_ids = torch.full(shape, -1, device=device, dtype=torch.long)
    for query_id, target_id in zip(query_ids, target_ids):
        gt_ids[targets[target_id] > 0.5] = int(query_id)
    return gt_ids


def boundary_f1(predicted, target, tolerance):
    pred_boundary = torch.zeros_like(predicted, dtype=torch.bool)
    gt_boundary = torch.zeros_like(target, dtype=torch.bool)
    pred_boundary[:, 1:] |= predicted[:, 1:] != predicted[:, :-1]
    pred_boundary[:, :-1] |= predicted[:, 1:] != predicted[:, :-1]
    pred_boundary[1:, :] |= predicted[1:, :] != predicted[:-1, :]
    pred_boundary[:-1, :] |= predicted[1:, :] != predicted[:-1, :]
    gt_boundary[:, 1:] |= target[:, 1:] != target[:, :-1]
    gt_boundary[:, :-1] |= target[:, 1:] != target[:, :-1]
    gt_boundary[1:, :] |= target[1:, :] != target[:-1, :]
    gt_boundary[:-1, :] |= target[1:, :] != target[:-1, :]
    kernel = tolerance * 2 + 1
    pred_dilated = F.max_pool2d(
        pred_boundary[None, None].float(), kernel, 1, tolerance
    )[0, 0] > 0
    gt_dilated = F.max_pool2d(
        gt_boundary[None, None].float(), kernel, 1, tolerance
    )[0, 0] > 0
    precision = (
        (pred_boundary & gt_dilated).sum().float()
        / pred_boundary.sum().clamp_min(1)
    )
    recall = (
        (gt_boundary & pred_dilated).sum().float()
        / gt_boundary.sum().clamp_min(1)
    )
    return float((2 * precision * recall / (precision + recall).clamp_min(1e-6)).cpu())


def instance_metrics(predicted, gt_ids, targets, query_ids, target_ids):
    ious = []
    size_groups = {"small": [], "medium": [], "large": []}
    image_area = float(gt_ids.numel())
    confusion = np.zeros((len(target_ids), predicted.max().clamp_min(0).item() + 1), dtype=np.int64)
    for local_id, (query_id, target_id) in enumerate(zip(query_ids, target_ids)):
        pred_mask = predicted == int(query_id)
        target_mask = targets[target_id] > 0.5
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().clamp_min(1)
        iou = float((intersection / union).cpu())
        ious.append(iou)
        area_ratio = float(target_mask.sum()) / image_area
        group = "small" if area_ratio < 0.02 else "medium" if area_ratio < 0.10 else "large"
        size_groups[group].append(iou)
        for predicted_query in np.unique(predicted[target_mask].detach().cpu().numpy()):
            if predicted_query >= 0 and predicted_query < confusion.shape[1]:
                confusion[local_id, predicted_query] = int(
                    ((predicted == int(predicted_query)) & target_mask).sum()
                )
    valid = gt_ids >= 0
    wrong_plane = valid & (predicted >= 0) & (predicted != gt_ids)
    leakage = float(wrong_plane.sum().float() / valid.sum().clamp_min(1))
    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    split_rate = (
        float(((normalized > 0.10).sum(axis=1) > 1).mean())
        if len(normalized)
        else 0.0
    )
    column_support = confusion / np.maximum(confusion.sum(axis=0, keepdims=True), 1)
    merge_rate = (
        float(((column_support > 0.10).sum(axis=0) > 1).mean())
        if column_support.shape[1]
        else 0.0
    )
    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "small_iou": float(np.mean(size_groups["small"])) if size_groups["small"] else None,
        "medium_iou": float(np.mean(size_groups["medium"])) if size_groups["medium"] else None,
        "large_iou": float(np.mean(size_groups["large"])) if size_groups["large"] else None,
        "leakage_rate": leakage,
        "split_rate": split_rate,
        "merge_rate": merge_rate,
    }


def fit_plane(points, mask):
    valid = mask & torch.isfinite(points).all(dim=-1)
    selected = points[valid]
    if selected.shape[0] < 3:
        return None
    centroid = selected.mean(dim=0)
    centered = selected - centroid
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    normal = F.normalize(vh[-1], dim=0)
    offset = -torch.dot(normal, centroid)
    residual = (selected @ normal + offset).abs().mean()
    return normal, offset, residual


def compare_plane_fits(points, target_masks, full_masks, core_masks, query_ids, target_ids):
    rows = []
    for query_id, target_id in zip(query_ids, target_ids):
        gt_fit = fit_plane(points, target_masks[target_id] > 0.5)
        full_fit = fit_plane(points, full_masks[query_id])
        core_fit = fit_plane(points, core_masks[query_id])
        row = {"query_id": int(query_id)}
        if gt_fit is not None:
            row["gt_residual"] = float(gt_fit[2].cpu())
        for name, fitted in (("full", full_fit), ("core", core_fit)):
            if gt_fit is None or fitted is None:
                row[f"{name}_normal_angle_deg"] = None
                row[f"{name}_offset_error"] = None
                row[f"{name}_residual"] = None
                continue
            dot = torch.dot(gt_fit[0], fitted[0])
            sign = 1.0 if float(dot) >= 0 else -1.0
            angle = torch.rad2deg(torch.acos(dot.abs().clamp(max=1.0)))
            offset_error = (gt_fit[1] - sign * fitted[1]).abs()
            row[f"{name}_normal_angle_deg"] = float(angle.cpu())
            row[f"{name}_offset_error"] = float(offset_error.cpu())
            row[f"{name}_residual"] = float(fitted[2].cpu())
        rows.append(row)
    return rows


@torch.no_grad()
def main():
    args = parse_args()
    import matplotlib.pyplot as plt

    from models.build_backbone import build_dust3r_backbone

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        image_size=(512, 512),
        input_mode=config.get("input_mode", "single"),
    )
    feature_indices = config.get("feature_indices", [0, 6, 9, 12])
    feature_dims = config.get("feature_dims", [1024, 768, 768, 768])
    backbone = build_dust3r_backbone(args.weights_path, device=device)
    backbone.eval()
    head = PlaneMaskHead(
        feature_dim=feature_dims[3],
        encoder_feature_dim=feature_dims[0],
        shallow_feature_dim=feature_dims[1],
        middle_feature_dim=feature_dims[2],
        hidden_dim=config["hidden_dim"],
        num_queries=config["num_queries"],
        num_decoder_layers=config["decoder_layers"],
        num_heads=config["decoder_heads"],
        use_rgb_edge=not config.get("disable_rgb_edge", False),
        refinement_margin=config.get("refinement_margin", 0.55),
    ).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    thresholds = {
        "existence_threshold": (
            args.existence_threshold
            if args.existence_threshold is not None
            else config.get("existence_threshold", 0.5)
        ),
        "full_threshold": (
            args.full_mask_threshold
            if args.full_mask_threshold is not None
            else config.get("full_mask_threshold", 0.5)
        ),
        "core_threshold": (
            args.core_mask_threshold
            if args.core_mask_threshold is not None
            else config.get("core_mask_threshold", 0.75)
        ),
        "core_margin_threshold": (
            args.core_margin_threshold
            if args.core_margin_threshold is not None
            else config.get("core_margin_threshold", 0.35)
        ),
    }

    rows = []
    for sample_idx in args.sample_indices:
        sample = dataset[sample_idx]
        batch = {
            key: value[None].to(device) if torch.is_tensor(value) else [value]
            for key, value in sample.items()
        }
        view1, view2 = build_views(batch, f"mask_vis_{sample_idx}")
        result1, _ = backbone(view1, view2)
        features = feature_maps_from_result(result1, view1["img"], feature_indices)
        output = head(
            features["deep"],
            middle_feature=features["middle"],
            shallow_feature=features["shallow"],
            encoder_feature=features["encoder"],
            image=view1["img"],
        )
        _, plane_ids = select_plane_ids(
            batch["gt_plane"],
            output["mask_logits_32"].shape[-2:],
            config["num_queries"],
            config["min_plane_pixels"],
        )
        labels32, targets32_list = masks_for_plane_ids(
            batch["gt_plane"],
            output["mask_logits_32"].shape[-2:],
            plane_ids,
        )
        targets32 = torch.stack(targets32_list[0])
        query_ids, target_ids = match_queries(
            output["mask_logits_32"][0],
            targets32,
            argparse.Namespace(**config),
        )

        scale_assignments = {}
        scale_metrics = {}
        scale_gt = {}
        target_masks128 = None
        masks128 = None
        for scale in (32, 64, 128):
            labels, target_lists = masks_for_plane_ids(
                batch["gt_plane"],
                output[f"mask_logits_{scale}"].shape[-2:],
                plane_ids,
            )
            targets = torch.stack(target_lists[0])
            gt_ids = make_gt_query_ids(
                targets,
                query_ids,
                target_ids,
                labels[0].shape,
                device,
            )
            masks = assignment_from_output(output, scale, thresholds)
            metrics = instance_metrics(
                masks["assignment"],
                gt_ids,
                targets,
                query_ids,
                target_ids,
            )
            metrics["boundary_f1_t2"] = boundary_f1(
                masks["assignment"], gt_ids, tolerance=2
            )
            metrics["boundary_f1_t4"] = boundary_f1(
                masks["assignment"], gt_ids, tolerance=4
            )
            scale_assignments[scale] = masks["assignment"]
            scale_metrics[scale] = metrics
            scale_gt[scale] = gt_ids
            if scale == 128:
                target_masks128 = targets
                masks128 = masks

        full_union = masks128["full_masks"].any(dim=0)
        core_union = masks128["core_masks"].any(dim=0)
        gt_valid = scale_gt[128] >= 0
        full_recall = float((full_union & gt_valid).sum().float() / gt_valid.sum().clamp_min(1))
        core_precision = float(
            (core_union & gt_valid & (masks128["assignment"] == scale_gt[128])).sum().float()
            / core_union.sum().clamp_min(1)
        )

        points = result1["pts3d"][0]
        points128 = F.interpolate(
            points.permute(2, 0, 1)[None],
            size=(128, 128),
            mode="nearest",
        )[0].permute(1, 2, 0)
        plane_fits = compare_plane_fits(
            points128,
            target_masks128,
            masks128["full_masks"],
            masks128["core_masks"],
            query_ids,
            target_ids,
        )

        rgb = image_uint8(batch["img"][0])
        error = (scale_gt[128] >= 0) & (
            scale_assignments[128] != scale_gt[128]
        )
        error_rgb = colorize(scale_assignments[128].cpu().numpy())
        error_rgb[error.cpu().numpy()] = np.asarray([255, 0, 255], dtype=np.uint8)
        full_assignment = masks128["assignment"].clone()
        full_assignment[~full_union] = -1
        core_assignment = masks128["assignment"].clone()
        core_assignment[~core_union] = -1

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        panels = [
            (rgb, "Input RGB"),
            (colorize(scale_gt[128].cpu().numpy()), f"GT ({len(targets32)})"),
            (colorize(scale_assignments[32].cpu().numpy()), "Coarse 32"),
            (colorize(scale_assignments[64].cpu().numpy()), "Middle 64"),
            (colorize(scale_assignments[128].cpu().numpy()), "Fine 128"),
            (colorize(full_assignment.cpu().numpy()), "Full support"),
            (colorize(core_assignment.cpu().numpy()), "Core fit support"),
            (error_rgb, "Fine errors (magenta)"),
        ]
        for axis, (image, title) in zip(axes.flat, panels):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(
            f"{args.split}_{sample_idx:06d} | "
            f"IoU 32/64/128={scale_metrics[32]['mean_iou']:.3f}/"
            f"{scale_metrics[64]['mean_iou']:.3f}/"
            f"{scale_metrics[128]['mean_iou']:.3f} | "
            f"alpha={float(head.alpha64):.3f},{float(head.alpha128):.3f}"
        )
        fig.tight_layout()
        image_path = output_dir / f"{args.split}_{sample_idx:06d}_stage1_multiscale_masks.png"
        fig.savefig(image_path, dpi=160)
        plt.close(fig)

        pp_boundary, pb_boundary = typed_boundary_maps(
            F.interpolate(
                batch["gt_plane"][:, None].float(),
                size=(128, 128),
                mode="nearest",
            )[0, 0].long()
        )
        gt_boundary = pp_boundary | pb_boundary
        boundary_band = F.max_pool2d(
            gt_boundary[None, None].float(),
            kernel_size=5,
            stride=1,
            padding=2,
        )[0, 0] > 0
        coarse128 = F.interpolate(
            scale_assignments[32][None, None].float(),
            size=(128, 128),
            mode="nearest",
        )[0, 0].long()
        changed = coarse128 != scale_assignments[128]
        valid_interior = (scale_gt[128] >= 0) & ~boundary_band
        boundary_change_rate = float(
            (changed & boundary_band).sum().float() / boundary_band.sum().clamp_min(1)
        )
        interior_change_rate = float(
            (changed & valid_interior).sum().float() / valid_interior.sum().clamp_min(1)
        )
        coarse_correct = coarse128 == scale_gt[128]
        fine_correct = scale_assignments[128] == scale_gt[128]
        corrected_rate = float(
            ((~coarse_correct) & fine_correct & (scale_gt[128] >= 0)).sum().float()
            / (scale_gt[128] >= 0).sum().clamp_min(1)
        )
        worsened_rate = float(
            (coarse_correct & (~fine_correct) & (scale_gt[128] >= 0)).sum().float()
            / (scale_gt[128] >= 0).sum().clamp_min(1)
        )
        row = {
            "sample_idx": sample_idx,
            "gt_planes": len(targets32),
            "predicted_planes": int(masks128["active"].sum()),
            "plane_count_abs_error": abs(int(masks128["active"].sum()) - len(targets32)),
            "scales": {str(scale): scale_metrics[scale] for scale in (32, 64, 128)},
            "full_mask_recall": full_recall,
            "core_mask_precision": core_precision,
            "core_pixels": int(core_union.sum()),
            "plane_plane_boundary_pixels": int(pp_boundary.sum()),
            "plane_background_boundary_pixels": int(pb_boundary.sum()),
            "boundary_change_rate_32_to_128": boundary_change_rate,
            "interior_change_rate_32_to_128": interior_change_rate,
            "corrected_pixel_rate_32_to_128": corrected_rate,
            "worsened_pixel_rate_32_to_128": worsened_rate,
            "plane_fits": plane_fits,
            "existence": output["existence_logits"][0].sigmoid().cpu().tolist(),
            "alpha64": float(head.alpha64.cpu()),
            "alpha128": float(head.alpha128.cpu()),
            "image": str(image_path),
        }
        rows.append(row)
        print(json.dumps(row))
    (output_dir / "stage1_multiscale_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
