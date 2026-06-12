import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone
from models.plane_mask_head import PlaneMaskHead
from train_stage1_plane_masks import (
    boundary_map,
    build_views,
    feature_map_from_result,
    instance_masks,
    match_queries,
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
    parser = argparse.ArgumentParser("Visualize Stage1 plane masks")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample_indices", type=int, nargs="+", required=True)
    parser.add_argument("--existence_threshold", type=float, default=0.5)
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


@torch.no_grad()
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        image_size=(512, 512),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    backbone = build_dust3r_backbone(args.weights_path, device=device)
    backbone.eval()
    head = PlaneMaskHead(
        feature_dim=768,
        hidden_dim=config["hidden_dim"],
        num_queries=config["num_queries"],
        num_decoder_layers=config["decoder_layers"],
        num_heads=config["decoder_heads"],
    ).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval()

    rows = []
    for sample_idx in args.sample_indices:
        sample = dataset[sample_idx]
        batch = {
            key: value[None].to(device) if torch.is_tensor(value) else [value]
            for key, value in sample.items()
        }
        view1, view2 = build_views(batch, f"mask_vis_{sample_idx}")
        result1, _ = backbone(view1, view2)
        feature_map = feature_map_from_result(result1, view1["img"])
        output = head(feature_map)
        labels, targets_list = instance_masks(
            batch["gt_plane"],
            feature_map.shape[-2:],
            config["num_queries"],
            config["min_plane_pixels"],
        )
        targets = torch.stack(targets_list[0])
        query_ids, target_ids = match_queries(output["mask_logits"][0], targets, argparse.Namespace(**config))
        existence = output["existence_logits"][0].sigmoid()
        active = existence > args.existence_threshold
        class_logits = torch.cat(
            (output["mask_logits"][0], output["background_logits"][0]),
            dim=0,
        )
        inactive = ~active
        class_logits[:-1][inactive] = -1e4
        predicted_query = class_logits.argmax(dim=0)
        predicted_query[predicted_query == config["num_queries"]] = -1

        gt_ids = torch.full_like(labels[0], -1)
        for query_id, target_id in zip(query_ids, target_ids):
            gt_ids[targets[target_id] > 0.5] = int(query_id)
        valid = gt_ids >= 0
        error = valid & (predicted_query != gt_ids)
        boundary = boundary_map(labels[0]) & valid
        matched_ious = []
        for query_id, target_id in zip(query_ids, target_ids):
            predicted_mask = predicted_query == int(query_id)
            target_mask = targets[target_id] > 0.5
            union = (predicted_mask | target_mask).sum()
            if union > 0:
                matched_ious.append(float(((predicted_mask & target_mask).sum() / union).cpu()))

        rgb = image_uint8(batch["img"][0])
        rgb_small = np.asarray(
            F.interpolate(
                torch.from_numpy(rgb).permute(2, 0, 1)[None].float(),
                size=feature_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0].permute(1, 2, 0).byte()
        )
        gt_np = gt_ids.cpu().numpy()
        pred_np = predicted_query.cpu().numpy()
        error_rgb = rgb_small.copy()
        error_rgb[error.cpu().numpy()] = np.asarray([255, 0, 255], dtype=np.uint8)
        error_rgb[boundary.cpu().numpy() & ~error.cpu().numpy()] = np.asarray(
            [0, 255, 255],
            dtype=np.uint8,
        )

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        panels = [
            (rgb, "Input RGB"),
            (colorize(gt_np), f"GT instances ({len(targets)})"),
            (colorize(pred_np), f"Predicted masks ({int(active.sum())})"),
            (error_rgb, "Errors: magenta; correct boundary: cyan"),
        ]
        for axis, (image, title) in zip(axes, panels):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(
            f"{args.split}_{sample_idx:06d} | IoU={np.mean(matched_ious):.3f} | "
            f"existence={np.round(existence.cpu().numpy(), 2).tolist()}"
        )
        fig.tight_layout()
        image_path = output_dir / f"{args.split}_{sample_idx:06d}_stage1_plane_masks.png"
        fig.savefig(image_path, dpi=160)
        plt.close(fig)
        row = {
            "sample_idx": sample_idx,
            "gt_planes": len(targets),
            "predicted_planes": int(active.sum()),
            "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
            "mask_accuracy": float((predicted_query[valid] == gt_ids[valid]).float().mean().cpu()),
            "boundary_accuracy": (
                float((predicted_query[boundary] == gt_ids[boundary]).float().mean().cpu())
                if boundary.any()
                else None
            ),
            "existence": existence.cpu().tolist(),
            "image": str(image_path),
        }
        rows.append(row)
        print(json.dumps(row))
    (output_dir / "stage1_plane_masks_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
