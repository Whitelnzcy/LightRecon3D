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
from models.build_backbone import build_dust3r_backbone
from models.plane_mask_head import PlaneMaskHead


def parse_args():
    parser = argparse.ArgumentParser("Train Stage1 bounded plane mask queries")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--input_mode", default="pair", choices=("pair", "single"))
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--small_train_size", type=int, default=128)
    parser.add_argument("--small_val_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_queries", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--min_plane_pixels", type=int, default=4)
    parser.add_argument("--match_bce_weight", type=float, default=1.0)
    parser.add_argument("--match_dice_weight", type=float, default=2.0)
    parser.add_argument("--mask_bce_weight", type=float, default=1.0)
    parser.add_argument("--mask_dice_weight", type=float, default=2.0)
    parser.add_argument("--existence_weight", type=float, default=0.5)
    parser.add_argument("--partition_weight", type=float, default=1.0)
    parser.add_argument("--boundary_weight", type=float, default=4.0)
    parser.add_argument("--log_every", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


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
    if limit > 0:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
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


def feature_map_from_result(result, image, patch_size=16):
    features = result["dec_features"]
    if isinstance(features, (list, tuple)):
        features = features[-1]
    batch_size, tokens, channels = features.shape
    height = image.shape[-2] // patch_size
    width = image.shape[-1] // patch_size
    if tokens != height * width:
        raise ValueError(f"Token count {tokens} does not match {height}x{width}")
    return features.transpose(1, 2).reshape(batch_size, channels, height, width)


def instance_masks(labels, target_hw, max_planes, min_pixels):
    labels = F.interpolate(
        labels[:, None].float(),
        size=target_hw,
        mode="nearest",
    )[:, 0].long()
    batch_masks = []
    for label_map in labels:
        candidates = []
        for plane_id in torch.unique(label_map):
            plane_id = int(plane_id)
            if plane_id <= 0 or plane_id == 255:
                continue
            mask = label_map == plane_id
            count = int(mask.sum())
            if count >= min_pixels:
                candidates.append((count, mask.float()))
        candidates.sort(key=lambda item: item[0], reverse=True)
        batch_masks.append([mask for _, mask in candidates[:max_planes]])
    return labels, batch_masks


def dice_cost(pred_prob, targets, eps=1e-6):
    intersection = torch.einsum("qhw,thw->qt", pred_prob, targets)
    denominator = pred_prob.sum(dim=(1, 2))[:, None] + targets.sum(dim=(1, 2))[None]
    return 1.0 - (2.0 * intersection + eps) / (denominator + eps)


def match_queries(mask_logits, targets, args):
    if targets.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    pred_prob = mask_logits.sigmoid()
    query_count, height, width = mask_logits.shape
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
    query_count, target_count = cost_np.shape
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
    query_ids = np.asarray(best_assignment, dtype=np.int64)
    target_ids = np.arange(target_count, dtype=np.int64)
    return query_ids, target_ids


def boundary_map(label_map):
    boundary = torch.zeros_like(label_map, dtype=torch.bool)
    horizontal = label_map[:, 1:] != label_map[:, :-1]
    vertical = label_map[1:, :] != label_map[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def sample_loss(output, labels, target_masks, args):
    mask_logits = output["mask_logits"]
    existence_logits = output["existence_logits"]
    background_logits = output["background_logits"]
    total = mask_logits.sum() * 0.0
    stats = defaultdict(float)
    batch_size = mask_logits.shape[0]

    for batch_id in range(batch_size):
        targets = (
            torch.stack(target_masks[batch_id])
            if target_masks[batch_id]
            else mask_logits.new_zeros((0, *mask_logits.shape[-2:]))
        )
        query_ids, target_ids = match_queries(mask_logits[batch_id], targets, args)
        query_ids_t = torch.as_tensor(query_ids, device=mask_logits.device, dtype=torch.long)
        target_ids_t = torch.as_tensor(target_ids, device=mask_logits.device, dtype=torch.long)

        existence_target = torch.zeros_like(existence_logits[batch_id])
        if len(query_ids):
            existence_target[query_ids_t] = 1.0
            matched_logits = mask_logits[batch_id, query_ids_t]
            matched_targets = targets[target_ids_t]
            mask_bce = F.binary_cross_entropy_with_logits(matched_logits, matched_targets)
            probabilities = matched_logits.sigmoid()
            intersection = (probabilities * matched_targets).sum(dim=(1, 2))
            denominator = probabilities.sum(dim=(1, 2)) + matched_targets.sum(dim=(1, 2))
            mask_dice = (1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
        else:
            mask_bce = total
            mask_dice = total

        existence = F.binary_cross_entropy_with_logits(
            existence_logits[batch_id],
            existence_target,
        )

        class_logits = torch.cat(
            (mask_logits[batch_id], background_logits[batch_id]),
            dim=0,
        )
        class_target = torch.full(
            labels[batch_id].shape,
            args.num_queries,
            device=labels.device,
            dtype=torch.long,
        )
        for query_id, target_id in zip(query_ids, target_ids):
            class_target[targets[target_id] > 0.5] = int(query_id)
        pixel_ce = F.cross_entropy(class_logits[None], class_target[None], reduction="none")[0]
        weights = torch.ones_like(pixel_ce)
        weights[boundary_map(labels[batch_id])] = args.boundary_weight
        partition = (pixel_ce * weights).sum() / weights.sum().clamp_min(1.0)

        loss = (
            args.mask_bce_weight * mask_bce
            + args.mask_dice_weight * mask_dice
            + args.existence_weight * existence
            + args.partition_weight * partition
        )
        total = total + loss

        predicted = class_logits.argmax(dim=0)
        valid = class_target < args.num_queries
        accuracy = (predicted[valid] == class_target[valid]).float().mean() if valid.any() else total * 0.0
        boundary = boundary_map(labels[batch_id]) & valid
        boundary_accuracy = (
            (predicted[boundary] == class_target[boundary]).float().mean()
            if boundary.any()
            else total * 0.0
        )
        matched_ious = []
        for query_id, target_id in zip(query_ids, target_ids):
            pred_mask = predicted == int(query_id)
            target_mask = targets[target_id] > 0.5
            union = (pred_mask | target_mask).sum()
            if union > 0:
                matched_ious.append((pred_mask & target_mask).sum().float() / union)
        mean_iou = torch.stack(matched_ious).mean() if matched_ious else total * 0.0

        stats["mask_bce"] += float(mask_bce.detach())
        stats["mask_dice"] += float(mask_dice.detach())
        stats["existence"] += float(existence.detach())
        stats["partition"] += float(partition.detach())
        stats["mask_accuracy"] += float(accuracy.detach())
        stats["boundary_accuracy"] += float(boundary_accuracy.detach())
        stats["mean_iou"] += float(mean_iou.detach())
        stats["gt_planes"] += float(targets.shape[0])
        stats["pred_planes"] += float((existence_logits[batch_id].sigmoid() > 0.5).sum())

    return total / batch_size, {key: value / batch_size for key, value in stats.items()}


def run_epoch(backbone, head, loader, optimizer, device, args, train):
    head.train(train)
    backbone.eval()
    totals = defaultdict(float)
    steps = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        view1, view2 = build_views(batch, "stage1")
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            result1, result2 = backbone(view1, view2)
            feature1 = feature_map_from_result(result1, view1["img"])
            feature2 = feature_map_from_result(result2, view2["img"])
        views = [(feature1, batch.get("gt_plane1", batch["gt_plane"]))]
        if args.input_mode == "pair" and "gt_plane2" in batch:
            views.append((feature2, batch["gt_plane2"]))

        losses = []
        rows = []
        for feature_map, gt_plane in views:
            output = head(feature_map)
            labels, masks = instance_masks(
                gt_plane,
                output["mask_logits"].shape[-2:],
                args.num_queries,
                args.min_plane_pixels,
            )
            loss, row = sample_loss(output, labels, masks, args)
            losses.append(loss)
            rows.append(row)
        loss = torch.stack(losses).mean()
        if train:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()

        totals["loss"] += float(loss.detach())
        for row in rows:
            for key, value in row.items():
                totals[key] += value / len(rows)
        steps += 1
        if step == 1 or step % args.log_every == 0 or step == len(loader):
            print(
                f"[{'Train' if train else 'Val'}] {step}/{len(loader)} "
                f"loss={float(loss.detach()):.4f} "
                f"iou={np.mean([row['mean_iou'] for row in rows]):.3f} "
                f"boundary={np.mean([row['boundary_accuracy'] for row in rows]):.3f} "
                f"planes={np.mean([row['pred_planes'] for row in rows]):.1f}/"
                f"{np.mean([row['gt_planes'] for row in rows]):.1f}",
                flush=True,
            )
    return {key: value / max(steps, 1) for key, value in totals.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(args, "train")
    val_loader = make_loader(args, "val")
    backbone = build_dust3r_backbone(args.weights_path, device=device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    head = PlaneMaskHead(
        feature_dim=768,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.decoder_layers,
        num_heads=args.decoder_heads,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        head.load_state_dict(checkpoint["head"])
        print(f"Initialized mask head from {args.init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_iou = -1.0
    for epoch in range(1, args.num_epochs + 1):
        train_stats = run_epoch(backbone, head, train_loader, optimizer, device, args, train=True)
        val_stats = run_epoch(backbone, head, val_loader, None, device, args, train=False)
        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "head": head.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "train_stats": train_stats,
            "val_stats": val_stats,
        }
        torch.save(checkpoint, os.path.join(args.save_dir, "latest.pt"))
        if val_stats["mean_iou"] > best_iou:
            best_iou = val_stats["mean_iou"]
            torch.save(checkpoint, os.path.join(args.save_dir, "best.pt"))
        with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)


if __name__ == "__main__":
    main()
