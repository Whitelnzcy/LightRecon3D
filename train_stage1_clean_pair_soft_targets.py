import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.build_backbone import build_dust3r_backbone
from models.clean_plane_mask_head import CleanPlaneMaskHead
from train_stage1_clean_baseline import (
    dice_loss,
    focal_bce,
    instance_metrics,
    set_seed,
)
from train_stage1_clean_pair_baseline import (
    cached_or_live_pair,
    combine_view_rows,
    load_initial_checkpoint,
    make_loader,
    prepare_pair_data,
    slice_output,
)
from train_stage1_plane_masks import match_queries


def parse_args():
    parser = argparse.ArgumentParser(
        "Train the clean real-pair head with full-resolution masks downsampled by area"
    )
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--pair_strategy", default="adjacent", choices=("adjacent", "all"))
    parser.add_argument("--pair_max_view_id_gap", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--small_train_size", type=int, default=128)
    parser.add_argument("--small_val_size", type=int, default=32)
    parser.add_argument("--train_indices", default="")
    parser.add_argument("--val_indices", default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_queries", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--feature_index", type=int, default=12)
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument(
        "--min_plane_area_tokens",
        type=float,
        default=0.25,
        help=(
            "Minimum area after area downsampling, measured in 32x32 token cells. "
            "Use 4.0 to approximate the old minimum-size policy; use 0.25 to retain "
            "thin and partially covered planes."
        ),
    )
    # Retained for checkpoint/evaluator compatibility. Soft-target selection uses
    # min_plane_area_tokens instead.
    parser.add_argument("--min_plane_pixels", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--match_bce_weight", type=float, default=1.0)
    parser.add_argument("--match_dice_weight", type=float, default=2.0)
    parser.add_argument("--mask_focal_weight", type=float, default=1.0)
    parser.add_argument("--mask_dice_weight", type=float, default=2.0)
    parser.add_argument("--existence_weight", type=float, default=0.1)
    parser.add_argument("--background_weight", type=float, default=1.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--existence_threshold", type=float, default=0.5)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--eval_before_train", action="store_true")
    parser.add_argument("--feature_cache_path", default="")
    parser.add_argument("--disable_feature_cache", action="store_true")
    parser.add_argument("--log_every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260616)
    # Required by shared pair-cache helper; this trainer always supervises both.
    parser.set_defaults(supervision="both")
    return parser.parse_args()


def nearest_labels(labels, target_hw):
    return F.interpolate(
        labels[:, None].float(),
        size=target_hw,
        mode="nearest",
    )[:, 0].long()


def soft_area_targets(labels, target_hw, max_planes, min_area_tokens):
    """Create one soft occupancy mask per full-resolution plane instance."""
    batch_targets = []
    for label_map in labels:
        candidates = []
        for plane_id_tensor in torch.unique(label_map):
            plane_id = int(plane_id_tensor)
            if plane_id <= 0 or plane_id == 255:
                continue
            full_mask = (label_map == plane_id).float()[None, None]
            soft_mask = F.interpolate(
                full_mask,
                size=target_hw,
                mode="area",
            )[0, 0]
            area_tokens = float(soft_mask.sum().detach())
            if area_tokens >= float(min_area_tokens):
                candidates.append((area_tokens, plane_id, soft_mask))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[:max_planes]
        if selected:
            targets = torch.stack([item[2] for item in selected], dim=0)
        else:
            targets = labels.new_zeros((0, *target_hw), dtype=torch.float32)
        batch_targets.append(targets)
    return batch_targets


def soft_partition_loss(mask_logits, background_logits, targets, query_ids, target_ids):
    """Soft cross entropy over matched queries plus explicit background occupancy."""
    num_queries, height, width = mask_logits.shape
    target_probs = mask_logits.new_zeros((num_queries + 1, height, width))

    for query_id, target_id in zip(query_ids, target_ids):
        target_probs[int(query_id)] = targets[int(target_id)]

    plane_occupancy = target_probs[:-1].sum(dim=0).clamp(0.0, 1.0)
    target_probs[-1] = 1.0 - plane_occupancy
    normalizer = target_probs.sum(dim=0, keepdim=True).clamp_min(1e-6)
    target_probs = target_probs / normalizer

    class_logits = torch.cat((mask_logits, background_logits), dim=0)
    log_probs = F.log_softmax(class_logits, dim=0)
    return -(target_probs * log_probs).sum(dim=0).mean()


def soft_sample_loss_and_metrics(output, labels, args):
    target_hw = output["mask_logits"].shape[-2:]
    hard_labels = nearest_labels(labels, target_hw)
    batch_targets = soft_area_targets(
        labels,
        target_hw,
        args.num_queries,
        args.min_plane_area_tokens,
    )

    losses = []
    rows = []
    for batch_id in range(output["mask_logits"].shape[0]):
        targets = batch_targets[batch_id]
        query_ids, target_ids = match_queries(
            output["mask_logits"][batch_id],
            targets,
            args,
        )

        existence_target = torch.zeros_like(output["existence_logits"][batch_id])
        if len(query_ids):
            existence_target[
                torch.as_tensor(query_ids, device=existence_target.device)
            ] = 1.0
        existence_loss = F.binary_cross_entropy_with_logits(
            output["existence_logits"][batch_id],
            existence_target,
        )

        if len(query_ids):
            query_tensor = torch.as_tensor(query_ids, device=targets.device)
            target_tensor = torch.as_tensor(target_ids, device=targets.device)
            mask_loss = (
                args.mask_focal_weight
                * focal_bce(
                    output["mask_logits"][batch_id][query_tensor],
                    targets[target_tensor],
                    args.focal_gamma,
                )
                + args.mask_dice_weight
                * dice_loss(
                    output["mask_logits"][batch_id][query_tensor],
                    targets[target_tensor],
                )
            )
        else:
            mask_loss = output["mask_logits"][batch_id].sum() * 0.0

        partition_loss = soft_partition_loss(
            output["mask_logits"][batch_id],
            output["background_logits"][batch_id],
            targets,
            query_ids,
            target_ids,
        )
        loss = (
            mask_loss
            + args.existence_weight * existence_loss
            + args.background_weight * partition_loss
        )
        losses.append(loss)

        row = instance_metrics(
            output["mask_logits"][batch_id],
            output["background_logits"][batch_id],
            output["existence_logits"][batch_id],
            targets,
            query_ids,
            target_ids,
            hard_labels[batch_id],
            args,
        )
        row["loss"] = float(loss.detach())
        row["mask_loss"] = float(mask_loss.detach())
        row["existence_loss"] = float(existence_loss.detach())
        row["partition_loss"] = float(partition_loss.detach())
        row["soft_gt_planes"] = float(targets.shape[0])
        rows.append(row)

    totals = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            totals[key] += float(value)
    return torch.stack(losses).mean(), {
        key: value / max(len(rows), 1) for key, value in totals.items()
    }


def run_epoch(backbone, head, loader, optimizer, device, args, train):
    head.train(train)
    if backbone is not None:
        backbone.eval()
    rows = []

    for step, batch in enumerate(loader, start=1):
        feature1, feature2, gt1, gt2 = cached_or_live_pair(
            batch,
            backbone,
            device,
            args,
            "soft_pair_train" if train else "soft_pair_val",
        )
        if train:
            optimizer.zero_grad(set_to_none=True)

        batch_size = feature1.shape[0]
        output = head(torch.cat((feature1, feature2), dim=0))
        output1 = slice_output(output, 0, batch_size)
        output2 = slice_output(output, batch_size, batch_size * 2)
        loss1, row1 = soft_sample_loss_and_metrics(output1, gt1, args)
        loss2, row2 = soft_sample_loss_and_metrics(output2, gt2, args)
        loss = 0.5 * (loss1 + loss2)
        row = combine_view_rows(row1, row2)
        row["loss"] = float(loss.detach())

        if train:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()

        rows.append(row)
        if step == 1 or step % args.log_every == 0 or step == len(loader):
            print(
                f"[{'Train' if train else 'Val'}] {step}/{len(loader)} "
                f"loss={row['loss']:.4f} iou={row['mean_iou']:.3f} "
                f"v1={row['view1_mean_iou']:.3f} v2={row['view2_mean_iou']:.3f} "
                f"leak={row['leakage_rate']:.3f} "
                f"soft_gt={row['soft_gt_planes']:.2f}",
                flush=True,
            )

    totals = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            totals[key] += float(value)
    return {key: value / max(len(rows), 1) for key, value in totals.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "experiment_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_dust3r_backbone(args.weights_path, device=device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    head = CleanPlaneMaskHead(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.decoder_layers,
        num_heads=args.decoder_heads,
    ).to(device)
    load_initial_checkpoint(head, args.init_checkpoint)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loader = make_loader(
        args,
        "train",
        args.small_train_size,
        args.train_indices,
        shuffle=True,
    )
    val_loader = make_loader(
        args,
        "val",
        args.small_val_size,
        args.val_indices,
        shuffle=False,
    )
    train_loader, val_loader = prepare_pair_data(
        backbone,
        train_loader,
        val_loader,
        device,
        args,
    )

    if args.eval_before_train:
        initial_val = run_epoch(
            backbone,
            head,
            val_loader,
            None,
            device,
            args,
            train=False,
        )
        print(json.dumps({"epoch": 0, "val": initial_val}, ensure_ascii=False))

    history = []
    best_iou = -1.0
    for epoch in range(1, args.num_epochs + 1):
        print(
            f"Epoch {epoch}/{args.num_epochs} "
            f"min_plane_area_tokens={args.min_plane_area_tokens}",
            flush=True,
        )
        train_stats = run_epoch(
            backbone,
            head,
            train_loader,
            optimizer,
            device,
            args,
            train=True,
        )
        val_stats = run_epoch(
            backbone,
            head,
            val_loader,
            None,
            device,
            args,
            train=False,
        )
        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        payload = {
            "head": head.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val": val_stats,
        }
        torch.save(payload, save_dir / "latest.pt")
        if val_stats["mean_iou"] > best_iou:
            best_iou = val_stats["mean_iou"]
            torch.save(payload, save_dir / "best.pt")
        (save_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
