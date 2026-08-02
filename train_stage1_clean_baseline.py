import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataloaders.s3d_dataset import Structured3DDataset
from models.clean_plane_mask_head import CleanPlaneMaskHead
from train_stage1_plane_masks import (
    build_views,
    feature_maps_from_result,
    masks_for_plane_ids,
    match_queries,
    select_plane_ids,
)


def parse_args():
    parser = argparse.ArgumentParser("Train clean Stage1 plane mask baseline")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--input_mode", default="pair", choices=("pair", "single"))
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--small_train_size", type=int, default=128)
    parser.add_argument("--small_val_size", type=int, default=32)
    parser.add_argument("--fixed_indices", default="")
    parser.add_argument("--disable_feature_cache", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_queries", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--feature_index", type=int, default=12)
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--min_plane_pixels", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--match_bce_weight", type=float, default=1.0)
    parser.add_argument("--match_dice_weight", type=float, default=2.0)
    parser.add_argument("--match_existence_weight", type=float, default=0.0)
    parser.add_argument("--mask_focal_weight", type=float, default=1.0)
    parser.add_argument("--mask_dice_weight", type=float, default=2.0)
    parser.add_argument("--mask_tversky_fp_weight", type=float, default=0.5)
    parser.add_argument("--mask_tversky_fn_weight", type=float, default=0.5)
    parser.add_argument("--existence_weight", type=float, default=1.0)
    parser.add_argument("--existence_positive_weight", type=float, default=1.0)
    parser.add_argument("--existence_negative_weight", type=float, default=1.0)
    parser.add_argument(
        "--class_score_weight",
        type=float,
        default=0.0,
        help="Add query objectness logits to mask logits for partition loss and prediction.",
    )
    parser.add_argument("--background_weight", type=float, default=1.0)
    parser.add_argument("--query_margin_weight", type=float, default=0.0)
    parser.add_argument("--query_margin", type=float, default=0.75)
    parser.add_argument("--unmatched_query_weight", type=float, default=0.0)
    parser.add_argument("--query_separation_weight", type=float, default=0.0)
    parser.add_argument("--query_separation_margin", type=float, default=0.5)
    parser.add_argument("--ownership_loss_weight", type=float, default=0.0)
    parser.add_argument("--ownership_overlap_threshold", type=float, default=0.08)
    parser.add_argument("--ownership_merge_threshold", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--existence_threshold", type=float, default=0.5)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260616)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def parse_indices(text):
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def make_loader(args, split, size, shuffle):
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=split,
        train_ratio=args.train_ratio,
        image_size=(512, 512),
        input_mode=args.input_mode,
    )
    fixed = parse_indices(args.fixed_indices)
    if fixed is not None:
        indices = [index for index in fixed if 0 <= index < len(dataset)]
    else:
        indices = list(range(min(size, len(dataset))))
    return DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def focal_bce(logits, targets, gamma):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = logits.sigmoid()
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    return ((1.0 - pt).pow(gamma) * bce).mean()


def dice_loss(logits, targets, eps=1e-6):
    prob = logits.sigmoid()
    intersection = (prob * targets).sum(dim=(1, 2))
    denominator = prob.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def tversky_loss(logits, targets, fp_weight=0.5, fn_weight=0.5, eps=1e-6):
    prob = logits.sigmoid()
    true_positive = (prob * targets).sum(dim=(1, 2))
    false_positive = (prob * (1.0 - targets)).sum(dim=(1, 2))
    false_negative = ((1.0 - prob) * targets).sum(dim=(1, 2))
    denominator = (
        true_positive
        + float(fp_weight) * false_positive
        + float(fn_weight) * false_negative
    )
    return (1.0 - (true_positive + eps) / (denominator + eps)).mean()


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


def instance_metrics(mask_logits, background_logits, existence_logits, targets, query_ids, target_ids, labels, args):
    scored_masks = apply_query_class_scores(mask_logits, existence_logits, args)
    class_logits = torch.cat((scored_masks, background_logits), dim=0)
    predicted = class_logits.argmax(dim=0)
    gt_query = torch.full_like(predicted, -1)
    ious = []
    for query_id, target_id in zip(query_ids, target_ids):
        gt = targets[target_id] > 0.5
        pred = predicted == int(query_id)
        gt_query[gt] = int(query_id)
        ious.append(float(((pred & gt).sum() / (pred | gt).sum().clamp_min(1)).detach()))
    valid = gt_query >= 0
    leakage = (
        float(((predicted != gt_query) & valid & (predicted < args.num_queries)).sum() / valid.sum().clamp_min(1))
        if valid.any()
        else 0.0
    )
    background = (labels <= 0) | (labels == 255)
    background_error = float(((predicted < args.num_queries) & background).sum() / background.sum().clamp_min(1))
    active = int((existence_logits.sigmoid() > args.existence_threshold).sum())
    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "leakage_rate": leakage,
        "background_error_rate": background_error,
        "gt_planes": len(target_ids),
        "pred_planes": active,
        "plane_count_abs_error": abs(active - len(target_ids)),
    }


def query_deduplication_loss(mask_logits, targets, query_ids, target_ids, args):
    """Penalize multiple queries explaining the same GT plane region.

    Hungarian matching assigns at most one query to each target, but pixel CE can
    be too weak to suppress all competing query logits inside the same plane.
    This term makes the matched query win by a margin on its target pixels and
    optionally suppresses unmatched queries on the union of all GT plane pixels.
    """
    zero = mask_logits.sum() * 0.0
    if len(query_ids) == 0 or targets.numel() == 0:
        return zero, zero

    margin_weight = float(getattr(args, "query_margin_weight", 0.0))
    unmatched_weight = float(getattr(args, "unmatched_query_weight", 0.0))
    if margin_weight <= 0.0 and unmatched_weight <= 0.0:
        return zero, zero

    device = mask_logits.device
    query_ids_t = torch.as_tensor(query_ids, device=device, dtype=torch.long)
    target_ids_t = torch.as_tensor(target_ids, device=device, dtype=torch.long)

    margin_loss = zero
    if margin_weight > 0.0:
        per_target = []
        for query_id, target_id in zip(query_ids_t, target_ids_t):
            target_mask = targets[target_id] > 0.5
            if not target_mask.any():
                continue
            correct = mask_logits[query_id][target_mask]
            wrong_logits = mask_logits[:, target_mask]
            wrong_query_mask = torch.ones(
                mask_logits.shape[0], dtype=torch.bool, device=device
            )
            wrong_query_mask[query_id] = False
            strongest_wrong = wrong_logits[wrong_query_mask].max(dim=0).values
            per_target.append(
                F.relu(float(getattr(args, "query_margin", 0.75)) + strongest_wrong - correct).mean()
            )
        if per_target:
            margin_loss = torch.stack(per_target).mean()

    unmatched_loss = zero
    if unmatched_weight > 0.0:
        matched = torch.zeros(mask_logits.shape[0], dtype=torch.bool, device=device)
        matched[query_ids_t] = True
        unmatched = ~matched
        gt_union = targets[target_ids_t].amax(dim=0) > 0.5
        if unmatched.any() and gt_union.any():
            unmatched_loss = F.softplus(mask_logits[unmatched][:, gt_union]).mean()

    return margin_loss, unmatched_loss


def query_separation_loss(mask_logits, targets, query_ids, target_ids, args):
    """Penalize matched queries leaking into other matched GT plane regions.

    The deduplication loss makes a plane's own query win inside its support,
    which helps split errors but can encourage stronger neighboring merges.
    This complementary term compares every pair of matched planes: on plane A,
    query A should beat query B, and on plane B, query B should beat query A.
    """
    zero = mask_logits.sum() * 0.0
    if (
        float(getattr(args, "query_separation_weight", 0.0)) <= 0.0
        or len(query_ids) < 2
        or targets.numel() == 0
    ):
        return zero

    device = mask_logits.device
    query_ids_t = torch.as_tensor(query_ids, device=device, dtype=torch.long)
    target_ids_t = torch.as_tensor(target_ids, device=device, dtype=torch.long)
    margin = float(getattr(args, "query_separation_margin", 0.5))
    pair_losses = []
    for first in range(len(query_ids_t)):
        q_a = query_ids_t[first]
        t_a = target_ids_t[first]
        mask_a = targets[t_a] > 0.5
        if not mask_a.any():
            continue
        for second in range(first + 1, len(query_ids_t)):
            q_b = query_ids_t[second]
            t_b = target_ids_t[second]
            mask_b = targets[t_b] > 0.5
            if not mask_b.any():
                continue
            pair_losses.append(
                F.relu(margin + mask_logits[q_b][mask_a] - mask_logits[q_a][mask_a]).mean()
            )
            pair_losses.append(
                F.relu(margin + mask_logits[q_a][mask_b] - mask_logits[q_b][mask_b]).mean()
            )
    if not pair_losses:
        return zero
    return torch.stack(pair_losses).mean()


def query_ownership_loss_and_metrics(
    mask_logits,
    background_logits,
    existence_logits,
    targets,
    query_ids,
    target_ids,
    labels,
    args,
):
    """Measure and penalize wrong query ownership.

    This is intentionally target-aware. Hungarian matching gives one query per
    GT plane, but extra queries can still take chunks of that same plane, and a
    matched query can still absorb pixels from neighboring GT planes. These two
    cases are the current split/merge failure mode.
    """
    zero = mask_logits.sum() * 0.0
    num_queries = mask_logits.shape[0]
    metrics = {
        "ownership_duplicate_gt_rate": 0.0,
        "ownership_duplicate_extra_queries": 0.0,
        "ownership_merge_query_rate": 0.0,
        "ownership_false_active_queries": 0.0,
    }
    if targets.numel() == 0:
        active = int((existence_logits.sigmoid() > args.existence_threshold).sum())
        metrics["ownership_false_active_queries"] = float(active)
        return zero, metrics

    scored_masks = apply_query_class_scores(mask_logits, existence_logits, args)
    class_logits = torch.cat((scored_masks, background_logits), dim=0)
    predicted = class_logits.argmax(dim=0)
    overlap_threshold = float(getattr(args, "ownership_overlap_threshold", 0.08))
    merge_threshold = float(getattr(args, "ownership_merge_threshold", 0.25))

    duplicate_planes = 0
    duplicate_extra = 0
    for target in targets:
        gt = target > 0.5
        area = gt.sum().clamp_min(1)
        owners = []
        for query_id in range(num_queries):
            fraction = ((predicted == query_id) & gt).sum().float() / area
            if float(fraction.detach()) >= overlap_threshold:
                owners.append(query_id)
        if len(owners) > 1:
            duplicate_planes += 1
            duplicate_extra += len(owners) - 1
    metrics["ownership_duplicate_gt_rate"] = duplicate_planes / max(int(targets.shape[0]), 1)
    metrics["ownership_duplicate_extra_queries"] = duplicate_extra / max(int(targets.shape[0]), 1)

    gt_plane = (labels > 0) & (labels != 255)
    merge_queries = 0
    active_queries = 0
    for query_id in range(num_queries):
        pred = (predicted == query_id) & gt_plane
        area = pred.sum()
        if int(area.detach()) == 0:
            continue
        active_queries += 1
        gt_values = labels[pred]
        counts = []
        for plane_id in gt_values.unique():
            if int(plane_id.detach()) <= 0 or int(plane_id.detach()) == 255:
                continue
            counts.append((gt_values == plane_id).sum().float())
        if len(counts) >= 2:
            counts = torch.stack(counts).sort(descending=True).values
            if float((counts[1] / counts.sum().clamp_min(1)).detach()) >= merge_threshold:
                merge_queries += 1
    metrics["ownership_merge_query_rate"] = merge_queries / max(active_queries, 1)

    matched = set(int(query_id) for query_id in query_ids)
    active = {
        int(query_id)
        for query_id in torch.nonzero(
            existence_logits.sigmoid() > args.existence_threshold,
            as_tuple=False,
        ).flatten().tolist()
    }
    metrics["ownership_false_active_queries"] = float(len(active - matched))

    if float(getattr(args, "ownership_loss_weight", 0.0)) <= 0.0 or len(query_ids) == 0:
        return zero, metrics

    device = mask_logits.device
    query_ids_t = torch.as_tensor(query_ids, device=device, dtype=torch.long)
    target_ids_t = torch.as_tensor(target_ids, device=device, dtype=torch.long)
    gt_union = targets[target_ids_t].amax(dim=0) > 0.5
    terms = []
    for query_id, target_id in zip(query_ids_t, target_ids_t):
        target_mask = targets[target_id] > 0.5
        if not target_mask.any():
            continue
        other = torch.ones(num_queries, dtype=torch.bool, device=device)
        other[query_id] = False
        terms.append(F.softplus(mask_logits[other][:, target_mask]).mean())
        outside = gt_union & (~target_mask)
        if outside.any():
            terms.append(F.softplus(mask_logits[query_id][outside]).mean())
    if not terms:
        return zero, metrics
    return torch.stack(terms).mean(), metrics


def sample_loss_and_metrics(output, labels, args):
    _, plane_ids = select_plane_ids(
        labels,
        output["mask_logits"].shape[-2:],
        args.num_queries,
        args.min_plane_pixels,
    )
    labels32, masks = masks_for_plane_ids(
        labels,
        output["mask_logits"].shape[-2:],
        plane_ids,
    )
    losses = []
    rows = []
    for batch_id in range(output["mask_logits"].shape[0]):
        targets = torch.stack(masks[batch_id]) if masks[batch_id] else output["mask_logits"].new_zeros((0, *output["mask_logits"].shape[-2:]))
        query_ids, target_ids = match_queries(
            output["mask_logits"][batch_id],
            targets,
            args,
            existence_logits=output["existence_logits"][batch_id],
        )
        existence_target = torch.zeros_like(output["existence_logits"][batch_id])
        if len(query_ids):
            existence_target[torch.as_tensor(query_ids, device=existence_target.device)] = 1.0
        existence_raw = F.binary_cross_entropy_with_logits(
            output["existence_logits"][batch_id],
            existence_target,
            reduction="none",
        )
        existence_weight = torch.where(
            existence_target > 0.5,
            existence_target.new_full((), float(getattr(args, "existence_positive_weight", 1.0))),
            existence_target.new_full((), float(getattr(args, "existence_negative_weight", 1.0))),
        )
        existence = (existence_raw * existence_weight).sum() / existence_weight.sum().clamp_min(1e-6)
        if len(query_ids):
            q = torch.as_tensor(query_ids, device=targets.device)
            t = torch.as_tensor(target_ids, device=targets.device)
            shape_loss = tversky_loss(
                output["mask_logits"][batch_id][q],
                targets[t],
                fp_weight=float(getattr(args, "mask_tversky_fp_weight", 0.5)),
                fn_weight=float(getattr(args, "mask_tversky_fn_weight", 0.5)),
            )
            mask_loss = (
                args.mask_focal_weight * focal_bce(output["mask_logits"][batch_id][q], targets[t], args.focal_gamma)
                + args.mask_dice_weight * shape_loss
            )
        else:
            mask_loss = output["mask_logits"][batch_id].sum() * 0.0
        class_target = class_target_from_matches(
            labels32[batch_id],
            targets,
            query_ids,
            target_ids,
            args.num_queries,
        )
        background = F.cross_entropy(
            torch.cat(
                (
                    apply_query_class_scores(
                        output["mask_logits"][batch_id],
                        output["existence_logits"][batch_id],
                        args,
                    ),
                    output["background_logits"][batch_id],
                ),
                dim=0,
            )[None],
            class_target[None],
        )
        query_margin, unmatched_query = query_deduplication_loss(
            output["mask_logits"][batch_id],
            targets,
            query_ids,
            target_ids,
            args,
        )
        query_separation = query_separation_loss(
            output["mask_logits"][batch_id],
            targets,
            query_ids,
            target_ids,
            args,
        )
        ownership_loss, ownership_metrics = query_ownership_loss_and_metrics(
            output["mask_logits"][batch_id],
            output["background_logits"][batch_id],
            output["existence_logits"][batch_id],
            targets,
            query_ids,
            target_ids,
            labels32[batch_id],
            args,
        )
        loss = (
            mask_loss
            + args.existence_weight * existence
            + args.background_weight * background
            + float(getattr(args, "query_margin_weight", 0.0)) * query_margin
            + float(getattr(args, "unmatched_query_weight", 0.0)) * unmatched_query
            + float(getattr(args, "query_separation_weight", 0.0)) * query_separation
            + float(getattr(args, "ownership_loss_weight", 0.0)) * ownership_loss
        )
        losses.append(loss)
        row = instance_metrics(
            output["mask_logits"][batch_id],
            output["background_logits"][batch_id],
            output["existence_logits"][batch_id],
            targets,
            query_ids,
            target_ids,
            labels32[batch_id],
            args,
        )
        row["loss"] = float(loss.detach())
        row["existence_loss"] = float(existence.detach())
        row["query_margin_loss"] = float(query_margin.detach())
        row["unmatched_query_loss"] = float(unmatched_query.detach())
        row["query_separation_loss"] = float(query_separation.detach())
        row["ownership_loss"] = float(ownership_loss.detach())
        row.update(ownership_metrics)
        rows.append(row)
    mean_loss = torch.stack(losses).mean()
    totals = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            totals[key] += float(value)
    return mean_loss, {key: value / len(rows) for key, value in totals.items()}


@torch.no_grad()
def cache_features(backbone, loader, device, args):
    backbone.eval()
    cached = []
    for batch in loader:
        batch = move_batch(batch, device)
        view1, view2 = build_views(batch, "clean_cache")
        result1, _ = backbone(view1, view2)
        features = feature_maps_from_result(result1, view1["img"], (0, 6, 9, args.feature_index))
        cached.append({
            "feature": features["deep"].detach().cpu(),
            "gt_plane": batch.get("gt_plane1", batch["gt_plane"]).detach().cpu(),
        })
    return cached


def run_epoch(backbone, head, loader, optimizer, device, args, train):
    head.train(train)
    if backbone is not None:
        backbone.eval()
    rows = []
    for step, batch in enumerate(loader, start=1):
        if isinstance(batch, dict) and "feature" in batch:
            feature = batch["feature"].to(device)
            gt_plane = batch["gt_plane"].to(device)
        else:
            batch = move_batch(batch, device)
            view1, view2 = build_views(batch, "clean_baseline")
            with torch.no_grad():
                result1, _ = backbone(view1, view2)
                features = feature_maps_from_result(result1, view1["img"], (0, 6, 9, args.feature_index))
            feature = features["deep"]
            gt_plane = batch.get("gt_plane1", batch["gt_plane"])
        if train:
            optimizer.zero_grad(set_to_none=True)
        output = head(feature)
        loss, row = sample_loss_and_metrics(output, gt_plane, args)
        if train:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()
        rows.append(row)
        if step % args.log_every == 0 or step == len(loader):
            print(
                f"[{'Train' if train else 'Val'}] {step}/{len(loader)} "
                f"loss={row['loss']:.4f} iou={row['mean_iou']:.3f} "
                f"count_err={row['plane_count_abs_error']:.2f}",
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models.build_backbone import build_dust3r_backbone

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
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(args, "train", args.small_train_size, shuffle=True)
    val_loader = make_loader(args, "val", args.small_val_size, shuffle=False)
    if not args.disable_feature_cache:
        print("Caching DUSt3R features for clean baseline...", flush=True)
        train_loader = cache_features(backbone, train_loader, device, args)
        val_loader = cache_features(backbone, val_loader, device, args)
    history = []
    best_iou = -1.0
    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}", flush=True)
        train_stats = run_epoch(backbone, head, train_loader, optimizer, device, args, train=True)
        val_stats = run_epoch(backbone, head, val_loader, optimizer, device, args, train=False)
        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"head": head.state_dict(), "args": vars(args), "epoch": epoch}, save_dir / "latest.pt")
        if val_stats["mean_iou"] > best_iou:
            best_iou = val_stats["mean_iou"]
            torch.save({"head": head.state_dict(), "args": vars(args), "epoch": epoch}, save_dir / "best.pt")
        (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
