import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from models.multiscale_plane_mask_head import MultiScalePlaneMaskHead
from train_stage1_clean_baseline import sample_loss_and_metrics, set_seed
from train_stage1_clean_pair_baseline import combine_view_rows, slice_output


STAGE_NAMES = ("encoder", "shallow", "middle", "deep")


def parse_args():
    parser = argparse.ArgumentParser(
        "Train a high-resolution multi-layer plane mask head on cached SymRef pairs"
    )
    parser.add_argument("--feature_cache_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_queries", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--output_size", type=int, default=128, choices=(32, 64, 128))
    parser.add_argument("--disable_rgb_skip", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
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
    parser.add_argument("--min_plane_pixels", type=int, default=4)
    parser.add_argument("--aux32_weight", type=float, default=0.25)
    parser.add_argument("--aux64_weight", type=float, default=0.5)
    parser.add_argument("--eval_before_train", action="store_true")
    parser.add_argument("--log_every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260617)
    return parser.parse_args()


def iter_sample_batches(samples, batch_size, shuffle, seed):
    indices = list(range(len(samples)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [samples[index] for index in indices[start : start + batch_size]]


def cat_feature_batch(items, key, device):
    return {
        stage: torch.cat(
            [item[key][stage] for item in items],
            dim=0,
        ).to(device=device, dtype=torch.float32, non_blocking=True)
        for stage in STAGE_NAMES
    }


def cat_tensor_batch(items, key, device, dtype=None):
    value = torch.cat([item[key] for item in items], dim=0).to(
        device=device,
        non_blocking=True,
    )
    return value.to(dtype=dtype) if dtype is not None else value


def primary_output(output):
    return {
        "mask_logits": output["mask_logits"],
        "background_logits": output["background_logits"],
        "existence_logits": output["existence_logits"],
    }


def resize_output(output, target_hw):
    return {
        "mask_logits": F.interpolate(
            output["mask_logits"],
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ),
        "background_logits": F.interpolate(
            output["background_logits"],
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ),
        "existence_logits": output["existence_logits"],
    }


def aux_args(args):
    cloned = copy.copy(args)
    cloned.existence_weight = 0.0
    return cloned


def evaluate_two_views(output, gt1, gt2, args):
    batch_size = gt1.shape[0]
    view1 = slice_output(output, 0, batch_size)
    view2 = slice_output(output, batch_size, batch_size * 2)
    loss1, row1 = sample_loss_and_metrics(view1, gt1, args)
    loss2, row2 = sample_loss_and_metrics(view2, gt2, args)
    return 0.5 * (loss1 + loss2), combine_view_rows(row1, row2)


def initialize_from_clean(head, checkpoint_path):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    clean_state = checkpoint.get("head", checkpoint)
    copied, skipped = head.load_clean_checkpoint(clean_state)

    # The old 32x32 head already learned a useful deep-feature projection.
    # Reuse it for every 768-channel decoder stage with a matching shape.
    old_proj_weight = clean_state.get("feature_proj.weight")
    old_proj_bias = clean_state.get("feature_proj.bias")
    if old_proj_weight is not None:
        for name in ("shallow", "middle", "deep"):
            layer = head.lateral[name]
            if layer.weight.shape == old_proj_weight.shape:
                layer.weight.data.copy_(old_proj_weight)
                if old_proj_bias is not None and layer.bias is not None:
                    layer.bias.data.copy_(old_proj_bias)
                copied.append(f"feature_proj -> lateral.{name}")

    old_bg_weight = clean_state.get("background_head.weight")
    old_bg_bias = clean_state.get("background_head.bias")
    for name in ("background32", "background64", "background128"):
        layer = getattr(head, name)
        if old_bg_weight is not None and layer.weight.shape == old_bg_weight.shape:
            layer.weight.data.copy_(old_bg_weight)
            if old_bg_bias is not None and layer.bias is not None:
                layer.bias.data.copy_(old_bg_bias)
            copied.append(f"background_head -> {name}")

    print(
        f"Warm-started multiscale head from {checkpoint_path}: "
        f"copied={len(copied)}, skipped={len(skipped)}",
        flush=True,
    )


def run_epoch(head, samples, optimizer, device, args, train, epoch):
    head.train(train)
    rows = []
    batches = list(
        iter_sample_batches(
            samples,
            args.batch_size,
            shuffle=train,
            seed=args.seed + epoch,
        )
    )
    auxiliary_args = aux_args(args)

    for step, items in enumerate(batches, start=1):
        features1 = cat_feature_batch(items, "features1", device)
        features2 = cat_feature_batch(items, "features2", device)
        rgb1 = cat_tensor_batch(items, "rgb1", device, torch.float32)
        rgb2 = cat_tensor_batch(items, "rgb2", device, torch.float32)
        gt1 = cat_tensor_batch(items, "gt_plane1", device)
        gt2 = cat_tensor_batch(items, "gt_plane2", device)

        features = {
            stage: torch.cat((features1[stage], features2[stage]), dim=0)
            for stage in STAGE_NAMES
        }
        rgb = torch.cat((rgb1, rgb2), dim=0)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            full_output = head(features, rgb=None if args.disable_rgb_skip else rgb)
            final_output = primary_output(full_output)
            final_loss, row = evaluate_two_views(final_output, gt1, gt2, args)
            loss = final_loss

            for auxiliary in full_output["aux_outputs"]:
                resolution = int(auxiliary["resolution"])
                weight = args.aux32_weight if resolution == 32 else args.aux64_weight
                if weight <= 0:
                    continue
                auxiliary_output = {
                    "mask_logits": auxiliary["mask_logits"],
                    "background_logits": auxiliary["background_logits"],
                    "existence_logits": auxiliary["existence_logits"],
                }
                auxiliary_loss, _ = evaluate_two_views(
                    auxiliary_output,
                    gt1,
                    gt2,
                    auxiliary_args,
                )
                loss = loss + weight * auxiliary_loss
                row[f"aux{resolution}_loss"] = float(auxiliary_loss.detach())

            # Apples-to-apples metric against the old 32x32 clean baseline.
            benchmark32 = resize_output(final_output, (32, 32))
            _, benchmark_row = evaluate_two_views(
                benchmark32,
                gt1,
                gt2,
                auxiliary_args,
            )
            for key, value in benchmark_row.items():
                row[f"benchmark32_{key}"] = float(value)

            stage_weights = full_output["stage_weights"].detach().cpu().tolist()
            for name, value in zip(STAGE_NAMES, stage_weights):
                row[f"stage_weight_{name}"] = float(value)
            row["loss"] = float(loss.detach())
            row["final_loss"] = float(final_loss.detach())

        if train:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()

        rows.append(row)
        if step == 1 or step % args.log_every == 0 or step == len(batches):
            print(
                f"[{'Train' if train else 'Val'}] {step}/{len(batches)} "
                f"loss={row['loss']:.4f} "
                f"iou{args.output_size}={row['mean_iou']:.3f} "
                f"iou32={row['benchmark32_mean_iou']:.3f} "
                f"v1={row['view1_mean_iou']:.3f} "
                f"v2={row['view2_mean_iou']:.3f} "
                f"leak={row['leakage_rate']:.3f}",
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

    cache = torch.load(args.feature_cache_path, map_location="cpu", weights_only=False)
    config = cache.get("config", {})
    input_dims = tuple(int(value) for value in config.get("input_dims", (1024, 768, 768, 768)))
    train_samples = cache["train"]
    val_samples = cache["val"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head = MultiScalePlaneMaskHead(
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.decoder_layers,
        num_heads=args.decoder_heads,
        output_size=args.output_size,
        use_rgb_skip=not args.disable_rgb_skip,
    ).to(device)
    initialize_from_clean(head, args.init_checkpoint)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.num_epochs, 1),
        eta_min=args.min_lr,
    )

    if args.eval_before_train:
        initial = run_epoch(
            head,
            val_samples,
            None,
            device,
            args,
            train=False,
            epoch=0,
        )
        print(json.dumps({"epoch": 0, "val": initial}, ensure_ascii=False), flush=True)

    history = []
    best_iou = -1.0
    for epoch in range(1, args.num_epochs + 1):
        print(
            f"Epoch {epoch}/{args.num_epochs} lr={optimizer.param_groups[0]['lr']:.6g}",
            flush=True,
        )
        train_stats = run_epoch(
            head,
            train_samples,
            optimizer,
            device,
            args,
            train=True,
            epoch=epoch,
        )
        val_stats = run_epoch(
            head,
            val_samples,
            None,
            device,
            args,
            train=False,
            epoch=epoch,
        )
        scheduler.step()

        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        payload = {
            "model_type": "MultiScalePlaneMaskHead",
            "head": head.state_dict(),
            "args": vars(args),
            "cache_config": config,
            "input_dims": list(input_dims),
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
