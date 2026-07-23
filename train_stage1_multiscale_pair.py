import argparse
import copy
import gc
import glob
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.multiscale_plane_mask_head import MultiScalePlaneMaskHead
from stage1_fast_assignment import solve_rectangular_assignment
import train_stage1_clean_baseline as clean_stage1_baseline
from train_stage1_clean_baseline import sample_loss_and_metrics, set_seed
from train_stage1_clean_pair_baseline import combine_view_rows, slice_output
from train_stage1_plane_masks import select_plane_ids


STAGE_NAMES = ("encoder", "shallow", "middle", "deep")


def fast_match_queries(mask_logits, targets, args, existence_logits=None):
    """Exact polynomial-time replacement for the exhaustive subset matcher."""

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
    intersection = torch.einsum("qhw,thw->qt", pred_prob, targets)
    denominator = pred_prob.sum(dim=(1, 2))[:, None] + targets.sum(dim=(1, 2))[None]
    dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    cost = args.match_bce_weight * bce + args.match_dice_weight * dice
    existence_weight = float(getattr(args, "match_existence_weight", 0.0))
    if existence_logits is not None and existence_weight > 0.0:
        existence_cost = F.binary_cross_entropy_with_logits(
            existence_logits,
            torch.ones_like(existence_logits),
            reduction="none",
        )
        cost = cost + existence_weight * existence_cost[:, None]
    return solve_rectangular_assignment(cost.detach().float().cpu().numpy())


# sample_loss_and_metrics resolves match_queries from its defining module at
# runtime.  Patch only this training entry point so other user experiments keep
# their existing behavior until they opt in explicitly.
clean_stage1_baseline.match_queries = fast_match_queries


def parse_args():
    parser = argparse.ArgumentParser(
        "Train a high-resolution multi-layer plane mask head on cached SymRef pairs"
    )
    parser.add_argument("--feature_cache_path", default="")
    parser.add_argument(
        "--feature_cache_glob",
        default="",
        help="Optional glob for sharded train caches. Each shard is loaded and released per epoch.",
    )
    parser.add_argument(
        "--val_cache_path",
        default="",
        help="Optional validation cache path when training from sharded train caches.",
    )
    parser.add_argument("--save_dir", required=True)
    parser.add_argument(
        "--train_quality_indices",
        default="",
        help="Optional cache-index list for filtering train samples, one index per line or comma-separated.",
    )
    parser.add_argument(
        "--train_quality_index_dir",
        default="",
        help=(
            "Optional directory containing per-shard cache-index lists named "
            "<shard_stem>_cache_indices.txt. This is preferred for sharded caches "
            "because --train_quality_indices is interpreted relative to each loaded cache."
        ),
    )
    parser.add_argument(
        "--val_quality_indices",
        default="",
        help="Optional cache-index list for filtering val samples, one index per line or comma-separated.",
    )
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument(
        "--resume_checkpoint",
        default="",
        help="Strictly resume a MultiScalePlaneMaskHead checkpoint for finetuning.",
    )
    parser.add_argument(
        "--allow_query_expansion_resume",
        action="store_true",
        help="Allow loading a checkpoint with fewer queries by copying overlapping query rows.",
    )
    parser.add_argument(
        "--allow_partial_resume",
        action="store_true",
        help=(
            "Allow warm-starting a larger head by copying overlapping tensor "
            "slices from a smaller checkpoint. New channels/layers stay randomly initialized."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_queries", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--decoder_ffn_multiplier", type=int, default=4)
    parser.add_argument("--fuse_refine_blocks", type=int, default=1)
    parser.add_argument("--pixel_refine_blocks", type=int, default=1)
    parser.add_argument("--output_size", type=int, default=128, choices=(32, 64, 128))
    parser.add_argument("--disable_rgb_skip", action="store_true")
    parser.add_argument(
        "--use_geometry",
        action="store_true",
        help="Fuse cached DUSt3R pointmap-derived geometry into the 128x128 decoder.",
    )
    parser.add_argument(
        "--use_masked_query_refine",
        action="store_true",
        help=(
            "Run a zero-initialized second query pass using mask-pooled pixel "
            "features. This should preserve old checkpoints at step 0."
        ),
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--match_bce_weight", type=float, default=1.0)
    parser.add_argument("--match_dice_weight", type=float, default=2.0)
    parser.add_argument(
        "--match_existence_weight",
        type=float,
        default=0.0,
        help="Add plane-existence probability to Hungarian matching cost.",
    )
    parser.add_argument("--mask_focal_weight", type=float, default=1.0)
    parser.add_argument("--mask_dice_weight", type=float, default=2.0)
    parser.add_argument("--mask_tversky_fp_weight", type=float, default=0.5)
    parser.add_argument("--mask_tversky_fn_weight", type=float, default=0.5)
    parser.add_argument("--existence_weight", type=float, default=0.1)
    parser.add_argument("--existence_positive_weight", type=float, default=1.0)
    parser.add_argument("--existence_negative_weight", type=float, default=1.0)
    parser.add_argument(
        "--class_score_weight",
        type=float,
        default=0.0,
        help="Add query objectness logits to mask logits for partition loss and prediction.",
    )
    parser.add_argument("--background_weight", type=float, default=1.0)
    parser.add_argument(
        "--query_margin_weight",
        type=float,
        default=0.0,
        help=(
            "Penalize competing query logits inside the matched GT plane. "
            "This targets query over-splitting without changing matching."
        ),
    )
    parser.add_argument("--query_margin", type=float, default=0.75)
    parser.add_argument(
        "--unmatched_query_weight",
        type=float,
        default=0.0,
        help="Suppress unmatched query logits on the union of GT plane pixels.",
    )
    parser.add_argument(
        "--query_separation_weight",
        type=float,
        default=0.0,
        help=(
            "Penalize matched plane queries leaking into each other's GT region. "
            "This targets query merging without changing the architecture."
        ),
    )
    parser.add_argument("--query_separation_margin", type=float, default=0.5)
    parser.add_argument(
        "--ownership_loss_weight",
        type=float,
        default=0.0,
        help="Target-aware loss for duplicate-query splits and multi-GT query merges.",
    )
    parser.add_argument("--ownership_overlap_threshold", type=float, default=0.08)
    parser.add_argument("--ownership_merge_threshold", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--existence_threshold", type=float, default=0.5)
    parser.add_argument("--min_plane_pixels", type=int, default=4)
    parser.add_argument("--aux32_weight", type=float, default=0.25)
    parser.add_argument("--aux64_weight", type=float, default=0.5)
    parser.add_argument(
        "--train_hflip_prob",
        type=float,
        default=0.0,
        help="Randomly flip cached features/RGB/geometry/GT horizontally during training.",
    )
    parser.add_argument(
        "--rgb_jitter_strength",
        type=float,
        default=0.0,
        help="Apply lightweight brightness/contrast jitter to RGB skip input during training.",
    )
    parser.add_argument(
        "--hard_case_sampling",
        action="store_true",
        help="Oversample cached samples with more planes, small planes, and imbalanced plane areas.",
    )
    parser.add_argument(
        "--hard_case_strength",
        type=float,
        default=1.0,
        help="Multiplier for hard-case oversampling. Weights are capped to avoid collapse.",
    )
    parser.add_argument("--eval_before_train", action="store_true")
    parser.add_argument("--log_every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260617)
    return parser.parse_args()


def iter_sample_batches(samples, batch_size, shuffle, seed, weights=None):
    indices = list(range(len(samples)))
    if shuffle and weights is not None:
        rng = random.Random(seed)
        indices = rng.choices(indices, weights=weights, k=len(indices))
    elif shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [samples[index] for index in indices[start : start + batch_size]]


def _view_hard_case_score(labels, args):
    target_hw = (int(args.output_size), int(args.output_size))
    resized, batch_plane_ids = select_plane_ids(
        labels,
        target_hw,
        args.num_queries,
        args.min_plane_pixels,
    )
    if not batch_plane_ids or not batch_plane_ids[0]:
        return 0.0
    label_map = resized[0]
    total = float(label_map.numel())
    areas = []
    for plane_id in batch_plane_ids[0]:
        areas.append(float((label_map == int(plane_id)).sum().item()) / total)
    plane_count = len(areas)
    small_count = sum(area < 0.035 for area in areas)
    imbalance = max(areas) / max(min(areas), 1.0 / total) if len(areas) >= 2 else 1.0
    many_planes_score = max(0.0, float(plane_count - 3)) / 5.0
    small_plane_score = min(1.0, float(small_count) / 3.0)
    imbalance_score = min(1.0, max(0.0, imbalance - 4.0) / 8.0)
    return 0.45 * many_planes_score + 0.35 * small_plane_score + 0.20 * imbalance_score


def build_hard_case_weights(samples, args):
    if not args.hard_case_sampling:
        return None
    weights = []
    for sample in samples:
        view1 = _view_hard_case_score(sample["gt_plane1"], args)
        view2 = _view_hard_case_score(sample["gt_plane2"], args)
        score = 0.5 * (view1 + view2)
        weight = 1.0 + float(args.hard_case_strength) * score
        weights.append(min(5.0, max(0.2, weight)))
    mean_weight = sum(weights) / max(len(weights), 1)
    print(
        "[hard-case sampling] "
        f"enabled strength={args.hard_case_strength:g} "
        f"min={min(weights):.3f} mean={mean_weight:.3f} max={max(weights):.3f}",
        flush=True,
    )
    return weights


def load_cache_index_filter(path_or_text):
    if not path_or_text:
        return None
    path = Path(path_or_text)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = path_or_text
    values = []
    for part in text.replace(",", "\n").splitlines():
        part = part.strip()
        if part:
            values.append(int(part))
    return set(values)


def filter_cached_samples(samples, index_filter, split_name):
    if index_filter is None:
        return samples
    filtered = [sample for index, sample in enumerate(samples) if index in index_filter]
    if not filtered:
        raise RuntimeError(f"{split_name} quality filter removed all cached samples")
    print(
        f"[{split_name} quality filter] kept {len(filtered)}/{len(samples)} samples",
        flush=True,
    )
    return filtered


def shard_quality_index_path(index_dir, shard_path):
    if not index_dir:
        return ""
    return str(Path(index_dir) / f"{Path(shard_path).stem}_cache_indices.txt")


def load_shard_quality_index_filter(index_dir, shard_path):
    path = shard_quality_index_path(index_dir, shard_path)
    if not path:
        return None
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing per-shard quality index file for {shard_path}: {path}"
        )
    return load_cache_index_filter(path)


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


def cat_optional_tensor_batch(items, key, device, dtype=None):
    if key not in items[0]:
        return None
    return cat_tensor_batch(items, key, device, dtype)


def apply_cached_augmentation(features, rgb, geometry, gt, args, train):
    if not train:
        return features, rgb, geometry, gt

    hflip_prob = float(getattr(args, "train_hflip_prob", 0.0))
    if hflip_prob > 0.0:
        batch_size = rgb.shape[0]
        flip_mask = torch.rand(batch_size, device=rgb.device) < hflip_prob
        if flip_mask.any():
            for stage in STAGE_NAMES:
                features[stage][flip_mask] = torch.flip(
                    features[stage][flip_mask],
                    dims=(-1,),
                )
            rgb[flip_mask] = torch.flip(rgb[flip_mask], dims=(-1,))
            gt[flip_mask] = torch.flip(gt[flip_mask], dims=(-1,))
            if geometry is not None:
                geometry[flip_mask] = torch.flip(geometry[flip_mask], dims=(-1,))

    jitter = float(getattr(args, "rgb_jitter_strength", 0.0))
    if jitter > 0.0:
        batch_size = rgb.shape[0]
        contrast = 1.0 + (torch.rand(batch_size, 1, 1, 1, device=rgb.device) * 2.0 - 1.0) * jitter
        brightness = (torch.rand(batch_size, 1, 1, 1, device=rgb.device) * 2.0 - 1.0) * jitter
        rgb = (rgb * contrast + brightness).clamp(0.0, 1.0)

    return features, rgb, geometry, gt


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
    cloned.query_margin_weight = 0.0
    cloned.unmatched_query_weight = 0.0
    cloned.query_separation_weight = 0.0
    cloned.ownership_loss_weight = 0.0
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


def _copy_overlapping_tensor(target, source):
    slices = tuple(slice(0, min(dst, src)) for dst, src in zip(target.shape, source.shape))
    copied = target.clone()
    copied[slices].copy_(source[slices])
    return copied


def resume_multiscale_checkpoint(
    head,
    checkpoint_path,
    allow_query_expansion=False,
    allow_partial_resume=False,
):
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("head", checkpoint)
    if allow_query_expansion or allow_partial_resume:
        own_state = head.state_dict()
        copied = []
        expanded = []
        skipped = []
        query_prefixes = ("query_embed.", "existence_head.")
        for key, value in state.items():
            if key not in own_state:
                skipped.append(key)
                continue
            if own_state[key].shape == value.shape:
                own_state[key] = value
                copied.append(key)
            elif key.startswith(query_prefixes) and own_state[key].ndim == value.ndim:
                own_state[key] = _copy_overlapping_tensor(own_state[key], value)
                expanded.append(key)
            elif allow_partial_resume and own_state[key].ndim == value.ndim:
                own_state[key] = _copy_overlapping_tensor(own_state[key], value)
                expanded.append(key)
            else:
                skipped.append(key)
        head.load_state_dict(own_state, strict=True)
        print(
            f"Partially-resumed MultiScalePlaneMaskHead from {checkpoint_path} "
            f"(epoch={checkpoint.get('epoch', 'unknown')}, "
            f"copied={len(copied)}, expanded={len(expanded)}, skipped={len(skipped)})",
            flush=True,
        )
        return checkpoint
    head.load_state_dict(state, strict=True)
    print(
        f"Resumed MultiScalePlaneMaskHead from {checkpoint_path} "
        f"(epoch={checkpoint.get('epoch', 'unknown')})",
        flush=True,
    )
    return checkpoint


def run_epoch(head, samples, optimizer, device, args, train, epoch, sample_weights=None):
    head.train(train)
    rows = []
    batches = list(
        iter_sample_batches(
            samples,
            args.batch_size,
            shuffle=train,
            seed=args.seed + epoch,
            weights=sample_weights if train else None,
        )
    )
    auxiliary_args = aux_args(args)
    epoch_started = time.perf_counter()
    interval_started = epoch_started
    interval_start_step = 0

    for step, items in enumerate(batches, start=1):
        features1 = cat_feature_batch(items, "features1", device)
        features2 = cat_feature_batch(items, "features2", device)
        rgb1 = cat_tensor_batch(items, "rgb1", device, torch.float32)
        rgb2 = cat_tensor_batch(items, "rgb2", device, torch.float32)
        geometry1 = cat_optional_tensor_batch(items, "geometry1", device, torch.float32)
        geometry2 = cat_optional_tensor_batch(items, "geometry2", device, torch.float32)
        gt1 = cat_tensor_batch(items, "gt_plane1", device)
        gt2 = cat_tensor_batch(items, "gt_plane2", device)

        features = {
            stage: torch.cat((features1[stage], features2[stage]), dim=0)
            for stage in STAGE_NAMES
        }
        rgb = torch.cat((rgb1, rgb2), dim=0)
        if args.use_geometry:
            if geometry1 is None or geometry2 is None:
                raise RuntimeError(
                    "--use_geometry requires a cache generated with geometry1/geometry2"
                )
            geometry = torch.cat((geometry1, geometry2), dim=0)
        else:
            geometry = None
        gt = torch.cat((gt1, gt2), dim=0)
        features, rgb, geometry, gt = apply_cached_augmentation(
            features,
            rgb,
            geometry,
            gt,
            args,
            train,
        )
        gt1 = gt[: gt1.shape[0]]
        gt2 = gt[gt1.shape[0] :]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            full_output = head(
                features,
                rgb=None if args.disable_rgb_skip else rgb,
                geometry=geometry,
            )
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
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            interval_steps = step - interval_start_step
            step_seconds = (now - interval_started) / max(interval_steps, 1)
            eta_minutes = step_seconds * (len(batches) - step) / 60.0
            print(
                f"[{'Train' if train else 'Val'}] {step}/{len(batches)} "
                f"loss={row['loss']:.4f} "
                f"iou{args.output_size}={row['mean_iou']:.3f} "
                f"iou32={row['benchmark32_mean_iou']:.3f} "
                f"v1={row['view1_mean_iou']:.3f} "
                f"v2={row['view2_mean_iou']:.3f} "
                f"leak={row['leakage_rate']:.3f} "
                f"dup={row.get('ownership_duplicate_gt_rate', 0.0):.3f} "
                f"merge={row.get('ownership_merge_query_rate', 0.0):.3f} "
                f"step_s={step_seconds:.2f} eta_min={eta_minutes:.1f}",
                flush=True,
            )
            interval_started = now
            interval_start_step = step

    totals = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            totals[key] += float(value)
    elapsed = time.perf_counter() - epoch_started
    print(
        f"[{'Train' if train else 'Val'} timing] steps={len(batches)} "
        f"elapsed_min={elapsed / 60.0:.1f} "
        f"mean_step_s={elapsed / max(len(batches), 1):.2f}",
        flush=True,
    )
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

    train_cache_paths = []
    if args.feature_cache_glob:
        train_cache_paths = sorted(glob.glob(args.feature_cache_glob))
        if not train_cache_paths:
            raise FileNotFoundError(f"No train cache shards matched: {args.feature_cache_glob}")
        if not args.val_cache_path:
            raise ValueError("--val_cache_path is required when using --feature_cache_glob")
        cache_for_config = torch.load(train_cache_paths[0], map_location="cpu", weights_only=False)
        val_cache_path = args.val_cache_path
        val_cache = torch.load(val_cache_path, map_location="cpu", weights_only=False)
        print(
            json.dumps(
                {
                    "sharded_training": {
                        "train_cache_glob": args.feature_cache_glob,
                        "train_shards": len(train_cache_paths),
                        "val_cache_path": val_cache_path,
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        if not args.feature_cache_path:
            raise ValueError("Provide --feature_cache_path or --feature_cache_glob")
        cache_for_config = torch.load(args.feature_cache_path, map_location="cpu", weights_only=False)
        val_cache = cache_for_config

    cache = cache_for_config
    config = cache.get("config", {})
    input_dims = tuple(int(value) for value in config.get("input_dims", (1024, 768, 768, 768)))
    if not train_cache_paths:
        train_samples = cache["train"]
        train_samples = filter_cached_samples(
            train_samples,
            load_cache_index_filter(args.train_quality_indices),
            "train",
        )
        train_sample_weights = build_hard_case_weights(train_samples, args)
    else:
        train_samples = None
        train_sample_weights = None
    val_samples = filter_cached_samples(
        val_cache["val"],
        load_cache_index_filter(args.val_quality_indices),
        "val",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head = MultiScalePlaneMaskHead(
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.decoder_layers,
        num_heads=args.decoder_heads,
        output_size=args.output_size,
        use_rgb_skip=not args.disable_rgb_skip,
        use_geometry=args.use_geometry,
        geometry_dim=int(config.get("geometry_channels", 9)),
        use_masked_query_refine=args.use_masked_query_refine,
        decoder_ffn_multiplier=args.decoder_ffn_multiplier,
        fuse_refine_blocks=args.fuse_refine_blocks,
        pixel_refine_blocks=args.pixel_refine_blocks,
    ).to(device)
    trainable_params = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in head.parameters())
    print(
        json.dumps(
            {
                "model_params": {
                    "total": int(total_params),
                    "trainable": int(trainable_params),
                    "hidden_dim": int(args.hidden_dim),
                    "num_queries": int(args.num_queries),
                    "decoder_layers": int(args.decoder_layers),
                    "decoder_ffn_multiplier": int(args.decoder_ffn_multiplier),
                    "fuse_refine_blocks": int(args.fuse_refine_blocks),
                    "pixel_refine_blocks": int(args.pixel_refine_blocks),
                }
            }
        ),
        flush=True,
    )
    initialize_from_clean(head, args.init_checkpoint)
    resume_multiscale_checkpoint(
        head,
        args.resume_checkpoint,
        allow_query_expansion=args.allow_query_expansion_resume,
        allow_partial_resume=args.allow_partial_resume,
    )

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

    initial = None
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
    best_iou = float(initial["mean_iou"]) if initial is not None else -1.0
    if initial is not None:
        payload = {
            "model_type": "MultiScalePlaneMaskHead",
            "head": head.state_dict(),
            "args": vars(args),
            "cache_config": config,
            "input_dims": list(input_dims),
            "epoch": 0,
            "val": initial,
        }
        torch.save(payload, save_dir / "best.pt")
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
            sample_weights=train_sample_weights,
        ) if not train_cache_paths else None
        if train_cache_paths:
            shard_stats = []
            for shard_index, shard_path in enumerate(train_cache_paths, start=1):
                print(
                    f"[Shard] epoch={epoch} {shard_index}/{len(train_cache_paths)} path={shard_path}",
                    flush=True,
                )
                shard_cache = torch.load(shard_path, map_location="cpu", weights_only=False)
                shard_index_filter = (
                    load_shard_quality_index_filter(args.train_quality_index_dir, shard_path)
                    if args.train_quality_index_dir
                    else load_cache_index_filter(args.train_quality_indices)
                )
                shard_samples = filter_cached_samples(
                    shard_cache["train"],
                    shard_index_filter,
                    "train",
                )
                shard_weights = build_hard_case_weights(shard_samples, args)
                shard_stats.append(
                    run_epoch(
                        head,
                        shard_samples,
                        optimizer,
                        device,
                        args,
                        train=True,
                        epoch=epoch,
                        sample_weights=shard_weights,
                    )
                )
                del shard_cache, shard_samples, shard_weights
                gc.collect()
            train_stats = {
                key: sum(float(row.get(key, 0.0)) for row in shard_stats) / max(len(shard_stats), 1)
                for key in shard_stats[0].keys()
            }
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
