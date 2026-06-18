import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataloaders.s3d_dataset import Structured3DDataset
from models.clean_plane_mask_head import CleanPlaneMaskHead
from train_stage1_clean_baseline import sample_loss_and_metrics, set_seed
from train_stage1_plane_masks import build_views, feature_maps_from_result


def parse_args():
    parser = argparse.ArgumentParser(
        "Train the clean Stage1 head with real Structured3D image pairs"
    )
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument(
        "--supervision",
        default="both",
        choices=("view1", "both"),
        help=(
            "view1: pair-conditioned view1-only training; "
            "both: one shared head is supervised on both pair views"
        ),
    )
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
    parser.add_argument("--min_plane_pixels", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--match_bce_weight", type=float, default=1.0)
    parser.add_argument("--match_dice_weight", type=float, default=2.0)
    parser.add_argument("--mask_focal_weight", type=float, default=1.0)
    parser.add_argument("--mask_dice_weight", type=float, default=2.0)
    parser.add_argument("--existence_weight", type=float, default=1.0)
    parser.add_argument("--background_weight", type=float, default=1.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--existence_threshold", type=float, default=0.5)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--eval_before_train", action="store_true")
    parser.add_argument(
        "--feature_cache_path",
        default="",
        help=(
            "Optional shared .pt cache. Pair features are stored in float16 on CPU; "
            "the same cache can be reused by view1 and both-supervision runs."
        ),
    )
    parser.add_argument("--disable_feature_cache", action="store_true")
    parser.add_argument("--log_every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260616)
    return parser.parse_args()


def parse_indices(text):
    if not text.strip():
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def make_loader(args, split, size, indices_text, shuffle):
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=split,
        train_ratio=args.train_ratio,
        image_size=(512, 512),
        input_mode="pair",
        pair_strategy=args.pair_strategy,
        pair_max_view_id_gap=args.pair_max_view_id_gap,
    )
    explicit = parse_indices(indices_text)
    if explicit is None:
        indices = list(range(min(size, len(dataset))))
    else:
        indices = [index for index in explicit if 0 <= index < len(dataset)]
    if not indices:
        raise RuntimeError(f"No valid {split} pair indices were selected")
    return DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def cache_config(args):
    return {
        "root_dir": str(Path(args.root_dir).resolve()),
        "train_ratio": float(args.train_ratio),
        "small_train_size": int(args.small_train_size),
        "small_val_size": int(args.small_val_size),
        "train_indices": args.train_indices,
        "val_indices": args.val_indices,
        "pair_strategy": args.pair_strategy,
        "pair_max_view_id_gap": args.pair_max_view_id_gap,
        "feature_index": int(args.feature_index),
        "feature_dim": int(args.feature_dim),
        "image_size": 512,
    }


def validate_cache_config(payload, args):
    expected = cache_config(args)
    actual = payload.get("config", {})
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Feature cache configuration does not match this experiment:\n"
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )


@torch.no_grad()
def cache_pair_features(backbone, loader, device, args, split):
    backbone.eval()
    cached = []
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        view1, view2 = build_views(batch, f"clean_pair_cache_{split}")
        result1, result2 = backbone(view1, view2)
        features1 = feature_maps_from_result(
            result1,
            view1["img"],
            (0, 6, 9, args.feature_index),
        )
        features2 = feature_maps_from_result(
            result2,
            view2["img"],
            (0, 6, 9, args.feature_index),
        )
        cached.append(
            {
                "feature1": features1["deep"].detach().half().cpu(),
                "feature2": features2["deep"].detach().half().cpu(),
                "gt_plane1": batch["gt_plane1"].detach().cpu(),
                "gt_plane2": batch["gt_plane2"].detach().cpu(),
            }
        )
        if step == 1 or step % args.log_every == 0 or step == len(loader):
            print(f"[Cache {split}] {step}/{len(loader)}", flush=True)
    return cached


def prepare_pair_data(backbone, train_loader, val_loader, device, args):
    if args.disable_feature_cache:
        return train_loader, val_loader

    cache_path = Path(args.feature_cache_path) if args.feature_cache_path else None
    if cache_path is not None and cache_path.exists():
        print(f"Loading shared pair feature cache: {cache_path}", flush=True)
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        validate_cache_config(payload, args)
        return payload["train"], payload["val"]

    print("Caching real-pair DUSt3R features for both views...", flush=True)
    train_cached = cache_pair_features(backbone, train_loader, device, args, "train")
    val_cached = cache_pair_features(backbone, val_loader, device, args, "val")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": cache_config(args),
                "train": train_cached,
                "val": val_cached,
            },
            cache_path,
        )
        print(f"Saved shared pair feature cache: {cache_path}", flush=True)
    return train_cached, val_cached


def cached_or_live_pair(batch, backbone, device, args, prefix):
    if isinstance(batch, dict) and "feature1" in batch:
        return (
            batch["feature1"].to(device=device, dtype=torch.float32),
            batch["feature2"].to(device=device, dtype=torch.float32),
            batch["gt_plane1"].to(device),
            batch["gt_plane2"].to(device),
        )

    batch = move_batch(batch, device)
    view1, view2 = build_views(batch, prefix)
    with torch.no_grad():
        result1, result2 = backbone(view1, view2)
        features1 = feature_maps_from_result(
            result1,
            view1["img"],
            (0, 6, 9, args.feature_index),
        )
        features2 = feature_maps_from_result(
            result2,
            view2["img"],
            (0, 6, 9, args.feature_index),
        )
    return (
        features1["deep"],
        features2["deep"],
        batch["gt_plane1"],
        batch["gt_plane2"],
    )


def slice_output(output, start, end):
    return {key: value[start:end] for key, value in output.items()}


def combine_view_rows(row1, row2):
    combined = {}
    for key in row1:
        combined[key] = 0.5 * (float(row1[key]) + float(row2[key]))
        combined[f"view1_{key}"] = float(row1[key])
        combined[f"view2_{key}"] = float(row2[key])
    return combined


def run_epoch(backbone, head, loader, optimizer, device, args, train):
    head.train(train)
    if backbone is not None:
        backbone.eval()
    rows = []

    for step, batch in enumerate(loader, start=1):
        feature1, feature2, gt_plane1, gt_plane2 = cached_or_live_pair(
            batch,
            backbone,
            device,
            args,
            "clean_pair_train" if train else "clean_pair_val",
        )

        if train:
            optimizer.zero_grad(set_to_none=True)

        supervise_both = (not train) or args.supervision == "both"
        if supervise_both:
            batch_size = feature1.shape[0]
            output = head(torch.cat((feature1, feature2), dim=0))
            output1 = slice_output(output, 0, batch_size)
            output2 = slice_output(output, batch_size, batch_size * 2)
            loss1, row1 = sample_loss_and_metrics(output1, gt_plane1, args)
            loss2, row2 = sample_loss_and_metrics(output2, gt_plane2, args)
            loss = 0.5 * (loss1 + loss2)
            row = combine_view_rows(row1, row2)
        else:
            output1 = head(feature1)
            loss, row1 = sample_loss_and_metrics(output1, gt_plane1, args)
            row = {key: float(value) for key, value in row1.items()}
            for key, value in row1.items():
                row[f"view1_{key}"] = float(value)

        if train:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()

        row["loss"] = float(loss.detach())
        rows.append(row)

        if step == 1 or step % args.log_every == 0 or step == len(loader):
            view1_iou = row.get("view1_mean_iou", row["mean_iou"])
            view2_text = (
                f" view2_iou={row['view2_mean_iou']:.3f}"
                if "view2_mean_iou" in row
                else ""
            )
            print(
                f"[{'Train' if train else 'Val'}] {step}/{len(loader)} "
                f"loss={row['loss']:.4f} mean_iou={row['mean_iou']:.3f} "
                f"view1_iou={view1_iou:.3f}{view2_text} "
                f"count_err={row['plane_count_abs_error']:.2f}",
                flush=True,
            )

    totals = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            totals[key] += float(value)
    return {key: value / max(len(rows), 1) for key, value in totals.items()}


def load_initial_checkpoint(head, checkpoint_path):
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("head", checkpoint)
    missing, unexpected = head.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing={missing}, unexpected={unexpected}"
        )
    print(
        f"Initialized clean head from {checkpoint_path} "
        f"(epoch={checkpoint.get('epoch', 'unknown')})",
        flush=True,
    )
    return checkpoint


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
        print("Evaluating the initialized head on both views before training...", flush=True)
        initial_val = run_epoch(
            backbone,
            head,
            val_loader,
            None,
            device,
            args,
            train=False,
        )
        print(
            json.dumps({"epoch": 0, "val": initial_val}, ensure_ascii=False),
            flush=True,
        )

    history = []
    best_iou = -1.0
    for epoch in range(1, args.num_epochs + 1):
        print(
            f"Epoch {epoch}/{args.num_epochs} supervision={args.supervision}",
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
