import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataloaders.s3d_dataset import Structured3DDataset
from models.stage1_feature_ablation_head import Stage1FeatureAblationHead
from train_stage1_clean_baseline import (
    apply_query_class_scores,
    masks_for_plane_ids,
    match_queries,
    move_batch,
    sample_loss_and_metrics,
    select_plane_ids,
)
from train_stage1_plane_masks import build_views, feature_maps_from_result


def parse_args():
    parser = argparse.ArgumentParser(
        "Parameter-matched Stage1 DUSt3R-feature versus RGB ablation"
    )
    parser.add_argument("--condition", required=True, choices=("dust3r", "rgb"))
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--weights_path", default="")
    parser.add_argument("--val_records_csv", required=True)
    parser.add_argument(
        "--val_limit",
        type=int,
        default=0,
        help="Technical smoke only; 0 keeps the full frozen validation set.",
    )
    parser.add_argument("--input_mode", default="pair", choices=("pair", "single"))
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--train_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_queries", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--feature_index", type=int, default=12)
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--patch_size", type=int, default=16)
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
    parser.add_argument("--class_score_weight", type=float, default=0.0)
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
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--cache_dtype",
        default="float16",
        choices=("float16", "float32"),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_val_records(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indices = [int(row["sample_idx"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("val_records_csv contains duplicate sample_idx values")
    if not indices:
        raise ValueError("val_records_csv contains no sample_idx values")
    return rows


def make_loader(args, split, indices):
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=split,
        train_ratio=args.train_ratio,
        image_size=(512, 512),
        input_mode=args.input_mode,
    )
    invalid = [index for index in indices if index < 0 or index >= len(dataset)]
    if invalid:
        raise ValueError(f"Indices outside {split} dataset: {invalid[:10]}")
    return DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


@torch.no_grad()
def cache_records(args, loader, device, backbone, prefix):
    dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
    records = []
    if backbone is not None:
        backbone.eval()
    for record_id, batch in enumerate(loader):
        batch = move_batch(batch, device)
        view1, view2 = build_views(batch, prefix)
        if args.condition == "dust3r":
            result1, _ = backbone(view1, view2)
            features = feature_maps_from_result(
                result1,
                view1["img"],
                (0, 6, 9, args.feature_index),
            )
            inputs = features["deep"]
        else:
            inputs = view1["img"]
        labels = batch.get("gt_plane1", batch["gt_plane"])
        records.append(
            {
                "record_id": record_id,
                "input": inputs.detach().to(device="cpu", dtype=dtype),
                "gt_plane": labels.detach().to(device="cpu"),
            }
        )
        if (record_id + 1) % args.log_every == 0 or record_id + 1 == len(loader):
            print(f"[Cache {prefix}] {record_id + 1}/{len(loader)}", flush=True)
    return records


def support_metrics(output, labels, args):
    resized_labels, plane_ids = select_plane_ids(
        labels,
        output["mask_logits"].shape[-2:],
        args.num_queries,
        args.min_plane_pixels,
    )
    resized_labels, masks = masks_for_plane_ids(
        labels,
        output["mask_logits"].shape[-2:],
        plane_ids,
    )
    targets = (
        torch.stack(masks[0])
        if masks[0]
        else output["mask_logits"].new_zeros(
            (0, *output["mask_logits"].shape[-2:])
        )
    )
    query_ids, target_ids = match_queries(
        output["mask_logits"][0],
        targets,
        args,
        existence_logits=output["existence_logits"][0],
    )
    scored = apply_query_class_scores(
        output["mask_logits"][0],
        output["existence_logits"][0],
        args,
    )
    predicted = torch.cat((scored, output["background_logits"][0]), dim=0).argmax(0)
    true_positive = 0
    false_positive = 0
    false_negative = 0
    matched_ious = []
    for query_id, target_id in zip(query_ids, target_ids):
        prediction = predicted == int(query_id)
        target = targets[int(target_id)] > 0.5
        intersection = int((prediction & target).sum())
        union = int((prediction | target).sum())
        true_positive += intersection
        false_positive += int((prediction & ~target).sum())
        false_negative += int((~prediction & target).sum())
        matched_ious.append(intersection / max(union, 1))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    active_queries = int(
        (output["existence_logits"][0].sigmoid() > args.existence_threshold).sum()
    )
    return {
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
        "matched_mask_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "matched_planes": len(matched_ious),
        "gt_planes": len(plane_ids[0]),
        "pred_planes": active_queries,
        "plane_count_abs_error": abs(active_queries - len(plane_ids[0])),
        "gt_plane_pixel_fraction": float(
            ((resized_labels[0] > 0) & (resized_labels[0] != 255)).float().mean()
        ),
    }


def aggregate(rows):
    totals = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            if key != "record_id":
                totals[key] += float(value)
    return {key: value / max(len(rows), 1) for key, value in totals.items()}


def run_epoch(model, records, optimizer, device, args, epoch, train):
    model.train(train)
    order = list(range(len(records)))
    if train:
        generator = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(records), generator=generator).tolist()
    rows = []
    batches = [order[start : start + args.batch_size] for start in range(0, len(order), args.batch_size)]
    for step, record_indices in enumerate(batches, start=1):
        inputs = torch.cat(
            [records[index]["input"] for index in record_indices], dim=0
        ).to(device=device, dtype=torch.float32)
        labels = torch.cat(
            [records[index]["gt_plane"] for index in record_indices], dim=0
        ).to(device=device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss, row = sample_loss_and_metrics(output, labels, args)
        if train:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        row = dict(row)
        row["loss"] = float(loss.detach())
        rows.append(row)
        if step % args.log_every == 0 or step == len(batches):
            print(
                f"[{'Train' if train else 'Val'}] {step}/{len(batches)} "
                f"loss={row['loss']:.4f} iou={row['mean_iou']:.4f}",
                flush=True,
            )
    return aggregate(rows)


@torch.no_grad()
def evaluate_records(model, records, device, args):
    model.eval()
    rows = []
    for record in records:
        inputs = record["input"].to(device=device, dtype=torch.float32)
        labels = record["gt_plane"].to(device=device)
        output = model(inputs)
        row = support_metrics(output, labels, args)
        row["record_id"] = int(record["record_id"])
        rows.append(row)
    return rows, aggregate(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    if save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_record_metadata = read_val_records(args.val_records_csv)
    if args.val_limit > 0:
        val_record_metadata = val_record_metadata[: args.val_limit]
    val_indices = [int(row["sample_idx"]) for row in val_record_metadata]
    train_dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split="train",
        train_ratio=args.train_ratio,
        image_size=(512, 512),
        input_mode=args.input_mode,
    )
    train_indices = list(range(min(args.train_size, len(train_dataset))))
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = make_loader(args, "val", val_indices)

    backbone = None
    if args.condition == "dust3r":
        if not args.weights_path:
            raise ValueError("--weights_path is required for the dust3r condition")
        from models.build_backbone import build_dust3r_backbone

        backbone = build_dust3r_backbone(args.weights_path, device=device)
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)

    cache_start = time.perf_counter()
    train_records = cache_records(args, train_loader, device, backbone, "train")
    val_records = cache_records(args, val_loader, device, backbone, "val")
    cache_seconds = time.perf_counter() - cache_start
    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = Stage1FeatureAblationHead(
        input_source=args.condition,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.decoder_layers,
        num_heads=args.decoder_heads,
        patch_size=args.patch_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_iou = -1.0
    best_epoch = 0
    train_start = time.perf_counter()
    for epoch in range(1, args.num_epochs + 1):
        train_stats = run_epoch(model, train_records, optimizer, device, args, epoch, True)
        val_stats = run_epoch(model, val_records, None, device, args, epoch, False)
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        print(json.dumps(history[-1]), flush=True)
        torch.save(
            {"model": model.state_dict(), "args": vars(args), "epoch": epoch},
            save_dir / "latest.pt",
        )
        if val_stats.get("mean_iou", 0.0) > best_iou:
            best_iou = val_stats.get("mean_iou", 0.0)
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                save_dir / "best.pt",
            )
        (save_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    train_seconds = time.perf_counter() - train_start

    best = torch.load(save_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    rows, summary = evaluate_records(model, val_records, device, args)
    for row, metadata in zip(rows, val_record_metadata):
        row["sample_idx"] = int(metadata["sample_idx"])
        row["scene_name"] = metadata.get("scene_name", "")
        row["item_id"] = metadata.get("item_id", "")
        row["condition"] = args.condition
        row["seed"] = args.seed
    with (save_dir / "per_record_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    source_files = [
        Path(__file__),
        Path(__file__).parent / "models" / "stage1_feature_ablation_head.py",
        Path(__file__).parent / "train_stage1_clean_baseline.py",
    ]
    manifest = {
        "condition": args.condition,
        "seed": args.seed,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "best_epoch": best_epoch,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "input_projection_parameters": model.projected_parameter_count(),
        "cache_seconds": cache_seconds,
        "train_seconds": train_seconds,
        "summary": summary,
        "args": vars(args),
        "val_records_csv_sha256": sha256(args.val_records_csv),
        "weights_sha256": sha256(args.weights_path) if args.weights_path else None,
        "source_sha256": {str(path): sha256(path) for path in source_files},
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    (save_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
