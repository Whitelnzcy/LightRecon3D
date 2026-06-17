import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone
from train_stage1_clean_pair_baseline import move_batch, parse_indices
from train_stage1_plane_masks import build_views, feature_maps_from_result


STAGE_NAMES = ("encoder", "shallow", "middle", "deep")


def parse_args():
    parser = argparse.ArgumentParser(
        "Cache multi-layer DUSt3R reference-branch features in both pair orders"
    )
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--pair_strategy", default="adjacent", choices=("adjacent", "all"))
    parser.add_argument("--pair_max_view_id_gap", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--small_train_size", type=int, default=128)
    parser.add_argument("--small_val_size", type=int, default=32)
    parser.add_argument("--train_indices", default="")
    parser.add_argument("--val_indices", default="")
    parser.add_argument(
        "--sampling_strategy",
        default="first",
        choices=("first", "space_balanced"),
        help=(
            "first preserves the old deterministic prefix subset. "
            "space_balanced round-robins across Structured3D spaces so that "
            "one large scene/room cannot dominate a small cache. Explicit "
            "train_indices/val_indices override this option."
        ),
    )
    parser.add_argument("--sampling_seed", type=int, default=20260617)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--feature_indices",
        default="0,6,9,12",
        help="Four comma-separated DUSt3R decoder output indices",
    )
    parser.add_argument("--rgb_cache_size", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=16)
    return parser.parse_args()


def parse_feature_indices(text):
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != 4:
        raise ValueError("feature_indices must contain exactly four integers")
    return values


def _space_key(sample):
    return sample["scene_name"], sample["pair_group"]


def select_space_balanced_indices(dataset, count, seed):
    if count <= 0 or count >= len(dataset):
        count = len(dataset)

    grouped = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        grouped[_space_key(sample)].append(index)

    rng = random.Random(seed)
    group_keys = sorted(grouped)
    rng.shuffle(group_keys)
    for key in group_keys:
        rng.shuffle(grouped[key])

    selected = []
    offsets = {key: 0 for key in group_keys}
    active = list(group_keys)
    while active and len(selected) < count:
        next_active = []
        for key in active:
            offset = offsets[key]
            values = grouped[key]
            if offset < len(values):
                selected.append(values[offset])
                offsets[key] = offset + 1
                if len(selected) >= count:
                    break
            if offsets[key] < len(values):
                next_active.append(key)
        active = next_active

    return selected


def summarize_indices(dataset, indices):
    scene_counts = Counter()
    space_counts = Counter()
    for index in indices:
        sample = dataset.samples[index]
        scene_counts[sample["scene_name"]] += 1
        space_counts[_space_key(sample)] += 1
    return {
        "pair_count": len(indices),
        "scene_count": len(scene_counts),
        "space_count": len(space_counts),
        "max_pairs_per_scene": max(scene_counts.values(), default=0),
        "max_pairs_per_space": max(space_counts.values(), default=0),
        "scene_pair_counts": dict(sorted(scene_counts.items())),
    }


def make_cache_loader(args, split, count, explicit_text):
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=split,
        train_ratio=args.train_ratio,
        image_size=(512, 512),
        input_mode="pair",
        pair_strategy=args.pair_strategy,
        pair_max_view_id_gap=args.pair_max_view_id_gap,
    )

    explicit = parse_indices(explicit_text)
    if explicit is not None:
        indices = [index for index in explicit if 0 <= index < len(dataset)]
        strategy = "explicit"
    elif args.sampling_strategy == "space_balanced":
        split_seed = args.sampling_seed + (0 if split == "train" else 1000003)
        indices = select_space_balanced_indices(dataset, count, split_seed)
        strategy = "space_balanced"
    else:
        indices = list(range(min(count, len(dataset))))
        strategy = "first"

    if not indices:
        raise RuntimeError(f"No valid {split} indices were selected")

    summary = summarize_indices(dataset, indices)
    summary["strategy"] = strategy
    summary["dataset_pair_count"] = len(dataset)
    print(
        f"[{split} selection] "
        + json.dumps(summary, ensure_ascii=False),
        flush=True,
    )

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader, indices, summary


def half_cpu_feature_dict(features):
    return {
        name: features[name].detach().to(dtype=torch.float16, device="cpu")
        for name in STAGE_NAMES
    }


@torch.no_grad()
def cache_split(backbone, loader, device, args, feature_indices, split):
    backbone.eval()
    cached = []
    feature_shapes = None

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        view1, view2 = build_views(batch, f"multiscale_symref_{split}")

        forward_ref, _ = backbone(view1, view2)
        reverse_ref, _ = backbone(view2, view1)
        features1 = feature_maps_from_result(
            forward_ref,
            view1["img"],
            feature_indices,
        )
        features2 = feature_maps_from_result(
            reverse_ref,
            view2["img"],
            feature_indices,
        )

        current_shapes = {
            name: list(features1[name].shape[1:]) for name in STAGE_NAMES
        }
        if feature_shapes is None:
            feature_shapes = current_shapes
            print(
                "Cached feature shapes: "
                + json.dumps(feature_shapes, ensure_ascii=False),
                flush=True,
            )
        elif current_shapes != feature_shapes:
            raise RuntimeError(
                f"Feature shape changed within cache: {current_shapes} vs {feature_shapes}"
            )

        rgb1 = F.interpolate(
            view1["img"],
            size=(args.rgb_cache_size, args.rgb_cache_size),
            mode="bilinear",
            align_corners=False,
        ).detach().to(dtype=torch.float16, device="cpu")
        rgb2 = F.interpolate(
            view2["img"],
            size=(args.rgb_cache_size, args.rgb_cache_size),
            mode="bilinear",
            align_corners=False,
        ).detach().to(dtype=torch.float16, device="cpu")

        cached.append(
            {
                "features1": half_cpu_feature_dict(features1),
                "features2": half_cpu_feature_dict(features2),
                "rgb1": rgb1,
                "rgb2": rgb2,
                "gt_plane1": batch["gt_plane1"].detach().cpu(),
                "gt_plane2": batch["gt_plane2"].detach().cpu(),
            }
        )

        if step == 1 or step % args.log_every == 0 or step == len(loader):
            print(f"[Multiscale cache {split}] {step}/{len(loader)}", flush=True)

    return cached, feature_shapes


def main():
    args = parse_args()
    feature_indices = parse_feature_indices(args.feature_indices)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_dust3r_backbone(args.weights_path, device=device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    train_loader, train_indices, train_selection = make_cache_loader(
        args,
        "train",
        args.small_train_size,
        args.train_indices,
    )
    val_loader, val_indices, val_selection = make_cache_loader(
        args,
        "val",
        args.small_val_size,
        args.val_indices,
    )

    train_cached, train_shapes = cache_split(
        backbone,
        train_loader,
        device,
        args,
        feature_indices,
        "train",
    )
    val_cached, val_shapes = cache_split(
        backbone,
        val_loader,
        device,
        args,
        feature_indices,
        "val",
    )
    if train_shapes != val_shapes:
        raise RuntimeError(
            f"Train and val feature shapes differ: {train_shapes} vs {val_shapes}"
        )

    input_dims = [int(train_shapes[name][0]) for name in STAGE_NAMES]
    payload = {
        "config": {
            "root_dir": str(Path(args.root_dir).resolve()),
            "train_ratio": float(args.train_ratio),
            "small_train_size": int(args.small_train_size),
            "small_val_size": int(args.small_val_size),
            "train_indices": args.train_indices,
            "val_indices": args.val_indices,
            "selected_train_indices": train_indices,
            "selected_val_indices": val_indices,
            "sampling_strategy": args.sampling_strategy,
            "sampling_seed": int(args.sampling_seed),
            "train_selection": train_selection,
            "val_selection": val_selection,
            "pair_strategy": args.pair_strategy,
            "pair_max_view_id_gap": args.pair_max_view_id_gap,
            "feature_indices": list(feature_indices),
            "feature_shapes": train_shapes,
            "input_dims": input_dims,
            "rgb_cache_size": int(args.rgb_cache_size),
            "image_size": 512,
            "feature_extraction": "symmetrized_reference_branch_multilayer",
        },
        "train": train_cached,
        "val": val_cached,
    }
    torch.save(payload, output_path)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "train_pairs": len(train_cached),
                "val_pairs": len(val_cached),
                "input_dims": input_dims,
                "feature_shapes": train_shapes,
                "train_selection": train_selection,
                "val_selection": val_selection,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
