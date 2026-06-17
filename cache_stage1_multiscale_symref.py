import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from models.build_backbone import build_dust3r_backbone
from train_stage1_clean_pair_baseline import make_loader, move_batch
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

    train_loader = make_loader(
        args,
        "train",
        args.small_train_size,
        args.train_indices,
        shuffle=False,
    )
    val_loader = make_loader(
        args,
        "val",
        args.small_val_size,
        args.val_indices,
        shuffle=False,
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
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
