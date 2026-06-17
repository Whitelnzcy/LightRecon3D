import argparse
import json
from pathlib import Path

import torch

from models.build_backbone import build_dust3r_backbone
from train_stage1_clean_pair_baseline import (
    cache_config,
    make_loader,
    move_batch,
)
from train_stage1_plane_masks import build_views, feature_maps_from_result


def parse_args():
    parser = argparse.ArgumentParser(
        "Cache real-pair DUSt3R features with each image evaluated as reference view"
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
    parser.add_argument("--feature_index", type=int, default=12)
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--log_every", type=int, default=16)
    return parser.parse_args()


@torch.no_grad()
def cache_split(backbone, loader, device, args, split):
    backbone.eval()
    cached = []
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        view1, view2 = build_views(batch, f"symref_cache_{split}")

        # Forward order: original view1 is the DUSt3R reference branch.
        forward_ref, _ = backbone(view1, view2)
        # Reversed order: original view2 is now evaluated through the same
        # reference branch, avoiding the asymmetric second decoder domain.
        reverse_ref, _ = backbone(view2, view1)

        features1 = feature_maps_from_result(
            forward_ref,
            view1["img"],
            (0, 6, 9, args.feature_index),
        )
        features2 = feature_maps_from_result(
            reverse_ref,
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
            print(f"[SymRef cache {split}] {step}/{len(loader)}", flush=True)
    return cached


def main():
    args = parse_args()
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

    print(
        "Caching each image through DUSt3R reference branch in both pair orders...",
        flush=True,
    )
    train_cached = cache_split(backbone, train_loader, device, args, "train")
    val_cached = cache_split(backbone, val_loader, device, args, "val")

    payload = {
        # Keep the exact configuration schema expected by the existing pair
        # baseline so this cache can be consumed without changing the trainer.
        "config": cache_config(args),
        "feature_extraction": "symmetrized_reference_branch",
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
                "feature_extraction": payload["feature_extraction"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
