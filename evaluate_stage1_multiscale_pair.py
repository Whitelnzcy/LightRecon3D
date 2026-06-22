import argparse
import json
from pathlib import Path

import torch

from models.multiscale_plane_mask_head import MultiScalePlaneMaskHead
from train_stage1_multiscale_pair import run_epoch


def parse_args():
    parser = argparse.ArgumentParser(
        "Evaluate a MultiScalePlaneMaskHead checkpoint on its cached pair split"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_cache_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=16)
    return parser.parse_args()


def main():
    cli = parse_args()
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    cache = torch.load(cli.feature_cache_path, map_location="cpu", weights_only=False)
    saved_args = argparse.Namespace(**checkpoint["args"])
    saved_args.batch_size = cli.batch_size
    saved_args.log_every = cli.log_every
    if not hasattr(saved_args, "use_geometry"):
        saved_args.use_geometry = False
    if not hasattr(saved_args, "use_masked_query_refine"):
        saved_args.use_masked_query_refine = False
    if not hasattr(saved_args, "decoder_ffn_multiplier"):
        saved_args.decoder_ffn_multiplier = 4
    if not hasattr(saved_args, "fuse_refine_blocks"):
        saved_args.fuse_refine_blocks = 1
    if not hasattr(saved_args, "pixel_refine_blocks"):
        saved_args.pixel_refine_blocks = 1

    input_dims = tuple(
        int(value)
        for value in checkpoint.get(
            "input_dims",
            cache.get("config", {}).get("input_dims", (1024, 768, 768, 768)),
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = MultiScalePlaneMaskHead(
        input_dims=input_dims,
        hidden_dim=saved_args.hidden_dim,
        num_queries=saved_args.num_queries,
        num_decoder_layers=saved_args.decoder_layers,
        num_heads=saved_args.decoder_heads,
        output_size=saved_args.output_size,
        use_rgb_skip=not saved_args.disable_rgb_skip,
        use_geometry=saved_args.use_geometry,
        geometry_dim=int(cache.get("config", {}).get("geometry_channels", 9)),
        use_masked_query_refine=saved_args.use_masked_query_refine,
        decoder_ffn_multiplier=saved_args.decoder_ffn_multiplier,
        fuse_refine_blocks=saved_args.fuse_refine_blocks,
        pixel_refine_blocks=saved_args.pixel_refine_blocks,
    ).to(device)
    head.load_state_dict(checkpoint["head"], strict=True)

    stats = run_epoch(
        head,
        cache[cli.split],
        None,
        device,
        saved_args,
        train=False,
        epoch=int(checkpoint.get("epoch", 0)),
    )
    payload = {
        "checkpoint": cli.checkpoint,
        "feature_cache_path": cli.feature_cache_path,
        "split": cli.split,
        "epoch": checkpoint.get("epoch"),
        "stats": stats,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    keys = [
        "mean_iou",
        "view1_mean_iou",
        "view2_mean_iou",
        "leakage_rate",
        "background_error_rate",
        "benchmark32_mean_iou",
        "benchmark32_view1_mean_iou",
        "benchmark32_view2_mean_iou",
        "benchmark32_leakage_rate",
        "stage_weight_encoder",
        "stage_weight_shallow",
        "stage_weight_middle",
        "stage_weight_deep",
    ]
    print("=" * 76)
    for key in keys:
        if key in stats:
            print(f"{key:<34} = {stats[key]:.6f}")
    print("=" * 76)
    print(f"Saved: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
