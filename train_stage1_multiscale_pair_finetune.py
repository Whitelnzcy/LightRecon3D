import json
from pathlib import Path

import torch

from models.multiscale_plane_mask_head import MultiScalePlaneMaskHead
from train_stage1_clean_baseline import set_seed
from train_stage1_multiscale_pair import (
    initialize_from_clean,
    parse_args,
    run_epoch,
)


def initialize_head(head, checkpoint_path):
    if not checkpoint_path:
        return "random"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") == "MultiScalePlaneMaskHead":
        state = checkpoint.get("head", checkpoint)
        missing, unexpected = head.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Multiscale checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        print(
            f"Fully initialized multiscale head from {checkpoint_path} "
            f"(epoch={checkpoint.get('epoch', 'unknown')})",
            flush=True,
        )
        return "multiscale_full"

    initialize_from_clean(head, checkpoint_path)
    return "clean_partial"


def main():
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cache = torch.load(args.feature_cache_path, map_location="cpu", weights_only=False)
    config = cache.get("config", {})
    input_dims = tuple(
        int(value)
        for value in config.get("input_dims", (1024, 768, 768, 768))
    )
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
    initialization_mode = initialize_head(head, args.init_checkpoint)

    experiment_config = vars(args).copy()
    experiment_config["initialization_mode"] = initialization_mode
    (save_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
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
            f"Epoch {epoch}/{args.num_epochs} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}",
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
            "initialization_mode": initialization_mode,
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
