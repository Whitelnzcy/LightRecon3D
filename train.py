import os
import sys
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset


# ============================================================
# 0. Path setup
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")

if DUST3R_REPO_ROOT not in sys.path:
    sys.path.insert(0, DUST3R_REPO_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel
from loss.losses import compute_losses


# ============================================================
# 1. Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser("LightRecon3D Training")

    # -------------------------
    # paths
    # -------------------------
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/data/zhucy23u/datasets/Structured3D",
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        default="/data/zhucy23u/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data/zhucy23u/checkpoints/lightrecon/plane_embedding_debug",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a LightRecon3D checkpoint.",
    )

    # -------------------------
    # dataset
    # -------------------------
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--small_train_size", type=int, default=-1)
    parser.add_argument("--small_val_size", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4)

    # -------------------------
    # training
    # -------------------------
    parser.add_argument("--num_epochs", "--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. Set <=0 to disable.",
    )

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--run_val", action="store_true")
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # -------------------------
    # model
    # -------------------------
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)
    parser.add_argument("--freeze_encoder", action="store_true")

    # -------------------------
    # loss weights
    # -------------------------
    parser.add_argument("--line_weight", type=float, default=1.0)
    parser.add_argument("--plane_weight", type=float, default=1.0)

    parser.add_argument(
        "--geo_weight",
        type=float,
        default=0.0,
        help="Weight of 3D coplanarity loss. Keep 0.0 during plane embedding stage.",
    )

    # -------------------------
    # line loss params
    # -------------------------
    parser.add_argument("--line_pos_weight", type=float, default=None)
    parser.add_argument("--line_bce_weight", type=float, default=1.0)
    parser.add_argument("--line_dice_weight", type=float, default=1.0)

    # -------------------------
    # plane embedding loss params
    # -------------------------
    parser.add_argument("--plane_min_pixels", type=int, default=64)
    parser.add_argument("--plane_max_pixels_per_plane", type=int, default=2048)
    parser.add_argument("--plane_max_planes_per_image", type=int, default=12)

    parser.add_argument("--plane_delta_var", type=float, default=0.5)
    parser.add_argument("--plane_delta_dist", type=float, default=1.5)
    parser.add_argument("--plane_var_weight", type=float, default=1.0)
    parser.add_argument("--plane_dist_weight", type=float, default=1.0)
    parser.add_argument("--plane_reg_weight", type=float, default=0.001)

    # -------------------------
    # coplanarity loss params
    # only used when geo_weight > 0
    # -------------------------
    parser.add_argument("--coplanarity_min_points", type=int, default=128)
    parser.add_argument("--coplanarity_max_points_per_plane", type=int, default=2048)
    parser.add_argument("--coplanarity_max_planes_per_image", type=int, default=8)

    parser.add_argument(
        "--coplanarity_normalize",
        action="store_true",
        help="Use normalized smallest eigenvalue for coplanarity loss.",
    )

    # -------------------------
    # swanlab
    # -------------------------
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--swanlab_project", type=str, default="LightRecon3D")
    parser.add_argument("--swanlab_run_name", type=str, default=None)
    parser.add_argument(
        "--swanlab_mode",
        type=str,
        default="cloud",
        choices=["cloud", "local", "disabled"],
    )
    parser.add_argument(
        "--swanlab_logdir",
        type=str,
        default="/data/zhucy23u/logs/swanlab",
    )

    return parser.parse_args()


# ============================================================
# 2. Utility functions
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def move_batch_to_device(batch, device):
    out = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value

    return out


def build_views_from_batch(batch, prefix="train"):
    """
    Current stage:
    single-image pseudo-pair for DUSt3R interface.

    view1 and view2 use the same image.
    Later this should be replaced by real image pairs.
    """
    batch_size = batch["img"].shape[0]

    view1 = {
        "img": batch["img"],
        "instance": [f"{prefix}_{i}" for i in range(batch_size)],
    }

    view2 = {
        "img": batch["img"],
        "instance": [f"{prefix}_{i}" for i in range(batch_size)],
    }

    return view1, view2


def make_subset(dataset, size):
    if size is None or size <= 0:
        return dataset

    size = min(size, len(dataset))
    indices = list(range(size))

    return Subset(dataset, indices)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")


def freeze_dust3r_encoder(model):
    """
    Freeze DUSt3R encoder-related parameters.

    Encoder names in DUSt3R/CroCo are usually:
    - patch_embed
    - enc_blocks
    - enc_norm

    Decoder and added heads remain trainable.
    """
    encoder_prefixes = (
        "patch_embed",
        "enc_blocks",
        "enc_norm",
    )

    frozen = 0
    total = 0

    for name, param in model.backbone.named_parameters():
        total += param.numel()

        if name.startswith(encoder_prefixes):
            param.requires_grad = False
            frozen += param.numel()

    print(f"Frozen DUSt3R encoder params: {frozen:,} / {total:,}")


def init_swanlab(args):
    if not args.use_swanlab:
        return None

    import swanlab

    os.makedirs(args.swanlab_logdir, exist_ok=True)

    run = swanlab.init(
        project=args.swanlab_project,
        experiment_name=args.swanlab_run_name,
        mode=args.swanlab_mode,
        logdir=args.swanlab_logdir,
        config=vars(args),
        description="LightRecon3D training: line supervision + plane embedding supervision.",
    )

    return run


def swanlab_log(run, metrics, step=None):
    if run is None:
        return

    clean_metrics = {}

    for key, value in metrics.items():
        if value is None:
            continue

        try:
            clean_metrics[key] = to_float(value)
        except Exception:
            continue

    if not clean_metrics:
        return

    try:
        run.log(clean_metrics, step=step)
    except AttributeError:
        import swanlab
        swanlab.log(clean_metrics, step=step)


def finish_swanlab(run):
    if run is None:
        return

    try:
        run.finish()
    except AttributeError:
        import swanlab
        swanlab.finish()


def save_checkpoint(
    save_dir,
    filename,
    model,
    optimizer,
    epoch,
    args,
    best_val=None,
):
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, filename)

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "args": vars(args),
        "best_val": best_val,
    }

    torch.save(ckpt, path)
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    print(f"Loading checkpoint from: {path}")

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[Resume] Missing keys: {missing[:10]}")
        if len(missing) > 10:
            print(f"... and {len(missing) - 10} more missing keys")

    if unexpected:
        print(f"[Resume] Unexpected keys: {unexpected[:10]}")
        if len(unexpected) > 10:
            print(f"... and {len(unexpected) - 10} more unexpected keys")

    if optimizer is not None and isinstance(ckpt, dict):
        opt_state = ckpt.get("optimizer", None)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

    start_epoch = 1
    best_val = None

    if isinstance(ckpt, dict):
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = ckpt.get("best_val", None)

    print(f"Resume start_epoch = {start_epoch}")
    print(f"Resume best_val    = {best_val}")

    return start_epoch, best_val


def compute_loss_wrapper(res, batch, args):
    """
    Only place that calls compute_losses.

    This avoids train / val using different loss APIs.
    """
    return compute_losses(
        res=res,
        batch=batch,

        line_weight=args.line_weight,
        plane_weight=args.plane_weight,
        geo_weight=args.geo_weight,

        line_pos_weight=args.line_pos_weight,
        line_bce_weight=args.line_bce_weight,
        line_dice_weight=args.line_dice_weight,

        plane_min_pixels=args.plane_min_pixels,
        plane_max_pixels_per_plane=args.plane_max_pixels_per_plane,
        plane_max_planes_per_image=args.plane_max_planes_per_image,

        plane_delta_var=args.plane_delta_var,
        plane_delta_dist=args.plane_delta_dist,
        plane_var_weight=args.plane_var_weight,
        plane_dist_weight=args.plane_dist_weight,
        plane_reg_weight=args.plane_reg_weight,

        coplanarity_min_points=args.coplanarity_min_points,
        coplanarity_max_points_per_plane=args.coplanarity_max_points_per_plane,
        coplanarity_max_planes_per_image=args.coplanarity_max_planes_per_image,
        coplanarity_normalize=args.coplanarity_normalize,
    )


def add_to_meters(meters, loss_dict):
    for key, value in loss_dict.items():
        try:
            meters[key] += to_float(value)
        except Exception:
            pass


def average_meters(meters, denom):
    if denom <= 0:
        return {}

    return {key: value / denom for key, value in meters.items()}


# ============================================================
# 3. Train / val
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    args,
    epoch,
    swanlab_run=None,
):
    model.train()

    meters = defaultdict(float)
    valid_batches = 0
    skipped_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        view1, view2 = build_views_from_batch(batch, prefix="train")

        res1, res2 = model(view1, view2)

        loss_total, loss_dict = compute_loss_wrapper(res1, batch, args)

        if not torch.isfinite(loss_total):
            print(
                f"[Warning] non-finite train loss at "
                f"epoch={epoch}, batch={batch_idx}: {loss_total.item()}"
            )
            optimizer.zero_grad(set_to_none=True)
            skipped_batches += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=args.grad_clip,
            )

        optimizer.step()

        valid_batches += 1
        add_to_meters(meters, loss_dict)

        global_step = (epoch - 1) * len(loader) + batch_idx

        log_metrics = {
            "train/batch_total_loss": to_float(loss_total),
            "train/batch_line_loss": loss_dict.get("loss_line", 0.0),
            "train/batch_plane_loss": loss_dict.get("loss_plane", 0.0),
            "train/lr": optimizer.param_groups[0]["lr"],
        }

        extra_keys = [
            "loss_line_bce",
            "loss_line_dice",
            "loss_plane_embedding",
            "loss_plane_var",
            "loss_plane_dist",
            "loss_plane_reg",
            "loss_plane_pull",
            "loss_plane_push",
            "num_planes",
            "loss_coplanarity",
            "num_geo_planes",
        ]

        for key in extra_keys:
            if key in loss_dict:
                log_metrics[f"train/{key}"] = loss_dict[key]

        swanlab_log(swanlab_run, log_metrics, step=global_step)

        if (
            batch_idx == 1
            or batch_idx == len(loader)
            or batch_idx % args.log_every == 0
        ):
            print(
                f"[Train] batch {batch_idx}/{len(loader)} | "
                f"total={to_float(loss_total):.4f}, "
                f"line={to_float(loss_dict.get('loss_line', 0.0)):.4f}, "
                f"plane={to_float(loss_dict.get('loss_plane', 0.0)):.4f}"
            )

    stats = average_meters(meters, valid_batches)

    if valid_batches == 0:
        stats = {
            "loss_total": float("nan"),
            "loss_line": float("nan"),
            "loss_plane": float("nan"),
        }

    stats["skipped_batches"] = skipped_batches

    print(
        f"Train | total: {stats.get('loss_total', float('nan')):.4f}, "
        f"line: {stats.get('loss_line', float('nan')):.4f}, "
        f"plane: {stats.get('loss_plane', float('nan')):.4f}, "
        f"skipped: {skipped_batches}"
    )

    epoch_metrics = {
        "train/epoch_total_loss": stats.get("loss_total", float("nan")),
        "train/epoch_line_loss": stats.get("loss_line", float("nan")),
        "train/epoch_plane_loss": stats.get("loss_plane", float("nan")),
        "train/epoch_skipped_batches": skipped_batches,
    }

    for key in [
        "loss_line_bce",
        "loss_line_dice",
        "loss_plane_embedding",
        "loss_plane_var",
        "loss_plane_dist",
        "loss_plane_reg",
        "loss_plane_pull",
        "loss_plane_push",
        "num_planes",
        "loss_coplanarity",
        "num_geo_planes",
    ]:
        if key in stats:
            epoch_metrics[f"train/epoch_{key}"] = stats[key]

    swanlab_log(swanlab_run, epoch_metrics, step=epoch)

    return stats


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
    args,
    epoch,
    swanlab_run=None,
):
    model.eval()

    meters = defaultdict(float)
    valid_batches = 0
    skipped_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        view1, view2 = build_views_from_batch(batch, prefix="val")

        res1, res2 = model(view1, view2)

        loss_total, loss_dict = compute_loss_wrapper(res1, batch, args)

        if not torch.isfinite(loss_total):
            print(
                f"[Warning] non-finite val loss at "
                f"epoch={epoch}, batch={batch_idx}: {loss_total.item()}"
            )
            skipped_batches += 1
            continue

        valid_batches += 1
        add_to_meters(meters, loss_dict)

        global_step = (epoch - 1) * len(loader) + batch_idx

        log_metrics = {
            "val/batch_total_loss": to_float(loss_total),
            "val/batch_line_loss": loss_dict.get("loss_line", 0.0),
            "val/batch_plane_loss": loss_dict.get("loss_plane", 0.0),
        }

        extra_keys = [
            "loss_line_bce",
            "loss_line_dice",
            "loss_plane_embedding",
            "loss_plane_var",
            "loss_plane_dist",
            "loss_plane_reg",
            "loss_plane_pull",
            "loss_plane_push",
            "num_planes",
            "loss_coplanarity",
            "num_geo_planes",
        ]

        for key in extra_keys:
            if key in loss_dict:
                log_metrics[f"val/{key}"] = loss_dict[key]

        swanlab_log(swanlab_run, log_metrics, step=global_step)

        if (
            batch_idx == 1
            or batch_idx == len(loader)
            or batch_idx % args.log_every == 0
        ):
            print(
                f"[Val] batch {batch_idx}/{len(loader)} | "
                f"total={to_float(loss_total):.4f}, "
                f"line={to_float(loss_dict.get('loss_line', 0.0)):.4f}, "
                f"plane={to_float(loss_dict.get('loss_plane', 0.0)):.4f}"
            )

    stats = average_meters(meters, valid_batches)

    if valid_batches == 0:
        stats = {
            "loss_total": float("nan"),
            "loss_line": float("nan"),
            "loss_plane": float("nan"),
        }

    stats["skipped_batches"] = skipped_batches

    print(
        f"Val   | total: {stats.get('loss_total', float('nan')):.4f}, "
        f"line: {stats.get('loss_line', float('nan')):.4f}, "
        f"plane: {stats.get('loss_plane', float('nan')):.4f}, "
        f"skipped: {skipped_batches}"
    )

    epoch_metrics = {
        "val/epoch_total_loss": stats.get("loss_total", float("nan")),
        "val/epoch_line_loss": stats.get("loss_line", float("nan")),
        "val/epoch_plane_loss": stats.get("loss_plane", float("nan")),
        "val/epoch_skipped_batches": skipped_batches,
    }

    for key in [
        "loss_line_bce",
        "loss_line_dice",
        "loss_plane_embedding",
        "loss_plane_var",
        "loss_plane_dist",
        "loss_plane_reg",
        "loss_plane_pull",
        "loss_plane_push",
        "num_planes",
        "loss_coplanarity",
        "num_geo_planes",
    ]:
        if key in stats:
            epoch_metrics[f"val/epoch_{key}"] = stats[key]

    swanlab_log(swanlab_run, epoch_metrics, step=epoch)

    return stats


# ============================================================
# 4. Main
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("LightRecon3D Training")
    print("=" * 80)
    print(f"device      : {device}")
    print(f"root_dir    : {args.root_dir}")
    print(f"weights_path: {args.weights_path}")
    print(f"save_dir    : {args.save_dir}")
    print(f"line_weight : {args.line_weight}")
    print(f"plane_weight: {args.plane_weight}")
    print(f"geo_weight  : {args.geo_weight}")
    print("=" * 80)

    # -------------------------
    # dataset
    # -------------------------
    train_dataset_full = Structured3DDataset(
        root_dir=args.root_dir,
        split="train",
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
    )

    val_dataset_full = Structured3DDataset(
        root_dir=args.root_dir,
        split="val",
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
    )

    print(f"Train scenes: {len(train_dataset_full.scenes)}")
    print(f"Val scenes  : {len(val_dataset_full.scenes)}")
    print(f"Full train samples: {len(train_dataset_full)}")
    print(f"Full val samples  : {len(val_dataset_full)}")

    train_dataset = make_subset(train_dataset_full, args.small_train_size)
    val_dataset = make_subset(val_dataset_full, args.small_val_size)

    print(f"Used train samples: {len(train_dataset)}")
    print(f"Used val samples  : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_train,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # -------------------------
    # model
    # -------------------------
    dust3r_backbone = build_dust3r_backbone(
        args.weights_path,
        device=device,
    )

    model = LightReconModel(
        dust3r_backbone=dust3r_backbone,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    ).to(device)

    if args.freeze_encoder:
        freeze_dust3r_encoder(model)

    count_parameters(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_val = None

    if args.resume is not None:
        start_epoch, best_val = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    swanlab_run = init_swanlab(args)

    # -------------------------
    # training loop
    # -------------------------
    for epoch in range(start_epoch, args.num_epochs + 1):
        print()
        print(f"========== Epoch {epoch}/{args.num_epochs} ==========")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            epoch=epoch,
            swanlab_run=swanlab_run,
        )

        val_stats = None

        if args.run_val:
            val_stats = validate_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                args=args,
                epoch=epoch,
                swanlab_run=swanlab_run,
            )

        # Always save latest.
        save_checkpoint(
            save_dir=args.save_dir,
            filename="latest.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            best_val=best_val,
        )

        # Best checkpoint is based on val total loss if validation is enabled.
        if args.run_val and val_stats is not None:
            current_val = val_stats.get("loss_total", float("nan"))

            if current_val == current_val:  # not NaN
                if best_val is None or current_val < best_val:
                    best_val = current_val

                    save_checkpoint(
                        save_dir=args.save_dir,
                        filename="best.pth",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        args=args,
                        best_val=best_val,
                    )

                    print(f"Best checkpoint updated at epoch {epoch}: val={best_val:.4f}")
        else:
            # If no validation, use train total loss for best.
            current_train = train_stats.get("loss_total", float("nan"))

            if current_train == current_train:
                if best_val is None or current_train < best_val:
                    best_val = current_train

                    save_checkpoint(
                        save_dir=args.save_dir,
                        filename="best.pth",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        args=args,
                        best_val=best_val,
                    )

                    print(f"Best checkpoint updated at epoch {epoch}: train={best_val:.4f}")

    finish_swanlab(swanlab_run)

    print()
    print("Training finished.")


if __name__ == "__main__":
    main()