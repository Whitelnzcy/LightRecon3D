import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, Subset

# =========================
# 0. 确保 dust3r 可导入
# =========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")
if DUST3R_REPO_ROOT not in sys.path:
    sys.path.insert(0, DUST3R_REPO_ROOT)

from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel
from loss.losses import total_loss_fn


# =========================
# 1. 参数解析
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train LightRecon3D")

    # 路径
    parser.add_argument(
        "--root_dir",
        type=str,
        default=r"E:\Study\code\LightRecon3D\data\Structured3D",
        help="Structured3D 根目录"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=r"checkpoints\DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        help="DUSt3R 预训练权重路径"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"checkpoints\lightrecon3d",
        help="checkpoint 保存目录"
    )

    # 训练超参数
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.9)

    # 小样本调试 / overfit 测试
    parser.add_argument(
        "--small_train_size",
        type=int,
        default=32,
        help=">0 时，只取 train dataset 的前 N 个样本"
    )
    parser.add_argument(
        "--small_val_size",
        type=int,
        default=8,
        help=">0 时，只取 val dataset 的前 N 个样本"
    )

    # loss 权重
    parser.add_argument("--line_weight", type=float, default=1.0)
    parser.add_argument("--plane_weight", type=float, default=1.0)
    parser.add_argument("--line_pos_weight", type=float, default=10.0)
    parser.add_argument("--line_dice_weight", type=float, default=1.0)

    # 训练策略
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="是否冻结 DUSt3R encoder"
    )
    parser.add_argument(
        "--run_val",
        action="store_true",
        help="是否每个 epoch 跑验证"
    )
    parser.add_argument(
        "--shuffle_train",
        action="store_true",
        help="是否打乱 train loader"
    )

    # 日志
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="每多少个 batch 打印一次 train 日志"
    )

    return parser.parse_args()


# =========================
# 2. 工具函数
# =========================
def move_batch_to_device(batch, device):
    new_batch = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            new_batch[k] = v.to(device, non_blocking=True)
        else:
            new_batch[k] = v
    return new_batch


def build_views_from_batch(batch, prefix="train"):
    """
    当前版本仍然使用“同图构造 view1/view2”的调试方式，
    目的是先把训练主循环稳定跑起来。
    """
    B = batch["img"].shape[0]

    view1 = {
        "img": batch["img"],
        "instance": [f"{prefix}_{i}" for i in range(B)],
    }
    view2 = {
        "img": batch["img"],
        "instance": [f"{prefix}_{i}" for i in range(B)],
    }
    return view1, view2


def freeze_dust3r_encoder(model):
    """
    冻结 LightReconModel.backbone 的 encoder 部分，
    只训练 decoder + 你新增的 line/plane heads。
    """
    backbone = model.backbone

    # patch embedding
    for p in backbone.patch_embed.parameters():
        p.requires_grad = False

    # encoder transformer
    for p in backbone.enc_blocks.parameters():
        p.requires_grad = False

    # encoder norm
    if hasattr(backbone, "enc_norm"):
        for p in backbone.enc_norm.parameters():
            p.requires_grad = False

    # mask token
    if hasattr(backbone, "mask_token") and backbone.mask_token is not None:
        backbone.mask_token.requires_grad = False


def count_trainable_parameters(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def save_checkpoint(model, optimizer, epoch, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckpt, save_path)


# =========================
# 3. train / val
# =========================
def train_one_epoch(model, loader, optimizer, device, args):
    model.train()

    total_loss_sum = 0.0
    total_line_sum = 0.0
    total_plane_sum = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        view1, view2 = build_views_from_batch(batch, prefix="train")

        optimizer.zero_grad()

        res1, res2 = model(view1, view2)

        loss_total, loss_dict = total_loss_fn(
            res1,
            batch,
            line_weight=args.line_weight,
            plane_weight=args.plane_weight,
            line_pos_weight=args.line_pos_weight,
            line_dice_weight=args.line_dice_weight
        )

        loss_total.backward()
        optimizer.step()

        total_loss_sum += loss_dict["loss_total"]
        total_line_sum += loss_dict["loss_line"]
        total_plane_sum += loss_dict["loss_plane"]
        num_batches += 1

        if batch_idx % args.log_every == 0:
            print(
                f"[Train] batch {batch_idx}/{len(loader)} | "
                f"total={loss_dict['loss_total']:.4f}, "
                f"line={loss_dict['loss_line']:.4f}, "
                f"plane={loss_dict['loss_plane']:.4f}"
            )

    avg_loss = total_loss_sum / max(num_batches, 1)
    avg_line = total_line_sum / max(num_batches, 1)
    avg_plane = total_plane_sum / max(num_batches, 1)

    return {
        "loss_total": avg_loss,
        "loss_line": avg_line,
        "loss_plane": avg_plane,
    }


@torch.no_grad()
def validate_one_epoch(model, loader, device, args):
    model.eval()

    total_loss_sum = 0.0
    total_line_sum = 0.0
    total_plane_sum = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        view1, view2 = build_views_from_batch(batch, prefix="val")

        res1, res2 = model(view1, view2)

        loss_total, loss_dict = total_loss_fn(
            res1,
            batch,
            line_weight=args.line_weight,
            plane_weight=args.plane_weight,
            line_pos_weight=args.line_pos_weight,
            line_dice_weight=args.line_dice_weight
        )

        total_loss_sum += loss_dict["loss_total"]
        total_line_sum += loss_dict["loss_line"]
        total_plane_sum += loss_dict["loss_plane"]
        num_batches += 1

        print(
            f"[Val] batch {batch_idx}/{len(loader)} | "
            f"total={loss_dict['loss_total']:.4f}, "
            f"line={loss_dict['loss_line']:.4f}, "
            f"plane={loss_dict['loss_plane']:.4f}"
        )

    avg_loss = total_loss_sum / max(num_batches, 1)
    avg_line = total_line_sum / max(num_batches, 1)
    avg_plane = total_plane_sum / max(num_batches, 1)

    return {
        "loss_total": avg_loss,
        "loss_line": avg_line,
        "loss_plane": avg_plane,
    }


# =========================
# 4. main
# =========================
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(args)

    # -------------------------
    # dataset
    # -------------------------
    full_train_dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split="train",
        train_ratio=args.train_ratio,
        image_size=(512, 512)
    )
    full_val_dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split="val",
        train_ratio=args.train_ratio,
        image_size=(512, 512)
    )

    print(f"Train scenes: {len(full_train_dataset.scenes)}")
    print(f"Val scenes: {len(full_val_dataset.scenes)}")
    print(f"Full train samples: {len(full_train_dataset)}")
    print(f"Full val samples: {len(full_val_dataset)}")

    # subset for smoke test / overfit test
    train_dataset = full_train_dataset
    val_dataset = full_val_dataset

    if args.small_train_size > 0:
        train_dataset = Subset(
            full_train_dataset,
            list(range(min(len(full_train_dataset), args.small_train_size)))
        )

    if args.small_val_size > 0:
        val_dataset = Subset(
            full_val_dataset,
            list(range(min(len(full_val_dataset), args.small_val_size)))
        )

    print(f"Subset train samples: {len(train_dataset)}")
    print(f"Subset val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_train,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # -------------------------
    # model
    # -------------------------
    dust3r_backbone = build_dust3r_backbone(args.weights_path, device=device)
    model = LightReconModel(dust3r_backbone, hidden_dim=768).to(device)

    if args.freeze_encoder:
        freeze_dust3r_encoder(model)
        print("Encoder frozen.")

    total_params, trainable_params = count_trainable_parameters(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # -------------------------
    # optimizer
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # -------------------------
    # train loop
    # -------------------------
    best_val_loss = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.num_epochs} ==========")

        train_stats = train_one_epoch(model, train_loader, optimizer, device, args)

        print(
            f"Train | total: {train_stats['loss_total']:.4f}, "
            f"line: {train_stats['loss_line']:.4f}, "
            f"plane: {train_stats['loss_plane']:.4f}"
        )

        # 每轮都保存 latest
        latest_path = os.path.join(args.save_dir, "latest.pth")
        save_checkpoint(model, optimizer, epoch, latest_path)

        # 可选验证
        if args.run_val:
            val_stats = validate_one_epoch(model, val_loader, device, args)

            print(
                f"Val   | total: {val_stats['loss_total']:.4f}, "
                f"line: {val_stats['loss_line']:.4f}, "
                f"plane: {val_stats['loss_plane']:.4f}"
            )

            if val_stats["loss_total"] < best_val_loss:
                best_val_loss = val_stats["loss_total"]
                best_path = os.path.join(args.save_dir, "best.pth")
                save_checkpoint(model, optimizer, epoch, best_path)
                print(f"Best checkpoint updated at epoch {epoch}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()