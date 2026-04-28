import os
import sys
import argparse

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")

if DUST3R_REPO_ROOT not in sys.path:
    sys.path.insert(0, DUST3R_REPO_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel


def parse_args():
    parser = argparse.ArgumentParser("Visualize LightRecon3D predictions")

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
        "--ckpt_path",
        type=str,
        default="/data/zhucy23u/checkpoints/lightrecon/plane_embedding_256_stable/latest.pth",
    )

    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--image_size", type=int, default=512)

    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)

    parser.add_argument("--line_threshold", type=float, default=0.3)

    parser.add_argument(
        "--save_path",
        type=str,
        default="/data/zhucy23u/logs/vis_plane_embedding.png",
    )

    return parser.parse_args()


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def build_views_from_batch(batch, prefix="vis"):
    bsz = batch["img"].shape[0]

    view1 = {
        "img": batch["img"],
        "instance": [f"{prefix}_{i}" for i in range(bsz)],
    }

    view2 = {
        "img": batch["img"],
        "instance": [f"{prefix}_{i}" for i in range(bsz)],
    }

    return view1, view2


def safe_load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_state_dict_flexible(model, ckpt):
    """
    只加载 shape 对得上的参数。
    这样旧 checkpoint / 新模型部分不一致时不会直接炸。
    """
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model_dict = model.state_dict()

    filtered = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    return missing, unexpected, skipped


def tensor_image_to_numpy(img_tensor):
    """
    img_tensor:
        [3, H, W]
    """
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img.clip(0.0, 1.0)
    return img


def embedding_pca_to_rgb(embedding, eps=1e-6):
    """
    将 plane embedding 可视化为 RGB。

    embedding:
        [C, H, W]

    return:
        [H, W, 3], numpy, range [0, 1]
    """
    if embedding.ndim != 3:
        raise ValueError(f"Expected embedding [C,H,W], got {embedding.shape}")

    C, H, W = embedding.shape

    # [C,H,W] -> [H*W,C]
    x = embedding.detach().float().permute(1, 2, 0).reshape(-1, C)

    finite = torch.isfinite(x).all(dim=1)

    if finite.sum() < 10:
        return torch.zeros((H, W, 3)).numpy()

    x_valid = x[finite]

    # center
    mean = x_valid.mean(dim=0, keepdim=True)
    x_centered = x_valid - mean

    # covariance [C,C]
    cov = x_centered.T @ x_centered / (x_centered.shape[0] + eps)

    eigvals, eigvecs = torch.linalg.eigh(cov)

    # 取最大三个主成分
    k = min(3, C)
    pcs = eigvecs[:, -k:]

    proj_valid = x_centered @ pcs  # [N,k]

    if k < 3:
        pad = torch.zeros(
            proj_valid.shape[0],
            3 - k,
            device=proj_valid.device,
            dtype=proj_valid.dtype,
        )
        proj_valid = torch.cat([proj_valid, pad], dim=1)

    proj = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
    proj[finite] = proj_valid

    proj = proj.reshape(H, W, 3)

    # robust min-max normalize per channel
    out = torch.zeros_like(proj)

    for c in range(3):
        channel = proj[..., c]
        valid_channel = channel[torch.isfinite(channel)]

        if valid_channel.numel() == 0:
            continue

        lo = torch.quantile(valid_channel, 0.01)
        hi = torch.quantile(valid_channel, 0.99)

        out[..., c] = (channel - lo) / (hi - lo + eps)

    out = out.clamp(0.0, 1.0)

    return out.cpu().numpy()


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("LightRecon3D Visualization")
    print("=" * 80)
    print(f"device       : {device}")
    print(f"root_dir     : {args.root_dir}")
    print(f"weights_path : {args.weights_path}")
    print(f"ckpt_path    : {args.ckpt_path}")
    print(f"split        : {args.split}")
    print(f"sample_idx   : {args.sample_idx}")
    print(f"save_path    : {args.save_path}")
    print("=" * 80)

    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
    )

    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        raise IndexError(
            f"sample_idx={args.sample_idx} out of range, dataset size={len(dataset)}"
        )

    sample = dataset[args.sample_idx]

    batch = {
        "img": sample["img"].unsqueeze(0),
        "gt_line": sample["gt_line"].unsqueeze(0),
        "gt_plane": sample["gt_plane"].unsqueeze(0),
    }

    batch = move_batch_to_device(batch, device)
    view1, view2 = build_views_from_batch(batch, prefix=args.split)

    print(f"Dataset size ({args.split}): {len(dataset)}")
    print(f"img shape     : {batch['img'].shape}")
    print(f"gt_line shape : {batch['gt_line'].shape}")
    print(f"gt_plane shape: {batch['gt_plane'].shape}")

    dust3r_backbone = build_dust3r_backbone(
        args.weights_path,
        device=device,
    )

    model = LightReconModel(
        dust3r_backbone=dust3r_backbone,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    ).to(device)

    ckpt = safe_load_checkpoint(args.ckpt_path, device)

    missing, unexpected, skipped = load_state_dict_flexible(model, ckpt)

    epoch = ckpt.get("epoch", "unknown") if isinstance(ckpt, dict) else "unknown"
    print(f"Loaded checkpoint epoch: {epoch}")

    if skipped:
        print(f"[Warning] skipped mismatched keys: {skipped[:10]}")
        if len(skipped) > 10:
            print(f"... and {len(skipped) - 10} more skipped keys")

    if missing:
        print(f"[Warning] missing keys: {missing[:10]}")
        if len(missing) > 10:
            print(f"... and {len(missing) - 10} more missing keys")

    if unexpected:
        print(f"[Warning] unexpected keys: {unexpected[:10]}")
        if len(unexpected) > 10:
            print(f"... and {len(unexpected) - 10} more unexpected keys")

    model.eval()

    res1, res2 = model(view1, view2)

    if "pred_line" not in res1:
        raise KeyError("res1 does not contain pred_line")

    if "pred_plane" not in res1:
        raise KeyError("res1 does not contain pred_plane embedding")

    img = tensor_image_to_numpy(batch["img"][0])

    gt_line = batch["gt_line"][0]
    if gt_line.ndim == 3:
        gt_line = gt_line[0]
    gt_line = gt_line.detach().cpu().numpy()

    gt_plane = batch["gt_plane"][0].detach().cpu().numpy()

    pred_line_logits = res1["pred_line"][0, 0]
    pred_line_prob = torch.sigmoid(pred_line_logits).detach().cpu().numpy()
    pred_line_bin = (pred_line_prob > args.line_threshold).astype("float32")

    pred_plane_embedding = res1["pred_plane"][0]  # [C,H,W]
    pred_plane_rgb = embedding_pca_to_rgb(pred_plane_embedding)

    print(f"pred_line shape : {res1['pred_line'].shape}")
    print(f"pred_plane shape: {res1['pred_plane'].shape}")

    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.title("Input RGB")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("GT Line")
    plt.imshow(gt_line, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Pred Line Prob")
    plt.imshow(pred_line_prob, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title(f"Pred Line Binary > {args.line_threshold}")
    plt.imshow(pred_line_bin, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("GT Plane Instance")
    plt.imshow(gt_plane, cmap="jet")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("Pred Plane Embedding PCA")
    plt.imshow(pred_plane_rgb)
    plt.axis("off")

    plt.tight_layout()

    save_dir = os.path.dirname(args.save_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to: {args.save_path}")


if __name__ == "__main__":
    main()