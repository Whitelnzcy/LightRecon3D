import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")

if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

if DUST3R_REPO_ROOT in sys.path:
    sys.path.remove(DUST3R_REPO_ROOT)
sys.path.insert(1, DUST3R_REPO_ROOT)

from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel
from train import build_views_from_batch


def safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(weights_path, ckpt_path, device, hidden_dim=768, plane_embed_dim=16):
    backbone = build_dust3r_backbone(weights_path, device=device)

    model = LightReconModel(
        dust3r_backbone=backbone,
        hidden_dim=hidden_dim,
        plane_embed_dim=plane_embed_dim,
    ).to(device)

    ckpt = safe_load(ckpt_path, device)
    state = ckpt.get("model", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[Load] ckpt: {ckpt_path}")
    print(f"[Load] missing keys   : {len(missing)}")
    print(f"[Load] unexpected keys: {len(unexpected)}")

    model.eval()
    return model


def tensor_img_to_numpy(img):
    """
    img: [3, H, W], usually normalized to roughly [-1, 1] or [0, 1].
    Return [H, W, 3] in [0, 1].
    """
    img = img.detach().float().cpu()

    if img.ndim != 3:
        raise ValueError(f"Expected image [3,H,W], got {tuple(img.shape)}")

    img = img.permute(1, 2, 0).numpy()

    # Robust visualization normalization.
    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99)
    img = (img - vmin) / max(vmax - vmin, 1e-8)
    img = np.clip(img, 0.0, 1.0)

    return img


def resize_gt_plane(gt_plane, size):
    """
    gt_plane: [B,H,W] or [B,1,H,W]
    """
    if gt_plane.ndim == 4 and gt_plane.shape[1] == 1:
        gt_plane = gt_plane[:, 0]

    if gt_plane.shape[-2:] == size:
        return gt_plane.long()

    x = gt_plane.unsqueeze(1).float()
    x = F.interpolate(x, size=size, mode="nearest")
    return x[:, 0].long()


def resize_label_np(label, size):
    """
    label: [H,W] numpy int
    size: (H,W)
    """
    x = torch.from_numpy(label).long()[None, None].float()
    y = F.interpolate(x, size=size, mode="nearest")
    return y[0, 0].long().numpy()


def get_plane_embedding(res, prefer_lowres=True):
    if prefer_lowres and "pred_plane_lowres" in res:
        return res["pred_plane_lowres"]
    if "pred_plane" in res:
        return res["pred_plane"]
    if "pred_plane_embedding" in res:
        return res["pred_plane_embedding"]
    raise KeyError(f"No plane embedding found. keys={list(res.keys())}")


def colorize_label(label, ignore_ids=(-1, 255)):
    """
    label: [H,W] integer map.
    Return RGB [H,W,3].
    """
    label = label.astype(np.int64)
    out = np.zeros((*label.shape, 3), dtype=np.float32)

    valid = np.ones(label.shape, dtype=bool)
    for ig in ignore_ids:
        valid &= label != ig

    ids = np.unique(label[valid])

    # Deterministic pseudo-colors.
    rng = np.random.default_rng(12345)
    colors = rng.uniform(0.15, 0.95, size=(max(len(ids), 1), 3)).astype(np.float32)

    for i, pid in enumerate(ids):
        out[label == pid] = colors[i % len(colors)]

    return out


def overlay(rgb, color_map, alpha=0.45):
    rgb = np.clip(rgb, 0.0, 1.0)
    color_map = np.clip(color_map, 0.0, 1.0)
    return np.clip((1 - alpha) * rgb + alpha * color_map, 0.0, 1.0)


def oracle_kmeans_cluster(
    emb_chw,
    gt_plane_hw,
    max_pixels=8192,
    min_plane_pixels=4,
    normalize_emb=True,
    random_state=0,
):
    """
    emb_chw: torch [C,H,W]
    gt_plane_hw: torch [H,W] at same resolution as emb
    return pred_cluster [H,W], gt_used [H,W], info dict
    """
    emb = emb_chw.detach().float().cpu()
    gt = gt_plane_hw.detach().cpu().long()

    c, h, w = emb.shape

    emb_np = emb.permute(1, 2, 0).numpy()
    gt_np = gt.numpy().astype(np.int64)

    valid = (gt_np >= 0) & (gt_np != 255)

    gt_ids = []
    for gid in np.unique(gt_np[valid]):
        cnt = int((gt_np == gid).sum())
        if cnt >= min_plane_pixels:
            gt_ids.append(int(gid))

    if len(gt_ids) < 2:
        raise RuntimeError(f"Not enough valid gt planes: {gt_ids}")

    valid = np.zeros_like(gt_np, dtype=bool)
    for gid in gt_ids:
        valid |= gt_np == gid

    x_all = emb_np[valid]
    coords_all = np.argwhere(valid)

    # Sample pixels for fitting KMeans.
    if x_all.shape[0] > max_pixels:
        sample_idx = np.linspace(0, x_all.shape[0] - 1, max_pixels).astype(np.int64)
        x_fit = x_all[sample_idx]
    else:
        x_fit = x_all

    if normalize_emb:
        norm = np.linalg.norm(x_fit, axis=1, keepdims=True)
        x_fit = x_fit / np.maximum(norm, 1e-8)

    k = len(gt_ids)

    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    km.fit(x_fit)

    # Predict all valid pixels.
    x_pred = x_all
    if normalize_emb:
        norm = np.linalg.norm(x_pred, axis=1, keepdims=True)
        x_pred = x_pred / np.maximum(norm, 1e-8)

    pred_valid = km.predict(x_pred)

    pred_map = np.full((h, w), fill_value=-1, dtype=np.int64)
    pred_map[valid] = pred_valid.astype(np.int64)

    info = {
        "K": k,
        "gt_ids": gt_ids,
        "num_valid_pixels": int(x_all.shape[0]),
        "num_fit_pixels": int(x_fit.shape[0]),
    }

    return pred_map, gt_np, info


def save_panel(rgb, gt_color, pred_color, gt_overlay, pred_overlay, save_path, title):
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    panels = [
        (rgb, "Input RGB"),
        (gt_color, "GT Plane"),
        (pred_color, "Pred Cluster"),
        (gt_overlay, "GT Overlay"),
        (pred_overlay, "Pred Overlay"),
    ]

    for ax, (img, name) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Visualize plane embedding clustering")

    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--ckpt_path", required=True)

    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--sample_idx", type=int, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)

    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)

    parser.add_argument("--prefer_lowres_plane", action="store_true")
    parser.add_argument("--min_plane_pixels", type=int, default=4)
    parser.add_argument("--max_pixels", type=int, default=8192)
    parser.add_argument("--no_normalize_emb", action="store_true")

    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Visualize Plane Clustering")
    print("=" * 80)
    print(f"sample_idx : {args.sample_idx}")
    print(f"ckpt_path  : {args.ckpt_path}")
    print(f"output_dir : {args.output_dir}")
    print("=" * 80)

    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
    )

    sample = dataset[args.sample_idx]

    batch = {
        "img": sample["img"].unsqueeze(0).to(device),
        "gt_line": sample["gt_line"].unsqueeze(0).to(device),
        "gt_plane": sample["gt_plane"].unsqueeze(0).to(device),
    }

    model = load_model(
        weights_path=args.weights_path,
        ckpt_path=args.ckpt_path,
        device=device,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    )

    view1, view2 = build_views_from_batch(batch, prefix=f"vis_cluster_{args.split}_{args.sample_idx}")

    res1, _ = model(view1, view2)

    emb = get_plane_embedding(res1, prefer_lowres=args.prefer_lowres_plane)
    gt_low = resize_gt_plane(batch["gt_plane"], size=emb.shape[-2:])

    pred_low, gt_low_np, info = oracle_kmeans_cluster(
        emb_chw=emb[0],
        gt_plane_hw=gt_low[0],
        max_pixels=args.max_pixels,
        min_plane_pixels=args.min_plane_pixels,
        normalize_emb=not args.no_normalize_emb,
        random_state=args.sample_idx,
    )

    rgb = tensor_img_to_numpy(batch["img"][0])

    # Upsample low-res labels to RGB resolution for visualization.
    h_img, w_img = rgb.shape[:2]
    gt_vis = resize_label_np(gt_low_np, size=(h_img, w_img))
    pred_vis = resize_label_np(pred_low, size=(h_img, w_img))

    gt_color = colorize_label(gt_vis)
    pred_color = colorize_label(pred_vis)

    gt_overlay = overlay(rgb, gt_color)
    pred_overlay = overlay(rgb, pred_color)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.split}_{args.sample_idx}"

    save_panel(
        rgb=rgb,
        gt_color=gt_color,
        pred_color=pred_color,
        gt_overlay=gt_overlay,
        pred_overlay=pred_overlay,
        save_path=output_dir / f"{prefix}_plane_cluster_panel.png",
        title=f"{prefix} | K={info['K']} | valid={info['num_valid_pixels']} | fit={info['num_fit_pixels']}",
    )

    plt.imsave(output_dir / f"{prefix}_rgb.png", rgb)
    plt.imsave(output_dir / f"{prefix}_gt_plane.png", gt_color)
    plt.imsave(output_dir / f"{prefix}_pred_cluster.png", pred_color)
    plt.imsave(output_dir / f"{prefix}_gt_overlay.png", gt_overlay)
    plt.imsave(output_dir / f"{prefix}_pred_overlay.png", pred_overlay)

    # Save raw label maps as npy for later analysis.
    np.save(output_dir / f"{prefix}_gt_plane_lowres.npy", gt_low_np)
    np.save(output_dir / f"{prefix}_pred_cluster_lowres.npy", pred_low)

    print("[Done]")
    print(f"K                : {info['K']}")
    print(f"valid pixels     : {info['num_valid_pixels']}")
    print(f"fit pixels       : {info['num_fit_pixels']}")
    print(f"panel saved to   : {output_dir / f'{prefix}_plane_cluster_panel.png'}")


if __name__ == "__main__":
    main()