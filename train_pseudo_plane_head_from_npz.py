import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class PlanePseudoLabelDataset(Dataset):
    def __init__(self, npz_paths, points_per_sample=8192, seed=42):
        self.npz_paths = [Path(p) for p in npz_paths]
        self.points_per_sample = int(points_per_sample)
        self.seed = int(seed)
        if not self.npz_paths:
            raise ValueError("No npz files were provided")

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx):
        path = self.npz_paths[idx]
        raw = np.load(path)
        points = raw["points"].astype(np.float32)
        colors = raw["colors"].astype(np.float32) / 255.0
        point_plane_ids = raw["point_plane_ids"].astype(np.int64)
        plane_ids = raw["plane_ids"].astype(np.int64)
        plane_normals = raw["plane_normals"].astype(np.float32)
        plane_offsets = raw["plane_offsets"].astype(np.float32)

        rng = np.random.default_rng(self.seed + idx)
        if len(points) > self.points_per_sample:
            choice = rng.choice(len(points), size=self.points_per_sample, replace=False)
        else:
            choice = np.arange(len(points))

        points = points[choice]
        colors = colors[choice]
        point_plane_ids = point_plane_ids[choice]

        center = points.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(points - center, axis=1).max()
        scale = max(float(scale), 1e-6)
        points_norm = (points - center) / scale

        id_to_plane = {int(pid): i for i, pid in enumerate(plane_ids)}
        target_normals = np.zeros((len(points), 3), dtype=np.float32)
        target_offsets = np.zeros((len(points), 1), dtype=np.float32)
        valid = np.zeros((len(points), 1), dtype=np.float32)

        for i, pid in enumerate(point_plane_ids):
            if int(pid) < 0 or int(pid) not in id_to_plane:
                continue
            plane_idx = id_to_plane[int(pid)]
            n = plane_normals[plane_idx]
            d = float(plane_offsets[plane_idx])
            # Convert world-space plane offset to the normalized coordinate frame.
            d_norm = float(np.dot(n, center[0]) + d) / scale
            target_normals[i] = n
            target_offsets[i, 0] = d_norm
            valid[i, 0] = 1.0

        inputs = np.concatenate([points_norm, colors], axis=1)
        return {
            "input": torch.from_numpy(inputs).float(),
            "points": torch.from_numpy(points_norm).float(),
            "target_normal": torch.from_numpy(target_normals).float(),
            "target_offset": torch.from_numpy(target_offsets).float(),
            "valid": torch.from_numpy(valid).float(),
            "path": str(path),
        }


class PseudoPlaneHead(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, x):
        pred = self.net(x)
        normal = F.normalize(pred[..., :3], dim=-1)
        offset = pred[..., 3:4]
        valid_logit = pred[..., 4:5]
        return normal, offset, valid_logit


def list_npz(input_dir, pattern):
    return sorted(str(p) for p in Path(input_dir).glob(pattern))


def split_paths(paths, val_ratio):
    if len(paths) == 1:
        return paths, paths
    n_val = max(1, int(round(len(paths) * val_ratio)))
    n_val = min(n_val, len(paths) - 1)
    return paths[:-n_val], paths[-n_val:]


def masked_mean(x, mask, eps=1e-6):
    return (x * mask).sum() / mask.sum().clamp_min(eps)


def compute_loss(batch, model, args):
    inputs = batch["input"]
    points = batch["points"]
    target_normal = batch["target_normal"]
    target_offset = batch["target_offset"]
    valid = batch["valid"]

    pred_normal, pred_offset, pred_valid_logit = model(inputs)
    sign = torch.where(
        (pred_normal * target_normal).sum(dim=-1, keepdim=True) < 0,
        -1.0,
        1.0,
    )
    pred_normal = pred_normal * sign
    pred_offset = pred_offset * sign

    cos = (pred_normal * target_normal).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    normal_loss = masked_mean(1.0 - cos, valid)
    offset_loss = masked_mean(F.smooth_l1_loss(pred_offset, target_offset, reduction="none"), valid)
    plane_dist = torch.abs((points * pred_normal).sum(dim=-1, keepdim=True) + pred_offset)
    plane_dist_loss = masked_mean(torch.clamp(plane_dist, max=args.plane_dist_clip), valid)
    valid_loss = F.binary_cross_entropy_with_logits(pred_valid_logit, valid)
    loss = (
        args.normal_weight * normal_loss
        + args.offset_weight * offset_loss
        + args.plane_dist_weight * plane_dist_loss
        + args.valid_weight * valid_loss
    )

    with torch.no_grad():
        angle_deg = torch.rad2deg(torch.acos(cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)))
        normal_angle = masked_mean(angle_deg, valid)
        offset_mae = masked_mean(torch.abs(pred_offset - target_offset), valid)
        valid_prob = torch.sigmoid(pred_valid_logit)
        valid_pred = valid_prob > 0.5
        valid_acc = (valid_pred == (valid > 0.5)).float().mean()

    stats = {
        "loss": float(loss.detach().cpu()),
        "normal_loss": float(normal_loss.detach().cpu()),
        "offset_loss": float(offset_loss.detach().cpu()),
        "plane_dist_loss": float(plane_dist_loss.detach().cpu()),
        "valid_loss": float(valid_loss.detach().cpu()),
        "normal_angle_deg": float(normal_angle.detach().cpu()),
        "offset_mae": float(offset_mae.detach().cpu()),
        "valid_acc": float(valid_acc.detach().cpu()),
        "valid_ratio": float(valid.mean().detach().cpu()),
    }
    return loss, stats


def run_epoch(loader, model, optimizer, device, args, train):
    model.train(train)
    sums = {}
    count = 0
    for batch in loader:
        batch = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        if train:
            optimizer.zero_grad(set_to_none=True)
        loss, stats = compute_loss(batch, model, args)
        if train:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        for key, value in stats.items():
            sums[key] = sums.get(key, 0.0) + value
        count += 1
    return {k: v / max(count, 1) for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser("Train a pseudo-label plane primitive head from full pointcloud npz files")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*_full_pointcloud_editable_planes_data.npz")
    parser.add_argument("--points_per_sample", type=int, default=8192)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal_weight", type=float, default=1.0)
    parser.add_argument("--offset_weight", type=float, default=0.25)
    parser.add_argument("--plane_dist_weight", type=float, default=0.2)
    parser.add_argument("--valid_weight", type=float, default=0.1)
    parser.add_argument("--plane_dist_clip", type=float, default=0.25)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = list_npz(args.input_dir, args.pattern)
    train_paths, val_paths = split_paths(paths, args.val_ratio)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = PlanePseudoLabelDataset(train_paths, args.points_per_sample, seed=args.seed)
    val_set = PlanePseudoLabelDataset(val_paths, args.points_per_sample, seed=args.seed + 10000)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PseudoPlaneHead(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val = math.inf
    best_path = output_dir / "best_pseudo_plane_head.pt"
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(train_loader, model, optimizer, device, args, train=True)
        val_stats = run_epoch(val_loader, model, optimizer, device, args, train=False)
        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_angle={val_stats['normal_angle_deg']:.2f} "
            f"val_offset={val_stats['offset_mae']:.4f} "
            f"val_dist={val_stats['plane_dist_loss']:.4f} "
            f"valid_acc={val_stats['valid_acc']:.3f}"
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "train_paths": train_paths,
                    "val_paths": val_paths,
                    "epoch": epoch,
                    "val": val_stats,
                },
                best_path,
            )

    summary = {
        "input_dir": args.input_dir,
        "num_files": len(paths),
        "train_files": train_paths,
        "val_files": val_paths,
        "best_checkpoint": str(best_path),
        "best_val_loss": best_val,
        "history": history,
    }
    summary_path = output_dir / "pseudo_plane_head_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(best_path)
    print(summary_path)


if __name__ == "__main__":
    main()
