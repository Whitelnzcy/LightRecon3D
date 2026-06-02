import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PLANE_COLORS = np.asarray(
    [
        [229, 57, 53],
        [30, 136, 229],
        [67, 160, 71],
        [251, 140, 0],
        [142, 36, 170],
        [0, 137, 123],
        [109, 76, 65],
        [57, 73, 171],
        [198, 40, 40],
        [0, 121, 107],
    ],
    dtype=np.uint8,
)


def sample_case(path, max_points, seed):
    raw = np.load(path)
    points = raw["points"].astype(np.float32)
    colors = raw["colors"].astype(np.uint8)
    rng = np.random.default_rng(seed)
    if max_points > 0 and len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    center = points.mean(axis=0, keepdims=True).astype(np.float32)
    scale = np.linalg.norm(points - center, axis=1).max()
    scale = max(float(scale), 1e-6)
    points_norm = ((points - center) / scale).astype(np.float32)
    colors_float = colors.astype(np.float32) / 255.0
    radius = np.linalg.norm(points_norm, axis=1, keepdims=True).astype(np.float32)
    features = np.concatenate([points_norm, colors_float, radius], axis=1).astype(np.float32)
    return {
        "path": str(path),
        "stem": Path(path).name.replace("_full_pointcloud_editable_planes_data.npz", ""),
        "points": points,
        "colors": colors,
        "points_norm": points_norm,
        "features": features,
        "center": center[0],
        "scale": scale,
    }


class MultiSamplePlaneTokens(nn.Module):
    def __init__(self, num_samples, num_planes, point_feature_dim=7, hidden_dim=192):
        super().__init__()
        self.num_samples = num_samples
        self.num_planes = num_planes
        self.normal_raw = nn.Parameter(torch.randn(num_samples, num_planes, 3) * 0.2)
        self.offset = nn.Parameter(torch.zeros(num_samples, num_planes))
        self.logit = nn.Parameter(torch.zeros(num_samples, num_planes))
        assign_input_dim = point_feature_dim + 3 + 1 + 1
        self.assignment_head = nn.Sequential(
            nn.Linear(assign_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_one(self, sample_idx, points, features, temperature, distance_logit_weight):
        normals = F.normalize(self.normal_raw[sample_idx], dim=-1)
        offsets = self.offset[sample_idx]
        dists = torch.abs(points @ normals.t() + offsets.view(1, -1))
        n_points = points.shape[0]
        n_planes = normals.shape[0]
        point_feat = features[:, None, :].expand(n_points, n_planes, -1)
        normal_feat = normals[None, :, :].expand(n_points, n_planes, -1)
        offset_feat = offsets.view(1, n_planes, 1).expand(n_points, n_planes, 1)
        pair_feat = torch.cat([point_feat, normal_feat, offset_feat, dists.unsqueeze(-1)], dim=-1)
        logits = self.assignment_head(pair_feat).squeeze(-1)
        logits = logits - distance_logit_weight * dists / temperature + self.logit[sample_idx].view(1, -1)
        assign = F.softmax(logits, dim=-1)
        return normals, offsets, dists, assign


def entropy(assign, eps=1e-8):
    return -(assign * (assign + eps).log()).sum(dim=-1)


def diversity_loss(normals, offsets, margin_normal=0.18, margin_offset=0.04):
    k = normals.shape[0]
    if k <= 1:
        return normals.sum() * 0.0
    sim = torch.abs(normals @ normals.t())
    off_dist = torch.abs(offsets[:, None] - offsets[None, :])
    eye = torch.eye(k, device=normals.device, dtype=torch.bool)
    sim = sim[~eye]
    off_dist = off_dist[~eye]
    return F.relu(sim - (1.0 - margin_normal)).mean() + F.relu(margin_offset - off_dist).mean()


def balance_loss(assign):
    coverage = assign.mean(dim=0)
    target = torch.full_like(coverage, 1.0 / max(1, coverage.numel()))
    return F.smooth_l1_loss(coverage, target), coverage


def coverage_loss(assign, min_coverage):
    coverage = assign.mean(dim=0)
    return F.relu(min_coverage - coverage).mean(), coverage


def confidence_separation_loss(assign, margin):
    if assign.shape[1] <= 1:
        return assign.sum() * 0.0
    top2 = torch.topk(assign, k=2, dim=-1).values
    return F.relu(margin - (top2[:, 0] - top2[:, 1])).mean()


def compute_one_loss(model, sample_idx, points, features, args, temperature):
    normals, offsets, dists, assign = model.forward_one(
        sample_idx,
        points,
        features,
        temperature,
        args.distance_logit_weight,
    )
    soft_fit = (assign * dists).sum(dim=-1).mean()
    hard_fit = dists.min(dim=-1).values.mean()
    ent = entropy(assign).mean()
    div = diversity_loss(normals, offsets, args.diversity_normal_margin, args.diversity_offset_margin)
    cov_loss, coverage = coverage_loss(assign, args.min_coverage)
    bal_loss, _ = balance_loss(assign)
    sep_loss = confidence_separation_loss(assign, args.assignment_margin)
    confidence = 1.0 - entropy(assign) / np.log(args.num_planes)
    confident_fit = (confidence.detach() * dists.min(dim=-1).values).mean()
    loss = (
        args.fit_weight * soft_fit
        + args.hard_fit_weight * hard_fit
        + args.entropy_weight * ent
        + args.diversity_weight * div
        + args.coverage_weight * cov_loss
        + args.balance_weight * bal_loss
        + args.assignment_margin_weight * sep_loss
        + args.confident_fit_weight * confident_fit
    )
    stats = {
        "loss": loss,
        "soft_fit": soft_fit.detach(),
        "hard_fit": hard_fit.detach(),
        "entropy": ent.detach(),
        "diversity": div.detach(),
        "coverage_loss": cov_loss.detach(),
        "balance_loss": bal_loss.detach(),
        "confidence": confidence.mean().detach(),
        "coverage": coverage.detach(),
    }
    return loss, stats


def export_case(model, case, sample_idx, output_dir, args, device):
    points = torch.from_numpy(case["points_norm"]).to(device)
    features = torch.from_numpy(case["features"]).to(device)
    with torch.no_grad():
        normals, offsets_norm, dists, assign = model.forward_one(
            sample_idx,
            points,
            features,
            args.min_temperature,
            args.distance_logit_weight,
        )
        hard = assign.argmax(dim=-1)
    normals_np = normals.detach().cpu().numpy()
    offsets_norm_np = offsets_norm.detach().cpu().numpy()
    hard_np = hard.detach().cpu().numpy().astype(np.int32)
    offsets_world = []
    for n, d_norm in zip(normals_np, offsets_norm_np):
        offsets_world.append(float(d_norm * case["scale"] - float(np.dot(n, case["center"]))))

    fit_per_plane = []
    dists_np = dists.detach().cpu().numpy()
    for i in range(args.num_planes):
        mask = hard_np == i
        fit_per_plane.append(float(dists_np[mask, i].mean()) if mask.any() else None)
    params = []
    for i in range(args.num_planes):
        params.append(
            {
                "id": int(i),
                "normal": [float(x) for x in normals_np[i]],
                "offset": offsets_world[i],
                "offset_normalized": float(offsets_norm_np[i]),
                "assigned_point_count": int(np.sum(hard_np == i)),
                "assigned_ratio": float(np.mean(hard_np == i)),
                "mean_abs_distance_normalized": fit_per_plane[i],
            }
        )
    learned_colors = PLANE_COLORS[hard_np % len(PLANE_COLORS)]
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{case['stem']}_multisample_learned_plane_tokens.json"
    npz_path = output_dir / f"{case['stem']}_multisample_learned_plane_tokens_assignment.npz"
    summary = {
        "input_npz": case["path"],
        "num_points_used": int(len(case["points"])),
        "num_planes": int(args.num_planes),
        "center": [float(x) for x in case["center"]],
        "scale": float(case["scale"]),
        "planes": params,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        points=case["points"].astype(np.float32),
        colors=learned_colors.astype(np.uint8),
        original_colors=case["colors"].astype(np.uint8),
        assignment=hard_np,
        plane_normals=normals_np.astype(np.float32),
        plane_offsets=np.asarray(offsets_world, dtype=np.float32),
        plane_offsets_normalized=offsets_norm_np.astype(np.float32),
    )
    return json_path, npz_path, summary


def main():
    parser = argparse.ArgumentParser("Multi-sample unsupervised plane-token decomposition")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*_full_pointcloud_editable_planes_data.npz")
    parser.add_argument("--sample_glob", default=None, help="Optional substring filter such as val_00003")
    parser.add_argument("--num_planes", type=int, default=4)
    parser.add_argument("--max_points_per_sample", type=int, default=30000)
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--assignment_hidden_dim", type=int, default=192)
    parser.add_argument("--distance_logit_weight", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--temperature_decay", type=float, default=0.999)
    parser.add_argument("--min_temperature", type=float, default=0.012)
    parser.add_argument("--fit_weight", type=float, default=1.0)
    parser.add_argument("--hard_fit_weight", type=float, default=0.2)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--diversity_weight", type=float, default=0.04)
    parser.add_argument("--coverage_weight", type=float, default=0.35)
    parser.add_argument("--balance_weight", type=float, default=0.16)
    parser.add_argument("--assignment_margin_weight", type=float, default=0.03)
    parser.add_argument("--confident_fit_weight", type=float, default=0.05)
    parser.add_argument("--min_coverage", type=float, default=0.08)
    parser.add_argument("--assignment_margin", type=float, default=0.12)
    parser.add_argument("--diversity_normal_margin", type=float, default=0.18)
    parser.add_argument("--diversity_offset_margin", type=float, default=0.04)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    paths = sorted(Path(args.input_dir).glob(args.pattern))
    if args.sample_glob:
        paths = [p for p in paths if args.sample_glob in p.name]
    if not paths:
        raise FileNotFoundError(f"No npz files matched under {args.input_dir}")
    cases = [sample_case(p, args.max_points_per_sample, args.seed + i * 97) for i, p in enumerate(paths)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = [
        (
            torch.from_numpy(case["points_norm"]).to(device),
            torch.from_numpy(case["features"]).to(device),
        )
        for case in cases
    ]
    model = MultiSamplePlaneTokens(
        len(cases),
        args.num_planes,
        point_feature_dim=tensors[0][1].shape[1],
        hidden_dim=args.assignment_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    history = []
    for step in range(1, args.steps + 1):
        temperature = max(args.min_temperature, args.temperature * (args.temperature_decay ** step))
        sample_ids = rng.choice(len(cases), size=min(args.sample_batch_size, len(cases)), replace=False)
        losses = []
        stat_rows = []
        for sid in sample_ids:
            points, features = tensors[int(sid)]
            loss, stats = compute_one_loss(model, int(sid), points, features, args, temperature)
            losses.append(loss)
            stat_rows.append(stats)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "soft_fit": float(torch.stack([s["soft_fit"] for s in stat_rows]).mean().cpu()),
            "hard_fit": float(torch.stack([s["hard_fit"] for s in stat_rows]).mean().cpu()),
            "entropy": float(torch.stack([s["entropy"] for s in stat_rows]).mean().cpu()),
            "confidence": float(torch.stack([s["confidence"] for s in stat_rows]).mean().cpu()),
            "balance_loss": float(torch.stack([s["balance_loss"] for s in stat_rows]).mean().cpu()),
            "temperature": float(temperature),
        }
        history.append(row)
        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:04d} loss={row['loss']:.5f} fit={row['soft_fit']:.5f} "
                f"hard={row['hard_fit']:.5f} ent={row['entropy']:.4f} "
                f"bal={row['balance_loss']:.4f} conf={row['confidence']:.3f}"
            )

    output_dir = Path(args.output_dir)
    exported = []
    for i, case in enumerate(cases):
        json_path, npz_path, summary = export_case(model, case, i, output_dir, args, device)
        exported.append({"json": str(json_path), "npz": str(npz_path), "summary": summary})

    overview = {
        "input_dir": args.input_dir,
        "num_samples": len(cases),
        "num_planes": args.num_planes,
        "max_points_per_sample": args.max_points_per_sample,
        "history": history,
        "exported": exported,
    }
    overview_path = output_dir / "multisample_learned_plane_tokens_summary.json"
    overview_path.write_text(json.dumps(overview, indent=2), encoding="utf-8")
    print(overview_path)
    print(f"samples={len(cases)}")


if __name__ == "__main__":
    main()
