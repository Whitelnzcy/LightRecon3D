import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_multisample_unsupervised_plane_tokens import (
    PLANE_COLORS,
    confidence_separation_loss,
    coverage_loss,
    dead_token_loss,
    diversity_loss,
    entropy,
    local_smoothness_loss,
    sample_case,
    trimmed_mean,
)


class AmortizedPlaneTokenHead(nn.Module):
    """Predict per-sample plane tokens from the input point cloud instead of optimizing them per sample."""

    def __init__(self, num_planes, point_feature_dim=7, hidden_dim=192, context_dim=256):
        super().__init__()
        self.num_planes = num_planes
        self.point_encoder = nn.Sequential(
            nn.Linear(point_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
            nn.GELU(),
        )
        self.token_head = nn.Sequential(
            nn.Linear(context_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_planes * 5),
        )
        assign_input_dim = point_feature_dim + 3 + 1 + 1
        self.assignment_head = nn.Sequential(
            nn.Linear(assign_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def predict_tokens(self, features):
        encoded = self.point_encoder(features)
        pooled = torch.cat([encoded.mean(dim=0), encoded.max(dim=0).values], dim=-1)
        raw = self.token_head(pooled).view(self.num_planes, 5)
        normals = F.normalize(raw[:, :3], dim=-1)
        offsets = raw[:, 3]
        token_logits = raw[:, 4]
        return normals, offsets, token_logits

    def forward_one(self, points, features, temperature, distance_logit_weight):
        normals, offsets, token_logits = self.predict_tokens(features)
        dists = torch.abs(points @ normals.t() + offsets.view(1, -1))
        n_points = points.shape[0]
        n_planes = normals.shape[0]
        point_feat = features[:, None, :].expand(n_points, n_planes, -1)
        normal_feat = normals[None, :, :].expand(n_points, n_planes, -1)
        offset_feat = offsets.view(1, n_planes, 1).expand(n_points, n_planes, 1)
        pair_feat = torch.cat([point_feat, normal_feat, offset_feat, dists.unsqueeze(-1)], dim=-1)
        logits = self.assignment_head(pair_feat).squeeze(-1)
        logits = logits - distance_logit_weight * dists / temperature + token_logits.view(1, -1)
        assign = F.softmax(logits, dim=-1)
        return normals, offsets, dists, assign


def compute_one_loss(model, points, features, smooth_i, smooth_j, smooth_w, args, temperature):
    normals, offsets, dists, assign = model.forward_one(
        points,
        features,
        temperature,
        args.distance_logit_weight,
    )
    soft_residual = (assign * dists).sum(dim=-1)
    soft_fit = trimmed_mean(soft_residual, args.trimmed_fit_ratio)
    hard_fit = dists.min(dim=-1).values.mean()
    ent = entropy(assign).mean()
    div = diversity_loss(normals, offsets, args.diversity_normal_margin, args.diversity_offset_margin)
    cov_loss, coverage = coverage_loss(assign, args.min_coverage)
    dead_loss, _ = dead_token_loss(assign, args.dead_token_min_coverage)
    sep_loss = confidence_separation_loss(assign, args.assignment_margin)
    smooth_loss = local_smoothness_loss(assign, smooth_i, smooth_j, smooth_w)
    confidence = 1.0 - entropy(assign) / np.log(args.num_planes)
    confident_fit = (confidence.detach() * dists.min(dim=-1).values).mean()
    loss = (
        args.fit_weight * soft_fit
        + args.hard_fit_weight * hard_fit
        + args.entropy_weight * ent
        + args.diversity_weight * div
        + args.coverage_weight * cov_loss
        + args.dead_token_weight * dead_loss
        + args.assignment_margin_weight * sep_loss
        + args.smooth_weight * smooth_loss
        + args.confident_fit_weight * confident_fit
    )
    stats = {
        "loss": loss,
        "soft_fit": soft_fit.detach(),
        "hard_fit": hard_fit.detach(),
        "entropy": ent.detach(),
        "diversity": div.detach(),
        "coverage_loss": cov_loss.detach(),
        "dead_token_loss": dead_loss.detach(),
        "smooth_loss": smooth_loss.detach(),
        "confidence": confidence.mean().detach(),
        "coverage": coverage.detach(),
    }
    return loss, stats


def export_case(model, case, output_dir, args, device):
    points = torch.from_numpy(case["points_norm"]).to(device)
    features = torch.from_numpy(case["features"]).to(device)
    with torch.no_grad():
        normals, offsets_norm, dists, assign = model.forward_one(
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

    dists_np = dists.detach().cpu().numpy()
    params = []
    for i in range(args.num_planes):
        mask = hard_np == i
        params.append(
            {
                "id": int(i),
                "normal": [float(x) for x in normals_np[i]],
                "offset": offsets_world[i],
                "offset_normalized": float(offsets_norm_np[i]),
                "assigned_point_count": int(mask.sum()),
                "assigned_ratio": float(mask.mean()),
                "mean_abs_distance_normalized": float(dists_np[mask, i].mean()) if mask.any() else None,
            }
        )
    learned_colors = PLANE_COLORS[hard_np % len(PLANE_COLORS)]
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{case['stem']}_amortized_plane_tokens.json"
    npz_path = output_dir / f"{case['stem']}_amortized_plane_tokens_assignment.npz"
    summary = {
        "input_npz": case["path"],
        "num_points_used": int(len(case["points"])),
        "num_planes": int(args.num_planes),
        "method": "amortized_plane_token_head",
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
    parser = argparse.ArgumentParser("Amortized unsupervised plane-token prediction")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*_full_pointcloud_editable_planes_data.npz")
    parser.add_argument("--sample_glob", default=None)
    parser.add_argument("--num_planes", type=int, default=4)
    parser.add_argument("--max_points_per_sample", type=int, default=30000)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--sample_batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--context_dim", type=int, default=256)
    parser.add_argument("--distance_logit_weight", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--temperature_decay", type=float, default=0.999)
    parser.add_argument("--min_temperature", type=float, default=0.012)
    parser.add_argument("--fit_weight", type=float, default=1.0)
    parser.add_argument("--trimmed_fit_ratio", type=float, default=0.8)
    parser.add_argument("--hard_fit_weight", type=float, default=0.2)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--diversity_weight", type=float, default=0.04)
    parser.add_argument("--coverage_weight", type=float, default=0.05)
    parser.add_argument("--dead_token_weight", type=float, default=0.8)
    parser.add_argument("--dead_token_min_coverage", type=float, default=0.02)
    parser.add_argument("--assignment_margin_weight", type=float, default=0.03)
    parser.add_argument("--confident_fit_weight", type=float, default=0.05)
    parser.add_argument("--smooth_weight", type=float, default=0.0)
    parser.add_argument("--smooth_pairs_per_sample", type=int, default=0)
    parser.add_argument("--smooth_candidates", type=int, default=24)
    parser.add_argument("--smooth_xyz_sigma", type=float, default=0.06)
    parser.add_argument("--smooth_rgb_sigma", type=float, default=0.25)
    parser.add_argument("--min_coverage", type=float, default=0.02)
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
    cases = [
        sample_case(
            p,
            args.max_points_per_sample,
            args.seed + i * 97,
            args.smooth_pairs_per_sample,
            args.smooth_candidates,
            args.smooth_xyz_sigma,
            args.smooth_rgb_sigma,
        )
        for i, p in enumerate(paths)
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = [
        (
            torch.from_numpy(case["points_norm"]).to(device),
            torch.from_numpy(case["features"]).to(device),
            torch.from_numpy(case["smooth_i"]).to(device),
            torch.from_numpy(case["smooth_j"]).to(device),
            torch.from_numpy(case["smooth_w"]).to(device),
        )
        for case in cases
    ]
    model = AmortizedPlaneTokenHead(
        args.num_planes,
        point_feature_dim=tensors[0][1].shape[1],
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
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
            points, features, smooth_i, smooth_j, smooth_w = tensors[int(sid)]
            loss, stats = compute_one_loss(model, points, features, smooth_i, smooth_j, smooth_w, args, temperature)
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
            "dead_token_loss": float(torch.stack([s["dead_token_loss"] for s in stat_rows]).mean().cpu()),
            "smooth_loss": float(torch.stack([s["smooth_loss"] for s in stat_rows]).mean().cpu()),
            "temperature": float(temperature),
        }
        history.append(row)
        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:04d} loss={row['loss']:.5f} fit={row['soft_fit']:.5f} "
                f"hard={row['hard_fit']:.5f} ent={row['entropy']:.4f} "
                f"dead={row['dead_token_loss']:.4f} smooth={row['smooth_loss']:.4f} "
                f"conf={row['confidence']:.3f}"
            )

    output_dir = Path(args.output_dir)
    exported = []
    for case in cases:
        json_path, npz_path, summary = export_case(model, case, output_dir, args, device)
        exported.append({"json": str(json_path), "npz": str(npz_path), "summary": summary})

    overview = {
        "input_dir": args.input_dir,
        "num_samples": len(cases),
        "num_planes": args.num_planes,
        "max_points_per_sample": args.max_points_per_sample,
        "method": "amortized_plane_token_head",
        "history": history,
        "exported": exported,
    }
    overview_path = output_dir / "amortized_plane_tokens_summary.json"
    overview_path.write_text(json.dumps(overview, indent=2), encoding="utf-8")
    print(overview_path)
    print(f"samples={len(cases)}")


if __name__ == "__main__":
    main()
