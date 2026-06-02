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


def load_npz(path, max_points, seed):
    raw = np.load(path)
    points = raw["points"].astype(np.float32)
    colors = raw["colors"].astype(np.uint8)
    rng = np.random.default_rng(seed)
    if max_points > 0 and len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    center = points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points - center, axis=1).max()
    scale = max(float(scale), 1e-6)
    points_norm = (points - center) / scale
    return points, colors, points_norm.astype(np.float32), center[0].astype(np.float32), scale


class PlaneTokenDecomposition(nn.Module):
    def __init__(self, num_planes):
        super().__init__()
        self.normal_raw = nn.Parameter(torch.randn(num_planes, 3) * 0.2)
        self.offset = nn.Parameter(torch.zeros(num_planes))
        self.logit = nn.Parameter(torch.zeros(num_planes))

    def forward(self, points, temperature):
        normals = F.normalize(self.normal_raw, dim=-1)
        offsets = self.offset
        dists = torch.abs(points @ normals.t() + offsets.view(1, -1))
        logits = -dists / temperature + self.logit.view(1, -1)
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
    normal_penalty = F.relu(sim - (1.0 - margin_normal)).mean()
    offset_penalty = F.relu(margin_offset - off_dist).mean()
    return normal_penalty + offset_penalty


def coverage_loss(assign, min_coverage):
    coverage = assign.mean(dim=0)
    return F.relu(min_coverage - coverage).mean(), coverage


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points_world, colors, points_norm, center, scale = load_npz(args.input_npz, args.max_points, args.seed)
    points = torch.from_numpy(points_norm).to(device)
    model = PlaneTokenDecomposition(args.num_planes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best = None
    for step in range(1, args.steps + 1):
        temperature = max(args.min_temperature, args.temperature * (args.temperature_decay ** step))
        normals, offsets, dists, assign = model(points, temperature)
        soft_dist = (assign * dists).sum(dim=-1).mean()
        min_dist = dists.min(dim=-1).values.mean()
        ent = entropy(assign).mean()
        div = diversity_loss(normals, offsets, args.diversity_normal_margin, args.diversity_offset_margin)
        cov_loss, coverage = coverage_loss(assign, args.min_coverage)
        confidence = 1.0 - entropy(assign) / np.log(args.num_planes)
        confident_fit = (confidence.detach() * dists.min(dim=-1).values).mean()
        loss = (
            args.fit_weight * soft_dist
            + args.hard_fit_weight * min_dist
            + args.entropy_weight * ent
            + args.diversity_weight * div
            + args.coverage_weight * cov_loss
            + args.confident_fit_weight * confident_fit
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "soft_fit": float(soft_dist.detach().cpu()),
            "hard_fit": float(min_dist.detach().cpu()),
            "entropy": float(ent.detach().cpu()),
            "diversity": float(div.detach().cpu()),
            "coverage_loss": float(cov_loss.detach().cpu()),
            "confidence": float(confidence.mean().detach().cpu()),
            "temperature": float(temperature),
            "coverage": [float(x) for x in coverage.detach().cpu()],
        }
        history.append(row)
        if best is None or row["loss"] < best["loss"]:
            best = row
        if step % args.log_every == 0 or step == 1:
            active = sum(c > args.min_coverage for c in row["coverage"])
            print(
                f"step={step:04d} loss={row['loss']:.5f} fit={row['soft_fit']:.5f} "
                f"hard={row['hard_fit']:.5f} ent={row['entropy']:.4f} "
                f"conf={row['confidence']:.3f} active={active}/{args.num_planes}"
            )

    with torch.no_grad():
        normals, offsets_norm, dists, assign = model(points, args.min_temperature)
        hard_assign = assign.argmax(dim=-1)
        coverage = torch.stack([(hard_assign == i).float().mean() for i in range(args.num_planes)])
        fit_per_plane = []
        for i in range(args.num_planes):
            mask = hard_assign == i
            if mask.any():
                fit_per_plane.append(float(dists[mask, i].mean().cpu()))
            else:
                fit_per_plane.append(None)

    normals_np = normals.detach().cpu().numpy()
    offsets_norm_np = offsets_norm.detach().cpu().numpy()
    offsets_world = []
    for n, d_norm in zip(normals_np, offsets_norm_np):
        offsets_world.append(float(d_norm * scale - float(np.dot(n, center))))

    hard_np = hard_assign.detach().cpu().numpy().astype(np.int32)
    learned_colors = PLANE_COLORS[hard_np % len(PLANE_COLORS)]
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input_npz).name.replace("_full_pointcloud_editable_planes_data.npz", "")
    json_path = output_dir / f"{stem}_learned_plane_tokens.json"
    npz_path = output_dir / f"{stem}_learned_plane_tokens_assignment.npz"
    summary = {
        "input_npz": args.input_npz,
        "num_points_used": int(len(points_world)),
        "num_planes": int(args.num_planes),
        "center": [float(x) for x in center],
        "scale": float(scale),
        "best": best,
        "final": history[-1],
        "planes": params,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        points=points_world.astype(np.float32),
        colors=learned_colors.astype(np.uint8),
        original_colors=colors.astype(np.uint8),
        assignment=hard_np,
        plane_normals=normals_np.astype(np.float32),
        plane_offsets=np.asarray(offsets_world, dtype=np.float32),
        plane_offsets_normalized=offsets_norm_np.astype(np.float32),
        history=np.asarray([h["loss"] for h in history], dtype=np.float32),
    )
    print(json_path)
    print(npz_path)
    return json_path, npz_path


def main():
    parser = argparse.ArgumentParser("Unsupervised plane-token decomposition from point cloud")
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_planes", type=int, default=6)
    parser.add_argument("--max_points", type=int, default=60000)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--temperature_decay", type=float, default=0.999)
    parser.add_argument("--min_temperature", type=float, default=0.012)
    parser.add_argument("--fit_weight", type=float, default=1.0)
    parser.add_argument("--hard_fit_weight", type=float, default=0.2)
    parser.add_argument("--entropy_weight", type=float, default=0.015)
    parser.add_argument("--diversity_weight", type=float, default=0.04)
    parser.add_argument("--coverage_weight", type=float, default=0.2)
    parser.add_argument("--confident_fit_weight", type=float, default=0.05)
    parser.add_argument("--min_coverage", type=float, default=0.025)
    parser.add_argument("--diversity_normal_margin", type=float, default=0.18)
    parser.add_argument("--diversity_offset_margin", type=float, default=0.04)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
