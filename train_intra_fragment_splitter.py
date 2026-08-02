"""Train a lightweight point-affinity and boundary model inside Stage2 fragments."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate_bias_aware_fragment_association import exact_gt_join


def stratified_sample_indices(
    labels: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) <= int(max_points):
        return np.arange(len(labels), dtype=np.int64)
    unique = np.unique(labels[labels >= 0])
    if len(unique) <= 1:
        return np.sort(rng.choice(len(labels), int(max_points), replace=False))
    selected = []
    quota = max(int(max_points) // len(unique), 1)
    for label in unique:
        candidates = np.flatnonzero(labels == label)
        take = min(len(candidates), quota)
        selected.extend(rng.choice(candidates, take, replace=False).tolist())
    selected = np.unique(np.asarray(selected, dtype=np.int64))
    if len(selected) < int(max_points):
        remaining = np.setdiff1d(
            np.arange(len(labels), dtype=np.int64),
            selected,
            assume_unique=True,
        )
        fill = rng.choice(
            remaining,
            min(int(max_points) - len(selected), len(remaining)),
            replace=False,
        )
        selected = np.concatenate((selected, fill))
    elif len(selected) > int(max_points):
        selected = rng.choice(selected, int(max_points), replace=False)
    return np.sort(selected.astype(np.int64))


def fit_fragment_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    covariance = (points - center).T @ (points - center)
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0:
        normal = -normal
    return normal.astype(np.float32), -float(normal @ center)


def point_and_edge_targets(
    labels: torch.Tensor,
    neighbors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_labels = (labels[:, None] == labels[neighbors]).float()
    boundary = (edge_labels < 0.5).any(dim=1).float()
    return edge_labels, boundary


def fragment_split_target(
    labels: torch.Tensor,
    purity_threshold: float = 0.8,
) -> torch.Tensor:
    _, counts = torch.unique(labels, return_counts=True)
    purity = counts.max().float() / max(int(len(labels)), 1)
    return (purity < float(purity_threshold)).float()


class MultiScaleGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, neighbor_limit: int):
        super().__init__()
        self.neighbor_limit = int(neighbor_limit)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, features: torch.Tensor, neighbors: torch.Tensor
    ) -> torch.Tensor:
        selected = neighbors[:, : min(self.neighbor_limit, neighbors.shape[1])]
        neighbor_features = features[selected]
        center = features[:, None, :].expand_as(neighbor_features)
        edge = self.edge_mlp(
            torch.cat((center, neighbor_features - center), dim=-1)
        )
        aggregate = torch.cat((edge.mean(dim=1), edge.max(dim=1).values), dim=1)
        update = self.update(torch.cat((features, aggregate), dim=1))
        return self.output_norm(features + update)


class LocalAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, attention_heads: int):
        super().__init__()
        if hidden_dim % int(attention_heads):
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.heads = int(attention_heads)
        self.head_dim = hidden_dim // self.heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, features: torch.Tensor, neighbors: torch.Tensor
    ) -> torch.Tensor:
        point_count = len(features)
        query = self.query(features).reshape(
            point_count, self.heads, self.head_dim
        )
        key = self.key(features[neighbors]).reshape(
            point_count, neighbors.shape[1], self.heads, self.head_dim
        )
        value = self.value(features[neighbors]).reshape_as(key)
        score = (
            query[:, None, :, :] * key
        ).sum(dim=-1) / math.sqrt(float(self.head_dim))
        attention = torch.softmax(score.float(), dim=1).to(value.dtype)
        context = (attention[..., None] * value).sum(dim=1).reshape(
            point_count, -1
        )
        features = self.attention_norm(features + self.projection(context))
        return self.output_norm(features + self.feed_forward(features))


class IntraFragmentSplitter(nn.Module):
    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 64,
        embedding_dim: int = 16,
        architecture: str = "light",
        graph_layers: int = 3,
        attention_layers: int = 4,
        attention_heads: int = 4,
    ):
        super().__init__()
        if architecture not in {"light", "multiscale"}:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.architecture = architecture
        activation = nn.ReLU(inplace=True) if architecture == "light" else nn.GELU()
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True) if architecture == "light" else nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        if architecture == "light":
            self.context_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dim),
            )
            self.graph_blocks = nn.ModuleList()
            self.attention_blocks = nn.ModuleList()
        else:
            limits = (8, 16, 32)
            self.graph_blocks = nn.ModuleList(
                MultiScaleGraphBlock(hidden_dim, limits[index % len(limits)])
                for index in range(int(graph_layers))
            )
            self.attention_blocks = nn.ModuleList(
                LocalAttentionBlock(hidden_dim, attention_heads)
                for _ in range(int(attention_layers))
            )
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim)
        self.boundary_head = nn.Linear(hidden_dim, 1)
        self.edge_head = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        if architecture == "multiscale":
            self.split_head = nn.Sequential(
                nn.Linear(embedding_dim * 3 + 6, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def encode(
        self,
        features: torch.Tensor,
        neighbors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.point_encoder(features)
        if self.architecture == "light":
            global_feature = local.max(dim=0, keepdim=True).values.expand_as(local)
            if neighbors is None:
                neighbor_mean = local
                neighbor_difference = torch.zeros_like(local)
            else:
                neighbor_features = local[neighbors]
                neighbor_mean = neighbor_features.mean(dim=1)
                neighbor_difference = torch.abs(
                    neighbor_features - local[:, None, :]
                ).max(dim=1).values
            fused = self.context_fusion(
                torch.cat(
                    (
                        local,
                        neighbor_mean,
                        neighbor_difference,
                        global_feature,
                    ),
                    dim=1,
                )
            )
        else:
            if neighbors is None:
                raise ValueError("multiscale architecture requires neighbors")
            fused = local
            for block in self.graph_blocks:
                fused = block(fused, neighbors)
            for block in self.attention_blocks:
                fused = block(fused, neighbors)
        embedding = F.normalize(self.embedding_head(fused), dim=1)
        boundary_logits = self.boundary_head(fused).squeeze(1)
        return embedding, boundary_logits

    def fragment_split_logit(
        self,
        embedding: torch.Tensor,
        boundary_logits: torch.Tensor,
        pair_logits: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not hasattr(self, "split_head"):
            return None
        if pair_logits is None or pair_logits.numel() == 0:
            pair_statistics = embedding.new_zeros((4,))
        else:
            probability = torch.sigmoid(pair_logits.float()).reshape(-1)
            pair_statistics = torch.stack(
                (
                    probability.mean(),
                    probability.std(unbiased=False),
                    probability.min(),
                    probability.max(),
                )
            ).to(embedding.dtype)
        pooled = torch.cat(
            (
                embedding.mean(dim=0),
                embedding.max(dim=0).values,
                embedding.std(dim=0, unbiased=False),
                boundary_logits.mean().reshape(1),
                boundary_logits.max().reshape(1),
                pair_statistics,
            )
        )
        return self.split_head(pooled).squeeze(0)

    def edge_logits(
        self,
        embedding: torch.Tensor,
        xyz: torch.Tensor,
        pixels: torch.Tensor,
        views: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        point_count, neighbor_count = neighbors.shape
        source = torch.arange(point_count, device=neighbors.device)[:, None].expand(
            point_count, neighbor_count
        )
        return self.pair_logits(
            embedding,
            xyz,
            pixels,
            views,
            source,
            neighbors,
        )

    def pair_logits(
        self,
        embedding: torch.Tensor,
        xyz: torch.Tensor,
        pixels: torch.Tensor,
        views: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        source_embedding = embedding[source]
        target_embedding = embedding[target]
        relative_xyz = xyz[target] - xyz[source]
        xyz_distance = relative_xyz.norm(dim=-1, keepdim=True)
        pixel_distance = (
            pixels[target] - pixels[source]
        ).norm(dim=-1, keepdim=True)
        same_view = (views[target] == views[source]).float().unsqueeze(-1)
        edge_features = torch.cat(
            (
                0.5 * (source_embedding + target_embedding),
                torch.abs(source_embedding - target_embedding),
                relative_xyz,
                xyz_distance,
                pixel_distance,
                same_view,
            ),
            dim=-1,
        )
        return self.edge_head(edge_features).squeeze(-1)


def local_neighbors(
    xyz: torch.Tensor,
    neighbor_count: int,
    pixels: torch.Tensor | None = None,
    views: torch.Tensor | None = None,
) -> torch.Tensor:
    count = min(max(int(neighbor_count), 1), max(len(xyz) - 1, 1))
    if pixels is None:
        distances = torch.cdist(xyz.float(), xyz.float())
    else:
        distances = torch.cdist(pixels.float(), pixels.float())
        if views is not None:
            distances = distances.masked_fill(
                views[:, None] != views[None, :], float("inf")
            )
    distances.fill_diagonal_(float("inf"))
    return torch.topk(distances, k=count, largest=False, dim=1).indices


def nearest_different_pairs(
    xyz: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(xyz.float(), xyz.float())
    different = labels[:, None] != labels[None, :]
    distances = distances.masked_fill(~different, float("inf"))
    nearest_distance, target = distances.min(dim=1)
    valid = torch.isfinite(nearest_distance)
    source = torch.arange(len(xyz), device=xyz.device)[valid]
    return source, target[valid]


def balanced_global_pairs(
    labels: torch.Tensor,
    max_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select deterministic non-local positive and negative supervision pairs."""
    point_count = int(len(labels))
    if point_count < 2 or int(max_pairs) < 1:
        empty = torch.empty((0,), dtype=torch.long, device=labels.device)
        return empty, empty, empty.float()
    pairs = torch.triu_indices(
        point_count,
        point_count,
        offset=1,
        device=labels.device,
    )
    targets = labels[pairs[0]] == labels[pairs[1]]
    per_class = max(int(max_pairs) // 2, 1)
    selected = []
    for target_value in (True, False):
        candidates = torch.nonzero(
            targets == target_value, as_tuple=False
        ).flatten()
        if len(candidates) > per_class:
            positions = torch.linspace(
                0,
                len(candidates) - 1,
                steps=per_class,
                device=labels.device,
            ).round().long()
            candidates = candidates[positions]
        selected.append(candidates)
    selected_indices = torch.cat(selected)
    return (
        pairs[0, selected_indices],
        pairs[1, selected_indices],
        targets[selected_indices].float(),
    )


def balanced_binary_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    positive = labels.mean()
    positive_weight = 0.5 / positive.clamp_min(1e-3)
    negative_weight = 0.5 / (1.0 - positive).clamp_min(1e-3)
    weights = torch.where(labels > 0.5, positive_weight, negative_weight).clamp_max(
        20.0
    )
    return F.binary_cross_entropy_with_logits(logits, labels, weight=weights)


class SceneStore:
    def __init__(self, manifest: dict):
        self.entries = {row["scene"]: row for row in manifest["scenes"]}
        self.cache: dict[str, dict[str, np.ndarray]] = {}

    def load(self, scene: str) -> dict[str, np.ndarray]:
        if scene in self.cache:
            return self.cache[scene]
        entry = self.entries[scene]
        with np.load(entry["support_npz"], allow_pickle=True) as support:
            points = support["points"].astype(np.float32)
            colors_key = (
                "original_colors"
                if "original_colors" in support.files
                else "colors"
            )
            colors = support[colors_key].astype(np.float32) / 255.0
            fragments = support["point_plane_ids"].astype(np.int64)
            views = support["alignment_view_indices"].astype(np.int64)
            pixels = support["pointmap_pixel_xy"].astype(np.float32)
        with np.load(entry["gt_npz"], allow_pickle=False) as gt:
            gt_labels = exact_gt_join(
                views,
                pixels.astype(np.int64),
                gt["view_indices"].astype(np.int64),
                gt["pixel_xy"].astype(np.int64),
                gt["point_plane_ids"].astype(np.int64),
            )
        self.cache[scene] = {
            "points": points,
            "colors": colors,
            "fragments": fragments,
            "views": views,
            "pixels": pixels,
            "gt_labels": gt_labels,
        }
        return self.cache[scene]


def prepare_fragment(
    arrays: dict[str, np.ndarray],
    fragment_id: int,
    max_points: int,
    pixel_size: float,
    rng: np.random.Generator,
    focus_single_view: bool,
    stratified_sampling: bool = True,
) -> dict[str, np.ndarray]:
    indices = np.flatnonzero(
        (arrays["fragments"] == int(fragment_id))
        & (arrays["gt_labels"] >= 0)
    )
    if len(indices) < 3:
        raise ValueError(f"Fragment {fragment_id} has fewer than three labeled points")
    if focus_single_view:
        candidate_views = []
        for view in np.unique(arrays["views"][indices]):
            view_indices = indices[arrays["views"][indices] == view]
            label_count = len(np.unique(arrays["gt_labels"][view_indices]))
            candidate_views.append(
                (int(label_count >= 2), len(view_indices), int(view))
            )
        mixed = [row for row in candidate_views if row[0]]
        pool = mixed if mixed else candidate_views
        maximum_count = max(row[1] for row in pool)
        strongest = [row for row in pool if row[1] == maximum_count]
        selected_view = strongest[int(rng.integers(len(strongest)))][2]
        indices = indices[arrays["views"][indices] == selected_view]
        if len(indices) < 3:
            raise ValueError(
                f"Fragment {fragment_id} view {selected_view} has fewer than three points"
            )
    local_labels = arrays["gt_labels"][indices]
    if stratified_sampling:
        sampled = stratified_sample_indices(local_labels, max_points, rng)
    elif len(indices) > int(max_points):
        sampled = np.sort(
            rng.choice(len(indices), int(max_points), replace=False)
        )
    else:
        sampled = np.arange(len(indices), dtype=np.int64)
    indices = indices[sampled]
    points = arrays["points"][indices].astype(np.float32)
    colors = arrays["colors"][indices].astype(np.float32)
    pixels = arrays["pixels"][indices].astype(np.float32)
    views = arrays["views"][indices].astype(np.int64)
    labels = arrays["gt_labels"][indices].astype(np.int64)
    center = points.mean(axis=0)
    scale = max(
        float(np.median(np.linalg.norm(points - center, axis=1))),
        1e-6,
    )
    xyz = (points - center) / scale
    normal, offset = fit_fragment_plane(points)
    signed_residual = (points @ normal + offset) / scale
    pixel_features = pixels / max(float(pixel_size), 1.0) * 2.0 - 1.0
    view_scale = max(float(views.max()), 1.0)
    view_feature = views.astype(np.float32)[:, None] / view_scale
    normal_features = np.broadcast_to(normal, (len(points), 3))
    features = np.concatenate(
        (
            xyz,
            colors,
            pixel_features,
            view_feature,
            signed_residual[:, None],
            normal_features,
        ),
        axis=1,
    ).astype(np.float32)
    return {
        "features": features,
        "xyz": xyz.astype(np.float32),
        "pixels": pixel_features.astype(np.float32),
        "views": views,
        "labels": labels,
    }


def tensor_case(case: dict[str, np.ndarray], device: torch.device) -> dict:
    return {
        "features": torch.from_numpy(case["features"]).to(device),
        "xyz": torch.from_numpy(case["xyz"]).to(device),
        "pixels": torch.from_numpy(case["pixels"]).to(device),
        "views": torch.from_numpy(case["views"]).to(device),
        "labels": torch.from_numpy(case["labels"]).to(device),
    }


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    prediction = logits >= 0.0
    truth = labels >= 0.5
    tp = int((prediction & truth).sum().item())
    fp = int((prediction & ~truth).sum().item())
    fn = int((~prediction & truth).sum().item())
    tn = int((~prediction & ~truth).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "different_recall": tn / max(tn + fp, 1),
        "balanced_accuracy": 0.5
        * (recall + tn / max(tn + fp, 1)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def merge_metric_counts(rows: list[dict[str, float]]) -> dict[str, float]:
    tp = int(sum(row["tp"] for row in rows))
    fp = int(sum(row["fp"] for row in rows))
    fn = int(sum(row["fn"] for row in rows))
    tn = int(sum(row["tn"] for row in rows))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "different_recall": tn / max(tn + fp, 1),
        "balanced_accuracy": 0.5
        * (recall + tn / max(tn + fp, 1)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@torch.no_grad()
def evaluate(
    model: IntraFragmentSplitter,
    store: SceneStore,
    records: list[dict],
    *,
    device: torch.device,
    max_points: int,
    pixel_size: float,
    neighbor_count: int,
    max_fragments: int,
    seed: int,
) -> dict:
    model.eval()
    selected = records[: int(max_fragments)] if max_fragments > 0 else records
    edge_rows = []
    global_pair_rows = []
    boundary_rows = []
    split_rows = []
    for index, record in enumerate(selected):
        rng = np.random.default_rng(seed + index)
        case = tensor_case(
            prepare_fragment(
                store.load(record["scene"]),
                record["fragment_id"],
                max_points,
                pixel_size,
                rng,
                True,
                model.architecture == "light",
            ),
            device,
        )
        neighbors = local_neighbors(
            case["xyz"], neighbor_count, case["pixels"], case["views"]
        )
        edge_target, boundary_target = point_and_edge_targets(
            case["labels"], neighbors
        )
        embedding, boundary_logits = model.encode(
            case["features"], neighbors
        )
        edge_logits = model.edge_logits(
            embedding,
            case["xyz"],
            case["pixels"],
            case["views"],
            neighbors,
        )
        edge_rows.append(binary_metrics(edge_logits, edge_target))
        global_source, global_target, global_labels = balanced_global_pairs(
            case["labels"], 4096
        )
        global_logits = model.pair_logits(
            embedding,
            case["xyz"],
            case["pixels"],
            case["views"],
            global_source,
            global_target,
        )
        global_pair_rows.append(binary_metrics(global_logits, global_labels))
        boundary_rows.append(binary_metrics(boundary_logits, boundary_target))
        split_logit = model.fragment_split_logit(
            embedding, boundary_logits, edge_logits
        )
        if split_logit is not None:
            split_target = torch.as_tensor(
                float(record["hard_impure"]),
                device=split_logit.device,
            )
            split_rows.append(
                binary_metrics(
                    split_logit.reshape(1),
                    split_target.reshape(1),
                )
            )
    result = {
        "fragment_count": len(selected),
        "edge": merge_metric_counts(edge_rows),
        "global_pair": merge_metric_counts(global_pair_rows),
        "boundary": merge_metric_counts(boundary_rows),
    }
    if split_rows:
        result["split_gate"] = merge_metric_counts(split_rows)
    return result


def flatten_records(manifest: dict, split: str) -> list[dict]:
    records = []
    for scene in manifest["scenes"]:
        if scene["split"] != split:
            continue
        for fragment in scene["fragments"]:
            if fragment["eligible"]:
                records.append({"scene": scene["scene"], **fragment})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train point affinity and boundary prediction inside Stage2 fragments."
    )
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--max_points", type=int, default=1024)
    parser.add_argument("--neighbor_count", type=int, default=8)
    parser.add_argument("--pixel_size", type=float, default=512.0)
    parser.add_argument(
        "--multi_view_crops",
        action="store_true",
        help="Disable the default single-view pointmap crop sampling.",
    )
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument(
        "--architecture",
        choices=("light", "multiscale"),
        default="light",
    )
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--attention_layers", type=int, default=4)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--hard_fragment_probability", type=float, default=0.7)
    parser.add_argument("--fragments_per_step", type=int, default=4)
    parser.add_argument("--boundary_loss_weight", type=float, default=1.0)
    parser.add_argument("--global_pair_count", type=int, default=2048)
    parser.add_argument("--global_pair_loss_weight", type=float, default=1.0)
    parser.add_argument("--split_loss_weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_val_fragments", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_records = flatten_records(manifest, "train")
    val_records = flatten_records(manifest, "val")
    hard_train = [record for record in train_records if record["hard_impure"]]
    pure_train = [
        record for record in train_records if not record["hard_impure"]
    ]
    hard_val = [record for record in val_records if record["hard_impure"]]
    if (
        not train_records
        or not val_records
        or not hard_train
        or not pure_train
        or not hard_val
    ):
        raise ValueError("Manifest must contain train/val and hard train/val fragments")
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = IntraFragmentSplitter(
        input_dim=13,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        architecture=args.architecture,
        graph_layers=args.graph_layers,
        attention_layers=args.attention_layers,
        attention_heads=args.attention_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    store = SceneStore(manifest)
    if use_amp:
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    best_score = -1.0
    best_step = 0
    started = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        step_edge_losses = []
        step_global_pair_losses = []
        step_boundary_losses = []
        step_split_losses = []
        fragment_batch = max(int(args.fragments_per_step), 1)
        for batch_index in range(fragment_batch):
            force_hard = fragment_batch > 1 and batch_index % 2 == 0
            force_pure = fragment_batch > 1 and batch_index % 2 == 1
            if force_hard:
                pool = hard_train
            elif force_pure:
                pool = pure_train
            elif rng.random() < float(args.hard_fragment_probability):
                pool = hard_train
            else:
                pool = pure_train
            record = pool[int(rng.integers(len(pool)))]
            case = tensor_case(
                prepare_fragment(
                    store.load(record["scene"]),
                    record["fragment_id"],
                    args.max_points,
                    args.pixel_size,
                    rng,
                    not args.multi_view_crops,
                    args.architecture == "light",
                ),
                device,
            )
            neighbors = local_neighbors(
                case["xyz"],
                args.neighbor_count,
                case["pixels"],
                case["views"],
            )
            edge_target, boundary_target = point_and_edge_targets(
                case["labels"], neighbors
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                embedding, boundary_logits = model.encode(
                    case["features"], neighbors
                )
                edge_logits = model.edge_logits(
                    embedding,
                    case["xyz"],
                    case["pixels"],
                    case["views"],
                    neighbors,
                )
                local_edge_logits = edge_logits
                hard_source, hard_target = nearest_different_pairs(
                    case["xyz"], case["labels"]
                )
                if len(hard_source):
                    hard_logits = model.pair_logits(
                        embedding,
                        case["xyz"],
                        case["pixels"],
                        case["views"],
                        hard_source,
                        hard_target,
                    )
                    edge_logits = torch.cat((edge_logits.reshape(-1), hard_logits))
                    edge_target = torch.cat(
                        (
                            edge_target.reshape(-1),
                            torch.zeros_like(hard_logits),
                        )
                    )
                edge_loss = balanced_binary_loss(edge_logits, edge_target)
                global_source, global_target, global_labels = (
                    balanced_global_pairs(
                        case["labels"], args.global_pair_count
                    )
                )
                global_logits = model.pair_logits(
                    embedding,
                    case["xyz"],
                    case["pixels"],
                    case["views"],
                    global_source,
                    global_target,
                )
                global_pair_loss = balanced_binary_loss(
                    global_logits, global_labels
                )
                boundary_loss = balanced_binary_loss(
                    boundary_logits, boundary_target
                )
                split_logit = model.fragment_split_logit(
                    embedding, boundary_logits, local_edge_logits
                )
                if split_logit is None:
                    split_loss = torch.zeros((), device=device)
                else:
                    split_target = torch.as_tensor(
                        float(record["hard_impure"]),
                        device=split_logit.device,
                    )
                    split_loss = F.binary_cross_entropy_with_logits(
                        split_logit,
                        split_target,
                    )
                loss = (
                    edge_loss
                    + float(args.global_pair_loss_weight) * global_pair_loss
                    + float(args.boundary_loss_weight) * boundary_loss
                    + float(args.split_loss_weight) * split_loss
                )
            scaler.scale(loss / fragment_batch).backward()
            step_losses.append(loss.detach())
            step_edge_losses.append(edge_loss.detach())
            step_global_pair_losses.append(global_pair_loss.detach())
            step_boundary_losses.append(boundary_loss.detach())
            step_split_losses.append(split_loss.detach())
        scaler.step(optimizer)
        scaler.update()
        loss = torch.stack(step_losses).mean()
        edge_loss = torch.stack(step_edge_losses).mean()
        global_pair_loss = torch.stack(step_global_pair_losses).mean()
        boundary_loss = torch.stack(step_boundary_losses).mean()
        split_loss = torch.stack(step_split_losses).mean()
        if step == 1 or step % int(args.eval_every) == 0 or step == args.steps:
            val_all = evaluate(
                model,
                store,
                val_records,
                device=device,
                max_points=args.max_points,
                pixel_size=args.pixel_size,
                neighbor_count=args.neighbor_count,
                max_fragments=args.max_val_fragments,
                seed=args.seed + 100_000,
            )
            val_hard = evaluate(
                model,
                store,
                hard_val,
                device=device,
                max_points=args.max_points,
                pixel_size=args.pixel_size,
                neighbor_count=args.neighbor_count,
                max_fragments=args.max_val_fragments,
                seed=args.seed + 200_000,
            )
            if "split_gate" in val_all:
                score = (
                    0.25 * val_hard["edge"]["balanced_accuracy"]
                    + 0.35 * val_hard["global_pair"]["balanced_accuracy"]
                    + 0.15 * val_hard["boundary"]["balanced_accuracy"]
                    + 0.25 * val_all["split_gate"]["balanced_accuracy"]
                )
            else:
                score = (
                    0.35 * val_hard["edge"]["balanced_accuracy"]
                    + 0.45 * val_hard["global_pair"]["balanced_accuracy"]
                    + 0.20 * val_hard["boundary"]["balanced_accuracy"]
                )
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "edge_loss": float(edge_loss.detach().cpu()),
                "global_pair_loss": float(global_pair_loss.detach().cpu()),
                "boundary_loss": float(boundary_loss.detach().cpu()),
                "split_loss": float(split_loss.detach().cpu()),
                "val_all": val_all,
                "val_hard": val_hard,
                "selection_score": score,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if score > best_score:
                best_score = score
                best_step = step
                torch.save(
                    {
                        "schema_version": 1,
                        "model": model.state_dict(),
                        "model_config": {
                            "input_dim": 13,
                            "hidden_dim": args.hidden_dim,
                            "embedding_dim": args.embedding_dim,
                            "architecture": args.architecture,
                            "graph_layers": args.graph_layers,
                            "attention_layers": args.attention_layers,
                            "attention_heads": args.attention_heads,
                        },
                        "feature_contract": manifest["feature_contract"],
                        "coordinate_contract": manifest["coordinate_contract"],
                        "step": step,
                        "val_all": val_all,
                        "val_hard": val_hard,
                        "args": vars(args),
                    },
                    output_dir / "best.pt",
                )
            model.train()
    runtime = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if use_amp else 0
    )
    summary = {
        "schema_version": 1,
        "task": "intra_fragment_point_affinity_and_boundary",
        "device": str(device),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "train_fragment_count": len(train_records),
        "hard_train_fragment_count": len(hard_train),
        "val_fragment_count": len(val_records),
        "hard_val_fragment_count": len(hard_val),
        "best_step": best_step,
        "best_selection_score": best_score,
        "runtime_seconds": runtime,
        "peak_cuda_memory_bytes": peak_memory,
        "config": vars(args),
        "history": history,
        "outputs": {
            "best_checkpoint": str(output_dir / "best.pt"),
            "summary_json": str(output_dir / "summary.json"),
        },
        "limitations": [
            "This is a crop-level affinity and boundary gate, not yet an end-to-end scene reconstruction.",
            "Validation deliberately uses three difficult development scenes and is not an unbiased final test split.",
            "DUSt3R intermediate descriptors are not present in the cached support NPZ and are not used.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in (
        "device",
        "parameter_count",
        "best_step",
        "best_selection_score",
        "runtime_seconds",
        "peak_cuda_memory_bytes",
        "outputs",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
