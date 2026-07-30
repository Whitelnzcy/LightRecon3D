"""Frozen prospective B16/M16 training and prediction implementation.

Final-scene GT is intentionally absent from this module. B16 and M16 share
every function; ``enable_split`` is their only method switch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


REGISTRATION_ID = "E16_PREREG_targeted_GT_sidecar_and_final_BM_20260730_v1"
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "e16_final_bm.json"
OUT = ROOT / "results" / "targeted_GT_sidecar_and_final_BM_e16_20260730_v1"
MODEL_PATH = OUT / "shared_identity_model.npz"
IDENTITY_MANIFEST = OUT / "shared_identity_manifest.json"
BIAS_EXCLUSIVE_FEATURES = {
    "bias_corrected_rmse_normalized",
    "bias_rmse_gain_normalized",
    "required_bias_std_normalized",
    "required_bias_range_normalized",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    if config.get("registration_id") != REGISTRATION_ID:
        raise ValueError("E16 config registration mismatch")
    return config


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("plane fit needs at least three finite points")
    center = points.mean(axis=0)
    covariance = (points - center).T @ (points - center)
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0:
        normal = -normal
    offset = -float(normal @ center)
    residual = points @ normal + offset
    rmse = math.sqrt(float(np.mean(residual**2)))
    return normal.astype(np.float64), offset, rmse


def _normalize_colors(colors: np.ndarray) -> np.ndarray:
    values = np.asarray(colors, dtype=np.float64)
    if values.size and float(np.nanmax(values)) > 1.5:
        values = values / 255.0
    return np.clip(values, 0.0, 1.0)


def _bbox_gap(first_min, first_max, second_min, second_max) -> float:
    first_min = np.asarray(first_min, dtype=np.float64)
    first_max = np.asarray(first_max, dtype=np.float64)
    second_min = np.asarray(second_min, dtype=np.float64)
    second_max = np.asarray(second_max, dtype=np.float64)
    gap = np.maximum(
        np.maximum(first_min - second_max, second_min - first_max), 0.0
    )
    return float(np.linalg.norm(gap))


def _bbox_iou(first: dict, second: dict) -> float:
    first_min = np.asarray(first["min"], dtype=np.float64)
    first_max = np.asarray(first["max"], dtype=np.float64)
    second_min = np.asarray(second["min"], dtype=np.float64)
    second_max = np.asarray(second["max"], dtype=np.float64)
    intersection = np.maximum(
        np.minimum(first_max, second_max) - np.maximum(first_min, second_min) + 1.0,
        0.0,
    )
    intersection_area = float(np.prod(intersection))
    first_area = float(np.prod(first_max - first_min + 1.0))
    second_area = float(np.prod(second_max - second_min + 1.0))
    return intersection_area / max(first_area + second_area - intersection_area, 1e-12)


def _robust_interval_relation(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    first_low, first_high = np.quantile(first, [0.02, 0.98])
    second_low, second_high = np.quantile(second, [0.02, 0.98])
    gap = max(first_low - second_high, second_low - first_high, 0.0)
    overlap = max(min(first_high, second_high) - max(first_low, second_low), 0.0)
    minimum_span = max(
        min(first_high - first_low, second_high - second_low), 1e-12
    )
    return float(gap), float(overlap / minimum_span)


def _tangent_relations(
    first: np.ndarray, second: np.ndarray, normal: np.ndarray
) -> dict[str, float]:
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    delta = second.mean(axis=0) - first.mean(axis=0)
    primary = delta - float(delta @ normal) * normal
    primary_norm = float(np.linalg.norm(primary))
    if primary_norm <= 1e-10:
        centered = np.concatenate((first, second), axis=0)
        centered -= centered.mean(axis=0)
        centered -= np.outer(centered @ normal, normal)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        primary = vectors[0] - float(vectors[0] @ normal) * normal
        primary_norm = float(np.linalg.norm(primary))
    if primary_norm <= 1e-10:
        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(reference @ normal)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        primary = np.cross(normal, reference)
        primary_norm = float(np.linalg.norm(primary))
    primary /= max(primary_norm, 1e-12)
    secondary = np.cross(normal, primary)
    secondary /= max(float(np.linalg.norm(secondary)), 1e-12)
    primary_gap, primary_overlap = _robust_interval_relation(
        first @ primary, second @ primary
    )
    secondary_gap, secondary_overlap = _robust_interval_relation(
        first @ secondary, second @ secondary
    )
    return {
        "primary_gap": primary_gap,
        "primary_overlap": primary_overlap,
        "secondary_gap": secondary_gap,
        "secondary_overlap": secondary_overlap,
        "bbox_gap": math.hypot(primary_gap, secondary_gap),
        "bbox_overlap": primary_overlap * secondary_overlap,
    }


def locate_npz(root: Path, scene: str, filename: str) -> Path:
    direct = root / scene / filename
    if direct.is_file():
        return direct
    matches = list((root / scene).rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"{scene}: expected one {filename}, found {len(matches)}")
    return matches[0]


def initial_observations(
    global_path: Path, support_path: Path, minimum_fragment_points: int
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    with np.load(global_path, allow_pickle=False) as global_cache:
        points = global_cache["points"].astype(np.float32)
        colors = global_cache["colors"].astype(np.uint8)
        global_views = global_cache["view_indices"].astype(np.int16)
        global_pixels = global_cache["pixel_xy"].astype(np.int32)
        scene_id = str(global_cache["scene_id"].item())
    with np.load(support_path, allow_pickle=False) as support:
        support_scene = str(support["scene_id"].item())
        labels = support["support_plane_labels"].astype(np.int32)
        source_views = support["support_source_view_indices"].astype(np.int16)
        partner_views = support["support_partner_view_indices"].astype(np.int16)
        orientations = support["support_pair_orientations"].astype(np.int8)
        mask_to_global = support[
            "view_mask_pixel_to_global_point_index"
        ].astype(np.int64)
    if scene_id != support_scene:
        raise ValueError("global/direct scene mismatch")
    output = {
        "points": [],
        "colors": [],
        "view_indices": [],
        "pixel_xy": [],
        "global_point_indices": [],
        "record_indices": [],
        "local_plane_labels": [],
        "initial_fragment_ids": [],
    }
    provenance: list[dict[str, Any]] = []
    next_fragment = 0
    for record_index in range(len(labels)):
        source_view = int(source_views[record_index])
        global_map = mask_to_global[source_view]
        for local_label in sorted(int(x) for x in np.unique(labels[record_index]) if x >= 0):
            mask = labels[record_index] == local_label
            indices = global_map[mask]
            indices = indices[
                (indices >= 0)
                & (indices < len(points))
            ]
            if len(indices):
                indices = indices[
                    np.isfinite(points[indices]).all(axis=1)
                    & (global_views[indices] == source_view)
                ]
            if len(indices) < int(minimum_fragment_points):
                continue
            count = len(indices)
            output["points"].append(points[indices])
            output["colors"].append(colors[indices])
            output["view_indices"].append(global_views[indices])
            output["pixel_xy"].append(global_pixels[indices])
            output["global_point_indices"].append(indices.astype(np.int64))
            output["record_indices"].append(
                np.full(count, record_index, dtype=np.int16)
            )
            output["local_plane_labels"].append(
                np.full(count, local_label, dtype=np.int16)
            )
            output["initial_fragment_ids"].append(
                np.full(count, next_fragment, dtype=np.int32)
            )
            provenance.append(
                {
                    "initial_fragment_id": next_fragment,
                    "support_record_index": record_index,
                    "source_view_index": source_view,
                    "partner_view_index": int(partner_views[record_index]),
                    "pair_orientation": int(orientations[record_index]),
                    "record_local_plane_label": local_label,
                    "point_count": count,
                }
            )
            next_fragment += 1
    if not output["points"]:
        raise ValueError(f"{scene_id}: no fragments pass minimum size")
    arrays = {
        key: np.concatenate(value, axis=0) for key, value in output.items()
    }
    # A global-cache point can be proposed by several directional pair records.
    # Freeze one support assignment per cache point using the first occurrence
    # in support-record/raster order. This is prediction-side initial-fragment
    # construction, shared byte-for-byte by B16 and M16.
    _, first_occurrence = np.unique(
        arrays["global_point_indices"], return_index=True
    )
    keep = np.sort(first_occurrence)
    arrays = {key: value[keep] for key, value in arrays.items()}
    retained_ids, retained_sizes = np.unique(
        arrays["initial_fragment_ids"], return_counts=True
    )
    sufficiently_large = retained_ids[
        retained_sizes >= int(minimum_fragment_points)
    ]
    keep = np.isin(arrays["initial_fragment_ids"], sufficiently_large)
    arrays = {key: value[keep] for key, value in arrays.items()}
    if not len(arrays["points"]):
        raise ValueError(f"{scene_id}: no unique fragments pass minimum size")
    old_ids_sorted = sorted(int(value) for value in np.unique(arrays["initial_fragment_ids"]))
    _, compact_fragments = np.unique(
        arrays["initial_fragment_ids"], return_inverse=True
    )
    arrays["initial_fragment_ids"] = compact_fragments.astype(np.int32)
    retained_counts = Counter(int(value) for value in arrays["initial_fragment_ids"])
    retained_provenance = []
    for compact_id, old_id in enumerate(old_ids_sorted):
        if compact_id not in retained_counts:
            continue
        row = dict(provenance[old_id])
        row["candidate_point_count"] = row.pop("point_count")
        row["initial_fragment_id"] = compact_id
        row["retained_unique_global_point_count"] = retained_counts[compact_id]
        retained_provenance.append(row)
    provenance = retained_provenance
    arrays["scene_id"] = np.asarray(scene_id)
    return arrays, provenance


class IntraFragmentSplitter(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 64, embedding_dim: int = 16):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
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

    def encode(
        self, features: torch.Tensor, neighbors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.point_encoder(features)
        neighbor_features = local[neighbors]
        neighbor_mean = neighbor_features.mean(dim=1)
        neighbor_difference = torch.abs(
            neighbor_features - local[:, None, :]
        ).max(dim=1).values
        global_feature = local.max(dim=0, keepdim=True).values.expand_as(local)
        fused = self.context_fusion(
            torch.cat(
                (local, neighbor_mean, neighbor_difference, global_feature), dim=1
            )
        )
        return (
            F.normalize(self.embedding_head(fused), dim=1),
            self.boundary_head(fused).squeeze(1),
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
        relative_xyz = xyz[target] - xyz[source]
        features = torch.cat(
            (
                0.5 * (embedding[source] + embedding[target]),
                torch.abs(embedding[source] - embedding[target]),
                relative_xyz,
                relative_xyz.norm(dim=-1, keepdim=True),
                (pixels[target] - pixels[source]).norm(dim=-1, keepdim=True),
                (views[target] == views[source]).float().unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.edge_head(features).squeeze(-1)


def splitter_case(
    points: np.ndarray,
    colors: np.ndarray,
    pixels: np.ndarray,
    views: np.ndarray,
    pixel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    scale = max(float(np.median(np.linalg.norm(points - center, axis=1))), 1e-6)
    xyz = (points - center) / scale
    normal, offset, _ = fit_plane(points)
    residual = (points @ normal + offset) / scale
    pixel_features = pixels / max(float(pixel_size), 1.0) * 2.0 - 1.0
    view_scale = max(float(views.max()), 1.0)
    features = np.concatenate(
        (
            xyz,
            _normalize_colors(colors),
            pixel_features,
            views.astype(np.float32)[:, None] / view_scale,
            residual[:, None],
            np.broadcast_to(normal, (len(points), 3)),
        ),
        axis=1,
    )
    return features.astype(np.float32), xyz.astype(np.float32), pixel_features.astype(np.float32)


def local_neighbors(pixels: torch.Tensor, views: torch.Tensor, count: int) -> torch.Tensor:
    neighbor_count = min(max(int(count), 1), max(len(pixels) - 1, 1))
    distances = torch.cdist(pixels.float(), pixels.float())
    distances = distances.masked_fill(views[:, None] != views[None, :], float("inf"))
    distances.fill_diagonal_(float("inf"))
    return torch.topk(distances, k=neighbor_count, largest=False, dim=1).indices


@torch.no_grad()
def dense_affinity(
    model: IntraFragmentSplitter,
    embedding: torch.Tensor,
    xyz: torch.Tensor,
    pixels: torch.Tensor,
    views: torch.Tensor,
) -> np.ndarray:
    count = len(embedding)
    affinity = np.eye(count, dtype=np.float32)
    if count < 2:
        return affinity
    pairs = torch.triu_indices(count, count, offset=1, device=embedding.device)
    for start in range(0, pairs.shape[1], 65536):
        source = pairs[0, start : start + 65536]
        target = pairs[1, start : start + 65536]
        forward = model.pair_logits(embedding, xyz, pixels, views, source, target)
        backward = model.pair_logits(embedding, xyz, pixels, views, target, source)
        probability = torch.sigmoid(0.5 * (forward + backward))
        source_np = source.cpu().numpy()
        target_np = target.cpu().numpy()
        values = probability.cpu().numpy()
        affinity[source_np, target_np] = values
        affinity[target_np, source_np] = values
    return affinity


def merge_small_clusters(
    labels: np.ndarray, affinity: np.ndarray, minimum_points: int
) -> np.ndarray:
    labels = labels.astype(np.int32, copy=True)
    unique, counts = np.unique(labels, return_counts=True)
    large = unique[counts >= int(minimum_points)]
    if len(large) < 2:
        return np.zeros_like(labels)
    for cluster in unique[counts < int(minimum_points)]:
        members = labels == cluster
        scores = [
            float(affinity[np.ix_(members, labels == candidate)].mean())
            for candidate in large
        ]
        labels[members] = int(large[int(np.argmax(scores))])
    return np.unique(labels, return_inverse=True)[1].astype(np.int32)


def nearest_anchor_labels(
    pixels: np.ndarray, anchor_pixels: np.ndarray, anchor_labels: np.ndarray
) -> np.ndarray:
    output = np.empty(len(pixels), dtype=np.int32)
    for start in range(0, len(pixels), 8192):
        chunk = pixels[start : start + 8192]
        squared = ((chunk[:, None] - anchor_pixels[None]) ** 2).sum(axis=2)
        output[start : start + len(chunk)] = anchor_labels[np.argmin(squared, axis=1)]
    return output


def load_splitter(config: dict[str, Any], device: torch.device) -> IntraFragmentSplitter:
    checkpoint_path = Path(config["m_checkpoint"])
    if sha256_file(checkpoint_path) != config["m_checkpoint_sha256"]:
        raise ValueError("M checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    model = IntraFragmentSplitter(**model_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 27730:
        raise ValueError("M checkpoint parameter count mismatch")
    return model.to(device).eval()


@torch.no_grad()
def apply_split(
    arrays: dict[str, np.ndarray],
    model: IntraFragmentSplitter,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    initial = arrays["initial_fragment_ids"]
    split_ids = np.full(len(initial), -1, dtype=np.int32)
    next_id = 0
    audit = []
    for fragment_id in sorted(int(x) for x in np.unique(initial)):
        indices = np.flatnonzero(initial == fragment_id)
        anchors = (
            np.arange(len(indices), dtype=np.int64)
            if len(indices) <= int(config["m_max_anchor_points"])
            else np.linspace(
                0,
                len(indices) - 1,
                int(config["m_max_anchor_points"]),
            ).round().astype(np.int64)
        )
        anchor_indices = indices[anchors]
        features_np, xyz_np, pixels_np = splitter_case(
            arrays["points"][anchor_indices],
            arrays["colors"][anchor_indices],
            arrays["pixel_xy"][anchor_indices].astype(np.float32),
            arrays["view_indices"][anchor_indices],
            config["m_pixel_size"],
        )
        features = torch.from_numpy(features_np).to(device)
        xyz = torch.from_numpy(xyz_np).to(device)
        pixels = torch.from_numpy(pixels_np).to(device)
        views = torch.from_numpy(
            arrays["view_indices"][anchor_indices].astype(np.int64)
        ).to(device)
        neighbors = local_neighbors(pixels, views, config["m_neighbor_count"])
        embedding, _ = model.encode(features, neighbors)
        affinity = dense_affinity(model, embedding, xyz, pixels, views)
        if len(anchors) < 2:
            anchor_labels = np.zeros(len(anchors), dtype=np.int32)
        else:
            distance = 1.0 - np.clip(0.5 * (affinity + affinity.T), 0.0, 1.0)
            np.fill_diagonal(distance, 0.0)
            tree = linkage(squareform(distance, checks=False), method="average")
            anchor_labels = fcluster(
                tree,
                t=1.0 - float(config["m_affinity_threshold"]),
                criterion="distance",
            )
            anchor_labels = np.unique(anchor_labels, return_inverse=True)[1].astype(
                np.int32
            )
            anchor_labels = merge_small_clusters(
                anchor_labels,
                affinity,
                config["m_minimum_anchor_cluster_points"],
            )
        local_labels = nearest_anchor_labels(
            arrays["pixel_xy"][indices].astype(np.float32),
            arrays["pixel_xy"][anchor_indices].astype(np.float32),
            anchor_labels,
        )
        for local_id in sorted(int(x) for x in np.unique(local_labels)):
            split_ids[indices[local_labels == local_id]] = next_id
            next_id += 1
        audit.append(
            {
                "initial_fragment_id": fragment_id,
                "point_count": len(indices),
                "anchor_count": len(anchors),
                "component_count": int(len(np.unique(local_labels))),
            }
        )
    return split_ids, audit


def fragment_descriptors(
    arrays: dict[str, np.ndarray],
    fragment_ids: np.ndarray,
    maximum_geometry_points: int,
) -> tuple[dict[int, dict[str, Any]], float]:
    points = arrays["points"].astype(np.float64)
    colors = _normalize_colors(arrays["colors"])
    center = np.median(points, axis=0)
    scene_scale = max(
        float(np.median(np.linalg.norm(points - center, axis=1))), 1e-8
    )
    descriptors = {}
    for fragment_id in sorted(int(x) for x in np.unique(fragment_ids)):
        indices = np.flatnonzero(fragment_ids == fragment_id)
        local_points = points[indices]
        local_views = arrays["view_indices"][indices]
        local_pixels = arrays["pixel_xy"][indices]
        if len(indices) > int(maximum_geometry_points):
            sample = np.linspace(
                0, len(indices) - 1, int(maximum_geometry_points), dtype=np.int64
            )
        else:
            sample = np.arange(len(indices), dtype=np.int64)
        geometry = local_points[sample]
        normal, offset, _ = fit_plane(geometry)
        centered = geometry - geometry.mean(axis=0)
        covariance = centered.T @ centered / max(len(centered), 1)
        view_boxes = {}
        for view in np.unique(local_views):
            mask = local_views == view
            pixels = local_pixels[mask]
            view_boxes[int(view)] = {
                "min": pixels.min(axis=0).astype(float),
                "max": pixels.max(axis=0).astype(float),
                "centroid": pixels.mean(axis=0).astype(float),
            }
        descriptors[fragment_id] = {
            "point_count": len(indices),
            "indices": indices,
            "points": geometry,
            "normal": normal,
            "offset": offset,
            "centroid": local_points.mean(axis=0),
            "bbox_min": local_points.min(axis=0),
            "bbox_max": local_points.max(axis=0),
            "color_mean": colors[indices].mean(axis=0),
            "color_std": colors[indices].std(axis=0),
            "eigenvalues": np.maximum(np.linalg.eigvalsh(covariance), 0.0),
            "view_set": set(int(x) for x in np.unique(local_views)),
            "view_boxes": view_boxes,
        }
    return descriptors, scene_scale


def pair_feature_values(
    first: dict[str, Any],
    second: dict[str, Any],
    scene_scale: float,
    pixel_diagonal: float,
) -> dict[str, float]:
    shared_views = sorted(first["view_set"] & second["view_set"])
    union_views = first["view_set"] | second["view_set"]
    delta = second["centroid"] - first["centroid"]
    sign = 1.0 if float(first["normal"] @ second["normal"]) >= 0 else -1.0
    mean_normal = first["normal"] + sign * second["normal"]
    mean_normal /= max(float(np.linalg.norm(mean_normal)), 1e-12)
    normal_separation = abs(float(delta @ mean_normal))
    tangent_distance = math.sqrt(
        max(float(delta @ delta) - normal_separation**2, 0.0)
    )
    residual_a = first["points"] @ second["normal"] + second["offset"]
    residual_b = second["points"] @ first["normal"] + first["offset"]
    cross_rmse = math.sqrt(
        0.5 * (float(np.mean(residual_a**2)) + float(np.mean(residual_b**2)))
    )
    pooled_points = np.concatenate((first["points"], second["points"]), axis=0)
    _, _, pooled_rmse = fit_plane(pooled_points)
    tangent = _tangent_relations(first["points"], second["points"], mean_normal)
    if shared_views:
        ious = [
            _bbox_iou(first["view_boxes"][view], second["view_boxes"][view])
            for view in shared_views
        ]
        gaps = [
            _bbox_gap(
                first["view_boxes"][view]["min"],
                first["view_boxes"][view]["max"],
                second["view_boxes"][view]["min"],
                second["view_boxes"][view]["max"],
            )
            for view in shared_views
        ]
        centroid_distances = [
            float(
                np.linalg.norm(
                    first["view_boxes"][view]["centroid"]
                    - second["view_boxes"][view]["centroid"]
                )
            )
            for view in shared_views
        ]
        iou_max = max(ious)
        iou_mean = float(np.mean(ious))
        pixel_gap = min(gaps) / pixel_diagonal
        pixel_centroid = min(centroid_distances) / pixel_diagonal
    else:
        iou_max = iou_mean = 0.0
        pixel_gap = pixel_centroid = 1.0
    features = {
        "normal_angle_deg": math.degrees(
            math.acos(np.clip(abs(float(first["normal"] @ second["normal"])), 0, 1))
        ),
        "pooled_rmse_normalized": pooled_rmse / scene_scale,
        "centroid_distance_normalized": float(np.linalg.norm(delta)) / scene_scale,
        "normal_centroid_separation_normalized": normal_separation / scene_scale,
        "tangent_centroid_distance_normalized": tangent_distance / scene_scale,
        "tangent_support_primary_gap_normalized": tangent["primary_gap"] / scene_scale,
        "tangent_support_primary_overlap": tangent["primary_overlap"],
        "tangent_support_secondary_gap_normalized": tangent["secondary_gap"]
        / scene_scale,
        "tangent_support_secondary_overlap": tangent["secondary_overlap"],
        "tangent_support_bbox_gap_normalized": tangent["bbox_gap"] / scene_scale,
        "tangent_support_bbox_overlap": tangent["bbox_overlap"],
        "bbox_3d_gap_normalized": _bbox_gap(
            first["bbox_min"],
            first["bbox_max"],
            second["bbox_min"],
            second["bbox_max"],
        )
        / scene_scale,
        "cross_plane_rmse_normalized": cross_rmse / scene_scale,
        "log_point_count_ratio_abs": abs(
            math.log(max(first["point_count"], 1) / max(second["point_count"], 1))
        ),
        "min_point_count_log": math.log1p(
            min(first["point_count"], second["point_count"])
        ),
        "color_mean_l2": float(
            np.linalg.norm(first["color_mean"] - second["color_mean"])
        ),
        "color_std_l2": float(
            np.linalg.norm(first["color_std"] - second["color_std"])
        ),
        "shared_view_count": float(len(shared_views)),
        "union_view_count": float(len(union_views)),
        "view_jaccard": len(shared_views) / max(len(union_views), 1),
        "disjoint_view_sets": float(not shared_views),
        "pixel_bbox_iou_max": iou_max,
        "pixel_bbox_iou_mean": iou_mean,
        "pixel_bbox_gap_min_normalized": pixel_gap,
        "pixel_centroid_distance_min_normalized": pixel_centroid,
    }
    first_eigen = first["eigenvalues"] / scene_scale**2
    second_eigen = second["eigenvalues"] / scene_scale**2
    for index in range(3):
        features[f"eigenvalue_{index}_difference"] = abs(
            float(first_eigen[index] - second_eigen[index])
        )
    return features


class StandardizedLogistic:
    """Deterministic class-balanced L2 logistic regression.

    The objective follows the common ``C * sum(logloss) + .5 * ||w||²``
    convention after dividing through by ``C * n``.
    """

    def __init__(self, regularization_c: float):
        self.regularization_c = float(regularization_c)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.optimization_: dict[str, Any] | None = None

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> "StandardizedLogistic":
        matrix = np.asarray(matrix, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        standardized = (matrix - self.mean_) / self.scale_
        positives = max(float(labels.sum()), 1.0)
        negatives = max(float(len(labels) - labels.sum()), 1.0)
        sample_weights = np.where(
            labels > 0.5,
            len(labels) / (2.0 * positives),
            len(labels) / (2.0 * negatives),
        )
        l2 = 1.0 / max(self.regularization_c * len(labels), 1e-12)

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            weights = parameters[:-1]
            intercept = parameters[-1]
            logits = standardized @ weights + intercept
            losses = np.logaddexp(0.0, logits) - labels * logits
            probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
            residual = sample_weights * (probability - labels) / len(labels)
            value = float(
                np.sum(sample_weights * losses) / len(labels)
                + 0.5 * l2 * np.dot(weights, weights)
            )
            gradient = np.concatenate(
                (
                    standardized.T @ residual + l2 * weights,
                    np.asarray([residual.sum()]),
                )
            )
            return value, gradient

        result = minimize(
            objective,
            np.zeros(matrix.shape[1] + 1, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(f"logistic optimization failed: {result.message}")
        self.coef_ = result.x[:-1]
        self.intercept_ = float(result.x[-1])
        self.optimization_ = {
            "iterations": int(result.nit),
            "objective": float(result.fun),
            "message": str(result.message),
        }
        return self

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        if (
            self.mean_ is None
            or self.scale_ is None
            or self.coef_ is None
            or self.intercept_ is None
        ):
            raise RuntimeError("model is not fitted")
        standardized = (np.asarray(matrix, dtype=np.float64) - self.mean_) / self.scale_
        logits = standardized @ self.coef_ + self.intercept_
        positive = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        return np.column_stack((1.0 - positive, positive))


def make_model(config: dict[str, Any], seed: int) -> StandardizedLogistic:
    del seed
    return StandardizedLogistic(config["regularization_c"])


def save_identity_model(model: StandardizedLogistic, feature_names: list[str]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL_PATH,
        mean=model.mean_,
        scale=model.scale_,
        coefficient=model.coef_,
        intercept=np.asarray(model.intercept_, dtype=np.float64),
        regularization_c=np.asarray(model.regularization_c, dtype=np.float64),
        feature_names=np.asarray(feature_names),
        optimization_json=np.asarray(
            json.dumps(model.optimization_, ensure_ascii=False, sort_keys=True)
        ),
    )


def load_identity_model(feature_names: list[str]) -> StandardizedLogistic:
    with np.load(MODEL_PATH, allow_pickle=False) as source:
        frozen_names = [str(value) for value in source["feature_names"]]
        if frozen_names != feature_names:
            raise ValueError("shared identity feature order mismatch")
        model = StandardizedLogistic(float(source["regularization_c"].item()))
        model.mean_ = source["mean"].astype(np.float64)
        model.scale_ = source["scale"].astype(np.float64)
        model.coef_ = source["coefficient"].astype(np.float64)
        model.intercept_ = float(source["intercept"].item())
        model.optimization_ = json.loads(str(source["optimization_json"].item()))
    return model


def feature_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"][name]) for name in names] for row in rows],
        dtype=np.float64,
    )


def pair_f1(labels: np.ndarray, clusters: np.ndarray) -> float:
    tp = fp = fn = 0
    for first in range(len(labels)):
        truth = labels[first] == labels[first + 1 :]
        prediction = clusters[first] == clusters[first + 1 :]
        tp += int(np.logical_and(truth, prediction).sum())
        fp += int(np.logical_and(~truth, prediction).sum())
        fn += int(np.logical_and(truth, ~prediction).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def linkage_context(rows: list[dict[str, Any]], costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt = {}
    for row in rows:
        gt[int(row["first_fragment_id"])] = int(row["first_gt_plane"])
        gt[int(row["second_fragment_id"])] = int(row["second_gt_plane"])
    nodes = sorted(gt)
    lookup = {node: index for index, node in enumerate(nodes)}
    distance = np.ones((len(nodes), len(nodes)), dtype=np.float64)
    np.fill_diagonal(distance, 0)
    for row, cost in zip(rows, costs, strict=True):
        first = lookup[int(row["first_fragment_id"])]
        second = lookup[int(row["second_fragment_id"])]
        distance[first, second] = distance[second, first] = min(
            distance[first, second], float(cost)
        )
    tree = (
        linkage(squareform(distance, checks=False), method="average")
        if len(nodes) > 1
        else np.empty((0, 4))
    )
    return np.asarray([gt[node] for node in nodes], dtype=np.int64), tree


def train_identity(config: dict[str, Any]) -> dict[str, Any]:
    edges_path = Path(config["development_edges_json"])
    rows = load_json(edges_path)
    rows_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_scene.setdefault(str(row["scene"]), []).append(row)
    scenes = sorted(rows_by_scene)
    feature_names = sorted(
        set(rows[0]["features"]) - BIAS_EXCLUSIVE_FEATURES
    )
    scored: dict[str, np.ndarray] = {}
    folds = int(config["calibration_folds"])
    for fold in range(folds):
        validation = [scene for index, scene in enumerate(scenes) if index % folds == fold]
        fitting = [scene for scene in scenes if scene not in validation]
        fitting_rows = [row for scene in fitting for row in rows_by_scene[scene]]
        model = make_model(config, int(config["seed"]) + fold)
        model.fit(
            feature_matrix(fitting_rows, feature_names),
            np.asarray([bool(row["same_gt_plane"]) for row in fitting_rows]),
        )
        for scene in validation:
            scored[scene] = 1.0 - model.predict_proba(
                feature_matrix(rows_by_scene[scene], feature_names)
            )[:, 1]
    all_costs = np.concatenate([scored[scene] for scene in scenes])
    thresholds = np.unique(
        np.quantile(all_costs, np.linspace(0, 1, int(config["threshold_count"])))
    )
    thresholds = np.concatenate(([float(all_costs.min()) - 1e-12], thresholds))
    contexts = {
        scene: linkage_context(rows_by_scene[scene], scored[scene]) for scene in scenes
    }
    macro_f1 = []
    for threshold in thresholds:
        values = []
        for scene in scenes:
            labels, tree = contexts[scene]
            clusters = (
                np.ones(1, dtype=np.int32)
                if len(labels) == 1
                else fcluster(tree, t=float(threshold), criterion="distance")
            )
            values.append(pair_f1(labels, clusters))
        macro_f1.append(float(np.mean(values)))
    best_value = max(macro_f1)
    best_indices = [
        index for index, value in enumerate(macro_f1) if value == best_value
    ]
    best_index = min(best_indices, key=lambda index: float(thresholds[index]))
    final_model = make_model(config, int(config["seed"]))
    final_model.fit(
        feature_matrix(rows, feature_names),
        np.asarray([bool(row["same_gt_plane"]) for row in rows]),
    )
    save_identity_model(final_model, feature_names)
    manifest = {
        "registration_id": REGISTRATION_ID,
        "status": "frozen",
        "development_scene_count": len(scenes),
        "development_scenes": scenes,
        "development_edge_count": len(rows),
        "development_edges_sha256": sha256_file(edges_path),
        "feature_group": "joint_without_bias_with_tangent_support",
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "model": "logistic",
        "regularization_c": config["regularization_c"],
        "calibration_folds": folds,
        "threshold_count": config["threshold_count"],
        "identity_cost_threshold": float(thresholds[best_index]),
        "oof_macro_partition_pair_f1": best_value,
        "primary_tie_count": len(best_indices),
        "tie_break": "smaller threshold; higher-order tie breaks not reached",
        "model_path": str(MODEL_PATH),
        "model_sha256": sha256_file(MODEL_PATH),
        "final_scene_usage_count": 0,
        "frozen_utc": utc_now(),
    }
    write_json(IDENTITY_MANIFEST, manifest)
    return manifest


def cluster_fragments(
    descriptors: dict[int, dict[str, Any]],
    scene_scale: float,
    model: Any,
    feature_names: list[str],
    threshold: float,
    pixel_diagonal: float,
) -> tuple[dict[int, int], int]:
    nodes = sorted(descriptors)
    if len(nodes) == 1:
        return {nodes[0]: 0}, 0
    rows = []
    for first_index, first in enumerate(nodes):
        for second in nodes[first_index + 1 :]:
            rows.append(
                {
                    "first": first,
                    "second": second,
                    "features": pair_feature_values(
                        descriptors[first],
                        descriptors[second],
                        scene_scale,
                        pixel_diagonal,
                    ),
                }
            )
    probabilities = model.predict_proba(feature_matrix(rows, feature_names))[:, 1]
    distances = np.ones((len(nodes), len(nodes)), dtype=np.float64)
    np.fill_diagonal(distances, 0)
    lookup = {node: index for index, node in enumerate(nodes)}
    for row, probability in zip(rows, probabilities, strict=True):
        first = lookup[row["first"]]
        second = lookup[row["second"]]
        distances[first, second] = distances[second, first] = 1.0 - probability
    tree = linkage(squareform(distances, checks=False), method="average")
    labels = fcluster(tree, t=float(threshold), criterion="distance")
    labels = np.unique(labels, return_inverse=True)[1].astype(np.int32)
    return {node: int(labels[index]) for index, node in enumerate(nodes)}, len(rows)


def plane_parameters(
    points: np.ndarray, plane_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals = []
    offsets = []
    counts = []
    for plane_id in sorted(int(x) for x in np.unique(plane_ids)):
        selected = points[plane_ids == plane_id]
        normal, offset, _ = fit_plane(selected)
        normals.append(normal)
        offsets.append(offset)
        counts.append(len(selected))
    return (
        np.asarray(normals, dtype=np.float32),
        np.asarray(offsets, dtype=np.float32),
        np.asarray(counts, dtype=np.int32),
    )


def predict_scene(
    scene_record: dict[str, Any],
    method_id: str,
    enable_split: bool,
    config: dict[str, Any],
    identity_manifest: dict[str, Any],
    model: Any,
    splitter: IntraFragmentSplitter | None,
    device: torch.device,
    source_commit: str,
) -> dict[str, Any]:
    scene = scene_record["scene_id"]
    global_path = locate_npz(Path(config["global_cache_root"]), scene, "global_cache_e14.npz")
    support_path = locate_npz(
        Path(config["direct_support_root"]), scene, "direct_support_e14.npz"
    )
    arrays, provenance = initial_observations(
        global_path, support_path, config["minimum_fragment_points"]
    )
    if enable_split:
        if splitter is None:
            raise ValueError("M16 requires splitter")
        working_fragments, split_audit = apply_split(
            arrays, splitter, config, device
        )
    else:
        working_fragments = arrays["initial_fragment_ids"].copy()
        split_audit = []
    descriptors, scene_scale = fragment_descriptors(
        arrays, working_fragments, config["maximum_geometry_points"]
    )
    cluster_map, edge_count = cluster_fragments(
        descriptors,
        scene_scale,
        model,
        identity_manifest["feature_names"],
        identity_manifest["identity_cost_threshold"],
        config["pixel_diagonal"],
    )
    plane_ids = np.asarray(
        [cluster_map[int(fragment)] for fragment in working_fragments],
        dtype=np.int32,
    )
    normals, offsets, counts = plane_parameters(arrays["points"], plane_ids)
    prediction_dir = Path(config["prediction_root"]) / method_id / scene
    prediction_dir.mkdir(parents=True, exist_ok=True)
    output = prediction_dir / "prediction_e16.npz"
    split_json = json.dumps(split_audit, ensure_ascii=False, sort_keys=True)
    provenance_json = json.dumps(
        {"initial_fragments": provenance, "split_audit": split_audit},
        ensure_ascii=False,
        sort_keys=True,
    )
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_name=np.asarray("LightRecon3D_E16_prediction"),
        schema_version=np.asarray(1, dtype=np.int16),
        registration_id=np.asarray(REGISTRATION_ID),
        scene_id=np.asarray(scene),
        method_id=np.asarray(method_id),
        learned_split_enabled=np.asarray(enable_split),
        view_count=np.asarray(scene_record["view_count"], dtype=np.int16),
        points=arrays["points"].astype(np.float32),
        colors=arrays["colors"].astype(np.uint8),
        view_indices=arrays["view_indices"].astype(np.int16),
        pixel_xy=arrays["pixel_xy"].astype(np.int32),
        global_point_indices=arrays["global_point_indices"].astype(np.int64),
        support_record_indices=arrays["record_indices"].astype(np.int16),
        record_local_plane_labels=arrays["local_plane_labels"].astype(np.int16),
        initial_fragment_ids=arrays["initial_fragment_ids"].astype(np.int32),
        post_split_fragment_ids=working_fragments.astype(np.int32),
        plane_ids=plane_ids,
        plane_normals=normals,
        plane_offsets=offsets,
        plane_point_counts=counts,
        fragment_provenance_json=np.asarray(provenance_json),
        split_audit_json=np.asarray(split_json),
        source_global_cache_sha256=np.asarray(sha256_file(global_path)),
        source_direct_support_sha256=np.asarray(sha256_file(support_path)),
        shared_identity_model_sha256=np.asarray(
            identity_manifest["model_sha256"]
        ),
        m_checkpoint_sha256=np.asarray(
            config["m_checkpoint_sha256"] if enable_split else ""
        ),
        config_sha256=np.asarray(sha256_file(DEFAULT_CONFIG)),
        source_commit=np.asarray(source_commit),
        pixel_coordinate_order=np.asarray("xy"),
        pixel_coordinate_space=np.asarray("dust3r_aligned_pointmap"),
    )
    os.replace(temporary, output)
    hashes = {
        "whole_file_sha256": sha256_file(output),
        "support_sha256": array_sha256(
            np.column_stack(
                (
                    arrays["view_indices"],
                    arrays["pixel_xy"],
                    arrays["global_point_indices"],
                )
            )
        ),
        "plane_id_sha256": array_sha256(plane_ids),
        "plane_parameter_sha256": array_sha256(
            np.column_stack((normals, offsets))
        ),
        "provenance_sha256": sha256_bytes(provenance_json.encode("utf-8")),
    }
    return {
        "scene_id": scene,
        "method_id": method_id,
        "view_count": int(scene_record["view_count"]),
        "prediction_path": str(output),
        "observation_count": len(plane_ids),
        "initial_fragment_count": len(np.unique(arrays["initial_fragment_ids"])),
        "post_split_fragment_count": len(np.unique(working_fragments)),
        "final_plane_count": len(normals),
        "identity_edge_count": edge_count,
        **hashes,
        "completed_utc": utc_now(),
    }


def run_predictions(config: dict[str, Any], source_commit: str) -> dict[str, Any]:
    identity_manifest = load_json(IDENTITY_MANIFEST)
    if sha256_file(MODEL_PATH) != identity_manifest["model_sha256"]:
        raise ValueError("shared identity model hash mismatch")
    model = load_identity_model(identity_manifest["feature_names"])
    view_manifest = load_json(Path(config["e14_view_manifest"]))
    records = view_manifest["records"]
    if len(records) != 128:
        raise ValueError("final scene count mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitter = load_splitter(config, device)
    state_path = OUT / "prediction_execution_state.json"
    state = (
        load_json(state_path)
        if state_path.is_file()
        else {"registration_id": REGISTRATION_ID, "predictions": {}}
    )
    predictions = state.setdefault("predictions", {})
    for method_id, enabled in (("B16", False), ("M16", True)):
        for record in records:
            key = f"{method_id}:{record['scene_id']}"
            existing = predictions.get(key)
            if existing:
                path = Path(existing["prediction_path"])
                if path.is_file() and sha256_file(path) == existing["whole_file_sha256"]:
                    continue
                raise ValueError(f"frozen prediction changed: {key}")
            receipt = predict_scene(
                record,
                method_id,
                enabled,
                config,
                identity_manifest,
                model,
                splitter,
                device,
                source_commit,
            )
            predictions[key] = receipt
            state.update(
                {
                    "status": "in_progress",
                    "completed_prediction_count": len(predictions),
                    "expected_prediction_count": 256,
                    "final_layout_json_parse_count": 0,
                    "final_camera_pose_numeric_parse_count": 0,
                    "final_annotation_semantic_parse_count": 0,
                    "updated_utc": utc_now(),
                }
            )
            write_json(state_path, state)
    ordered = [
        predictions[f"{method}:{record['scene_id']}"]
        for method in ("B16", "M16")
        for record in records
    ]
    manifest = {
        "registration_id": REGISTRATION_ID,
        "status": "frozen",
        "prediction_count": len(ordered),
        "prediction_frozen_before_GT_evaluation": len(ordered) == 256,
        "final_layout_json_parse_count": 0,
        "final_camera_pose_numeric_parse_count": 0,
        "final_annotation_semantic_parse_count": 0,
        "predictions": ordered,
        "frozen_utc": utc_now(),
    }
    manifest_path = OUT / "prediction_manifest.json"
    write_json(manifest_path, manifest)
    (OUT / "prediction_manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  prediction_manifest.json\n", encoding="utf-8"
    )
    state.update(
        {
            "status": "frozen",
            "completed_prediction_count": len(ordered),
            "prediction_manifest_sha256": sha256_file(manifest_path),
            "completed_utc": utc_now(),
        }
    )
    write_json(state_path, state)
    return manifest


def run_development_smoke(config: dict[str, Any]) -> dict[str, Any]:
    identity_manifest = load_json(IDENTITY_MANIFEST)
    if sha256_file(MODEL_PATH) != identity_manifest["model_sha256"]:
        raise ValueError("shared identity model hash mismatch")
    model = load_identity_model(identity_manifest["feature_names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitter = load_splitter(config, device)
    smoke_config = dict(config)
    smoke_config["global_cache_root"] = config["historical_smoke_global_root"]
    smoke_config["direct_support_root"] = config["historical_smoke_support_root"]
    smoke_config["prediction_root"] = str(OUT / "development_smoke_predictions")
    records = []
    for path in sorted(
        Path(smoke_config["global_cache_root"]).rglob("global_cache_e14.npz")
    ):
        with np.load(path, allow_pickle=False) as source:
            records.append(
                {
                    "scene_id": str(source["scene_id"].item()),
                    "view_count": int(source["view_count"].item()),
                }
            )
    if sorted(row["view_count"] for row in records) != [4, 5]:
        raise ValueError("historical smoke must contain one 4-view and one 5-view case")
    receipts = []
    for method_id, enabled in (("B16", False), ("M16", True)):
        for record in records:
            receipts.append(
                predict_scene(
                    record,
                    method_id,
                    enabled,
                    smoke_config,
                    identity_manifest,
                    model,
                    splitter,
                    device,
                    "development-smoke-only",
                )
            )
    report = {
        "registration_id": REGISTRATION_ID,
        "status": "passed",
        "development_only": True,
        "case_count": len(receipts),
        "view_counts": sorted({row["view_count"] for row in receipts}),
        "methods": sorted({row["method_id"] for row in receipts}),
        "receipts": receipts,
        "final_scene_prediction_count": 0,
        "final_layout_json_parse_count": 0,
        "final_camera_pose_numeric_parse_count": 0,
        "final_annotation_semantic_parse_count": 0,
        "completed_utc": utc_now(),
    }
    write_json(OUT / "development_smoke_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase", choices=("develop", "smoke", "predict"), required=True
    )
    parser.add_argument("--source-commit", default="")
    args = parser.parse_args()
    config = load_config(args.config)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "develop":
        result = train_identity(config)
    elif args.phase == "smoke":
        result = run_development_smoke(config)
    else:
        if not args.source_commit:
            raise ValueError("--source-commit is required for formal predictions")
        result = run_predictions(config, args.source_commit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
