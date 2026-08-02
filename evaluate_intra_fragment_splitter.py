"""Evaluate learned intra-fragment affinities as actual connected components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as sparse_components
from scipy.spatial.distance import squareform

from train_intra_fragment_splitter import (
    IntraFragmentSplitter,
    SceneStore,
    flatten_records,
    local_neighbors,
    prepare_fragment,
    tensor_case,
)


def connected_components(
    point_count: int,
    neighbors: np.ndarray,
    edge_probability: np.ndarray,
    boundary_probability: np.ndarray,
    *,
    affinity_threshold: float,
    boundary_threshold: float,
) -> np.ndarray:
    neighbors = np.asarray(neighbors, dtype=np.int64)
    edge_probability = np.asarray(edge_probability)
    boundary_probability = np.asarray(boundary_probability)
    source = np.repeat(
        np.arange(point_count, dtype=np.int64), neighbors.shape[1]
    )
    target = neighbors.reshape(-1)
    accepted = edge_probability.reshape(-1) >= float(affinity_threshold)
    accepted &= (
        np.maximum(
            boundary_probability[source],
            boundary_probability[target],
        )
        < float(boundary_threshold)
    )
    graph = coo_matrix(
        (
            np.ones(int(accepted.sum()), dtype=np.uint8),
            (source[accepted], target[accepted]),
        ),
        shape=(point_count, point_count),
    ).tocsr()
    _, labels = sparse_components(
        graph,
        directed=False,
        return_labels=True,
    )
    return labels.astype(np.int32)


def partition_pair_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have identical shapes")

    def pairs_from_counts(counts: np.ndarray) -> int:
        return int(np.sum(counts * (counts - 1) // 2))

    _, pred_counts = np.unique(prediction, return_counts=True)
    _, truth_counts = np.unique(truth, return_counts=True)
    joint = np.stack((prediction, truth), axis=1)
    _, joint_counts = np.unique(joint, axis=0, return_counts=True)
    true_positive = pairs_from_counts(joint_counts)
    predicted_positive = pairs_from_counts(pred_counts)
    actual_positive = pairs_from_counts(truth_counts)
    precision = true_positive / max(predicted_positive, 1)
    recall = true_positive / max(actual_positive, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_component_count": int(len(pred_counts)),
        "gt_component_count": int(len(truth_counts)),
    }


def average_linkage_tree(affinity: np.ndarray) -> np.ndarray:
    """Build an average-linkage tree from a symmetric learned affinity matrix."""
    affinity = np.asarray(affinity, dtype=np.float64)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("affinity must be a square matrix")
    if len(affinity) < 2:
        return np.empty((0, 4), dtype=np.float64)
    symmetric = np.clip(0.5 * (affinity + affinity.T), 0.0, 1.0)
    distance = 1.0 - symmetric
    np.fill_diagonal(distance, 0.0)
    return linkage(squareform(distance, checks=False), method="average")


def average_linkage_components(
    tree: np.ndarray,
    point_count: int,
    *,
    affinity_threshold: float,
) -> np.ndarray:
    """Cut a cached average-linkage tree at a learned affinity threshold."""
    if point_count < 2:
        return np.zeros((point_count,), dtype=np.int32)
    labels = fcluster(
        tree,
        t=1.0 - float(affinity_threshold),
        criterion="distance",
    )
    _, compact = np.unique(labels, return_inverse=True)
    return compact.astype(np.int32)


@torch.no_grad()
def dense_pair_affinity(
    model: IntraFragmentSplitter,
    embedding: torch.Tensor,
    xyz: torch.Tensor,
    pixels: torch.Tensor,
    views: torch.Tensor,
    *,
    pair_chunk_size: int = 65_536,
) -> np.ndarray:
    """Evaluate all unordered point pairs, averaging both directed logits."""
    point_count = int(len(embedding))
    affinity = np.eye(point_count, dtype=np.float32)
    if point_count < 2:
        return affinity
    pairs = torch.triu_indices(
        point_count,
        point_count,
        offset=1,
        device=embedding.device,
    )
    for start in range(0, pairs.shape[1], int(pair_chunk_size)):
        source = pairs[0, start : start + int(pair_chunk_size)]
        target = pairs[1, start : start + int(pair_chunk_size)]
        forward = model.pair_logits(
            embedding, xyz, pixels, views, source, target
        )
        backward = model.pair_logits(
            embedding, xyz, pixels, views, target, source
        )
        probability = torch.sigmoid(0.5 * (forward + backward))
        source_np = source.cpu().numpy()
        target_np = target.cpu().numpy()
        probability_np = probability.cpu().numpy()
        affinity[source_np, target_np] = probability_np
        affinity[target_np, source_np] = probability_np
    return affinity


@torch.no_grad()
def predict_records(
    model: IntraFragmentSplitter,
    store: SceneStore,
    records: list[dict],
    *,
    device: torch.device,
    max_points: int,
    pixel_size: float,
    neighbor_count: int,
    seed: int,
    compute_dense_affinity: bool = True,
) -> list[dict]:
    model.eval()
    predictions = []
    for index, record in enumerate(records):
        rng = np.random.default_rng(seed + index)
        case_np = prepare_fragment(
            store.load(record["scene"]),
            record["fragment_id"],
            max_points,
            pixel_size,
            rng,
            True,
            model.architecture == "light",
        )
        case = tensor_case(case_np, device)
        neighbors = local_neighbors(
            case["xyz"], neighbor_count, case["pixels"], case["views"]
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
        split_logit = model.fragment_split_logit(
            embedding, boundary_logits, edge_logits
        )
        dense_tree = None
        if compute_dense_affinity:
            dense_affinity = dense_pair_affinity(
                model,
                embedding,
                case["xyz"],
                case["pixels"],
                case["views"],
            )
            dense_tree = average_linkage_tree(dense_affinity)
        predictions.append(
            {
                "scene": record["scene"],
                "fragment_id": int(record["fragment_id"]),
                "truth": case_np["labels"],
                "neighbors": neighbors.cpu().numpy(),
                "edge_probability": torch.sigmoid(edge_logits).cpu().numpy(),
                "boundary_probability": torch.sigmoid(
                    boundary_logits
                ).cpu().numpy(),
                "average_linkage_tree": dense_tree,
                "split_probability": (
                    float(torch.sigmoid(split_logit).cpu())
                    if split_logit is not None
                    else None
                ),
            }
        )
    return predictions


def evaluate_predictions(
    predictions: list[dict],
    affinity_threshold: float,
    boundary_threshold: float,
    *,
    partition_method: str = "local_connected_components",
    split_gate_threshold: float | None = None,
) -> dict:
    rows = []
    for prediction in predictions:
        gate_rejects = (
            split_gate_threshold is not None
            and prediction.get("split_probability") is not None
            and prediction["split_probability"] < float(split_gate_threshold)
        )
        if gate_rejects:
            clusters = np.zeros((len(prediction["truth"]),), dtype=np.int32)
        elif partition_method == "local_connected_components":
            clusters = connected_components(
                len(prediction["truth"]),
                prediction["neighbors"],
                prediction["edge_probability"],
                prediction["boundary_probability"],
                affinity_threshold=affinity_threshold,
                boundary_threshold=boundary_threshold,
            )
        elif partition_method == "dense_average_linkage":
            clusters = average_linkage_components(
                prediction["average_linkage_tree"],
                len(prediction["truth"]),
                affinity_threshold=affinity_threshold,
            )
        else:
            raise ValueError(f"Unsupported partition method: {partition_method}")
        rows.append(
            {
                "scene": prediction["scene"],
                "fragment_id": prediction["fragment_id"],
                **partition_pair_metrics(clusters, prediction["truth"]),
            }
        )
    return {
        "fragment_count": len(rows),
        "mean_precision": float(np.mean([row["precision"] for row in rows])),
        "mean_recall": float(np.mean([row["recall"] for row in rows])),
        "mean_f1": float(np.mean([row["f1"] for row in rows])),
        "mean_predicted_component_count": float(
            np.mean([row["predicted_component_count"] for row in rows])
        ),
        "mean_gt_component_count": float(
            np.mean([row["gt_component_count"] for row in rows])
        ),
        "rows": rows,
    }


def baseline_metrics(predictions: list[dict]) -> dict:
    rows = []
    for prediction in predictions:
        rows.append(
            partition_pair_metrics(
                np.zeros((len(prediction["truth"]),), dtype=np.int32),
                prediction["truth"],
            )
        )
    return {
        "fragment_count": len(rows),
        "mean_precision": float(np.mean([row["precision"] for row in rows])),
        "mean_recall": float(np.mean([row["recall"] for row in rows])),
        "mean_f1": float(np.mean([row["f1"] for row in rows])),
        "mean_predicted_component_count": 1.0,
        "mean_gt_component_count": float(
            np.mean([row["gt_component_count"] for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate and evaluate learned intra-fragment splitting."
    )
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_points", type=int, default=512)
    parser.add_argument("--neighbor_count", type=int, default=8)
    parser.add_argument("--pixel_size", type=float, default=512.0)
    parser.add_argument("--max_calibration_fragments", type=int, default=64)
    parser.add_argument("--max_calibration_pure_fragments", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--partition_methods",
        default="local_connected_components,dense_average_linkage",
        help="Comma-separated subset of local_connected_components,dense_average_linkage.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest = json.loads(
        Path(args.manifest_json).resolve().read_text(encoding="utf-8")
    )
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    checkpoint = torch.load(
        Path(args.checkpoint).resolve(), map_location=device, weights_only=False
    )
    model = IntraFragmentSplitter(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    partition_methods = tuple(
        method.strip()
        for method in args.partition_methods.split(",")
        if method.strip()
    )
    supported_methods = {
        "local_connected_components",
        "dense_average_linkage",
    }
    if not partition_methods or not set(partition_methods) <= supported_methods:
        raise ValueError(f"Unsupported partition_methods: {partition_methods}")
    compute_dense_affinity = "dense_average_linkage" in partition_methods
    store = SceneStore(manifest)
    hard_train = [
        record
        for record in flatten_records(manifest, "train")
        if record["hard_impure"]
    ][: int(args.max_calibration_fragments)]
    pure_train = [
        record
        for record in flatten_records(manifest, "train")
        if not record["hard_impure"]
    ][: int(args.max_calibration_pure_fragments)]
    hard_val = [
        record
        for record in flatten_records(manifest, "val")
        if record["hard_impure"]
    ]
    calibration_predictions = predict_records(
        model,
        store,
        hard_train + pure_train,
        device=device,
        max_points=args.max_points,
        pixel_size=args.pixel_size,
        neighbor_count=args.neighbor_count,
        seed=args.seed + 10_000,
        compute_dense_affinity=compute_dense_affinity,
    )
    val_predictions = predict_records(
        model,
        store,
        hard_val,
        device=device,
        max_points=args.max_points,
        pixel_size=args.pixel_size,
        neighbor_count=args.neighbor_count,
        seed=args.seed + 20_000,
        compute_dense_affinity=compute_dense_affinity,
    )
    gate_thresholds = (
        (0.2, 0.35, 0.5, 0.65, 0.8)
        if calibration_predictions
        and calibration_predictions[0].get("split_probability") is not None
        else (None,)
    )
    methods = {}
    if "local_connected_components" in partition_methods:
        local_candidates = []
        for affinity_threshold in np.linspace(0.3, 0.9, 13):
            for boundary_threshold in (0.4, 0.6, 0.8, 1.01):
                for split_gate_threshold in gate_thresholds:
                    metrics = evaluate_predictions(
                        calibration_predictions,
                        float(affinity_threshold),
                        float(boundary_threshold),
                        split_gate_threshold=split_gate_threshold,
                    )
                    local_candidates.append(
                        {
                            "affinity_threshold": float(affinity_threshold),
                            "boundary_threshold": float(boundary_threshold),
                            "split_gate_threshold": split_gate_threshold,
                            "mean_f1": metrics["mean_f1"],
                        }
                    )
        best_local = max(local_candidates, key=lambda row: row["mean_f1"])
        local_calibration = evaluate_predictions(
            calibration_predictions,
            best_local["affinity_threshold"],
            best_local["boundary_threshold"],
            split_gate_threshold=best_local["split_gate_threshold"],
        )
        local_validation = evaluate_predictions(
            val_predictions,
            best_local["affinity_threshold"],
            best_local["boundary_threshold"],
            split_gate_threshold=best_local["split_gate_threshold"],
        )
        methods["local_connected_components"] = {
            "selected_thresholds": best_local,
            "calibration": local_calibration,
            "validation": local_validation,
        }
    if "dense_average_linkage" in partition_methods:
        average_candidates = []
        for affinity_threshold in np.linspace(0.1, 0.9, 33):
            for split_gate_threshold in gate_thresholds:
                metrics = evaluate_predictions(
                    calibration_predictions,
                    float(affinity_threshold),
                    1.01,
                    partition_method="dense_average_linkage",
                    split_gate_threshold=split_gate_threshold,
                )
                average_candidates.append(
                    {
                        "affinity_threshold": float(affinity_threshold),
                        "boundary_threshold": None,
                        "split_gate_threshold": split_gate_threshold,
                        "mean_f1": metrics["mean_f1"],
                    }
                )
        best_average = max(average_candidates, key=lambda row: row["mean_f1"])
        average_calibration = evaluate_predictions(
            calibration_predictions,
            best_average["affinity_threshold"],
            1.01,
            partition_method="dense_average_linkage",
            split_gate_threshold=best_average["split_gate_threshold"],
        )
        average_validation = evaluate_predictions(
            val_predictions,
            best_average["affinity_threshold"],
            1.01,
            partition_method="dense_average_linkage",
            split_gate_threshold=best_average["split_gate_threshold"],
        )
        methods["dense_average_linkage"] = {
            "selected_thresholds": best_average,
            "calibration": average_calibration,
            "validation": average_validation,
        }
    selected_method = max(
        methods,
        key=lambda name: methods[name]["calibration"]["mean_f1"],
    )
    selected = methods[selected_method]
    validation = selected["validation"]
    baseline = baseline_metrics(val_predictions)
    result = {
        "schema_version": 2,
        "task": "intra_fragment_learned_affinity_partition",
        "calibration_protocol": (
            "partition method and thresholds tuned only on hard "
            "and pure training-scene fragments with equal caps"
        ),
        "calibration_composition": {
            "hard_fragment_count": len(hard_train),
            "pure_fragment_count": len(pure_train),
        },
        "selected_method": selected_method,
        "selected_thresholds": selected["selected_thresholds"],
        "calibration": selected["calibration"],
        "validation": validation,
        "methods": methods,
        "unsplit_baseline": baseline,
        "validation_delta": {
            "mean_precision": validation["mean_precision"]
            - baseline["mean_precision"],
            "mean_recall": validation["mean_recall"] - baseline["mean_recall"],
            "mean_f1": validation["mean_f1"] - baseline["mean_f1"],
        },
        "config": vars(args),
        "outputs": {"summary_json": str(output_dir / "summary.json")},
        "limitations": [
            "Metrics are computed on deterministic 512-point single-view crops.",
            "The three validation scenes are difficult development scenes, not a final held-out benchmark.",
            "The split components are not yet reinserted into global fragment identity inference.",
            "Dense all-pairs affinity is currently a crop-level evaluator, not a scalable full-resolution inference path.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_method": selected_method,
                "selected_thresholds": selected["selected_thresholds"],
                "calibration": {
                    key: value
                    for key, value in selected["calibration"].items()
                    if key != "rows"
                },
                "validation": {
                    key: value
                    for key, value in validation.items()
                    if key != "rows"
                },
                "unsplit_baseline": baseline,
                "method_validation": {
                    name: {
                        key: value
                        for key, value in method["validation"].items()
                        if key != "rows"
                    }
                    for name, method in methods.items()
                },
                "validation_delta": result["validation_delta"],
                "summary_json": result["outputs"]["summary_json"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
