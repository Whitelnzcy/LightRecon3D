"""Create the frozen W3 accuracy/efficiency bundle for the practice report."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 1
IMAGE_SIZE = 512
STAGE2_MANIFEST = "learned_region_merge_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def percentile(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(milliseconds: list[float]) -> dict[str, Any]:
    if not milliseconds:
        raise ValueError("latency list is empty")
    return {
        "samples": len(milliseconds),
        "mean_ms": float(statistics.fmean(milliseconds)),
        "p50_ms": percentile(milliseconds, 50),
        "p95_ms": percentile(milliseconds, 95),
        "min_ms": min(milliseconds),
        "max_ms": max(milliseconds),
        "raw_ms": [float(value) for value in milliseconds],
    }


def pair_count(count: int) -> int:
    return int(count) * (int(count) - 1) // 2


def support_partition_metrics(prediction: Any, ground_truth: Any) -> dict[str, Any]:
    import numpy as np

    pred = np.asarray(prediction, dtype=np.int32).reshape(-1)
    gt = np.asarray(ground_truth, dtype=np.int32).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError("predicted and GT support arrays must have identical shapes")
    assigned = pred >= 0
    gt_labeled = gt >= 0
    domain = assigned & gt_labeled
    pred_ids = sorted(int(value) for value in np.unique(pred[domain]) if value >= 0)
    gt_ids = sorted(int(value) for value in np.unique(gt[domain]) if value >= 0)
    contingency = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)
    for row, pred_id in enumerate(pred_ids):
        for column, gt_id in enumerate(gt_ids):
            contingency[row, column] = int(
                ((pred == pred_id) & (gt == gt_id) & domain).sum()
            )
    same_both = sum(pair_count(value) for value in contingency.ravel())
    same_pred = sum(pair_count(value) for value in contingency.sum(axis=1))
    same_gt = sum(pair_count(value) for value in contingency.sum(axis=0))
    precision = same_both / same_pred if same_pred else 0.0
    recall = same_both / same_gt if same_gt else 0.0
    pairwise_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    domain_count = int(domain.sum())
    purity = (
        float(contingency.max(axis=1).sum()) / domain_count
        if domain_count and contingency.shape[1]
        else 0.0
    )
    completeness = (
        float(contingency.max(axis=0).sum()) / domain_count
        if domain_count and contingency.shape[0]
        else 0.0
    )
    purity_completeness_f1 = (
        2.0 * purity * completeness / (purity + completeness)
        if purity + completeness
        else 0.0
    )
    return {
        "points": int(len(pred)),
        "predicted_planes": len(pred_ids),
        "observed_gt_planes": len(gt_ids),
        "assignment_rate": float(assigned.mean()) if len(pred) else 0.0,
        "gt_labeled_coverage": (
            float(domain.sum() / gt_labeled.sum()) if bool(gt_labeled.any()) else 0.0
        ),
        "pairwise_precision": float(precision),
        "pairwise_recall": float(recall),
        "pairwise_f1": float(pairwise_f1),
        "predicted_purity": float(purity),
        "gt_completeness": float(completeness),
        "purity_completeness_f1": float(purity_completeness_f1),
    }


def resolve_source_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def stage1_accuracy_rows(final_manifest_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np

    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for item in final_manifest.get("items", []):
        item_id = str(item.get("id", ""))
        scene_name = str(item.get("expected_scene_name", ""))
        stage2_dir = Path(str(item.get("input_dir", "")))
        stage2_manifest = stage2_dir / STAGE2_MANIFEST
        if not stage2_manifest.is_file():
            raise FileNotFoundError(
                f"{item_id} missing Stage2 source manifest: {stage2_manifest}"
            )
        payload = json.loads(stage2_manifest.read_text(encoding="utf-8"))
        files = payload.get("files", [])
        if not files:
            raise ValueError(f"{item_id} Stage2 source manifest has no files")
        source_manifests.append(
            {
                "item_id": item_id,
                "scene_name": scene_name,
                **file_record(stage2_manifest),
                "records": len(files),
            }
        )
        for record in files:
            source_path = resolve_source_path(str(record.get("input", "")), stage2_manifest)
            key = os.path.normcase(os.path.abspath(str(source_path)))
            if key in seen_sources:
                raise ValueError(f"duplicate Stage1 source in final set: {source_path}")
            seen_sources.add(key)
            if not source_path.is_file():
                raise FileNotFoundError(f"missing Stage1 support file: {source_path}")
            with np.load(source_path, allow_pickle=False) as raw:
                required = ("point_plane_ids", "gt_point_plane_ids")
                missing = [name for name in required if name not in raw.files]
                if missing:
                    raise ValueError(f"{source_path} missing arrays: {missing}")
                metrics = support_partition_metrics(
                    raw["point_plane_ids"], raw["gt_point_plane_ids"]
                )
                sample_idx = (
                    int(raw["sample_idx"].item()) if "sample_idx" in raw.files else -1
                )
            rows.append(
                {
                    "item_id": item_id,
                    "scene_name": scene_name,
                    "sample_idx": sample_idx,
                    "source_npz": str(source_path),
                    **metrics,
                }
            )
    if len({row["scene_name"] for row in rows}) != 8:
        raise ValueError("Stage1 accuracy input must contain eight independent scenes")
    return rows, source_manifests


ACCURACY_KEYS = (
    "assignment_rate",
    "gt_labeled_coverage",
    "pairwise_precision",
    "pairwise_recall",
    "pairwise_f1",
    "predicted_purity",
    "gt_completeness",
    "purity_completeness_f1",
)


def aggregate_accuracy(rows: list[dict[str, Any]], group_key: str | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    if group_key is None:
        grouped["all"] = rows
    else:
        for row in rows:
            grouped.setdefault(str(row[group_key]), []).append(row)
    output = []
    for group, group_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "group": group,
            "records": len(group_rows),
            "points": int(sum(row["points"] for row in group_rows)),
        }
        for key in ACCURACY_KEYS:
            values = [float(row[key]) for row in group_rows]
            summary[f"{key}_mean"] = float(statistics.fmean(values))
            summary[f"{key}_median"] = float(statistics.median(values))
        output.append(summary)
    return output


def parameter_record(model: Any, label: str) -> dict[str, Any]:
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    return {
        "component": label,
        "parameters": int(sum(value.numel() for value in parameters)),
        "trainable_parameters": int(
            sum(value.numel() for value in parameters if value.requires_grad)
        ),
        "parameter_bytes": int(
            sum(value.numel() * value.element_size() for value in parameters)
        ),
        "buffer_bytes": int(
            sum(value.numel() * value.element_size() for value in buffers)
        ),
    }


def benchmark_cuda(
    torch: Any,
    label: str,
    function: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    use_inference_mode: bool = True,
    cleanup_between: bool = False,
) -> dict[str, Any]:
    if warmup < 0 or repeats < 1:
        raise ValueError("invalid benchmark warmup/repeat count")
    context = torch.inference_mode() if use_inference_mode else contextlib.nullcontext()
    with context:
        for _ in range(warmup):
            value = function()
            torch.cuda.synchronize()
            del value
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline_allocated = int(torch.cuda.memory_allocated())
        baseline_reserved = int(torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            value = function()
            torch.cuda.synchronize()
            durations.append((time.perf_counter() - started) * 1000.0)
            del value
            if cleanup_between:
                gc.collect()
                torch.cuda.empty_cache()
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    return {
        "component": label,
        "warmup": int(warmup),
        "repeats": int(repeats),
        **latency_summary(durations),
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": max(0, peak_allocated - baseline_allocated),
    }


def load_benchmark_sample(
    selection_plan_path: Path,
    *,
    root_dir: Path,
    feature_cache_path: Path,
) -> tuple[Any, dict[str, Any]]:
    import argparse as argparse_module
    import torch
    from torch.utils.data import DataLoader, Subset

    from export_stage1_pred_support_teacher_npz import build_dataset
    from train_stage1_clean_pair_baseline import move_batch
    from train_stage1_plane_masks import build_views

    selection = json.loads(selection_plan_path.read_text(encoding="utf-8"))
    items = selection.get("items", [])
    if not items:
        raise ValueError("selection plan has no items")
    selected = items[0]
    indices = selected.get("selected_indices", [])
    if not indices:
        raise ValueError("first selected scene has no dataset indices")
    index = int(indices[0])
    cache = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
    cache_config = cache.get("config", {})
    del cache
    args = argparse_module.Namespace(
        root_dir=str(root_dir),
        split="val",
        train_ratio=0.9,
        image_size=IMAGE_SIZE,
        pair_strategy="all",
        pair_max_view_id_gap=None,
        indices=str(index),
        start_idx=0,
        count=0,
    )
    dataset, validated_indices = build_dataset(args, cache_config=cache_config)
    loader = DataLoader(
        Subset(dataset, validated_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    batch = move_batch(next(iter(loader)), torch.device("cuda"))
    view1, view2 = build_views(batch, f"efficiency_val_{index}")
    return (view1, view2), {
        "sample_idx": index,
        "scene_name": str(selected.get("scene_name", "")),
        "pair_group": str(selected.get("pair_group", "")),
        "image_size": IMAGE_SIZE,
    }


def first_alignment_images(batch_execution: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    import numpy as np

    for item in batch_execution.get("items", []):
        artifact = item.get("artifacts", {}).get("global_cloud_cache", {})
        cache_path = Path(str(artifact.get("path", "")))
        if not cache_path.is_file():
            continue
        with np.load(cache_path, allow_pickle=True) as raw:
            if "dust3r_view_registry_json" in raw.files:
                registry = json.loads(str(raw["dust3r_view_registry_json"].item()))
                registry = sorted(
                    registry, key=lambda row: int(row["alignment_view_index"])
                )
                paths = [str(row["image_path"]) for row in registry]
                path_source = "dust3r_view_registry_json"
            elif "dust3r_image_paths" in raw.files:
                paths = [str(value) for value in raw["dust3r_image_paths"].tolist()]
                path_source = "dust3r_image_paths"
            else:
                continue
        if len(paths) >= 2 and all(Path(path).is_file() for path in paths):
            return paths, {
                "item_id": str(item.get("id", "")),
                "scene_name": str(item.get("scene_name", "")),
                "global_cloud_cache": str(cache_path),
                "global_cloud_cache_sha256": str(artifact.get("sha256", "")),
                "views": len(paths),
                "image_path_source": path_source,
            }
    raise FileNotFoundError("no final global cache with valid image paths")


def gpu_benchmarks(
    *,
    selection_plan_path: Path,
    batch_execution: dict[str, Any],
    root_dir: Path,
    weights_path: Path,
    stage1_checkpoint: Path,
    feature_cache_path: Path,
    stage2_checkpoint: Path,
    head_warmup: int,
    head_repeats: int,
    backbone_warmup: int,
    backbone_repeats: int,
    alignment_repeats: int,
) -> dict[str, Any]:
    import argparse as argparse_module
    import numpy as np
    import torch

    from export_stage1_pred_support_teacher_npz import (
        load_stage1_head,
        predict_stage1,
    )
    from export_stage2_learned_region_merge_editables import load_model as load_stage2
    from export_stage3_scene_plane_fusion import run_dust3r_global_alignment
    from models.build_backbone import build_dust3r_backbone
    from train_stage1_plane_masks import point_map_from_result

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen W3 benchmark")
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda")
    cache = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
    cache_config = cache.get("config", {})
    del cache
    (view1, view2), sample = load_benchmark_sample(
        selection_plan_path,
        root_dir=root_dir,
        feature_cache_path=feature_cache_path,
    )
    backbone = build_dust3r_backbone(str(weights_path), device=device)
    backbone.eval()
    head, saved_args, _ = load_stage1_head(
        str(stage1_checkpoint), cache_config, device
    )
    stage2_model, stage2_payload = load_stage2(str(stage2_checkpoint), device)
    models = [
        parameter_record(backbone, "shared_dust3r_backbone"),
        parameter_record(head, "stage1_plane_support_head"),
        parameter_record(stage2_model, "stage2_region_merge_mlp"),
    ]

    backbone_latency = benchmark_cuda(
        torch,
        "dust3r_pair_backbone_2_images",
        lambda: backbone(view1, view2),
        warmup=backbone_warmup,
        repeats=backbone_repeats,
    )
    with torch.inference_mode():
        result1, result2 = backbone(view1, view2)
        torch.cuda.synchronize()
    target_hw1 = tuple(int(value) for value in point_map_from_result(result1)[0].shape[:2])
    target_hw2 = tuple(int(value) for value in point_map_from_result(result2)[0].shape[:2])
    head_views = [(result1, view1["img"], target_hw1), (result2, view2["img"], target_hw2)]
    head_counter = {"value": 0}

    def run_head() -> Any:
        index = head_counter["value"] % 2
        head_counter["value"] += 1
        result, image, target_hw = head_views[index]
        return predict_stage1(head, saved_args, result, image, target_hw)

    head_latency = benchmark_cuda(
        torch,
        "stage1_plane_support_head_per_image",
        run_head,
        warmup=head_warmup,
        repeats=head_repeats,
    )
    stage2_input_dim = int(stage2_payload.get("input_dim", 14))
    stage2_input = torch.zeros((64, stage2_input_dim), dtype=torch.float32, device=device)
    stage2_latency = benchmark_cuda(
        torch,
        "stage2_region_merge_mlp_64_candidate_pairs",
        lambda: stage2_model(stage2_input),
        warmup=head_warmup,
        repeats=head_repeats,
    )

    del result1, result2, head_views, head, stage2_model, stage2_input
    gc.collect()
    torch.cuda.empty_cache()
    alignment_images, alignment_source = first_alignment_images(batch_execution)
    alignment_args = argparse_module.Namespace(
        image_size=IMAGE_SIZE,
        scene_graph="complete",
        batch_size=1,
        niter=300,
        schedule="cosine",
        lr=0.01,
    )

    def run_alignment() -> Any:
        views, loss, scene = run_dust3r_global_alignment(
            alignment_images, backbone, "cuda", alignment_args
        )
        return views, loss, scene

    alignment_latency = benchmark_cuda(
        torch,
        "dust3r_five_view_inference_and_global_alignment",
        run_alignment,
        warmup=0,
        repeats=alignment_repeats,
        use_inference_mode=False,
        cleanup_between=True,
    )
    return {
        "sample": sample,
        "alignment_source": alignment_source,
        "models": models,
        "latency": [
            backbone_latency,
            head_latency,
            stage2_latency,
            alignment_latency,
        ],
    }


def archived_stage_timings(batch: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for item in batch.get("items", []):
        for stage in item.get("stages", []):
            rows.append(
                {
                    "item_id": str(item.get("id", "")),
                    "scene_name": str(item.get("scene_name", "")),
                    "stage": str(stage.get("stage", "")),
                    "runtime_seconds": float(stage.get("runtime_seconds", 0.0)),
                    "reused_global_cache": bool(item.get("reused_global_cloud_cache")),
                }
            )
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["stage"], []).append(row["runtime_seconds"])
    summaries = []
    for stage, values in sorted(grouped.items()):
        summaries.append(
            {
                "stage": stage,
                "scenes": len(values),
                "mean_seconds": float(statistics.fmean(values)),
                "median_seconds": float(statistics.median(values)),
                "p95_seconds": percentile(values, 95),
            }
        )
    uncached = [
        row["runtime_seconds"]
        for row in rows
        if row["stage"] == "direct_support" and not row["reused_global_cache"]
    ]
    if uncached:
        summaries.append(
            {
                "stage": "direct_support_uncached_alignment_plus_export_proxy",
                "scenes": len(uncached),
                "mean_seconds": float(statistics.fmean(uncached)),
                "median_seconds": float(statistics.median(uncached)),
                "p95_seconds": percentile(uncached, 95),
            }
        )
    return rows, summaries


def hardware_record() -> dict[str, Any]:
    import numpy as np
    import torch

    row: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "gpu_total_memory_bytes": (
            int(torch.cuda.get_device_properties(0).total_memory)
            if torch.cuda.is_available()
            else 0
        ),
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        row["nvidia_smi"] = completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        row["nvidia_smi_error"] = str(error)
    return row


def markdown_report(result: dict[str, Any]) -> str:
    overall = result["stage1_accuracy"]["overall"][0]
    lines = [
        "# Research-practice final efficiency and Stage1 accuracy",
        "",
        f"Git SHA: `{result['git_sha']}`",
        "",
        f"Final method decision: `{result['final_audit_decision']}`",
        "",
        f"Frozen input resolution: `{IMAGE_SIZE} x {IMAGE_SIZE}`.",
        "",
        "## Stage1 support accuracy",
        "",
        (
            f"Evaluated {overall['records']} pair records from "
            f"{len(result['stage1_accuracy']['per_scene'])} independent scenes."
        ),
        "",
        "| Metric | Mean | Median |",
        "|---|---:|---:|",
    ]
    for key in ACCURACY_KEYS:
        lines.append(
            f"| {key} | {overall[f'{key}_mean']:.6f} | "
            f"{overall[f'{key}_median']:.6f} |"
        )
    lines.extend(
        [
            "",
            "These are support-record partition metrics on the frozen sampled Stage1 outputs; they are not full-image semantic AP.",
            "",
            "## Model footprint",
            "",
            "| Component | Parameters | Trainable | Parameter MiB |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in result["gpu_benchmarks"]["models"]:
        lines.append(
            f"| {row['component']} | {row['parameters']} | "
            f"{row['trainable_parameters']} | {row['parameter_bytes'] / 2**20:.3f} |"
        )
    lines.extend(
        [
            "",
            "DUSt3R is the shared reconstruction backbone. The Stage1 head and Stage2 MLP are the added lightweight components.",
            "",
            "| Checkpoint | MiB | SHA256 |",
            "|---|---:|---|",
        ]
    )
    for label, record in result["checkpoints"].items():
        lines.append(
            f"| {label} | {record['bytes'] / 2**20:.3f} | `{record['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Frozen GPU latency",
            "",
            "| Component | Samples | P50 (ms) | P95 (ms) | Peak allocated (MiB) | Incremental peak (MiB) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["gpu_benchmarks"]["latency"]:
        lines.append(
            f"| {row['component']} | {row['samples']} | {row['p50_ms']:.3f} | "
            f"{row['p95_ms']:.3f} | {row['peak_allocated_bytes'] / 2**20:.3f} | "
            f"{row['incremental_peak_allocated_bytes'] / 2**20:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Archived final-batch stages",
            "",
            "| Stage | Scenes | Mean (s) | Median (s) | P95 (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["archived_stage_summary"]:
        lines.append(
            f"| {row['stage']} | {row['scenes']} | {row['mean_seconds']:.3f} | "
            f"{row['median_seconds']:.3f} | {row['p95_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The final method was promoted through the quality path, not the efficiency path. Runtime varies with scene structure; no universal acceleration claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = set().union(*(row.keys() for row in rows))
    fields = [key for key in preferred if key in keys]
    fields.extend(sorted(keys - set(fields)))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    paths = {
        "final_manifest": Path(args.final_manifest),
        "selection_plan": Path(args.selection_plan),
        "batch_execution": Path(args.batch_execution_json),
        "final_audit": Path(args.final_audit_json),
        "weights": Path(args.weights_path),
        "stage1_checkpoint": Path(args.stage1_checkpoint),
        "feature_cache": Path(args.feature_cache_path),
        "stage2_checkpoint": Path(args.stage2_checkpoint),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    final_audit = json.loads(paths["final_audit"].read_text(encoding="utf-8"))
    decision = str(final_audit.get("gate", {}).get("decision", ""))
    if decision != "promote_learning_guided_ransac_final":
        raise ValueError(f"unexpected final audit decision: {decision}")
    batch = json.loads(paths["batch_execution"].read_text(encoding="utf-8"))
    if any(item.get("status") != "pass" for item in batch.get("items", [])):
        raise ValueError("final batch contains failed items")
    accuracy_rows, source_manifests = stage1_accuracy_rows(paths["final_manifest"])
    per_scene_accuracy = aggregate_accuracy(accuracy_rows, "scene_name")
    overall_accuracy = aggregate_accuracy(accuracy_rows)
    stage_rows, stage_summary = archived_stage_timings(batch)
    gpu = gpu_benchmarks(
        selection_plan_path=paths["selection_plan"],
        batch_execution=batch,
        root_dir=Path(args.root_dir),
        weights_path=paths["weights"],
        stage1_checkpoint=paths["stage1_checkpoint"],
        feature_cache_path=paths["feature_cache"],
        stage2_checkpoint=paths["stage2_checkpoint"],
        head_warmup=args.head_warmup,
        head_repeats=args.head_repeats,
        backbone_warmup=args.backbone_warmup,
        backbone_repeats=args.backbone_repeats,
        alignment_repeats=args.alignment_repeats,
    )
    input_records = {label: file_record(path) for label, path in paths.items()}
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "research_practice_final_efficiency",
        "git_sha": str(args.git_sha),
        "frozen_image_size": IMAGE_SIZE,
        "hardware": hardware_record(),
        "inputs": input_records,
        "final_audit_decision": decision,
        "stage1_accuracy": {
            "records": accuracy_rows,
            "per_scene": per_scene_accuracy,
            "overall": overall_accuracy,
            "source_stage2_manifests": source_manifests,
        },
        "checkpoints": {
            "shared_dust3r_weights": input_records["weights"],
            "stage1_plane_support_head": input_records["stage1_checkpoint"],
            "stage2_region_merge_mlp": input_records["stage2_checkpoint"],
        },
        "gpu_benchmarks": gpu,
        "archived_stage_rows": stage_rows,
        "archived_stage_summary": stage_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "efficiency_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "stage1_accuracy_records.csv",
        accuracy_rows,
        ["item_id", "scene_name", "sample_idx", "source_npz"],
    )
    write_csv(
        output_dir / "stage1_accuracy_per_scene.csv",
        per_scene_accuracy,
        ["group", "records", "points"],
    )
    write_csv(
        output_dir / "archived_stage_timings.csv",
        stage_rows,
        ["item_id", "scene_name", "stage", "runtime_seconds"],
    )
    write_csv(
        output_dir / "gpu_latency.csv",
        gpu["latency"],
        ["component", "samples", "mean_ms", "p50_ms", "p95_ms"],
    )
    (output_dir / "efficiency_report.md").write_text(
        markdown_report(result), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser("Frozen W3 research-practice efficiency benchmark")
    parser.add_argument("--final_manifest", required=True)
    parser.add_argument("--selection_plan", required=True)
    parser.add_argument("--batch_execution_json", required=True)
    parser.add_argument("--final_audit_json", required=True)
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--feature_cache_path", required=True)
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--git_sha", required=True)
    parser.add_argument("--head_warmup", type=int, default=10)
    parser.add_argument("--head_repeats", type=int, default=50)
    parser.add_argument("--backbone_warmup", type=int, default=2)
    parser.add_argument("--backbone_repeats", type=int, default=10)
    parser.add_argument("--alignment_repeats", type=int, default=3)
    args = parser.parse_args()
    result = run_benchmark(args)
    overall = result["stage1_accuracy"]["overall"][0]
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "stage1_records": overall["records"],
                "stage1_pairwise_f1_mean": overall["pairwise_f1_mean"],
                "stage1_gt_labeled_coverage_mean": overall[
                    "gt_labeled_coverage_mean"
                ],
                "final_audit_decision": result["final_audit_decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
