"""Render honest 3D comparisons for the final research-practice batch.

The renderer consumes the *actual* full-cache NPZ outputs recorded by
``batch_execution.json``.  For every passed scene it validates that ordinary
RANSAC and learning-guided RANSAC use the same ordered global point cloud, then
renders RGB, ordinary RANSAC plane instances, and guided RANSAC plane instances
from identical cameras.

Only NumPy and Pillow are required.  The script is CPU-only and never reruns
DUSt3R, plane extraction, training, or evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCHEMA_VERSION = 1
RANSAC_ARTIFACT = "global_ransac"
GUIDED_ARTIFACT = "learning_guided_ransac"
RANSAC_METHOD = "global_ransac_cc"
GUIDED_METHOD = "learning_guided_ransac_cc"

BACKGROUND = np.asarray([247, 249, 252], dtype=np.uint8)
UNASSIGNED = np.asarray([188, 194, 202], dtype=np.uint8)
PLANE_PALETTE = np.asarray(
    [
        [0, 114, 178],
        [230, 159, 0],
        [0, 158, 115],
        [213, 94, 0],
        [204, 121, 167],
        [86, 180, 233],
        [240, 228, 66],
        [0, 0, 0],
        [80, 80, 210],
        [226, 110, 40],
        [40, 155, 75],
        [190, 55, 90],
        [125, 85, 180],
        [35, 165, 180],
        [180, 150, 45],
        [90, 105, 115],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class CameraView:
    name: str
    yaw_deg: float
    pitch_deg: float


@dataclass
class Prediction:
    path: Path
    method: str
    points: np.ndarray
    original_colors: np.ndarray
    point_plane_ids: np.ndarray
    plane_ids: np.ndarray
    runtime_seconds: float | None


@dataclass
class SceneInput:
    item_id: str
    scene_name: str
    baseline_path: Path
    guided_path: Path


@dataclass
class Projection:
    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray
    indices: np.ndarray
    width: int
    height: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar")
    item = array.reshape(()).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def scalar_float(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar")
    return float(array.reshape(()).item())


def load_prediction(path: Path, expected_method: str) -> Prediction:
    if not path.is_file():
        raise FileNotFoundError(f"missing prediction NPZ: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "points",
            "point_plane_ids",
            "plane_ids",
            "method",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        points = np.asarray(payload["points"], dtype=np.float32)
        assignments = np.asarray(payload["point_plane_ids"], dtype=np.int32).reshape(-1)
        plane_ids = np.asarray(payload["plane_ids"], dtype=np.int32).reshape(-1)
        method = scalar_text(payload["method"], "method")
        if "original_colors" in payload.files:
            colors = np.asarray(payload["original_colors"])
        elif "colors" in payload.files:
            colors = np.asarray(payload["colors"])
        else:
            raise ValueError(f"{path} is missing original_colors/colors")
        runtime = (
            scalar_float(payload["runtime_seconds"], "runtime_seconds")
            if "runtime_seconds" in payload.files
            else None
        )

    if method != expected_method:
        raise ValueError(f"{path} method is {method!r}, expected {expected_method!r}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path} points must have shape (N, 3), got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(f"{path} colors shape {colors.shape} != points {points.shape}")
    if len(assignments) != len(points):
        raise ValueError(f"{path} point_plane_ids length does not match points")
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains non-finite 3D points")
    if len(points) < 3:
        raise ValueError(f"{path} contains fewer than three 3D points")
    colors = np.clip(colors, 0, 255).astype(np.uint8, copy=False)
    return Prediction(
        path=path,
        method=method,
        points=points,
        original_colors=colors,
        point_plane_ids=assignments,
        plane_ids=plane_ids,
        runtime_seconds=runtime,
    )


def resolve_artifact(record: Any, execution_dir: Path, label: str) -> Path:
    if isinstance(record, str):
        raw_path = record
    elif isinstance(record, dict) and isinstance(record.get("path"), str):
        raw_path = record["path"]
    else:
        raise ValueError(f"artifact {label} must contain a path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = execution_dir / path
    return path


def scene_inputs(batch_execution_json: Path) -> list[SceneInput]:
    payload = json.loads(batch_execution_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("batch execution JSON must contain an items list")
    result: list[SceneInput] = []
    seen_items: set[str] = set()
    seen_scenes: set[str] = set()
    for item in payload["items"]:
        if not isinstance(item, dict) or item.get("status") != "pass":
            continue
        item_id = str(item.get("id", ""))
        scene_name = str(item.get("scene_name", ""))
        if not item_id or not scene_name:
            raise ValueError("passed batch items require id and scene_name")
        if item_id in seen_items or scene_name in seen_scenes:
            raise ValueError(f"duplicate item or scene in final batch: {item_id}/{scene_name}")
        artifacts = item.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"{item_id} has no artifacts map")
        missing = [
            key for key in (RANSAC_ARTIFACT, GUIDED_ARTIFACT) if key not in artifacts
        ]
        if missing:
            raise ValueError(f"{item_id} is missing final artifacts: {missing}")
        result.append(
            SceneInput(
                item_id=item_id,
                scene_name=scene_name,
                baseline_path=resolve_artifact(
                    artifacts[RANSAC_ARTIFACT], batch_execution_json.parent, RANSAC_ARTIFACT
                ),
                guided_path=resolve_artifact(
                    artifacts[GUIDED_ARTIFACT], batch_execution_json.parent, GUIDED_ARTIFACT
                ),
            )
        )
        seen_items.add(item_id)
        seen_scenes.add(scene_name)
    if not result:
        raise ValueError("batch execution contains no passed scenes")
    return result


def validate_identical_cache(baseline: Prediction, guided: Prediction) -> None:
    if baseline.points.shape != guided.points.shape:
        raise ValueError(
            f"final methods do not use the same cache shape: "
            f"{baseline.points.shape} vs {guided.points.shape}"
        )
    if not np.array_equal(baseline.points, guided.points):
        delta = float(np.max(np.abs(baseline.points - guided.points)))
        raise ValueError(
            "final methods do not use the identical ordered global point cache; "
            f"maximum coordinate delta is {delta:.8g}"
        )
    if not np.array_equal(baseline.original_colors, guided.original_colors):
        raise ValueError("final methods do not preserve identical original RGB colors")


def stable_sample_indices(count: int, max_points: int) -> np.ndarray:
    if max_points < 1000:
        raise ValueError("max_points must be at least 1000")
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    # Endpoint-inclusive, deterministic sampling makes repeated renders auditable.
    return np.linspace(0, count - 1, num=max_points, dtype=np.int64)


def canonical_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample = points[stable_sample_indices(len(points), min(100_000, len(points)))]
    center = np.median(sample.astype(np.float64), axis=0)
    centered = sample.astype(np.float64) - center
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, np.argsort(values)[::-1]]
    for axis in range(3):
        column = basis[:, axis]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            basis[:, axis] *= -1
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
    return center.astype(np.float64), basis.astype(np.float64)


def view_axes(view: CameraView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(view.yaw_deg)
    pitch = math.radians(view.pitch_deg)
    forward = np.asarray(
        [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)],
        dtype=np.float64,
    )
    up_reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up_reference)
    norm = float(np.linalg.norm(right))
    if norm < 1e-8:
        raise ValueError(f"view {view.name} is parallel to the up axis")
    right /= norm
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    return right, up, forward


def project_values(
    points: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    view: CameraView,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canonical = (points.astype(np.float64) - center) @ basis
    right, up, forward = view_axes(view)
    return canonical @ right, canonical @ up, canonical @ forward


def robust_extent(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.005, 0.995])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0
    margin = 0.045 * (high - low)
    return float(low - margin), float(high + margin)


def prepare_projection(
    points: np.ndarray,
    sample_indices: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    view: CameraView,
    width: int,
    height: int,
    padding: int = 18,
) -> Projection:
    sampled = points[sample_indices]
    horizontal, vertical, depth = project_values(sampled, center, basis, view)
    x_min, x_max = robust_extent(horizontal)
    y_min, y_max = robust_extent(vertical)
    available_width = max(1, width - 2 * padding)
    available_height = max(1, height - 2 * padding)
    scale = min(available_width / (x_max - x_min), available_height / (y_max - y_min))
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    x = np.rint((horizontal - x_center) * scale + width / 2).astype(np.int32)
    y = np.rint(height / 2 - (vertical - y_center) * scale).astype(np.int32)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if int(valid.sum()) < 100:
        raise ValueError(f"view {view.name} retains fewer than 100 projected points")
    return Projection(
        x=x[valid],
        y=y[valid],
        depth=depth[valid].astype(np.float32),
        indices=sample_indices[valid],
        width=width,
        height=height,
    )


def visible_raster_samples(
    projection: Projection, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    if radius < 0 or radius > 3:
        raise ValueError("point_radius must be between 0 and 3")
    offsets = [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius + 1
    ]
    if not offsets:
        offsets = [(0, 0)]
    pixel_chunks: list[np.ndarray] = []
    depth_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    for dx, dy in offsets:
        x = projection.x + dx
        y = projection.y + dy
        valid = (x >= 0) & (x < projection.width) & (y >= 0) & (y < projection.height)
        pixel_chunks.append((y[valid] * projection.width + x[valid]).astype(np.int64))
        depth_chunks.append(projection.depth[valid])
        index_chunks.append(projection.indices[valid])
    pixels = np.concatenate(pixel_chunks)
    depths = np.concatenate(depth_chunks)
    expanded_indices = np.concatenate(index_chunks)
    # Sort by pixel first and depth second.  The last sample for each pixel is
    # the nearest one (largest camera-forward depth), giving a real z-buffer.
    order = np.lexsort((depths, pixels))
    sorted_pixels = pixels[order]
    keep = np.r_[sorted_pixels[1:] != sorted_pixels[:-1], True]
    selected = order[keep]
    selected_pixels = pixels[selected]
    selected_indices = expanded_indices[selected]
    return selected_pixels, selected_indices


def zbuffer_render(
    projection: Projection,
    point_colors: np.ndarray,
    raster_samples: tuple[np.ndarray, np.ndarray],
    background: np.ndarray = BACKGROUND,
) -> Image.Image:
    selected_pixels, selected_indices = raster_samples
    canvas = np.broadcast_to(
        np.asarray(background, dtype=np.uint8),
        (projection.height, projection.width, 3),
    ).copy()
    canvas.reshape(-1, 3)[selected_pixels] = point_colors[selected_indices]
    return Image.fromarray(canvas)


def plane_colors(assignments: np.ndarray) -> np.ndarray:
    output = np.broadcast_to(UNASSIGNED, (len(assignments), 3)).copy()
    assigned = assignments >= 0
    output[assigned] = PLANE_PALETTE[assignments[assigned] % len(PLANE_PALETTE)]
    return output


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def labeled_panel(
    render: Image.Image,
    title: str,
    subtitle: str,
    width: int,
    footer_height: int = 58,
) -> Image.Image:
    panel = Image.new("RGB", (width, render.height + footer_height), "white")
    panel.paste(render, (0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, width - 1, render.height - 1), outline=(196, 205, 216), width=2)
    draw.rectangle((0, render.height, width, panel.height), fill=(241, 245, 249))
    draw.text((12, render.height + 7), title, fill=(25, 52, 83), font=load_font(17, True))
    draw.text((12, render.height + 31), subtitle, fill=(78, 91, 107), font=load_font(12))
    return panel


def parse_views(text: str) -> list[CameraView]:
    if text.strip().lower() == "auto":
        return []
    result: list[CameraView] = []
    for spec in text.split(","):
        fields = spec.strip().split(":")
        if len(fields) != 3 or not fields[0]:
            raise ValueError(
                "views must be comma-separated name:yaw_deg:pitch_deg specifications"
            )
        result.append(CameraView(fields[0], float(fields[1]), float(fields[2])))
    if not result:
        raise ValueError("at least one camera view is required")
    return result


def automatic_views(
    points: np.ndarray, center: np.ndarray, basis: np.ndarray
) -> list[CameraView]:
    sample = points[stable_sample_indices(len(points), min(60_000, len(points)))]
    candidates: list[tuple[float, float]] = []
    for yaw in range(-165, 180, 30):
        view = CameraView("candidate", float(yaw), 20.0)
        horizontal, vertical, _ = project_values(sample, center, basis, view)
        x0, x1 = np.quantile(horizontal, [0.005, 0.995])
        y0, y1 = np.quantile(vertical, [0.005, 0.995])
        dx = max(float(x1 - x0), 1e-9)
        dy = max(float(y1 - y0), 1e-9)
        balance = min(dx, dy) / max(dx, dy)
        candidates.append((math.log(dx * dy) + 0.35 * balance, float(yaw)))
    candidates.sort(reverse=True)
    first_yaw = candidates[0][1]

    def circular_distance(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    second_yaw = next(
        yaw
        for _, yaw in candidates[1:]
        if 45.0 <= circular_distance(first_yaw, yaw) <= 135.0
    )
    return [
        CameraView("auto_overview", first_yaw, 20.0),
        CameraView("auto_alternate", second_yaw, 20.0),
        CameraView("auto_elevated", first_yaw, 58.0),
    ]


def audit_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("per_scene") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("audit JSON must contain a per_scene list")
    return {str(row["item_id"]): row for row in rows if isinstance(row, dict)}


def metric_caption(row: dict[str, Any] | None) -> str:
    if not row:
        return "F1 not supplied (render-only run)"
    baseline = float(row["ransac_pairwise_f1"])
    guided = float(row["guided_pairwise_f1"])
    delta = float(row["delta_pairwise_f1"])
    return f"pairwise F1 {baseline:.3f} -> {guided:.3f} (delta {delta:+.3f})"


def prediction_summary(prediction: Prediction) -> str:
    assigned = int(np.count_nonzero(prediction.point_plane_ids >= 0))
    rate = assigned / len(prediction.points)
    return f"{len(prediction.plane_ids)} planes | assigned {rate:.1%}"


def compose_scene_sheet(
    item: SceneInput,
    baseline: Prediction,
    guided: Prediction,
    views: Sequence[CameraView],
    output_dir: Path,
    metric_row: dict[str, Any] | None,
    width: int,
    height: int,
    max_points: int,
    point_radius: int,
) -> dict[str, Any]:
    validate_identical_cache(baseline, guided)
    scene_dir = output_dir / item.item_id
    scene_dir.mkdir(parents=True, exist_ok=False)
    sample_indices = stable_sample_indices(len(baseline.points), max_points)
    center, basis = canonical_frame(baseline.points)
    selected_views = list(views) if views else automatic_views(baseline.points, center, basis)
    rgb = baseline.original_colors
    baseline_colors = plane_colors(baseline.point_plane_ids)
    guided_colors = plane_colors(guided.point_plane_ids)
    render_sets = [
        ("RGB global cloud", "shared DUSt3R-aligned points", rgb),
        ("Ordinary RANSAC", prediction_summary(baseline), baseline_colors),
        ("Guided RANSAC", prediction_summary(guided), guided_colors),
    ]
    rows: list[list[Image.Image]] = [[], [], []]
    per_view_paths: list[str] = []
    for column, view in enumerate(selected_views):
        projection = prepare_projection(
            baseline.points, sample_indices, center, basis, view, width, height
        )
        raster_samples = visible_raster_samples(projection, point_radius)
        view_panels: list[Image.Image] = []
        for row_index, (title, subtitle, colors) in enumerate(render_sets):
            rendered = zbuffer_render(
                projection,
                colors,
                raster_samples,
                np.asarray([22, 29, 38], dtype=np.uint8) if row_index == 0 else BACKGROUND,
            )
            panel = labeled_panel(
                rendered,
                title,
                f"{subtitle} | view {view.name} ({view.yaw_deg:g}, {view.pitch_deg:g})",
                width,
            )
            rows[row_index].append(panel)
            view_panels.append(panel)
        view_sheet = Image.new(
            "RGB",
            (width * len(view_panels), view_panels[0].height),
            "white",
        )
        for index, panel in enumerate(view_panels):
            view_sheet.paste(panel, (index * width, 0))
        view_path = scene_dir / f"view_{column:02d}_{view.name}.png"
        view_sheet.save(view_path, optimize=True)
        per_view_paths.append(str(view_path))

    header_height = 94
    note_height = 42
    panel_height = rows[0][0].height
    sheet = Image.new(
        "RGB",
        (width * len(selected_views), header_height + panel_height * 3 + note_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (18, 12),
        f"{item.scene_name}: final 3D reconstruction comparison",
        fill=(22, 51, 82),
        font=load_font(25, True),
    )
    draw.text((18, 48), metric_caption(metric_row), fill=(45, 66, 89), font=load_font(16))
    draw.text(
        (18, 71),
        f"same ordered cache: {len(baseline.points):,} points | rendered sample: {len(sample_indices):,}",
        fill=(83, 96, 110),
        font=load_font(13),
    )
    for row_index, row in enumerate(rows):
        for column, panel in enumerate(row):
            sheet.paste(panel, (column * width, header_height + row_index * panel_height))
    note_y = header_height + panel_height * 3
    draw.rectangle((0, note_y, sheet.width, sheet.height), fill=(234, 240, 247))
    draw.text(
        (18, note_y + 12),
        "Plane colors are method-local instance IDs. Compare boundaries, fragmentation and merging; matching colors do not imply correspondence.",
        fill=(49, 69, 92),
        font=load_font(13),
    )
    sheet_path = scene_dir / f"{item.scene_name}_multiview_comparison.png"
    sheet.save(sheet_path, optimize=True)
    return {
        "item_id": item.item_id,
        "scene_name": item.scene_name,
        "points": int(len(baseline.points)),
        "rendered_points": int(len(sample_indices)),
        "baseline_planes": int(len(baseline.plane_ids)),
        "guided_planes": int(len(guided.plane_ids)),
        "baseline_assigned_points": int(np.count_nonzero(baseline.point_plane_ids >= 0)),
        "guided_assigned_points": int(np.count_nonzero(guided.point_plane_ids >= 0)),
        "baseline_npz": str(baseline.path),
        "baseline_npz_sha256": file_sha256(baseline.path),
        "guided_npz": str(guided.path),
        "guided_npz_sha256": file_sha256(guided.path),
        "metric_caption": metric_caption(metric_row),
        "multiview_png": str(sheet_path),
        "per_view_pngs": per_view_paths,
        "camera_views": [view.__dict__ for view in selected_views],
        "canonical_center": center.tolist(),
        "canonical_basis_columns": basis.tolist(),
    }


def representative_tile(path: Path, item: dict[str, Any], tile_width: int) -> Image.Image:
    with Image.open(path) as source_file:
        source = source_file.convert("RGB")
        # The source is a 3-column sheet: RGB, ordinary RANSAC, guided RANSAC.
        third = source.width // 3
        baseline = source.crop((third, 0, 2 * third, source.height))
        guided = source.crop((2 * third, 0, source.width, source.height))
        content = Image.new("RGB", (third * 2, source.height), "white")
        content.paste(baseline, (0, 0))
        content.paste(guided, (third, 0))
    target_height = max(1, round(content.height * tile_width / content.width))
    content = content.resize((tile_width, target_height), Image.Resampling.LANCZOS)
    header = 54
    tile = Image.new("RGB", (tile_width, target_height + header), "white")
    tile.paste(content, (0, header))
    draw = ImageDraw.Draw(tile)
    draw.text((10, 7), item["scene_name"], fill=(22, 51, 82), font=load_font(18, True))
    draw.text((10, 31), item["metric_caption"], fill=(70, 84, 101), font=load_font(12))
    draw.rectangle((0, 0, tile.width - 1, tile.height - 1), outline=(191, 201, 213), width=2)
    return tile


def compose_batch_contact_sheet(
    scene_rows: list[dict[str, Any]], output_dir: Path, columns: int = 2
) -> Path:
    tile_width = 1000
    tiles = [
        representative_tile(Path(row["per_view_pngs"][0]), row, tile_width)
        for row in scene_rows
    ]
    tile_height = max(tile.height for tile in tiles)
    header = 96
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (tile_width * columns, header + tile_height * rows), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (18, 12),
        "Validation-scene 3D comparison: ordinary vs learning-guided RANSAC",
        fill=(22, 51, 82),
        font=load_font(26, True),
    )
    draw.text(
        (18, 52),
        "Each pair uses the same global point cache and the same camera. Plane colors are method-local IDs.",
        fill=(65, 82, 100),
        font=load_font(15),
    )
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_width
        y = header + (index // columns) * tile_height
        sheet.paste(tile, (x, y))
    path = output_dir / "all_scenes_final_3d_contact_sheet.png"
    sheet.save(path, optimize=True)
    return path


def markdown_summary(result: dict[str, Any]) -> str:
    lines = [
        "# Final research-practice 3D visualizations",
        "",
        (
            f"Rendered {len(result['scenes'])} passed scenes from the real final batch. "
            "Ordinary and guided RANSAC are rendered on identical ordered global point caches."
        ),
        "",
        "| Scene | Points | RANSAC planes | Guided planes | Metric | Multiview |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in result["scenes"]:
        lines.append(
            f"| {row['scene_name']} | {row['points']} | {row['baseline_planes']} | "
            f"{row['guided_planes']} | {row['metric_caption']} | {row['multiview_png']} |"
        )
    lines.extend(
        [
            "",
            "## Reading the figures",
            "",
            "Rows are RGB global cloud, ordinary RANSAC, and learning-guided RANSAC. "
            "Columns are identical 3D cameras. Plane colors are local to each method; "
            "compare geometric support boundaries, fragmentation, and over-merging.",
            "",
            "These are render-only outputs. No reconstruction, training, or metric was recomputed.",
            "",
        ]
    )
    return "\n".join(lines)


def run_visualization(
    batch_execution_json: Path,
    output_dir: Path,
    audit_json: Path | None,
    views: Sequence[CameraView],
    width: int,
    height: int,
    max_points: int,
    point_radius: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if width < 320 or height < 240:
        raise ValueError("panel width/height must be at least 320x240")
    inputs = scene_inputs(batch_execution_json)
    metrics = audit_index(audit_json)
    output_dir.mkdir(parents=True, exist_ok=False)
    scene_rows: list[dict[str, Any]] = []
    for item in inputs:
        baseline = load_prediction(item.baseline_path, RANSAC_METHOD)
        guided = load_prediction(item.guided_path, GUIDED_METHOD)
        row = compose_scene_sheet(
            item,
            baseline,
            guided,
            views,
            output_dir,
            metrics.get(item.item_id),
            width,
            height,
            max_points,
            point_radius,
        )
        scene_rows.append(row)
        print(
            json.dumps(
                {
                    "scene": item.scene_name,
                    "points": row["points"],
                    "ransac_planes": row["baseline_planes"],
                    "guided_planes": row["guided_planes"],
                    "output": row["multiview_png"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    contact_sheet = compose_batch_contact_sheet(scene_rows, output_dir)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "research_practice_final_3d_visualization",
        "render_only": True,
        "source_batch_execution_json": str(batch_execution_json),
        "source_batch_execution_sha256": file_sha256(batch_execution_json),
        "source_audit_json": str(audit_json) if audit_json is not None else None,
        "source_audit_sha256": file_sha256(audit_json) if audit_json is not None else None,
        "camera_mode": "explicit" if views else "automatic_per_scene",
        "requested_camera_views": [view.__dict__ for view in views],
        "render_settings": {
            "panel_width": width,
            "panel_height": height,
            "max_points": max_points,
            "point_radius": point_radius,
            "projection": "orthographic_pca_canonical",
            "occlusion": "nearest_depth_z_buffer",
        },
        "contact_sheet_png": str(contact_sheet),
        "scenes": scene_rows,
    }
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(markdown_summary(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        "Render actual final-batch RGB/RANSAC/guided-RANSAC 3D comparisons"
    )
    parser.add_argument("--batch_execution_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audit_json")
    parser.add_argument(
        "--views",
        default="auto",
        help="comma-separated name:yaw_deg:pitch_deg cameras in the PCA frame",
    )
    parser.add_argument("--panel_width", type=int, default=720)
    parser.add_argument("--panel_height", type=int, default=500)
    parser.add_argument("--max_points", type=int, default=220_000)
    parser.add_argument("--point_radius", type=int, default=1)
    args = parser.parse_args()
    result = run_visualization(
        Path(args.batch_execution_json),
        Path(args.output_dir),
        Path(args.audit_json) if args.audit_json else None,
        parse_views(args.views),
        args.panel_width,
        args.panel_height,
        args.max_points,
        args.point_radius,
    )
    print(
        json.dumps(
            {
                "scenes": len(result["scenes"]),
                "contact_sheet": result["contact_sheet_png"],
                "manifest": str(Path(args.output_dir) / "visualization_manifest.json"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
