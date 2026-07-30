"""Render the actual frozen E16 reconstruction outputs.

This is a reporting-only utility. It reads the frozen RGB, DUSt3R global
point cloud, and B16/M16 predictions. It never changes predictions, metrics,
plane parameters, support, or GT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results" / "targeted_GT_sidecar_and_final_BM_e16_20260730_v1"
FIGURE_ROOT = RESULT_ROOT / "figures" / "actual_reconstruction"
CONFIG_PATH = ROOT / "configs" / "e16_final_bm.json"
SIDECAR_CONFIG_PATH = ROOT / "configs" / "e16_targeted_gt_sidecars.json"
SCENES = ("scene_01922", "scene_02817", "scene_00597")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sidecar_config = json.loads(SIDECAR_CONFIG_PATH.read_text(encoding="utf-8"))
    config["raw_rgb_root"] = sidecar_config["raw_rgb_root"]
    return config


def load_prediction_manifest() -> dict[tuple[str, str], Path]:
    payload = json.loads(
        (RESULT_ROOT / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    return {
        (row["scene_id"], row["method_id"]): Path(row["prediction_path"])
        for row in payload["predictions"]
    }


def matched_m_labels(b_labels: np.ndarray, m_labels: np.ndarray) -> dict[int, int]:
    """Map M display labels onto B colours by maximum support overlap."""
    mapping: dict[int, int] = {}
    used_b: set[int] = set()
    pairs: list[tuple[int, int, int]] = []
    for m_id in np.unique(m_labels):
        mask = m_labels == m_id
        b_ids, counts = np.unique(b_labels[mask], return_counts=True)
        for b_id, count in zip(b_ids.tolist(), counts.tolist()):
            pairs.append((int(count), int(m_id), int(b_id)))
    for _, m_id, b_id in sorted(pairs, reverse=True):
        if m_id not in mapping and b_id not in used_b:
            mapping[m_id] = b_id
            used_b.add(b_id)
    next_id = int(max(np.max(b_labels), np.max(m_labels))) + 1
    for m_id in np.unique(m_labels).tolist():
        if int(m_id) not in mapping:
            mapping[int(m_id)] = next_id
            next_id += 1
    return mapping


def label_colours(labels: np.ndarray, colour_index: dict[int, int] | None = None):
    from matplotlib import colormaps

    palette = colormaps["tab20"]
    mapped = np.asarray(
        [colour_index.get(int(v), int(v)) if colour_index else int(v) for v in labels]
    )
    return palette((mapped % 20) / 20.0)


def choose_view(prediction: dict[str, np.ndarray]) -> int:
    view_ids, counts = np.unique(prediction["view_indices"], return_counts=True)
    return int(view_ids[np.argmax(counts)])


def make_overlay(
    image: np.ndarray,
    pixel_xy: np.ndarray,
    labels: np.ndarray,
    colour_index: dict[int, int] | None = None,
) -> np.ndarray:
    from PIL import Image

    target_w, target_h = 512, 288
    resized = np.asarray(
        Image.fromarray(image).resize((target_w, target_h), Image.Resampling.LANCZOS)
    ).copy()
    colours = (label_colours(labels, colour_index)[:, :3] * 255).astype(np.uint8)
    x = np.clip(pixel_xy[:, 0], 0, target_w - 1)
    y = np.clip(pixel_xy[:, 1], 0, target_h - 1)
    overlay = resized.astype(np.float32)
    overlay[y, x] = 0.24 * overlay[y, x] + 0.76 * colours
    return overlay.astype(np.uint8)


def robust_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    centre = np.median(points, axis=0)
    distance = np.linalg.norm(points - centre, axis=1)
    threshold = np.quantile(distance, 0.985)
    keep = points[distance <= threshold]
    if len(keep) <= max_points:
        return keep
    rng = np.random.default_rng(seed)
    return keep[rng.choice(len(keep), size=max_points, replace=False)]


def shared_transform(points: np.ndarray):
    centre = np.median(points, axis=0)
    centred = points - centre
    covariance = np.cov(centred.T)
    _, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, ::-1]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return centre, basis


def transform(points: np.ndarray, centre: np.ndarray, basis: np.ndarray):
    return (points - centre) @ basis


def set_equal_axes(ax, points: np.ndarray):
    lower = np.quantile(points, 0.01, axis=0)
    upper = np.quantile(points, 0.99, axis=0)
    centre = (lower + upper) / 2
    radius = max(float(np.max(upper - lower)) / 2, 1e-6)
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect((1, 1, 0.72))
    ax.view_init(elev=24, azim=-52)
    ax.set_axis_off()


def render_point_cloud(
    ax,
    background_points: np.ndarray,
    background_colours: np.ndarray,
    foreground_points: np.ndarray | None,
    foreground_labels: np.ndarray | None,
    centre: np.ndarray,
    basis: np.ndarray,
    colour_index: dict[int, int] | None = None,
):
    bg = transform(background_points, centre, basis)
    ax.scatter(
        bg[:, 0],
        bg[:, 1],
        bg[:, 2],
        c=background_colours / 255.0,
        s=0.45,
        alpha=0.34 if foreground_points is not None else 0.78,
        linewidths=0,
        rasterized=True,
    )
    if foreground_points is not None and foreground_labels is not None:
        fg = transform(foreground_points, centre, basis)
        ax.scatter(
            fg[:, 0],
            fg[:, 1],
            fg[:, 2],
            c=label_colours(foreground_labels, colour_index),
            s=2.2,
            alpha=0.96,
            linewidths=0,
            rasterized=True,
        )
    set_equal_axes(ax, bg)


def render_scene(
    scene: str,
    config: dict,
    predictions: dict[tuple[str, str], Path],
    output_path: Path,
) -> dict:
    import matplotlib.pyplot as plt
    from PIL import Image

    global_path = (
        Path(config["global_cache_root"]) / scene / "global_cache_e14.npz"
    )
    with np.load(global_path, allow_pickle=False) as global_npz:
        global_data = {key: global_npz[key] for key in global_npz.files}
    with np.load(predictions[(scene, "B16")], allow_pickle=False) as b_npz:
        b = {key: b_npz[key] for key in b_npz.files}
    with np.load(predictions[(scene, "M16")], allow_pickle=False) as m_npz:
        m = {key: m_npz[key] for key in m_npz.files}

    if not np.array_equal(b["global_point_indices"], m["global_point_indices"]):
        raise ValueError(f"{scene}: B/M frozen support differs")
    display_map = matched_m_labels(b["plane_ids"], m["plane_ids"])
    view_index = choose_view(b)
    relative_path = str(global_data["relative_image_paths"][view_index])
    rgb_path = Path(config["raw_rgb_root"]) / scene / relative_path
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))

    b_view = b["view_indices"] == view_index
    m_view = m["view_indices"] == view_index
    b_overlay = make_overlay(rgb, b["pixel_xy"][b_view], b["plane_ids"][b_view])
    m_overlay = make_overlay(
        rgb,
        m["pixel_xy"][m_view],
        m["plane_ids"][m_view],
        display_map,
    )

    rng = np.random.default_rng(int(scene[-5:]))
    valid = np.all(np.isfinite(global_data["points"]), axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) > 55000:
        indices = rng.choice(indices, size=55000, replace=False)
    bg_points = global_data["points"][indices]
    bg_colours = global_data["colors"][indices]
    clipped_bg = robust_points(bg_points, max_points=48000, seed=int(scene[-5:]))
    # Retain colours for the robust subset by applying the same geometric mask.
    centre0 = np.median(bg_points, axis=0)
    distance = np.linalg.norm(bg_points - centre0, axis=1)
    threshold = np.quantile(distance, 0.985)
    mask = distance <= threshold
    bg_points = bg_points[mask]
    bg_colours = bg_colours[mask]
    if len(bg_points) > len(clipped_bg):
        keep = rng.choice(len(bg_points), size=len(clipped_bg), replace=False)
        bg_points = bg_points[keep]
        bg_colours = bg_colours[keep]

    centre, basis = shared_transform(bg_points)

    fig = plt.figure(figsize=(16, 9.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(0.78, 1.22))
    axes_2d = [fig.add_subplot(grid[0, i]) for i in range(3)]
    axes_3d = [
        fig.add_subplot(grid[1, i], projection="3d", computed_zorder=False)
        for i in range(3)
    ]

    axes_2d[0].imshow(rgb)
    axes_2d[1].imshow(b_overlay)
    axes_2d[2].imshow(m_overlay)
    top_titles = (
        f"Input RGB · selected view {view_index}",
        f"B16 image support · {len(b['plane_normals'])} planes",
        f"M16 image support · {len(m['plane_normals'])} planes",
    )
    for ax, title in zip(axes_2d, top_titles):
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_axis_off()

    render_point_cloud(
        axes_3d[0],
        bg_points,
        bg_colours,
        None,
        None,
        centre,
        basis,
    )
    render_point_cloud(
        axes_3d[1],
        bg_points,
        bg_colours,
        b["points"],
        b["plane_ids"],
        centre,
        basis,
    )
    render_point_cloud(
        axes_3d[2],
        bg_points,
        bg_colours,
        m["points"],
        m["plane_ids"],
        centre,
        basis,
        display_map,
    )
    bottom_titles = (
        "DUSt3R globally aligned RGB point cloud",
        "B16 final 3D plane identities",
        "M16 final 3D plane identities",
    )
    for ax, title in zip(axes_3d, bottom_titles):
        ax.set_title(title, fontsize=13, pad=-2)

    fig.suptitle(
        f"{scene} · actual frozen reconstruction output",
        fontsize=17,
        fontweight="normal",
    )
    fig.text(
        0.5,
        0.008,
        "Coloured points are the method's actual sparse support; translucent RGB points show the full aligned scene.",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "scene_id": scene,
        "selected_view": view_index,
        "rgb_path": str(rgb_path),
        "b_plane_count": int(len(b["plane_normals"])),
        "m_plane_count": int(len(m["plane_normals"])),
        "support_point_count": int(len(b["points"])),
        "output_path": str(output_path),
        "sha256": sha256_file(output_path),
    }


def render_b16_gallery(
    scenes: list[str],
    config: dict,
    predictions: dict[tuple[str, str], Path],
    output_path: Path,
) -> dict:
    """Render the default method as input, image support, and final 3D output."""
    import matplotlib.pyplot as plt
    from PIL import Image

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    grid = fig.add_gridspec(len(scenes), 3, width_ratios=(1, 1, 1.2))
    records = []

    for row_index, scene in enumerate(scenes):
        global_path = (
            Path(config["global_cache_root"]) / scene / "global_cache_e14.npz"
        )
        with np.load(global_path, allow_pickle=False) as global_npz:
            global_data = {key: global_npz[key] for key in global_npz.files}
        with np.load(predictions[(scene, "B16")], allow_pickle=False) as b_npz:
            b = {key: b_npz[key] for key in b_npz.files}

        view_index = choose_view(b)
        relative_path = str(global_data["relative_image_paths"][view_index])
        rgb_path = Path(config["raw_rgb_root"]) / scene / relative_path
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        b_view = b["view_indices"] == view_index
        overlay = make_overlay(
            rgb,
            b["pixel_xy"][b_view],
            b["plane_ids"][b_view],
        )

        ax_rgb = fig.add_subplot(grid[row_index, 0])
        ax_overlay = fig.add_subplot(grid[row_index, 1])
        ax_3d = fig.add_subplot(
            grid[row_index, 2], projection="3d", computed_zorder=False
        )
        ax_rgb.imshow(rgb)
        ax_overlay.imshow(overlay)
        ax_rgb.set_axis_off()
        ax_overlay.set_axis_off()

        seed = int(scene[-5:])
        rng = np.random.default_rng(seed)
        valid = np.all(np.isfinite(global_data["points"]), axis=1)
        indices = np.flatnonzero(valid)
        if len(indices) > 50000:
            indices = rng.choice(indices, size=50000, replace=False)
        bg_points = global_data["points"][indices]
        bg_colours = global_data["colors"][indices]
        centre0 = np.median(bg_points, axis=0)
        distance = np.linalg.norm(bg_points - centre0, axis=1)
        mask = distance <= np.quantile(distance, 0.985)
        bg_points = bg_points[mask]
        bg_colours = bg_colours[mask]
        centre, basis = shared_transform(bg_points)
        render_point_cloud(
            ax_3d,
            bg_points,
            bg_colours,
            b["points"],
            b["plane_ids"],
            centre,
            basis,
        )

        if row_index == 0:
            ax_rgb.set_title("Selected input RGB", fontsize=14, pad=8)
            ax_overlay.set_title("B16 plane identities in the image", fontsize=14, pad=8)
            ax_3d.set_title("B16 final 3D plane reconstruction", fontsize=14, pad=-2)
        ax_rgb.text(
            0.02,
            0.96,
            scene,
            transform=ax_rgb.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.58, "pad": 4, "edgecolor": "none"},
        )
        ax_overlay.text(
            0.02,
            0.96,
            f"{len(b['plane_normals'])} recovered identities",
            transform=ax_overlay.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.58, "pad": 4, "edgecolor": "none"},
        )
        records.append(
            {
                "scene_id": scene,
                "selected_view": view_index,
                "rgb_path": str(rgb_path),
                "plane_count": int(len(b["plane_normals"])),
                "support_point_count": int(len(b["points"])),
            }
        )

    fig.suptitle(
        "LightRecon3D final default output · B16",
        fontsize=18,
        fontweight="normal",
    )
    fig.text(
        0.5,
        0.006,
        "The coloured samples are actual recovered plane identities on frozen direct support; the faint RGB cloud is the aligned scene context.",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "output_path": str(output_path),
        "sha256": sha256_file(output_path),
        "scenes": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    args = parser.parse_args()
    config = load_config()
    predictions = load_prediction_manifest()
    records = []
    for scene in args.scenes:
        records.append(
            render_scene(
                scene,
                config,
                predictions,
                FIGURE_ROOT / f"{scene}_actual_reconstruction.png",
            )
        )
    b16_gallery = render_b16_gallery(
        list(args.scenes),
        config,
        predictions,
        FIGURE_ROOT / "b16_final_method_gallery.png",
    )
    manifest = {
        "registration_id": "E16_PREREG_targeted_GT_sidecar_and_final_BM_20260730_v1",
        "reporting_only": True,
        "scientific_outputs_modified": False,
        "description": "Actual frozen RGB, global point cloud, and B16/M16 reconstruction outputs.",
        "scenes": records,
        "b16_final_method_gallery": b16_gallery,
    }
    manifest_path = FIGURE_ROOT / "actual_reconstruction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
