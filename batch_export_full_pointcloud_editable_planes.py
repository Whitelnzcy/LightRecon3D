import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dataloaders.s3d_dataset import Structured3DDataset
from export_full_pointcloud_editable_planes import (
    PLANE_COLORS,
    clean_points,
    get_conf_from_res,
    get_pts3d_from_res,
    make_html,
    make_plane_mesh,
    resize_rgb,
    sequential_ransac,
    tensor_img_to_uint8,
    write_ply,
)
from models.lightrecon_net import LightReconModel
from train import build_dust3r_backbone, build_views_from_batch


def export_one(args, dataset, model, device, sample_idx):
    out_dir = Path(args.output_dir)
    npz_path = out_dir / f"{args.split}_{sample_idx:06d}_full_pointcloud_editable_planes_data.npz"
    if args.skip_existing and npz_path.exists():
        return {"sample_idx": sample_idx, "status": "skipped", "npz": str(npz_path)}

    sample = dataset[sample_idx]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}
    view1, view2 = build_views_from_batch(batch, prefix=f"{args.split}_{sample_idx}")
    with torch.no_grad():
        res1, _ = model(view1, view2)
    pts_hw3 = get_pts3d_from_res(res1)[0].detach().float().cpu().numpy()
    conf_hw = get_conf_from_res(res1, pts_hw3.shape[:2])
    rgb = resize_rgb(tensor_img_to_uint8(batch["img"][0].cpu()), pts_hw3.shape[:2])

    points = pts_hw3.reshape(-1, 3)
    colors = rgb.reshape(-1, 3)
    conf = conf_hw.reshape(-1)
    ok = clean_points(points, args.max_abs_coord)
    ok &= np.isfinite(conf)
    points = points[ok]
    colors = colors[ok]
    conf = conf[ok]
    order = np.argsort(conf)[::-1]
    points = points[order]
    colors = colors[order]

    planes = sequential_ransac(
        points,
        max_planes=args.max_planes,
        threshold=args.threshold,
        min_inliers=args.min_inliers,
        iterations=args.iterations,
        max_fit_points=args.max_fit_points,
        seed=args.seed + sample_idx,
    )
    for plane in planes:
        inlier_points = points[plane["mask"]]
        verts, faces, center, u, v, extent = make_plane_mesh(
            inlier_points,
            plane["normal"],
            plane["offset"],
        )
        color = PLANE_COLORS[plane["id"] % len(PLANE_COLORS)]
        plane["color"] = color
        plane["mesh_vertices"] = verts
        plane["mesh_faces"] = faces
        plane["center"] = center
        plane["u"] = u
        plane["v"] = v
        plane["extent"] = extent

    full_point_plane_ids = np.full(len(points), -1, dtype=np.int32)
    if planes:
        dist_stack = np.stack([np.abs(points @ p["normal"] + p["offset"]) for p in planes], axis=1)
        nearest = np.argmin(dist_stack, axis=1)
        nearest_dist = dist_stack[np.arange(len(points)), nearest]
        for i, (plane_idx, dist) in enumerate(zip(nearest, nearest_dist)):
            if dist <= args.threshold:
                full_point_plane_ids[i] = int(planes[int(plane_idx)]["id"])

    full_colors = colors.copy()
    for i, pid in enumerate(full_point_plane_ids):
        if pid >= 0:
            full_colors[i] = np.asarray(planes[int(pid)]["color"], dtype=np.uint8)

    display_n = min(args.max_display_points, len(points))
    display_idx = np.linspace(0, len(points) - 1, display_n).astype(np.int64)
    display_points = points[display_idx]
    display_colors = full_colors[display_idx]
    point_plane_ids = full_point_plane_ids[display_idx]

    json_path = out_dir / f"{args.split}_{sample_idx:06d}_full_pointcloud_plane_params.json"
    txt_path = out_dir / f"{args.split}_{sample_idx:06d}_full_pointcloud_plane_params.txt"
    ply_path = out_dir / f"{args.split}_{sample_idx:06d}_full_pointcloud_editable_planes.ply"
    html_path = out_dir / f"{args.split}_{sample_idx:06d}_full_pointcloud_editable_planes.html"
    if not args.npz_only:
        write_ply(ply_path, display_points, display_colors, planes)

    params = []
    for plane in planes:
        params.append(
            {
                "id": plane["id"],
                "equation": "nx*x + ny*y + nz*z + d = 0",
                "normal": [float(x) for x in plane["normal"]],
                "offset": float(plane["offset"]),
                "inlier_count": int(plane["inlier_count"]),
                "mean_abs_distance": float(plane["mean_abs_distance"]),
                "color": plane["color"],
            }
        )
    json_path.write_text(json.dumps({"sample_idx": sample_idx, "planes": params}, indent=2), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        points=points.astype(np.float32),
        colors=full_colors.astype(np.uint8),
        point_plane_ids=full_point_plane_ids.astype(np.int32),
        plane_ids=np.asarray([p["id"] for p in planes], dtype=np.int32),
        plane_normals=np.asarray([p["normal"] for p in planes], dtype=np.float32),
        plane_offsets=np.asarray([p["offset"] for p in planes], dtype=np.float32),
        plane_inlier_counts=np.asarray([p["inlier_count"] for p in planes], dtype=np.int32),
    )
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Equation: nx*x + ny*y + nz*z + d = 0\n")
        f.write(f"Full point cloud points used for visualization: {len(display_points)} / {len(points)}\n")
        f.write(f"Full editable point cloud data: {npz_path.name}\n")
        f.write(f"Extracted major planes: {len(planes)}\n\n")
        for p in params:
            n = p["normal"]
            f.write(
                f"plane {p['id']}: normal=({n[0]:.6f}, {n[1]:.6f}, {n[2]:.6f}) "
                f"d={p['offset']:.6f} inliers={p['inlier_count']} "
                f"mean_abs_distance={p['mean_abs_distance']:.6f}\n"
            )

    if not args.npz_only:
        all_xyz = np.concatenate([display_points] + [p["mesh_vertices"] for p in planes], axis=0)
        extent = np.maximum(all_xyz.max(axis=0) - all_xyz.min(axis=0), 1e-6)
        scale = float(0.82 / extent.max())
        center = all_xyz.mean(axis=0)
        data = {
            "points": display_points.astype(float).tolist(),
            "colors": [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in display_colors],
            "point_plane_ids": point_plane_ids.astype(int).tolist(),
            "planes": params,
            "center": center.astype(float).tolist(),
            "scale": scale,
            "total_points": int(len(points)),
            "display_points": int(len(display_points)),
        }
        make_html(html_path, data)
    return {
        "sample_idx": sample_idx,
        "status": "ok",
        "planes": len(planes),
        "points": int(len(points)),
        "npz": str(npz_path),
    }


def main():
    parser = argparse.ArgumentParser("Batch export DUSt3R point clouds and RANSAC plane proposals")
    parser.add_argument("--root_dir", default="/data/zhucy23u/datasets/Structured3D")
    parser.add_argument("--weights_path", default="/data/zhucy23u/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_planes", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.035)
    parser.add_argument("--min_inliers", type=int, default=900)
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--max_fit_points", type=int, default=65000)
    parser.add_argument("--max_display_points", type=int, default=12000)
    parser.add_argument("--max_abs_coord", type=float, default=1e4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--npz_only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
    )
    end_idx = min(len(dataset), args.start_idx + args.count)
    backbone = build_dust3r_backbone(args.weights_path, device=device)
    model = LightReconModel(dust3r_backbone=backbone).to(device)
    model.eval()
    manifest = []
    for sample_idx in range(args.start_idx, end_idx):
        try:
            row = export_one(args, dataset, model, device, sample_idx)
        except Exception as exc:
            row = {"sample_idx": sample_idx, "status": "error", "error": repr(exc)}
        manifest.append(row)
        print(json.dumps(row), flush=True)
        (out_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out_dir / "batch_manifest.json")


if __name__ == "__main__":
    main()
