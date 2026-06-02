import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")

if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

if DUST3R_REPO_ROOT in sys.path:
    sys.path.remove(DUST3R_REPO_ROOT)
sys.path.insert(1, DUST3R_REPO_ROOT)

from dataloaders.s3d_dataset import Structured3DDataset
from evaluate_plane_params_head import geometry_descriptor
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel
from train import build_views_from_batch
from visualize_pointmap_compare import tensor_img_to_uint8


def safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def resize_label(gt_plane, target_hw):
    if gt_plane.shape[-2:] == target_hw:
        return gt_plane.long()
    return F.interpolate(gt_plane.unsqueeze(1).float(), size=target_hw, mode="nearest")[:, 0].long()


def get_pts3d_from_res(res):
    for key in ["pts3d", "pts3d_in_other_view", "pointmap", "pred_pts3d"]:
        if key in res:
            pts = res[key]
            break
    else:
        raise KeyError(f"Cannot find pointmap keys in {list(res.keys())}")
    if pts.shape[-1] == 3:
        return pts
    if pts.shape[1] == 3:
        return pts.permute(0, 2, 3, 1).contiguous()
    raise ValueError(f"Cannot interpret pointmap shape {pts.shape}")


def clean_points(points):
    ok = torch.isfinite(points).all(dim=1)
    points = points[ok]
    ok = points.abs().amax(dim=1) < 1e4
    return points[ok]


def plane_basis(normal):
    normal = normal / np.linalg.norm(normal)
    helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = np.cross(normal, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    return u, v


def make_plane_mesh(points_np, normal_np, offset, move_delta=0.0, grid=12):
    normal_np = normal_np / np.linalg.norm(normal_np)
    centroid = points_np.mean(axis=0)
    # For n*x+d=0, increasing d by delta moves the plane by -delta*n.
    center = centroid - (float(np.dot(normal_np, centroid)) + offset) * normal_np
    center = center - move_delta * normal_np
    u, v = plane_basis(normal_np)
    coords_u = (points_np - centroid) @ u
    coords_v = (points_np - centroid) @ v
    lo_u, hi_u = np.percentile(coords_u, [2, 98])
    lo_v, hi_v = np.percentile(coords_v, [2, 98])
    if hi_u - lo_u < 1e-3:
        lo_u, hi_u = -0.5, 0.5
    if hi_v - lo_v < 1e-3:
        lo_v, hi_v = -0.5, 0.5
    us = np.linspace(lo_u, hi_u, grid)
    vs = np.linspace(lo_v, hi_v, grid)
    verts = []
    for vv in vs:
        for uu in us:
            verts.append(center + uu * u + vv * v)
    verts = np.asarray(verts, dtype=np.float32)
    faces = []
    for y in range(grid - 1):
        for x in range(grid - 1):
            a = y * grid + x
            b = a + 1
            c = a + grid
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    return verts, faces


def write_ply(path, point_vertices, point_colors, mesh_entries):
    vertices = []
    colors = []
    faces = []
    for p, c in zip(point_vertices, point_colors):
        vertices.append(p)
        colors.append(c)
    for verts, local_faces, color in mesh_entries:
        base = len(vertices)
        for v in verts:
            vertices.append(v)
            colors.append(color)
        for f in local_faces:
            faces.append(tuple(base + idx for idx in f))

    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(vertices, colors):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")


def set_axes_equal(ax, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float((maxs - mins).max() * 0.55)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_views(out_path, points, colors, mesh_entries, title):
    fig = plt.figure(figsize=(14, 10))
    views = [(20, -70, "view A"), (20, 25, "view B"), (65, -60, "top-ish"), (5, -90, "side")]
    all_mesh_verts = [m[0] for m in mesh_entries]
    all_xyz = np.concatenate([points] + all_mesh_verts, axis=0)
    stride = max(1, len(points) // 7000)
    p = points[::stride]
    c = colors[::stride] / 255.0
    for i, (elev, azim, name) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, s=0.5, alpha=0.45, depthshade=False)
        for verts, faces, color in mesh_entries:
            tris = verts[np.asarray(faces)]
            ax.plot_trisurf(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                triangles=np.asarray(faces),
                color=np.asarray(color) / 255.0,
                alpha=0.32,
                linewidth=0.2,
                edgecolor="k",
            )
        set_axes_equal(ax, all_xyz)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--sample_idx", type=int, required=True)
    parser.add_argument(
        "--param_head_type",
        default="geom_token_conf",
        choices=["geom_token", "geom_token_conf", "geom_token_point_anchor", "geom_token_point_anchor_conf"],
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--plane_offset_scale", type=float, default=1000.0)
    parser.add_argument("--min_pixels", type=int, default=64)
    parser.add_argument("--max_points", type=int, default=25000)
    parser.add_argument("--offset_delta", type=float, default=0.35)
    parser.add_argument("--offset_mode", default="predicted", choices=["predicted", "point_anchor"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Structured3DDataset(
        root_dir=args.root_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        plane_offset_scale=args.plane_offset_scale,
    )
    sample = dataset[args.sample_idx]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}

    backbone = build_dust3r_backbone(args.weights_path, device=device)
    model = LightReconModel(backbone).to(device)
    state = safe_load(args.ckpt_path, device).get("model")
    model.load_state_dict(state, strict=False)
    model.eval()

    view1, view2 = build_views_from_batch(batch, prefix=f"editable_{args.sample_idx}")
    res1, _ = model(view1, view2)

    pts3d = get_pts3d_from_res(res1)[0]
    h_pts, w_pts = pts3d.shape[:2]
    gt_plane_pts = resize_label(batch["gt_plane"], (h_pts, w_pts))[0]

    img_uint8 = tensor_img_to_uint8(batch["img"])
    img_t = torch.from_numpy(img_uint8).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_resized = F.interpolate(img_t, size=(h_pts, w_pts), mode="bilinear", align_corners=False)[0]
    img_colors = img_resized.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()

    feat = res1["dec_feature_map"][0]
    h_feat, w_feat = feat.shape[-2:]
    feat_hw = feat.permute(1, 2, 0).contiguous()
    gt_plane_feat = resize_label(batch["gt_plane"], (h_feat, w_feat))[0]

    flat_pts = pts3d.reshape(-1, 3)
    flat_rgb = img_colors.reshape(-1, 3)
    ok = torch.isfinite(flat_pts).all(dim=1) & (flat_pts.abs().amax(dim=1) < 1e4)
    ok_idx = torch.nonzero(ok).flatten()
    if len(ok_idx) > args.max_points:
        keep = torch.linspace(0, len(ok_idx) - 1, steps=args.max_points, device=ok_idx.device).long()
        ok_idx = ok_idx[keep]
    points_np = flat_pts[ok_idx].detach().cpu().numpy()
    colors_np = flat_rgb[ok_idx.detach().cpu().numpy()]

    plane_rows = []
    mesh_entries = []
    plane_colors = [
        np.array([30, 144, 255], dtype=np.uint8),
        np.array([34, 197, 94], dtype=np.uint8),
        np.array([168, 85, 247], dtype=np.uint8),
        np.array([20, 184, 166], dtype=np.uint8),
    ]
    moved_color = np.array([239, 68, 68], dtype=np.uint8)

    for pi, pid in enumerate(torch.unique(gt_plane_feat)):
        pid_int = int(pid.item())
        if pid_int in (-1, 0, 255):
            continue
        mask_feat = gt_plane_feat == pid
        if int(mask_feat.sum()) < args.min_pixels:
            continue
        mask_pts = gt_plane_pts == pid
        plane_points = clean_points(pts3d[mask_pts])
        if plane_points.shape[0] < 32:
            continue
        token = feat_hw[mask_feat].mean(dim=0, keepdim=True)
        geom = geometry_descriptor(plane_points).to(dtype=token.dtype).view(1, -1)
        if args.param_head_type in ("geom_token_conf", "geom_token_point_anchor_conf"):
            pred = model.geom_plane_token_conf_head(torch.cat([token, geom], dim=1))[0]
            conf = float(torch.sigmoid(pred[4]).detach().cpu())
        else:
            pred = model.geom_plane_token_param_head(torch.cat([token, geom], dim=1))[0]
            conf = -1.0
        plane_np = plane_points.detach().cpu().numpy()
        normal = normalize(pred[:3], dim=0).detach().cpu().numpy().astype(np.float32)
        if args.offset_mode == "point_anchor":
            offset = -float(np.dot(normal, plane_np.mean(axis=0)))
        elif args.param_head_type in ("geom_token_point_anchor", "geom_token_point_anchor_conf"):
            offset = -float(np.dot(normal, plane_np.mean(axis=0))) + float(pred[3].detach().cpu())
        else:
            offset = float(pred[3].detach().cpu())
        verts, faces = make_plane_mesh(plane_np, normal, offset, move_delta=0.0)
        moved_verts, moved_faces = make_plane_mesh(plane_np, normal, offset, move_delta=args.offset_delta)
        color = plane_colors[len(mesh_entries) % len(plane_colors)]
        mesh_entries.append((verts, faces, color))
        mesh_entries.append((moved_verts, moved_faces, moved_color))
        plane_rows.append(
            {
                "plane_id": pid_int,
                "normal": normal,
                "offset": offset,
                "moved_offset": offset + args.offset_delta,
                "move_delta": args.offset_delta,
                "confidence": conf,
                "points": int(plane_points.shape[0]),
            }
        )

    stem = f"{args.split}_{args.sample_idx:06d}_editable_planes_3d"
    ply_path = out_dir / f"{stem}.ply"
    png_path = out_dir / f"{stem}.png"
    txt_path = out_dir / f"{stem}_params.txt"
    write_ply(ply_path, points_np, colors_np, mesh_entries)
    render_views(
        png_path,
        points_np,
        colors_np,
        mesh_entries,
        f"{args.split} sample {args.sample_idx}: predicted planes + moved offset planes (red)",
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Equation: n_x*x + n_y*y + n_z*z + d = 0\n")
        f.write("Red planes in the PLY/PNG use d + offset_delta.\n\n")
        f.write(f"offset_mode={args.offset_mode}\n")
        if args.offset_mode == "point_anchor":
            f.write("point_anchor means d is recomputed as -n dot centroid(points) in DUSt3R pointmap coordinates.\n")
            f.write("This tests editability in the actual reconstructed 3D coordinate frame.\n\n")
        for row in plane_rows:
            n = row["normal"]
            f.write(
                f"plane_id={row['plane_id']} points={row['points']} confidence={row['confidence']:.4f}\n"
                f"  normal=({n[0]:.6f}, {n[1]:.6f}, {n[2]:.6f})\n"
                f"  offset={row['offset']:.6f}\n"
                f"  moved_offset={row['moved_offset']:.6f}  delta={row['move_delta']:.6f}\n\n"
            )
    print(ply_path)
    print(png_path)
    print(txt_path)


if __name__ == "__main__":
    main()
