import argparse
import json
import os
import random
import sys
from pathlib import Path

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
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel
from train import build_views_from_batch


PLANE_COLORS = [
    [65, 129, 191],
    [142, 92, 181],
    [46, 158, 120],
    [221, 144, 55],
    [196, 74, 74],
    [76, 163, 190],
    [184, 110, 155],
    [134, 154, 65],
    [90, 105, 190],
    [190, 125, 80],
]


def normalize(x, eps=1e-8):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def tensor_img_to_uint8(img):
    if img.ndim == 4:
        img = img[0]
    img_np = img.detach().float().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    finite = np.isfinite(img_np)
    if not finite.any():
        return np.zeros((*img_np.shape[:2], 3), dtype=np.uint8)
    vmin = float(np.nanmin(img_np))
    vmax = float(np.nanmax(img_np))
    if vmin >= 0.0 and vmax <= 1.5:
        img_np = np.clip(img_np, 0.0, 1.0) * 255.0
    elif vmin >= 0.0 and vmax <= 255.0:
        img_np = np.clip(img_np, 0.0, 255.0)
    else:
        img_np = (img_np - vmin) / (vmax - vmin + 1e-8)
        img_np = np.clip(img_np, 0.0, 1.0) * 255.0
    return img_np.astype(np.uint8)


def resize_rgb(rgb, target_hw):
    if rgb.shape[:2] == target_hw:
        return rgb
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()


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


def get_conf_from_res(res, shape_hw):
    for key in ["conf", "confidence", "pred_conf"]:
        if key in res:
            conf = res[key]
            if conf.ndim == 4:
                if conf.shape[1] == 1:
                    conf = conf[:, 0]
                else:
                    conf = conf[..., 0]
            return conf[0].detach().float().cpu().numpy()
    return np.ones(shape_hw, dtype=np.float32)


def clean_points(points, max_abs_coord=1e4):
    ok = np.isfinite(points).all(axis=1)
    ok &= np.max(np.abs(points), axis=1) < max_abs_coord
    return ok


def fit_plane_svd(points):
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    x = points - centroid
    cov = x.T @ x / max(1, len(points) - 1)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    normal = normalize(vecs[:, 0])
    offset = -float(np.dot(normal, centroid))
    return normal.astype(np.float32), offset, centroid.astype(np.float32)


def fit_plane_3pts(points):
    a, b, c = points
    normal = np.cross(b - a, c - a)
    norm = np.linalg.norm(normal)
    if norm < 1e-7:
        return None
    normal = normal / norm
    offset = -float(np.dot(normal, a))
    return normal.astype(np.float32), offset


def align_plane(normal, offset):
    idx = int(np.argmax(np.abs(normal)))
    if normal[idx] < 0:
        normal = -normal
        offset = -offset
    return normal.astype(np.float32), float(offset)


def ransac_one(points, iterations, threshold, seed):
    rng = np.random.default_rng(seed)
    if len(points) < 3:
        return None
    best = None
    sample_n = len(points)
    for _ in range(iterations):
        ids = rng.choice(sample_n, size=3, replace=False)
        fit = fit_plane_3pts(points[ids])
        if fit is None:
            continue
        normal, offset = fit
        dist = np.abs(points @ normal + offset)
        inliers = dist <= threshold
        count = int(inliers.sum())
        if count < 3:
            continue
        mean_dist = float(dist[inliers].mean())
        score = (count, -mean_dist)
        if best is None or score > best[0]:
            best = (score, normal, offset, inliers)
    if best is None:
        return None
    _, normal, offset, inliers = best
    refined = fit_plane_svd(points[inliers])
    if refined is not None:
        normal, offset, centroid = refined
    else:
        centroid = points[inliers].mean(axis=0)
    normal, offset = align_plane(normal, offset)
    dist = np.abs(points @ normal + offset)
    inliers = dist <= threshold
    return normal, offset, inliers, centroid, float(dist[inliers].mean())


def sequential_ransac(points, max_planes, threshold, min_inliers, iterations, max_fit_points, seed):
    rng = np.random.default_rng(seed)
    if len(points) > max_fit_points:
        fit_ids = rng.choice(len(points), size=max_fit_points, replace=False)
    else:
        fit_ids = np.arange(len(points))
    fit_points = points[fit_ids]
    remaining = np.ones(len(fit_points), dtype=bool)
    planes = []
    for plane_id in range(max_planes):
        current = fit_points[remaining]
        if len(current) < min_inliers:
            break
        result = ransac_one(
            current,
            iterations=iterations,
            threshold=threshold,
            seed=seed + plane_id * 997,
        )
        if result is None:
            break
        normal, offset, inliers_local, centroid, mean_dist = result
        inlier_count = int(inliers_local.sum())
        if inlier_count < min_inliers:
            break
        dist_full = np.abs(points @ normal + offset)
        inliers_full = dist_full <= threshold
        full_count = int(inliers_full.sum())
        plane = {
            "id": plane_id,
            "normal": normal,
            "offset": float(offset),
            "centroid": points[inliers_full].mean(axis=0).astype(np.float32),
            "inlier_count": full_count,
            "fit_inlier_count": inlier_count,
            "mean_abs_distance": float(dist_full[inliers_full].mean()),
            "mask": inliers_full,
        }
        planes.append(plane)
        idx_remaining = np.flatnonzero(remaining)
        idx_remove = idx_remaining[inliers_local]
        remaining[idx_remove] = False
    return planes


def plane_basis(normal):
    normal = np.asarray(normal, dtype=np.float32)
    normal = normal / max(np.linalg.norm(normal), 1e-8)
    helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = np.cross(normal, helper)
    u = u / max(np.linalg.norm(u), 1e-8)
    v = np.cross(normal, u)
    v = v / max(np.linalg.norm(v), 1e-8)
    return u.astype(np.float32), v.astype(np.float32)


def make_plane_mesh(points, normal, offset, grid=8):
    normal = np.asarray(normal, dtype=np.float32)
    centroid = points.mean(axis=0).astype(np.float32)
    center = centroid - (float(np.dot(normal, centroid)) + float(offset)) * normal
    u, v = plane_basis(normal)
    coords_u = (points - centroid) @ u
    coords_v = (points - centroid) @ v
    lo_u, hi_u = np.percentile(coords_u, [3, 97])
    lo_v, hi_v = np.percentile(coords_v, [3, 97])
    if hi_u - lo_u < 1e-3:
        lo_u, hi_u = -0.2, 0.2
    if hi_v - lo_v < 1e-3:
        lo_v, hi_v = -0.2, 0.2
    verts = []
    for vv in np.linspace(lo_v, hi_v, grid):
        for uu in np.linspace(lo_u, hi_u, grid):
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
    return verts, faces, center, u, v, [float(lo_u), float(hi_u), float(lo_v), float(hi_v)]


def write_ply(path, points, colors, planes):
    vertices = []
    vertex_colors = []
    faces = []
    for p, c in zip(points, colors):
        vertices.append(p)
        vertex_colors.append(c)
    for plane in planes:
        base = len(vertices)
        color = np.asarray(plane["color"], dtype=np.uint8)
        for v in plane["mesh_vertices"]:
            vertices.append(v)
            vertex_colors.append(color)
        for f in plane["mesh_faces"]:
            faces.append(tuple(base + idx for idx in f))
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(vertices, vertex_colors):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")


def make_html(path, data):
    html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>Full Point Cloud Editable Planes</title>
<style>
body { margin:0; font-family: Arial, sans-serif; background:#f6f6f4; color:#202124; }
.bar { display:flex; align-items:center; gap:14px; flex-wrap:wrap; min-height:60px; padding:10px 14px; background:white; border-bottom:1px solid #d8d8d8; box-sizing:border-box; }
.bar strong { margin-right:10px; }
.ctrl { display:flex; align-items:center; gap:6px; font-size:13px; padding:3px 6px; border:1px solid #ddd; background:#fafafa; }
.ctrl input { width:118px; }
canvas { display:block; width:100vw; height:calc(100vh - 82px); }
</style>
</head>
<body>
<div class="bar" id="bar">
  <strong>Full point cloud: editable major planes</strong>
</div>
<canvas id="c"></canvas>
<script>
const DATA = __DATA__;
const bar = document.getElementById('bar');
for (const pl of DATA.planes) {
  const d = document.createElement('span');
  d.className = 'ctrl';
  d.innerHTML = `plane ${pl.id} <input id="delta_${pl.id}" type="range" min="-0.45" max="0.45" step="0.01" value="0"> <span id="txt_${pl.id}">0.00</span>`;
  bar.appendChild(d);
}
const hint = document.createElement('span');
hint.className = 'ctrl';
hint.textContent = 'drag canvas to rotate; each slider changes plane offset d';
bar.appendChild(hint);
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let yaw = -0.75, pitch = 0.58, dragging=false, lastX=0, lastY=0;
function resize(){ canvas.width=innerWidth*devicePixelRatio; canvas.height=(innerHeight-bar.offsetHeight)*devicePixelRatio; draw(); }
addEventListener('resize', resize);
canvas.addEventListener('mousedown', e=>{ dragging=true; lastX=e.clientX; lastY=e.clientY; });
addEventListener('mouseup', ()=>dragging=false);
addEventListener('mousemove', e=>{ if(!dragging)return; yaw+=(e.clientX-lastX)*0.008; pitch+=(e.clientY-lastY)*0.008; pitch=Math.max(-1.45,Math.min(1.45,pitch)); lastX=e.clientX; lastY=e.clientY; draw(); });
for (const pl of DATA.planes) document.getElementById(`delta_${pl.id}`).addEventListener('input', draw);
function rot(p){ let [x,y,z]=p; let cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch); let x1=cy*x+sy*z, z1=-sy*x+cy*z; let y1=cp*y-sp*z1, z2=sp*y+cp*z1; return [x1,y1,z2]; }
function project(p){ const r=rot(p); const s=Math.min(canvas.width,canvas.height)*DATA.scale; return [canvas.width/2+r[0]*s, canvas.height/2-r[1]*s, r[2]]; }
function planeVerts(pl, moved){ const delta=moved ? Number(document.getElementById(`delta_${pl.id}`).value) : 0; const n=pl.normal,u=pl.u,v=pl.v,e=pl.extent; const center=pl.center.map((x,i)=>x-delta*n[i]); return [[e[0],e[2]],[e[1],e[2]],[e[1],e[3]],[e[0],e[3]]].map(([a,b])=>center.map((x,i)=>x+a*u[i]+b*v[i])); }
function editedPoint(i){
  const p = DATA.points[i];
  const pid = DATA.point_plane_ids[i];
  if (pid < 0) return p;
  const pl = DATA.planeById[pid];
  if (!pl) return p;
  const delta = Number(document.getElementById(`delta_${pid}`).value);
  return p.map((x,j)=>x-delta*pl.normal[j]);
}
function drawPoly(verts, rgba){ const pts=verts.map(project); ctx.beginPath(); ctx.moveTo(pts[0][0],pts[0][1]); for(let i=1;i<pts.length;i++)ctx.lineTo(pts[i][0],pts[i][1]); ctx.closePath(); ctx.fillStyle=rgba; ctx.strokeStyle='rgba(0,0,0,0.35)'; ctx.lineWidth=1.1*devicePixelRatio; ctx.fill(); ctx.stroke(); }
function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle='#f6f6f4'; ctx.fillRect(0,0,canvas.width,canvas.height);
  for (const pl of DATA.planes) document.getElementById(`txt_${pl.id}`).textContent = Number(document.getElementById(`delta_${pl.id}`).value).toFixed(2);
  const items=[];
  for(let i=0;i<DATA.points.length;i++){ const pr=project(editedPoint(i)); items.push(['pt',pr[2],pr,DATA.colors[i],DATA.point_plane_ids[i]]); }
  items.sort((a,b)=>a[1]-b[1]);
  for(const it of items){ const c=it[3]; const alpha=it[4] < 0 ? 0.30 : 0.62; ctx.fillStyle=`rgba(${c[0]},${c[1]},${c[2]},${alpha})`; ctx.fillRect(it[2][0],it[2][1],1.45*devicePixelRatio,1.45*devicePixelRatio); }
  for(const pl of DATA.planes){ const c=pl.color; drawPoly(planeVerts(pl,false),`rgba(${c[0]},${c[1]},${c[2]},0.30)`); }
  for(const pl of DATA.planes){ drawPoly(planeVerts(pl,true),'rgba(220,55,45,0.26)'); }
  ctx.fillStyle='#222'; ctx.font=`${13*devicePixelRatio}px Arial`; let y=22*devicePixelRatio;
  for(const pl of DATA.planes){ const delta=Number(document.getElementById(`delta_${pl.id}`).value); ctx.fillText(`plane ${pl.id}: n=(${pl.normal.map(x=>x.toFixed(3)).join(', ')}) d=${pl.offset.toFixed(3)} -> ${(pl.offset+delta).toFixed(3)} pts=${pl.inlier_count}`, 16*devicePixelRatio, y); y+=18*devicePixelRatio; }
}
resize();
</script>
</body>
</html>
"""
    path.write_text(html.replace("__DATA__", json.dumps(data)), encoding="utf-8")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="/data/zhucy23u/datasets/Structured3D")
    parser.add_argument("--weights_path", default="/data/zhucy23u/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--sample_idx", type=int, default=76)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_planes", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.025)
    parser.add_argument("--min_inliers", type=int, default=2500)
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--max_fit_points", type=int, default=50000)
    parser.add_argument("--max_display_points", type=int, default=18000)
    parser.add_argument("--max_abs_coord", type=float, default=1e4)
    parser.add_argument("--seed", type=int, default=17)
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
    sample = dataset[args.sample_idx]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}
    view1, view2 = build_views_from_batch(batch, prefix=f"{args.split}_{args.sample_idx}")

    backbone = build_dust3r_backbone(args.weights_path, device=device)
    model = LightReconModel(dust3r_backbone=backbone).to(device)
    model.eval()
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
        seed=args.seed,
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

    display_n = min(args.max_display_points, len(points))
    display_idx = np.linspace(0, len(points) - 1, display_n).astype(np.int64)
    display_points = points[display_idx]
    display_colors = colors[display_idx]
    point_plane_ids = np.full(len(display_points), -1, dtype=np.int32)
    if planes:
        dist_stack = []
        for plane in planes:
            dist_stack.append(np.abs(display_points @ plane["normal"] + plane["offset"]))
        dist_stack = np.stack(dist_stack, axis=1)
        nearest = np.argmin(dist_stack, axis=1)
        nearest_dist = dist_stack[np.arange(len(display_points)), nearest]
        for i, (plane_idx, dist) in enumerate(zip(nearest, nearest_dist)):
            if dist <= args.threshold:
                point_plane_ids[i] = int(planes[int(plane_idx)]["id"])
                display_colors[i] = np.asarray(planes[int(plane_idx)]["color"], dtype=np.uint8)

    ply_path = out_dir / f"{args.split}_{args.sample_idx:06d}_full_pointcloud_editable_planes.ply"
    json_path = out_dir / f"{args.split}_{args.sample_idx:06d}_full_pointcloud_plane_params.json"
    txt_path = out_dir / f"{args.split}_{args.sample_idx:06d}_full_pointcloud_plane_params.txt"
    html_path = out_dir / f"{args.split}_{args.sample_idx:06d}_full_pointcloud_editable_planes.html"
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
    json_path.write_text(json.dumps({"sample_idx": args.sample_idx, "planes": params}, indent=2), encoding="utf-8")
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Equation: nx*x + ny*y + nz*z + d = 0\n")
        f.write(f"Full point cloud points used for visualization: {len(display_points)} / {len(points)}\n")
        f.write(f"Extracted major planes: {len(planes)}\n\n")
        for p in params:
            n = p["normal"]
            f.write(
                f"plane {p['id']}: normal=({n[0]:.6f}, {n[1]:.6f}, {n[2]:.6f}) "
                f"d={p['offset']:.6f} inliers={p['inlier_count']} "
                f"mean_abs_distance={p['mean_abs_distance']:.6f}\n"
            )

    all_xyz = np.concatenate([display_points] + [p["mesh_vertices"] for p in planes], axis=0)
    extent = np.maximum(all_xyz.max(axis=0) - all_xyz.min(axis=0), 1e-6)
    scale = float(0.82 / extent.max())
    html_data = {
        "points": display_points.round(5).tolist(),
        "colors": display_colors.astype(int).tolist(),
        "point_plane_ids": point_plane_ids.astype(int).tolist(),
        "scale": scale,
        "planes": [
            {
                "id": int(p["id"]),
                "normal": p["normal"].round(6).tolist(),
                "offset": float(p["offset"]),
                "inlier_count": int(p["inlier_count"]),
                "color": p["color"],
                "center": p["center"].round(6).tolist(),
                "u": p["u"].round(6).tolist(),
                "v": p["v"].round(6).tolist(),
                "extent": p["extent"],
            }
            for p in planes
        ],
    }
    html_data["planeById"] = {str(p["id"]): p for p in html_data["planes"]}
    make_html(html_path, html_data)
    print(ply_path)
    print(json_path)
    print(txt_path)
    print(html_path)
    print(f"planes={len(planes)} points={len(points)} display_points={len(display_points)}")


if __name__ == "__main__":
    main()
