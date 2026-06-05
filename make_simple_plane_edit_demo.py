import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "outputs" / "weekly_report_2026-06-05" / "assets"
DATA_DIR = ROOT / "outputs" / "plane_proposal_refinement_train3997_v6_soft_assignment_refit"


PALETTE = [
    (239, 68, 68),
    (37, 99, 235),
    (34, 197, 94),
    (249, 115, 22),
    (168, 85, 247),
    (20, 184, 166),
]


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(30, True)
F_H = font(22, True)
F_BODY = font(17)
F_SMALL = font(14)
F_MONO = font(13)


def project(points, width, height, yaw=-0.55, pitch=0.28):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = 0.80 * min(width, height) / max(float(np.max(maxs - mins)), 1e-6)
    p = points - center
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x1 = cy * p[:, 0] + sy * p[:, 2]
    z1 = -sy * p[:, 0] + cy * p[:, 2]
    y1 = cp * p[:, 1] - sp * z1
    z2 = sp * p[:, 1] + cp * z1
    return np.stack([width / 2 + x1 * scale, height / 2 - y1 * scale, z2], axis=1)


def sample_indices(point_ids, max_points=42000):
    idxs = []
    valid = [pid for pid in np.unique(point_ids) if pid >= 0]
    per_plane = max_points // max(len(valid), 1)
    for pid in valid:
        ids = np.flatnonzero(point_ids == pid)
        stride = max(1, len(ids) // max(per_plane, 1))
        idxs.extend(ids[::stride][:per_plane].tolist())
    return np.asarray(idxs[:max_points], dtype=np.int64)


def draw_cloud(draw, panel, points, point_ids, plane_ids, title, edit_plane=None, normal=None, delta=0.22):
    x1, y1, x2, y2 = panel
    draw.rounded_rectangle(panel, radius=16, fill="#ffffff", outline="#dbe1ea")
    draw.text((x1 + 18, y1 + 14), title, fill="#172033", font=F_H)
    px, py = x1 + 18, y1 + 58
    pw, ph = x2 - x1 - 36, y2 - y1 - 86
    pts = points.copy()
    moved = np.zeros(len(points), dtype=bool)
    if edit_plane is not None and normal is not None:
        moved = point_ids == edit_plane
        pts[moved] = pts[moved] + delta * normal[None, :]
    idx = sample_indices(point_ids)
    proj = project(pts[idx], pw, ph)
    order = np.argsort(proj[:, 2])
    for oi in order:
        gi = idx[oi]
        sx, sy = int(px + proj[oi, 0]), int(py + proj[oi, 1])
        if not (px <= sx < px + pw and py <= sy < py + ph):
            continue
        pid = int(point_ids[gi])
        if pid < 0:
            color = (160, 160, 160)
            radius = 1
        elif moved[gi]:
            color = (225, 29, 72)
            radius = 2
        else:
            color = PALETTE[pid % len(PALETTE)]
            radius = 2
        draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius), fill=color)
    if edit_plane is not None:
        draw.text((x1 + 22, y2 - 34), f"Edited plane {edit_plane}: offset changed, moved points shown in pink", fill="#e11d48", font=F_SMALL)


def make_demo(input_npz=None, sample="val_000026", edit_plane=None, output_name=None):
    input_npz = Path(input_npz) if input_npz else DATA_DIR / f"{sample}_refined_plane_proposals_data.npz"
    data = np.load(input_npz)
    points = data["points"].astype(np.float32)
    point_ids = data["point_plane_ids"].astype(np.int32)
    plane_ids = data["plane_ids"].astype(np.int32)
    normals = data["plane_normals"].astype(np.float32)
    offsets = data["plane_offsets"].astype(np.float32)
    if edit_plane is None:
        counts = [(int((point_ids == pid).sum()), int(pid)) for pid in plane_ids]
        edit_plane = max(counts)[1]
    n = normals[list(plane_ids).index(edit_plane)]
    d = float(offsets[list(plane_ids).index(edit_plane)])
    moved_count = int((point_ids == edit_plane).sum())

    img = Image.new("RGB", (1600, 980), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    draw.text((46, 34), "最直观结果：一个颜色代表一个面，选中一个面移动", fill="#172033", font=F_TITLE)
    draw.text((46, 80), "这张图只展示当前能讲清楚的部分：平面分组、平面参数、以及通过修改 offset 进行编辑。", fill="#4b5563", font=F_BODY)
    draw_cloud(draw, (45, 130, 775, 745), points, point_ids, plane_ids, "Before: 每个颜色是一类平面 primitive")
    draw_cloud(draw, (825, 130, 1555, 745), points, point_ids, plane_ids, "After: 修改选中平面的 offset", edit_plane=edit_plane, normal=n)

    draw.rounded_rectangle((58, 790, 1542, 930), radius=16, fill="#ffffff", outline="#dbe1ea")
    draw.text((86, 812), "这张图应该怎么讲", fill="#172033", font=F_H)
    notes = [
        f"面的体现：同一种颜色表示绑定到同一个平面 primitive 的点；这里一共有 {len(plane_ids)} 个主要平面。",
        f"参数体现：选中平面 {edit_plane} 的方程是 n*x + d = 0，n=[{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}], d={d:.3f}。",
        f"可编辑体现：右图修改 offset 后，绑定到该平面的 {moved_count:,} 个点整体移动，粉色区域就是被编辑的结构面。",
        "诚实说明：这只是当前最清楚的可编辑机制展示，不代表所有验证样本都已经学得很好。",
    ]
    yy = 850
    for note in notes:
        draw.text((90, yy), note, fill="#4b5563", font=F_BODY)
        yy += 26
    out = ASSETS / (output_name or f"fig7_simple_plane_edit_{sample}.png")
    img.save(out, quality=95)
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create a simple one-color-per-plane edit demo figure.")
    parser.add_argument("--input_npz", default=None)
    parser.add_argument("--sample", default="val_000026")
    parser.add_argument("--edit_plane", type=int, default=None)
    parser.add_argument("--output_name", default=None)
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_demo(args.input_npz, args.sample, args.edit_plane, args.output_name)
