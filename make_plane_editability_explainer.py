from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "outputs" / "weekly_report_2026-06-05" / "assets"
DATA_DIR = ROOT / "outputs" / "plane_proposal_refinement_train3997_v6_soft_assignment_refit"


PALETTE = [
    (37, 99, 235),
    (22, 163, 74),
    (225, 29, 72),
    (245, 158, 11),
    (124, 58, 237),
    (8, 145, 178),
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


def basis_from_normal(n):
    n = n / max(float(np.linalg.norm(n)), 1e-8)
    helper = np.array([0, 0, 1], dtype=np.float32)
    if abs(float(np.dot(helper, n))) > 0.85:
        helper = np.array([0, 1, 0], dtype=np.float32)
    u = np.cross(n, helper)
    u = u / max(float(np.linalg.norm(u)), 1e-8)
    v = np.cross(n, u)
    v = v / max(float(np.linalg.norm(v)), 1e-8)
    return u.astype(np.float32), v.astype(np.float32)


def plane_patch(points, normal, offset, mask, expand=0.08):
    pts = points[mask]
    if len(pts) < 10:
        return None
    u, v = basis_from_normal(normal)
    center = pts.mean(axis=0)
    center = center - normal * (float(np.dot(normal, center)) + float(offset))
    rel = pts - center[None, :]
    a = rel @ u
    b = rel @ v
    amin, amax = np.percentile(a, [2, 98])
    bmin, bmax = np.percentile(b, [2, 98])
    da = (amax - amin) * expand
    db = (bmax - bmin) * expand
    amin, amax = amin - da, amax + da
    bmin, bmax = bmin - db, bmax + db
    corners = np.stack(
        [
            center + amin * u + bmin * v,
            center + amax * u + bmin * v,
            center + amax * u + bmax * v,
            center + amin * u + bmax * v,
        ]
    )
    return corners.astype(np.float32)


def projection(points, width, height, yaw=-0.55, pitch=0.18):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = 0.72 * min(width, height) / max(float(np.max(maxs - mins)), 1e-6)
    p = points - center
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x1 = cy * p[:, 0] + sy * p[:, 2]
    z1 = -sy * p[:, 0] + cy * p[:, 2]
    y1 = cp * p[:, 1] - sp * z1
    z2 = sp * p[:, 1] + cp * z1
    return np.stack([width / 2 + x1 * scale, height / 2 - y1 * scale, z2], axis=1), center, scale


def sample_points(points, point_ids, max_points=36000):
    idxs = []
    for pid in np.unique(point_ids):
        if pid < 0:
            continue
        ids = np.flatnonzero(point_ids == pid)
        keep = min(len(ids), max_points // 8)
        stride = max(1, len(ids) // max(keep, 1))
        idxs.extend(ids[::stride][:keep].tolist())
    if len(idxs) < max_points:
        remaining = max_points - len(idxs)
        stride = max(1, len(points) // remaining)
        idxs.extend(np.arange(0, len(points), stride)[:remaining].tolist())
    return np.unique(np.asarray(idxs, dtype=np.int64))[:max_points]


def draw_scene(base_img, panel, points, colors, point_ids, plane_ids, normals, offsets, title, edit_plane=None, edit_delta=0.0):
    x1, y1, x2, y2 = panel
    d = ImageDraw.Draw(base_img)
    d.rounded_rectangle(panel, radius=16, fill="#ffffff", outline="#dbe1ea")
    d.text((x1 + 18, y1 + 14), title, fill="#172033", font=F_H)
    px, py = x1 + 18, y1 + 54
    pw, ph = x2 - x1 - 36, y2 - y1 - 72

    draw_points = points.copy()
    if edit_plane is not None:
        mask = point_ids == edit_plane
        n = normals[list(plane_ids).index(edit_plane)]
        draw_points[mask] = draw_points[mask] - edit_delta * n[None, :]

    sample_idx = sample_points(draw_points, point_ids)
    proj, _, _ = projection(draw_points[sample_idx], pw, ph)
    order = np.argsort(proj[:, 2])
    local = Image.new("RGBA", (pw, ph), (255, 255, 255, 255))
    ld = ImageDraw.Draw(local, "RGBA")

    all_proj_points = []
    for pid, n, off in zip(plane_ids, normals, offsets):
        mask = point_ids == int(pid)
        patch = plane_patch(draw_points, n, off + (edit_delta if edit_plane == int(pid) else 0.0), mask)
        if patch is None:
            continue
        p2, _, _ = projection(patch, pw, ph)
        all_proj_points.extend([(float(a), float(b)) for a, b, _ in p2])
        color = PALETTE[int(pid) % len(PALETTE)]
        poly = [(float(a), float(b)) for a, b, _ in p2]
        ld.polygon(poly, fill=(*color, 72), outline=(*color, 230))

    for oi in order:
        gi = sample_idx[oi]
        sx, sy = int(proj[oi, 0]), int(proj[oi, 1])
        if not (0 <= sx < pw and 0 <= sy < ph):
            continue
        pid = int(point_ids[gi])
        if pid >= 0:
            c = PALETTE[pid % len(PALETTE)]
            r = 1
            alpha = 180
        else:
            rgb = colors[gi]
            c = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            r = 1
            alpha = 80
        ld.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(*c, alpha))

    if edit_plane is not None:
        n = normals[list(plane_ids).index(edit_plane)]
        mask = point_ids == edit_plane
        pts = points[mask]
        center = pts.mean(axis=0)
        moved_center = center - edit_delta * n
        both = np.stack([center, moved_center])
        p2, _, _ = projection(both, pw, ph)
        a = (int(p2[0, 0]), int(p2[0, 1]))
        b = (int(p2[1, 0]), int(p2[1, 1]))
        ld.line((a, b), fill=(225, 29, 72, 255), width=6)
        ld.ellipse((b[0] - 8, b[1] - 8, b[0] + 8, b[1] + 8), fill=(225, 29, 72, 255))
        ld.text((20, ph - 34), f"Edit: Plane {edit_plane} offset d changes by {edit_delta:+.2f}", fill=(225, 29, 72, 255), font=F_SMALL)

    base_img.paste(local, (px, py))


def make_figure(sample="val_000030", edit_plane=0, edit_delta=0.12):
    data = np.load(DATA_DIR / f"{sample}_refined_plane_proposals_data.npz")
    points = data["points"].astype(np.float32)
    colors = data["colors"].astype(np.uint8)
    point_ids = data["point_plane_ids"].astype(np.int32)
    plane_ids = data["plane_ids"].astype(np.int32)
    normals = data["plane_normals"].astype(np.float32)
    offsets = data["plane_offsets"].astype(np.float32)

    img = Image.new("RGB", (1600, 1030), "#f7f8fb")
    d = ImageDraw.Draw(img)
    d.text((46, 34), "把结果画成“面”和“可编辑操作”", fill="#172033", font=F_TITLE)
    d.text((46, 80), "半透明彩色区域是拟合出的平面 primitive；右图演示修改 offset 后，被绑定到该平面的点整体移动。", fill="#4b5563", font=F_BODY)
    draw_scene(img, (50, 130, 765, 790), points, colors, point_ids, plane_ids, normals, offsets, "A. 学到的平面 primitive")
    draw_scene(
        img,
        (835, 130, 1550, 790),
        points,
        colors,
        point_ids,
        plane_ids,
        normals,
        offsets,
        "B. 可编辑演示：移动一个平面的 offset",
        edit_plane=edit_plane,
        edit_delta=edit_delta,
    )
    d.rounded_rectangle((60, 830, 1540, 985), radius=16, fill="#ffffff", outline="#dbe1ea")
    d.text((86, 852), "这张图想说明什么？", fill="#172033", font=F_H)
    lines = [
        "1. 面的体现：每个半透明彩色矩形就是一个 learned/refined plane primitive，对应一个方程 n*x + d = 0。",
        "2. 点的归属：同色点表示被绑定到该平面的点，说明模型不只是画线，而是在维护 point-to-plane assignment。",
        "3. 可编辑体现：右图修改某个平面的 offset 后，该平面的 support points 会沿法向整体移动，这就是后续编辑墙面/地面的基础。",
        "4. 当前局限：有些平面还不够准，所以这张图适合作为“初步可行性”，不能包装成最终完美结果。",
    ]
    yy = 890
    for line in lines:
        d.text((90, yy), line, fill="#4b5563", font=F_BODY)
        yy += 28
    out = ASSETS / f"fig5_plane_editability_{sample}.png"
    img.save(out, quality=95)
    print(out)


def draw_selected_panel(img, panel, points, colors, point_ids, normal, offset, plane_id, title, mode="before", edit_delta=0.12):
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = panel
    draw.rounded_rectangle(panel, radius=16, fill="#ffffff", outline="#dbe1ea")
    draw.text((x1 + 18, y1 + 14), title, fill="#172033", font=F_H)
    px, py = x1 + 18, y1 + 56
    pw, ph = x2 - x1 - 36, y2 - y1 - 76

    mask = point_ids == plane_id
    base_points = points.copy()
    moved_points = points.copy()
    moved_points[mask] = moved_points[mask] - edit_delta * normal[None, :]
    draw_points = moved_points if mode == "after" else base_points
    focus = np.flatnonzero(mask)
    sample = sample_points(points, np.where(mask, plane_id, -1), max_points=30000)
    proj, _, _ = projection(draw_points[sample], pw, ph, yaw=-0.58, pitch=0.16)
    order = np.argsort(proj[:, 2])
    local = Image.new("RGBA", (pw, ph), (255, 255, 255, 255))
    ld = ImageDraw.Draw(local, "RGBA")

    patch = plane_patch(draw_points, normal, offset + (edit_delta if mode == "after" else 0.0), mask)
    if patch is not None:
        pp, _, _ = projection(patch, pw, ph, yaw=-0.58, pitch=0.16)
        poly = [(float(a), float(b)) for a, b, _ in pp]
        ld.polygon(poly, fill=(37, 99, 235, 92), outline=(37, 99, 235, 255))

    if mode == "after":
        before_patch = plane_patch(base_points, normal, offset, mask)
        if before_patch is not None:
            bp, _, _ = projection(before_patch, pw, ph, yaw=-0.58, pitch=0.16)
            ld.line([tuple(x[:2]) for x in bp] + [tuple(bp[0, :2])], fill=(148, 163, 184, 180), width=3)

    for oi in order:
        gi = sample[oi]
        sx, sy = int(proj[oi, 0]), int(proj[oi, 1])
        if not (0 <= sx < pw and 0 <= sy < ph):
            continue
        if mask[gi]:
            c = (225, 29, 72) if mode != "after" else (37, 99, 235)
            r = 2
            alpha = 230
        else:
            rgb = colors[gi]
            gray = int(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
            c = (gray, gray, gray)
            r = 1
            alpha = 55
        ld.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(*c, alpha))

    if mode == "after":
        center = points[mask].mean(axis=0)
        moved_center = center - edit_delta * normal
        arrow_pts, _, _ = projection(np.stack([center, moved_center]), pw, ph, yaw=-0.58, pitch=0.16)
        a = (int(arrow_pts[0, 0]), int(arrow_pts[0, 1]))
        b = (int(arrow_pts[1, 0]), int(arrow_pts[1, 1]))
        ld.line((a, b), fill=(225, 29, 72, 255), width=7)
        ld.ellipse((b[0] - 9, b[1] - 9, b[0] + 9, b[1] + 9), fill=(225, 29, 72, 255))
        ld.text((18, ph - 32), f"offset d -> d {edit_delta:+.2f}: selected support moves together", fill=(225, 29, 72, 255), font=F_SMALL)
    elif mode == "equation":
        ld.text((18, ph - 58), f"Plane {plane_id}: n*x + d = 0", fill=(23, 32, 51, 255), font=F_SMALL)
        ld.text(
            (18, ph - 34),
            f"n=[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}], d={offset:.3f}",
            fill=(23, 32, 51, 255),
            font=F_MONO,
        )
    img.paste(local.convert("RGB"), (px, py))


def make_selected_plane_figure(sample="val_000030", plane_id=3, edit_delta=0.16):
    data = np.load(DATA_DIR / f"{sample}_refined_plane_proposals_data.npz")
    points = data["points"].astype(np.float32)
    colors = data["colors"].astype(np.uint8)
    point_ids = data["point_plane_ids"].astype(np.int32)
    plane_ids = data["plane_ids"].astype(np.int32)
    normals = data["plane_normals"].astype(np.float32)
    offsets = data["plane_offsets"].astype(np.float32)
    idx = list(plane_ids).index(plane_id)
    normal = normals[idx]
    offset = float(offsets[idx])
    count = int((point_ids == plane_id).sum())

    img = Image.new("RGB", (1600, 980), "#f7f8fb")
    d = ImageDraw.Draw(img)
    d.text((46, 34), "更直观地展示：一个平面 primitive 和一次编辑", fill="#172033", font=F_TITLE)
    d.text((46, 80), "这里只突出一个平面：红/蓝色点是绑定到该平面的 support points，半透明矩形是对应平面方程。", fill="#4b5563", font=F_BODY)
    draw_selected_panel(img, (45, 130, 535, 705), points, colors, point_ids, normal, offset, plane_id, "1. 选中一个结构平面", mode="before")
    draw_selected_panel(img, (555, 130, 1045, 705), points, colors, point_ids, normal, offset, plane_id, "2. 平面参数方程", mode="equation")
    draw_selected_panel(img, (1065, 130, 1555, 705), points, colors, point_ids, normal, offset, plane_id, "3. 修改 offset 后的编辑效果", mode="after", edit_delta=edit_delta)

    d.rounded_rectangle((58, 750, 1542, 925), radius=16, fill="#ffffff", outline="#dbe1ea")
    d.text((86, 772), "这张图比前面的线图更适合回答老师的问题", fill="#172033", font=F_H)
    notes = [
        f"面的体现：半透明蓝色矩形就是 Plane {plane_id} 的 primitive，来自方程 n*x + d = 0，而不是手动画的装饰。",
        f"绑定体现：该平面绑定了 {count:,} 个点，说明我们维护了 point-to-plane assignment。",
        "可编辑体现：右图改变 offset 后，绑定到该面的点和面 patch 一起沿法向移动。",
        "诚实说明：当前平面还不是完美语义墙面，但已经能展示结构参数化和编辑机制。",
    ]
    yy = 812
    for note in notes:
        d.text((90, yy), note, fill="#4b5563", font=F_BODY)
        yy += 28
    out = ASSETS / f"fig6_selected_plane_editability_{sample}_plane{plane_id}.png"
    img.save(out, quality=95)
    print(out)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_figure()
    make_selected_plane_figure()
