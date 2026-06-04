import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "weekly_report_2026-06-05"
ASSETS = OUT / "assets"


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
F_MONO = font(14)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_metric_chart():
    reports = {
        "v3 conservative": read_json(ROOT / "outputs/plane_refinement_report_v1/plane_refinement_report_summary.json"),
        "v4 binding+": read_json(ROOT / "outputs/plane_refinement_report_v4_binding_stronger/plane_refinement_report_summary.json"),
        "v6 soft assign + refit": read_json(
            ROOT / "outputs/plane_refinement_report_v6_soft_assignment_refit/plane_refinement_report_summary.json"
        ),
    }
    w, h = 1400, 760
    img = Image.new("RGB", (w, h), "#f7f8fb")
    d = ImageDraw.Draw(img)
    d.text((48, 34), "本周核心实验对比", fill="#172033", font=F_TITLE)
    d.text((48, 78), "v6 将 learned soft assignment 反馈到 plane equation refit，指标明显优于前两版。", fill="#4b5563", font=F_BODY)

    metrics = [
        ("Plane residual improvement", "relative_improvement", "%", "#2563eb"),
        ("Binding residual improvement", "binding_relative_improvement", "%", "#16a34a"),
    ]
    x0, y0 = 90, 170
    panel_w, panel_h = 570, 440
    for mi, (title, key, unit, color) in enumerate(metrics):
        px = x0 + mi * 650
        d.rounded_rectangle((px, y0, px + panel_w, y0 + panel_h), radius=18, fill="#ffffff", outline="#dbe1ea")
        d.text((px + 28, y0 + 24), title, fill="#172033", font=F_H)
        vals = [reports[name][key] * 100.0 for name in reports]
        max_v = max(vals) * 1.18
        bar_w = 110
        gap = 55
        base_y = y0 + 360
        for i, (name, summary) in enumerate(reports.items()):
            val = summary[key] * 100.0
            bh = int((val / max_v) * 240) if max_v else 0
            bx = px + 52 + i * (bar_w + gap)
            d.rounded_rectangle((bx, base_y - bh, bx + bar_w, base_y), radius=8, fill=color)
            d.text((bx + 8, base_y - bh - 28), f"{val:.2f}%", fill="#172033", font=F_SMALL)
            label = name.replace(" + ", "\n+ ")
            yy = base_y + 18
            for line in label.split("\n"):
                d.text((bx - 8, yy), line, fill="#4b5563", font=F_SMALL)
                yy += 18
        d.line((px + 38, base_y, px + panel_w - 30, base_y), fill="#9ca3af", width=2)

    rows = [
        ("v3", reports["v3 conservative"], "参数 refine + confidence + keep/remove"),
        ("v4", reports["v4 binding+"], "单纯加大 binding loss，效果下降"),
        ("v6", reports["v6 soft assign + refit"], "soft assignment 指导重新拟合平面方程"),
    ]
    table_y = 645
    d.text((70, table_y), "结论：v6 是主结果；v4 作为消融说明“不是简单加大绑定损失就有效”。", fill="#172033", font=F_BODY)
    img.save(ASSETS / "fig1_metrics_comparison.png", quality=95)


def project_points(points, width, height, yaw=-0.62, pitch=0.34):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = 0.78 * min(width, height) / max(float(np.max(maxs - mins)), 1e-6)
    p = points - center
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x1 = cy * p[:, 0] + sy * p[:, 2]
    z1 = -sy * p[:, 0] + cy * p[:, 2]
    y1 = cp * p[:, 1] - sp * z1
    z2 = sp * p[:, 1] + cp * z1
    x = width / 2 + x1 * scale
    y = height / 2 - y1 * scale
    return np.stack([x, y, z2], axis=1)


def sample_indices(total, focus, max_points=28000):
    focus = np.asarray(focus, dtype=np.int64)
    if total <= max_points:
        return np.arange(total)
    keep_focus = focus[: min(len(focus), max_points // 2)]
    remaining = max_points - len(keep_focus)
    stride = max(1, total // max(remaining, 1))
    background = np.arange(0, total, stride)[:remaining]
    return np.unique(np.concatenate([keep_focus, background]))[:max_points]


def draw_point_panel(draw, x, y, w, h, points, colors, states, title):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=16, fill="#ffffff", outline="#dbe1ea")
    draw.text((x + 18, y + 16), title, fill="#172033", font=F_H)
    plot_x, plot_y = x + 20, y + 58
    plot_w, plot_h = w - 40, h - 78
    proj = project_points(points, plot_w, plot_h)
    order = np.argsort(proj[:, 2])
    for idx in order:
        px = int(plot_x + proj[idx, 0])
        py = int(plot_y + proj[idx, 1])
        if not (plot_x <= px < plot_x + plot_w and plot_y <= py < plot_y + plot_h):
            continue
        state = states[idx]
        if state == "kept":
            c = "#16a34a"
            r = 2
        elif state == "added":
            c = "#2563eb"
            r = 3
        elif state == "removed":
            c = "#dc2626"
            r = 3
        elif state == "bound":
            c = "#16a34a"
            r = 2
        else:
            rgb = colors[idx]
            c = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            r = 1
        draw.ellipse((px - r, py - r, px + r, py + r), fill=c)


def make_binding_figure(sample="val_000027", plane_id=1):
    base_npz = OUT / "data" / f"{sample}_full_pointcloud_editable_planes_data.npz"
    refined_npz = ROOT / "outputs" / "plane_proposal_refinement_train3997_v6_soft_assignment_refit" / f"{sample}_refined_plane_proposals_data.npz"
    base = np.load(base_npz)
    refined = np.load(refined_npz)
    points = base["points"].astype(np.float32)
    colors = base["colors"].astype(np.uint8)
    base_mask = base["point_plane_ids"].astype(np.int32) == plane_id
    refined_mask = refined["point_plane_ids"].astype(np.int32) == plane_id
    kept = base_mask & refined_mask
    added = (~base_mask) & refined_mask
    removed = base_mask & (~refined_mask)
    focus = np.flatnonzero(base_mask | refined_mask)
    idx = sample_indices(len(points), focus)
    before_states = np.where(base_mask[idx], "bound", "other")
    after_states = np.full(len(idx), "other", dtype=object)
    after_states[kept[idx]] = "kept"
    after_states[added[idx]] = "added"
    after_states[removed[idx]] = "removed"
    w, h = 1500, 780
    img = Image.new("RGB", (w, h), "#f7f8fb")
    d = ImageDraw.Draw(img)
    d.text((48, 32), f"v6 Soft Assignment Binding Refinement ({sample}, Plane {plane_id})", fill="#172033", font=F_TITLE)
    d.text((48, 76), "绿色=保留/绑定，蓝色=新增绑定，红色=移除绑定。v6 学到的 assignment 会参与后续平面方程 refit。", fill="#4b5563", font=F_BODY)
    draw_point_panel(d, 50, 125, 675, 565, points[idx], colors[idx], before_states, "Before: RANSAC proposal support")
    draw_point_panel(d, 775, 125, 675, 565, points[idx], colors[idx], after_states, "After: learned assignment support")
    metrics = [
        f"Base bound: {int(base_mask.sum()):,}",
        f"Refined bound: {int(refined_mask.sum()):,}",
        f"Kept: {int(kept.sum()):,}",
        f"Added: {int(added.sum()):,}",
        f"Removed: {int(removed.sum()):,}",
    ]
    x = 62
    for m in metrics:
        d.rounded_rectangle((x, 710, x + 245, 752), radius=10, fill="#ffffff", outline="#dbe1ea")
        d.text((x + 12, 721), m, fill="#172033", font=F_BODY)
        x += 278
    img.save(ASSETS / f"fig2_binding_{sample}.png", quality=95)


def line_from_planes(n1, d1, n2, d2):
    direction = np.cross(n1, n2).astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction /= norm
    a = np.stack([n1, n2, direction], axis=0).astype(np.float32)
    b = np.asarray([-d1, -d2, 0.0], dtype=np.float32)
    point = np.linalg.solve(a, b).astype(np.float32)
    return point, direction


def make_line_figure(sample="val_000027"):
    data = np.load(ROOT / "outputs" / "plane_proposal_refinement_train3997_v6_soft_assignment_refit" / f"{sample}_refined_plane_proposals_data.npz")
    lines_json = read_json(ROOT / "outputs" / "plane_line_primitives_v6" / f"{sample}_plane_intersection_lines.json")
    points = data["points"].astype(np.float32)
    colors = data["colors"].astype(np.uint8)
    focus = []
    for line in lines_json["lines"]:
        p = np.asarray(line["point"], dtype=np.float32)
        v = np.asarray(line["direction"], dtype=np.float32)
        dist = np.linalg.norm(np.cross(points - p[None, :], v[None, :]), axis=1)
        focus.extend(np.flatnonzero(dist < 0.035)[:3000].tolist())
    idx = sample_indices(len(points), np.asarray(focus), max_points=30000)
    w, h = 1500, 820
    img = Image.new("RGB", (w, h), "#f7f8fb")
    d = ImageDraw.Draw(img)
    d.text((48, 32), f"Plane-to-Line Primitive Extension ({sample})", fill="#172033", font=F_TITLE)
    d.text((48, 76), "由 refined plane equations 直接求交线：x = p + t*v。线 primitive 可作为后续 line head 的弱监督和结构一致性约束。", fill="#4b5563", font=F_BODY)
    panel = (50, 124, 1040, 760)
    d.rounded_rectangle(panel, radius=18, fill="#ffffff", outline="#dbe1ea")
    plot_x, plot_y = panel[0] + 20, panel[1] + 20
    plot_w, plot_h = panel[2] - panel[0] - 40, panel[3] - panel[1] - 40
    proj = project_points(points[idx], plot_w, plot_h)
    order = np.argsort(proj[:, 2])
    for oi in order:
        px = int(plot_x + proj[oi, 0])
        py = int(plot_y + proj[oi, 1])
        if plot_x <= px < plot_x + plot_w and plot_y <= py < plot_y + plot_h:
            c = colors[idx[oi]]
            d.ellipse((px - 1, py - 1, px + 1, py + 1), fill=(int(c[0]), int(c[1]), int(c[2])))
    palette = ["#e11d48", "#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2", "#db2777", "#4b5563"]
    all_line_points = []
    for line in lines_json["lines"][:8]:
        all_line_points.append(np.asarray(line["segment_start"], dtype=np.float32))
        all_line_points.append(np.asarray(line["segment_end"], dtype=np.float32))
    if all_line_points:
        line_proj = project_points(np.stack(all_line_points), plot_w, plot_h)
        for li in range(0, len(line_proj), 2):
            c = palette[(li // 2) % len(palette)]
            a = (int(plot_x + line_proj[li, 0]), int(plot_y + line_proj[li, 1]))
            b = (int(plot_x + line_proj[li + 1, 0]), int(plot_y + line_proj[li + 1, 1]))
            d.line((a, b), fill=c, width=5)
    side_x = 1085
    d.rounded_rectangle((side_x, 124, 1450, 760), radius=18, fill="#ffffff", outline="#dbe1ea")
    d.text((side_x + 24, 148), "Line primitive summary", fill="#172033", font=F_H)
    d.text((side_x + 24, 190), f"Planes: {lines_json['plane_count']}", fill="#172033", font=F_BODY)
    d.text((side_x + 24, 222), f"Lines: {len(lines_json['lines'])}", fill="#172033", font=F_BODY)
    d.text((side_x + 24, 254), f"Mean support: {lines_json['mean_support']:.1f}", fill="#172033", font=F_BODY)
    y = 310
    for i, line in enumerate(lines_json["lines"][:6]):
        c = palette[i % len(palette)]
        d.rounded_rectangle((side_x + 24, y, side_x + 334, y + 58), radius=9, fill="#f7f8fb", outline="#e1e6ee")
        d.rectangle((side_x + 38, y + 18, side_x + 58, y + 38), fill=c)
        d.text((side_x + 70, y + 10), f"Line {i}: P{line['plane_a']} x P{line['plane_b']}", fill="#172033", font=F_SMALL)
        d.text((side_x + 70, y + 31), f"support {line['support_count']:,}", fill="#4b5563", font=F_SMALL)
        y += 70
    img.save(ASSETS / f"fig3_lines_{sample}.png", quality=95)


def make_report():
    summary = (ROOT / "outputs" / "report_ready_summary_2026-06-04.md").read_text(encoding="utf-8")
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>周内小报告：可编辑结构 Primitive 重建实验进展</title>
<style>
body {{ font-family: "Microsoft YaHei", Arial, sans-serif; max-width: 1060px; margin: 36px auto; color: #172033; line-height: 1.62; }}
h1 {{ font-size: 30px; margin-bottom: 8px; }}
h2 {{ margin-top: 34px; border-bottom: 1px solid #dbe1ea; padding-bottom: 6px; }}
.lead {{ color:#4b5563; font-size: 16px; }}
.card {{ background:#f7f8fb; border:1px solid #dbe1ea; border-radius:12px; padding:16px 18px; margin:16px 0; }}
img {{ width:100%; border:1px solid #dbe1ea; border-radius:12px; margin:10px 0 4px; }}
.caption {{ color:#5b6371; font-size: 14px; margin-bottom:20px; }}
table {{ border-collapse: collapse; width:100%; margin: 12px 0; }}
th,td {{ border:1px solid #dbe1ea; padding:8px 10px; }}
th {{ background:#eef2f7; }}
code {{ background:#f1f5f9; padding:2px 4px; border-radius:4px; }}
</style>
</head>
<body>
<h1>周内小报告：可编辑结构 Primitive 重建实验进展</h1>
<p class="lead">本周围绕 DUSt3R/MASt3R 点云结果，探索如何从点云中得到可编辑、可解释的结构 primitive 参数，重点是平面方程、点到平面的绑定关系，以及由平面诱导出的线结构。</p>

<h2>1. 本周主要完成内容</h2>
<div class="card">
<ul>
<li>建立了 <b>RANSAC proposal + learned refinement head</b> 的实验链路。</li>
<li>从硬 keep/remove 分类升级到 <b>soft point-to-plane assignment</b>。</li>
<li>提出并验证 v6：用 learned assignment 反向重新拟合平面方程。</li>
<li>增加 plane-plane intersection line primitive 可视化，为后续 line head 做准备。</li>
</ul>
</div>

<h2>2. 核心指标</h2>
<img src="assets/fig1_metrics_comparison.png" alt="核心实验指标对比">
<p class="caption">图1：v6 的参数残差和 binding 残差均优于 v3/v4，说明“soft assignment 指导平面 refit”比单纯调大 binding loss 更有效。</p>

<table>
<tr><th>版本</th><th>方法含义</th><th>平面残差提升</th><th>Binding 残差提升</th></tr>
<tr><td>v3</td><td>参数 refine + confidence + keep/remove</td><td>2.94%</td><td>10.59%</td></tr>
<tr><td>v4</td><td>加大 binding loss 的消融</td><td>1.71%</td><td>7.89%</td></tr>
<tr><td><b>v6</b></td><td><b>soft assignment + learned support refit</b></td><td><b>4.39%</b></td><td><b>19.32%</b></td></tr>
</table>

<h2>3. 可视化结果：点到平面的绑定关系</h2>
<img src="assets/fig2_binding_val_000027.png" alt="v6 binding refinement">
<p class="caption">图2：以 val_000027 的一个主平面为例。左侧为 RANSAC proposal support，右侧为 v6 learned assignment 后的 support。蓝色点表示新增绑定，红色点表示移除绑定，绿色点表示保留。</p>

<h2>4. 可视化结果：由平面参数诱导线 Primitive</h2>
<img src="assets/fig3_lines_val_000027.png" alt="plane induced line primitives">
<p class="caption">图3：由 refined plane equations 计算 plane-plane intersection，得到线 primitive。该结果说明平面参数不仅能用于编辑平面，也能进一步产生线结构约束。</p>

<h2>5. 当前结论</h2>
<div class="card">
<p>本周结果说明，当前工作不只是 RANSAC 后处理。RANSAC 只作为 coarse proposal，真正的创新点在于学习点到结构 primitive 的归属关系，并将 learned assignment 反馈到结构参数方程估计中。</p>
<p>目前 v6 是主结果：平均平面残差从 <code>0.007434</code> 降到 <code>0.007107</code>，提升 <b>4.39%</b>；largest-plane binding 残差提升 <b>19.32%</b>。</p>
</div>

<h2>6. 下周计划</h2>
<ol>
<li>把 plane-plane intersection 得到的线作为弱监督，训练一个 line parameter head。</li>
<li>加入 line-plane consistency loss，使线必须落在对应两个平面上。</li>
<li>继续加入 RGB / 局部几何一致性，提升 soft assignment 的稳定性。</li>
<li>补充更多样本的成功/失败分析，筛选适合放进最终报告的可视化案例。</li>
</ol>

<h2>附：结果位置</h2>
<p>主报告数据目录：<code>outputs/plane_refinement_report_v6_soft_assignment_refit</code></p>
<p>Binding 可视化目录：<code>outputs/plane_binding_refinement_v6_soft_assignment_refit</code></p>
<p>Line primitive 可视化目录：<code>outputs/plane_line_primitives_v6</code></p>
</body>
</html>
"""
    md = """# 周内小报告：可编辑结构 Primitive 重建实验进展

本周围绕 DUSt3R/MASt3R 点云结果，探索如何从点云中得到可编辑、可解释的结构 primitive 参数。

![核心实验指标对比](assets/fig1_metrics_comparison.png)

![v6 binding refinement](assets/fig2_binding_val_000027.png)

![plane induced line primitives](assets/fig3_lines_val_000027.png)

## 核心结论

- v6 是主结果：soft assignment + learned support refit。
- 平面残差提升 4.39%。
- largest-plane binding 残差提升 19.32%。
- plane equation 可以进一步诱导 line primitive，为后续 line head 提供弱监督。

## 下周计划

1. 训练 line parameter head。
2. 加入 line-plane consistency loss。
3. 加入 RGB / 局部几何一致性。
4. 补充更多成功/失败案例。
"""
    (OUT / "weekly_progress_report_2026-06-05.html").write_text(html, encoding="utf-8")
    (OUT / "weekly_progress_report_2026-06-05.md").write_text(md, encoding="utf-8")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_metric_chart()
    make_binding_figure()
    make_line_figure()
    make_report()
    print(OUT / "weekly_progress_report_2026-06-05.html")


if __name__ == "__main__":
    main()
