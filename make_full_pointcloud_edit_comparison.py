import argparse
import json
from pathlib import Path

import numpy as np


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Full Pointcloud Plane Edit Comparison</title>
<style>
:root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }
body { margin:0; background:#f5f6f8; color:#171a1f; }
.shell { display:grid; grid-template-columns: minmax(0,1fr) 360px; min-height:100vh; }
.stage { display:grid; grid-template-columns: 1fr 1fr; gap:1px; background:#d9dde5; }
.view { position:relative; background:#ffffff; min-height:100vh; }
canvas { width:100%; height:100%; display:block; cursor:grab; }
canvas:active { cursor:grabbing; }
.label { position:absolute; left:14px; top:12px; padding:6px 9px; background:rgba(255,255,255,.92); border:1px solid #d8dce3; border-radius:6px; font-size:13px; font-weight:700; }
.side { border-left:1px solid #d8dce3; background:#ffffff; padding:18px 18px 24px; overflow:auto; }
h1 { margin:0 0 12px; font-size:20px; line-height:1.2; }
h2 { margin:22px 0 9px; font-size:14px; color:#383f4c; }
.metricGrid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.metric { border:1px solid #dde1e8; border-radius:6px; padding:9px 10px; background:#fbfcfe; }
.metric b { display:block; font-size:18px; margin-top:3px; }
.plane { border-top:1px solid #e4e7ed; padding:11px 0; }
.planeHeader { display:flex; align-items:center; gap:8px; font-weight:700; }
.swatch { width:14px; height:14px; border-radius:3px; border:1px solid rgba(0,0,0,.2); flex:0 0 auto; }
.eq { margin:6px 0 0; font-family:Consolas, Menlo, monospace; font-size:12px; line-height:1.4; color:#242936; overflow-wrap:anywhere; }
.small { margin-top:5px; font-size:12px; color:#5b6371; }
.hint { margin-top:14px; font-size:12px; color:#6b7280; line-height:1.45; }
.pill { display:inline-block; padding:2px 7px; border-radius:999px; background:#edf2ff; color:#214a9a; font-size:12px; font-weight:700; }
</style>
</head>
<body>
<div class="shell">
  <main class="stage">
    <section class="view"><canvas id="before"></canvas><div class="label">Before: DUSt3R full point cloud</div></section>
    <section class="view"><canvas id="after"></canvas><div class="label">After: plane offset edit</div></section>
  </main>
  <aside class="side">
    <h1>Editable Plane Parameters</h1>
    <div class="pill">nx*x + ny*y + nz*z + d = 0</div>
    <div class="metricGrid">
      <div class="metric">Full points<b id="totalPoints"></b></div>
      <div class="metric">Major planes<b id="planeCount"></b></div>
      <div class="metric">Edited plane<b id="editedPlane"></b></div>
      <div class="metric">Moved points<b id="movedPoints"></b></div>
    </div>
    <h2>Offset Edit</h2>
    <div id="editInfo" class="small"></div>
    <h2>Plane Equations</h2>
    <div id="planes"></div>
    <div class="hint">The page displays a deterministic sample for browser speed. Counts, equations, assignments, and the edit are computed from the complete point cloud stored in the NPZ.</div>
  </aside>
</div>
<script>
const DATA = __DATA__;
let yaw = -0.62, pitch = 0.34, scale = DATA.scale;
let dragging = false, lastX = 0, lastY = 0;

function fmt(x, n=4) { return Number(x).toFixed(n); }
function rgb(c) { return `rgb(${c[0]},${c[1]},${c[2]})`; }
function equation(p, after=false) {
  const d = after && p.id === DATA.edit.plane_id ? p.offset + DATA.edit.delta : p.offset;
  return `${fmt(p.normal[0],6)}*x + ${fmt(p.normal[1],6)}*y + ${fmt(p.normal[2],6)}*z + ${fmt(d,6)} = 0`;
}
function renderSide() {
  totalPoints.textContent = DATA.total_points.toLocaleString();
  planeCount.textContent = DATA.planes.length;
  editedPlane.textContent = DATA.edit.plane_id;
  movedPoints.textContent = DATA.moved_points.toLocaleString();
  editInfo.textContent = `Plane ${DATA.edit.plane_id}: d ${fmt(DATA.edit.offset_before,6)} -> ${fmt(DATA.edit.offset_after,6)} (delta ${fmt(DATA.edit.delta,3)}), moving assigned full-pointcloud points along the plane normal.`;
  planes.innerHTML = DATA.planes.map(p => `
    <div class="plane">
      <div class="planeHeader"><span class="swatch" style="background:${rgb(p.color)}"></span>Plane ${p.id}</div>
      <div class="eq">before: ${equation(p, false)}</div>
      <div class="eq">after: ${equation(p, true)}</div>
      <div class="small">inliers ${p.inlier_count.toLocaleString()} · assigned full points ${p.assigned_point_count.toLocaleString()}</div>
    </div>
  `).join('');
}
function resizeCanvas(canvas) {
  canvas.width = Math.max(1, canvas.clientWidth * devicePixelRatio);
  canvas.height = Math.max(1, canvas.clientHeight * devicePixelRatio);
}
function rotate(p) {
  const x = p[0] - DATA.center[0], y = p[1] - DATA.center[1], z = p[2] - DATA.center[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy*x + sy*z, z1 = -sy*x + cy*z;
  const y1 = cp*y - sp*z1, z2 = sp*y + cp*z1;
  return [x1, y1, z2];
}
function project(p, canvas) {
  const r = rotate(p);
  const s = Math.min(canvas.width, canvas.height) * scale;
  return [canvas.width/2 + r[0]*s, canvas.height/2 - r[1]*s, r[2]];
}
function draw(canvas, pointsKey) {
  resizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const pts = DATA[pointsKey].map((p, i) => [project(p, canvas), i]).sort((a,b) => a[0][2] - b[0][2]);
  const r = Math.max(1.0, 1.45 * devicePixelRatio);
  for (const [pp, i] of pts) {
    const moved = DATA.sample_moved[i];
    ctx.fillStyle = moved && pointsKey === 'after_points' ? '#e11d48' : DATA.sample_colors[i];
    ctx.globalAlpha = moved ? 0.96 : 0.74;
    ctx.beginPath();
    ctx.arc(pp[0], pp[1], moved ? r*1.25 : r, 0, Math.PI*2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}
function drawAll() {
  draw(document.getElementById('before'), 'before_points');
  draw(document.getElementById('after'), 'after_points');
}
function attach(canvas) {
  canvas.addEventListener('pointerdown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    yaw += (e.clientX - lastX) * 0.008;
    pitch += (e.clientY - lastY) * 0.008;
    pitch = Math.max(-1.35, Math.min(1.35, pitch));
    lastX = e.clientX; lastY = e.clientY;
    drawAll();
  });
  canvas.addEventListener('pointerup', () => { dragging = false; });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    scale *= e.deltaY > 0 ? 0.92 : 1.08;
    scale = Math.max(0.02, Math.min(5, scale));
    drawAll();
  }, { passive:false });
}
renderSide();
attach(document.getElementById('before'));
attach(document.getElementById('after'));
addEventListener('resize', drawAll);
drawAll();
</script>
</body>
</html>
"""


PLANE_COLORS = [
    [229, 57, 53],
    [30, 136, 229],
    [67, 160, 71],
    [251, 140, 0],
    [142, 36, 170],
    [0, 137, 123],
    [109, 76, 65],
    [57, 73, 171],
    [198, 40, 40],
    [0, 121, 107],
]


def load_npz(path):
    raw = np.load(path)
    return {
        "points": raw["points"].astype(np.float32),
        "colors": raw["colors"].astype(np.uint8),
        "point_plane_ids": raw["point_plane_ids"].astype(np.int32),
        "plane_ids": raw["plane_ids"].astype(np.int32),
        "plane_normals": raw["plane_normals"].astype(np.float32),
        "plane_offsets": raw["plane_offsets"].astype(np.float32),
        "plane_inlier_counts": raw["plane_inlier_counts"].astype(np.int32),
    }


def select_plane(data, edit_plane):
    plane_counts = {}
    for plane_id in data["plane_ids"]:
        plane_counts[int(plane_id)] = int(np.sum(data["point_plane_ids"] == int(plane_id)))
    if edit_plane == "largest":
        return max(plane_counts, key=plane_counts.get)
    return int(edit_plane)


def build_planes(data):
    planes = []
    for i, plane_id in enumerate(data["plane_ids"]):
        pid = int(plane_id)
        planes.append(
            {
                "id": pid,
                "normal": [float(x) for x in data["plane_normals"][i]],
                "offset": float(data["plane_offsets"][i]),
                "inlier_count": int(data["plane_inlier_counts"][i]),
                "assigned_point_count": int(np.sum(data["point_plane_ids"] == pid)),
                "color": PLANE_COLORS[i % len(PLANE_COLORS)],
            }
        )
    return planes


def deterministic_sample(total, moved_mask, max_display_points):
    moved_idx = np.flatnonzero(moved_mask)
    base_n = max_display_points - min(len(moved_idx), max_display_points // 2)
    base_n = max(0, base_n)
    base_idx = np.linspace(0, total - 1, min(base_n, total), dtype=np.int64)
    if len(moved_idx) > 0:
        keep_moved = moved_idx[
            np.linspace(0, len(moved_idx) - 1, min(len(moved_idx), max_display_points - len(base_idx)), dtype=np.int64)
        ]
        idx = np.unique(np.concatenate([base_idx, keep_moved]))
    else:
        idx = base_idx
    if len(idx) > max_display_points:
        idx = idx[np.linspace(0, len(idx) - 1, max_display_points, dtype=np.int64)]
    return idx.astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--edit_plane", default="largest")
    parser.add_argument("--edit_delta", type=float, default=0.25)
    parser.add_argument("--max_display_points", type=int, default=28000)
    args = parser.parse_args()

    data = load_npz(args.input_npz)
    edit_plane_id = select_plane(data, args.edit_plane)
    plane_index = np.where(data["plane_ids"] == edit_plane_id)[0]
    if len(plane_index) != 1:
        raise ValueError(f"Cannot find plane {edit_plane_id}")
    plane_index = int(plane_index[0])
    normal = data["plane_normals"][plane_index]
    offset_before = float(data["plane_offsets"][plane_index])
    moved_mask = data["point_plane_ids"] == edit_plane_id

    edited_points = data["points"].copy()
    edited_points[moved_mask] = edited_points[moved_mask] - float(args.edit_delta) * normal

    sample_idx = deterministic_sample(len(data["points"]), moved_mask, args.max_display_points)
    sample_colors = []
    for pid, color in zip(data["point_plane_ids"][sample_idx], data["colors"][sample_idx]):
        if int(pid) >= 0:
            plane_pos = np.where(data["plane_ids"] == int(pid))[0]
            if len(plane_pos) == 1:
                sample_colors.append(f"rgb({','.join(str(x) for x in PLANE_COLORS[int(plane_pos[0]) % len(PLANE_COLORS)])})")
                continue
        sample_colors.append(f"rgb({int(color[0])},{int(color[1])},{int(color[2])})")

    mins = data["points"].min(axis=0)
    maxs = data["points"].max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    scale = 0.74 / max(span, 1e-6)

    html_data = {
        "input_npz": str(args.input_npz),
        "total_points": int(len(data["points"])),
        "display_points": int(len(sample_idx)),
        "moved_points": int(np.sum(moved_mask)),
        "center": [float(x) for x in center],
        "scale": scale,
        "planes": build_planes(data),
        "edit": {
            "plane_id": int(edit_plane_id),
            "delta": float(args.edit_delta),
            "offset_before": offset_before,
            "offset_after": offset_before + float(args.edit_delta),
        },
        "before_points": data["points"][sample_idx].round(5).tolist(),
        "after_points": edited_points[sample_idx].round(5).tolist(),
        "sample_moved": moved_mask[sample_idx].astype(bool).tolist(),
        "sample_colors": sample_colors,
    }

    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        HTML_TEMPLATE.replace("__DATA__", json.dumps(html_data, separators=(",", ":"))),
        encoding="utf-8",
    )
    print(output_html)
    print(
        f"total_points={html_data['total_points']} display_points={html_data['display_points']} "
        f"edit_plane={edit_plane_id} moved_points={html_data['moved_points']}"
    )


if __name__ == "__main__":
    main()
