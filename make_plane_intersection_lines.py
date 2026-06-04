import argparse
import csv
import json
from pathlib import Path

import numpy as np


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Plane Intersection Line Primitives</title>
<style>
:root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }
body { margin:0; background:#f5f6f8; color:#171a1f; }
.shell { display:grid; grid-template-columns:minmax(0,1fr) 390px; min-height:100vh; }
.view { position:relative; background:#fff; min-height:100vh; }
canvas { width:100%; height:100%; display:block; cursor:grab; }
canvas:active { cursor:grabbing; }
.label { position:absolute; left:14px; top:12px; padding:6px 9px; background:rgba(255,255,255,.94); border:1px solid #d8dce3; border-radius:6px; font-size:13px; font-weight:700; }
.side { border-left:1px solid #d8dce3; background:#fff; padding:18px 18px 24px; overflow:auto; }
h1 { margin:0 0 12px; font-size:20px; line-height:1.2; }
h2 { margin:20px 0 8px; font-size:14px; color:#383f4c; }
.metricGrid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.metric { border:1px solid #dde1e8; border-radius:6px; padding:9px 10px; background:#fbfcfe; }
.metric b { display:block; font-size:18px; margin-top:3px; }
.row { border-top:1px solid #e4e7ed; padding:10px 0; }
.small { font-size:12px; color:#5b6371; line-height:1.45; }
.eq { font-family:Consolas, Menlo, monospace; font-size:12px; line-height:1.45; overflow-wrap:anywhere; color:#242936; }
.swatch { display:inline-block; width:13px; height:13px; border-radius:3px; margin-right:6px; vertical-align:-2px; border:1px solid rgba(0,0,0,.2); }
</style>
</head>
<body>
<div class="shell">
  <main class="view"><canvas id="canvas"></canvas><div class="label">Plane-induced line primitives</div></main>
  <aside class="side">
    <h1>Plane-Line Structure</h1>
    <div class="metricGrid">
      <div class="metric">Points<b id="pointCount"></b></div>
      <div class="metric">Planes<b id="planeCount"></b></div>
      <div class="metric">Lines<b id="lineCount"></b></div>
      <div class="metric">Mean support<b id="meanSupport"></b></div>
    </div>
    <h2>Line Equation</h2>
    <div class="small">Each line is derived from two learned plane equations. It is represented as x = p + t*v.</div>
    <div id="lines"></div>
    <h2>Interpretation</h2>
    <div class="small">This page shows that editable plane equations can induce editable line primitives. These lines can become weak targets for a future line head and line-plane consistency loss.</div>
  </aside>
</div>
<script>
const DATA = __DATA__;
let yaw = -0.62, pitch = 0.34, scale = DATA.scale;
let dragging = false, lastX = 0, lastY = 0;
function fmt(x, n=4) { return Number(x).toFixed(n); }
function renderSide() {
  pointCount.textContent = DATA.total_points.toLocaleString();
  planeCount.textContent = DATA.plane_count;
  lineCount.textContent = DATA.lines.length;
  meanSupport.textContent = fmt(DATA.mean_support, 1);
  lines.innerHTML = DATA.lines.map((l, i) => `
    <div class="row">
      <div><span class="swatch" style="background:${l.color}"></span><b>Line ${i}</b> from Plane ${l.plane_a} & Plane ${l.plane_b}</div>
      <div class="small">support ${l.support_count.toLocaleString()} pts, angle ${fmt(l.angle_deg,1)} deg</div>
      <div class="eq">p = [${l.point.map(x => fmt(x,5)).join(', ')}]</div>
      <div class="eq">v = [${l.direction.map(x => fmt(x,5)).join(', ')}]</div>
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
function draw() {
  const canvas = document.getElementById('canvas');
  resizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const pts = DATA.points.map((p, i) => [project(p, canvas), i]).sort((a,b) => a[0][2] - b[0][2]);
  const r = Math.max(1.0, 1.35 * devicePixelRatio);
  for (const [pp, i] of pts) {
    ctx.fillStyle = DATA.colors[i];
    ctx.globalAlpha = 0.54;
    ctx.beginPath();
    ctx.arc(pp[0], pp[1], r, 0, Math.PI*2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  ctx.lineCap = 'round';
  for (const l of DATA.lines) {
    const a = project(l.segment_start, canvas);
    const b = project(l.segment_end, canvas);
    ctx.strokeStyle = l.color;
    ctx.lineWidth = 4 * devicePixelRatio;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }
}
function attach(canvas) {
  canvas.addEventListener('mousedown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener('mouseup', () => dragging = false);
  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    yaw += (e.clientX - lastX) * 0.008;
    pitch += (e.clientY - lastY) * 0.008;
    pitch = Math.max(-1.45, Math.min(1.45, pitch));
    lastX = e.clientX; lastY = e.clientY;
    draw();
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    scale *= Math.exp(-e.deltaY * 0.001);
    draw();
  }, {passive:false});
}
renderSide();
attach(document.getElementById('canvas'));
draw();
window.addEventListener('resize', draw);
</script>
</body>
</html>
"""


COLORS = ["#e11d48", "#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2", "#db2777", "#4b5563"]


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


def deterministic_sample(total, focus_idx, max_display_points):
    focus_idx = np.asarray(focus_idx, dtype=np.int64)
    if total <= max_display_points:
        return np.arange(total, dtype=np.int64)
    focus_keep = focus_idx[: min(len(focus_idx), max_display_points // 2)]
    remaining = max_display_points - len(focus_keep)
    stride = max(1, total // max(remaining, 1))
    background = np.arange(0, total, stride, dtype=np.int64)[:remaining]
    return np.unique(np.concatenate([focus_keep, background]))[:max_display_points]


def build_lines(points, plane_ids, point_plane_ids, normals, offsets, args):
    lines = []
    focus = []
    for i in range(len(plane_ids)):
        for j in range(i + 1, len(plane_ids)):
            n1, n2 = normals[i], normals[j]
            angle_cos = abs(float(np.dot(n1, n2)))
            angle_deg = float(np.degrees(np.arccos(np.clip(angle_cos, -1.0, 1.0))))
            if angle_deg < args.min_angle_deg or angle_deg > 180.0 - args.min_angle_deg:
                continue
            result = line_from_planes(n1, float(offsets[i]), n2, float(offsets[j]))
            if result is None:
                continue
            line_point, direction = result
            candidate = (point_plane_ids == int(plane_ids[i])) | (point_plane_ids == int(plane_ids[j]))
            if not np.any(candidate):
                continue
            cand_idx = np.flatnonzero(candidate)
            cand_pts = points[cand_idx]
            dist = np.linalg.norm(np.cross(cand_pts - line_point[None, :], direction[None, :]), axis=1)
            support_local = np.flatnonzero(dist <= args.line_support_threshold)
            if len(support_local) < args.min_support_points:
                continue
            support_idx = cand_idx[support_local]
            support_pts = points[support_idx]
            t = (support_pts - line_point[None, :]) @ direction
            lo, hi = np.percentile(t, [2, 98])
            if float(hi - lo) < args.min_line_length:
                continue
            focus.extend(support_idx.tolist())
            lines.append(
                {
                    "plane_a": int(plane_ids[i]),
                    "plane_b": int(plane_ids[j]),
                    "point": line_point.tolist(),
                    "direction": direction.tolist(),
                    "segment_start": (line_point + direction * lo).tolist(),
                    "segment_end": (line_point + direction * hi).tolist(),
                    "support_count": int(len(support_idx)),
                    "angle_deg": angle_deg,
                    "length": float(hi - lo),
                }
            )
    lines.sort(key=lambda x: (x["support_count"], x["length"]), reverse=True)
    return lines[: args.max_lines], np.asarray(focus, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser("Extract line primitives from learned plane equations.")
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--summary_csv", default=None)
    parser.add_argument("--max_display_points", type=int, default=35000)
    parser.add_argument("--max_lines", type=int, default=8)
    parser.add_argument("--min_angle_deg", type=float, default=20.0)
    parser.add_argument("--line_support_threshold", type=float, default=0.035)
    parser.add_argument("--min_support_points", type=int, default=800)
    parser.add_argument("--min_line_length", type=float, default=0.08)
    args = parser.parse_args()

    raw = np.load(args.input_npz)
    points = raw["points"].astype(np.float32)
    colors = raw["colors"].astype(np.uint8)
    point_plane_ids = raw["point_plane_ids"].astype(np.int32)
    plane_ids = raw["plane_ids"].astype(np.int32)
    normals = raw["plane_normals"].astype(np.float32)
    offsets = raw["plane_offsets"].astype(np.float32)

    lines, focus_idx = build_lines(points, plane_ids, point_plane_ids, normals, offsets, args)
    sample_idx = deterministic_sample(len(points), focus_idx, args.max_display_points)
    sample_points = points[sample_idx]
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = float(0.74 / max(float(np.max(maxs - mins)), 1e-6))
    mean_support = float(np.mean([line["support_count"] for line in lines])) if lines else 0.0
    for i, line in enumerate(lines):
        line["color"] = COLORS[i % len(COLORS)]

    data = {
        "total_points": int(len(points)),
        "plane_count": int(len(plane_ids)),
        "mean_support": mean_support,
        "center": [float(x) for x in center],
        "scale": scale,
        "points": sample_points.round(5).tolist(),
        "colors": [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in colors[sample_idx]],
        "lines": lines,
    }
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(data, indent=2), encoding="utf-8")
    if args.summary_csv:
        csv_path = Path(args.summary_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample", "planes", "lines", "mean_support", "max_support"],
            )
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "sample": output_html.name.replace("_plane_intersection_lines.html", ""),
                    "planes": int(len(plane_ids)),
                    "lines": int(len(lines)),
                    "mean_support": mean_support,
                    "max_support": int(max([line["support_count"] for line in lines], default=0)),
                }
            )
    print(output_html)
    print(f"planes={len(plane_ids)} lines={len(lines)} mean_support={mean_support:.1f}")


if __name__ == "__main__":
    main()
