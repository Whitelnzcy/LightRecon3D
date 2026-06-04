import argparse
import csv
import json
from pathlib import Path

import numpy as np


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Plane Binding Refinement Comparison</title>
<style>
:root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }
body { margin:0; background:#f5f6f8; color:#171a1f; }
.shell { display:grid; grid-template-columns:minmax(0,1fr) 380px; min-height:100vh; }
.stage { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:#d9dde5; }
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
.legend { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px; }
.legendItem { font-size:12px; display:flex; align-items:center; gap:7px; color:#4b5563; }
.swatch { width:13px; height:13px; border-radius:3px; border:1px solid rgba(0,0,0,.2); flex:0 0 auto; }
.eq { font-family:Consolas, Menlo, monospace; font-size:12px; line-height:1.45; overflow-wrap:anywhere; color:#242936; }
.small { font-size:12px; color:#5b6371; line-height:1.45; }
.plane { border-top:1px solid #e4e7ed; padding:11px 0; }
.pill { display:inline-block; padding:2px 7px; border-radius:999px; background:#edf2ff; color:#214a9a; font-size:12px; font-weight:700; }
</style>
</head>
<body>
<div class="shell">
  <main class="stage">
    <section class="view"><canvas id="before"></canvas><div class="label">RANSAC binding mask</div></section>
    <section class="view"><canvas id="after"></canvas><div class="label">Learned refined binding mask</div></section>
  </main>
  <aside class="side">
    <h1>Plane Binding Refinement</h1>
    <div class="pill">selected plane: <span id="planeId"></span></div>
    <div class="metricGrid">
      <div class="metric">Base bound<b id="baseCount"></b></div>
      <div class="metric">Refined bound<b id="refinedCount"></b></div>
      <div class="metric">Kept<b id="keptCount"></b></div>
      <div class="metric">Added / removed<b id="changeCount"></b></div>
    </div>
    <div class="legend">
      <div class="legendItem"><span class="swatch" style="background:#16a34a"></span>kept</div>
      <div class="legendItem"><span class="swatch" style="background:#2563eb"></span>added</div>
      <div class="legendItem"><span class="swatch" style="background:#dc2626"></span>removed</div>
      <div class="legendItem"><span class="swatch" style="background:#9ca3af"></span>other points</div>
    </div>
    <h2>Residual</h2>
    <div id="residualInfo" class="small"></div>
    <h2>Plane Equation</h2>
    <div id="equations" class="eq"></div>
    <h2>All Planes</h2>
    <div id="planes"></div>
    <div class="small">This page compares point binding masks directly. It is meant to judge whether the learned keep/remove head changes the editable structural support, not just whether the plane equation residual is lower.</div>
  </aside>
</div>
<script>
const DATA = __DATA__;
let yaw = -0.62, pitch = 0.34, scale = DATA.scale;
let dragging = false, lastX = 0, lastY = 0;
function fmt(x, n=4) { return Number(x).toFixed(n); }
function eq(p) { return `${fmt(p.normal[0],6)}*x + ${fmt(p.normal[1],6)}*y + ${fmt(p.normal[2],6)}*z + ${fmt(p.offset,6)} = 0`; }
function renderSide() {
  planeId.textContent = DATA.plane_id;
  baseCount.textContent = DATA.base_count.toLocaleString();
  refinedCount.textContent = DATA.refined_count.toLocaleString();
  keptCount.textContent = DATA.kept_count.toLocaleString();
  changeCount.textContent = `${DATA.added_count.toLocaleString()} / ${DATA.removed_count.toLocaleString()}`;
  residualInfo.innerHTML = `base mean abs distance: <b>${fmt(DATA.base_residual,6)}</b><br>refined mean abs distance on refined support: <b>${fmt(DATA.refined_residual,6)}</b>`;
  equations.innerHTML = `base: ${eq(DATA.base_plane)}<br>refined: ${eq(DATA.refined_plane)}`;
  planes.innerHTML = DATA.plane_rows.map(p => `
    <div class="plane">
      <b>Plane ${p.id}</b>
      <div class="small">base ${p.base_count.toLocaleString()} · refined ${p.refined_count.toLocaleString()} · added ${p.added.toLocaleString()} · removed ${p.removed.toLocaleString()}</div>
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
function draw(canvas, mode) {
  resizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const pts = DATA.points.map((p, i) => [project(p, canvas), i]).sort((a,b) => a[0][2] - b[0][2]);
  const r = Math.max(1.0, 1.45 * devicePixelRatio);
  for (const [pp, i] of pts) {
    const state = DATA.states[i];
    let color = '#9ca3af', alpha = 0.24, rr = r;
    if (mode === 'base') {
      if (state === 'kept' || state === 'removed') { color = state === 'kept' ? '#16a34a' : '#dc2626'; alpha = 0.92; rr = r*1.25; }
    } else {
      if (state === 'kept' || state === 'added') { color = state === 'kept' ? '#16a34a' : '#2563eb'; alpha = 0.92; rr = r*1.25; }
    }
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.arc(pp[0], pp[1], rr, 0, Math.PI*2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}
function drawAll() { draw(before, 'base'); draw(after, 'refined'); }
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
attach(before); attach(after);
addEventListener('resize', drawAll);
drawAll();
</script>
</body>
</html>
"""


def load_npz(path):
    raw = np.load(path)
    return {
        "points": raw["points"].astype(np.float32),
        "point_plane_ids": raw["point_plane_ids"].astype(np.int32),
        "plane_ids": raw["plane_ids"].astype(np.int32),
        "plane_normals": raw["plane_normals"].astype(np.float32),
        "plane_offsets": raw["plane_offsets"].astype(np.float32),
    }


def select_plane(base, refined, edit_plane):
    if edit_plane == "largest":
        counts = {int(pid): int(np.sum(base["point_plane_ids"] == int(pid))) for pid in base["plane_ids"]}
        return max(counts, key=counts.get)
    return int(edit_plane)


def plane_by_id(data, plane_id):
    idx = np.where(data["plane_ids"] == int(plane_id))[0]
    if len(idx) != 1:
        raise ValueError(f"Cannot find plane {plane_id}")
    idx = int(idx[0])
    return {
        "id": int(plane_id),
        "normal": [float(x) for x in data["plane_normals"][idx]],
        "offset": float(data["plane_offsets"][idx]),
    }


def residual(points, mask, plane):
    if not np.any(mask):
        return None
    n = np.asarray(plane["normal"], dtype=np.float32)
    d = float(plane["offset"])
    return float(np.mean(np.abs(points[mask] @ n + d)))


def deterministic_sample(total, focus_mask, max_display_points):
    focus_idx = np.flatnonzero(focus_mask)
    base_n = max_display_points - min(len(focus_idx), max_display_points // 2)
    base_idx = np.linspace(0, total - 1, min(max(base_n, 0), total), dtype=np.int64)
    if len(focus_idx):
        focus_keep = focus_idx[np.linspace(0, len(focus_idx) - 1, min(len(focus_idx), max_display_points - len(base_idx)), dtype=np.int64)]
        idx = np.unique(np.concatenate([base_idx, focus_keep]))
    else:
        idx = base_idx
    if len(idx) > max_display_points:
        idx = idx[np.linspace(0, len(idx) - 1, max_display_points, dtype=np.int64)]
    return idx.astype(np.int64)


def main():
    parser = argparse.ArgumentParser("Visualize RANSAC vs learned plane binding masks")
    parser.add_argument("--base_npz", required=True)
    parser.add_argument("--refined_npz", required=True)
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--plane", default="largest")
    parser.add_argument("--max_display_points", type=int, default=30000)
    parser.add_argument("--summary_csv", default=None)
    args = parser.parse_args()

    base = load_npz(args.base_npz)
    refined = load_npz(args.refined_npz)
    if len(base["points"]) != len(refined["points"]):
        raise ValueError("Base and refined NPZ must contain aligned point arrays")
    plane_id = select_plane(base, refined, args.plane)
    base_mask = base["point_plane_ids"] == plane_id
    refined_mask = refined["point_plane_ids"] == plane_id
    kept = base_mask & refined_mask
    added = (~base_mask) & refined_mask
    removed = base_mask & (~refined_mask)
    focus = base_mask | refined_mask
    sample_idx = deterministic_sample(len(base["points"]), focus, args.max_display_points)
    states = []
    for i in sample_idx:
        if kept[i]:
            states.append("kept")
        elif added[i]:
            states.append("added")
        elif removed[i]:
            states.append("removed")
        else:
            states.append("other")

    mins = base["points"].min(axis=0)
    maxs = base["points"].max(axis=0)
    center = (mins + maxs) / 2.0
    scale = float(0.74 / max(float(np.max(maxs - mins)), 1e-6))
    base_plane = plane_by_id(base, plane_id)
    refined_plane = plane_by_id(refined, plane_id)
    plane_rows = []
    for pid in base["plane_ids"]:
        b = base["point_plane_ids"] == int(pid)
        r = refined["point_plane_ids"] == int(pid)
        plane_rows.append(
            {
                "id": int(pid),
                "base_count": int(b.sum()),
                "refined_count": int(r.sum()),
                "added": int((~b & r).sum()),
                "removed": int((b & ~r).sum()),
            }
        )
    data = {
        "plane_id": int(plane_id),
        "points": base["points"][sample_idx].round(5).tolist(),
        "states": states,
        "center": [float(x) for x in center],
        "scale": scale,
        "base_count": int(base_mask.sum()),
        "refined_count": int(refined_mask.sum()),
        "kept_count": int(kept.sum()),
        "added_count": int(added.sum()),
        "removed_count": int(removed.sum()),
        "base_residual": residual(base["points"], base_mask, base_plane),
        "refined_residual": residual(refined["points"], refined_mask, refined_plane),
        "base_plane": base_plane,
        "refined_plane": refined_plane,
        "plane_rows": plane_rows,
    }
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    if args.summary_csv:
        csv_path = Path(args.summary_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample",
                    "plane_id",
                    "base_count",
                    "refined_count",
                    "kept_count",
                    "added_count",
                    "removed_count",
                    "base_residual",
                    "refined_residual",
                ],
            )
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "sample": output_html.name.replace("_binding_refinement_comparison.html", ""),
                    "plane_id": int(plane_id),
                    "base_count": int(base_mask.sum()),
                    "refined_count": int(refined_mask.sum()),
                    "kept_count": int(kept.sum()),
                    "added_count": int(added.sum()),
                    "removed_count": int(removed.sum()),
                    "base_residual": data["base_residual"],
                    "refined_residual": data["refined_residual"],
                }
            )
    print(output_html)
    print(
        f"plane={plane_id} base={base_mask.sum()} refined={refined_mask.sum()} "
        f"added={added.sum()} removed={removed.sum()}"
    )


if __name__ == "__main__":
    main()
