import argparse
import json
import re
from pathlib import Path


def extract_data(input_html):
    text = Path(input_html).read_text(encoding="utf-8")
    match = re.search(r"const DATA = (.*?);\n(?:DATA\.planeById|const bar)", text, re.S)
    if not match:
        raise RuntimeError(f"Cannot find DATA block in {input_html}")
    data = json.loads(match.group(1))
    counts = {str(p["id"]): 0 for p in data["planes"]}
    counts["-1"] = 0
    for pid in data["point_plane_ids"]:
        counts[str(pid)] = counts.get(str(pid), 0) + 1
    for plane in data["planes"]:
        plane["assigned_display_points"] = counts.get(str(plane["id"]), 0)
    return data


def build_html(data):
    data_json = json.dumps(data)
    total_points = len(data["points"])
    plane_count = len(data["planes"])
    assigned = sum(1 for pid in data["point_plane_ids"] if pid >= 0)
    unassigned = total_points - assigned
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>Editable Structured Planes Demo</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: Arial, "Microsoft YaHei", sans-serif; background:#f4f4f1; color:#202124; }}
.app {{ display:grid; grid-template-columns:minmax(0,1fr) 390px; height:100vh; }}
.stage {{ position:relative; min-width:0; }}
canvas {{ width:100%; height:100%; display:block; background:#f6f6f4; }}
.panel {{ border-left:1px solid #d5d5d0; background:#fff; padding:16px 16px 18px; overflow:auto; }}
h1 {{ font-size:18px; margin:0 0 6px; font-weight:700; }}
.sub {{ color:#5f6368; font-size:12px; line-height:1.45; margin-bottom:14px; }}
.summary {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:14px; }}
.stat {{ border:1px solid #ddd; background:#fafafa; padding:8px; border-radius:6px; }}
.stat b {{ display:block; font-size:17px; }}
.stat span {{ color:#666; font-size:12px; }}
.toolrow {{ display:flex; gap:8px; margin:10px 0 14px; align-items:center; flex-wrap:wrap; }}
button {{ border:1px solid #c7c7c7; background:#f8f8f8; padding:6px 9px; border-radius:5px; cursor:pointer; font-size:12px; }}
button.active {{ background:#202124; color:white; border-color:#202124; }}
.card {{ border:1px solid #d6d6d6; border-radius:7px; padding:10px; margin:10px 0; background:#fff; }}
.card.selected {{ outline:2px solid #202124; }}
.cardHead {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
.badge {{ display:inline-flex; align-items:center; gap:6px; font-size:12px; }}
.swatch {{ width:12px; height:12px; display:inline-block; border-radius:2px; border:1px solid rgba(0,0,0,.2); }}
.eq {{ font-family:Consolas, monospace; font-size:12px; background:#f7f7f7; border:1px solid #e2e2e2; padding:7px; border-radius:5px; word-break:break-all; line-height:1.35; }}
.meta {{ display:grid; grid-template-columns:1fr 1fr; gap:6px; color:#555; font-size:12px; margin:8px 0; }}
.sliderRow {{ display:grid; grid-template-columns:44px 1fr 48px; gap:8px; align-items:center; }}
input[type=range] {{ width:100%; }}
.overlay {{ position:absolute; top:12px; left:12px; background:rgba(255,255,255,.88); border:1px solid #ddd; border-radius:6px; padding:8px 10px; font-size:12px; color:#333; max-width:460px; }}
.overlay b {{ font-size:13px; }}
</style>
</head>
<body>
<div class="app">
  <div class="stage">
    <canvas id="canvas"></canvas>
    <div class="overlay"><b>完整点云结构面编辑</b><br>拖动画布旋转；右侧选择 plane 并调整 offset d。被选中/归属的点会跟随平面方程沿法线移动。</div>
  </div>
  <aside class="panel">
    <h1>主要平面参数方程</h1>
    <div class="sub">每个平面采用 <b>n·x + d = 0</b>。修改 offset d 时，平面 mesh 和归属点云一起沿法线方向移动。</div>
    <div class="summary">
      <div class="stat"><b>{total_points}</b><span>display points</span></div>
      <div class="stat"><b>{plane_count}</b><span>major planes</span></div>
      <div class="stat"><b>{assigned}</b><span>assigned points</span></div>
      <div class="stat"><b>{unassigned}</b><span>unassigned points</span></div>
    </div>
    <div class="toolrow">
      <button id="showAll" class="active">全部点云</button>
      <button id="focusSelected">只看选中面</button>
      <button id="resetAll">重置 offset</button>
    </div>
    <div id="cards"></div>
  </aside>
</div>
<script>
const DATA = {data_json};
DATA.planeById = Object.fromEntries(DATA.planes.map(p => [p.id, p]));
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const cards = document.getElementById('cards');
let yaw = -0.78, pitch = 0.55, dragging = false, lastX = 0, lastY = 0;
let selectedPlane = DATA.planes[0]?.id ?? -1;
let focusOnly = false;
const offsets = Object.fromEntries(DATA.planes.map(p => [p.id, 0]));
function fmt(x, n=3) {{ return Number(x).toFixed(n); }}
function equation(pl) {{
  const n = pl.normal;
  return `${{fmt(n[0])}}x + ${{fmt(n[1])}}y + ${{fmt(n[2])}}z + ${{fmt(pl.offset + offsets[pl.id])}} = 0`;
}}
function renderCards() {{
  cards.innerHTML = '';
  for (const pl of DATA.planes) {{
    const c = pl.color;
    const div = document.createElement('div');
    div.className = 'card' + (pl.id === selectedPlane ? ' selected' : '');
    div.innerHTML = `
      <div class="cardHead">
        <span class="badge"><span class="swatch" style="background:rgb(${{c[0]}},${{c[1]}},${{c[2]}})"></span><b>plane ${{pl.id}}</b></span>
        <button data-select="${{pl.id}}">选择</button>
      </div>
      <div class="eq" id="eq_${{pl.id}}">${{equation(pl)}}</div>
      <div class="meta">
        <span>inliers: ${{pl.inlier_count}}</span>
        <span>shown pts: ${{pl.assigned_display_points}}</span>
        <span>n: (${{pl.normal.map(v=>fmt(v,2)).join(', ')}})</span>
        <span>d0: ${{fmt(pl.offset)}}</span>
      </div>
      <div class="sliderRow"><span>Δd</span><input data-slider="${{pl.id}}" type="range" min="-0.45" max="0.45" step="0.01" value="${{offsets[pl.id]}}"><span id="delta_${{pl.id}}">${{fmt(offsets[pl.id],2)}}</span></div>
    `;
    cards.appendChild(div);
  }}
  cards.querySelectorAll('[data-select]').forEach(btn => btn.addEventListener('click', () => {{ selectedPlane = Number(btn.dataset.select); renderCards(); draw(); }}));
  cards.querySelectorAll('[data-slider]').forEach(sl => sl.addEventListener('input', () => {{ offsets[Number(sl.dataset.slider)] = Number(sl.value); updateEquations(); draw(); }}));
}}
function updateEquations() {{
  for (const pl of DATA.planes) {{
    const eq = document.getElementById(`eq_${{pl.id}}`);
    const de = document.getElementById(`delta_${{pl.id}}`);
    if (eq) eq.textContent = equation(pl);
    if (de) de.textContent = fmt(offsets[pl.id], 2);
  }}
}}
document.getElementById('showAll').onclick = () => {{ focusOnly=false; document.getElementById('showAll').classList.add('active'); document.getElementById('focusSelected').classList.remove('active'); draw(); }};
document.getElementById('focusSelected').onclick = () => {{ focusOnly=true; document.getElementById('focusSelected').classList.add('active'); document.getElementById('showAll').classList.remove('active'); draw(); }};
document.getElementById('resetAll').onclick = () => {{ for (const pl of DATA.planes) offsets[pl.id]=0; renderCards(); draw(); }};
function resize() {{ canvas.width = canvas.clientWidth * devicePixelRatio; canvas.height = canvas.clientHeight * devicePixelRatio; draw(); }}
addEventListener('resize', resize);
canvas.addEventListener('mousedown', e => {{ dragging=true; lastX=e.clientX; lastY=e.clientY; }});
addEventListener('mouseup', () => dragging=false);
addEventListener('mousemove', e => {{ if(!dragging) return; yaw += (e.clientX-lastX)*0.008; pitch += (e.clientY-lastY)*0.008; pitch=Math.max(-1.45,Math.min(1.45,pitch)); lastX=e.clientX; lastY=e.clientY; draw(); }});
function rot(p) {{ let [x,y,z]=p; const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch); const x1=cy*x+sy*z, z1=-sy*x+cy*z; const y1=cp*y-sp*z1, z2=sp*y+cp*z1; return [x1,y1,z2]; }}
function project(p) {{ const r=rot(p); const s=Math.min(canvas.width,canvas.height)*DATA.scale; return [canvas.width/2+r[0]*s, canvas.height/2-r[1]*s, r[2]]; }}
function editedPoint(i) {{
  const pid = DATA.point_plane_ids[i];
  const p = DATA.points[i];
  if (pid < 0) return p;
  const pl = DATA.planeById[pid];
  const d = offsets[pid] ?? 0;
  return p.map((x,j)=>x-d*pl.normal[j]);
}}
function planeVerts(pl, moved=true) {{
  const d = moved ? offsets[pl.id] : 0;
  const n=pl.normal, u=pl.u, v=pl.v, e=pl.extent;
  const center = pl.center.map((x,i)=>x-d*n[i]);
  return [[e[0],e[2]],[e[1],e[2]],[e[1],e[3]],[e[0],e[3]]].map(([a,b])=>center.map((x,i)=>x+a*u[i]+b*v[i]));
}}
function drawPoly(verts, rgba, line='rgba(0,0,0,.38)') {{
  const pts = verts.map(project);
  ctx.beginPath(); ctx.moveTo(pts[0][0],pts[0][1]);
  for(let i=1;i<pts.length;i++) ctx.lineTo(pts[i][0],pts[i][1]);
  ctx.closePath(); ctx.fillStyle=rgba; ctx.strokeStyle=line; ctx.lineWidth=1.1*devicePixelRatio; ctx.fill(); ctx.stroke();
}}
function draw() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#f6f6f4'; ctx.fillRect(0,0,canvas.width,canvas.height);
  const items=[];
  for(let i=0;i<DATA.points.length;i++) {{
    const pid = DATA.point_plane_ids[i];
    if(focusOnly && pid !== selectedPlane) continue;
    const pr = project(editedPoint(i));
    items.push([pr[2], pr, DATA.colors[i], pid]);
  }}
  items.sort((a,b)=>a[0]-b[0]);
  for(const it of items) {{
    const c = it[2];
    const isSel = it[3] === selectedPlane;
    const alpha = it[3] < 0 ? 0.18 : (isSel ? 0.86 : 0.35);
    const size = isSel ? 1.8 : 1.25;
    ctx.fillStyle = `rgba(${{c[0]}},${{c[1]}},${{c[2]}},${{alpha}})`;
    ctx.fillRect(it[1][0], it[1][1], size*devicePixelRatio, size*devicePixelRatio);
  }}
  for(const pl of DATA.planes) {{
    const c = pl.color;
    const alpha = pl.id === selectedPlane ? 0.42 : 0.20;
    drawPoly(planeVerts(pl,true), `rgba(${{c[0]}},${{c[1]}},${{c[2]}},${{alpha}})`);
  }}
  const sel = DATA.planeById[selectedPlane];
  if(sel) drawPoly(planeVerts(sel,false), 'rgba(255,255,255,0.06)', 'rgba(0,0,0,0.75)');
}}
renderCards();
resize();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_html", required=True)
    parser.add_argument("--output_html", required=True)
    args = parser.parse_args()
    data = extract_data(args.input_html)
    Path(args.output_html).write_text(build_html(data), encoding="utf-8")
    assigned = sum(1 for pid in data["point_plane_ids"] if pid >= 0)
    print(args.output_html)
    print(f"planes={len(data['planes'])} points={len(data['points'])} assigned={assigned}")


if __name__ == "__main__":
    main()
