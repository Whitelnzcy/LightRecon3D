#!/usr/bin/env bash
set -euo pipefail

PROJECT="/gemini/code/LightRecon3D"
DATA_ROOT="/gemini/data-1/Structured3D"
WEIGHTS="/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
TOP_UNIQUE="${TOP_UNIQUE:-6}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
MAX_IMAGES="${MAX_IMAGES:-10}"
MAX_POINTS="${MAX_POINTS:-800000}"

RUN_CALIB="${RUN_CALIB:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_ROOT="/gemini/data-1/important_results/badcase_archive_${TS}"
MINING_DIR="${ARCHIVE_ROOT}/00_failure_mining"
CASES_DIR="${ARCHIVE_ROOT}/cases"

mkdir -p "${ARCHIVE_ROOT}" "${MINING_DIR}" "${CASES_DIR}"

cd "${PROJECT}"

set +u
source "${PROJECT}/scripts/activate_gemini_env.sh"
set -u

echo "============================================================"
echo "Badcase Archive Run"
echo "============================================================"
echo "PROJECT      : ${PROJECT}"
echo "DATA_ROOT    : ${DATA_ROOT}"
echo "WEIGHTS      : ${WEIGHTS}"
echo "GPU          : ${GPU}"
echo "NUM_SAMPLES  : ${NUM_SAMPLES}"
echo "TOP_UNIQUE   : ${TOP_UNIQUE}"
echo "ARCHIVE_ROOT : ${ARCHIVE_ROOT}"
echo "RUN_CALIB    : ${RUN_CALIB}"
echo "============================================================"

cat > "${ARCHIVE_ROOT}/README.md" <<EOF
# Badcase Archive

Created: ${TS}

Purpose:
- Re-run baseline failure mining.
- Archive top bad scenes.
- Save input images, candidate panels, raw DUSt3R multi-view point clouds, and logs.
- This run is for diagnosis only. It does not claim improvement.

Paths:
- Project: \`${PROJECT}\`
- Dataset: \`${DATA_ROOT}\`
- DUSt3R weights: \`${WEIGHTS}\`

Main files:
- \`00_failure_mining/candidates.csv\`
- \`00_failure_mining/top_panels/\`
- \`top_cases.tsv\`
- \`cases/*/input_images/\`
- \`cases/*/recon/*.ply\`
- \`cases/*/logs/\`
EOF

echo "============================================================"
echo "[1/3] Run baseline failure mining"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU}" python baseline_failure_mining.py \
  --root_dir "${DATA_ROOT}" \
  --weights_path "${WEIGHTS}" \
  --output_dir "${MINING_DIR}" \
  --split val \
  --num_samples "${NUM_SAMPLES}" \
  --image_size "${IMAGE_SIZE}" \
  --conf_thr 1.5 \
  --min_plane_pixels 5000 \
  --planar_dist_ratio 0.006 \
  --top_k 40 \
  --save_top_panels 40 \
  2>&1 | tee "${ARCHIVE_ROOT}/00_failure_mining.log"

echo "============================================================"
echo "[2/3] Select top unique bad scenes"
echo "============================================================"

python - "${MINING_DIR}/candidates.csv" "${ARCHIVE_ROOT}/top_cases.tsv" "${TOP_UNIQUE}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
out_tsv = Path(sys.argv[2])
top_unique = int(sys.argv[3])

rows = []
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            r["_score"] = float(r["score"])
        except Exception:
            r["_score"] = -1.0
        rows.append(r)

rows.sort(key=lambda x: x["_score"], reverse=True)

selected = []
seen_base = set()

def parse_scene_render(rgb_path):
    p = Path(rgb_path)
    parts = p.parts

    scene = "unknown_scene"
    render = "unknown_render"

    if "Structured3D" in parts:
        i = parts.index("Structured3D")
        if i + 1 < len(parts):
            scene = parts[i + 1]

    if "2D_rendering" in parts:
        i = parts.index("2D_rendering")
        if i + 1 < len(parts):
            render = parts[i + 1]

    base_dir = str(p.parent.parent)
    return scene, render, base_dir

for r in rows:
    rgb_path = r.get("rgb_path", "")
    if not rgb_path:
        continue

    scene, render, base_dir = parse_scene_render(rgb_path)
    key = base_dir

    if key in seen_base:
        continue

    seen_base.add(key)
    selected.append({
        "rank": len(selected) + 1,
        "sample_idx": r.get("sample_idx", ""),
        "plane_id": r.get("plane_id", ""),
        "score": r.get("score", ""),
        "scene": scene,
        "render": render,
        "base_dir": base_dir,
        "rgb_path": rgb_path,
        "flatness_p90_norm": r.get("flatness_p90_norm", ""),
        "high_conf_ratio": r.get("high_conf_ratio", ""),
        "low_conf_but_planar_ratio": r.get("low_conf_but_planar_ratio", ""),
    })

    if len(selected) >= top_unique:
        break

with open(out_tsv, "w", encoding="utf-8") as f:
    for r in selected:
        fields = [
            str(r["rank"]),
            str(r["sample_idx"]),
            str(r["plane_id"]),
            str(r["score"]),
            str(r["scene"]),
            str(r["render"]),
            str(r["base_dir"]),
            str(r["rgb_path"]),
            str(r["flatness_p90_norm"]),
            str(r["high_conf_ratio"]),
            str(r["low_conf_but_planar_ratio"]),
        ]
        f.write("\t".join(fields) + "\n")

print("Selected cases:")
for r in selected:
    print(
        f"rank={r['rank']} "
        f"scene={r['scene']} render={r['render']} "
        f"sample={r['sample_idx']} plane={r['plane_id']} "
        f"score={r['score']} "
        f"low_planar={r['low_conf_but_planar_ratio']} "
        f"high_conf={r['high_conf_ratio']} "
        f"base={r['base_dir']}"
    )
PY

echo "Saved top cases:"
cat "${ARCHIVE_ROOT}/top_cases.tsv"

echo "============================================================"
echo "[3/3] Re-run raw DUSt3R multi-view for top cases"
echo "============================================================"

while IFS=$'\t' read -r RANK SAMPLE_IDX PLANE_ID SCORE SCENE RENDER BASE_DIR RGB_PATH P90N HIGHCONF LOWPLANAR; do
  SCORE_TAG="$(printf "%.3f" "${SCORE}" | sed 's/\./p/g')"
  CASE_NAME="$(printf "case_%02d_%s_%s_idx%s_plane%s_score%s" "${RANK}" "${SCENE}" "${RENDER}" "${SAMPLE_IDX}" "${PLANE_ID}" "${SCORE_TAG}")"
  CASE_DIR="${CASES_DIR}/${CASE_NAME}"

  mkdir -p "${CASE_DIR}/input_images" "${CASE_DIR}/candidate_panels" "${CASE_DIR}/recon" "${CASE_DIR}/logs"

  echo "------------------------------------------------------------"
  echo "Case ${RANK}: ${CASE_NAME}"
  echo "BASE_DIR: ${BASE_DIR}"
  echo "------------------------------------------------------------"

  mapfile -t IMGS < <(find "${BASE_DIR}" -maxdepth 2 -name "rgb_rawlight.png" | sort -V)

  if [[ "${#IMGS[@]}" -lt 2 ]]; then
    echo "[Skip] less than 2 images in ${BASE_DIR}" | tee "${CASE_DIR}/logs/skip.log"
    continue
  fi

  if [[ "${#IMGS[@]}" -gt "${MAX_IMAGES}" ]]; then
    IMGS=("${IMGS[@]:0:${MAX_IMAGES}}")
  fi

  : > "${CASE_DIR}/used_images.txt"

  for i in "${!IMGS[@]}"; do
    SRC="${IMGS[$i]}"
    DST="${CASE_DIR}/input_images/view$(printf "%02d" "${i}")_rgb_rawlight.png"
    cp "${SRC}" "${DST}"
    echo "${SRC}" >> "${CASE_DIR}/used_images.txt"
  done

  find "${MINING_DIR}/top_panels" -maxdepth 1 -type f -name "*idx${SAMPLE_IDX}_*.png" -exec cp {} "${CASE_DIR}/candidate_panels/" \; || true

  cat > "${CASE_DIR}/case_info.txt" <<EOF
case_name: ${CASE_NAME}
rank: ${RANK}
sample_idx: ${SAMPLE_IDX}
plane_id: ${PLANE_ID}
score: ${SCORE}
scene: ${SCENE}
render: ${RENDER}
base_dir: ${BASE_DIR}
rgb_path: ${RGB_PATH}
flatness_p90_norm: ${P90N}
high_conf_ratio: ${HIGHCONF}
low_conf_but_planar_ratio: ${LOWPLANAR}
num_images: ${#IMGS[@]}
EOF

  python - "${CASE_DIR}/input_images" "${CASE_DIR}/input_contact_sheet.png" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageDraw

img_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])

paths = sorted(img_dir.glob("*.png"))
if not paths:
    raise SystemExit

thumb_w, thumb_h = 240, 240
pad = 20
label_h = 30
cols = min(5, len(paths))
rows = (len(paths) + cols - 1) // cols

canvas = Image.new("RGB", (cols * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad), "white")
draw = ImageDraw.Draw(canvas)

for idx, p in enumerate(paths):
    img = Image.open(p).convert("RGB")
    img.thumbnail((thumb_w, thumb_h))

    x = pad + (idx % cols) * (thumb_w + pad)
    y = pad + (idx // cols) * (thumb_h + label_h + pad)

    canvas.paste(img, (x, y))
    draw.text((x, y + thumb_h + 5), p.name, fill=(0, 0, 0))

canvas.save(out_path)
PY

  echo "[Run] raw DUSt3R baseline none"

  CUDA_VISIBLE_DEVICES="${GPU}" python "${PROJECT}/run_raw_dust3r_multiview.py" \
    --weights_path "${WEIGHTS}" \
    --image_paths "${IMGS[@]}" \
    --output_dir "${CASE_DIR}/recon" \
    --run_tag "${CASE_NAME}_raw_none" \
    --image_size "${IMAGE_SIZE}" \
    --max_images "${MAX_IMAGES}" \
    --scene_graph complete \
    --batch_size 1 \
    --niter 300 \
    --lr 0.01 \
    --schedule cosine \
    --max_points "${MAX_POINTS}" \
    --conf_reweight none \
    --export_thresholds 0.0 1.0 1.5 2.0 \
    2>&1 | tee "${CASE_DIR}/logs/run_raw_none.log"

  if [[ "${RUN_CALIB}" == "1" ]]; then
    echo "[Run] plane confidence calibration diagnostic"

    CUDA_VISIBLE_DEVICES="${GPU}" python "${PROJECT}/run_raw_dust3r_multiview.py" \
      --weights_path "${WEIGHTS}" \
      --image_paths "${IMGS[@]}" \
      --output_dir "${CASE_DIR}/recon" \
      --run_tag "${CASE_NAME}_plane_conf_calib" \
      --image_size "${IMAGE_SIZE}" \
      --max_images "${MAX_IMAGES}" \
      --scene_graph complete \
      --batch_size 1 \
      --niter 300 \
      --lr 0.01 \
      --schedule cosine \
      --max_points "${MAX_POINTS}" \
      --conf_reweight plane_conf_calib \
      --export_thresholds 0.0 1.0 1.5 2.0 \
      --plane_seed_quantile 0.70 \
      --plane_dist_ratio 0.006 \
      --plane_near_scale 1.5 \
      --plane_floor 0.90 \
      --plane_blend_alpha 0.80 \
      --plane_max_planes 5 \
      --plane_num_iters 120 \
      --plane_max_seed_points 20000 \
      --plane_min_inliers 1200 \
      --plane_exclude_jump_quantile 0.985 \
      --plane_exclude_jump_dilate 1 \
      --save_conf_calib_debug \
      2>&1 | tee "${CASE_DIR}/logs/run_plane_conf_calib.log"
  fi

done < "${ARCHIVE_ROOT}/top_cases.tsv"

echo "============================================================"
echo "Archive finished"
echo "============================================================"
echo "ARCHIVE_ROOT=${ARCHIVE_ROOT}"
echo ""
echo "Quick view:"
find "${ARCHIVE_ROOT}" -maxdepth 3 -type f \( -name "*.png" -o -name "*.ply" -o -name "*.csv" -o -name "*.tsv" -o -name "*.log" -o -name "*.txt" \) | sort | head -n 200