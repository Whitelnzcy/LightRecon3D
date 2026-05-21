#!/usr/bin/env bash
set -euo pipefail

PROJECT="/gemini/code/LightRecon3D"
WEIGHTS="/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

IMAGE_SIZE="${IMAGE_SIZE:-512}"
NITER="${NITER:-300}"
LR="${LR:-0.01}"
MAX_POINTS="${MAX_POINTS:-800000}"

ARCHIVE_ROOT="$(ls -td /gemini/data-1/important_results/badcase_archive_* | head -n 1)"
CASE_DIR="$(find "${ARCHIVE_ROOT}/cases" -maxdepth 1 -type d -name "case_01_*" | head -n 1)"

if [[ -z "${CASE_DIR}" ]]; then
  echo "[Error] Cannot find case_01_* under ${ARCHIVE_ROOT}/cases"
  exit 1
fi

AUDIT_DIR="${CASE_DIR}/pairwise_audit"
mkdir -p "${AUDIT_DIR}"

cd "${PROJECT}"
source "${PROJECT}/scripts/activate_gemini_env.sh"

echo "============================================================"
echo "Pairwise Audit for Case 01"
echo "============================================================"
echo "ARCHIVE_ROOT : ${ARCHIVE_ROOT}"
echo "CASE_DIR     : ${CASE_DIR}"
echo "AUDIT_DIR    : ${AUDIT_DIR}"
echo "GPU          : ${GPU}"
echo "============================================================"

mapfile -t IMGS < "${CASE_DIR}/used_images.txt"

echo "Input images:"
for i in "${!IMGS[@]}"; do
  echo "view${i}: ${IMGS[$i]}"
done

NUM=${#IMGS[@]}
if [[ "${NUM}" -lt 2 ]]; then
  echo "[Error] Need at least 2 images."
  exit 1
fi

echo "============================================================"
echo "[1/3] Run all pairwise DUSt3R reconstructions"
echo "============================================================"

for ((i=0; i<NUM; i++)); do
  for ((j=i+1; j<NUM; j++)); do
    PAIR_NAME="pair_v${i}_v${j}"
    OUT_DIR="${AUDIT_DIR}/${PAIR_NAME}"
    mkdir -p "${OUT_DIR}"

    echo "------------------------------------------------------------"
    echo "[Run] ${PAIR_NAME}"
    echo "img_i: ${IMGS[$i]}"
    echo "img_j: ${IMGS[$j]}"
    echo "------------------------------------------------------------"

    cat > "${OUT_DIR}/pair_info.txt" <<EOF
pair_name: ${PAIR_NAME}
view_i: ${i}
view_j: ${j}
img_i: ${IMGS[$i]}
img_j: ${IMGS[$j]}
EOF

    CUDA_VISIBLE_DEVICES="${GPU}" python "${PROJECT}/run_raw_dust3r_multiview.py" \
      --weights_path "${WEIGHTS}" \
      --image_paths "${IMGS[$i]}" "${IMGS[$j]}" \
      --output_dir "${OUT_DIR}" \
      --run_tag "${PAIR_NAME}_raw_none" \
      --image_size "${IMAGE_SIZE}" \
      --max_images 2 \
      --scene_graph complete \
      --batch_size 1 \
      --niter "${NITER}" \
      --lr "${LR}" \
      --schedule cosine \
      --max_points "${MAX_POINTS}" \
      --conf_reweight none \
      --export_thresholds 0.0 1.0 1.5 2.0 \
      2>&1 | tee "${OUT_DIR}/run_pair.log"
  done
done

echo "============================================================"
echo "[2/3] Summarize pairwise logs"
echo "============================================================"

python - <<'PY'
import csv
import re
from pathlib import Path

archive_root = Path("/gemini/data-1/important_results")
latest = sorted(archive_root.glob("badcase_archive_*"), key=lambda p: p.stat().st_mtime, reverse=True)[0]
case_dir = sorted((latest / "cases").glob("case_01_*"))[0]
audit_dir = case_dir / "pairwise_audit"

rows = []

for pair_dir in sorted(audit_dir.glob("pair_v*_v*")):
    log_path = pair_dir / "run_pair.log"
    if not log_path.exists():
        continue

    text = log_path.read_text(errors="ignore")

    final_loss = ""
    m = re.search(r"final loss:\s*([0-9.eE+-]+)", text)
    if m:
        final_loss = m.group(1)

    saved_blocks = []
    lines = text.splitlines()
    for k, line in enumerate(lines):
        if "[Saved]" in line and line.strip().endswith(".ply"):
            ply_path = line.split("[Saved]", 1)[1].strip()
            points = ""
            if k + 1 < len(lines):
                m2 = re.search(r"\[Saved\]\s*points:\s*(\d+)", lines[k + 1])
                if m2:
                    points = m2.group(1)

            conf = ""
            m3 = re.search(r"conf([0-9.]+)\.ply", ply_path)
            if m3:
                conf = m3.group(1)

            saved_blocks.append((conf, points, ply_path))

    if not saved_blocks:
        rows.append({
            "pair": pair_dir.name,
            "conf": "",
            "points": "",
            "final_loss": final_loss,
            "ply_path": "",
            "status": "no_saved_ply",
        })
    else:
        for conf, points, ply_path in saved_blocks:
            rows.append({
                "pair": pair_dir.name,
                "conf": conf,
                "points": points,
                "final_loss": final_loss,
                "ply_path": ply_path,
                "status": "ok",
            })

out_csv = audit_dir / "pairwise_summary.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["pair", "conf", "points", "final_loss", "status", "ply_path"],
    )
    writer.writeheader()
    writer.writerows(rows)

print("saved:", out_csv)
print()
for r in rows:
    print(
        f"{r['pair']:12s} conf={r['conf']:>4s} "
        f"points={r['points']:>8s} loss={r['final_loss']} {r['status']}"
    )
PY

echo "============================================================"
echo "[3/3] Create lightweight zip for upload"
echo "============================================================"

python - <<'PY'
import os
import zipfile
from pathlib import Path

archive_root = sorted(
    Path("/gemini/data-1/important_results").glob("badcase_archive_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)[0]

case_dir = sorted((archive_root / "cases").glob("case_01_*"))[0]

out = Path("/gemini/data-1/important_results/case01_pairwise_light_report.zip")
include_suffix = {".md", ".tsv", ".csv", ".txt", ".log", ".png", ".jpg", ".jpeg"}
exclude_suffix = {".ply", ".pth", ".pt", ".npy", ".npz"}

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    # 总表
    for p in [
        archive_root / "README.md",
        archive_root / "top_cases.tsv",
        archive_root / "00_failure_mining" / "candidates.csv",
    ]:
        if p.exists():
            z.write(p, p.relative_to(archive_root).as_posix())

    # case01 相关文件，不含 ply
    for p in case_dir.rglob("*"):
        if not p.is_file():
            continue

        suffix = p.suffix.lower()
        if suffix in exclude_suffix:
            continue
        if suffix not in include_suffix:
            continue

        z.write(p, p.relative_to(archive_root).as_posix())

print("saved:", out)
print("size MB:", out.stat().st_size / 1024 / 1024)
PY

echo "============================================================"
echo "Done"
echo "============================================================"
echo "Upload this file:"
echo "/gemini/data-1/important_results/case01_pairwise_light_report.zip"