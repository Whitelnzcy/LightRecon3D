#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
ROOT_DIR="${ROOT_DIR:-/gemini/data-1/Structured3D}"
EXISTING_ROOT="${EXISTING_ROOT:-/gemini/data-1/lightrecon_runs/stage3_val_showcase_v1}"
SMOKE_BATCH_DIR="${SMOKE_BATCH_DIR:-/gemini/data-1/lightrecon_runs/research_practice_batch_smoke_three_groups_20260715_v1}"
EXPANSION_ROOT="${EXPANSION_ROOT:-/gemini/data-1/lightrecon_runs/research_practice_final_unique_scene_inputs_20260716_v1}"
OUT_DIR="${OUT_DIR:-/gemini/data-1/lightrecon_runs/research_practice_final_selection_20260716_v1}"
TARGET_SCENES="${TARGET_SCENES:-8}"
MIN_PAIRS="${MIN_PAIRS:-10}"
RUN_LOG="${OUT_DIR}_launcher.log"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable project Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import cv2, numpy, torch' >/dev/null 2>&1; then
  echo "Selected lightrecon Python is missing cv2, NumPy or torch: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import roma' >/dev/null 2>&1; then
  echo "roma is missing; installing roma==1.5.6 into ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install --no-deps roma==1.5.6
fi
for required in \
  "${ROOT_DIR}" \
  "${EXISTING_ROOT}/selected_groups.tsv" \
  "${SMOKE_BATCH_DIR}/batch_execution.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required selection input: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
  exit 2
fi
if [[ -e "${RUN_LOG}" ]]; then
  echo "Refusing to overwrite existing launcher log: ${RUN_LOG}" >&2
  exit 2
fi

exec > >(tee "${RUN_LOG}") 2>&1

echo "LightRecon3D final independent-scene selection preflight"
echo "project=${PROJECT_DIR}"
echo "python=${PYTHON_BIN}"
echo "dataset=${ROOT_DIR}"
echo "existing_stage2=${EXISTING_ROOT}"
echo "smoke_batch=${SMOKE_BATCH_DIR}"
echo "planned_expansion=${EXPANSION_ROOT}"
echo "output=${OUT_DIR}"
echo "target_scenes=${TARGET_SCENES}"
echo "git_sha=$(git rev-parse HEAD)"
"${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version
import cv2
import numpy
import torch

print(f"torch={torch.__version__}")
print(f"numpy={numpy.__version__}")
print(f"cv2={cv2.__version__}")
print(f"roma={version('roma')}")
print(f"cuda_used_by_this_command=False")
PY

"${PYTHON_BIN}" prepare_research_practice_final_manifest.py \
  --root_dir "${ROOT_DIR}" \
  --existing_root "${EXISTING_ROOT}" \
  --expansion_root "${EXPANSION_ROOT}" \
  --reuse_batch_execution_json "${SMOKE_BATCH_DIR}/batch_execution.json" \
  --target_scenes "${TARGET_SCENES}" \
  --min_pairs "${MIN_PAIRS}" \
  --split val \
  --train_ratio 0.9 \
  --pair_strategy all \
  --output_dir "${OUT_DIR}"

echo "Final selection preflight complete: ${OUT_DIR}"
cat "${OUT_DIR}/selection_plan.md"
