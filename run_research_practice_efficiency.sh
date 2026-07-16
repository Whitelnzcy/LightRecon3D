#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
RUNROOT="${RUNROOT:-/gemini/data-1/lightrecon_runs}"
ROOT_DIR="${ROOT_DIR:-/gemini/data-1/Structured3D}"
SELECTION_DIR="${SELECTION_DIR:-${RUNROOT}/research_practice_final_selection_20260716_v1}"
FINAL_BATCH_DIR="${FINAL_BATCH_DIR:-${RUNROOT}/research_practice_final_eight_scenes_20260716_v1}"
FINAL_AUDIT_DIR="${FINAL_AUDIT_DIR:-${FINAL_BATCH_DIR}_audit_v1}"

WEIGHTS_PATH="${WEIGHTS_PATH:-/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${RUNROOT}/checkpoints/stage1_large_80m_train2048_sharded_v1/best.pt}"
FEATURE_CACHE_PATH="${FEATURE_CACHE_PATH:-${RUNROOT}/cache/stage1_train2048_shards/val128.pt}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${RUNROOT}/stage2_region_merge_v1/checkpoint/best.pt}"

OUT_DIR="${OUT_DIR:-${RUNROOT}/research_practice_efficiency_20260716_v1}"
RUN_LOG="${OUT_DIR}_run.log"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable project Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import cv2, numpy, scipy, torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  echo "The fixed lightrecon Python must provide cv2, NumPy, SciPy, torch and CUDA." >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import roma' >/dev/null 2>&1; then
  echo "roma is missing; installing roma==1.5.6 into ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install --no-deps roma==1.5.6
fi

FINAL_MANIFEST="${SELECTION_DIR}/final_unique_scenes_execute.json"
SELECTION_PLAN="${SELECTION_DIR}/selection_plan.json"
BATCH_EXECUTION_JSON="${FINAL_BATCH_DIR}/batch_execution.json"
FINAL_AUDIT_JSON="${FINAL_AUDIT_DIR}/final_method_audit.json"
for required in \
  "${ROOT_DIR}" \
  "${FINAL_MANIFEST}" \
  "${SELECTION_PLAN}" \
  "${BATCH_EXECUTION_JSON}" \
  "${FINAL_AUDIT_JSON}" \
  "${WEIGHTS_PATH}" \
  "${STAGE1_CHECKPOINT}" \
  "${FEATURE_CACHE_PATH}" \
  "${STAGE2_CHECKPOINT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing final efficiency input: ${required}" >&2
    exit 2
  fi
done
for output in "${OUT_DIR}" "${RUN_LOG}"; do
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite existing efficiency output: ${output}" >&2
    exit 2
  fi
done

exec > >(tee "${RUN_LOG}") 2>&1

GIT_SHA="$(git rev-parse HEAD)"
echo "LightRecon3D final research-practice efficiency benchmark"
echo "project=${PROJECT_DIR}"
echo "python=${PYTHON_BIN}"
echo "final_batch=${FINAL_BATCH_DIR}"
echo "final_audit=${FINAL_AUDIT_DIR}"
echo "output=${OUT_DIR}"
echo "git_sha=${GIT_SHA}"
echo "frozen_resolution=512x512"
"${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version
import cv2
import numpy
import scipy
import torch

print(f"torch={torch.__version__}")
print(f"numpy={numpy.__version__}")
print(f"scipy={scipy.__version__}")
print(f"cv2={cv2.__version__}")
print(f"roma={version('roma')}")
PY
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

"${PYTHON_BIN}" benchmark_research_practice_efficiency.py \
  --final_manifest "${FINAL_MANIFEST}" \
  --selection_plan "${SELECTION_PLAN}" \
  --batch_execution_json "${BATCH_EXECUTION_JSON}" \
  --final_audit_json "${FINAL_AUDIT_JSON}" \
  --root_dir "${ROOT_DIR}" \
  --weights_path "${WEIGHTS_PATH}" \
  --stage1_checkpoint "${STAGE1_CHECKPOINT}" \
  --feature_cache_path "${FEATURE_CACHE_PATH}" \
  --stage2_checkpoint "${STAGE2_CHECKPOINT}" \
  --output_dir "${OUT_DIR}" \
  --git_sha "${GIT_SHA}"

cat "${OUT_DIR}/efficiency_report.md"
