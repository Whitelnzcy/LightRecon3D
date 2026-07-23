#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
ROOT_DIR="${ROOT_DIR:-/gemini/data-1/Structured3D}"
SELECTION_DIR="${SELECTION_DIR:-/gemini/data-1/lightrecon_runs/research_practice_final_selection_20260716_v1}"
SELECTION_PLAN="${SELECTION_PLAN:-${SELECTION_DIR}/selection_plan.json}"
MANIFEST="${MANIFEST:-${SELECTION_DIR}/final_unique_scenes_execute.json}"

RUNROOT="${RUNROOT:-/gemini/data-1/lightrecon_runs}"
WEIGHTS_PATH="${WEIGHTS_PATH:-/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${RUNROOT}/checkpoints/stage1_large_80m_train2048_sharded_v1/best.pt}"
FEATURE_CACHE_PATH="${FEATURE_CACHE_PATH:-${RUNROOT}/cache/stage1_train2048_shards/val128.pt}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${RUNROOT}/stage2_region_merge_v1/checkpoint/best.pt}"

MATERIALIZATION_DIR="${MATERIALIZATION_DIR:-${RUNROOT}/research_practice_final_materialization_20260716_v1}"
PREFLIGHT_DIR="${PREFLIGHT_DIR:-${RUNROOT}/research_practice_final_preflight_20260716_v1}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-${RUNROOT}/research_practice_final_eight_scenes_20260716_v1}"
RUN_LOG="${FINAL_OUT_DIR}_launcher.log"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable project Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import cv2, numpy, scipy, torch' >/dev/null 2>&1; then
  echo "Selected lightrecon Python is missing cv2, NumPy, SciPy or torch: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  echo "CUDA is required for three missing Stage1/Stage2 groups and uncached DUSt3R alignments." >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import roma' >/dev/null 2>&1; then
  echo "roma is missing; installing roma==1.5.6 into ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install --no-deps roma==1.5.6
fi
if ! "${PYTHON_BIN}" -c 'import trimesh; import importlib.metadata as m; assert m.version("trimesh") == "4.9.0"' >/dev/null 2>&1; then
  echo "Installing trimesh==4.9.0 without changing NumPy"
  "${PYTHON_BIN}" -m pip install --no-deps trimesh==4.9.0
fi

for required in \
  "${ROOT_DIR}" \
  "${SELECTION_PLAN}" \
  "${MANIFEST}" \
  "${WEIGHTS_PATH}" \
  "${STAGE1_CHECKPOINT}" \
  "${FEATURE_CACHE_PATH}" \
  "${STAGE2_CHECKPOINT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required final-batch input: ${required}" >&2
    exit 2
  fi
done
for output in "${MATERIALIZATION_DIR}" "${PREFLIGHT_DIR}" "${FINAL_OUT_DIR}" "${RUN_LOG}"; do
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite existing final-batch output: ${output}" >&2
    exit 2
  fi
done

exec > >(tee "${RUN_LOG}") 2>&1

GIT_SHA="$(git rev-parse HEAD)"
echo "LightRecon3D final eight-independent-scene batch"
echo "project=${PROJECT_DIR}"
echo "python=${PYTHON_BIN}"
echo "selection_plan=${SELECTION_PLAN}"
echo "manifest=${MANIFEST}"
echo "materialization=${MATERIALIZATION_DIR}"
echo "preflight=${PREFLIGHT_DIR}"
echo "final_output=${FINAL_OUT_DIR}"
echo "git_sha=${GIT_SHA}"
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
print(f"trimesh={version('trimesh')}")
PY
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

echo "[1/3] Materialize only missing Stage1/Stage2 groups"
"${PYTHON_BIN}" materialize_research_practice_final_inputs.py \
  --selection_plan "${SELECTION_PLAN}" \
  --output_dir "${MATERIALIZATION_DIR}" \
  --project_dir "${PROJECT_DIR}" \
  --python_bin "${PYTHON_BIN}" \
  --root_dir "${ROOT_DIR}" \
  --weights_path "${WEIGHTS_PATH}" \
  --stage1_checkpoint "${STAGE1_CHECKPOINT}" \
  --feature_cache_path "${FEATURE_CACHE_PATH}" \
  --stage2_checkpoint "${STAGE2_CHECKPOINT}" \
  --git_sha "${GIT_SHA}"

echo "[2/3] Strict eight-scene input preflight"
"${PYTHON_BIN}" research_practice_batch.py \
  --manifest "${MANIFEST}" \
  --output_dir "${PREFLIGHT_DIR}" \
  --git_sha "${GIT_SHA}"

echo "[3/3] Identical-cache final methods and metrics"
"${PYTHON_BIN}" execute_research_practice_batch.py \
  --manifest "${MANIFEST}" \
  --output_dir "${FINAL_OUT_DIR}" \
  --project_dir "${PROJECT_DIR}" \
  --python_bin "${PYTHON_BIN}" \
  --weights_path "${WEIGHTS_PATH}" \
  --git_sha "${GIT_SHA}"

echo "Final eight-scene batch complete: ${FINAL_OUT_DIR}"
cat "${MATERIALIZATION_DIR}/materialization.md"
cat "${PREFLIGHT_DIR}/batch_preflight.md"
cat "${FINAL_OUT_DIR}/batch_execution.md"
