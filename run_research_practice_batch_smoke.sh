#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/docs/research_practice/manifests/three_group_smoke_execute.json}"
WEIGHTS_PATH="${WEIGHTS_PATH:-/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
OUT_DIR="${OUT_DIR:-/gemini/data-1/lightrecon_runs/research_practice_batch_smoke_three_groups_20260715_v1}"
RUN_LOG="${OUT_DIR}_launcher.log"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable project Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import numpy, scipy, torch' >/dev/null 2>&1; then
  echo "Selected lightrecon Python is missing NumPy, SciPy or torch: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  echo "CUDA is required because uncached smoke groups need DUSt3R global alignment." >&2
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
if ! "${PYTHON_BIN}" -c 'import cv2' >/dev/null 2>&1; then
  echo "Installing OpenCV headless without changing NumPy"
  "${PYTHON_BIN}" -m pip install --no-deps 'opencv-python-headless>=4.8,<5'
fi
for required in "${MANIFEST}" "${WEIGHTS_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
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

GIT_SHA="$(git rev-parse HEAD)"
echo "LightRecon3D research-practice identical-cache smoke"
echo "project=${PROJECT_DIR}"
echo "python=${PYTHON_BIN}"
echo "manifest=${MANIFEST}"
echo "weights=${WEIGHTS_PATH}"
echo "output=${OUT_DIR}"
echo "git_sha=${GIT_SHA}"
"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import cv2
import numpy
import scipy
import torch

print(f"torch={torch.__version__}")
print(f"numpy={numpy.__version__}")
print(f"scipy={scipy.__version__}")
print(f"cv2={cv2.__version__}")
print(f"roma={metadata.version('roma')}")
print(f"trimesh={metadata.version('trimesh')}")
PY
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

"${PYTHON_BIN}" execute_research_practice_batch.py \
  --manifest "${MANIFEST}" \
  --output_dir "${OUT_DIR}" \
  --project_dir "${PROJECT_DIR}" \
  --python_bin "${PYTHON_BIN}" \
  --weights_path "${WEIGHTS_PATH}" \
  --git_sha "${GIT_SHA}"

echo "Research-practice batch smoke complete: ${OUT_DIR}"
cat "${OUT_DIR}/batch_execution.md"
