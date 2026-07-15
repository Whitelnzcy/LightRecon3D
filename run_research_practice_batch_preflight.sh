#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/docs/research_practice/manifests/three_group_smoke.json}"
OUT_DIR="${OUT_DIR:-/gemini/data-1/lightrecon_runs/research_practice_batch_preflight_three_group_20260715_v1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Required lightrecon Python is missing or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import numpy, torch' >/dev/null 2>&1; then
  echo "NumPy or torch is missing from ${PYTHON_BIN}; activate/repair the lightrecon environment first." >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import roma' >/dev/null 2>&1; then
  echo "roma is missing; installing roma==1.5.6 into ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install --no-deps roma==1.5.6
fi
"${PYTHON_BIN}" -c 'from importlib.metadata import version; import roma; print("roma=" + version("roma"))'
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Batch manifest is missing: ${MANIFEST}" >&2
  exit 2
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
GIT_SHA="$(git rev-parse HEAD)"

echo "LightRecon3D research-practice three-group preflight"
echo "project=${PROJECT_DIR}"
echo "python=${PYTHON_BIN}"
echo "manifest=${MANIFEST}"
echo "output=${OUT_DIR}"
echo "git_sha=${GIT_SHA}"

"${PYTHON_BIN}" research_practice_batch.py \
  --manifest "${MANIFEST}" \
  --output_dir "${OUT_DIR}" \
  --git_sha "${GIT_SHA}"

echo "Preflight complete: ${OUT_DIR}"
