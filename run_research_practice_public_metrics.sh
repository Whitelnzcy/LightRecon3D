#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
BATCH_EXECUTION_JSON="${BATCH_EXECUTION_JSON:-/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_merged/combined_batch/batch_execution.json}"
OUT_DIR="${OUT_DIR:-/gemini/data-1/lightrecon_runs/research_practice_public_metrics_20260717_v1}"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable project Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import roma' >/dev/null 2>&1; then
  echo "roma is missing; installing roma==1.5.6 into ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install --no-deps roma==1.5.6
fi
if ! "${PYTHON_BIN}" -c 'import numpy' >/dev/null 2>&1; then
  echo "The fixed lightrecon Python must provide NumPy." >&2
  exit 2
fi
if [[ ! -f "${BATCH_EXECUTION_JSON}" ]]; then
  echo "Missing combined batch ledger: ${BATCH_EXECUTION_JSON}" >&2
  exit 2
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
  exit 2
fi

echo "LightRecon3D publication-style plane metric recomputation"
echo "project=${PROJECT_DIR}"
echo "python=${PYTHON_BIN}"
echo "batch_execution=${BATCH_EXECUTION_JSON}"
echo "output=${OUT_DIR}"
echo "git_sha=$(git rev-parse HEAD)"
"${PYTHON_BIN}" -c 'from importlib.metadata import version; import roma; print("roma=" + version("roma"))'

"${PYTHON_BIN}" recompute_public_plane_metrics.py \
  --batch_execution_json "${BATCH_EXECUTION_JSON}" \
  --output_dir "${OUT_DIR}"

cat "${OUT_DIR}/public_plane_metrics.md"
