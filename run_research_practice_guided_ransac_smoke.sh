#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
BATCH_DIR="${BATCH_DIR:-/gemini/data-1/lightrecon_runs/research_practice_batch_smoke_three_groups_20260715_v1}"
OUT_DIR="${OUT_DIR:-/gemini/data-1/lightrecon_runs/research_practice_guided_ransac_smoke_20260716_v1}"
RUN_LOG="${OUT_DIR}_launcher.log"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable project Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import numpy' >/dev/null 2>&1; then
  echo "Selected lightrecon Python is missing NumPy: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import roma' >/dev/null 2>&1; then
  echo "roma is missing; installing roma==1.5.6 into ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install --no-deps roma==1.5.6
fi
if [[ ! -f "${BATCH_DIR}/batch_execution.json" ]]; then
  echo "Missing archived smoke batch: ${BATCH_DIR}/batch_execution.json" >&2
  exit 2
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
  exit 2
fi
if [[ -e "${RUN_LOG}" ]]; then
  echo "Refusing to overwrite existing launcher log: ${RUN_LOG}" >&2
  exit 2
fi

exec > >(tee "${RUN_LOG}") 2>&1

echo "LightRecon3D learning-guided RANSAC identical-cache smoke"
echo "project=${PROJECT_DIR}"
echo "python=${PYTHON_BIN}"
echo "batch=${BATCH_DIR}"
echo "output=${OUT_DIR}"
echo "git_sha=$(git rev-parse HEAD)"
"${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version
import numpy

print(f"numpy={numpy.__version__}")
print(f"roma={version('roma')}")
PY

"${PYTHON_BIN}" evaluate_guided_ransac_smoke.py \
  --batch_execution_json "${BATCH_DIR}/batch_execution.json" \
  --output_dir "${OUT_DIR}" \
  --project_dir "${PROJECT_DIR}" \
  --python_bin "${PYTHON_BIN}"

echo "Learning-guided RANSAC smoke complete: ${OUT_DIR}"
cat "${OUT_DIR}/guided_smoke.md"
