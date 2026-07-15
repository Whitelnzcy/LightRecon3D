#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
BATCH_DIR="${BATCH_DIR:-/gemini/data-1/lightrecon_runs/research_practice_batch_smoke_three_groups_20260715_v1}"
OUT_DIR="${OUT_DIR:-${BATCH_DIR}_gate_v1}"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable project Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import roma' >/dev/null 2>&1; then
  echo "roma is missing; installing roma==1.5.6 into ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install --no-deps roma==1.5.6
fi
for required in "${BATCH_DIR}/aggregate_metrics.json" "${BATCH_DIR}/batch_execution.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing smoke result: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
  exit 2
fi

echo "LightRecon3D research-practice smoke gate"
echo "project=${PROJECT_DIR}"
echo "batch=${BATCH_DIR}"
echo "output=${OUT_DIR}"
echo "git_sha=$(git rev-parse HEAD)"
"${PYTHON_BIN}" -c 'from importlib.metadata import version; import roma; print("roma=" + version("roma"))'

"${PYTHON_BIN}" audit_research_practice_batch_results.py \
  --aggregate_metrics_json "${BATCH_DIR}/aggregate_metrics.json" \
  --batch_execution_json "${BATCH_DIR}/batch_execution.json" \
  --output_dir "${OUT_DIR}" \
  --minimum_independent_scenes 8

cat "${OUT_DIR}/smoke_gate.md"
