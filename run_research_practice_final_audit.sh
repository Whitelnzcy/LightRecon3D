#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
FINAL_BATCH_DIR="${FINAL_BATCH_DIR:-/gemini/data-1/lightrecon_runs/research_practice_final_eight_scenes_20260716_v1}"
OUT_DIR="${OUT_DIR:-${FINAL_BATCH_DIR}_audit_v1}"
RUN_LOG="${OUT_DIR}_run.log"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable project Python: ${PYTHON_BIN}" >&2
  exit 2
fi
for required in \
  "${FINAL_BATCH_DIR}/aggregate_metrics.json" \
  "${FINAL_BATCH_DIR}/batch_execution.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing final-batch result: ${required}" >&2
    exit 2
  fi
done
for output in "${OUT_DIR}" "${RUN_LOG}"; do
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite existing final audit output: ${output}" >&2
    exit 2
  fi
done

exec > >(tee "${RUN_LOG}") 2>&1

echo "LightRecon3D research-practice final result audit (CPU only)"
echo "project=${PROJECT_DIR}"
echo "python=${PYTHON_BIN}"
echo "final_batch=${FINAL_BATCH_DIR}"
echo "output=${OUT_DIR}"
echo "git_sha=$(git rev-parse HEAD)"
echo "cuda_used=false"

"${PYTHON_BIN}" audit_research_practice_final_results.py \
  --aggregate_metrics_json "${FINAL_BATCH_DIR}/aggregate_metrics.json" \
  --batch_execution_json "${FINAL_BATCH_DIR}/batch_execution.json" \
  --output_dir "${OUT_DIR}"

"${PYTHON_BIN}" audit_research_practice_batch_results.py \
  --aggregate_metrics_json "${FINAL_BATCH_DIR}/aggregate_metrics.json" \
  --batch_execution_json "${FINAL_BATCH_DIR}/batch_execution.json" \
  --output_dir "${OUT_DIR}/manual_identity_gate" \
  --minimum_independent_scenes 8

cat "${OUT_DIR}/final_method_audit.md"
cat "${OUT_DIR}/manual_identity_gate/smoke_gate.md"
