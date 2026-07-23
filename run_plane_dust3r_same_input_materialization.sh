#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
BATCH_JSON="${BATCH_JSON:-/gemini/data-1/lightrecon_runs/research_practice_all_validation_20260717_v1_failed_recovery_v1_merged/combined_batch/batch_execution.json}"
OUT_DIR="${OUT_DIR:-/gemini/data-1/lightrecon_runs/plane_dust3r_same_input_20260717_v1}"
LINK_MODE="${LINK_MODE:-symlink}"
MAXIMUM_SCENES="${MAXIMUM_SCENES:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing fixed lightrecon Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${BATCH_JSON}" ]]; then
  echo "Missing frozen batch ledger: ${BATCH_JSON}" >&2
  exit 2
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
  exit 2
fi

if ! "${PYTHON_BIN}" -c 'import roma' >/dev/null 2>&1; then
  echo "roma is missing; installing roma==1.5.6 into ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install --no-deps roma==1.5.6
fi
"${PYTHON_BIN}" -c 'from importlib.metadata import version; print("roma=" + version("roma"))'

echo "LightRecon3D Plane-DUSt3R same-input materialization"
echo "project=${PROJECT_DIR}"
echo "batch=${BATCH_JSON}"
echo "output=${OUT_DIR}"
echo "link_mode=${LINK_MODE}"
echo "maximum_scenes=${MAXIMUM_SCENES}"
echo "git_sha=$(git -C "${PROJECT_DIR}" rev-parse HEAD)"

"${PYTHON_BIN}" "${PROJECT_DIR}/materialize_plane_dust3r_same_input.py" \
  --batch_execution_json "${BATCH_JSON}" \
  --output_dir "${OUT_DIR}" \
  --link_mode "${LINK_MODE}" \
  --maximum_scenes "${MAXIMUM_SCENES}" \
  --git_sha "$(git -C "${PROJECT_DIR}" rev-parse HEAD)"

cat "${OUT_DIR}/same_input_manifest.md"
