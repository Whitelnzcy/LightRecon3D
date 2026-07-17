#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
RUNROOT="${RUNROOT:-/gemini/data-1/lightrecon_runs}"
FINAL_BATCH_DIR="${FINAL_BATCH_DIR:-${RUNROOT}/research_practice_final_eight_scenes_20260716_v1}"
FINAL_AUDIT_DIR="${FINAL_AUDIT_DIR:-${FINAL_BATCH_DIR}_audit_v1}"
OUT_DIR="${OUT_DIR:-${RUNROOT}/research_practice_final_3d_visualization_20260717_v1}"
RUN_LOG="${OUT_DIR}_run.log"

# Override this when a PCA view is poor for a particular dataset.  The same
# cameras are still used for RGB, ordinary RANSAC, and guided RANSAC.
VIEWS="${VIEWS:-auto}"
PANEL_WIDTH="${PANEL_WIDTH:-720}"
PANEL_HEIGHT="${PANEL_HEIGHT:-500}"
MAX_POINTS="${MAX_POINTS:-220000}"
POINT_RADIUS="${POINT_RADIUS:-1}"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable project Python: ${PYTHON_BIN}" >&2
  exit 2
fi

# Keep the server-entry convention used by this project.  --no-deps prevents
# pip from silently upgrading NumPy in the frozen lightrecon environment.
if ! "${PYTHON_BIN}" -c 'from importlib.metadata import version; version("roma")' >/dev/null 2>&1; then
  echo "roma is missing; installing roma==1.5.6 into ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install --no-deps roma==1.5.6
fi

if ! "${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version
import numpy
from PIL import Image

print(f"numpy={numpy.__version__}")
print(f"Pillow={Image.__version__}")
print(f"roma={version('roma')}")
PY
then
  echo "Visualization requires NumPy and Pillow in the fixed lightrecon Python." >&2
  exit 2
fi

BATCH_EXECUTION_JSON="${FINAL_BATCH_DIR}/batch_execution.json"
FINAL_AUDIT_JSON="${FINAL_AUDIT_DIR}/final_method_audit.json"
for required in "${BATCH_EXECUTION_JSON}" "${FINAL_AUDIT_JSON}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing final visualization input: ${required}" >&2
    exit 2
  fi
done
for output in "${OUT_DIR}" "${RUN_LOG}"; do
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite existing visualization output: ${output}" >&2
    exit 2
  fi
done

exec > >(tee "${RUN_LOG}") 2>&1

echo "LightRecon3D final-batch 3D visualization (CPU render only)"
echo "project=${PROJECT_DIR}"
echo "python=${PYTHON_BIN}"
echo "final_batch=${FINAL_BATCH_DIR}"
echo "final_audit=${FINAL_AUDIT_DIR}"
echo "output=${OUT_DIR}"
echo "git_sha=$(git rev-parse HEAD)"
echo "cuda_used=false"
echo "reconstruction_recomputed=false"
echo "views=${VIEWS}"
echo "max_points=${MAX_POINTS}"

"${PYTHON_BIN}" visualize_research_practice_final_3d.py \
  --batch_execution_json "${BATCH_EXECUTION_JSON}" \
  --audit_json "${FINAL_AUDIT_JSON}" \
  --output_dir "${OUT_DIR}" \
  --views "${VIEWS}" \
  --panel_width "${PANEL_WIDTH}" \
  --panel_height "${PANEL_HEIGHT}" \
  --max_points "${MAX_POINTS}" \
  --point_radius "${POINT_RADIUS}"

cat "${OUT_DIR}/README.md"
echo "Primary report figure: ${OUT_DIR}/all_scenes_final_3d_contact_sheet.png"
echo "Per-scene multiview figures: ${OUT_DIR}/final_*/scene_*_multiview_comparison.png"
