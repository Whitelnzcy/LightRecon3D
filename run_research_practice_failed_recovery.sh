#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
ROOT_DIR="${ROOT_DIR:-/gemini/data-1/Structured3D}"
RUNROOT="${RUNROOT:-/gemini/data-1/lightrecon_runs}"
WEIGHTS_PATH="${WEIGHTS_PATH:-/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${RUNROOT}/checkpoints/stage1_large_80m_train2048_sharded_v1/best.pt}"
FEATURE_CACHE_PATH="${FEATURE_CACHE_PATH:-${RUNROOT}/cache/stage1_train2048_shards/val128.pt}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${RUNROOT}/stage2_region_merge_v1/checkpoint/best.pt}"

ORIGINAL_NAME="${ORIGINAL_NAME:-research_practice_all_validation_20260717_v1}"
ORIGINAL_SELECTION_DIR="${ORIGINAL_SELECTION_DIR:-${RUNROOT}/${ORIGINAL_NAME}_selection}"
ORIGINAL_MATERIALIZATION_DIR="${ORIGINAL_MATERIALIZATION_DIR:-${RUNROOT}/${ORIGINAL_NAME}_materialization}"
ORIGINAL_PREFLIGHT_DIR="${ORIGINAL_PREFLIGHT_DIR:-${RUNROOT}/${ORIGINAL_NAME}_preflight}"
ORIGINAL_BATCH_DIR="${ORIGINAL_BATCH_DIR:-${RUNROOT}/${ORIGINAL_NAME}_batch}"

RECOVERY_NAME="${RECOVERY_NAME:-${ORIGINAL_NAME}_failed_recovery_v1}"
RECOVERY_PLAN_DIR="${RECOVERY_PLAN_DIR:-${RUNROOT}/${RECOVERY_NAME}_plan}"
RECOVERY_INPUT_ROOT="${RECOVERY_INPUT_ROOT:-${RUNROOT}/${RECOVERY_NAME}_inputs}"
RECOVERY_MATERIALIZATION_DIR="${RECOVERY_MATERIALIZATION_DIR:-${RUNROOT}/${RECOVERY_NAME}_materialization}"
RECOVERY_PREFLIGHT_DIR="${RECOVERY_PREFLIGHT_DIR:-${RUNROOT}/${RECOVERY_NAME}_preflight}"
RECOVERY_BATCH_DIR="${RECOVERY_BATCH_DIR:-${RUNROOT}/${RECOVERY_NAME}_batch}"
MERGED_DIR="${MERGED_DIR:-${RUNROOT}/${RECOVERY_NAME}_merged}"
AUDIT_DIR="${AUDIT_DIR:-${RUNROOT}/${RECOVERY_NAME}_audit}"
VIS_DIR="${VIS_DIR:-${RUNROOT}/${RECOVERY_NAME}_3d_visualization}"
BUNDLE_DIR="${BUNDLE_DIR:-${RUNROOT}/${RECOVERY_NAME}_bundle}"
RUN_LOG="${RUN_LOG:-${RUNROOT}/${RECOVERY_NAME}_launcher.log}"
MAX_VIS_POINTS="${MAX_VIS_POINTS:-180000}"

SELECTION_PLAN="${ORIGINAL_SELECTION_DIR}/selection_plan.json"
ORIGINAL_MATERIALIZATION="${ORIGINAL_MATERIALIZATION_DIR}/materialization.json"
ORIGINAL_PREFLIGHT="${ORIGINAL_PREFLIGHT_DIR}/batch_preflight.json"
ORIGINAL_BATCH="${ORIGINAL_BATCH_DIR}/batch_execution.json"
RECOVERY_SELECTION="${RECOVERY_PLAN_DIR}/recovery_selection_plan.json"
RECOVERY_MANIFEST="${RECOVERY_PLAN_DIR}/recovery_execute.json"

cd "${PROJECT_DIR}"

for required in \
  "${PYTHON_BIN}" \
  "${ROOT_DIR}" \
  "${WEIGHTS_PATH}" \
  "${STAGE1_CHECKPOINT}" \
  "${FEATURE_CACHE_PATH}" \
  "${STAGE2_CHECKPOINT}" \
  "${SELECTION_PLAN}" \
  "${ORIGINAL_MATERIALIZATION}" \
  "${ORIGINAL_PREFLIGHT}" \
  "${ORIGINAL_BATCH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing recovery input: ${required}" >&2
    exit 2
  fi
done

for output in \
  "${RECOVERY_PLAN_DIR}" \
  "${RECOVERY_INPUT_ROOT}" \
  "${RECOVERY_MATERIALIZATION_DIR}" \
  "${RECOVERY_PREFLIGHT_DIR}" \
  "${RECOVERY_BATCH_DIR}" \
  "${MERGED_DIR}" \
  "${AUDIT_DIR}" \
  "${VIS_DIR}" \
  "${BUNDLE_DIR}" \
  "${RUN_LOG}"; do
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite recovery output: ${output}" >&2
    echo "Use a new RECOVERY_NAME for another immutable attempt." >&2
    exit 2
  fi
done

if ! "${PYTHON_BIN}" - <<'PY'
import cv2
import numpy
import scipy
import torch
from PIL import Image

print(
    f"numpy={numpy.__version__} cv2={cv2.__version__} "
    f"scipy={scipy.__version__} torch={torch.__version__} Pillow={Image.__version__}"
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
probe = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print(f"cuda_probe={probe.device}")
PY
then
  echo "The fixed lightrecon Python or CUDA probe failed; do not start recovery." >&2
  nvidia-smi || true
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'from importlib.metadata import version; version("roma")' >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install --no-deps roma==1.5.6
fi
if ! "${PYTHON_BIN}" -c 'from importlib.metadata import version; assert version("trimesh") == "4.9.0"' >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install --no-deps trimesh==4.9.0
fi

exec > >(tee "${RUN_LOG}") 2>&1
GIT_SHA="$(git rev-parse HEAD)"
echo "LightRecon3D failed-scene immutable recovery"
echo "original_name=${ORIGINAL_NAME}"
echo "recovery_name=${RECOVERY_NAME}"
echo "git_sha=${GIT_SHA}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

echo "[1/8] Prepare a frozen plan containing only original failed items"
"${PYTHON_BIN}" recover_research_practice_failed_batch.py prepare \
  --selection_plan_json "${SELECTION_PLAN}" \
  --materialization_json "${ORIGINAL_MATERIALIZATION}" \
  --retry_input_root "${RECOVERY_INPUT_ROOT}" \
  --output_dir "${RECOVERY_PLAN_DIR}"
cat "${RECOVERY_PLAN_DIR}/recovery_plan_summary.json"

echo "[2/8] Re-materialize failed Stage1/Stage2 inputs in new directories"
set +e
"${PYTHON_BIN}" materialize_research_practice_final_inputs.py \
  --selection_plan "${RECOVERY_SELECTION}" \
  --output_dir "${RECOVERY_MATERIALIZATION_DIR}" \
  --project_dir "${PROJECT_DIR}" \
  --python_bin "${PYTHON_BIN}" \
  --root_dir "${ROOT_DIR}" \
  --weights_path "${WEIGHTS_PATH}" \
  --stage1_checkpoint "${STAGE1_CHECKPOINT}" \
  --feature_cache_path "${FEATURE_CACHE_PATH}" \
  --stage2_checkpoint "${STAGE2_CHECKPOINT}" \
  --git_sha "${GIT_SHA}"
MATERIALIZATION_RC=$?
set -e
echo "recovery_materialization_return_code=${MATERIALIZATION_RC}"
test -f "${RECOVERY_MATERIALIZATION_DIR}/materialization.json"
cat "${RECOVERY_MATERIALIZATION_DIR}/materialization.md"

echo "[3/8] Preflight recovered inputs"
"${PYTHON_BIN}" research_practice_batch.py \
  --manifest "${RECOVERY_MANIFEST}" \
  --output_dir "${RECOVERY_PREFLIGHT_DIR}" \
  --git_sha "${GIT_SHA}"
cat "${RECOVERY_PREFLIGHT_DIR}/batch_preflight.md"

echo "[4/8] Execute full reconstruction and metrics only for recovered items"
set +e
"${PYTHON_BIN}" execute_research_practice_batch.py \
  --manifest "${RECOVERY_MANIFEST}" \
  --output_dir "${RECOVERY_BATCH_DIR}" \
  --project_dir "${PROJECT_DIR}" \
  --python_bin "${PYTHON_BIN}" \
  --weights_path "${WEIGHTS_PATH}" \
  --git_sha "${GIT_SHA}"
RECOVERY_BATCH_RC=$?
set -e
echo "recovery_batch_return_code=${RECOVERY_BATCH_RC}"
test -f "${RECOVERY_BATCH_DIR}/batch_execution.json"

echo "[5/8] Merge successful recovery rows with the untouched original 17-scene ledger"
"${PYTHON_BIN}" recover_research_practice_failed_batch.py merge \
  --selection_plan_json "${SELECTION_PLAN}" \
  --original_materialization_json "${ORIGINAL_MATERIALIZATION}" \
  --recovery_materialization_json "${RECOVERY_MATERIALIZATION_DIR}/materialization.json" \
  --original_preflight_json "${ORIGINAL_PREFLIGHT}" \
  --recovery_preflight_json "${RECOVERY_PREFLIGHT_DIR}/batch_preflight.json" \
  --original_batch_json "${ORIGINAL_BATCH}" \
  --recovery_batch_json "${RECOVERY_BATCH_DIR}/batch_execution.json" \
  --output_dir "${MERGED_DIR}"
cat "${MERGED_DIR}/recovery_merge.json"

COMBINED_MATERIALIZATION="${MERGED_DIR}/combined_materialization.json"
COMBINED_PREFLIGHT="${MERGED_DIR}/combined_preflight.json"
COMBINED_BATCH="${MERGED_DIR}/combined_batch/batch_execution.json"
COMBINED_METRICS="${MERGED_DIR}/combined_batch/aggregate_metrics.json"

echo "[6/8] Recompute the paired audit over the combined scene ledger"
"${PYTHON_BIN}" audit_research_practice_final_results.py \
  --aggregate_metrics_json "${COMBINED_METRICS}" \
  --batch_execution_json "${COMBINED_BATCH}" \
  --output_dir "${AUDIT_DIR}" \
  --allow_failed_items
cat "${AUDIT_DIR}/final_method_audit.md"

echo "[7/8] Render actual same-camera 3D results for every combined passed scene"
"${PYTHON_BIN}" visualize_research_practice_final_3d.py \
  --batch_execution_json "${COMBINED_BATCH}" \
  --audit_json "${AUDIT_DIR}/final_method_audit.json" \
  --output_dir "${VIS_DIR}" \
  --views auto \
  --panel_width 720 \
  --panel_height 500 \
  --max_points "${MAX_VIS_POINTS}" \
  --point_radius 1

echo "[8/8] Collect the combined metrics, history and 3D paths"
"${PYTHON_BIN}" collect_research_practice_large_scale.py \
  --selection_plan_json "${SELECTION_PLAN}" \
  --materialization_json "${COMBINED_MATERIALIZATION}" \
  --preflight_json "${COMBINED_PREFLIGHT}" \
  --batch_execution_json "${COMBINED_BATCH}" \
  --audit_json "${AUDIT_DIR}/final_method_audit.json" \
  --visualization_manifest_json "${VIS_DIR}/visualization_manifest.json" \
  --output_dir "${BUNDLE_DIR}"

cat "${BUNDLE_DIR}/large_scale_summary.md"
echo "Recovery complete."
echo "Summary: ${BUNDLE_DIR}/large_scale_summary.md"
echo "Recovery history: ${MERGED_DIR}/recovery_merge.json"
echo "Per-scene paths: ${BUNDLE_DIR}/scene_artifact_index.csv"
echo "Failures after recovery: ${BUNDLE_DIR}/failures.csv"
echo "3D contact sheet: ${VIS_DIR}/all_scenes_final_3d_contact_sheet.png"
