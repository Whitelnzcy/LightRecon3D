#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
ROOT_DIR="${ROOT_DIR:-/gemini/data-1/Structured3D}"
RUNROOT="${RUNROOT:-/gemini/data-1/lightrecon_runs}"
EXISTING_ROOT="${EXISTING_ROOT:-${RUNROOT}/stage3_val_showcase_v1}"
REUSE_BATCH_DIR="${REUSE_BATCH_DIR:-${RUNROOT}/research_practice_final_eight_scenes_20260716_v1}"

WEIGHTS_PATH="${WEIGHTS_PATH:-/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${RUNROOT}/checkpoints/stage1_large_80m_train2048_sharded_v1/best.pt}"
FEATURE_CACHE_PATH="${FEATURE_CACHE_PATH:-${RUNROOT}/cache/stage1_train2048_shards/val128.pt}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${RUNROOT}/stage2_region_merge_v1/checkpoint/best.pt}"

RUN_NAME="${RUN_NAME:-research_practice_all_validation_20260717_v1}"
SELECTION_DIR="${SELECTION_DIR:-${RUNROOT}/${RUN_NAME}_selection}"
EXPANSION_ROOT="${EXPANSION_ROOT:-${RUNROOT}/${RUN_NAME}_inputs}"
MATERIALIZATION_DIR="${MATERIALIZATION_DIR:-${RUNROOT}/${RUN_NAME}_materialization}"
PREFLIGHT_DIR="${PREFLIGHT_DIR:-${RUNROOT}/${RUN_NAME}_preflight}"
BATCH_DIR="${BATCH_DIR:-${RUNROOT}/${RUN_NAME}_batch}"
AUDIT_DIR="${AUDIT_DIR:-${RUNROOT}/${RUN_NAME}_audit}"
VIS_DIR="${VIS_DIR:-${RUNROOT}/${RUN_NAME}_3d_visualization}"
BUNDLE_DIR="${BUNDLE_DIR:-${RUNROOT}/${RUN_NAME}_bundle}"
RUN_LOG="${RUN_LOG:-${RUNROOT}/${RUN_NAME}_launcher.log}"

# Zero means every eligible independent scene in the frozen validation split.
TARGET_SCENES="${TARGET_SCENES:-0}"
MINIMUM_VALID_ITEMS="${MINIMUM_VALID_ITEMS:-8}"
MIN_PAIRS="${MIN_PAIRS:-10}"
RESUME="${RESUME:-0}"
MAX_VIS_POINTS="${MAX_VIS_POINTS:-180000}"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable project Python: ${PYTHON_BIN}" >&2
  exit 2
fi
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
    raise SystemExit("CUDA is required for missing Stage1/Stage2 groups and DUSt3R alignment")
probe = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print(f"cuda_probe={probe.device}")
PY
then
  echo "The fixed lightrecon Python environment check failed." >&2
  nvidia-smi || true
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'from importlib.metadata import version; version("roma")' >/dev/null 2>&1; then
  echo "roma is missing; installing roma==1.5.6 into ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install --no-deps roma==1.5.6
fi
if ! "${PYTHON_BIN}" -c 'from importlib.metadata import version; assert version("trimesh") == "4.9.0"' >/dev/null 2>&1; then
  echo "Installing trimesh==4.9.0 without changing NumPy"
  "${PYTHON_BIN}" -m pip install --no-deps trimesh==4.9.0
fi

for required in \
  "${ROOT_DIR}" \
  "${EXISTING_ROOT}/selected_groups.tsv" \
  "${WEIGHTS_PATH}" \
  "${STAGE1_CHECKPOINT}" \
  "${FEATURE_CACHE_PATH}" \
  "${STAGE2_CHECKPOINT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing large-scale experiment input: ${required}" >&2
    exit 2
  fi
done

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ "${RESUME}" == "0" ]]; then
  for output in \
    "${SELECTION_DIR}" \
    "${EXPANSION_ROOT}" \
    "${MATERIALIZATION_DIR}" \
    "${PREFLIGHT_DIR}" \
    "${BATCH_DIR}" \
    "${AUDIT_DIR}" \
    "${VIS_DIR}" \
    "${BUNDLE_DIR}" \
    "${RUN_LOG}"; do
    if [[ -e "${output}" ]]; then
      echo "Refusing to overwrite existing large-scale output: ${output}" >&2
      echo "Use RESUME=1 only for this exact frozen run." >&2
      exit 2
    fi
  done
fi

if [[ "${RESUME}" == "1" ]]; then
  exec > >(tee -a "${RUN_LOG}") 2>&1
else
  exec > >(tee "${RUN_LOG}") 2>&1
fi

GIT_SHA="$(git rev-parse HEAD)"
echo "LightRecon3D all-eligible-validation experiment"
echo "project=${PROJECT_DIR}"
echo "python=${PYTHON_BIN}"
echo "dataset=${ROOT_DIR}"
echo "run_name=${RUN_NAME}"
echo "target_scenes=${TARGET_SCENES} (0 means all eligible validation scenes)"
echo "minimum_valid_items=${MINIMUM_VALID_ITEMS}"
echo "resume=${RESUME}"
echo "git_sha=${GIT_SHA}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

SELECTION_PLAN="${SELECTION_DIR}/selection_plan.json"
MANIFEST="${SELECTION_DIR}/final_unique_scenes_execute.json"

echo "[1/7] Deterministic inventory and all-validation selection"
if [[ -f "${SELECTION_PLAN}" && -f "${MANIFEST}" && "${RESUME}" == "1" ]]; then
  echo "[resume] reusing frozen selection: ${SELECTION_DIR}"
else
  REUSE_ARGS=()
  if [[ -f "${REUSE_BATCH_DIR}/batch_execution.json" ]]; then
    REUSE_ARGS=(--reuse_batch_execution_json "${REUSE_BATCH_DIR}/batch_execution.json")
  fi
  "${PYTHON_BIN}" prepare_research_practice_final_manifest.py \
    --root_dir "${ROOT_DIR}" \
    --existing_root "${EXISTING_ROOT}" \
    --expansion_root "${EXPANSION_ROOT}" \
    "${REUSE_ARGS[@]}" \
    --target_scenes "${TARGET_SCENES}" \
    --minimum_valid_items "${MINIMUM_VALID_ITEMS}" \
    --min_pairs "${MIN_PAIRS}" \
    --split val \
    --train_ratio 0.9 \
    --pair_strategy all \
    --output_dir "${SELECTION_DIR}"
fi
cat "${SELECTION_DIR}/selection_plan.md"

echo "[2/7] Materialize every missing Stage1/Stage2 input"
MATERIALIZE_ARGS=()
if [[ "${RESUME}" == "1" && -f "${MATERIALIZATION_DIR}/materialization.json" ]]; then
  MATERIALIZE_ARGS=(--resume)
fi
set +e
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
  --git_sha "${GIT_SHA}" \
  "${MATERIALIZE_ARGS[@]}"
MATERIALIZATION_RC=$?
set -e
echo "materialization_return_code=${MATERIALIZATION_RC} (individual failures are retained)"
test -f "${MATERIALIZATION_DIR}/materialization.json"

echo "[3/7] Preflight every selected group"
if [[ -f "${PREFLIGHT_DIR}/batch_preflight.json" && "${RESUME}" == "1" ]]; then
  echo "[resume] reusing preflight: ${PREFLIGHT_DIR}"
else
  "${PYTHON_BIN}" research_practice_batch.py \
    --manifest "${MANIFEST}" \
    --output_dir "${PREFLIGHT_DIR}" \
    --git_sha "${GIT_SHA}"
fi
cat "${PREFLIGHT_DIR}/batch_preflight.md"

echo "[4/7] Full identical-cache methods, metrics and 3D artifacts"
BATCH_ARGS=()
if [[ "${RESUME}" == "1" && -f "${BATCH_DIR}/batch_execution.json" ]]; then
  BATCH_ARGS=(--resume)
fi
set +e
"${PYTHON_BIN}" execute_research_practice_batch.py \
  --manifest "${MANIFEST}" \
  --output_dir "${BATCH_DIR}" \
  --project_dir "${PROJECT_DIR}" \
  --python_bin "${PYTHON_BIN}" \
  --weights_path "${WEIGHTS_PATH}" \
  --git_sha "${GIT_SHA}" \
  "${BATCH_ARGS[@]}"
BATCH_RC=$?
set -e
echo "batch_return_code=${BATCH_RC} (individual failures are retained)"
test -f "${BATCH_DIR}/batch_execution.json"
"${PYTHON_BIN}" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); n=int(p["summary"]["passed_items"]); m=int(sys.argv[2]); print(f"passed_scenes={n}, required={m}"); raise SystemExit(0 if n >= m else 2)' "${BATCH_DIR}/batch_execution.json" "${MINIMUM_VALID_ITEMS}"

echo "[5/7] Paired audit over every successful independent scene"
if [[ -f "${AUDIT_DIR}/final_method_audit.json" && "${RESUME}" == "1" ]]; then
  echo "[resume] reusing audit: ${AUDIT_DIR}"
else
  "${PYTHON_BIN}" audit_research_practice_final_results.py \
    --aggregate_metrics_json "${BATCH_DIR}/aggregate_metrics.json" \
    --batch_execution_json "${BATCH_DIR}/batch_execution.json" \
    --output_dir "${AUDIT_DIR}" \
    --allow_failed_items
fi
cat "${AUDIT_DIR}/final_method_audit.md"

echo "[6/7] Same-camera 3D visualization for every successful scene"
if [[ -f "${VIS_DIR}/visualization_manifest.json" && "${RESUME}" == "1" ]]; then
  echo "[resume] reusing visualization: ${VIS_DIR}"
else
  "${PYTHON_BIN}" visualize_research_practice_final_3d.py \
    --batch_execution_json "${BATCH_DIR}/batch_execution.json" \
    --audit_json "${AUDIT_DIR}/final_method_audit.json" \
    --output_dir "${VIS_DIR}" \
    --views auto \
    --panel_width 720 \
    --panel_height 500 \
    --max_points "${MAX_VIS_POINTS}" \
    --point_radius 1
fi

echo "[7/7] Collect metrics, failures and artifact paths into one bundle"
if [[ -f "${BUNDLE_DIR}/large_scale_summary.json" && "${RESUME}" == "1" ]]; then
  echo "[resume] reusing result bundle: ${BUNDLE_DIR}"
else
  "${PYTHON_BIN}" collect_research_practice_large_scale.py \
    --selection_plan_json "${SELECTION_PLAN}" \
    --materialization_json "${MATERIALIZATION_DIR}/materialization.json" \
    --preflight_json "${PREFLIGHT_DIR}/batch_preflight.json" \
    --batch_execution_json "${BATCH_DIR}/batch_execution.json" \
    --audit_json "${AUDIT_DIR}/final_method_audit.json" \
    --visualization_manifest_json "${VIS_DIR}/visualization_manifest.json" \
    --output_dir "${BUNDLE_DIR}"
fi

cat "${BUNDLE_DIR}/large_scale_summary.md"
echo "Large-scale experiment complete."
echo "Summary: ${BUNDLE_DIR}/large_scale_summary.md"
echo "All per-scene artifacts: ${BUNDLE_DIR}/scene_artifact_index.csv"
echo "Failures: ${BUNDLE_DIR}/failures.csv"
echo "3D contact sheet: ${VIS_DIR}/all_scenes_final_3d_contact_sheet.png"
