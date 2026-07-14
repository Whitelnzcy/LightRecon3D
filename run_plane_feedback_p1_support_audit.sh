#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}
GT_NPZ=${GT_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_gt_scene00180_20260714_v1/scene00180_point_aligned_plane_gt.npz}
RANSAC_NPZ=${RANSAC_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_ransac_scene00180_20260714_v1/scene00180_global_ransac_cc_full_pointcloud_editable_planes_data.npz}
SUPPORT_PRED_NPZ=${SUPPORT_PRED_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_support_scene00180_20260714_v1/scene00180_stage2_manual_merge_support_full_pointcloud_editable_planes_data.npz}
OUT_DIR=${OUT_DIR:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_support_audit_scene00180_20260714_v1}

cd "$PROJ"

if ! "$PYTHON" -c "import roma" >/dev/null 2>&1; then
  "$PYTHON" -m pip install --disable-pip-version-check roma==1.5.6
fi
for required in "$GT_NPZ" "$RANSAC_NPZ" "$SUPPORT_PRED_NPZ"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required NPZ: $required" >&2
    exit 2
  fi
done
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/run.log"

{
  echo "LightRecon3D P1 sparse-support identity audit"
  echo "project=$PROJ"
  echo "gt=$GT_NPZ"
  echo "ransac=$RANSAC_NPZ"
  echo "support_prediction=$SUPPORT_PRED_NPZ"
  echo "output=$OUT_DIR"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "roma=$($PYTHON -c 'import importlib.metadata as m; print(m.version("roma"))')"
  sha256sum "$GT_NPZ" "$RANSAC_NPZ" "$SUPPORT_PRED_NPZ"

  PYTHONUNBUFFERED=1 "$PYTHON" evaluate_global_plane_baselines.py \
    --gt_npz "$GT_NPZ" \
    --pred_npz "$GT_NPZ" "$RANSAC_NPZ" "$SUPPORT_PRED_NPZ" \
    --output_csv "$OUT_DIR/scene00180_support_conditioned_metrics.csv" \
    --match_iou 0.5 \
    --fragmentation_iou 0.1 \
    --min_observed_plane_points 64
} 2>&1 | tee "$LOG_FILE"

echo "P1 sparse-support identity audit complete: $OUT_DIR"
