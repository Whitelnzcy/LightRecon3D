#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}
GLOBAL_CLOUD_NPZ=${GLOBAL_CLOUD_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p0_scene00180_20260714_v3/scene_00180__gemini_data-1_Structured3D_scene_00180_2D_rendering_445895_perspective_empty_dust3r_global_cloud_cache.npz}
GT_NPZ=${GT_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_gt_scene00180_20260714_v1/scene00180_point_aligned_plane_gt.npz}
RANSAC_NPZ=${RANSAC_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_ransac_scene00180_20260714_v1/scene00180_global_ransac_cc_full_pointcloud_editable_planes_data.npz}
SUPPORT_NPZ=${SUPPORT_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p0_scene00180_20260714_v3/scene_00180__gemini_data-1_Structured3D_scene_00180_2D_rendering_445895_perspective_empty_stage3_dust3r_plane_feedback_full_pointcloud_editable_planes_data.npz}
OUT_DIR=${OUT_DIR:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_support_scene00180_20260714_v1}

cd "$PROJ"

if ! "$PYTHON" -c "import roma" >/dev/null 2>&1; then
  "$PYTHON" -m pip install --disable-pip-version-check roma==1.5.6
fi
for required in "$GLOBAL_CLOUD_NPZ" "$GT_NPZ" "$RANSAC_NPZ" "$SUPPORT_NPZ"; do
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
PRED_NPZ="$OUT_DIR/scene00180_stage2_manual_merge_support_full_pointcloud_editable_planes_data.npz"

{
  echo "LightRecon3D P1 identical-cache Stage2/manual-merge support baseline"
  echo "project=$PROJ"
  echo "global_cloud=$GLOBAL_CLOUD_NPZ"
  echo "support_prediction=$SUPPORT_NPZ"
  echo "gt=$GT_NPZ"
  echo "ransac=$RANSAC_NPZ"
  echo "output=$OUT_DIR"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "roma=$($PYTHON -c 'import importlib.metadata as m; print(m.version("roma"))')"

  PYTHONUNBUFFERED=1 "$PYTHON" lift_support_prediction_to_global_cache.py \
    --global_cloud_npz "$GLOBAL_CLOUD_NPZ" \
    --support_npz "$SUPPORT_NPZ" \
    --output_dir "$OUT_DIR" \
    --scene_key scene00180 \
    --method stage2_manual_merge_support \
    --min_conf 1.0 \
    --conflict_policy drop \
    --min_points_per_plane 3

  PYTHONUNBUFFERED=1 "$PYTHON" evaluate_global_plane_baselines.py \
    --gt_npz "$GT_NPZ" \
    --pred_npz "$GT_NPZ" "$RANSAC_NPZ" "$PRED_NPZ" \
    --output_csv "$OUT_DIR/scene00180_plane_metrics.csv"
} 2>&1 | tee "$LOG_FILE"

echo "P1 support baseline complete: $OUT_DIR"
