#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}
GLOBAL_CLOUD_NPZ=${GLOBAL_CLOUD_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p0_scene00180_20260714_v3/scene_00180__gemini_data-1_Structured3D_scene_00180_2D_rendering_445895_perspective_empty_dust3r_global_cloud_cache.npz}
GT_NPZ=${GT_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_gt_scene00180_20260714_v1/scene00180_point_aligned_plane_gt.npz}
OUT_DIR=${OUT_DIR:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_ransac_scene00180_20260714_v1}

cd "$PROJ"

if ! "$PYTHON" -c "import roma" >/dev/null 2>&1; then
  "$PYTHON" -m pip install --disable-pip-version-check roma==1.5.6
fi
if [[ ! -f "$GLOBAL_CLOUD_NPZ" ]]; then
  echo "Missing global-cloud cache: $GLOBAL_CLOUD_NPZ" >&2
  exit 2
fi
if [[ ! -f "$GT_NPZ" ]]; then
  echo "Missing point-aligned GT: $GT_NPZ" >&2
  exit 2
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/run.log"
PRED_NPZ="$OUT_DIR/scene00180_global_ransac_cc_full_pointcloud_editable_planes_data.npz"

{
  echo "LightRecon3D P1 identical-cache RANSAC baseline"
  echo "project=$PROJ"
  echo "global_cloud=$GLOBAL_CLOUD_NPZ"
  echo "gt=$GT_NPZ"
  echo "output=$OUT_DIR"
  echo "git_sha=$(git rev-parse HEAD)"

  PYTHONUNBUFFERED=1 "$PYTHON" global_plane_baselines.py \
    --global_cloud_npz "$GLOBAL_CLOUD_NPZ" \
    --output_dir "$OUT_DIR" \
    --scene_key scene00180 \
    --min_conf 1.0 \
    --distance_threshold 0.03 \
    --iterations 300 \
    --min_inliers 2000 \
    --cluster_radius 0.08 \
    --min_component_points 1000 \
    --max_planes 32 \
    --seed 0 \
    --hypothesis_max_points 50000 \
    --component_exact_max_points 20000

  PYTHONUNBUFFERED=1 "$PYTHON" evaluate_global_plane_baselines.py \
    --gt_npz "$GT_NPZ" \
    --pred_npz "$GT_NPZ" "$PRED_NPZ" \
    --output_csv "$OUT_DIR/scene00180_plane_metrics.csv"
} 2>&1 | tee "$LOG_FILE"

echo "P1 RANSAC baseline complete: $OUT_DIR"
