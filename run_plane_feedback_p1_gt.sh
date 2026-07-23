#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}
GLOBAL_CLOUD_NPZ=${GLOBAL_CLOUD_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p0_scene00180_20260714_v3/scene_00180__gemini_data-1_Structured3D_scene_00180_2D_rendering_445895_perspective_empty_dust3r_global_cloud_cache.npz}
OUT_DIR=${OUT_DIR:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_gt_scene00180_20260714_v1}

cd "$PROJ"

if ! "$PYTHON" -c "import roma" >/dev/null 2>&1; then
  "$PYTHON" -m pip install --disable-pip-version-check roma==1.5.6
fi
"$PYTHON" -c "import cv2, numpy, roma; print('cv2=' + cv2.__version__); print('numpy=' + numpy.__version__)"

if [[ ! -f "$GLOBAL_CLOUD_NPZ" ]]; then
  echo "Missing P0 global-cloud cache: $GLOBAL_CLOUD_NPZ" >&2
  exit 2
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/run.log"

{
  echo "LightRecon3D P1 point-aligned Structured3D plane GT"
  echo "project=$PROJ"
  echo "global_cloud=$GLOBAL_CLOUD_NPZ"
  echo "output=$OUT_DIR"
  echo "git_sha=$(git rev-parse HEAD)"

  PYTHONUNBUFFERED=1 "$PYTHON" build_structured3d_point_aligned_gt.py \
    --global_cloud_npz "$GLOBAL_CLOUD_NPZ" \
    --output_npz "$OUT_DIR/scene00180_point_aligned_plane_gt.npz" \
    --output_ply "$OUT_DIR/scene00180_point_aligned_plane_gt.ply" \
    --output_manifest "$OUT_DIR/scene00180_point_aligned_plane_gt_manifest.json" \
    --min_conf 1.0 \
    --min_plane_points 64 \
    --image_size 512 \
    --patch_size 16 \
    --boundary_ignore_radius 1
} 2>&1 | tee "$LOG_FILE"

echo "P1 point-aligned GT complete: $OUT_DIR"
