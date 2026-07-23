#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lightrecon/bin/python}"
GLOBAL_CLOUD_NPZ="${GLOBAL_CLOUD_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p0_scene00180_20260714_v3/scene_00180__gemini_data-1_Structured3D_scene_00180_2D_rendering_445895_perspective_empty_dust3r_global_cloud_cache.npz}"
PLANE_PREDICTION_NPZ="${PLANE_PREDICTION_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_support_scene00180_20260714_v1/scene00180_stage2_manual_merge_support_full_pointcloud_editable_planes_data.npz}"
OUT_DIR="${OUT_DIR:-/gemini/data-1/lightrecon_runs/research_practice_line_smoke_scene00180_20260715_v1}"
RUN_LOG="${OUT_DIR}_run.log"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable project Python: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN explicitly; refusing to fall back to the base Miniconda environment." >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
  echo "Selected Python does not provide torch: $PYTHON_BIN" >&2
  echo "Use the LightRecon3D environment, normally /root/miniconda3/envs/lightrecon/bin/python." >&2
  exit 1
fi

if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUT_DIR" >&2
  exit 1
fi
if [[ -e "$RUN_LOG" ]]; then
  echo "Refusing to overwrite existing run log: $RUN_LOG" >&2
  exit 1
fi
if [[ ! -f "$GLOBAL_CLOUD_NPZ" ]]; then
  echo "Missing global cloud cache: $GLOBAL_CLOUD_NPZ" >&2
  exit 1
fi
if [[ ! -f "$PLANE_PREDICTION_NPZ" ]]; then
  echo "Missing plane prediction: $PLANE_PREDICTION_NPZ" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import roma' >/dev/null 2>&1; then
  echo "roma is missing; installing roma==1.5.6 into $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
  "$PYTHON_BIN" -m pip install --no-deps roma==1.5.6
fi

if ! "$PYTHON_BIN" -c 'import cv2' >/dev/null 2>&1; then
  echo "OpenCV is missing; installing a pre-5 headless build without changing NumPy"
  "$PYTHON_BIN" -m pip install --no-deps 'opencv-python-headless>=4.8,<5'
fi

exec > >(tee "$RUN_LOG") 2>&1

echo "LightRecon3D research-practice structural-line smoke"
echo "project=$PROJECT_DIR"
echo "global_cloud=$GLOBAL_CLOUD_NPZ"
echo "plane_prediction=$PLANE_PREDICTION_NPZ"
echo "output=$OUT_DIR"
echo "git_sha=$(git rev-parse HEAD)"
"$PYTHON_BIN" - <<'PY'
import importlib.metadata as metadata

import cv2
import numpy
import roma

print(f"cv2={cv2.__version__}")
print(f"numpy={numpy.__version__}")
print(f"roma={metadata.version('roma')}")
PY
sha256sum "$GLOBAL_CLOUD_NPZ" "$PLANE_PREDICTION_NPZ"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

"$PYTHON_BIN" extract_structural_lines.py \
  --global_cloud_npz "$GLOBAL_CLOUD_NPZ" \
  --plane_prediction_npz "$PLANE_PREDICTION_NPZ" \
  --output_dir "$OUT_DIR" \
  --min_conf 1.0 \
  --min_length_px 24 \
  --max_lines_per_view 256 \
  --sample_step_px 2 \
  --min_valid_samples 6 \
  --plane_side_offset_px 2 \
  --max_3d_gap_factor 8 \
  --association_filter all

"$PYTHON_BIN" - "$OUT_DIR/structural_lines_manifest.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
print(json.dumps({
    "manifest": path,
    "scene_key": manifest["scene_key"],
    "line_count": manifest["line_count"],
    "association_counts": manifest["association_counts"],
    "runtime_seconds": manifest["runtime_seconds"],
    "views": manifest["views"],
}, indent=2))
PY

echo "Research-practice structural-line smoke complete: $OUT_DIR"
