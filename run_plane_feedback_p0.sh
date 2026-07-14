#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}
INPUT_DIR=${INPUT_DIR:-/gemini/data-1/lightrecon_runs/stage3_val_showcase_v1/group_000_pairs_10/stage2_merge}
OUT_DIR=${OUT_DIR:-/gemini/data-1/lightrecon_runs/plane_feedback_p0_scene00180_20260714_v1}
WEIGHTS=${WEIGHTS:-/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}

cd "$PROJ"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Missing Stage2 input directory: $INPUT_DIR" >&2
  exit 2
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "Missing DUSt3R weights: $WEIGHTS" >&2
  exit 2
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/run.log"

{
  echo "LightRecon3D plane-feedback P0 smoke"
  echo "project=$PROJ"
  echo "input=$INPUT_DIR"
  echo "output=$OUT_DIR"
  echo "weights=$WEIGHTS"
  echo "git_sha=$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

  PYTHONUNBUFFERED=1 "$PYTHON" export_stage3_scene_plane_fusion.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUT_DIR" \
    --pattern "*_learned_region_merge_full_pointcloud_editable_planes_data.npz" \
    --group_by pair_group \
    --fusion_mode dust3r_global \
    --merge_mode manual \
    --weights_path "$WEIGHTS" \
    --include_second_view \
    --batch_size 1 \
    --niter 300 \
    --lr 0.01 \
    --schedule cosine \
    --global_min_conf 0.0 \
    --plane_feedback \
    --plane_feedback_niter 100 \
    --plane_feedback_lr 0.002 \
    --plane_feedback_weight 0.2 \
    --plane_feedback_huber_delta 0.01 \
    --plane_feedback_min_views 2 \
    --plane_feedback_min_points 64 \
    --plane_feedback_max_base_loss_increase 0.03 \
    --plane_feedback_min_relative_improvement 0.0001 \
    --plane_feedback_log_every 20
} 2>&1 | tee "$LOG_FILE"

echo "P0 smoke complete: $OUT_DIR"
