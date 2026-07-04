#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
ROOT=${ROOT:-/gemini/data-1/Structured3D}
WEIGHTS=${WEIGHTS:-/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}
RUNROOT=${RUNROOT:-/gemini/data-1/lightrecon_runs}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}

STAGE1_CKPT=${STAGE1_CKPT:-$RUNROOT/checkpoints/stage1_large_80m_train2048_sharded_v1/best.pt}
TRAIN_CACHE=${TRAIN_CACHE:-$RUNROOT/cache/stage1_large_80m_train4096_sharded_v1/train_0_of_16.pt}
VAL_CACHE=${VAL_CACHE:-$RUNROOT/cache/stage1_train2048_shards/val128.pt}

TRAIN_COUNT=${TRAIN_COUNT:-512}
VAL_COUNT=${VAL_COUNT:-128}
VIS_INDICES=${VIS_INDICES:-"7,10,25,42,44,74,86,105"}

OUTROOT=${OUTROOT:-$RUNROOT/stage2_region_merge_v1}
TRAIN_TEACHER=$OUTROOT/train_teacher
VAL_TEACHER=$OUTROOT/val_teacher
CKPT_DIR=$OUTROOT/checkpoint
MERGED_NPZ=$OUTROOT/learned_merge_npz
HTML_DIR=$OUTROOT/html
LOGDIR=$OUTROOT/logs

cd "$PROJ"
mkdir -p "$TRAIN_TEACHER" "$VAL_TEACHER" "$CKPT_DIR" "$MERGED_NPZ" "$HTML_DIR" "$LOGDIR"

echo "[1/5] Export Stage1 train teachers"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 "$PYTHON" \
  export_stage1_pred_support_teacher_npz.py \
  --root_dir "$ROOT" \
  --weights_path "$WEIGHTS" \
  --stage1_checkpoint "$STAGE1_CKPT" \
  --feature_cache_path "$TRAIN_CACHE" \
  --output_dir "$TRAIN_TEACHER" \
  --split train \
  --start_idx 0 \
  --count "$TRAIN_COUNT" \
  --max_planes 8 \
  --max_points 24000 \
  2>&1 | tee "$LOGDIR/export_train_teacher.log"

echo "[2/5] Export Stage1 val teachers"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 "$PYTHON" \
  export_stage1_pred_support_teacher_npz.py \
  --root_dir "$ROOT" \
  --weights_path "$WEIGHTS" \
  --stage1_checkpoint "$STAGE1_CKPT" \
  --feature_cache_path "$VAL_CACHE" \
  --output_dir "$VAL_TEACHER" \
  --split val \
  --start_idx 0 \
  --count "$VAL_COUNT" \
  --max_planes 8 \
  --max_points 24000 \
  2>&1 | tee "$LOGDIR/export_val_teacher.log"

echo "[3/5] Train learned Stage2 region merge"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 "$PYTHON" \
  train_stage2_region_merge_net.py \
  --train_dir "$TRAIN_TEACHER" \
  --val_dir "$VAL_TEACHER" \
  --output_dir "$CKPT_DIR" \
  --epochs 120 \
  --hidden_dim 256 \
  --depth 4 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --threshold 0.5 \
  2>&1 | tee "$LOGDIR/train_region_merge.log"

echo "[4/5] Apply learned merge/refit on selected val samples"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 "$PYTHON" \
  export_stage1_pred_support_teacher_npz.py \
  --root_dir "$ROOT" \
  --weights_path "$WEIGHTS" \
  --stage1_checkpoint "$STAGE1_CKPT" \
  --feature_cache_path "$VAL_CACHE" \
  --output_dir "$VAL_TEACHER/selected" \
  --split val \
  --indices "$VIS_INDICES" \
  --max_planes 8 \
  --max_points 24000 \
  2>&1 | tee "$LOGDIR/export_selected_teacher.log"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 "$PYTHON" \
  export_stage2_learned_region_merge_editables.py \
  --input_dir "$VAL_TEACHER/selected" \
  --checkpoint "$CKPT_DIR/best.pt" \
  --output_dir "$MERGED_NPZ" \
  --threshold 0.5 \
  --use_safety_gate \
  2>&1 | tee "$LOGDIR/apply_learned_merge.log"

echo "[5/5] Export editable HTML/PLY"
PYTHONUNBUFFERED=1 "$PYTHON" \
  export_stage2_geometry_refit_editables.py \
  --input_dir "$MERGED_NPZ" \
  --output_dir "$HTML_DIR" \
  --pattern "*_learned_region_merge_full_pointcloud_editable_planes_data.npz" \
  --max_display_points 18000 \
  2>&1 | tee "$LOGDIR/export_html.log"

echo "Done."
echo "Checkpoint: $CKPT_DIR/best.pt"
echo "HTML: $HTML_DIR"
