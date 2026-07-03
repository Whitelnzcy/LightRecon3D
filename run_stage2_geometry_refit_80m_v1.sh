#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
ROOT=${ROOT:-/gemini/data-1/Structured3D}
WEIGHTS=${WEIGHTS:-/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}
RUNROOT=${RUNROOT:-/gemini/data-1/lightrecon_runs}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}

CKPT=${CKPT:-$RUNROOT/checkpoints/stage1_large_80m_train2048_sharded_v1/best.pt}
VALCACHE=${VALCACHE:-$RUNROOT/cache/stage1_train2048_shards/val128.pt}
INDICES=${INDICES:-7,10,25,42,44,74,86,105}
COUNT=${COUNT:-0}

TEACHER_DIR=${TEACHER_DIR:-$RUNROOT/outputs/stage2_geometry_refit_80m_v1_teacher}
EDITABLE_DIR=${EDITABLE_DIR:-$RUNROOT/outputs/stage2_geometry_refit_80m_v1_editables}
LOGDIR=${LOGDIR:-$RUNROOT/logs/stage2_geometry_refit_80m_v1}

cd "$PROJ"
mkdir -p "$TEACHER_DIR" "$EDITABLE_DIR" "$LOGDIR"

echo "[stage2 geometry] checkpoint=$CKPT"
echo "[stage2 geometry] output teacher=$TEACHER_DIR"
echo "[stage2 geometry] output editables=$EDITABLE_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 "$PYTHON" \
  export_stage1_pred_support_teacher_npz.py \
  --root_dir "$ROOT" \
  --weights_path "$WEIGHTS" \
  --stage1_checkpoint "$CKPT" \
  --feature_cache_path "$VALCACHE" \
  --output_dir "$TEACHER_DIR" \
  --split val \
  --indices "$INDICES" \
  --count "$COUNT" \
  --max_planes 8 \
  --min_plane_points 128 \
  --min_core_points 64 \
  --core_confidence 0.55 \
  --core_margin 0.12 \
  --enable_safe_geometry_merge \
  --geometry_merge_angle_deg 5.0 \
  --geometry_merge_offset 0.03 \
  --geometry_merge_residual 0.04 \
  --geometry_merge_max_boundary_rgb_edge 0.05 \
  --geometry_merge_min_area_ratio 0.05 \
  --geometry_merge_adjacency_radius 2 \
  --enable_duplicate_geometry_merge \
  --duplicate_merge_angle_deg 3.0 \
  --duplicate_merge_offset 0.02 \
  --duplicate_merge_residual 0.025 \
  --duplicate_merge_min_area_ratio 0.015 \
  --max_points 28000 \
  --num_workers 2 \
  2>&1 | tee "$LOGDIR/export_teacher.log"

PYTHONUNBUFFERED=1 "$PYTHON" \
  export_stage2_geometry_refit_editables.py \
  --input_dir "$TEACHER_DIR" \
  --output_dir "$EDITABLE_DIR" \
  --pattern "*_stage1_teacher_full_pointcloud_editable_planes_data.npz" \
  --edit_plane largest \
  --edit_delta 0.25 \
  --max_display_points 28000 \
  2>&1 | tee "$LOGDIR/export_editables.log"

echo "[stage2 geometry] done"
echo "$EDITABLE_DIR"
