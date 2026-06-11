#!/usr/bin/env bash
set -euo pipefail

cd /home/zhucy23u/projects/LightRecon3D

PYTHON=/data/zhucy23u/conda_envs/lightrecon/bin/python
INPUT_DIR=/data/zhucy23u/logs/learned_plane_params_v1/stage1_teacher_val32_v1_line_v3
OUTPUT_DIR=/data/zhucy23u/logs/learned_plane_params_v1/bounded_support_head_eval_val32_v1
CKPT=/data/zhucy23u/checkpoints/lightrecon_param/bounded_support_head_v1/train128.pt

mkdir -p "$OUTPUT_DIR"

"$PYTHON" train_bounded_plane_support_head.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --load_checkpoint "$CKPT" \
  --export_only \
  --num_planes 8 \
  --max_points_per_sample 18000 \
  --patch_pixel_size 0.08 \
  --min_patch_points 12 \
  --hidden_dim 192 \
  --support_logit_weight 0.15 \
  --boundary_support_logit_weight 0.05 \
  --outside_support_penalty 0.40 \
  --support_grid_size 0.055 \
  --support_dilate_cells 2 \
  --support_min_label_conf 0.60 \
  --seed 20260608
