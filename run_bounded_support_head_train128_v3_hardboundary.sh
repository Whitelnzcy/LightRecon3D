#!/usr/bin/env bash
set -euo pipefail

cd /home/zhucy23u/projects/LightRecon3D

PYTHON=/data/zhucy23u/conda_envs/lightrecon/bin/python
INPUT_DIR=/data/zhucy23u/logs/learned_plane_params_v1/stage1_teacher_train128_v1_line_v3
OUTPUT_DIR=/data/zhucy23u/logs/learned_plane_params_v1/bounded_support_head_train128_v3_hardboundary
INIT_CKPT=/data/zhucy23u/checkpoints/lightrecon_param/bounded_support_head_v2/train128_boundaryweighted.pt
CKPT=/data/zhucy23u/checkpoints/lightrecon_param/bounded_support_head_v3/train128_hardboundary.pt

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$CKPT")"

/usr/bin/time -p "$PYTHON" train_bounded_plane_support_head.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --load_checkpoint "$INIT_CKPT" \
  --save_checkpoint "$CKPT" \
  --num_planes 8 \
  --max_points_per_sample 18000 \
  --patch_pixel_size 0.08 \
  --min_patch_points 12 \
  --steps 2200 \
  --sample_batch_size 4 \
  --hidden_dim 192 \
  --lr 0.0003 \
  --weight_decay 0.0001 \
  --teacher_weight 1.10 \
  --smooth_weight 0.05 \
  --boundary_weight 0.12 \
  --hard_boundary_weight 0.80 \
  --hard_boundary_min_edge_conf 0.25 \
  --residual_weight 0.025 \
  --line_smooth_suppress 0.60 \
  --boundary_margin 0.08 \
  --boundary_error_weight 3.5 \
  --boundary_error_min_edge_conf 0.25 \
  --support_logit_weight 0.12 \
  --boundary_support_logit_weight 0.04 \
  --outside_support_penalty 0.45 \
  --support_grid_size 0.055 \
  --support_dilate_cells 2 \
  --support_min_label_conf 0.60 \
  --log_every 100 \
  --seed 20260611
