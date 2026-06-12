#!/usr/bin/env bash
set -euo pipefail

cd /home/zhucy23u/projects/LightRecon3D

PYTHON=/data/zhucy23u/conda_envs/lightrecon/bin/python
INPUT_DIR=/data/zhucy23u/logs/learned_plane_params_v1/stage1_teacher_train128_v1_line_v3
OUTPUT_DIR=/data/zhucy23u/logs/learned_plane_params_v1/bounded_support_head_train128_v4_diffsvd
INIT_CKPT=/data/zhucy23u/checkpoints/lightrecon_param/bounded_support_head_v3/train128_hardboundary.pt
CKPT=/data/zhucy23u/checkpoints/lightrecon_param/bounded_support_head_v4/train128_diffsvd.pt

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$CKPT")"

"$PYTHON" train_bounded_plane_support_head.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --load_checkpoint "$INIT_CKPT" \
  --save_checkpoint "$CKPT" \
  --num_planes 8 \
  --max_points_per_sample 18000 \
  --patch_pixel_size 0.08 \
  --min_patch_points 12 \
  --steps 1200 \
  --sample_batch_size 4 \
  --hidden_dim 192 \
  --lr 0.0002 \
  --weight_decay 0.0001 \
  --teacher_weight 1.0 \
  --smooth_weight 0.06 \
  --boundary_weight 0.18 \
  --hard_boundary_weight 0.10 \
  --hard_boundary_min_edge_conf 0.25 \
  --residual_weight 0.02 \
  --diff_plane_fit_weight 0.05 \
  --coverage_weight 0.02 \
  --min_plane_mass 30 \
  --cov_jitter 0.00001 \
  --fit_charbonnier_eps 0.001 \
  --grad_clip 1.0 \
  --line_smooth_suppress 0.60 \
  --boundary_margin 0.10 \
  --boundary_error_weight 3.5 \
  --boundary_error_min_edge_conf 0.25 \
  --support_logit_weight 0.15 \
  --boundary_support_logit_weight 0.05 \
  --outside_support_penalty 0.40 \
  --support_grid_size 0.055 \
  --support_dilate_cells 2 \
  --support_min_label_conf 0.60 \
  --log_every 100 \
  --seed 20260611
