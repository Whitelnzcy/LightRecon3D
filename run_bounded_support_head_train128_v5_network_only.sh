#!/usr/bin/env bash
set -euo pipefail

cd /home/zhucy23u/projects/LightRecon3D

PYTHON=/data/zhucy23u/conda_envs/lightrecon/bin/python
INPUT_DIR=/data/zhucy23u/logs/learned_plane_params_v1/stage1_teacher_train128_v1_line_v3
OUTPUT_DIR=/data/zhucy23u/logs/learned_plane_params_v1/bounded_support_head_train128_v5_network_only
CKPT=/data/zhucy23u/checkpoints/lightrecon_param/bounded_support_head_v5/train128_network_only.pt

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$CKPT")"

"$PYTHON" train_bounded_plane_support_head.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --save_checkpoint "$CKPT" \
  --num_planes 8 \
  --max_points_per_sample 18000 \
  --patch_pixel_size 0.08 \
  --min_patch_points 12 \
  --hide_teacher_feature_conf \
  --disable_support_prior \
  --steps 2400 \
  --sample_batch_size 4 \
  --hidden_dim 192 \
  --lr 0.0007 \
  --weight_decay 0.0001 \
  --teacher_weight 1.0 \
  --smooth_weight 0.06 \
  --boundary_weight 0.18 \
  --hard_boundary_weight 0.10 \
  --hard_boundary_min_edge_conf 0.25 \
  --residual_weight 0.02 \
  --diff_plane_fit_weight 0.05 \
  --coverage_weight 0.02 \
  --min_plane_mass 300 \
  --cov_jitter 0.00001 \
  --fit_charbonnier_eps 0.001 \
  --use_patch_count_weight \
  --grad_clip 1.0 \
  --line_smooth_suppress 0.60 \
  --boundary_margin 0.10 \
  --boundary_error_weight 3.5 \
  --boundary_error_min_edge_conf 0.25 \
  --log_every 100 \
  --seed 20260611
