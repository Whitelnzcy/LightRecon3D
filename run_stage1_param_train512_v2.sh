#!/usr/bin/env bash
set -euo pipefail

cd /home/zhucy23u/projects/LightRecon3D

PYTHON=/data/zhucy23u/conda_envs/lightrecon/bin/python
ROOT=/data/zhucy23u/datasets/Structured3D
WEIGHTS=/data/zhucy23u/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
INIT=/data/zhucy23u/checkpoints/lightrecon_param/geom_token_gt_stage1_v1/best.pth
SAVE=/data/zhucy23u/checkpoints/lightrecon_param/geom_token_gt_stage1_v2_train512
LOG=/data/zhucy23u/logs/learned_plane_params_v1/stage1_train512_v2.log

mkdir -p "$SAVE"
mkdir -p "$(dirname "$LOG")"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" train_plane_params.py \
  --root_dir "$ROOT" \
  --weights_path "$WEIGHTS" \
  --save_dir "$SAVE" \
  --init_checkpoint "$INIT" \
  --small_train_size 512 \
  --small_val_size 64 \
  --num_workers 4 \
  --num_epochs 5 \
  --batch_size 1 \
  --lr 0.00003 \
  --weight_decay 0.0001 \
  --grad_clip 1.0 \
  --log_every 64 \
  --run_val \
  --seed 20260612 \
  --param_head_type geom_token_point_anchor_conf \
  --param_normal_weight 1.0 \
  --param_offset_weight 0.25 \
  --point_plane_weight 0.15 \
  --point_plane_max_points 1024 \
  --point_plane_clip 0.25 \
  --confidence_weight 0.05 \
  --param_min_pixels 64 \
  --param_max_pixels_per_plane 2048 \
  --param_max_planes_per_image 8 \
  2>&1 | tee "$LOG"
