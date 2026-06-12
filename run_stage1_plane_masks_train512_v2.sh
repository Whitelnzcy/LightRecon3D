#!/usr/bin/env bash
set -euo pipefail

cd /home/zhucy23u/projects/LightRecon3D

PYTHON=/data/zhucy23u/conda_envs/lightrecon/bin/python
SAVE=/data/zhucy23u/checkpoints/lightrecon_param/stage1_plane_masks_train512_v2
LOG=/data/zhucy23u/logs/learned_plane_params_v1/stage1_plane_masks_train512_v2.log

mkdir -p "$SAVE"
mkdir -p "$(dirname "$LOG")"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" train_stage1_plane_masks.py \
  --root_dir /data/zhucy23u/datasets/Structured3D \
  --weights_path /data/zhucy23u/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --save_dir "$SAVE" \
  --init_checkpoint /data/zhucy23u/checkpoints/lightrecon_param/stage1_plane_masks_train128_v1/best.pt \
  --input_mode single \
  --small_train_size 512 \
  --small_val_size 64 \
  --num_workers 4 \
  --num_epochs 8 \
  --batch_size 1 \
  --num_queries 8 \
  --hidden_dim 256 \
  --decoder_layers 3 \
  --decoder_heads 8 \
  --lr 0.00015 \
  --weight_decay 0.0001 \
  --mask_bce_weight 1.0 \
  --mask_dice_weight 2.0 \
  --existence_weight 1.0 \
  --partition_weight 1.0 \
  --boundary_weight 4.0 \
  --log_every 64 \
  --seed 20260612 \
  2>&1 | tee "$LOG"
