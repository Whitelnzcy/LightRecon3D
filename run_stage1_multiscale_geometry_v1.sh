#!/usr/bin/env bash
set -euo pipefail

cd /home/zhucy23u/projects/LightRecon3D

PYTHON=${PYTHON:-/data/zhucy23u/conda_envs/lightrecon/bin/python}
ROOT_DIR=${ROOT_DIR:-/data/zhucy23u/datasets/Structured3D}
WEIGHTS=${WEIGHTS:-/data/zhucy23u/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}

CACHE=${CACHE:-/data/zhucy23u/logs/learned_plane_params_v1/stage1_multiscale_symref_balanced512_geometry_v1.pt}
SAVE=${SAVE:-/data/zhucy23u/checkpoints/lightrecon_param/stage1_multiscale_symref_balanced512_geometry_v1}
EVAL=${EVAL:-/data/zhucy23u/logs/learned_plane_params_v1/eval_stage1_multiscale_symref_balanced512_geometry_v1}
INIT=${INIT:-/data/zhucy23u/checkpoints/lightrecon_param/stage1_multiscale_symref_balanced512_v1/best.pt}

mkdir -p "$(dirname "$CACHE")" "$SAVE" "$EVAL"

if [[ ! -f "$CACHE" ]]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" cache_stage1_multiscale_symref.py \
    --root_dir "$ROOT_DIR" \
    --weights_path "$WEIGHTS" \
    --output_path "$CACHE" \
    --small_train_size 512 \
    --small_val_size 128 \
    --sampling_strategy space_balanced \
    --rgb_cache_size 128 \
    --num_workers 4 \
    --batch_size 1 \
    --log_every 32
fi

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" train_stage1_multiscale_pair_finetune.py \
  --feature_cache_path "$CACHE" \
  --save_dir "$SAVE" \
  --init_checkpoint "$INIT" \
  --num_epochs 8 \
  --batch_size 2 \
  --num_queries 8 \
  --hidden_dim 256 \
  --decoder_layers 3 \
  --decoder_heads 8 \
  --output_size 128 \
  --use_geometry \
  --lr 0.0001 \
  --min_lr 0.000005 \
  --weight_decay 0.0001 \
  --existence_weight 0.1 \
  --aux32_weight 0.25 \
  --aux64_weight 0.5 \
  --log_every 32 \
  --seed 20260619

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" evaluate_stage1_multiscale_pair.py \
  --checkpoint "$SAVE/best.pt" \
  --feature_cache_path "$CACHE" \
  --output_dir "$EVAL" \
  --split val \
  --batch_size 2 \
  --log_every 32
