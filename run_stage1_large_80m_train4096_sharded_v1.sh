#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
ROOT=${ROOT:-/gemini/data-1/Structured3D}
WEIGHTS=${WEIGHTS:-/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}
RUNROOT=${RUNROOT:-/gemini/data-1/lightrecon_runs}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}

TRAIN_SIZE=${TRAIN_SIZE:-4096}
VAL_SIZE=${VAL_SIZE:-256}
TRAIN_SHARDS=${TRAIN_SHARDS:-16}
RUN_NAME=${RUN_NAME:-stage1_large_80m_train4096_sharded_v1}

CACHE_DIR=${CACHE_DIR:-$RUNROOT/cache/$RUN_NAME}
LOGDIR=${LOGDIR:-$RUNROOT/logs/$RUN_NAME}
SAVE_DIR=${SAVE_DIR:-$RUNROOT/checkpoints/$RUN_NAME}
VAL_CACHE=${VAL_CACHE:-$CACHE_DIR/val${VAL_SIZE}.pt}
RESUME_CKPT=${RESUME_CKPT:-$RUNROOT/checkpoints/stage1_large_80m_train2048_sharded_v1/best.pt}

cd "$PROJ"
mkdir -p "$CACHE_DIR" "$LOGDIR" "$SAVE_DIR"

echo "[config] project=$PROJ"
echo "[config] dataset=$ROOT"
echo "[config] run=$RUN_NAME train=$TRAIN_SIZE val=$VAL_SIZE shards=$TRAIN_SHARDS"
echo "[config] resume=$RESUME_CKPT"

if [[ ! -f "$VAL_CACHE" ]]; then
  echo "[cache] building val cache: $VAL_CACHE"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 "$PYTHON" \
    cache_stage1_multiscale_symref.py \
    --root_dir "$ROOT" \
    --weights_path "$WEIGHTS" \
    --output_path "$VAL_CACHE" \
    --small_train_size 1 \
    --small_val_size "$VAL_SIZE" \
    --sampling_strategy space_balanced \
    --skip_train \
    --num_workers 4 \
    --batch_size 1 \
    --log_every 16 \
    --flush_every 32 \
    --resume_partial \
    2>&1 | tee "$LOGDIR/cache_val.log"
else
  echo "[cache] val cache exists: $VAL_CACHE"
fi

for shard in $(seq 0 $((TRAIN_SHARDS - 1))); do
  shard_path="$CACHE_DIR/train_${shard}_of_${TRAIN_SHARDS}.pt"
  if [[ -f "$shard_path" ]]; then
    echo "[cache] train shard exists: $shard_path"
    continue
  fi
  echo "[cache] building train shard $shard/$((TRAIN_SHARDS - 1)): $shard_path"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 "$PYTHON" \
    cache_stage1_multiscale_symref.py \
    --root_dir "$ROOT" \
    --weights_path "$WEIGHTS" \
    --output_path "$shard_path" \
    --small_train_size "$TRAIN_SIZE" \
    --small_val_size 1 \
    --sampling_strategy space_balanced \
    --skip_val \
    --train_num_shards "$TRAIN_SHARDS" \
    --train_shard_index "$shard" \
    --num_workers 4 \
    --batch_size 1 \
    --log_every 16 \
    --flush_every 32 \
    --resume_partial \
    2>&1 | tee "$LOGDIR/cache_train_shard_${shard}.log"
done

echo "[train] starting 80M Stage1 training"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 "$PYTHON" \
  train_stage1_multiscale_pair.py \
  --feature_cache_glob "$CACHE_DIR/train_*_of_${TRAIN_SHARDS}.pt" \
  --val_cache_path "$VAL_CACHE" \
  --save_dir "$SAVE_DIR" \
  --resume_checkpoint "$RESUME_CKPT" \
  --allow_partial_resume \
  --num_epochs 32 \
  --batch_size 1 \
  --num_queries 12 \
  --hidden_dim 512 \
  --decoder_layers 6 \
  --decoder_heads 8 \
  --decoder_ffn_multiplier 4 \
  --fuse_refine_blocks 1 \
  --pixel_refine_blocks 1 \
  --output_size 128 \
  --use_geometry \
  --use_masked_query_refine \
  --lr 0.000035 \
  --min_lr 0.000001 \
  --weight_decay 0.0001 \
  --existence_weight 0.15 \
  --class_score_weight 0.25 \
  --query_margin_weight 0.1 \
  --unmatched_query_weight 0.05 \
  --query_separation_weight 0.05 \
  --ownership_loss_weight 0.05 \
  --aux32_weight 0.15 \
  --aux64_weight 0.35 \
  --train_hflip_prob 0.5 \
  --rgb_jitter_strength 0.18 \
  --eval_before_train \
  --log_every 32 \
  --seed 20260703 \
  2>&1 | tee "$LOGDIR/train.log"

echo "[done] best checkpoint should be under $SAVE_DIR"
