#!/usr/bin/env bash
set -e

cd /gemini/code/LightRecon3D
source /gemini/code/LightRecon3D/scripts/activate_gemini_env.sh

declare -A BASES
BASES["scene00022_141"]="/gemini/data-1/Structured3D/scene_00022/2D_rendering/141/perspective/empty"
BASES["scene00198_4953"]="/gemini/data-1/Structured3D/scene_00198/2D_rendering/4953/perspective/empty"
BASES["scene00198_4955"]="/gemini/data-1/Structured3D/scene_00198/2D_rendering/4955/perspective/empty"
BASES["scene00199_322268"]="/gemini/data-1/Structured3D/scene_00199/2D_rendering/322268/perspective/empty"
BASES["scene00199_322269"]="/gemini/data-1/Structured3D/scene_00199/2D_rendering/322269/perspective/empty"
BASES["scene00027_822157"]="/gemini/data-1/Structured3D/scene_00027/2D_rendering/822157/perspective/empty"

for TAG in "${!BASES[@]}"; do
  BASE="${BASES[$TAG]}"
  echo "============================================================"
  echo "TAG  = $TAG"
  echo "BASE = $BASE"
  echo "============================================================"

  mapfile -t IMGS < <(find "$BASE" -maxdepth 2 -name "rgb_rawlight.png" | sort -V)

  echo "Found ${#IMGS[@]} images"
  printf '%s\n' "${IMGS[@]}"

  if [ "${#IMGS[@]}" -lt 2 ]; then
    echo "[Skip] less than 2 RGB images"
    continue
  fi

  CUDA_VISIBLE_DEVICES=0 python run_raw_dust3r_multiview.py \
    --weights_path /gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    --image_paths "${IMGS[@]}" \
    --output_dir /gemini/data-1/logs/raw_dust3r_multiview_${TAG}_rerun \
    --image_size 512 \
    --max_images 10 \
    --scene_graph complete \
    --batch_size 1 \
    --niter 300 \
    --lr 0.01 \
    --schedule cosine \
    --min_conf 1.0 \
    --max_points 800000 \
    2>&1 | tee /gemini/data-1/logs/raw_dust3r_multiview_${TAG}_rerun.log
done