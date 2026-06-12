#!/usr/bin/env bash
set -euo pipefail

cd /home/zhucy23u/projects/LightRecon3D

PYTHON=/data/zhucy23u/conda_envs/lightrecon/bin/python
INPUT_DIR=/data/zhucy23u/logs/learned_plane_params_v1/bounded_support_head_eval_val32_v4_diffsvd
OUTPUT_DIR=/data/zhucy23u/logs/learned_plane_params_v1/bounded_support_head_eval_val32_v4_diffsvd_visuals

mkdir -p "$OUTPUT_DIR"

for NPZ in "$INPUT_DIR"/*_bounded_support_head_assignment.npz; do
  STEM=$(basename "$NPZ" _assignment.npz)
  "$PYTHON" make_learned_plane_token_comparison.py \
    --learned_npz "$NPZ" \
    --learned_json "$INPUT_DIR/${STEM}.json" \
    --output_html "$OUTPUT_DIR/${STEM}_edit.html" \
    --edit_plane 0 \
    --edit_delta 0.25 \
    --max_display_points 18000
done
