#!/usr/bin/env bash
set -euo pipefail

cd /home/zhucy23u/projects/LightRecon3D

OUT=/data/zhucy23u/logs/learned_plane_params_v1/bounded_support_head_eval_val32_v3_hardboundary_visuals
SRC=/data/zhucy23u/logs/learned_plane_params_v1/bounded_support_head_eval_val32_v3_hardboundary
mkdir -p "${OUT}"

count=0
for json_path in "${SRC}"/*_bounded_support_head.json; do
  stem=$(basename "${json_path}" _bounded_support_head.json)
  npz_path="${SRC}/${stem}_bounded_support_head_assignment.npz"
  html_path="${OUT}/${stem}_bounded_support_head_edit.html"
  /data/zhucy23u/conda_envs/lightrecon/bin/python make_learned_plane_token_comparison.py \
    --learned_npz "${npz_path}" \
    --learned_json "${json_path}" \
    --output_html "${html_path}" \
    --edit_plane largest \
    --edit_delta 0.25 \
    --max_display_points 28000
  count=$((count + 1))
  if [ "${count}" -ge 12 ]; then
    break
  fi
done
echo "visualized=${count}"
