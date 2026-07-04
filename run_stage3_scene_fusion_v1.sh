#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
RUNROOT=${RUNROOT:-/gemini/data-1/lightrecon_runs}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}

STAGE2_ROOT=${STAGE2_ROOT:-$RUNROOT/stage2_region_merge_v1}
INPUT_DIR=${INPUT_DIR:-$STAGE2_ROOT/learned_merge_npz}
OUT_DIR=${OUT_DIR:-$RUNROOT/stage3_scene_fusion_v1}
LOGDIR=$OUT_DIR/logs

cd "$PROJ"
mkdir -p "$OUT_DIR" "$LOGDIR"

echo "[1/1] Fuse local Stage2 primitives into scene-level planes"
PYTHONUNBUFFERED=1 "$PYTHON" \
  export_stage3_scene_plane_fusion.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUT_DIR" \
  --pattern "*_learned_region_merge_full_pointcloud_editable_planes_data.npz" \
  --group_by reference_view \
  --min_group_size 1 \
  --max_angle_deg 8.0 \
  --max_offset 0.06 \
  --max_mutual_residual 0.06 \
  --max_centroid_distance 2.5 \
  --max_display_points 32000 \
  2>&1 | tee "$LOGDIR/stage3_scene_fusion.log"

echo "Done."
echo "Stage3 outputs: $OUT_DIR"
