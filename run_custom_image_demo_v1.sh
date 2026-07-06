#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}
RUNROOT=${RUNROOT:-/gemini/data-1/lightrecon_runs}
IMAGE_DIR=${IMAGE_DIR:-/gemini/data-1/lightrecon_custom_demo/images}
OUT_ROOT=${OUT_ROOT:-$RUNROOT/custom_image_demo_v1}
WEIGHTS=${WEIGHTS:-/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-$RUNROOT/checkpoints/stage1_large_80m_train2048_sharded_v1/best.pt}
FEATURE_CACHE=${FEATURE_CACHE:-$RUNROOT/cache/stage1_train2048_shards/val128.pt}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-$RUNROOT/stage2_region_merge_v1/checkpoint/best.pt}
PAIR_STRATEGY=${PAIR_STRATEGY:-all}
SCENE_NAME=${SCENE_NAME:-custom_scene}
MAX_POINTS=${MAX_POINTS:-4000}
STAGE3_VIS_ROOT=${STAGE3_VIS_ROOT:-$OUT_ROOT/stage3_global_visual_v1}

export PROJ PYTHON RUNROOT IMAGE_DIR OUT_ROOT WEIGHTS
export STAGE1_CHECKPOINT FEATURE_CACHE STAGE2_CHECKPOINT STAGE3_VIS_ROOT
export PYTHONPATH="$PROJ:$PROJ/dust3r:${PYTHONPATH:-}"

cd "$PROJ"
mkdir -p \
  "$OUT_ROOT/stage1_teacher" \
  "$OUT_ROOT/stage2_merge" \
  "$OUT_ROOT/registry" \
  "$OUT_ROOT/logs" \
  "$STAGE3_VIS_ROOT/logs"

echo "[0/5] Environment"
echo "PROJ=$PROJ"
echo "PYTHON=$PYTHON"
echo "IMAGE_DIR=$IMAGE_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "WEIGHTS=$WEIGHTS"
echo "STAGE1_CHECKPOINT=$STAGE1_CHECKPOINT"
echo "STAGE2_CHECKPOINT=$STAGE2_CHECKPOINT"
"$PYTHON" -c "import sys; print(sys.executable)"
"$PYTHON" -c "import torch; print('torch', torch.__version__)"
"$PYTHON" -c "import roma; print('roma ok')"

echo "[1/5] Export custom Stage1 dual-view support"
PYTHONUNBUFFERED=1 "$PYTHON" export_custom_image_stage1_support_npz.py \
  --image_dir "$IMAGE_DIR" \
  --weights_path "$WEIGHTS" \
  --stage1_checkpoint "$STAGE1_CHECKPOINT" \
  --feature_cache_path "$FEATURE_CACHE" \
  --output_dir "$OUT_ROOT/stage1_teacher" \
  --scene_name "$SCENE_NAME" \
  --pair_group "$IMAGE_DIR" \
  --pair_strategy "$PAIR_STRATEGY" \
  --max_planes 8 \
  --max_points "$MAX_POINTS" \
  2>&1 | tee "$OUT_ROOT/logs/export_custom_stage1.log"

echo "[2/5] Run Stage2 learned region merge"
PYTHONUNBUFFERED=1 "$PYTHON" export_stage2_learned_region_merge_editables.py \
  --input_dir "$OUT_ROOT/stage1_teacher" \
  --checkpoint "$STAGE2_CHECKPOINT" \
  --output_dir "$OUT_ROOT/stage2_merge" \
  --threshold 0.5 \
  --use_safety_gate \
  2>&1 | tee "$OUT_ROOT/logs/export_stage2.log"

echo "[3/5] Validate Stage3 view registry"
PYTHONUNBUFFERED=1 "$PYTHON" validate_stage3_view_registry.py \
  --input_dir "$OUT_ROOT/stage2_merge" \
  --output_json "$OUT_ROOT/registry/view_registry_summary.json" \
  --pattern "*_learned_region_merge_full_pointcloud_editable_planes_data.npz" \
  --check_files \
  2>&1 | tee "$OUT_ROOT/logs/validate_view_registry.log"

echo "[4/5] Run Stage3 DUSt3R global fusion visualization"
PYTHONUNBUFFERED=1 "$PYTHON" export_stage3_scene_plane_fusion.py \
  --input_dir "$OUT_ROOT/stage2_merge" \
  --output_dir "$STAGE3_VIS_ROOT" \
  --pattern "*_learned_region_merge_full_pointcloud_editable_planes_data.npz" \
  --group_by pair_group \
  --fusion_mode dust3r_global \
  --weights_path "$WEIGHTS" \
  --include_second_view \
  --image_size 512 \
  --scene_graph complete \
  --batch_size 1 \
  --niter 300 \
  --lr 0.01 \
  --schedule cosine \
  --min_group_size 1 \
  --max_display_points 80000 \
  2>&1 | tee "$STAGE3_VIS_ROOT/logs/stage3_global_visual.log"

echo "[5/5] Manifest summary"
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["STAGE3_VIS_ROOT"])
manifest = root / "stage3_scene_fusion_manifest.json"
rows = json.load(open(manifest, encoding="utf-8"))
print("manifest:", manifest)
print("scenes:", len(rows))
for row in rows:
    print("scene_key:", row.get("scene_key"))
    print("views:", len(row.get("dust3r_view_registry", [])))
    print("loss:", row.get("dust3r_global_alignment_loss"))
    print("points:", row.get("points"))
    print("local/global/merged:", row.get("local_planes"), row.get("global_planes"), row.get("merged_pairs"))
    print("quality:", row.get("quality_summary"))
print("outputs:")
for pattern in ("*.html", "*.ply", "*.npz"):
    for path in sorted(root.glob(pattern)):
        print(path)
PY

echo "Done."
