#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}
ROOT=${ROOT:-/gemini/data-1/Structured3D}
RUNROOT=${RUNROOT:-/gemini/data-1/lightrecon_runs}
WEIGHTS=${WEIGHTS:-/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}

INDICES=${INDICES:-466,467,468,469,470,471,472,473,474,475}
DEBUG_ROOT=${DEBUG_ROOT:-$RUNROOT/debug_stage3_onegroup_dualview_v1}
STAGE3_VIS_ROOT=${STAGE3_VIS_ROOT:-$DEBUG_ROOT/stage3_global_visual_v1}

STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-$RUNROOT/checkpoints/stage1_large_80m_train2048_sharded_v1/best.pt}
FEATURE_CACHE=${FEATURE_CACHE:-$RUNROOT/cache/stage1_train2048_shards/val128.pt}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-$RUNROOT/stage2_region_merge_v1/checkpoint/best.pt}

export PROJ PYTHON ROOT RUNROOT WEIGHTS INDICES DEBUG_ROOT STAGE3_VIS_ROOT
export STAGE1_CHECKPOINT FEATURE_CACHE STAGE2_CHECKPOINT

cd "$PROJ"
mkdir -p \
  "$DEBUG_ROOT/stage1_teacher" \
  "$DEBUG_ROOT/stage2_merge" \
  "$DEBUG_ROOT/registry" \
  "$DEBUG_ROOT/logs" \
  "$STAGE3_VIS_ROOT/logs"

echo "[0/5] Environment"
echo "PROJ=$PROJ"
echo "PYTHON=$PYTHON"
echo "ROOT=$ROOT"
echo "RUNROOT=$RUNROOT"
echo "WEIGHTS=$WEIGHTS"
echo "DEBUG_ROOT=$DEBUG_ROOT"
echo "STAGE3_VIS_ROOT=$STAGE3_VIS_ROOT"
echo "INDICES=$INDICES"
"$PYTHON" -c "import sys; print(sys.executable)"
"$PYTHON" -c "import torch; print('torch', torch.__version__)"
"$PYTHON" -c "import roma; print('roma ok')"
"$PYTHON" tests/test_stage3_scene_plane_fusion_mapping.py

echo "[1/5] Export Stage1 dual-view support"
PYTHONUNBUFFERED=1 "$PYTHON" export_stage1_pred_support_teacher_npz.py \
  --root_dir "$ROOT" \
  --weights_path "$WEIGHTS" \
  --stage1_checkpoint "$STAGE1_CHECKPOINT" \
  --feature_cache_path "$FEATURE_CACHE" \
  --output_dir "$DEBUG_ROOT/stage1_teacher" \
  --split val \
  --indices "$INDICES" \
  --count 0 \
  --pair_strategy all \
  --max_planes 8 \
  --max_points 4000 \
  --export_second_view_support \
  --num_workers 0 \
  2>&1 | tee "$DEBUG_ROOT/logs/export_stage1_dualview.log"

echo "[2/5] Run Stage2 learned region merge"
PYTHONUNBUFFERED=1 "$PYTHON" export_stage2_learned_region_merge_editables.py \
  --input_dir "$DEBUG_ROOT/stage1_teacher" \
  --checkpoint "$STAGE2_CHECKPOINT" \
  --output_dir "$DEBUG_ROOT/stage2_merge" \
  --threshold 0.5 \
  --use_safety_gate \
  2>&1 | tee "$DEBUG_ROOT/logs/export_stage2_dualview.log"

echo "[3/5] Validate Stage3 view registry"
PYTHONUNBUFFERED=1 "$PYTHON" validate_stage3_view_registry.py \
  --input_dir "$DEBUG_ROOT/stage2_merge" \
  --output_json "$DEBUG_ROOT/registry/view_registry_summary.json" \
  --pattern "*_learned_region_merge_full_pointcloud_editable_planes_data.npz" \
  --check_files \
  2>&1 | tee "$DEBUG_ROOT/logs/validate_view_registry.log"

echo "[4/5] Run Stage3 DUSt3R global fusion visualization"
PYTHONUNBUFFERED=1 "$PYTHON" export_stage3_scene_plane_fusion.py \
  --input_dir "$DEBUG_ROOT/stage2_merge" \
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
    for rec in row.get("mapping_records", [])[:5]:
        print("mapping:", {
            "source_counts": rec.get("source_counts"),
            "kept_counts": rec.get("kept_counts"),
            "registered_view_indices": rec.get("registered_view_indices"),
            "dropped_total": rec.get("dropped_total"),
            "mean_kept_conf": rec.get("mean_kept_conf"),
        })
print("outputs:")
for pattern in ("*.html", "*.ply", "*.npz"):
    for path in sorted(root.glob(pattern)):
        print(path)
PY

echo "Done."
