#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}
INPUT_DIR=${INPUT_DIR:-/gemini/data-1/lightrecon_runs/stage3_val_showcase_v1/group_000_pairs_10/stage2_merge}
GLOBAL_CLOUD_CACHE=${GLOBAL_CLOUD_CACHE:-/gemini/data-1/lightrecon_runs/plane_feedback_p0_scene00180_20260714_v3/scene_00180__gemini_data-1_Structured3D_scene_00180_2D_rendering_445895_perspective_empty_dust3r_global_cloud_cache.npz}
GT_NPZ=${GT_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_gt_scene00180_20260714_v1/scene00180_point_aligned_plane_gt.npz}
RANSAC_NPZ=${RANSAC_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_ransac_scene00180_20260714_v1/scene00180_global_ransac_cc_full_pointcloud_editable_planes_data.npz}
MANUAL_NPZ=${MANUAL_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p0_scene00180_20260714_v3/scene_00180__gemini_data-1_Structured3D_scene_00180_2D_rendering_445895_perspective_empty_stage3_dust3r_plane_feedback_full_pointcloud_editable_planes_data.npz}
OUT_DIR=${OUT_DIR:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_direct_svd_scene00180_20260714_v1}

cd "$PROJ"

ensure_python_package() {
  local module=$1
  local distribution=$2
  local version=$3
  if ! "$PYTHON" -c "import ${module}; import importlib.metadata as m; assert m.version('${distribution}') == '${version}'" >/dev/null 2>&1; then
    "$PYTHON" -m pip install --disable-pip-version-check "${distribution}==${version}"
  fi
  "$PYTHON" -c "import importlib.metadata as m; print('${distribution}=' + m.version('${distribution}'))"
}

ensure_python_package roma roma 1.5.6
ensure_python_package trimesh trimesh 4.9.0

for required in "$INPUT_DIR" "$GLOBAL_CLOUD_CACHE" "$GT_NPZ" "$RANSAC_NPZ" "$MANUAL_NPZ"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/run.log"

{
  echo "LightRecon3D P1 direct-global-SVD and repeated-record identity audit"
  echo "project=$PROJ"
  echo "input=$INPUT_DIR"
  echo "global_cloud_cache=$GLOBAL_CLOUD_CACHE"
  echo "gt=$GT_NPZ"
  echo "ransac=$RANSAC_NPZ"
  echo "manual=$MANUAL_NPZ"
  echo "output=$OUT_DIR"
  echo "git_sha=$(git rev-parse HEAD)"
  sha256sum "$GLOBAL_CLOUD_CACHE" "$GT_NPZ" "$RANSAC_NPZ" "$MANUAL_NPZ"

  PYTHONUNBUFFERED=1 "$PYTHON" export_stage3_scene_plane_fusion.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUT_DIR" \
    --pattern "*_learned_region_merge_full_pointcloud_editable_planes_data.npz" \
    --group_by pair_group \
    --fusion_mode dust3r_global \
    --merge_mode none \
    --global_cloud_cache "$GLOBAL_CLOUD_CACHE" \
    --include_second_view \
    --global_min_conf 0.0 \
    --min_points 64 \
    --max_display_points 32000

  mapfile -t direct_files < <(find "$OUT_DIR" -maxdepth 1 -type f -name '*_stage3_dust3r_global_fusion_full_pointcloud_editable_planes_data.npz' -print)
  if [[ ${#direct_files[@]} -ne 1 ]]; then
    echo "Expected one direct-SVD NPZ, found ${#direct_files[@]}" >&2
    exit 2
  fi
  DIRECT_NPZ=${direct_files[0]}
  echo "direct_svd=$DIRECT_NPZ"
  sha256sum "$DIRECT_NPZ"

  PYTHONUNBUFFERED=1 "$PYTHON" evaluate_support_record_partitions.py \
    --gt_npz "$GT_NPZ" \
    --support_reference_npz "$MANUAL_NPZ" \
    --pred_npz "$GT_NPZ" "$RANSAC_NPZ" "$MANUAL_NPZ" "$DIRECT_NPZ" \
    --method_names point_aligned_gt_oracle global_ransac_cc stage2_manual_merge_support stage2_support_direct_global_svd \
    --output_json "$OUT_DIR/scene00180_support_record_partition_audit.json" \
    --match_iou 0.5 \
    --fragmentation_iou 0.1 \
    --min_observed_plane_points 64 \
    --allow_legacy_cache_xy
} 2>&1 | tee "$LOG_FILE"

echo "P1 direct-global-SVD audit complete: $OUT_DIR"
