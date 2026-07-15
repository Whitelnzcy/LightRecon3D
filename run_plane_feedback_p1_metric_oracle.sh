#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}
GLOBAL_CLOUD_CACHE=${GLOBAL_CLOUD_CACHE:-/gemini/data-1/lightrecon_runs/plane_feedback_p0_scene00180_20260714_v3/scene_00180__gemini_data-1_Structured3D_scene_00180_2D_rendering_445895_perspective_empty_dust3r_global_cloud_cache.npz}
MANUAL_SUPPORT_NPZ=${MANUAL_SUPPORT_NPZ:-/gemini/data-1/lightrecon_runs/plane_feedback_p0_scene00180_20260714_v3/scene_00180__gemini_data-1_Structured3D_scene_00180_2D_rendering_445895_perspective_empty_stage3_dust3r_plane_feedback_full_pointcloud_editable_planes_data.npz}
OUT_DIR=${OUT_DIR:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_metric_oracle_scene00180_20260715_v1}

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

# Keep the server environment invariant requested for all DUSt3R-related runs.
ensure_python_package roma roma 1.5.6
"$PYTHON" -c "import cv2, numpy; print('cv2=' + cv2.__version__); print('numpy=' + numpy.__version__)"

for required in "$GLOBAL_CLOUD_CACHE" "$MANUAL_SUPPORT_NPZ"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR/manual_ba" "$OUT_DIR/oracle_ba"
LOG_FILE="$OUT_DIR/run.log"
METRIC_GT="$OUT_DIR/scene00180_metric_structural_gt.npz"
METRIC_GT_MANIFEST="$OUT_DIR/scene00180_metric_structural_gt_manifest.json"
MANUAL_BA="$OUT_DIR/manual_ba/scene00180_manual_planegraph_ba_v0_full_pointcloud_editable_planes_data.npz"
ORACLE_BA="$OUT_DIR/oracle_ba/scene00180_oracle_planegraph_ba_v0_full_pointcloud_editable_planes_data.npz"
METRICS_JSON="$OUT_DIR/scene00180_metric_geometry_metrics.json"

{
  echo "LightRecon3D P1 metric structural GT and oracle-identity PlaneGraph-BA"
  echo "project=$PROJ"
  echo "global_cloud_cache=$GLOBAL_CLOUD_CACHE"
  echo "manual_support=$MANUAL_SUPPORT_NPZ"
  echo "output=$OUT_DIR"
  echo "git_sha=$(git rev-parse HEAD)"
  sha256sum "$GLOBAL_CLOUD_CACHE" "$MANUAL_SUPPORT_NPZ"

  PYTHONUNBUFFERED=1 "$PYTHON" build_structured3d_point_aligned_gt.py \
    --global_cloud_npz "$GLOBAL_CLOUD_CACHE" \
    --output_npz "$METRIC_GT" \
    --output_ply "$OUT_DIR/scene00180_identity_gt_in_dust3r_frame.ply" \
    --output_metric_ply "$OUT_DIR/scene00180_metric_structural_gt_world_m.ply" \
    --output_manifest "$METRIC_GT_MANIFEST" \
    --min_conf 1.0 \
    --min_plane_points 64 \
    --boundary_ignore_radius 1

  PYTHONUNBUFFERED=1 "$PYTHON" planegraph_ba.py \
    --global_cloud_npz "$GLOBAL_CLOUD_CACHE" \
    --support_npz "$MANUAL_SUPPORT_NPZ" \
    --output_dir "$OUT_DIR/manual_ba" \
    --scene_key scene00180_manual \
    --min_conf 1.0 \
    --min_plane_points 64 \
    --min_plane_views 2 \
    --iterations 10

  PYTHONUNBUFFERED=1 "$PYTHON" planegraph_ba.py \
    --global_cloud_npz "$GLOBAL_CLOUD_CACHE" \
    --support_npz "$METRIC_GT" \
    --output_dir "$OUT_DIR/oracle_ba" \
    --scene_key scene00180_oracle \
    --min_conf 1.0 \
    --min_plane_points 64 \
    --min_plane_views 2 \
    --iterations 10

  for generated in "$METRIC_GT" "$MANUAL_BA" "$ORACLE_BA"; do
    if [[ ! -f "$generated" ]]; then
      echo "Expected output was not generated: $generated" >&2
      exit 2
    fi
  done
  sha256sum "$METRIC_GT" "$MANUAL_BA" "$ORACLE_BA"

  PYTHONUNBUFFERED=1 "$PYTHON" evaluate_structured3d_metric_geometry.py \
    --metric_gt_npz "$METRIC_GT" \
    --prediction "original_dust3r=$GLOBAL_CLOUD_CACHE" \
    --prediction "manual_planegraph_ba=$MANUAL_BA" \
    --prediction "oracle_identity_planegraph_ba=$ORACLE_BA" \
    --reference original_dust3r \
    --alignment_trim_quantile 0.9 \
    --output_json "$METRICS_JSON"

  METRICS_JSON="$METRICS_JSON" "$PYTHON" -c 'import json, os; p=json.load(open(os.environ["METRICS_JSON"])); print(json.dumps({"metric_gt_points":p["metric_gt_points"], "methods":[{"name":r["name"], "rmse_m":r["independent_alignment_metrics"]["correspondence_error_m"]["rmse"], "plane_mean_m":r["independent_alignment_metrics"]["gt_plane_residual_m"]["mean"], "delta":r["delta_vs_reference"]} for r in p["methods"]]}, indent=2))'
} 2>&1 | tee "$LOG_FILE"

echo "P1 metric/oracle audit complete: $OUT_DIR"
