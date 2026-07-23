#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}
SOURCE_DIR=${SOURCE_DIR:-/gemini/data-1/lightrecon_runs/plane_feedback_p1_metric_oracle_scene00180_20260715_v1}
GLOBAL_CLOUD_CACHE=${GLOBAL_CLOUD_CACHE:-/gemini/data-1/lightrecon_runs/plane_feedback_p0_scene00180_20260714_v3/scene_00180__gemini_data-1_Structured3D_scene_00180_2D_rendering_445895_perspective_empty_dust3r_global_cloud_cache.npz}
METRIC_GT=${METRIC_GT:-$SOURCE_DIR/scene00180_metric_structural_gt.npz}
MANUAL_BA=${MANUAL_BA:-$SOURCE_DIR/manual_ba/scene00180_manual_planegraph_ba_v0_full_pointcloud_editable_planes_data.npz}
ORACLE_BA=${ORACLE_BA:-$SOURCE_DIR/oracle_ba/scene00180_oracle_planegraph_ba_v0_full_pointcloud_editable_planes_data.npz}
OUT_JSON=${OUT_JSON:-$SOURCE_DIR/scene00180_metric_geometry_gate_v2.json}
LOG_FILE=${LOG_FILE:-$SOURCE_DIR/metric_gate_recheck.log}

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

for required in "$GLOBAL_CLOUD_CACHE" "$METRIC_GT" "$MANUAL_BA" "$ORACLE_BA"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done
for output in "$OUT_JSON" "$LOG_FILE"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing output: $output" >&2
    exit 2
  fi
done

{
  echo "LightRecon3D P1 oracle per-view keep/rollback stop-go"
  echo "project=$PROJ"
  echo "source_dir=$SOURCE_DIR"
  echo "output=$OUT_JSON"
  echo "git_sha=$(git rev-parse HEAD)"
  sha256sum "$GLOBAL_CLOUD_CACHE" "$METRIC_GT" "$MANUAL_BA" "$ORACLE_BA"

  PYTHONUNBUFFERED=1 "$PYTHON" evaluate_structured3d_metric_geometry.py \
    --metric_gt_npz "$METRIC_GT" \
    --prediction "original_dust3r=$GLOBAL_CLOUD_CACHE" \
    --prediction "manual_planegraph_ba=$MANUAL_BA" \
    --prediction "oracle_identity_planegraph_ba=$ORACLE_BA" \
    --reference original_dust3r \
    --alignment_trim_quantile 0.9 \
    --output_json "$OUT_JSON"

  OUT_JSON="$OUT_JSON" "$PYTHON" -c 'import json, os; p=json.load(open(os.environ["OUT_JSON"])); print(json.dumps({"metric_gt_points":p["metric_gt_points"], "methods":[{"name":r["name"], "fixed_delta":r["shared_reference_delta_vs_reference"], "per_view":r["oracle_view_switch_upper_bound"].get("per_view_decisions", []), "correspondence_oracle":r["oracle_view_switch_upper_bound"].get("correspondence_oracle"), "joint_pareto_oracle":r["oracle_view_switch_upper_bound"].get("joint_pareto_oracle")} for r in p["methods"]]}, indent=2))'
} 2>&1 | tee "$LOG_FILE"

echo "P1 metric gate recheck complete: $OUT_JSON"
