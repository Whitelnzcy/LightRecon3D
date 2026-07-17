#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
OFFICIAL_REPO="${OFFICIAL_REPO:-/gemini/code/Plane-DUSt3R}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-9a1ae50650ec6d706bf329352aaaf49efded90a0}"
PYTHON_BIN="${PYTHON_BIN:-/gemini/data-1/lightrecon_envs/planedust3r-py311-torch220-cu118-v2/bin/python}"
PLANE_CHECKPOINT="${PLANE_CHECKPOINT:-/gemini/pretrain/Plane-DUSt3R/checkpoint-best-onlyencoder.pth}"
NONCUBOID_CHECKPOINT="${NONCUBOID_CHECKPOINT:-/gemini/pretrain/Plane-DUSt3R/Structured3D_pretrained.pt}"
SAME_INPUT_MANIFEST="${SAME_INPUT_MANIFEST:-/gemini/data-1/lightrecon_runs/plane_dust3r_same_input_20260717_v1/same_input_manifest.json}"
SCENE_NAME="${SCENE_NAME:-scene_00180}"
OUT_DIR="${OUT_DIR:-/gemini/data-1/lightrecon_runs/plane_dust3r_same_input_smoke_scene00180_20260717_v2}"

if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
  exit 2
fi

"${PYTHON_BIN}" -c 'from importlib.metadata import version; import numpy,scipy,torch,roma; print("torch=" + torch.__version__); print("torch_cuda=" + str(torch.version.cuda)); print("numpy=" + numpy.__version__); print("scipy=" + scipy.__version__); print("roma=" + version("roma")); assert torch.__version__.split("+")[0] == "2.2.0"; assert torch.version.cuda == "11.8"; assert numpy.__version__ == "1.26.4"; assert scipy.__version__ == "1.11.4"; assert torch.cuda.is_available()'
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader

echo "LightRecon3D Plane-DUSt3R same-input smoke"
echo "scene=${SCENE_NAME}"
echo "official_repo=${OFFICIAL_REPO}"
echo "expected_commit=${EXPECTED_COMMIT}"
echo "python=${PYTHON_BIN}"
echo "output=${OUT_DIR}"
echo "project_git_sha=$(git -C "${PROJECT_DIR}" rev-parse HEAD)"

"${PYTHON_BIN}" "${PROJECT_DIR}/run_plane_dust3r_same_input_smoke.py" \
  --same_input_manifest "${SAME_INPUT_MANIFEST}" \
  --scene_name "${SCENE_NAME}" \
  --official_repo "${OFFICIAL_REPO}" \
  --expected_commit "${EXPECTED_COMMIT}" \
  --python_bin "${PYTHON_BIN}" \
  --plane_checkpoint "${PLANE_CHECKPOINT}" \
  --noncuboid_checkpoint "${NONCUBOID_CHECKPOINT}" \
  --output_dir "${OUT_DIR}" \
  --project_git_sha "$(git -C "${PROJECT_DIR}" rev-parse HEAD)"

cat "${OUT_DIR}/smoke_manifest.json"
