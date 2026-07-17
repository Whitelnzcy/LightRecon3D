#!/usr/bin/env bash
set -euo pipefail

OFFICIAL_REPO="${OFFICIAL_REPO:-/gemini/code/Plane-DUSt3R}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-9a1ae50650ec6d706bf329352aaaf49efded90a0}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
ENV_DIR="${ENV_DIR:-/gemini/data-1/lightrecon_envs/planedust3r-py311-torch220-cu118}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/gemini/pretrain/Plane-DUSt3R}"
PLANE_CHECKPOINT="${PLANE_CHECKPOINT:-${CHECKPOINT_DIR}/checkpoint-best-onlyencoder.pth}"
NONCUBOID_CHECKPOINT="${NONCUBOID_CHECKPOINT:-${CHECKPOINT_DIR}/Structured3D_pretrained.pt}"
OUT_DIR="${OUT_DIR:-/gemini/data-1/lightrecon_runs/plane_dust3r_external_setup_20260717_v1}"
DOWNLOAD_BACKEND="${DOWNLOAD_BACKEND:-aria2}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
ARIA2_CONNECTIONS="${ARIA2_CONNECTIONS:-16}"
PLANE_URL="${PLANE_URL:-${HF_ENDPOINT%/}/yxuan/Plane-DUSt3R/resolve/main/checkpoint-best-onlyencoder.pth}"
NONCUBOID_URL="${NONCUBOID_URL:-https://drive.google.com/uc?id=1DZnnOUMh6llVwhBvb-yo9ENVmN4o42x8}"

if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUT_DIR}" "${CHECKPOINT_DIR}" "$(dirname "${ENV_DIR}")"

test -x "${CONDA_BIN}"
test -d "${OFFICIAL_REPO}/.git"
actual_commit="$(git -C "${OFFICIAL_REPO}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Official commit ${actual_commit} != expected ${EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${OFFICIAL_REPO}" status --short)" ]]; then
  echo "Official repository must be clean" >&2
  exit 2
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${CONDA_BIN}" create -y --prefix "${ENV_DIR}" python=3.11 cmake=3.14.0
fi

if ! "${ENV_DIR}/bin/python" -c 'import torch; assert torch.__version__.startswith("2.2.0")' >/dev/null 2>&1; then
  "${CONDA_BIN}" install -y --prefix "${ENV_DIR}" \
    pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 \
    -c pytorch -c nvidia
fi

repo_marker="${ENV_DIR}/.plane_dust3r_requirements_${EXPECTED_COMMIT}"
if [[ ! -f "${repo_marker}" ]]; then
  "${ENV_DIR}/bin/python" -m pip install -r "${OFFICIAL_REPO}/MASt3R/requirements.txt"
  "${ENV_DIR}/bin/python" -m pip install -r "${OFFICIAL_REPO}/MASt3R/dust3r/requirements.txt"
  "${ENV_DIR}/bin/python" -m pip install -r "${OFFICIAL_REPO}/NonCuboidRoom/requirements.txt"
  "${ENV_DIR}/bin/python" -m pip install roma==1.5.6 gdown
  touch "${repo_marker}"
fi

checkpoint_loadable() {
  local checkpoint="$1"
  [[ -f "${checkpoint}" ]] && \
    "${ENV_DIR}/bin/python" -c \
      'import sys,torch; value=torch.load(sys.argv[1], map_location="cpu"); assert isinstance(value, dict)' \
      "${checkpoint}" >/dev/null 2>&1
}

download_plane_checkpoint() {
  case "${DOWNLOAD_BACKEND}" in
    aria2)
      if [[ ! -x "${ENV_DIR}/bin/aria2c" ]]; then
        "${CONDA_BIN}" install -y --prefix "${ENV_DIR}" -c conda-forge aria2
      fi
      "${ENV_DIR}/bin/aria2c" \
        --continue=true \
        --max-connection-per-server="${ARIA2_CONNECTIONS}" \
        --split="${ARIA2_CONNECTIONS}" \
        --min-split-size=4M \
        --file-allocation=none \
        --max-tries=5 \
        --retry-wait=3 \
        --timeout=30 \
        --dir="${CHECKPOINT_DIR}" \
        --out="$(basename "${PLANE_CHECKPOINT}")" \
        "${PLANE_URL}"
      ;;
    hf_xet)
      "${ENV_DIR}/bin/python" -m pip install --upgrade "huggingface_hub>=0.32.0"
      HF_ENDPOINT="${HF_ENDPOINT}" HF_XET_HIGH_PERFORMANCE=1 \
        "${ENV_DIR}/bin/python" -c \
        'import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id="yxuan/Plane-DUSt3R", filename="checkpoint-best-onlyencoder.pth", local_dir=sys.argv[1]))' \
        "${CHECKPOINT_DIR}"
      ;;
    wget)
      command -v wget >/dev/null 2>&1 || {
        echo "wget is unavailable" >&2
        exit 2
      }
      wget --continue --tries=3 --timeout=30 \
        "${PLANE_URL}" -O "${PLANE_CHECKPOINT}"
      ;;
    *)
      echo "Unknown DOWNLOAD_BACKEND=${DOWNLOAD_BACKEND}; use aria2, hf_xet or wget" >&2
      exit 2
      ;;
  esac
}

if ! checkpoint_loadable "${PLANE_CHECKPOINT}"; then
  existing_bytes="$(stat -c%s "${PLANE_CHECKPOINT}" 2>/dev/null || echo 0)"
  echo "Plane checkpoint is missing or incomplete (${existing_bytes} bytes); downloading with ${DOWNLOAD_BACKEND}."
  download_plane_checkpoint
fi
if ! checkpoint_loadable "${PLANE_CHECKPOINT}"; then
  echo "Plane-DUSt3R checkpoint is still incomplete or invalid: ${PLANE_CHECKPOINT}" >&2
  exit 2
fi

if ! checkpoint_loadable "${NONCUBOID_CHECKPOINT}"; then
  "${ENV_DIR}/bin/python" -m gdown --continue \
    "${NONCUBOID_URL}" \
    -O "${NONCUBOID_CHECKPOINT}"
fi
if ! checkpoint_loadable "${NONCUBOID_CHECKPOINT}"; then
  echo "NonCuboidRoom checkpoint is incomplete or invalid: ${NONCUBOID_CHECKPOINT}" >&2
  exit 2
fi

"${ENV_DIR}/bin/python" - <<'PY' > "${OUT_DIR}/environment.txt"
from importlib.metadata import version
import torch
print(f"python_torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"roma={version('roma')}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
"${ENV_DIR}/bin/python" -m pip freeze > "${OUT_DIR}/pip_freeze.txt"
"${CONDA_BIN}" list --prefix "${ENV_DIR}" > "${OUT_DIR}/conda_list.txt"
sha256sum "${PLANE_CHECKPOINT}" "${NONCUBOID_CHECKPOINT}" \
  > "${OUT_DIR}/checkpoints.sha256"
git -C "${OFFICIAL_REPO}" rev-parse HEAD > "${OUT_DIR}/official_commit.txt"
git -C "${OFFICIAL_REPO}" status --short > "${OUT_DIR}/working_tree_status.txt"
{
  echo "download_backend=${DOWNLOAD_BACKEND}"
  echo "hf_endpoint=${HF_ENDPOINT}"
  echo "aria2_connections=${ARIA2_CONNECTIONS}"
  echo "plane_url=${PLANE_URL}"
  echo "noncuboid_url=${NONCUBOID_URL}"
  echo "plane_bytes=$(stat -c%s "${PLANE_CHECKPOINT}")"
  echo "noncuboid_bytes=$(stat -c%s "${NONCUBOID_CHECKPOINT}")"
} > "${OUT_DIR}/download_provenance.txt"

cat "${OUT_DIR}/environment.txt"
cat "${OUT_DIR}/checkpoints.sha256"
cat "${OUT_DIR}/download_provenance.txt"
echo "Plane-DUSt3R external environment and checkpoints are ready."
