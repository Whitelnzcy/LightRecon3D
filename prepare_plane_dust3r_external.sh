#!/usr/bin/env bash
set -euo pipefail

OFFICIAL_REPO="${OFFICIAL_REPO:-/gemini/code/Plane-DUSt3R}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-9a1ae50650ec6d706bf329352aaaf49efded90a0}"
PROJECT_DIR="${PROJECT_DIR:-/gemini/code/LightRecon3D}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
ENV_DIR="${ENV_DIR:-/gemini/data-1/lightrecon_envs/planedust3r-py311-torch220-cu118-v2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/gemini/pretrain/Plane-DUSt3R}"
PLANE_CHECKPOINT="${PLANE_CHECKPOINT:-${CHECKPOINT_DIR}/checkpoint-best-onlyencoder.pth}"
NONCUBOID_CHECKPOINT="${NONCUBOID_CHECKPOINT:-${CHECKPOINT_DIR}/Structured3D_pretrained.pt}"
OUT_DIR="${OUT_DIR:-/gemini/data-1/lightrecon_runs/plane_dust3r_external_setup_20260717_v4}"
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
test -f "${PROJECT_DIR}/prepare_plane_dust3r_requirements.py"
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

verify_torch_runtime() {
  "${ENV_DIR}/bin/python" - <<'PY'
import torch

assert torch.__version__.split("+")[0] == "2.2.0", torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
matrix = torch.arange(9, dtype=torch.float32).reshape(3, 3)
result = matrix @ matrix.T
assert result.shape == (3, 3)
assert float(result[0, 0]) == 5.0
print(f"PyTorch runtime verified: torch={torch.__version__} cuda={torch.version.cuda}")
PY
}

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${CONDA_BIN}" create -y --prefix "${ENV_DIR}" python=3.11 cmake=3.14.0
elif "${ENV_DIR}/bin/python" -c 'import torch' >/dev/null 2>&1 && \
    ! "${ENV_DIR}/bin/python" -c \
      'import torch; assert torch.__version__.split("+")[0] == "2.2.0"; assert torch.version.cuda == "11.8"' \
      >/dev/null 2>&1; then
  echo "Refusing to repair an existing environment with incompatible PyTorch: ${ENV_DIR}" >&2
  echo "Use the new default v2 environment or choose a fresh ENV_DIR." >&2
  exit 2
fi

if ! "${ENV_DIR}/bin/python" -c \
  'import torch; assert torch.__version__.split("+")[0] == "2.2.0"; assert torch.version.cuda == "11.8"' \
  >/dev/null 2>&1; then
  "${CONDA_BIN}" install -y --prefix "${ENV_DIR}" \
    pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 \
    mkl=2024.0 intel-openmp=2024.0 \
    -c pytorch -c nvidia -c defaults
fi
verify_torch_runtime

requirements_dir="${OUT_DIR}/sanitized_requirements"
"${ENV_DIR}/bin/python" "${PROJECT_DIR}/prepare_plane_dust3r_requirements.py" \
  --official_repo "${OFFICIAL_REPO}" \
  --output_dir "${requirements_dir}"
constraints="${requirements_dir}/constraints.txt"

verify_environment() {
  "${ENV_DIR}/bin/python" - <<'PY'
from importlib.metadata import version
import cv2
import numpy
import scipy
import torch
import torchvision

expected = {
    "torch": "2.2.0",
    "torchvision": "0.17.0",
    "torchaudio": "2.2.0",
    "numpy": "1.26.4",
    "scipy": "1.11.4",
    "opencv-python": "4.10.0.84",
    "mmcv": "1.7.2",
    "numba": "0.59.1",
    "pillow": "9.5.0",
    "roma": "1.5.6",
    "setuptools": "80.9.0",
    "wheel": "0.45.1",
}
for package, required in expected.items():
    actual = version(package).split("+")[0]
    assert actual == required, f"{package}={actual}, expected {required}"
assert torch.version.cuda == "11.8", torch.version.cuda
assert numpy.__version__ == "1.26.4", numpy.__version__
assert scipy.__version__ == "1.11.4", scipy.__version__
assert cv2.__version__ == "4.10.0", cv2.__version__
matrix = torch.arange(9, dtype=torch.float32).reshape(3, 3)
assert float((matrix @ matrix.T)[0, 0]) == 5.0
print("Plane-DUSt3R Python environment pins verified")
PY
  "${ENV_DIR}/bin/python" -m pip check
}

repo_marker="${ENV_DIR}/.plane_dust3r_requirements_${EXPECTED_COMMIT}_py311v4"
if [[ ! -f "${repo_marker}" ]]; then
  "${ENV_DIR}/bin/python" -m pip install \
    --constraint "${constraints}" \
    setuptools==80.9.0 wheel==0.45.1
  "${ENV_DIR}/bin/python" -c 'import pkg_resources; print("pkg_resources bootstrap verified")'
  "${ENV_DIR}/bin/python" -m pip install \
    --constraint "${constraints}" \
    --requirement "${requirements_dir}/python311_binary_compatibility.txt"
  "${ENV_DIR}/bin/python" -m pip install \
    --no-build-isolation \
    --constraint "${constraints}" \
    --requirement "${requirements_dir}/python311_source_compatibility.txt"
  "${ENV_DIR}/bin/python" -m pip install \
    --constraint "${constraints}" \
    --requirement "${requirements_dir}/MASt3R__requirements.txt"
  "${ENV_DIR}/bin/python" -m pip install \
    --constraint "${constraints}" \
    --requirement "${requirements_dir}/MASt3R__dust3r__requirements.txt"
  "${ENV_DIR}/bin/python" -m pip install \
    --constraint "${constraints}" \
    --requirement "${requirements_dir}/NonCuboidRoom__requirements.txt"
  "${ENV_DIR}/bin/python" -m pip install \
    --constraint "${constraints}" gdown
  verify_environment
  touch "${repo_marker}"
fi
verify_environment

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
        "${CONDA_BIN}" install -y --freeze-installed \
          --prefix "${ENV_DIR}" -c conda-forge aria2
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
      "${ENV_DIR}/bin/python" -m pip install \
        --constraint "${constraints}" --upgrade-strategy only-if-needed \
        "huggingface_hub>=0.32.0"
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

verify_environment

"${ENV_DIR}/bin/python" - <<'PY' > "${OUT_DIR}/environment.txt"
from importlib.metadata import version
import cv2
import numpy
import scipy
import torch
print(f"python_torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"numpy={numpy.__version__}")
print(f"scipy={scipy.__version__}")
print(f"cv2={cv2.__version__}")
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
