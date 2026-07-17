#!/usr/bin/env bash
set -euo pipefail

OFFICIAL_REPO="${OFFICIAL_REPO:-/gemini/code/Plane-DUSt3R}"
OFFICIAL_URL="${OFFICIAL_URL:-https://github.com/justacar/Plane-DUSt3R.git}"
OUT_DIR="${OUT_DIR:-/gemini/data-1/lightrecon_runs/plane_dust3r_repository_setup_20260717_v1}"

if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUT_DIR}"
LOG_PATH="${OUT_DIR}/repository_setup.log"

if [[ ! -d "${OFFICIAL_REPO}/.git" ]]; then
  if [[ -e "${OFFICIAL_REPO}" ]]; then
    echo "Refusing non-Git external repo path: ${OFFICIAL_REPO}" >&2
    exit 2
  fi
  git -c http.version=HTTP/1.1 clone \
    "${OFFICIAL_URL}" "${OFFICIAL_REPO}" 2>&1 | tee "${LOG_PATH}"
else
  actual_url="$(git -C "${OFFICIAL_REPO}" remote get-url origin)"
  if [[ "${actual_url}" != "${OFFICIAL_URL}" ]]; then
    echo "Unexpected Plane-DUSt3R origin: ${actual_url}" >&2
    exit 2
  fi
fi

git -c http.version=HTTP/1.1 -C "${OFFICIAL_REPO}" \
  submodule update --init --recursive 2>&1 | tee -a "${LOG_PATH}"

git -C "${OFFICIAL_REPO}" rev-parse HEAD > "${OUT_DIR}/official_commit.txt"
git -C "${OFFICIAL_REPO}" submodule status --recursive > "${OUT_DIR}/submodule_status.txt"
git -C "${OFFICIAL_REPO}" status --short > "${OUT_DIR}/working_tree_status.txt"

for required in evaluate_planedust3r.py metric.py plane_merge_planedust3r.py; do
  test -f "${OFFICIAL_REPO}/${required}" || {
    echo "Missing official file: ${required}" >&2
    exit 2
  }
done
test -d "${OFFICIAL_REPO}/MASt3R"
test -d "${OFFICIAL_REPO}/NonCuboidRoom"

echo "official_repo=${OFFICIAL_REPO}"
echo "official_commit=$(cat "${OUT_DIR}/official_commit.txt")"
cat "${OUT_DIR}/submodule_status.txt"
echo "Repository setup complete; checkpoints and Python environment are not installed by this step."
