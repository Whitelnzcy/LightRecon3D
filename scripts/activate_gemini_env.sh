#!/usr/bin/env bash

PROJECT_DIR="/gemini/code/LightRecon3D"
ENV_DIR="/gemini/data-1/envs/lightrecon"
PIP_CACHE="/gemini/data-1/pip_cache"
CONDA_PKGS="/gemini/data-1/conda_pkgs"

mkdir -p "$PIP_CACHE"
mkdir -p "$CONDA_PKGS"

export PIP_CACHE_DIR="$PIP_CACHE"
export CONDA_PKGS_DIRS="$CONDA_PKGS"

# ============================================================
# Configure mirrors every time.
# $HOME may be ephemeral, so ~/.pip and ~/.condarc may disappear.
# ============================================================

cat > ~/.condarc <<'EO_CONDARC'
channels:
  - defaults

default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  nvidia: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

show_channel_urls: true
channel_priority: flexible
EO_CONDARC

mkdir -p ~/.pip

cat > ~/.pip/pip.conf <<'EO_PIP'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
EO_PIP

export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
export PIP_DEFAULT_TIMEOUT=120

# ============================================================
# Activate persistent environment without relying on conda command.
# This avoids problems when conda itself is not available in the shell.
# ============================================================

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "[Error] Persistent env not found: $ENV_DIR"
    echo "You need to run setup once:"
    echo "  bash /gemini/code/LightRecon3D/scripts/setup_gemini_env.sh"
    return 1 2>/dev/null || exit 1
fi

export CONDA_PREFIX="$ENV_DIR"
export CONDA_DEFAULT_ENV="lightrecon"
export PATH="$ENV_DIR/bin:$PATH"

cd "$PROJECT_DIR"

echo "============================================================"
echo "LightRecon3D environment activated"
echo "Project        : $PROJECT_DIR"
echo "Env            : $ENV_DIR"
echo "Python         : $(which python)"
python --version
echo "Pip mirror     : $PIP_INDEX_URL"
echo "Pip cache      : $PIP_CACHE_DIR"
echo "Conda pkgs     : $CONDA_PKGS_DIRS"
echo "============================================================"