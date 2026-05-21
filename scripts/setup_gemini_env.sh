#!/usr/bin/env bash
set -e

echo "============================================================"
echo "LightRecon3D Gemini Environment Setup"
echo "============================================================"

# ============================================================
# 0. Persistent paths
# ============================================================

PROJECT_DIR="/gemini/code/LightRecon3D"
ENV_DIR="/gemini/data-1/envs/lightrecon"
PIP_CACHE="/gemini/data-1/pip_cache"
CONDA_PKGS="/gemini/data-1/conda_pkgs"
LOG_DIR="/gemini/data-1/logs"
CKPT_DIR="/gemini/data-1/checkpoints/lightrecon"

mkdir -p "$PIP_CACHE"
mkdir -p "$CONDA_PKGS"
mkdir -p "$LOG_DIR"
mkdir -p "$CKPT_DIR"
mkdir -p "$(dirname "$ENV_DIR")"

export PIP_CACHE_DIR="$PIP_CACHE"
export CONDA_PKGS_DIRS="$CONDA_PKGS"

cd "$PROJECT_DIR"

echo "[Info] Project dir: $PROJECT_DIR"
echo "[Info] Env dir    : $ENV_DIR"
echo "[Info] Pip cache  : $PIP_CACHE"
echo "[Info] Conda pkgs : $CONDA_PKGS"
echo "[Info] Log dir    : $LOG_DIR"
echo "[Info] Ckpt dir   : $CKPT_DIR"

# ============================================================
# 1. Configure Tsinghua mirrors
# ============================================================

echo "============================================================"
echo "[Step 1] Configuring Tsinghua mirrors"
echo "============================================================"

cat > ~/.condarc <<'EOF'
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
EOF

mkdir -p ~/.pip

cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
EOF

export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
export PIP_DEFAULT_TIMEOUT=120

echo "[Info] Conda mirror:"
conda config --show default_channels || true

echo "[Info] Pip mirror:"
python -m pip config list || true

# Clean old conda index cache. Failure is non-fatal.
conda clean -i -y || true

# ============================================================
# 2. Check conda
# ============================================================

echo "============================================================"
echo "[Step 2] Checking conda"
echo "============================================================"

if ! command -v conda >/dev/null 2>&1; then
    echo "[Error] conda not found."
    echo "If conda is ephemeral on this server, install Miniconda under /gemini/data-1 first."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[Info] Conda base: $(conda info --base)"

# ============================================================
# 3. Create persistent conda environment
# ============================================================

echo "============================================================"
echo "[Step 3] Creating / checking persistent conda env"
echo "============================================================"

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "[Info] Creating conda env at $ENV_DIR"
    conda create -y -p "$ENV_DIR" python=3.10
else
    echo "[Info] Existing env found at $ENV_DIR"
fi

conda activate "$ENV_DIR"

echo "[Info] Python:"
which python
python --version

echo "[Info] Pip:"
which pip
pip --version

# Make sure pip itself is usable and reasonably new.
python -m pip install --upgrade pip setuptools wheel

# ============================================================
# 4. Install PyTorch
# ============================================================

echo "============================================================"
echo "[Step 4] Installing / checking PyTorch"
echo "============================================================"

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") is not None else 1)
PY
then
    echo "[Info] torch already installed."
else
    echo "[Info] Installing PyTorch from Tsinghua pip mirror."
    echo "[Info] If CUDA is unavailable after install, adjust torch version manually."

    # 先用默认 pip 清华源安装。多数云端镜像会给到 CUDA 版或可用版本。
    # 如果装完 torch.cuda.is_available() 为 False，再单独处理。
    pip install torch torchvision torchaudio
fi

# ============================================================
# 5. Install project dependencies
# ============================================================

echo "============================================================"
echo "[Step 5] Installing project dependencies"
echo "============================================================"

if [ -f "requirements.txt" ]; then
    echo "[Info] Found requirements.txt. Installing..."
    pip install -r requirements.txt
else
    echo "[Warn] requirements.txt not found. Installing common dependencies..."
fi

# 即使 requirements.txt 存在，也补装这些常用依赖，重复安装不会有问题。
pip install \
    numpy \
    scipy \
    matplotlib \
    pillow \
    opencv-python \
    tqdm \
    einops \
    pyyaml \
    huggingface_hub \
    safetensors \
    scikit-learn \
    swanlab \
    packaging \
    ninja \
    cython

# ============================================================
# 6. Sanity import check
# ============================================================

echo "============================================================"
echo "[Step 6] Python package sanity check"
echo "============================================================"

python - <<'PY'
import sys
print("python:", sys.executable)

import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
    print("cuda version:", torch.version.cuda)
else:
    print("[Warn] torch.cuda.is_available() is False.")

import numpy
import scipy
import matplotlib
import PIL
import cv2
import tqdm
import einops
print("basic packages ok")
PY

# ============================================================
# 7. Check project files
# ============================================================

echo "============================================================"
echo "[Step 7] Checking LightRecon3D project files"
echo "============================================================"

if [ ! -f "train.py" ]; then
    echo "[Error] train.py not found in $PROJECT_DIR"
    exit 1
fi

if [ ! -d "dust3r" ]; then
    echo "[Warn] dust3r directory not found under $PROJECT_DIR"
    echo "[Warn] If train.py imports dust3r, please make sure dust3r is included."
fi

if [ ! -f "/gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" ]; then
    echo "[Warn] DUSt3R checkpoint not found:"
    echo "       /gemini/pretrain/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
fi

if [ ! -d "/gemini/data-1/Structured3D" ]; then
    echo "[Warn] Structured3D dataset not found:"
    echo "       /gemini/data-1/Structured3D"
fi

# ============================================================
# 8. Check train.py --help
# ============================================================

echo "============================================================"
echo "[Step 8] Checking train.py --help"
echo "============================================================"

python train.py --help >/tmp/lightrecon_train_help.txt

echo "[Info] train.py --help ok."

# ============================================================
# 9. Create activate script if missing
# ============================================================

echo "============================================================"
echo "[Step 9] Writing activate script"
echo "============================================================"

mkdir -p scripts

cat > scripts/activate_gemini_env.sh <<'EOF'
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
# Configure Tsinghua mirrors every time.
# Useful when $HOME is ephemeral on cloud servers.
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

if ! command -v conda >/dev/null 2>&1; then
    echo "[Error] conda not found."
    echo "If conda is also ephemeral, install Miniconda under /gemini/data-1 first."
    return 1 2>/dev/null || exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "[Error] Env not found: $ENV_DIR"
    echo "Please run:"
    echo "  bash /gemini/code/LightRecon3D/scripts/setup_gemini_env.sh"
    return 1 2>/dev/null || exit 1
fi

conda activate "$ENV_DIR"

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
EOF

chmod +x scripts/activate_gemini_env.sh

# ============================================================
# Done
# ============================================================

echo "============================================================"
echo "Environment setup finished."
echo "============================================================"
echo "Next time after server restart, run:"
echo ""
echo "  source /gemini/code/LightRecon3D/scripts/activate_gemini_env.sh"
echo ""
echo "Then check:"
echo ""
echo "  python train.py --help"
echo ""
echo "============================================================"