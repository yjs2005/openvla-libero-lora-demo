#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_NAME="${ENV_NAME:-openvla}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTORCH_CUDA_VERSION="${PYTORCH_CUDA_VERSION:-12.4}"
OPENVLA_DIR="${PROJECT_ROOT}/external/openvla"
LIBERO_DIR="${PROJECT_ROOT}/external/LIBERO"

echo "=== OpenVLA cloud environment setup template ==="
echo "Target platform: Ubuntu 22.04 or similar Linux GPU instance"
echo "Target Python: ${PYTHON_VERSION}"
echo "Safety: A100 40GB+ is recommended for OpenVLA-7B; A100 80GB is closer to official LoRA examples."
echo "Safety: the local RTX 4060 Laptop 8GB GPU is not recommended for OpenVLA-7B inference or LoRA."
echo "This script installs code dependencies only."
echo "It does not download OpenVLA model checkpoints."
echo "It does not download LIBERO datasets."
echo "It does not run evaluation or training."
echo

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: setup_env.sh is intended for Linux cloud GPU instances."
  exit 1
fi

echo "Step 1/10: checking OS information"
if [[ -f /etc/os-release ]]; then
  cat /etc/os-release
else
  uname -a
fi
echo

echo "Step 2/10: checking NVIDIA GPU visibility"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi was not found. Use a GPU instance with NVIDIA drivers installed."
  exit 1
fi
nvidia-smi
echo

echo "Step 3/10: checking disk space"
df -h "${PROJECT_ROOT}"
echo "Warning: OpenVLA checkpoints, LIBERO data, and logs can consume many GB. Do not download BridgeData V2 at this stage."
echo

echo "Step 4/10: checking conda"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found. Install Miniconda or Anaconda before running this script."
  exit 1
fi
eval "$(conda shell.bash hook)"
echo

echo "Step 5/10: creating or reusing conda environment '${ENV_NAME}'"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists; reusing it."
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"
python --version
echo

echo "Step 6/10: installing CUDA-enabled PyTorch"
echo "Using pytorch-cuda=${PYTORCH_CUDA_VERSION}. If this fails, check the current PyTorch install matrix and retry with PYTORCH_CUDA_VERSION=12.1 or another supported version."
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio "pytorch-cuda=${PYTORCH_CUDA_VERSION}" -c pytorch -c nvidia -y
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
echo

echo "Step 7/10: cloning or updating official OpenVLA repository"
mkdir -p "${PROJECT_ROOT}/external"
if [[ -d "${OPENVLA_DIR}/.git" ]]; then
  echo "external/openvla already exists; pulling latest main branch."
  git -C "${OPENVLA_DIR}" pull --ff-only
else
  git clone https://github.com/openvla/openvla.git "${OPENVLA_DIR}"
fi
echo

echo "Step 8/10: installing OpenVLA as an editable package"
cd "${OPENVLA_DIR}"
pip install -e .
echo

echo "Step 9/10: installing flash-attn dependencies"
pip install packaging ninja
ninja --version
echo "Installing flash-attn==2.5.5. If this fails, try: pip cache remove flash_attn, verify CUDA/PyTorch compatibility, then rerun this step."
pip install "flash-attn==2.5.5" --no-build-isolation
echo

echo "Step 10/10: installing LIBERO evaluation dependencies"
if [[ -d "${LIBERO_DIR}/.git" ]]; then
  echo "external/LIBERO already exists; pulling latest main branch."
  git -C "${LIBERO_DIR}" pull --ff-only
else
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
fi
pip install -e "${LIBERO_DIR}"
pip install -r "${OPENVLA_DIR}/experiments/robot/libero/libero_requirements.txt"
echo

cd "${PROJECT_ROOT}"
echo "Setup template completed."
echo "Next safe step: run python scripts/check_env.py on the cloud machine."
echo "Do not run full evaluation or LoRA until the smoke test plan is confirmed."
