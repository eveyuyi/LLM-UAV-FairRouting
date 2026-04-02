#!/usr/bin/env bash
set -euo pipefail

# Reproducible installer for VERL GRPO environment.
# Usage:
#   bash env/install_verl_grpo.sh [env_name]
#
# Default env name matches env/verl-grpo-conda.yaml.

ENV_NAME="${1:-verl310_grpo_stable}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="${ROOT_DIR}/env/verl-grpo-requirements-lock.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi

echo "[1/4] Create minimal conda env: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python=3.10 pip setuptools wheel

echo "[2/4] Install PyTorch CUDA 12.8 wheels first"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0

echo "[3/4] Install project lock requirements"
conda run -n "${ENV_NAME}" python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r "${REQ_FILE}"

echo "[4/4] Verify critical imports (with PYTHONNOUSERSITE=1)"
conda run -n "${ENV_NAME}" env PYTHONNOUSERSITE=1 python -c \
  "import torch, vllm, verl, transformers, datasets, ray, hydra, tensordict, pydantic; \
print('OK:', torch.__version__, vllm.__version__, verl.__version__)"

echo "Environment ready: ${ENV_NAME}"
