#!/bin/bash
# Run this script on a GPU node (requires nvcc / CUDA).
# Usage: bash env/install_vllm_on_gpu_node.sh

set -e

if ! command -v nvcc &>/dev/null; then
    module load cuda/12.2.0 2>/dev/null || true
fi

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate verl310_grpo_stable

echo "Python: $(which python)"

if [ -z "$CUDA_HOME" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    echo "Set CUDA_HOME=$CUDA_HOME"
fi

pip install --no-user vllm==0.12.0

echo ""
echo "Verifying..."
python -c "import vllm; print('vllm version:', vllm.__version__)"
echo "Done."
