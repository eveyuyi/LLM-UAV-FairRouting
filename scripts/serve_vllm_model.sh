#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------- 只改这里 ----------
CONDA_ENV="${CONDA_ENV:-}"
MODEL_PATH="${MODEL_PATH:-data/checkpoints/llm3_sft_merged_hf/global_step_8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-local}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-bfloat16}" # auto | half | bfloat16 | float16 | float32
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ -n "${CONDA_ENV}" ]]; then
  _py=(conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python)
else
  _py=(env PYTHONNOUSERSITE=1 python)
fi

cmd=(
  "${_py[@]}"
  -m vllm.entrypoints.openai.api_server
  --host "${HOST}"
  --port "${PORT}"
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

echo "Starting vLLM server..."
echo "  model_path   : ${MODEL_PATH}"
echo "  model_name   : ${SERVED_MODEL_NAME}"
echo "  endpoint     : http://${HOST}:${PORT}/v1"
echo "  tp_size      : ${TENSOR_PARALLEL_SIZE}"
echo "  dtype        : ${DTYPE}"
echo "  max_model_len: ${MAX_MODEL_LEN}"
echo "  gpu_mem_util : ${GPU_MEMORY_UTILIZATION}"
echo ""
echo "Tip: set OPENAI_BASE_URL=http://127.0.0.1:${PORT}/v1"
echo "     set LLM4FAIRROUTING_MODEL=${SERVED_MODEL_NAME}"
echo ""

exec "${cmd[@]}"
