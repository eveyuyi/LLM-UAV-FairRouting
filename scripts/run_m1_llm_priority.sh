#!/usr/bin/env bash
# M1: LLM Priority Inference + NSGA-III
# 使用本地 vLLM 服务（默认 http://127.0.0.1:8000/v1）或 OpenAI API 进行优先级推断
#
# 前提：vLLM 服务已在 GPU 节点启动（参考 scripts/serve_vllm_model.sh）
# 或设置 OPENAI_API_KEY + OPENAI_BASE_URL 指向 OpenAI API。
#
# 用法示例：
#   # 本地 vLLM
#   SEED=4111 bash scripts/run_m1_llm_priority.sh
#   # OpenAI API
#   OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.openai.com/v1 \
#     MODEL=gpt-4o SEED=4111 bash scripts/run_m1_llm_priority.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${LLM4FAIRROUTING_ENV_FILE:-$ROOT_DIR/.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi
CONDA_ENV="${LLM4FAIRROUTING_CONDA_ENV:-}"

cd "${ROOT_DIR}"

SEED="${SEED:-4111}"
SPLIT="${SPLIT:-norm_eval}"
DATASET_DIR="${DATASET_DIR:-data/test/test_seeds/${SPLIT}/seed_${SEED}}"
EXTRACTED_DEMANDS="${EXTRACTED_DEMANDS:-${DATASET_DIR}/llm3_sft_pipeline.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/eval_runs/m1_llm_priority_${SPLIT}_seed${SEED}}"
SOLVER_BACKEND="${SOLVER_BACKEND:-nsga3_heuristic}"
NSGA3_POP_SIZE="${NSGA3_POP_SIZE:-8}"
NSGA3_N_GENERATIONS="${NSGA3_N_GENERATIONS:-5}"

# LLM config — override via env vars
API_BASE="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
MODEL="${MODEL:-qwen3-local}"

if [[ ! -f "${EXTRACTED_DEMANDS}" ]]; then
  echo "[error] Extracted demands file not found: ${EXTRACTED_DEMANDS}" >&2
  exit 1
fi

echo "[M1] seed=${SEED} split=${SPLIT}"
echo "[M1] extracted_demands=${EXTRACTED_DEMANDS}"
echo "[M1] output_dir=${OUTPUT_DIR}"
echo "[M1] api_base=${API_BASE} model=${MODEL}"

if [[ -n "${CONDA_ENV}" ]]; then
  PYTHONPATH=src conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python -u \
    -m llm4fairrouting.workflow.run_workflow \
    --output-dir "${OUTPUT_DIR}" \
    --priority-mode llm-only \
    --api-base "${API_BASE}" \
    --api-key "${API_KEY}" \
    --model "${MODEL}" \
    --solver-backend "${SOLVER_BACKEND}" \
    --dialogues "${DATASET_DIR}/dialogues.jsonl" \
    --extracted-demands "${EXTRACTED_DEMANDS}" \
    --nsga3-pop-size "${NSGA3_POP_SIZE}" \
    --nsga3-n-generations "${NSGA3_N_GENERATIONS}" \
    "$@"
else
  PYTHONPATH=src python -u \
    -m llm4fairrouting.workflow.run_workflow \
    --output-dir "${OUTPUT_DIR}" \
    --priority-mode llm-only \
    --api-base "${API_BASE}" \
    --api-key "${API_KEY}" \
    --model "${MODEL}" \
    --solver-backend "${SOLVER_BACKEND}" \
    --dialogues "${DATASET_DIR}/dialogues.jsonl" \
    --extracted-demands "${EXTRACTED_DEMANDS}" \
    --nsga3-pop-size "${NSGA3_POP_SIZE}" \
    --nsga3-n-generations "${NSGA3_N_GENERATIONS}" \
    "$@"
fi
