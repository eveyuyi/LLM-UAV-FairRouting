#!/usr/bin/env bash
# P3: GPT-4o 零样本（pre）vs Fine-tuned Qwen3（post）
# 测试集与 P1/P2 完全一致：data/test/test_seeds/{norm_eval,hard_eval}/seed_xxxx
#
# 用法示例（单 seed）：
#   OPENAI_API_KEY=sk-... bash scripts/eval_p3_gpt4o_vs_finetuned.sh
#
# 批量跑全部 norm + hard seeds：
#   for seed in 4111 4112 5111 5112 5113; do
#     SEED=$seed SPLIT=norm_eval bash scripts/eval_p3_gpt4o_vs_finetuned.sh
#   done
#   for seed in 5101 5102 5103 5104 5105 5106; do
#     SEED=$seed SPLIT=hard_eval bash scripts/eval_p3_gpt4o_vs_finetuned.sh
#   done
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEED="${SEED:-4111}"
SPLIT="${SPLIT:-norm_eval}"
DATASET_DIR="${DATASET_DIR:-data/test/test_seeds/${SPLIT}/seed_${SEED}}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "[error] Dataset dir not found: ${DATASET_DIR}" >&2
  echo "  Ensure data/test/test_seeds/ has been generated first." >&2
  exit 1
fi

# PRE: GPT-4o 零样本
export PRE_API_BASE="${PRE_API_BASE:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
export PRE_MODEL="${PRE_MODEL:-gpt-4o}"
export PRE_PRIORITY_MODE="llm-only"

# POST: Fine-tuned Qwen3（本地 vLLM 服务，默认端口 8001）
export POST_API_BASE="${POST_API_BASE:-http://127.0.0.1:8001/v1}"
export POST_MODEL="${POST_MODEL:-qwen3-finetuned}"
export POST_PRIORITY_MODE="llm-only"

export OUTPUT_ROOT="${OUTPUT_ROOT:-data/eval_runs/p3_gpt4o_vs_finetuned_${SPLIT}_seed${SEED}}"

echo "[P3] seed=${SEED} split=${SPLIT}"
echo "[P3] pre_model=${PRE_MODEL} post_model=${POST_MODEL}"
echo "[P3] output_root=${OUTPUT_ROOT}"

bash scripts/eval_pre_post_priority_alignment.sh "${DATASET_DIR}"
