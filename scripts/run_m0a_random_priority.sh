#!/usr/bin/env bash
# M0a: Random Priority + NSGA-III
# 使用 data/test/test_seeds/norm_eval/seed_xxxx 作为标准测试集
# 与 M0b/M0c/M1 共享相同 extracted_demands（固定 Module 2 输出），仅优先级注入为随机
#
# 用法示例：
#   bash scripts/run_m0a_random_priority.sh
#   SEED=4112 bash scripts/run_m0a_random_priority.sh
#   bash scripts/run_m0a_random_priority.sh --skip-solver    # 冒烟测试
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
OUTPUT_DIR="${OUTPUT_DIR:-data/eval_runs/m0a_random_priority_${SPLIT}_seed${SEED}}"
SOLVER_BACKEND="${SOLVER_BACKEND:-nsga3_heuristic}"
NSGA3_POP_SIZE="${NSGA3_POP_SIZE:-8}"
NSGA3_N_GENERATIONS="${NSGA3_N_GENERATIONS:-5}"

if [[ ! -f "${EXTRACTED_DEMANDS}" ]]; then
  echo "[error] Extracted demands file not found: ${EXTRACTED_DEMANDS}" >&2
  echo "  Ensure data/test/test_seeds/ has been generated first." >&2
  exit 1
fi

echo "[M0a] seed=${SEED} split=${SPLIT}"
echo "[M0a] extracted_demands=${EXTRACTED_DEMANDS}"
echo "[M0a] output_dir=${OUTPUT_DIR}"

if [[ -n "${CONDA_ENV}" ]]; then
  PYTHONPATH=src conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python -u \
    -m llm4fairrouting.workflow.run_workflow \
    --output-dir "${OUTPUT_DIR}" \
    --priority-mode random \
    --offline \
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
    --priority-mode random \
    --offline \
    --solver-backend "${SOLVER_BACKEND}" \
    --dialogues "${DATASET_DIR}/dialogues.jsonl" \
    --extracted-demands "${EXTRACTED_DEMANDS}" \
    --nsga3-pop-size "${NSGA3_POP_SIZE}" \
    --nsga3-n-generations "${NSGA3_N_GENERATIONS}" \
    "$@"
fi
