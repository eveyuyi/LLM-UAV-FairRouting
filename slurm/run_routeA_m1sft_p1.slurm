#!/bin/bash
#SBATCH -J routeA_m1sft_p1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -n 8
#SBATCH --mem=40G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/yy6120/code/LLM-UAV-FairRouting/slurm/logs/routeA_m1sft_p1_%j.out
#SBATCH -e /scratch/yy6120/code/LLM-UAV-FairRouting/slurm/logs/routeA_m1sft_p1_%j.err

# Route A — M1_sft Phase 1: LLM inference only (SFT-only model, before GRPO)
# Generates weight_configs to data/eval_runs/formal_m1_sft_seed4111/
# Run Phase 2 (solver) via run_routeA_m1sft_p2.slurm after this completes.

ROOT=/scratch/yy6120/code/LLM-UAV-FairRouting
SEED=4111
SPLIT=norm_eval
DATASET_DIR="${ROOT}/data/test/test_seeds/${SPLIT}/seed_${SEED}"
EXTRACTED_DEMANDS="${DATASET_DIR}/llm3_sft_pipeline.jsonl"
VLLM_ENV=/scratch/yy6120/conda/envs/vllm-serve
VLLM_PORT=8310

TAG="m1_sft"
MODEL_PATH="${ROOT}/checkpoints/expA_sft_baseline/global_step_594/huggingface_lora_merged"
SERVED_MODEL_NAME="qwen3-sft"
OUTPUT_DIR="${ROOT}/data/eval_runs/formal_${TAG}_seed${SEED}"

export LLM4FAIRROUTING_ENV_FILE=/dev/null

cd "${ROOT}"
echo "[Phase1-SFT] model=${MODEL_PATH}"
echo "[Phase1-SFT] output=${OUTPUT_DIR}"

PYTHONNOUSERSITE=1 "${VLLM_ENV}/bin/python" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port "${VLLM_PORT}" \
  --model "${MODEL_PATH}" --served-model-name "${SERVED_MODEL_NAME}" \
  --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --trust-remote-code --enforce-eager &
VLLM_PID=$!

for i in $(seq 1 60); do
  sleep 2
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "[vLLM] ready after $((i*2))s"; break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[error] vLLM died" >&2; exit 1
  fi
done

PYTHONPATH=src conda run --no-capture-output -n llm-uav-fairrouting env PYTHONNOUSERSITE=1 \
  python -u -m llm4fairrouting.workflow.run_workflow \
  --output-dir "${OUTPUT_DIR}" \
  --priority-mode llm-only \
  --skip-solver \
  --model "${SERVED_MODEL_NAME}" \
  --api-base "http://127.0.0.1:${VLLM_PORT}/v1" --api-key EMPTY \
  --dialogues "${DATASET_DIR}/dialogues.jsonl" \
  --extracted-demands "${EXTRACTED_DEMANDS}" \
  --time-slots $(seq 0 47 | tr '\n' ' ')

kill "${VLLM_PID}" 2>/dev/null || true
echo "[Phase1-SFT] DONE — weight_configs → ${OUTPUT_DIR}"
