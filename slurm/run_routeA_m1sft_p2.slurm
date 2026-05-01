#!/bin/bash
#SBATCH -J routeA_m1sft_p2
#SBATCH -p compute
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH -o /scratch/yy6120/code/LLM-UAV-FairRouting/slurm/logs/routeA_m1sft_p2_%A_%a.out
#SBATCH -e /scratch/yy6120/code/LLM-UAV-FairRouting/slurm/logs/routeA_m1sft_p2_%A_%a.err
#SBATCH --array=0-1

# Route A — M1_sft Phase 2: solver-only (continuous simulation)
# task 0 → 7 UAVs   task 1 → 10 UAVs
# Must run after Phase 1 (use --dependency=afterok:<P1_JOB_ID>)

DRONES=(7 10)
N=${DRONES[$SLURM_ARRAY_TASK_ID]}

ROOT=/scratch/yy6120/code/LLM-UAV-FairRouting
SEED=4111
SPLIT=norm_eval
DATASET_DIR="${ROOT}/data/test/test_seeds/${SPLIT}/seed_${SEED}"
EXTRACTED_DEMANDS="${DATASET_DIR}/llm3_sft_pipeline.jsonl"

TAG="m1_sft"
PHASE1_BASE="${ROOT}/data/eval_runs/formal_${TAG}_seed${SEED}"
PHASE1_RUN=$(ls -td "${PHASE1_BASE}"/run_*/ 2>/dev/null | head -1)
if [[ -z "${PHASE1_RUN}" ]]; then
  echo "[error] No Phase-1 run found in ${PHASE1_BASE}" >&2; exit 1
fi
WEIGHT_CONFIGS="${PHASE1_RUN}weight_configs"
if [[ ! -d "${WEIGHT_CONFIGS}" ]]; then
  echo "[error] weight_configs not found: ${WEIGHT_CONFIGS}" >&2; exit 1
fi

OUTPUT_DIR="${ROOT}/data/eval_runs/routeA_${N}uav_${TAG}_seed${SEED}"

export LLM4FAIRROUTING_ENV_FILE=/dev/null

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate llm-uav-fairrouting

cd "${ROOT}"
echo "[Phase2-SFT] task=${SLURM_ARRAY_TASK_ID} drones=${N}"
echo "[Phase2-SFT] weight_configs: ${WEIGHT_CONFIGS}"

PYTHONPATH=src python -u -m llm4fairrouting.workflow.run_workflow \
  --output-dir "${OUTPUT_DIR}" \
  --priority-mode random \
  --no-independent-windows \
  --max-drones-per-station "${N}" \
  --solver-backend nsga3_heuristic \
  --nsga3-pop-size 8 \
  --nsga3-n-generations 5 \
  --dialogues "${DATASET_DIR}/dialogues.jsonl" \
  --extracted-demands "${EXTRACTED_DEMANDS}" \
  --load-weight-configs "${WEIGHT_CONFIGS}" \
  --time-slots $(seq 0 47 | tr '\n' ' ')

echo "[Phase2-SFT] ${N}uav DONE"
