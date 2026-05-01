#!/bin/bash
#SBATCH -J routeB_m0abc
#SBATCH -p compute
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o /scratch/yy6120/code/LLM-UAV-FairRouting/slurm/logs/routeB_m0abc_%A_%a.out
#SBATCH -e /scratch/yy6120/code/LLM-UAV-FairRouting/slurm/logs/routeB_m0abc_%A_%a.err
#SBATCH --array=0-2

# Route B: independent windows, 3 UAVs, all demands at t=0 (single-batch solve)
# tasks: 0=m0a  1=m0b  2=m0c

METHODS=(random  uniform  rule-only)
TAGS=(   m0a     m0b      m0c      )

METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

ROOT=/scratch/yy6120/code/LLM-UAV-FairRouting
SEED=4111
SPLIT=norm_eval
DATASET_DIR="${ROOT}/data/test/test_seeds/${SPLIT}/seed_${SEED}"
EXTRACTED_DEMANDS="${DATASET_DIR}/llm3_sft_pipeline.jsonl"
OUTPUT_DIR="${ROOT}/data/eval_runs/routeB_3uav_${TAG}_seed${SEED}"

export LLM4FAIRROUTING_ENV_FILE=/dev/null

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate llm-uav-fairrouting

cd "${ROOT}"
echo "[SLURM] task=${SLURM_ARRAY_TASK_ID} method=${TAG} route=B 3uav seed=${SEED}"

PYTHONPATH=src python -u -m llm4fairrouting.workflow.run_workflow \
  --output-dir "${OUTPUT_DIR}" \
  --priority-mode "${METHOD}" \
  --offline \
  --independent-windows \
  --max-drones-per-station 3 \
  --solver-backend nsga3_heuristic \
  --nsga3-pop-size 8 \
  --nsga3-n-generations 5 \
  --dialogues "${DATASET_DIR}/dialogues.jsonl" \
  --extracted-demands "${EXTRACTED_DEMANDS}" \
  --time-slots $(seq 0 47 | tr '\n' ' ')

echo "[SLURM] task=${SLURM_ARRAY_TASK_ID} ${TAG} routeB DONE"
