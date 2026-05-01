#!/bin/bash
#SBATCH -J routeA_m0abc
#SBATCH -p compute
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH -o /scratch/yy6120/code/LLM-UAV-FairRouting/slurm/logs/routeA_m0abc_%A_%a.out
#SBATCH -e /scratch/yy6120/code/LLM-UAV-FairRouting/slurm/logs/routeA_m0abc_%A_%a.err
#SBATCH --array=0-5

# Route A: Continuous simulation (no --independent-windows)
# Array tasks:  0=m0a/7uav  1=m0b/7uav  2=m0c/7uav
#               3=m0a/10uav 4=m0b/10uav 5=m0c/10uav

METHODS=(random  uniform  rule-only  random  uniform  rule-only)
TAGS=(   m0a     m0b      m0c        m0a     m0b      m0c      )
DRONES=( 7       7        7          10      10       10       )

METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
N=${DRONES[$SLURM_ARRAY_TASK_ID]}

ROOT=/scratch/yy6120/code/LLM-UAV-FairRouting
SEED=4111
SPLIT=norm_eval
DATASET_DIR="${ROOT}/data/test/test_seeds/${SPLIT}/seed_${SEED}"
EXTRACTED_DEMANDS="${DATASET_DIR}/llm3_sft_pipeline.jsonl"
OUTPUT_DIR="${ROOT}/data/eval_runs/routeA_${N}uav_${TAG}_seed${SEED}"

export LLM4FAIRROUTING_ENV_FILE=/dev/null

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate llm-uav-fairrouting

cd "${ROOT}"

echo "[SLURM] task=${SLURM_ARRAY_TASK_ID} method=${TAG} drones=${N} seed=${SEED}"

PYTHONPATH=src python -u -m llm4fairrouting.workflow.run_workflow \
  --output-dir "${OUTPUT_DIR}" \
  --priority-mode "${METHOD}" \
  --offline \
  --no-independent-windows \
  --max-drones-per-station "${N}" \
  --solver-backend nsga3_heuristic \
  --nsga3-pop-size 8 \
  --nsga3-n-generations 5 \
  --dialogues "${DATASET_DIR}/dialogues.jsonl" \
  --extracted-demands "${EXTRACTED_DEMANDS}" \
  --time-slots $(seq 0 47 | tr '\n' ' ')

echo "[SLURM] task=${SLURM_ARRAY_TASK_ID} ${TAG}/${N}uav DONE"
