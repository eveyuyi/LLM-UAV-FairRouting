#!/usr/bin/env bash
# Submit Route B experiments (independent windows, 3 UAVs, single-batch solve)
# M1_sft weight_configs depend on routeA Phase-1 job 15298097; pass its ID if still running.
# Usage:  bash slurm/submit_routeB.sh [M1_SFT_P1_JOB_ID]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p slurm/logs

M1SFT_P1_JOB="${1:-}"

JOB_M0=$(sbatch --parsable slurm/run_routeB_m0abc.slurm)
echo "Submitted M0abc routeB: ${JOB_M0}"

if [[ -n "${M1SFT_P1_JOB}" ]]; then
  JOB_M1=$(sbatch --parsable --dependency=afterok:${M1SFT_P1_JOB} slurm/run_routeB_m1.slurm)
  echo "Submitted M1 routeB (depends on ${M1SFT_P1_JOB}): ${JOB_M1}"
else
  JOB_M1=$(sbatch --parsable slurm/run_routeB_m1.slurm)
  echo "Submitted M1 routeB: ${JOB_M1}"
fi

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    slurm/logs/routeB_*"
