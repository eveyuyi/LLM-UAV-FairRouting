#!/usr/bin/env bash
# Submit Route A experiments (continuous simulation, 7 & 10 UAVs)
# Usage: bash slurm/submit_routeA.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p slurm/logs

JOB_M0=$(sbatch --parsable slurm/run_routeA_m0abc.slurm)
echo "Submitted M0abc job: ${JOB_M0}"

JOB_M1=$(sbatch --parsable slurm/run_routeA_m1.slurm)
echo "Submitted M1 job:    ${JOB_M1}"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    slurm/logs/routeA_*"
echo ""
echo "When complete, run analysis:"
echo "  python scripts/eval_formal_comparison.py  # needs routeA result dirs wired in"
