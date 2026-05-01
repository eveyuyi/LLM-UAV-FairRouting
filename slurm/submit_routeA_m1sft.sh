#!/usr/bin/env bash
# Submit M1_sft Phase 1 (GPU) then Phase 2 (CPU, chained via dependency)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p slurm/logs

P1_JOB=$(sbatch --parsable slurm/run_routeA_m1sft_p1.slurm)
echo "Submitted Phase 1 (GPU): ${P1_JOB}"

P2_JOB=$(sbatch --parsable --dependency=afterok:${P1_JOB} slurm/run_routeA_m1sft_p2.slurm)
echo "Submitted Phase 2 (CPU, depends on ${P1_JOB}): ${P2_JOB}"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Phase 1 log: slurm/logs/routeA_m1sft_p1_${P1_JOB}.out"
echo "Phase 2 log: slurm/logs/routeA_m1sft_p2_${P2_JOB}_*.out"
