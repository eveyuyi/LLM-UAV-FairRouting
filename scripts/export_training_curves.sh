#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-base}"
INPUT_ROOT="${INPUT_ROOT:-data/hydra_outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-data/plots}"
MODE="${MODE:-auto}" # auto|sft|grpo|all
MAX_CURVES_PER_RUN="${MAX_CURVES_PER_RUN:-30}"

if [[ -n "${CONDA_ENV}" ]]; then
  _py=(conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python)
else
  _py=(env PYTHONNOUSERSITE=1 python)
fi

echo "Export curves: input=${INPUT_ROOT} output=${OUTPUT_DIR} mode=${MODE} conda=${CONDA_ENV:-<none>}"
"${_py[@]}" scripts/export_training_curves.py \
  --input-root "${INPUT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --mode "${MODE}" \
  --max-curves-per-run "${MAX_CURVES_PER_RUN}"

echo "Done. See ${OUTPUT_DIR}/README.md and PNG files under ${OUTPUT_DIR}/"
