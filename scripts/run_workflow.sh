#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${LLM4FAIRROUTING_ENV_FILE:-$ROOT_DIR/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

CONDA_ENV="${LLM4FAIRROUTING_CONDA_ENV:-}"

cd "$ROOT_DIR"

if [[ -n "$CONDA_ENV" ]]; then
  PYTHONPATH=src conda run -n "$CONDA_ENV" python -m llm4fairrouting.workflow.run_workflow "$@"
else
  PYTHONPATH=src python -m llm4fairrouting.workflow.run_workflow "$@"
fi
