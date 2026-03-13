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
  PYTHONPATH=src conda run --no-capture-output -n "$CONDA_ENV" python -u -m llm4fairrouting.data.demand_dialogue_dataset "$@"
else
  PYTHONPATH=src python -u -m llm4fairrouting.data.demand_dialogue_dataset "$@"
fi
