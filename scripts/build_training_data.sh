#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${LLM3_SMOKE_CONDA_ENV:-verl}"
OUTPUT_ROOT="${1:-data/train/quality_pilot}"
SEEDS_START="${SEEDS_START:-3009}"
SEEDS_END="${SEEDS_END:-3009}"
SEEDS=($(seq "$SEEDS_START" "$SEEDS_END"))
PARALLEL_SEEDS="${PARALLEL_SEEDS:-1}"
BATCH_SIZE="${BATCH_SIZE:-6}"
DIALOGUE_CONCURRENCY="${DIALOGUE_CONCURRENCY:-4}"
EXTRACTION_CONCURRENCY="${EXTRACTION_CONCURRENCY:-6}"
WINDOW="${WINDOW:-60}"
WINDOW_MINUTES="${WINDOW_MINUTES:-60}"
DEMANDS_PER_WINDOW_MIN="${DEMANDS_PER_WINDOW_MIN:-5}"
DEMANDS_PER_WINDOW_MAX="${DEMANDS_PER_WINDOW_MAX:-10}"
MEDICAL_RATIO="${MEDICAL_RATIO:-0.35}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

build_one_seed() {
  local seed="$1"
  local output_dir="${OUTPUT_ROOT}/seed_${seed}"
  local complete_markers=(
    "${output_dir}/events_manifest.jsonl"
    "${output_dir}/dialogues.jsonl"
    "${output_dir}/llm2_sft.jsonl"
    "${output_dir}/llm3_sft_clean.jsonl"
    "${output_dir}/llm3_sft_pipeline.jsonl"
    "${output_dir}/llm3_grpo_hard.jsonl"
    "${output_dir}/quality_report.json"
    "${output_dir}/release_manifest.json"
    "${output_dir}/dataset_manifest.json"
  )

  if [[ "$SKIP_COMPLETED" == "1" ]]; then
    local marker
    local missing_marker=0
    for marker in "${complete_markers[@]}"; do
      if [[ ! -s "$marker" ]]; then
        missing_marker=1
        break
      fi
    done
    if [[ "$missing_marker" -eq 0 ]]; then
      echo "[$(date '+%F %T')] skipping seed ${seed}: complete outputs already exist in ${output_dir}"
      return 0
    fi
  fi

  echo "[$(date '+%F %T')] building dataset for seed ${seed}"
  conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH=src \
    python -u -m llm4fairrouting.data.training_dataset_builder \
    --no-offline \
    --output-dir "$output_dir" \
    --window "$WINDOW" \
    --window-minutes "$WINDOW_MINUTES" \
    --demands-per-window-min "$DEMANDS_PER_WINDOW_MIN" \
    --demands-per-window-max "$DEMANDS_PER_WINDOW_MAX" \
    --medical-ratio "$MEDICAL_RATIO" \
    --styles direct technical \
    --batch-size "$BATCH_SIZE" \
    --dialogue-concurrency "$DIALOGUE_CONCURRENCY" \
    --extraction-concurrency "$EXTRACTION_CONCURRENCY" \
    --temperature 0.2 \
    --model gpt-4o-mini \
    --seed "$seed"
}

run_generation() {
  cd "$ROOT_DIR"
  mkdir -p "$OUTPUT_ROOT"

  if ! [[ "$PARALLEL_SEEDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "PARALLEL_SEEDS must be a positive integer, got: $PARALLEL_SEEDS" >&2
    exit 1
  fi
  if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer, got: $BATCH_SIZE" >&2
    exit 1
  fi
  if ! [[ "$DIALOGUE_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    echo "DIALOGUE_CONCURRENCY must be a positive integer, got: $DIALOGUE_CONCURRENCY" >&2
    exit 1
  fi
  if ! [[ "$EXTRACTION_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    echo "EXTRACTION_CONCURRENCY must be a positive integer, got: $EXTRACTION_CONCURRENCY" >&2
    exit 1
  fi

  echo "[$(date '+%F %T')] using conda env: $CONDA_ENV"
  echo "[$(date '+%F %T')] output root: $OUTPUT_ROOT"
  echo "[$(date '+%F %T')] seeds: ${SEEDS[*]}"
  echo "[$(date '+%F %T')] parallel seeds: $PARALLEL_SEEDS, batch size: $BATCH_SIZE, dialogue concurrency: $DIALOGUE_CONCURRENCY, extraction concurrency: $EXTRACTION_CONCURRENCY, skip completed: $SKIP_COMPLETED"

  if [[ "$PARALLEL_SEEDS" -eq 1 ]]; then
    for seed in "${SEEDS[@]}"; do
      build_one_seed "$seed"
    done
    return 0
  fi

  local active=0
  local seed
  for seed in "${SEEDS[@]}"; do
    build_one_seed "$seed" &
    ((active+=1))
    if (( active >= PARALLEL_SEEDS )); then
      wait -n
      ((active-=1))
    fi
  done
  wait
}

if [[ "${2:-}" == "--foreground" ]]; then
  run_generation
else
  LOG_DIR="$ROOT_DIR/logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/build_quality_pilot_training_data_$(date '+%Y%m%d_%H%M%S').log"

  nohup "$0" "$OUTPUT_ROOT" --foreground >"$LOG_FILE" 2>&1 &
  PID=$!

  echo "Started in background."
  echo "PID: $PID"
  echo "Log: $LOG_FILE"
  echo "Tail logs with: tail -f \"$LOG_FILE\""
fi
