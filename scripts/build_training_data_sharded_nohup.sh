#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-data/train/quality_pilot_v2}"

SEEDS_START="${SEEDS_START:-3001}"
SEEDS_END="${SEEDS_END:-3008}"
WORKERS="${WORKERS:-2}"
INNER_PARALLEL_SEEDS="${INNER_PARALLEL_SEEDS:-1}"

# Keep defaults aligned with scripts/build_training_data.sh.
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
BATCH_SIZE="${BATCH_SIZE:-6}"
DIALOGUE_CONCURRENCY="${DIALOGUE_CONCURRENCY:-4}"
EXTRACTION_CONCURRENCY="${EXTRACTION_CONCURRENCY:-6}"
WINDOW="${WINDOW:-60}"
WINDOW_MINUTES="${WINDOW_MINUTES:-60}"
DEMANDS_PER_WINDOW_MIN="${DEMANDS_PER_WINDOW_MIN:-5}"
DEMANDS_PER_WINDOW_MAX="${DEMANDS_PER_WINDOW_MAX:-10}"
MEDICAL_RATIO="${MEDICAL_RATIO:-0.35}"
LLM3_SMOKE_CONDA_ENV="${LLM3_SMOKE_CONDA_ENV:-verl}"
CONDA_BIN="${CONDA_BIN:-}"

is_pos_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_int() {
  [[ "$1" =~ ^-?[0-9]+$ ]]
}

if ! is_int "$SEEDS_START" || ! is_int "$SEEDS_END"; then
  echo "SEEDS_START/SEEDS_END must be integers. Got: $SEEDS_START .. $SEEDS_END" >&2
  exit 1
fi
if (( SEEDS_END < SEEDS_START )); then
  echo "SEEDS_END must be >= SEEDS_START. Got: $SEEDS_START .. $SEEDS_END" >&2
  exit 1
fi
if ! is_pos_int "$WORKERS"; then
  echo "WORKERS must be a positive integer. Got: $WORKERS" >&2
  exit 1
fi
if ! is_pos_int "$INNER_PARALLEL_SEEDS"; then
  echo "INNER_PARALLEL_SEEDS must be a positive integer. Got: $INNER_PARALLEL_SEEDS" >&2
  exit 1
fi

resolve_conda_bin() {
  if [[ -n "$CONDA_BIN" && -x "$CONDA_BIN" ]]; then
    echo "$CONDA_BIN"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  local candidates=(
    "/root/miniconda3/bin/conda"
    "$HOME/miniconda3/bin/conda"
    "/opt/conda/bin/conda"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if ! CONDA_BIN_RESOLVED="$(resolve_conda_bin)"; then
  echo "Unable to find conda binary. Set CONDA_BIN explicitly, e.g.:" >&2
  echo "  CONDA_BIN=/root/miniconda3/bin/conda bash scripts/build_training_data_sharded_nohup.sh ..." >&2
  exit 1
fi
CONDA_DIR="$(dirname "$CONDA_BIN_RESOLVED")"

TOTAL_SEEDS=$((SEEDS_END - SEEDS_START + 1))
if (( WORKERS > TOTAL_SEEDS )); then
  WORKERS="$TOTAL_SEEDS"
fi

CHUNK_SIZE=$(((TOTAL_SEEDS + WORKERS - 1) / WORKERS))

LOG_PARENT="$ROOT_DIR/logs"
RUN_TAG="build_training_sharded_$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="$LOG_PARENT/$RUN_TAG"
mkdir -p "$RUN_LOG_DIR"

echo "[$(date '+%F %T')] root: $ROOT_DIR"
echo "[$(date '+%F %T')] output root: $OUTPUT_ROOT"
echo "[$(date '+%F %T')] seeds: $SEEDS_START..$SEEDS_END (total=$TOTAL_SEEDS)"
echo "[$(date '+%F %T')] workers: $WORKERS, chunk size: $CHUNK_SIZE"
echo "[$(date '+%F %T')] inner parallel seeds per worker: $INNER_PARALLEL_SEEDS"
echo "[$(date '+%F %T')] conda bin: $CONDA_BIN_RESOLVED"
echo "[$(date '+%F %T')] logs dir: $RUN_LOG_DIR"
echo

declare -a PIDS=()

for ((i=0; i<WORKERS; i++)); do
  shard_start=$((SEEDS_START + i * CHUNK_SIZE))
  shard_end=$((shard_start + CHUNK_SIZE - 1))
  if (( shard_start > SEEDS_END )); then
    continue
  fi
  if (( shard_end > SEEDS_END )); then
    shard_end=$SEEDS_END
  fi

  worker_id=$((i + 1))
  worker_log="$RUN_LOG_DIR/worker_${worker_id}_seed_${shard_start}_${shard_end}.log"

  cmd="cd \"$ROOT_DIR\" && \
export PATH=\"$CONDA_DIR:\$PATH\" && \
SEEDS_START=$shard_start \
SEEDS_END=$shard_end \
PARALLEL_SEEDS=$INNER_PARALLEL_SEEDS \
SKIP_COMPLETED=$SKIP_COMPLETED \
BATCH_SIZE=$BATCH_SIZE \
DIALOGUE_CONCURRENCY=$DIALOGUE_CONCURRENCY \
EXTRACTION_CONCURRENCY=$EXTRACTION_CONCURRENCY \
WINDOW=$WINDOW \
WINDOW_MINUTES=$WINDOW_MINUTES \
DEMANDS_PER_WINDOW_MIN=$DEMANDS_PER_WINDOW_MIN \
DEMANDS_PER_WINDOW_MAX=$DEMANDS_PER_WINDOW_MAX \
MEDICAL_RATIO=$MEDICAL_RATIO \
LLM3_SMOKE_CONDA_ENV=$LLM3_SMOKE_CONDA_ENV \
bash scripts/build_training_data.sh \"$OUTPUT_ROOT\" --foreground"

  nohup bash -lc "$cmd" >"$worker_log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo "worker ${worker_id}: seeds ${shard_start}-${shard_end}, pid=${pid}, log=${worker_log}"
done

echo
echo "Started ${#PIDS[@]} worker process(es)."
echo "Tail all logs: tail -f \"$RUN_LOG_DIR\"/*.log"
echo "Check status: ps -fp ${PIDS[*]}"
