#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-verl}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy_api_key}"

RAW_MODEL="${RAW_MODEL:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/model--Qwen-Qwen3-4B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-instruct-2507}"
TAG="${TAG:-qwen3_4b_2507}"
SEED="${SEED:-4112}"
SPLIT="${SPLIT:-norm_eval}"

# Keep defaults aligned with the two resume scripts.
LLM_ONLY_PORT="${LLM_ONLY_PORT:-8200}"
HYBRID_PORT="${HYBRID_PORT:-8600}"
LLM_ONLY_GPU_MEMORY_UTILIZATION="${LLM_ONLY_GPU_MEMORY_UTILIZATION:-0.5}"
HYBRID_GPU_MEMORY_UTILIZATION="${HYBRID_GPU_MEMORY_UTILIZATION:-0.45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
RANK_ONLY_MODE="${RANK_ONLY_MODE:-llm3_only}"
CUDA_VISIBLE_DEVICES_ONE_SHOT="${CUDA_VISIBLE_DEVICES_ONE_SHOT:-0}"
MAX_ATTEMPTS_PER_MODE="${MAX_ATTEMPTS_PER_MODE:-5}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-3}"
LLM4FAIRROUTING_CONTINUE_ON_WINDOW_ERROR="${LLM4FAIRROUTING_CONTINUE_ON_WINDOW_ERROR:-1}"

DATASET_DIR="data/test/test_seeds/${SPLIT}/seed_${SEED}"

resolve_model_path() {
  local raw="$1"

  if [[ ! -d "${raw}" ]]; then
    return 1
  fi
  if [[ -f "${raw}/config.json" ]]; then
    echo "${raw}"
    return 0
  fi

  if [[ -d "${raw}/snapshots" ]]; then
    local candidate
    candidate="$(ls -1dt "${raw}/snapshots"/* 2>/dev/null | head -n 1 || true)"
    if [[ -n "${candidate}" && -f "${candidate}/config.json" ]]; then
      echo "${candidate}"
      return 0
    fi
  fi
  return 1
}

pids_on_port() {
  local port="$1"
  ss -ltnp "( sport = :${port} )" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u
}

cleanup_port() {
  local port="$1"
  local pids=()
  mapfile -t pids < <(pids_on_port "${port}")
  if ((${#pids[@]} == 0)); then
    return 0
  fi

  echo "[cleanup] port=${port} stale pids: ${pids[*]}"
  kill "${pids[@]}" >/dev/null 2>&1 || true
  sleep 1

  local alive=()
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      alive+=("${pid}")
    fi
  done
  if ((${#alive[@]})); then
    kill -9 "${alive[@]}" >/dev/null 2>&1 || true
  fi
}

wait_ready() {
  local port="$1"
  local served="$2"

  for i in $(seq 1 600); do
    local payload
    payload="$(curl -sSf --connect-timeout 1 --max-time 2 "http://127.0.0.1:${port}/v1/models" 2>/dev/null || true)"
    if [[ -n "${payload}" ]]; then
      if python - "$served" "$payload" <<'PY'
import json
import sys

served = sys.argv[1]
payload = sys.argv[2]
try:
    data = json.loads(payload)
except Exception:
    raise SystemExit(1)

models = data.get("data", []) if isinstance(data, dict) else []
ids = {str(item.get("id", "")) for item in models if isinstance(item, dict)}
raise SystemExit(0 if served in ids else 2)
PY
      then
        echo "[ready] port=${port} model=${served}"
        return 0
      fi
    fi

    if (( i % 15 == 0 )); then
      echo "[wait] port=${port} still not ready (${i}/600)"
    fi
    sleep 2
  done

  echo "[not_ready] port=${port} timeout"
  return 1
}

run_one_mode_once() {
  local mode="$1"
  local port="$2"
  local gpu_mem="$3"
  local out="data/eval_runs/test_seeds_${SPLIT}/${TAG}_seed${SEED}_${mode}"
  local log_file="logs/${mode}_${TAG}_${port}_oneshot.log"

  if [[ -s "${out}/evals/alignment.json" ]]; then
    echo "[skip] ${mode} already done: ${out}/evals/alignment.json"
    return 0
  fi

  cleanup_port "${port}"

  local model_path
  model_path="$(resolve_model_path "${RAW_MODEL}" || true)"
  if [[ -z "${model_path}" ]]; then
    echo "[fatal] invalid RAW_MODEL path: ${RAW_MODEL}" >&2
    exit 1
  fi

  mkdir -p logs
  echo "[start] mode=${mode} port=${port} tag=${TAG} model_path=${model_path}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_ONE_SHOT}" \
  CONDA_ENV="${CONDA_ENV}" \
  MODEL_PATH="${model_path}" \
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  PORT="${port}" \
  GPU_MEMORY_UTILIZATION="${gpu_mem}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  nohup bash scripts/serve_vllm_model.sh > "${log_file}" 2>&1 &

  local server_pid="$!"
  cleanup_server() {
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" 2>/dev/null || true
  }
  trap cleanup_server RETURN

  wait_ready "${port}" "${SERVED_MODEL_NAME}" || {
    echo "[error] server startup failed, log: ${log_file}" >&2
    return 1
  }

  echo "[eval] mode=${mode} split=${SPLIT} seed=${SEED}"
  CONDA_ENV="${CONDA_ENV}" OPENAI_API_KEY="${OPENAI_API_KEY}" \
  API_BASE="http://127.0.0.1:${port}/v1" MODEL_NAME="${SERVED_MODEL_NAME}" \
  PRIORITY_MODE="${mode}" RANK_ONLY_MODE="${RANK_ONLY_MODE}" AUTO_TIMESTAMP_OUTPUT_ROOT=0 \
  LLM4FAIRROUTING_CONTINUE_ON_WINDOW_ERROR="${LLM4FAIRROUTING_CONTINUE_ON_WINDOW_ERROR}" \
  OUTPUT_ROOT="${out}" \
  bash scripts/eval_single_model_priority_alignment.sh "${DATASET_DIR}"

  if [[ ! -s "${out}/evals/alignment.json" ]]; then
    echo "[error] ${mode} finished without alignment.json: ${out}/evals/alignment.json" >&2
    return 1
  fi

  echo "[done] ${mode}: ${out}/evals/alignment.json"
}

run_one_mode() {
  local mode="$1"
  local port="$2"
  local gpu_mem="$3"

  local attempt
  for attempt in $(seq 1 "${MAX_ATTEMPTS_PER_MODE}"); do
    echo "[attempt] mode=${mode} try=${attempt}/${MAX_ATTEMPTS_PER_MODE}"
    if run_one_mode_once "${mode}" "${port}" "${gpu_mem}"; then
      return 0
    fi
    if (( attempt < MAX_ATTEMPTS_PER_MODE )); then
      echo "[retry] mode=${mode} retry after ${RETRY_SLEEP_SECONDS}s"
      sleep "${RETRY_SLEEP_SECONDS}"
    fi
  done

  echo "[fatal] mode=${mode} failed after ${MAX_ATTEMPTS_PER_MODE} attempts" >&2
  return 1
}

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "[fatal] missing dataset dir: ${DATASET_DIR}" >&2
  exit 1
fi

run_one_mode "llm-only" "${LLM_ONLY_PORT}" "${LLM_ONLY_GPU_MEMORY_UTILIZATION}"
run_one_mode "hybrid" "${HYBRID_PORT}" "${HYBRID_GPU_MEMORY_UTILIZATION}"

echo "[all done] Backfill completed for ${TAG}, ${SPLIT}, seed ${SEED}."
