#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-verl}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy_api_key}"
PRIORITY_MODE="${PRIORITY_MODE:-hybrid}"
RANK_ONLY_MODE="${RANK_ONLY_MODE:-llm3_only}"

# Reserved default port segment for hybrid 3GPU runs.
# Optional override: set PORT_BASE manually per job.
PORT_BASE="${PORT_BASE:-8600}"
GPUS=(${GPUS_OVERRIDE:-0 1 2})
PORTS=("${PORT_BASE}" "$((PORT_BASE + 1))" "$((PORT_BASE + 2))")
CLEAN_PORTS_BEFORE_START="${CLEAN_PORTS_BEFORE_START:-1}"

HARD_SEEDS=(5101 5102 5103 5104 5105 5106)
NORM_SEEDS=(4111 4112 5111 5112 5113)

# format: tag|served_model_name|model_path
MODELS=(
  "expA_gs594|expA-sft-baseline-gs594|data/checkpoints/expA_sft_baseline/global_step_594/huggingface_lora_merged"
  "expB_gs50|expB-grpo-hard-gs50|data/checkpoints/large_v1/expB_grpo_hard_merged_gs50"
  "expB_gs100|expB-grpo-hard-gs100|data/checkpoints/large_v1/expB_grpo_hard_merged_gs100"
  "expB_gs150|expB-grpo-hard-gs150|data/checkpoints/large_v1/expB_grpo_hard_merged_gs150"
  "expB_gs171|expB-grpo-hard-gs171|data/checkpoints/large_v1/expB_grpo_hard_merged_gs171"
  "expC_gs100|expC-grpo-mixed-8c-gs100|data/checkpoints/large_v1/expC_grpo_mixed_8cards_merged_gs100"
  "expC_gs300|expC-grpo-mixed-8c-gs300|data/checkpoints/large_v1/expC_grpo_mixed_8cards_merged_gs300"
  "expC_gs500|expC-grpo-mixed-8c-gs500|data/checkpoints/large_v1/expC_grpo_mixed_8cards_merged_gs500"
  "expC_gs600|expC-grpo-mixed-8c-gs600|data/checkpoints/large_v1/expC_grpo_mixed_8cards_merged_gs600"
  "expC_gs700|expC-grpo-mixed-8c-gs700|data/checkpoints/large_v1/expC_grpo_mixed_8cards_merged_gs700"
  "expC_gs800|expC-grpo-mixed-8c-gs800|data/checkpoints/large_v1/expC_grpo_mixed_8cards_merged_gs800"
  "expC_gs900|expC-grpo-mixed-8c-gs900|data/checkpoints/large_v1/expC_grpo_mixed_8cards_merged_gs900"
  "expC_gs992|expC-grpo-mixed-8c-gs992|data/checkpoints/large_v1/expC_grpo_mixed_8cards_merged_gs992"
  "qwen3_4b_2507|qwen3-4b-instruct-2507|/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/model--Qwen-Qwen3-4B-Instruct-2507"
)

# 14 models -> 5 rounds (3+3+3+3+2)
ROUNDS=(
  "0 1 2"
  "3 4 5"
  "6 7 8"
  "9 10 11"
  "12 13"
)

mkdir -p logs
FAILED_LOG="logs/${PRIORITY_MODE}_4gpu_failed_cases.log"
touch "${FAILED_LOG}"

server_pids=()

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

fail() {
  echo "[fatal] $*" >&2
  exit 1
}

resolve_model_path() {
  local raw="$1"

  if [[ ! -d "${raw}" ]]; then
    return 1
  fi
  if [[ -f "${raw}/config.json" ]]; then
    echo "${raw}"
    return 0
  fi

  # HF cache root form: model--xxx/snapshots/<rev>/config.json
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

port_is_busy() {
  local port="$1"
  ss -ltn "( sport = :${port} )" 2>/dev/null | awk 'NR>1{print}' | wc -l | awk '{exit !($1>0)}'
}

pids_on_port() {
  local port="$1"
  ss -ltnp "( sport = :${port} )" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u
}

kill_port_listeners() {
  local port="$1"
  local pids=()
  mapfile -t pids < <(pids_on_port "${port}")
  if ((${#pids[@]} == 0)); then
    return 0
  fi

  echo "[cleanup] port=${port} killing stale pids: ${pids[*]}"
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

ensure_ports_clean() {
  local port
  for port in "${PORTS[@]}"; do
    if port_is_busy "${port}" && [[ "${CLEAN_PORTS_BEFORE_START}" == "1" ]]; then
      kill_port_listeners "${port}"
    fi

    # vLLM workers may take a moment to exit after kill.
    local _i
    for _i in $(seq 1 8); do
      if ! port_is_busy "${port}"; then
        break
      fi
      sleep 1
      if [[ "${CLEAN_PORTS_BEFORE_START}" == "1" ]]; then
        kill_port_listeners "${port}"
      fi
    done

    if port_is_busy "${port}"; then
      fail "port still in use after cleanup retries: ${port} (set another PORT_BASE)"
    fi
  done
}

model_served_on_port() {
  local port="$1"
  local served="$2"
  local payload
  payload="$(curl -sSf --connect-timeout 1 --max-time 2 "http://127.0.0.1:${port}/v1/models" 2>/dev/null)" || return 1

  python - "$served" "$payload" <<'PY'
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
}

preflight_checks() {
  have_cmd conda || fail "conda not found in PATH"
  have_cmd curl || fail "curl not found in PATH"
  have_cmd ss || fail "ss not found in PATH (install iproute2)"

  conda run --no-capture-output -n "${CONDA_ENV}" python -V >/dev/null \
    || fail "conda env not usable: ${CONDA_ENV}"

  ensure_ports_clean

  local item tag served path resolved
  for item in "${MODELS[@]}"; do
    IFS='|' read -r tag served path <<< "${item}"
    resolved="$(resolve_model_path "${path}" || true)"
    if [[ -z "${resolved}" ]]; then
      fail "model path invalid for ${tag}: ${path} (need config.json or snapshots/*/config.json)"
    fi
  done
}

start_server() {
  local gpu="$1" port="$2" tag="$3" served="$4" raw_path="$5"
  local model_path

  model_path="$(resolve_model_path "${raw_path}" || true)"
  if [[ -z "${model_path}" ]]; then
    echo "[start_failed] tag=${tag} raw_path=${raw_path}" >&2
    return 1
  fi

  local log_file="logs/${PRIORITY_MODE}_${tag}_${port}.log"
  echo "[start] gpu=${gpu} port=${port} tag=${tag} model_path=${model_path}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  CONDA_ENV="${CONDA_ENV}" \
  MODEL_PATH="${model_path}" \
  SERVED_MODEL_NAME="${served}" \
  PORT="${port}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}" \
  nohup bash scripts/serve_vllm_model.sh > "${log_file}" 2>&1 &

  server_pids+=($!)
}

wait_ready() {
  local port="$1"
  local log_file="$2"
  local pid="$3"
  local served="$4"

  # up to 20 minutes
  local _i
  for _i in $(seq 1 600); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "[not_ready] port=${port} pid_exited=${pid}" >&2
      sed -n '1,200p' "${log_file}" >&2 || true
      return 1
    fi
    if model_served_on_port "${port}" "${served}"; then
      echo "[ready] port=${port} model=${served}"
      return 0
    fi
    sleep 2
  done

  echo "[not_ready] port=${port} timeout" >&2
  sed -n '1,240p' "${log_file}" >&2 || true
  return 1
}

stop_servers() {
  if ((${#server_pids[@]})); then
    echo "[stop] ${server_pids[*]}"
    kill "${server_pids[@]}" >/dev/null 2>&1 || true
    wait "${server_pids[@]}" 2>/dev/null || true
  fi
  server_pids=()
}

already_done() {
  local split="$1" seed="$2" tag="$3"
  local out="data/eval_runs/test_seeds_${split}/${tag}_seed${seed}_${PRIORITY_MODE}"
  [[ -s "${out}/evals/alignment.json" ]]
}

eval_one() {
  local split="$1" seed="$2" port="$3" served="$4" tag="$5"
  local out="data/eval_runs/test_seeds_${split}/${tag}_seed${seed}_${PRIORITY_MODE}"

  if already_done "${split}" "${seed}" "${tag}"; then
    echo "[skip] ${split} seed=${seed} ${tag}"
    return 0
  fi

  local d="data/test/test_seeds/${split}/seed_${seed}"
  echo "[eval] split=${split} seed=${seed} tag=${tag} port=${port}"

  if ! CONDA_ENV="${CONDA_ENV}" OPENAI_API_KEY="${OPENAI_API_KEY}" \
    API_BASE="http://127.0.0.1:${port}/v1" MODEL_NAME="${served}" \
    PRIORITY_MODE="${PRIORITY_MODE}" RANK_ONLY_MODE="${RANK_ONLY_MODE}" AUTO_TIMESTAMP_OUTPUT_ROOT=0 \
    LLM4FAIRROUTING_PRIORITY_RANK_CHUNK_SIZE="${LLM4FAIRROUTING_PRIORITY_RANK_CHUNK_SIZE:-3}" \
    OUTPUT_ROOT="${out}" \
    bash scripts/eval_single_model_priority_alignment.sh "${d}"; then
    echo "[failed] split=${split} seed=${seed} tag=${tag}" >&2
    echo "${split} ${seed} ${tag}" >> "${FAILED_LOG}"
    return 1
  fi
}

wait_eval_pids() {
  local ec=0
  local p
  for p in "$@"; do
    wait "$p" || ec=1
  done
  return "${ec}"
}

trap stop_servers EXIT

preflight_checks

for ridx in "${!ROUNDS[@]}"; do
  echo "===== Round $((ridx + 1)) / ${#ROUNDS[@]} ====="
  ensure_ports_clean

  read -r -a ids <<< "${ROUNDS[$ridx]}"
  round_tags=()
  round_served=()
  round_ports=()
  round_logs=()
  round_pids=()

  for i in "${!ids[@]}"; do
    mid="${ids[$i]}"
    IFS='|' read -r tag served path <<< "${MODELS[$mid]}"
    gpu="${GPUS[$i]}"
    port="${PORTS[$i]}"

    round_tags+=("${tag}")
    round_served+=("${served}")
    round_ports+=("${port}")
    round_logs+=("logs/${PRIORITY_MODE}_${tag}_${port}.log")

    start_server "${gpu}" "${port}" "${tag}" "${served}" "${path}" || {
      echo "startup_failed round=$((ridx + 1)) tag=${tag}" >> "${FAILED_LOG}"
      stop_servers
      continue 2
    }
    round_pids+=("${server_pids[-1]}")
  done

  for i in "${!round_ports[@]}"; do
    port="${round_ports[$i]}"
    tag="${round_tags[$i]}"
    log_file="${round_logs[$i]}"
    pid="${round_pids[$i]}"
    if ! wait_ready "${port}" "${log_file}" "${pid}" "${round_served[$i]}"; then
      echo "[warn] startup failed for tag=${tag} port=${port}, skip this round" >&2
      for t in "${round_tags[@]}"; do
        echo "startup_failed round=$((ridx + 1)) tag=${t}" >> "${FAILED_LOG}"
      done
      stop_servers
      continue 2
    fi
  done

  for seed in "${HARD_SEEDS[@]}"; do
    pids=()
    for i in "${!round_tags[@]}"; do
      eval_one "hard_eval" "${seed}" "${round_ports[$i]}" "${round_served[$i]}" "${round_tags[$i]}" &
      pids+=($!)
    done
    if ! wait_eval_pids "${pids[@]}"; then
      echo "[warn] hard_eval failed at seed=${seed}, continue" >&2
    fi
  done

  for seed in "${NORM_SEEDS[@]}"; do
    pids=()
    for i in "${!round_tags[@]}"; do
      eval_one "norm_eval" "${seed}" "${round_ports[$i]}" "${round_served[$i]}" "${round_tags[$i]}" &
      pids+=($!)
    done
    if ! wait_eval_pids "${pids[@]}"; then
      echo "[warn] norm_eval failed at seed=${seed}, continue" >&2
    fi
  done

  stop_servers
done

echo "All rounds finished."
echo "Failed cases logged to: ${FAILED_LOG}"
