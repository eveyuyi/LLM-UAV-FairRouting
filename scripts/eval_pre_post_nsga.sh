#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------- 只改这里 ----------
CONDA_ENV="${CONDA_ENV:-}"
API_KEY="${OPENAI_API_KEY:-}"

# 训练前/后模型（OpenAI-compatible API）
PRE_API_BASE="${PRE_API_BASE:-http://127.0.0.1:8000/v1}"
PRE_MODEL="${PRE_MODEL:-qwen3-pre}"
POST_API_BASE="${POST_API_BASE:-http://127.0.0.1:8001/v1}"
POST_MODEL="${POST_MODEL:-qwen3-post}"

# 数据与输出
DIALOGUES="${DIALOGUES:-data/seed/daily_demand_dialogues.jsonl}"
GROUND_TRUTH="${GROUND_TRUTH:-data/seed/daily_demand_events_manifest.jsonl}"
STATIONS="${STATIONS:-data/seed/drone_station_locations.csv}"
BUILDING_DATA="${BUILDING_DATA:-data/seed/building_information.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/eval_runs/pre_post_nsga}"

# 评估窗口：metadata.time_slot（5分钟槽位）
TIME_SLOTS_STR="${TIME_SLOTS_STR:-0 1 2 3 4 5 6 7 8 9}"

# 求解与运行参数
SOLVER_BACKEND="${SOLVER_BACKEND:-nsga3}" # nsga3 | nsga3_heuristic | cplex
WINDOW_MINUTES="${WINDOW_MINUTES:-5}"
TIME_LIMIT="${TIME_LIMIT:-180}"
MAX_SOLVER_STATIONS="${MAX_SOLVER_STATIONS:-1}"
MAX_DRONES_PER_STATION="${MAX_DRONES_PER_STATION:-3}"
MAX_PAYLOAD="${MAX_PAYLOAD:-60.0}"
MAX_RANGE="${MAX_RANGE:-200000.0}"
NOISE_WEIGHT="${NOISE_WEIGHT:-0.5}"
DRONE_ACTIVATION_COST="${DRONE_ACTIVATION_COST:-1000.0}"
DRONE_SPEED="${DRONE_SPEED:-60.0}"

# NSGA参数
NSGA3_POP_SIZE="${NSGA3_POP_SIZE:-8}"
NSGA3_N_GENERATIONS="${NSGA3_N_GENERATIONS:-5}"
NSGA3_SEED="${NSGA3_SEED:-42}"

URGENT_THRESHOLD="${URGENT_THRESHOLD:-2}"

if [[ -z "${API_KEY}" ]]; then
  echo "Missing OPENAI_API_KEY. Please export OPENAI_API_KEY first." >&2
  exit 1
fi

read -r -a TIME_SLOTS <<< "${TIME_SLOTS_STR}"
if [[ "${#TIME_SLOTS[@]}" -eq 0 ]]; then
  echo "TIME_SLOTS_STR is empty." >&2
  exit 1
fi

if [[ -n "${CONDA_ENV}" ]]; then
  _py=(conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python)
else
  _py=(env PYTHONNOUSERSITE=1 python)
fi

latest_run_dir() {
  local root="$1"
  local runs=()
  shopt -s nullglob
  runs=("${root}"/run_*)
  shopt -u nullglob
  if [[ "${#runs[@]}" -eq 0 ]]; then
    echo "" && return 0
  fi
  local latest="${runs[0]}"
  local path
  for path in "${runs[@]}"; do
    if [[ "${path}" -nt "${latest}" ]]; then
      latest="${path}"
    fi
  done
  echo "${latest}"
}

run_workflow_once() {
  local tag="$1"
  local api_base="$2"
  local model="$3"
  local out_base="${OUTPUT_ROOT}/${tag}"
  mkdir -p "${out_base}"

  echo "[${tag}] Run workflow with model=${model}, api_base=${api_base}, backend=${SOLVER_BACKEND}" >&2
  OPENAI_API_KEY="${API_KEY}" OPENAI_BASE_URL="${api_base}" \
  "${_py[@]}" -m llm4fairrouting.workflow.run_workflow \
    --output-dir "${out_base}" \
    --dialogues "${DIALOGUES}" \
    --stations "${STATIONS}" \
    --building-data "${BUILDING_DATA}" \
    --model "${model}" \
    --window "${WINDOW_MINUTES}" \
    --time-limit "${TIME_LIMIT}" \
    --max-solver-stations "${MAX_SOLVER_STATIONS}" \
    --max-drones-per-station "${MAX_DRONES_PER_STATION}" \
    --max-payload "${MAX_PAYLOAD}" \
    --max-range "${MAX_RANGE}" \
    --noise-weight "${NOISE_WEIGHT}" \
    --drone-activation-cost "${DRONE_ACTIVATION_COST}" \
    --drone-speed "${DRONE_SPEED}" \
    --solver-backend "${SOLVER_BACKEND}" \
    --nsga3-pop-size "${NSGA3_POP_SIZE}" \
    --nsga3-n-generations "${NSGA3_N_GENERATIONS}" \
    --nsga3-seed "${NSGA3_SEED}" \
    --time-slots "${TIME_SLOTS[@]}"

  local run_dir
  run_dir="$(latest_run_dir "${out_base}")"
  if [[ -z "${run_dir}" ]]; then
    echo "[${tag}] Failed: no run_* dir under ${out_base}" >&2
    exit 1
  fi
  echo "[${tag}] run_dir=${run_dir}" >&2
  echo "${run_dir}"
}

PRE_RUN_DIR="$(run_workflow_once "pre" "${PRE_API_BASE}" "${PRE_MODEL}")"
POST_RUN_DIR="$(run_workflow_once "post" "${POST_API_BASE}" "${POST_MODEL}")"

mkdir -p "${OUTPUT_ROOT}/evals"

echo "[eval] priority alignment (pre)"
"${_py[@]}" evals/eval_priority_alignment.py \
  --weights "${PRE_RUN_DIR}/weight_configs" \
  --demands "${PRE_RUN_DIR}/extracted_demands.json" \
  --dialogues "${DIALOGUES}" \
  --ground-truth "${GROUND_TRUTH}" \
  --urgent-threshold "${URGENT_THRESHOLD}" \
  --output "${OUTPUT_ROOT}/evals/pre_alignment.json"

echo "[eval] priority alignment (post)"
"${_py[@]}" evals/eval_priority_alignment.py \
  --weights "${POST_RUN_DIR}/weight_configs" \
  --demands "${POST_RUN_DIR}/extracted_demands.json" \
  --dialogues "${DIALOGUES}" \
  --ground-truth "${GROUND_TRUTH}" \
  --urgent-threshold "${URGENT_THRESHOLD}" \
  --output "${OUTPUT_ROOT}/evals/post_alignment.json"

echo "[eval] operational impact (post vs pre)"
"${_py[@]}" evals/eval_priority_operational_impact.py \
  --run "pre=${PRE_RUN_DIR}" \
  --run "post=${POST_RUN_DIR}" \
  --dialogues "${DIALOGUES}" \
  --ground-truth "${GROUND_TRUTH}" \
  --urgent-threshold "${URGENT_THRESHOLD}" \
  --reference-method pre \
  --output "${OUTPUT_ROOT}/evals/post_vs_pre_operational_impact.json"

cat > "${OUTPUT_ROOT}/evals/eval_manifest.json" <<EOF
{
  "pre_run_dir": "${PRE_RUN_DIR}",
  "post_run_dir": "${POST_RUN_DIR}",
  "pre_alignment": "${OUTPUT_ROOT}/evals/pre_alignment.json",
  "post_alignment": "${OUTPUT_ROOT}/evals/post_alignment.json",
  "post_vs_pre_operational_impact": "${OUTPUT_ROOT}/evals/post_vs_pre_operational_impact.json",
  "solver_backend": "${SOLVER_BACKEND}",
  "time_slots": "${TIME_SLOTS_STR}",
  "nsga3_pop_size": ${NSGA3_POP_SIZE},
  "nsga3_n_generations": ${NSGA3_N_GENERATIONS},
  "nsga3_seed": ${NSGA3_SEED}
}
EOF

echo ""
echo "Finished pre/post evaluation."
echo "  pre_run_dir  : ${PRE_RUN_DIR}"
echo "  post_run_dir : ${POST_RUN_DIR}"
echo "  eval outputs : ${OUTPUT_ROOT}/evals"
