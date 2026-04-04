#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------- Config ----------
CONDA_ENV="${CONDA_ENV:-}"
API_KEY="${OPENAI_API_KEY:-}"

PRE_API_BASE="${PRE_API_BASE:-http://127.0.0.1:8000/v1}"
PRE_MODEL="${PRE_MODEL:-qwen3-pre}"
POST_API_BASE="${POST_API_BASE:-http://127.0.0.1:8001/v1}"
POST_MODEL="${POST_MODEL:-qwen3-post}"

DIALOGUES="${DIALOGUES:-data/seed/daily_demand_dialogues.jsonl}"
GROUND_TRUTH="${GROUND_TRUTH:-data/seed/daily_demand_events_manifest.jsonl}"
STATIONS="${STATIONS:-data/seed/drone_station_locations.csv}"
BUILDING_DATA="${BUILDING_DATA:-data/seed/building_information.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/eval_runs/pre_post_tiered}"

URGENT_THRESHOLD="${URGENT_THRESHOLD:-2}"

# Stage 1: ranking-only
STAGE1_TIME_SLOTS_STR="${STAGE1_TIME_SLOTS_STR:-0 1 2 3 4 5 6 7 8 9}"

# Stage 2: sampled operational impact
STAGE2_TIME_SLOTS_STR="${STAGE2_TIME_SLOTS_STR:-}"
STAGE2_SAMPLE_N_SLOTS="${STAGE2_SAMPLE_N_SLOTS:-3}"
STAGE2_SAMPLE_SEED="${STAGE2_SAMPLE_SEED:-42}"
SOLVER_BACKEND_STAGE2="${SOLVER_BACKEND_STAGE2:-nsga3_heuristic}" # nsga3_heuristic | nsga3 | cplex
WINDOW_MINUTES="${WINDOW_MINUTES:-5}"
TIME_LIMIT="${TIME_LIMIT:-120}"
MAX_SOLVER_STATIONS="${MAX_SOLVER_STATIONS:-1}"
MAX_DRONES_PER_STATION="${MAX_DRONES_PER_STATION:-3}"
MAX_PAYLOAD="${MAX_PAYLOAD:-60.0}"
MAX_RANGE="${MAX_RANGE:-200000.0}"
NOISE_WEIGHT="${NOISE_WEIGHT:-0.5}"
DRONE_ACTIVATION_COST="${DRONE_ACTIVATION_COST:-1000.0}"
DRONE_SPEED="${DRONE_SPEED:-60.0}"
NSGA3_POP_SIZE="${NSGA3_POP_SIZE:-4}"
NSGA3_N_GENERATIONS="${NSGA3_N_GENERATIONS:-2}"
NSGA3_SEED="${NSGA3_SEED:-42}"

if [[ -z "${API_KEY}" ]]; then
  echo "Missing OPENAI_API_KEY. Please export OPENAI_API_KEY first." >&2
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

run_stage1_rank_only_once() {
  local tag="$1"
  local api_base="$2"
  local model="$3"
  local out_base="${OUTPUT_ROOT}/stage1_rank/${tag}"
  mkdir -p "${out_base}"

  read -r -a stage1_slots <<< "${STAGE1_TIME_SLOTS_STR}"
  if [[ "${#stage1_slots[@]}" -eq 0 ]]; then
    echo "STAGE1_TIME_SLOTS_STR is empty." >&2
    exit 1
  fi

  echo "[stage1:${tag}] rank-only workflow (skip solver), model=${model}, api_base=${api_base}" >&2
  OPENAI_API_KEY="${API_KEY}" OPENAI_BASE_URL="${api_base}" PYTHONPATH=src \
  "${_py[@]}" -m llm4fairrouting.workflow.run_workflow \
    --output-dir "${out_base}" \
    --dialogues "${DIALOGUES}" \
    --stations "${STATIONS}" \
    --building-data "${BUILDING_DATA}" \
    --model "${model}" \
    --time-slots "${stage1_slots[@]}" \
    --skip-solver \
    1>&2

  local run_dir
  run_dir="$(latest_run_dir "${out_base}")"
  if [[ -z "${run_dir}" ]]; then
    echo "[stage1:${tag}] Failed: no run_* dir under ${out_base}" >&2
    exit 1
  fi
  echo "[stage1:${tag}] run_dir=${run_dir}" >&2
  RUN_DIR_RESULT="${run_dir}"
}

sample_stage2_slots_if_needed() {
  if [[ -n "${STAGE2_TIME_SLOTS_STR}" ]]; then
    return 0
  fi

  STAGE2_TIME_SLOTS_STR="$(
    PYTHONPATH=src "${_py[@]}" - <<'PY'
import json
import os
import random
from pathlib import Path

dialogues = Path(os.environ["DIALOGUES"])
n_slots = int(os.environ["STAGE2_SAMPLE_N_SLOTS"])
seed = int(os.environ["STAGE2_SAMPLE_SEED"])

counts = {}
with dialogues.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        slot = (obj.get("metadata") or {}).get("time_slot")
        if slot is None:
            continue
        slot = int(slot)
        counts[slot] = counts.get(slot, 0) + 1

slots = sorted(counts)
if not slots:
    print("")
    raise SystemExit(0)

if n_slots <= 0 or n_slots >= len(slots):
    selected = slots
else:
    rng = random.Random(seed)
    selected = sorted(rng.sample(slots, n_slots))

print(" ".join(str(s) for s in selected))
PY
  )"

  if [[ -z "${STAGE2_TIME_SLOTS_STR}" ]]; then
    echo "Failed to sample Stage2 time slots from ${DIALOGUES}" >&2
    exit 1
  fi
}

run_stage2_solver_once() {
  local tag="$1"
  local api_base="$2"
  local model="$3"
  local out_base="${OUTPUT_ROOT}/stage2_sampled_solver/${tag}"
  mkdir -p "${out_base}"

  read -r -a stage2_slots <<< "${STAGE2_TIME_SLOTS_STR}"
  if [[ "${#stage2_slots[@]}" -eq 0 ]]; then
    echo "Stage2 time slots are empty." >&2
    exit 1
  fi

  echo "[stage2:${tag}] sampled solver workflow, backend=${SOLVER_BACKEND_STAGE2}, model=${model}, slots=${STAGE2_TIME_SLOTS_STR}" >&2
  OPENAI_API_KEY="${API_KEY}" OPENAI_BASE_URL="${api_base}" PYTHONPATH=src \
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
    --solver-backend "${SOLVER_BACKEND_STAGE2}" \
    --nsga3-pop-size "${NSGA3_POP_SIZE}" \
    --nsga3-n-generations "${NSGA3_N_GENERATIONS}" \
    --nsga3-seed "${NSGA3_SEED}" \
    --time-slots "${stage2_slots[@]}" \
    1>&2

  local run_dir
  run_dir="$(latest_run_dir "${out_base}")"
  if [[ -z "${run_dir}" ]]; then
    echo "[stage2:${tag}] Failed: no run_* dir under ${out_base}" >&2
    exit 1
  fi
  echo "[stage2:${tag}] run_dir=${run_dir}" >&2
  RUN_DIR_RESULT="${run_dir}"
}

mkdir -p "${OUTPUT_ROOT}/evals"

# ---------------- Stage 1 ----------------
RUN_DIR_RESULT=""
run_stage1_rank_only_once "pre" "${PRE_API_BASE}" "${PRE_MODEL}"
STAGE1_PRE_RUN_DIR="${RUN_DIR_RESULT}"
run_stage1_rank_only_once "post" "${POST_API_BASE}" "${POST_MODEL}"
STAGE1_POST_RUN_DIR="${RUN_DIR_RESULT}"

echo "[eval][stage1] priority alignment (pre)"
PYTHONPATH=src "${_py[@]}" evals/eval_priority_alignment.py \
  --weights "${STAGE1_PRE_RUN_DIR}/weight_configs" \
  --demands "${STAGE1_PRE_RUN_DIR}/extracted_demands.json" \
  --dialogues "${DIALOGUES}" \
  --ground-truth "${GROUND_TRUTH}" \
  --urgent-threshold "${URGENT_THRESHOLD}" \
  --output "${OUTPUT_ROOT}/evals/stage1_pre_alignment.json"

echo "[eval][stage1] priority alignment (post)"
PYTHONPATH=src "${_py[@]}" evals/eval_priority_alignment.py \
  --weights "${STAGE1_POST_RUN_DIR}/weight_configs" \
  --demands "${STAGE1_POST_RUN_DIR}/extracted_demands.json" \
  --dialogues "${DIALOGUES}" \
  --ground-truth "${GROUND_TRUTH}" \
  --urgent-threshold "${URGENT_THRESHOLD}" \
  --output "${OUTPUT_ROOT}/evals/stage1_post_alignment.json"

echo "[eval][stage1] alignment delta (post - pre)"
PYTHONPATH=src "${_py[@]}" - <<'PY'
import json
import os
from pathlib import Path

output_root = Path(os.environ["OUTPUT_ROOT"])
pre_path = output_root / "evals" / "stage1_pre_alignment.json"
post_path = output_root / "evals" / "stage1_post_alignment.json"
delta_path = output_root / "evals" / "stage1_post_vs_pre_alignment_delta.json"

with pre_path.open("r", encoding="utf-8") as f:
    pre = json.load(f)
with post_path.open("r", encoding="utf-8") as f:
    post = json.load(f)

def diff(metric):
    a = pre.get(metric)
    b = post.get(metric)
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 6)

pre_topk = (pre.get("top_k_hit_rate") or {}).get("hit_rate")
post_topk = (post.get("top_k_hit_rate") or {}).get("hit_rate")
topk_delta = None
if pre_topk is not None and post_topk is not None:
    topk_delta = round(float(post_topk) - float(pre_topk), 6)

payload = {
    "pre_alignment": str(pre_path),
    "post_alignment": str(post_path),
    "delta_metrics_post_minus_pre": {
        "accuracy": diff("accuracy"),
        "macro_f1": diff("macro_f1"),
        "weighted_f1": diff("weighted_f1"),
        "spearman": diff("spearman"),
        "kendall_tau": diff("kendall_tau"),
        "top_k_hit_rate": topk_delta,
    },
    "n_aligned_demands": {
        "pre": pre.get("n_aligned_demands"),
        "post": post.get("n_aligned_demands"),
    },
}

with delta_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(f"Stage1 alignment delta saved to {delta_path}")
PY

# ---------------- Stage 2 ----------------
sample_stage2_slots_if_needed
read -r -a STAGE2_SLOTS_ARRAY <<< "${STAGE2_TIME_SLOTS_STR}"
if [[ "${#STAGE2_SLOTS_ARRAY[@]}" -eq 0 ]]; then
  echo "Stage2 slots are empty after sampling." >&2
  exit 1
fi
echo "[stage2] using sampled time slots: ${STAGE2_TIME_SLOTS_STR}" >&2

RUN_DIR_RESULT=""
run_stage2_solver_once "pre" "${PRE_API_BASE}" "${PRE_MODEL}"
STAGE2_PRE_RUN_DIR="${RUN_DIR_RESULT}"
run_stage2_solver_once "post" "${POST_API_BASE}" "${POST_MODEL}"
STAGE2_POST_RUN_DIR="${RUN_DIR_RESULT}"

echo "[eval][stage2] operational impact (post vs pre)"
PYTHONPATH=src "${_py[@]}" evals/eval_priority_operational_impact.py \
  --run "pre=${STAGE2_PRE_RUN_DIR}" \
  --run "post=${STAGE2_POST_RUN_DIR}" \
  --dialogues "${DIALOGUES}" \
  --ground-truth "${GROUND_TRUTH}" \
  --urgent-threshold "${URGENT_THRESHOLD}" \
  --reference-method pre \
  --output "${OUTPUT_ROOT}/evals/stage2_post_vs_pre_operational_impact.json"

cat > "${OUTPUT_ROOT}/evals/eval_manifest.json" <<EOF
{
  "mode": "tiered_pre_post_eval",
  "stage1": {
    "time_slots": "${STAGE1_TIME_SLOTS_STR}",
    "pre_run_dir": "${STAGE1_PRE_RUN_DIR}",
    "post_run_dir": "${STAGE1_POST_RUN_DIR}",
    "pre_alignment": "${OUTPUT_ROOT}/evals/stage1_pre_alignment.json",
    "post_alignment": "${OUTPUT_ROOT}/evals/stage1_post_alignment.json",
    "post_vs_pre_alignment_delta": "${OUTPUT_ROOT}/evals/stage1_post_vs_pre_alignment_delta.json"
  },
  "stage2": {
    "sampled_time_slots": "${STAGE2_TIME_SLOTS_STR}",
    "solver_backend": "${SOLVER_BACKEND_STAGE2}",
    "nsga3_pop_size": ${NSGA3_POP_SIZE},
    "nsga3_n_generations": ${NSGA3_N_GENERATIONS},
    "nsga3_seed": ${NSGA3_SEED},
    "pre_run_dir": "${STAGE2_PRE_RUN_DIR}",
    "post_run_dir": "${STAGE2_POST_RUN_DIR}",
    "post_vs_pre_operational_impact": "${OUTPUT_ROOT}/evals/stage2_post_vs_pre_operational_impact.json"
  }
}
EOF

echo ""
echo "Finished tiered pre/post evaluation."
echo "  Stage1 alignment outputs : ${OUTPUT_ROOT}/evals/stage1_*"
echo "  Stage2 impact output     : ${OUTPUT_ROOT}/evals/stage2_post_vs_pre_operational_impact.json"
echo "  Manifest                 : ${OUTPUT_ROOT}/evals/eval_manifest.json"
