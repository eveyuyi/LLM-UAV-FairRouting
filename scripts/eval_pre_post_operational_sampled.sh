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
OUTPUT_ROOT="${OUTPUT_ROOT:-data/eval_runs/pre_post_operational_sampled}"

# If TIME_SLOTS_STR is set, use it directly; otherwise sample stratified slots.
TIME_SLOTS_STR="${TIME_SLOTS_STR:-}"
SAMPLE_TOTAL_SLOTS="${SAMPLE_TOTAL_SLOTS:-9}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

SOLVER_BACKEND="${SOLVER_BACKEND:-nsga3_heuristic}" # nsga3 | nsga3_heuristic | cplex
WINDOW_MINUTES="${WINDOW_MINUTES:-5}"
TIME_LIMIT="${TIME_LIMIT:-180}"
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

URGENT_THRESHOLD="${URGENT_THRESHOLD:-2}"

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

mkdir -p "${OUTPUT_ROOT}/evals"

if [[ -z "${TIME_SLOTS_STR}" ]]; then
  echo "[sample] build stratified time-slot sample from dialogues" >&2
  TIME_SLOTS_STR="$(
    DIALOGUES="${DIALOGUES}" \
    SAMPLE_TOTAL_SLOTS="${SAMPLE_TOTAL_SLOTS}" \
    SAMPLE_SEED="${SAMPLE_SEED}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    PYTHONPATH=src "${_py[@]}" - <<'PY'
import json
import os
import random
from collections import Counter
from pathlib import Path

dialogues = Path(os.environ["DIALOGUES"])
sample_total = max(1, int(os.environ.get("SAMPLE_TOTAL_SLOTS", "9")))
seed = int(os.environ.get("SAMPLE_SEED", "42"))
output_root = Path(os.environ["OUTPUT_ROOT"])

counts = Counter()
with dialogues.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        slot = (obj.get("metadata") or {}).get("time_slot")
        if slot is None:
            continue
        counts[int(slot)] += 1

slots_sorted = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
all_slots = [s for s, _ in slots_sorted]

if not all_slots:
    print("")
    raise SystemExit(0)

if sample_total >= len(all_slots):
    selected = sorted(all_slots)
    mode = "all_slots"
else:
    rng = random.Random(seed)
    n = len(slots_sorted)
    low = slots_sorted[: max(1, n // 3)]
    mid = slots_sorted[max(1, n // 3): max(2, 2 * n // 3)]
    high = slots_sorted[max(2, 2 * n // 3):]
    buckets = [low, mid, high]

    # Allocate quotas as evenly as possible across buckets.
    base = sample_total // 3
    rem = sample_total % 3
    quotas = [base, base, base]
    for i in range(rem):
        quotas[i] += 1

    selected_set = set()
    for bucket, q in zip(buckets, quotas):
        pool = [slot for slot, _ in bucket if slot not in selected_set]
        if not pool or q <= 0:
            continue
        q = min(q, len(pool))
        picks = rng.sample(pool, q)
        selected_set.update(picks)

    # Fill from remaining slots if any bucket was short.
    if len(selected_set) < sample_total:
        remaining = [slot for slot in all_slots if slot not in selected_set]
        need = min(sample_total - len(selected_set), len(remaining))
        if need > 0:
            selected_set.update(rng.sample(remaining, need))

    selected = sorted(selected_set)
    mode = "stratified_sample"

sampling_payload = {
    "mode": mode,
    "sample_total_slots": sample_total,
    "seed": seed,
    "selected_time_slots": selected,
    "slot_counts": {str(slot): int(cnt) for slot, cnt in slots_sorted},
}
sampling_path = output_root / "evals" / "slot_sampling.json"
sampling_path.parent.mkdir(parents=True, exist_ok=True)
with sampling_path.open("w", encoding="utf-8") as f:
    json.dump(sampling_payload, f, ensure_ascii=False, indent=2)

print(" ".join(str(s) for s in selected))
PY
  )"
fi

read -r -a TIME_SLOTS <<< "${TIME_SLOTS_STR}"
if [[ "${#TIME_SLOTS[@]}" -eq 0 ]]; then
  echo "TIME_SLOTS_STR is empty (explicit or sampled)." >&2
  exit 1
fi
echo "[sample] use time slots: ${TIME_SLOTS_STR}" >&2

run_workflow_once() {
  local tag="$1"
  local api_base="$2"
  local model="$3"
  local out_base="${OUTPUT_ROOT}/${tag}"
  mkdir -p "${out_base}"

  echo "[${tag}] sampled workflow with solver=${SOLVER_BACKEND}, model=${model}, api_base=${api_base}" >&2
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
    --solver-backend "${SOLVER_BACKEND}" \
    --nsga3-pop-size "${NSGA3_POP_SIZE}" \
    --nsga3-n-generations "${NSGA3_N_GENERATIONS}" \
    --nsga3-seed "${NSGA3_SEED}" \
    --time-slots "${TIME_SLOTS[@]}" \
    1>&2

  local run_dir
  run_dir="$(latest_run_dir "${out_base}")"
  if [[ -z "${run_dir}" ]]; then
    echo "[${tag}] Failed: no run_* dir under ${out_base}" >&2
    exit 1
  fi
  echo "[${tag}] run_dir=${run_dir}" >&2
  RUN_DIR_RESULT="${run_dir}"
}

RUN_DIR_RESULT=""
run_workflow_once "pre" "${PRE_API_BASE}" "${PRE_MODEL}"
PRE_RUN_DIR="${RUN_DIR_RESULT}"
run_workflow_once "post" "${POST_API_BASE}" "${POST_MODEL}"
POST_RUN_DIR="${RUN_DIR_RESULT}"

echo "[eval] priority alignment (pre)"
PYTHONPATH=src "${_py[@]}" evals/eval_priority_alignment.py \
  --weights "${PRE_RUN_DIR}/weight_configs" \
  --demands "${PRE_RUN_DIR}/extracted_demands.json" \
  --dialogues "${DIALOGUES}" \
  --ground-truth "${GROUND_TRUTH}" \
  --urgent-threshold "${URGENT_THRESHOLD}" \
  --output "${OUTPUT_ROOT}/evals/pre_alignment.json"

echo "[eval] priority alignment (post)"
PYTHONPATH=src "${_py[@]}" evals/eval_priority_alignment.py \
  --weights "${POST_RUN_DIR}/weight_configs" \
  --demands "${POST_RUN_DIR}/extracted_demands.json" \
  --dialogues "${DIALOGUES}" \
  --ground-truth "${GROUND_TRUTH}" \
  --urgent-threshold "${URGENT_THRESHOLD}" \
  --output "${OUTPUT_ROOT}/evals/post_alignment.json"

echo "[eval] operational impact (post vs pre)"
PYTHONPATH=src "${_py[@]}" evals/eval_priority_operational_impact.py \
  --run "pre=${PRE_RUN_DIR}" \
  --run "post=${POST_RUN_DIR}" \
  --dialogues "${DIALOGUES}" \
  --ground-truth "${GROUND_TRUTH}" \
  --urgent-threshold "${URGENT_THRESHOLD}" \
  --reference-method pre \
  --output "${OUTPUT_ROOT}/evals/post_vs_pre_operational_impact.json"

cat > "${OUTPUT_ROOT}/evals/eval_manifest.json" <<EOF
{
  "mode": "operational_impact_sampled",
  "pre_run_dir": "${PRE_RUN_DIR}",
  "post_run_dir": "${POST_RUN_DIR}",
  "pre_alignment": "${OUTPUT_ROOT}/evals/pre_alignment.json",
  "post_alignment": "${OUTPUT_ROOT}/evals/post_alignment.json",
  "post_vs_pre_operational_impact": "${OUTPUT_ROOT}/evals/post_vs_pre_operational_impact.json",
  "solver_backend": "${SOLVER_BACKEND}",
  "time_slots": "${TIME_SLOTS_STR}",
  "sample_total_slots": ${SAMPLE_TOTAL_SLOTS},
  "sample_seed": ${SAMPLE_SEED},
  "nsga3_pop_size": ${NSGA3_POP_SIZE},
  "nsga3_n_generations": ${NSGA3_N_GENERATIONS},
  "nsga3_seed": ${NSGA3_SEED}
}
EOF

echo ""
echo "Finished sampled operational-impact evaluation."
echo "  pre_run_dir  : ${PRE_RUN_DIR}"
echo "  post_run_dir : ${POST_RUN_DIR}"
echo "  eval outputs : ${OUTPUT_ROOT}/evals"
