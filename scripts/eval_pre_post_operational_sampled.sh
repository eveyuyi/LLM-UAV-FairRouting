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
FIXED_EXTRACTED_DEMANDS="${FIXED_EXTRACTED_DEMANDS:-}"
OPERATIONAL_MODE="${OPERATIONAL_MODE:-auto}" # auto | llm3_only | end_to_end
TRUTH_SOURCE="${TRUTH_SOURCE:-auto}"
WINDOW_INDICES_STR="${WINDOW_INDICES_STR:-}"
TIME_WINDOWS_STR="${TIME_WINDOWS_STR:-}"

# If TIME_SLOTS_STR is set, use it directly; otherwise sample stratified slots.
TIME_SLOTS_STR="${TIME_SLOTS_STR:-}"
SAMPLE_TOTAL_SLOTS="${SAMPLE_TOTAL_SLOTS:-9}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

SOLVER_BACKEND="${SOLVER_BACKEND:-cplex}" # cplex | nsga3 | nsga3_heuristic
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

if [[ ! -f "${GROUND_TRUTH}" ]]; then
  echo "Missing ground-truth file: ${GROUND_TRUTH}" >&2
  echo "Operational impact evaluation requires event priorities from ground-truth manifest." >&2
  echo "Please set GROUND_TRUTH=/path/to/events_manifest.jsonl, then rerun." >&2
  echo "Example to generate default seed manifest:" >&2
  echo "  llm4fairrouting-demand-events --manifest-output data/seed/daily_demand_events_manifest.jsonl" >&2
  exit 1
fi

if [[ "${SOLVER_BACKEND}" == "nsga3" || "${SOLVER_BACKEND}" == "nsga3_heuristic" ]]; then
  echo "[warn] SOLVER_BACKEND=${SOLVER_BACKEND} adds an outer search loop." >&2
  echo "[warn] For pre/post model comparison, CPLEX is usually the cleaner and more stable choice." >&2
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

if [[ -z "${FIXED_EXTRACTED_DEMANDS}" ]]; then
  _candidate_fixed="$(dirname "${DIALOGUES}")/llm3_sft_pipeline.jsonl"
  if [[ -f "${_candidate_fixed}" ]]; then
    FIXED_EXTRACTED_DEMANDS="${_candidate_fixed}"
  fi
fi

if [[ "${OPERATIONAL_MODE}" == "auto" ]]; then
  if [[ -n "${FIXED_EXTRACTED_DEMANDS}" ]]; then
    OPERATIONAL_MODE="llm3_only"
  else
    OPERATIONAL_MODE="end_to_end"
  fi
fi

ALIGNMENT_DEMANDS_PATH=""
TRUTH_DEMANDS_PATH=""
RUN_WORKFLOW_EXTRA_ARGS=()
TIME_SLOT_ARGS=()
TRUTH_ARGS=()
SELECTION_MANIFEST_PATH=""
SLOT_SAMPLING_PATH="${OUTPUT_ROOT}/evals/slot_sampling.json"

mkdir -p "${OUTPUT_ROOT}/evals"

select_fixed_windows() {
  if [[ -z "${FIXED_EXTRACTED_DEMANDS}" ]]; then
    echo "OPERATIONAL_MODE=llm3_only requires FIXED_EXTRACTED_DEMANDS or a sibling llm3_sft_pipeline.jsonl file." >&2
    exit 1
  fi

  local fixed_dir="${OUTPUT_ROOT}/fixed_inputs"
  mkdir -p "${fixed_dir}"
  local selected_demands="${fixed_dir}/selected_demands.json"
  local selection_manifest="${fixed_dir}/selection_manifest.json"

  FIXED_EXTRACTED_DEMANDS="${FIXED_EXTRACTED_DEMANDS}" \
  SELECTED_DEMANDS="${selected_demands}" \
  SELECTION_MANIFEST="${selection_manifest}" \
  TIME_SLOTS_STR="${TIME_SLOTS_STR}" \
  TIME_WINDOWS_STR="${TIME_WINDOWS_STR}" \
  WINDOW_INDICES_STR="${WINDOW_INDICES_STR}" \
  PYTHONPATH=src "${_py[@]}" - <<'PY'
import json
import os
import re
from pathlib import Path

WINDOW_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T(\d{2}):(\d{2})-\d{2}:\d{2}$")


def load_windows(path: Path):
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON/JSONL list of windows: {path}")
    return payload


def window_start_slot(window):
    label = str(window.get("time_window", "")).split("::", 1)[0].strip()
    match = WINDOW_RE.match(label)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        return (hour * 60 + minute) // 5
    for demand in window.get("demands", []) or []:
        ts = str(demand.get("request_timestamp", "")).strip()
        if not ts:
            continue
        try:
            hour = int(ts[11:13])
            minute = int(ts[14:16])
        except (TypeError, ValueError, IndexError):
            continue
        return (hour * 60 + minute) // 5
    return None


source = Path(os.environ["FIXED_EXTRACTED_DEMANDS"])
selected_path = Path(os.environ["SELECTED_DEMANDS"])
manifest_path = Path(os.environ["SELECTION_MANIFEST"])
windows = load_windows(source)

time_windows = [item for item in os.environ.get("TIME_WINDOWS_STR", "").split() if item]
window_indices = [int(item) for item in os.environ.get("WINDOW_INDICES_STR", "").split() if item]
time_slots = [int(item) for item in os.environ.get("TIME_SLOTS_STR", "").split() if item]

selection_mode = "all_windows"
selected = list(windows)
if time_windows:
    allowed = set(time_windows)
    selected = [window for window in windows if str(window.get("time_window", "")) in allowed]
    selection_mode = "time_windows"
elif window_indices:
    selected = [
        windows[index]
        for index in window_indices
        if 0 <= index < len(windows)
    ]
    selection_mode = "window_indices"
elif time_slots:
    allowed = set(time_slots)
    selected = [window for window in windows if window_start_slot(window) in allowed]
    selection_mode = "time_slots"

selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
manifest = {
    "source_path": str(source),
    "selection_mode": selection_mode,
    "requested_time_slots": time_slots,
    "requested_time_windows": time_windows,
    "requested_window_indices": window_indices,
    "n_windows_source": len(windows),
    "n_windows_selected": len(selected),
    "selected_time_windows": [str(window.get("time_window", "")) for window in selected],
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY

  ALIGNMENT_DEMANDS_PATH="${selected_demands}"
  TRUTH_DEMANDS_PATH="${selected_demands}"
  RUN_WORKFLOW_EXTRA_ARGS=(--extracted-demands "${selected_demands}")
  SELECTION_MANIFEST_PATH="${selection_manifest}"
}

if [[ "${OPERATIONAL_MODE}" == "llm3_only" && -z "${TIME_WINDOWS_STR}" && -z "${WINDOW_INDICES_STR}" && -z "${TIME_SLOTS_STR}" ]]; then
  echo "[sample] build stratified fixed-window sample from ${FIXED_EXTRACTED_DEMANDS}" >&2
  TIME_WINDOWS_STR="$(
    FIXED_EXTRACTED_DEMANDS="${FIXED_EXTRACTED_DEMANDS}" \
    SAMPLE_TOTAL_SLOTS="${SAMPLE_TOTAL_SLOTS}" \
    SAMPLE_SEED="${SAMPLE_SEED}" \
    URGENT_THRESHOLD="${URGENT_THRESHOLD}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    PYTHONPATH=src "${_py[@]}" - <<'PY'
import json
import os
import random
from pathlib import Path


def load_windows(path: Path):
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON/JSONL list of windows: {path}")
    return payload


def demand_priority(demand):
    labels = demand.get("labels", {}) or {}
    value = labels.get("extraction_observable_priority", demand.get("extraction_observable_priority", 4))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 4


def is_vulnerable(demand):
    vuln = ((demand.get("priority_evaluation_signals") or {}).get("population_vulnerability") or {})
    return any(bool(vuln.get(key, False)) for key in ("children_involved", "elderly_involved", "vulnerable_community"))


source = Path(os.environ["FIXED_EXTRACTED_DEMANDS"])
sample_total = max(1, int(os.environ.get("SAMPLE_TOTAL_SLOTS", "9")))
seed = int(os.environ.get("SAMPLE_SEED", "42"))
urgent_threshold = int(os.environ.get("URGENT_THRESHOLD", "2"))
output_root = Path(os.environ["OUTPUT_ROOT"])
rng = random.Random(seed)
windows = load_windows(source)

summaries = []
for index, window in enumerate(windows):
    demands = list(window.get("demands", []) or [])
    priorities = [demand_priority(demand) for demand in demands]
    labels = []
    if priorities and min(priorities) == 1:
        labels.append("priority_1")
    if priorities and min(priorities) <= urgent_threshold:
        labels.append("urgent")
    if any(is_vulnerable(demand) for demand in demands):
        labels.append("vulnerable")
    if len(priorities) >= 2:
        ordered = sorted(priorities)
        if ordered[1] - ordered[0] <= 1:
            labels.append("near_tie")
    if len(demands) >= 5:
        labels.append("dense")
    if not labels:
        labels.append("routine")
    summaries.append({
        "index": index,
        "time_window": str(window.get("time_window", "")),
        "n_demands": len(demands),
        "labels": labels,
        "min_priority": min(priorities) if priorities else None,
    })

if sample_total >= len(summaries):
    selected = summaries
    mode = "all_windows"
else:
    selected = []
    selected_idx = set()
    for label in ("priority_1", "urgent", "vulnerable", "near_tie", "dense", "routine"):
        candidates = [item for item in summaries if label in item["labels"] and item["index"] not in selected_idx]
        if not candidates or len(selected) >= sample_total:
            continue
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: (-len(item["labels"]), -item["n_demands"], item["time_window"]))
        choice = candidates[0]
        selected.append(choice)
        selected_idx.add(choice["index"])

    if len(selected) < sample_total:
        remaining = [item for item in summaries if item["index"] not in selected_idx]
        rng.shuffle(remaining)
        remaining.sort(key=lambda item: (-len(item["labels"]), -item["n_demands"], item["time_window"]))
        selected.extend(remaining[: sample_total - len(selected)])
    selected.sort(key=lambda item: item["index"])
    mode = "fixed_window_stratified_sample"

selected_labels = [item["time_window"] for item in selected]
sampling_payload = {
    "mode": mode,
    "sample_total_slots": sample_total,
    "seed": seed,
    "fixed_extracted_demands": str(source),
    "selected_time_windows": selected_labels,
    "window_summaries": summaries,
}
sampling_path = output_root / "evals" / "slot_sampling.json"
sampling_path.parent.mkdir(parents=True, exist_ok=True)
sampling_path.write_text(json.dumps(sampling_payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(" ".join(selected_labels))
PY
  )"
elif [[ "${OPERATIONAL_MODE}" == "end_to_end" && -z "${TIME_SLOTS_STR}" ]]; then
  echo "[sample] build stratified time-slot sample from dialogues + ground truth" >&2
  TIME_SLOTS_STR="$(
    DIALOGUES="${DIALOGUES}" \
    GROUND_TRUTH="${GROUND_TRUTH}" \
    SAMPLE_TOTAL_SLOTS="${SAMPLE_TOTAL_SLOTS}" \
    SAMPLE_SEED="${SAMPLE_SEED}" \
    URGENT_THRESHOLD="${URGENT_THRESHOLD}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    PYTHONPATH=src "${_py[@]}" - <<'PY'
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

dialogues = Path(os.environ["DIALOGUES"])
ground_truth = Path(os.environ["GROUND_TRUTH"])
sample_total = max(1, int(os.environ.get("SAMPLE_TOTAL_SLOTS", "9")))
seed = int(os.environ.get("SAMPLE_SEED", "42"))
urgent_threshold = int(os.environ.get("URGENT_THRESHOLD", "2"))
output_root = Path(os.environ["OUTPUT_ROOT"])
rng = random.Random(seed)

dialogue_counts = Counter()
with dialogues.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        slot = (obj.get("metadata") or {}).get("time_slot")
        if slot is not None:
            dialogue_counts[int(slot)] += 1

priority_by_slot = defaultdict(list)
with ground_truth.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        slot = obj.get("time_slot")
        priority = obj.get("latent_priority")
        if slot is None or priority is None:
            continue
        priority_by_slot[int(slot)].append(int(priority))

slots = sorted(set(dialogue_counts) | set(priority_by_slot))
if not slots:
    print("")
    raise SystemExit(0)

slot_meta = []
for slot in slots:
    priorities = sorted(priority_by_slot.get(slot, []))
    labels = []
    if priorities and priorities[0] == 1:
        labels.append("priority_1")
    if priorities and priorities[0] <= urgent_threshold:
        labels.append("urgent")
    if len(set(priorities)) >= 2:
        labels.append("mixed")
    if dialogue_counts.get(slot, 0) >= 5:
        labels.append("dense")
    if not labels:
        labels.append("routine")
    slot_meta.append({
        "slot": slot,
        "dialogue_count": dialogue_counts.get(slot, 0),
        "min_priority": priorities[0] if priorities else None,
        "labels": labels,
    })

if sample_total >= len(slot_meta):
    selected = slot_meta
    mode = "all_slots"
else:
    selected = []
    selected_slots = set()
    for label in ("priority_1", "urgent", "mixed", "dense", "routine"):
        candidates = [item for item in slot_meta if label in item["labels"] and item["slot"] not in selected_slots]
        if not candidates or len(selected) >= sample_total:
            continue
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: (-len(item["labels"]), -item["dialogue_count"], item["slot"]))
        choice = candidates[0]
        selected.append(choice)
        selected_slots.add(choice["slot"])

    if len(selected) < sample_total:
        remaining = [item for item in slot_meta if item["slot"] not in selected_slots]
        rng.shuffle(remaining)
        remaining.sort(key=lambda item: (-len(item["labels"]), -item["dialogue_count"], item["slot"]))
        selected.extend(remaining[: sample_total - len(selected)])
    selected.sort(key=lambda item: item["slot"])
    mode = "stratified_priority_slot_sample"

selected_slots = [item["slot"] for item in selected]
sampling_payload = {
    "mode": mode,
    "sample_total_slots": sample_total,
    "seed": seed,
    "selected_time_slots": selected_slots,
    "slot_meta": slot_meta,
}
sampling_path = output_root / "evals" / "slot_sampling.json"
sampling_path.parent.mkdir(parents=True, exist_ok=True)
sampling_path.write_text(json.dumps(sampling_payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(" ".join(str(slot) for slot in selected_slots))
PY
  )"
fi

if [[ "${OPERATIONAL_MODE}" == "llm3_only" ]]; then
  select_fixed_windows
  if [[ "${TRUTH_SOURCE}" == "auto" ]]; then
    TRUTH_SOURCE="fixed_demands"
  fi
else
  read -r -a TIME_SLOTS <<< "${TIME_SLOTS_STR}"
  if [[ "${#TIME_SLOTS[@]}" -eq 0 ]]; then
    echo "TIME_SLOTS_STR is empty (explicit or sampled)." >&2
    exit 1
  fi
  echo "[sample] use time slots: ${TIME_SLOTS_STR}" >&2
  TIME_SLOT_ARGS=(--time-slots "${TIME_SLOTS[@]}")
  if [[ "${TRUTH_SOURCE}" == "auto" ]]; then
    TRUTH_SOURCE="run_extracted"
  fi
fi

if [[ "${TRUTH_SOURCE}" == "fixed_demands" ]]; then
  if [[ -z "${TRUTH_DEMANDS_PATH}" ]]; then
    echo "TRUTH_SOURCE=fixed_demands requires a resolved TRUTH_DEMANDS_PATH." >&2
    exit 1
  fi
  TRUTH_ARGS=(--truth-demands "${TRUTH_DEMANDS_PATH}")
fi

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
    "${RUN_WORKFLOW_EXTRA_ARGS[@]}" \
    "${TIME_SLOT_ARGS[@]}" \
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
  --demands "${ALIGNMENT_DEMANDS_PATH:-${PRE_RUN_DIR}/extracted_demands.json}" \
  --dialogues "${DIALOGUES}" \
  --ground-truth "${GROUND_TRUTH}" \
  --urgent-threshold "${URGENT_THRESHOLD}" \
  --truth-source "${TRUTH_SOURCE}" \
  "${TRUTH_ARGS[@]}" \
  --output "${OUTPUT_ROOT}/evals/pre_alignment.json"

echo "[eval] priority alignment (post)"
PYTHONPATH=src "${_py[@]}" evals/eval_priority_alignment.py \
  --weights "${POST_RUN_DIR}/weight_configs" \
  --demands "${ALIGNMENT_DEMANDS_PATH:-${POST_RUN_DIR}/extracted_demands.json}" \
  --dialogues "${DIALOGUES}" \
  --ground-truth "${GROUND_TRUTH}" \
  --urgent-threshold "${URGENT_THRESHOLD}" \
  --truth-source "${TRUTH_SOURCE}" \
  "${TRUTH_ARGS[@]}" \
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
  "operational_mode": "${OPERATIONAL_MODE}",
  "truth_source": "${TRUTH_SOURCE}",
  "fixed_extracted_demands": "${FIXED_EXTRACTED_DEMANDS}",
  "selected_alignment_demands": "${ALIGNMENT_DEMANDS_PATH}",
  "selection_manifest": "${SELECTION_MANIFEST_PATH}",
  "slot_sampling": "${SLOT_SAMPLING_PATH}",
  "pre_run_dir": "${PRE_RUN_DIR}",
  "post_run_dir": "${POST_RUN_DIR}",
  "pre_alignment": "${OUTPUT_ROOT}/evals/pre_alignment.json",
  "post_alignment": "${OUTPUT_ROOT}/evals/post_alignment.json",
  "post_vs_pre_operational_impact": "${OUTPUT_ROOT}/evals/post_vs_pre_operational_impact.json",
  "summary_json": "${OUTPUT_ROOT}/evals/summary.json",
  "summary_md": "${OUTPUT_ROOT}/evals/summary.md",
  "solver_backend": "${SOLVER_BACKEND}",
  "time_slots": "${TIME_SLOTS_STR}",
  "sample_total_slots": ${SAMPLE_TOTAL_SLOTS},
  "sample_seed": ${SAMPLE_SEED},
  "nsga3_pop_size": ${NSGA3_POP_SIZE},
  "nsga3_n_generations": ${NSGA3_N_GENERATIONS},
  "nsga3_seed": ${NSGA3_SEED}
}
EOF

PYTHONPATH=src "${_py[@]}" evals/build_pre_post_eval_summary.py \
  --manifest "${OUTPUT_ROOT}/evals/eval_manifest.json" \
  --output-json "${OUTPUT_ROOT}/evals/summary.json" \
  --output-md "${OUTPUT_ROOT}/evals/summary.md"

echo ""
echo "Finished sampled operational-impact evaluation."
echo "  pre_run_dir  : ${PRE_RUN_DIR}"
echo "  post_run_dir : ${POST_RUN_DIR}"
echo "  eval outputs : ${OUTPUT_ROOT}/evals"
echo "  quick summary: ${OUTPUT_ROOT}/evals/summary.md"
