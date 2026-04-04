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
OUTPUT_ROOT="${OUTPUT_ROOT:-data/eval_runs/pre_post_rank_only}"
FIXED_EXTRACTED_DEMANDS="${FIXED_EXTRACTED_DEMANDS:-}"
RANK_ONLY_MODE="${RANK_ONLY_MODE:-auto}" # auto | llm3_only | end_to_end
TRUTH_SOURCE="${TRUTH_SOURCE:-auto}"     # auto | run_extracted | fixed_demands | ground_truth_manifest
WINDOW_INDICES_STR="${WINDOW_INDICES_STR:-}"
TIME_WINDOWS_STR="${TIME_WINDOWS_STR:-}"

TIME_SLOTS_STR="${TIME_SLOTS_STR:-0 1 2 3 4 5 6 7 8 9}"
URGENT_THRESHOLD="${URGENT_THRESHOLD:-2}"

if [[ -z "${API_KEY}" ]]; then
  echo "Missing OPENAI_API_KEY. Please export OPENAI_API_KEY first." >&2
  exit 1
fi

if [[ ! -f "${GROUND_TRUTH}" ]]; then
  echo "Missing ground-truth file: ${GROUND_TRUTH}" >&2
  echo "Priority alignment evaluation requires ground-truth priorities." >&2
  echo "Please set GROUND_TRUTH=/path/to/events_manifest.jsonl, then rerun." >&2
  echo "Example to generate default seed manifest:" >&2
  echo "  llm4fairrouting-demand-events --manifest-output data/seed/daily_demand_events_manifest.jsonl" >&2
  exit 1
fi

read -r -a TIME_SLOTS <<< "${TIME_SLOTS_STR}"
if [[ -n "${TIME_SLOTS_STR}" && "${#TIME_SLOTS[@]}" -eq 0 ]]; then
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

if [[ -z "${FIXED_EXTRACTED_DEMANDS}" ]]; then
  _candidate_fixed="$(dirname "${DIALOGUES}")/llm3_sft_pipeline.jsonl"
  if [[ -f "${_candidate_fixed}" ]]; then
    FIXED_EXTRACTED_DEMANDS="${_candidate_fixed}"
  fi
fi

if [[ "${RANK_ONLY_MODE}" == "auto" ]]; then
  if [[ -n "${FIXED_EXTRACTED_DEMANDS}" ]]; then
    RANK_ONLY_MODE="llm3_only"
  else
    RANK_ONLY_MODE="end_to_end"
  fi
fi

ALIGNMENT_DEMANDS_PATH=""
TRUTH_DEMANDS_PATH=""
RUN_WORKFLOW_EXTRA_ARGS=()
TIME_SLOT_ARGS=()
TRUTH_ARGS=()

prepare_llm3_only_inputs() {
  if [[ -z "${FIXED_EXTRACTED_DEMANDS}" ]]; then
    echo "RANK_ONLY_MODE=llm3_only requires FIXED_EXTRACTED_DEMANDS or a sibling llm3_sft_pipeline.jsonl file." >&2
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
print(selected_path)
PY

  ALIGNMENT_DEMANDS_PATH="${selected_demands}"
  TRUTH_DEMANDS_PATH="${selected_demands}"
  RUN_WORKFLOW_EXTRA_ARGS=(--extracted-demands "${selected_demands}")

  if [[ "${TRUTH_SOURCE}" == "auto" ]]; then
    TRUTH_SOURCE="fixed_demands"
  fi
}

if [[ "${RANK_ONLY_MODE}" == "llm3_only" ]]; then
  prepare_llm3_only_inputs
else
  if [[ "${TRUTH_SOURCE}" == "auto" ]]; then
    TRUTH_SOURCE="run_extracted"
  fi
  if [[ "${#TIME_SLOTS[@]}" -gt 0 ]]; then
    TIME_SLOT_ARGS=(--time-slots "${TIME_SLOTS[@]}")
  fi
fi

if [[ "${TRUTH_SOURCE}" == "fixed_demands" ]]; then
  if [[ -z "${TRUTH_DEMANDS_PATH}" ]]; then
    echo "TRUTH_SOURCE=fixed_demands requires a resolved TRUTH_DEMANDS_PATH." >&2
    exit 1
  fi
  TRUTH_ARGS=(--truth-demands "${TRUTH_DEMANDS_PATH}")
fi

run_rank_only_once() {
  local tag="$1"
  local api_base="$2"
  local model="$3"
  local out_base="${OUTPUT_ROOT}/${tag}"
  mkdir -p "${out_base}"

  echo "[${tag}] rank-only workflow (skip solver), model=${model}, api_base=${api_base}" >&2
  OPENAI_API_KEY="${API_KEY}" OPENAI_BASE_URL="${api_base}" PYTHONPATH=src \
  "${_py[@]}" -m llm4fairrouting.workflow.run_workflow \
    --output-dir "${out_base}" \
    --dialogues "${DIALOGUES}" \
    --stations "${STATIONS}" \
    --building-data "${BUILDING_DATA}" \
    --model "${model}" \
    "${RUN_WORKFLOW_EXTRA_ARGS[@]}" \
    "${TIME_SLOT_ARGS[@]}" \
    --skip-solver \
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

mkdir -p "${OUTPUT_ROOT}/evals"

RUN_DIR_RESULT=""
run_rank_only_once "pre" "${PRE_API_BASE}" "${PRE_MODEL}"
PRE_RUN_DIR="${RUN_DIR_RESULT}"
run_rank_only_once "post" "${POST_API_BASE}" "${POST_MODEL}"
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

echo "[eval] alignment delta (post - pre)"
PYTHONPATH=src "${_py[@]}" - <<'PY'
import json
from pathlib import Path

output_root = Path("data/eval_runs/pre_post_rank_only")
if "OUTPUT_ROOT" in __import__("os").environ:
    output_root = Path(__import__("os").environ["OUTPUT_ROOT"])

pre_path = output_root / "evals" / "pre_alignment.json"
post_path = output_root / "evals" / "post_alignment.json"
delta_path = output_root / "evals" / "post_vs_pre_alignment_delta.json"

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

def nested_diff(path):
    a = pre
    b = post
    for key in path:
        a = (a or {}).get(key)
        b = (b or {}).get(key)
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 6)

payload = {
    "pre_alignment": str(pre_path),
    "post_alignment": str(post_path),
    "truth_source": post.get("truth_source") or pre.get("truth_source"),
    "delta_metrics_post_minus_pre": {
        "accuracy": diff("accuracy"),
        "macro_f1": diff("macro_f1"),
        "weighted_f1": diff("weighted_f1"),
        "spearman": diff("spearman"),
        "kendall_tau": diff("kendall_tau"),
        "top_k_hit_rate": None,
        "priority_1_recall": nested_diff(("priority_1_metrics", "recall")),
        "priority_1_f1": nested_diff(("priority_1_metrics", "f1")),
        "urgent_recall": nested_diff(("urgent_metrics", "recall")),
        "urgent_f1": nested_diff(("urgent_metrics", "f1")),
    },
    "n_aligned_demands": {
        "pre": pre.get("n_aligned_demands"),
        "post": post.get("n_aligned_demands"),
    },
}

# Top-k hit-rate is nested.
pre_topk = (pre.get("top_k_hit_rate") or {}).get("hit_rate")
post_topk = (post.get("top_k_hit_rate") or {}).get("hit_rate")
if pre_topk is None or post_topk is None:
    payload["delta_metrics_post_minus_pre"]["top_k_hit_rate"] = None
else:
    payload["delta_metrics_post_minus_pre"]["top_k_hit_rate"] = round(float(post_topk) - float(pre_topk), 6)

delta_path.parent.mkdir(parents=True, exist_ok=True)
with delta_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"Alignment delta saved to {delta_path}")
PY

cat > "${OUTPUT_ROOT}/evals/eval_manifest.json" <<EOF
{
  "mode": "rank_only_alignment",
  "rank_only_mode": "${RANK_ONLY_MODE}",
  "truth_source": "${TRUTH_SOURCE}",
  "fixed_extracted_demands": "${FIXED_EXTRACTED_DEMANDS}",
  "selected_alignment_demands": "${ALIGNMENT_DEMANDS_PATH}",
  "pre_run_dir": "${PRE_RUN_DIR}",
  "post_run_dir": "${POST_RUN_DIR}",
  "pre_alignment": "${OUTPUT_ROOT}/evals/pre_alignment.json",
  "post_alignment": "${OUTPUT_ROOT}/evals/post_alignment.json",
  "post_vs_pre_alignment_delta": "${OUTPUT_ROOT}/evals/post_vs_pre_alignment_delta.json",
  "time_slots": "${TIME_SLOTS_STR}",
  "urgent_threshold": ${URGENT_THRESHOLD}
}
EOF

echo ""
echo "Finished rank-only pre/post evaluation."
echo "  pre_run_dir  : ${PRE_RUN_DIR}"
echo "  post_run_dir : ${POST_RUN_DIR}"
echo "  eval outputs : ${OUTPUT_ROOT}/evals"
