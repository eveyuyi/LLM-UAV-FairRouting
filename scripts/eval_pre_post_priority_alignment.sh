#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/eval_pre_post_priority_alignment.sh [DATASET_DIR]

Convenience shortcuts:
  1. Pass a dataset shard directory as the only positional argument, e.g.
       bash scripts/eval_pre_post_priority_alignment.sh data/train/llm3_medium_5min_v1/seed_4111
  2. Or set DATASET_DIR=/path/to/seed_xxxx

When DATASET_DIR is provided, the script will auto-resolve:
  - DIALOGUES=<DATASET_DIR>/dialogues.jsonl
  - GROUND_TRUTH=<DATASET_DIR>/events_manifest.jsonl
  - FIXED_EXTRACTED_DEMANDS=<DATASET_DIR>/llm3_sft_pipeline.jsonl
    (falls back to llm3_sft_clean.jsonl when pipeline is absent)

For DATASET_DIR-based runs, if TIME_SLOTS_STR is not explicitly set,
the script evaluates all windows in the shard by default.

By default, AUTO_TIMESTAMP_OUTPUT_ROOT=1, so each run appends a timestamp
suffix to OUTPUT_ROOT and keeps historical evaluation summaries.

Priority mode:
  - Set PRIORITY_MODE=rule-only|llm-only|hybrid for both pre/post runs.
  - Or override separately with PRE_PRIORITY_MODE=... and POST_PRIORITY_MODE=...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

if [[ "$#" -gt 1 ]]; then
  echo "Expected at most one positional argument: DATASET_DIR" >&2
  show_help >&2
  exit 1
fi

DIALOGUES_WAS_SET="${DIALOGUES+x}"
GROUND_TRUTH_WAS_SET="${GROUND_TRUTH+x}"
OUTPUT_ROOT_WAS_SET="${OUTPUT_ROOT+x}"
FIXED_EXTRACTED_DEMANDS_WAS_SET="${FIXED_EXTRACTED_DEMANDS+x}"
TIME_SLOTS_STR_WAS_SET="${TIME_SLOTS_STR+x}"
DATASET_DIR="${1:-${DATASET_DIR:-}}"

# ---------- Config ----------
CONDA_ENV="${CONDA_ENV:-}"
API_KEY="${OPENAI_API_KEY:-}"
DEFAULT_OUTPUT_ROOT="data/eval_runs/pre_post_rank_only"
DEFAULT_TIME_SLOTS_STR="0 1 2 3 4 5 6 7 8 9"
AUTO_TIMESTAMP_OUTPUT_ROOT="${AUTO_TIMESTAMP_OUTPUT_ROOT:-1}"
PRIORITY_MODE="${PRIORITY_MODE:-hybrid}"
PRE_PRIORITY_MODE="${PRE_PRIORITY_MODE:-${PRIORITY_MODE}}"
POST_PRIORITY_MODE="${POST_PRIORITY_MODE:-${PRIORITY_MODE}}"

PRE_API_BASE="${PRE_API_BASE:-http://127.0.0.1:8000/v1}"
PRE_MODEL="${PRE_MODEL:-qwen3-pre}"
POST_API_BASE="${POST_API_BASE:-http://127.0.0.1:8001/v1}"
POST_MODEL="${POST_MODEL:-qwen3-post}"

DIALOGUES="${DIALOGUES:-data/seed/daily_demand_dialogues.jsonl}"
GROUND_TRUTH="${GROUND_TRUTH:-data/seed/daily_demand_events_manifest.jsonl}"
STATIONS="${STATIONS:-data/seed/drone_station_locations.csv}"
BUILDING_DATA="${BUILDING_DATA:-data/seed/building_information.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
FIXED_EXTRACTED_DEMANDS="${FIXED_EXTRACTED_DEMANDS:-}"
RANK_ONLY_MODE="${RANK_ONLY_MODE:-auto}" # auto | llm3_only | end_to_end
TRUTH_SOURCE="${TRUTH_SOURCE:-auto}"     # auto | run_extracted | fixed_demands | ground_truth_manifest
WINDOW_INDICES_STR="${WINDOW_INDICES_STR:-}"
TIME_WINDOWS_STR="${TIME_WINDOWS_STR:-}"

if [[ -n "${TIME_SLOTS_STR_WAS_SET}" ]]; then
  TIME_SLOTS_STR="${TIME_SLOTS_STR-}"
else
  TIME_SLOTS_STR="${DEFAULT_TIME_SLOTS_STR}"
fi
URGENT_THRESHOLD="${URGENT_THRESHOLD:-2}"

if [[ -n "${DATASET_DIR}" ]]; then
  if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "DATASET_DIR does not exist or is not a directory: ${DATASET_DIR}" >&2
    exit 1
  fi

  if [[ -z "${DIALOGUES_WAS_SET}" ]]; then
    DIALOGUES="${DATASET_DIR}/dialogues.jsonl"
  fi
  if [[ -z "${GROUND_TRUTH_WAS_SET}" ]]; then
    GROUND_TRUTH="${DATASET_DIR}/events_manifest.jsonl"
  fi
  if [[ -z "${FIXED_EXTRACTED_DEMANDS_WAS_SET}" ]]; then
    if [[ -f "${DATASET_DIR}/llm3_sft_pipeline.jsonl" ]]; then
      FIXED_EXTRACTED_DEMANDS="${DATASET_DIR}/llm3_sft_pipeline.jsonl"
    elif [[ -f "${DATASET_DIR}/llm3_sft_clean.jsonl" ]]; then
      FIXED_EXTRACTED_DEMANDS="${DATASET_DIR}/llm3_sft_clean.jsonl"
    fi
  fi
  if [[ -z "${OUTPUT_ROOT_WAS_SET}" ]]; then
    dataset_parent="$(basename "$(dirname "${DATASET_DIR}")")"
    dataset_name="$(basename "${DATASET_DIR}")"
    OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}_${dataset_parent}_${dataset_name}"
  fi
  if [[ -z "${TIME_SLOTS_STR_WAS_SET}" ]]; then
    TIME_SLOTS_STR=""
  fi
fi

if [[ "${AUTO_TIMESTAMP_OUTPUT_ROOT}" != "0" && "${AUTO_TIMESTAMP_OUTPUT_ROOT}" != "1" ]]; then
  echo "AUTO_TIMESTAMP_OUTPUT_ROOT must be 0 or 1, got: ${AUTO_TIMESTAMP_OUTPUT_ROOT}" >&2
  exit 1
fi

VALID_MODES=("rule-only" "llm-only" "hybrid" "random" "uniform")
for mode in "${PRE_PRIORITY_MODE}" "${POST_PRIORITY_MODE}"; do
  valid=0
  for vm in "${VALID_MODES[@]}"; do [[ "$mode" == "$vm" ]] && valid=1 && break; done
  if [[ $valid -eq 0 ]]; then
    echo "Unsupported priority mode: ${mode}. Expected one of: ${VALID_MODES[*]}" >&2
    exit 1
  fi
done

if [[ "${AUTO_TIMESTAMP_OUTPUT_ROOT}" == "1" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT%/}_$(date '+%Y%m%d_%H%M%S')"
fi

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

if [[ ! -f "${DIALOGUES}" ]]; then
  echo "Missing dialogue file: ${DIALOGUES}" >&2
  exit 1
fi

if [[ -n "${FIXED_EXTRACTED_DEMANDS}" && ! -f "${FIXED_EXTRACTED_DEMANDS}" ]]; then
  echo "Missing fixed extracted demands file: ${FIXED_EXTRACTED_DEMANDS}" >&2
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

echo "[config] dataset_dir=${DATASET_DIR:-<default>}" >&2
echo "[config] dialogues=${DIALOGUES}" >&2
echo "[config] ground_truth=${GROUND_TRUTH}" >&2
echo "[config] fixed_extracted_demands=${FIXED_EXTRACTED_DEMANDS:-<none>}" >&2
echo "[config] output_root=${OUTPUT_ROOT}" >&2
echo "[config] auto_timestamp_output_root=${AUTO_TIMESTAMP_OUTPUT_ROOT}" >&2
echo "[config] time_slots=${TIME_SLOTS_STR:-<all>}" >&2
echo "[config] pre_priority_mode=${PRE_PRIORITY_MODE}" >&2
echo "[config] post_priority_mode=${POST_PRIORITY_MODE}" >&2

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
SELECTION_MANIFEST_PATH=""

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
  SELECTION_MANIFEST_PATH="${selection_manifest}"

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
  local priority_mode="$4"
  local out_base="${OUTPUT_ROOT}/${tag}"
  mkdir -p "${out_base}"

  echo "[${tag}] rank-only workflow (skip solver), model=${model}, api_base=${api_base}, priority_mode=${priority_mode}" >&2
  OPENAI_API_KEY="${API_KEY}" OPENAI_BASE_URL="${api_base}" LLM4FAIRROUTING_TIME_SLOTS="${TIME_SLOTS_STR}" PYTHONPATH=src \
  "${_py[@]}" -m llm4fairrouting.workflow.run_workflow \
    --output-dir "${out_base}" \
    --dialogues "${DIALOGUES}" \
    --stations "${STATIONS}" \
    --building-data "${BUILDING_DATA}" \
    --model "${model}" \
    --priority-mode "${priority_mode}" \
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
run_rank_only_once "pre" "${PRE_API_BASE}" "${PRE_MODEL}" "${PRE_PRIORITY_MODE}"
PRE_RUN_DIR="${RUN_DIR_RESULT}"
run_rank_only_once "post" "${POST_API_BASE}" "${POST_MODEL}" "${POST_PRIORITY_MODE}"
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
OUTPUT_ROOT="${OUTPUT_ROOT}" PRE_PRIORITY_MODE="${PRE_PRIORITY_MODE}" POST_PRIORITY_MODE="${POST_PRIORITY_MODE}" PYTHONPATH=src "${_py[@]}" - <<'PY'
import json
import os
from pathlib import Path

output_root = Path(os.environ["OUTPUT_ROOT"])
pre_priority_mode = os.environ["PRE_PRIORITY_MODE"]
post_priority_mode = os.environ["POST_PRIORITY_MODE"]

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
    "priority_modes": {
        "pre": pre_priority_mode,
        "post": post_priority_mode,
    },
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
  "pre_priority_mode": "${PRE_PRIORITY_MODE}",
  "post_priority_mode": "${POST_PRIORITY_MODE}",
  "fixed_extracted_demands": "${FIXED_EXTRACTED_DEMANDS}",
  "selected_alignment_demands": "${ALIGNMENT_DEMANDS_PATH}",
  "selection_manifest": "${SELECTION_MANIFEST_PATH}",
  "slot_sampling": "",
  "pre_run_dir": "${PRE_RUN_DIR}",
  "post_run_dir": "${POST_RUN_DIR}",
  "pre_alignment": "${OUTPUT_ROOT}/evals/pre_alignment.json",
  "post_alignment": "${OUTPUT_ROOT}/evals/post_alignment.json",
  "post_vs_pre_alignment_delta": "${OUTPUT_ROOT}/evals/post_vs_pre_alignment_delta.json",
  "summary_json": "${OUTPUT_ROOT}/evals/summary.json",
  "summary_md": "${OUTPUT_ROOT}/evals/summary.md",
  "time_slots": "${TIME_SLOTS_STR}",
  "urgent_threshold": ${URGENT_THRESHOLD}
}
EOF

PYTHONPATH=src "${_py[@]}" evals/build_pre_post_eval_summary.py \
  --manifest "${OUTPUT_ROOT}/evals/eval_manifest.json" \
  --output-json "${OUTPUT_ROOT}/evals/summary.json" \
  --output-md "${OUTPUT_ROOT}/evals/summary.md"

echo ""
echo "Finished rank-only pre/post evaluation."
echo "  pre_run_dir  : ${PRE_RUN_DIR}"
echo "  post_run_dir : ${POST_RUN_DIR}"
echo "  eval outputs : ${OUTPUT_ROOT}/evals"
echo "  quick summary: ${OUTPUT_ROOT}/evals/summary.md"
