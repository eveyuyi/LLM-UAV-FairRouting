#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

show_help() {
  cat <<'EOF'
Usage:
  bash scripts/eval_single_model_priority_alignment.sh [DATASET_DIR]

Same inputs as eval_pre_post_priority_alignment.sh, but evaluates one served
model (rank-only / skip-solver) and writes:
  ${OUTPUT_ROOT}/evals/alignment.json
  ${OUTPUT_ROOT}/evals/eval_manifest.json
  ${OUTPUT_ROOT}/evals/summary.json
  ${OUTPUT_ROOT}/evals/summary.md

Model endpoint:
  API_BASE (default http://127.0.0.1:8000/v1)
  MODEL_NAME (default: SERVED_MODEL_NAME, else POST_MODEL, else qwen3-local)

Example:
  API_BASE=http://127.0.0.1:8010/v1 MODEL_NAME=my-model \
  PRIORITY_MODE=llm-only RANK_ONLY_MODE=llm3_only AUTO_TIMESTAMP_OUTPUT_ROOT=0 \
  OUTPUT_ROOT=data/eval_runs/single/my_run \
  bash scripts/eval_single_model_priority_alignment.sh data/test/test_seeds/hard_eval/seed_5101
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
DEFAULT_OUTPUT_ROOT="data/eval_runs/single_model_rank_only"
DEFAULT_TIME_SLOTS_STR="0 1 2 3 4 5 6 7 8 9"
AUTO_TIMESTAMP_OUTPUT_ROOT="${AUTO_TIMESTAMP_OUTPUT_ROOT:-1}"
PRIORITY_MODE="${PRIORITY_MODE:-hybrid}"

API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME:-${POST_MODEL:-qwen3-local}}}"

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

if [[ "${PRIORITY_MODE}" != "rule-only" && "${PRIORITY_MODE}" != "llm-only" && "${PRIORITY_MODE}" != "hybrid" ]]; then
  echo "Unsupported priority mode: ${PRIORITY_MODE}. Expected one of: rule-only, llm-only, hybrid" >&2
  exit 1
fi

if [[ "${AUTO_TIMESTAMP_OUTPUT_ROOT}" == "1" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT%/}_$(date '+%Y%m%d_%H%M%S')"
fi

if [[ -z "${API_KEY}" ]]; then
  echo "Missing OPENAI_API_KEY. Please export OPENAI_API_KEY first." >&2
  exit 1
fi

if [[ ! -f "${GROUND_TRUTH}" ]]; then
  echo "Missing ground-truth file: ${GROUND_TRUTH}" >&2
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
echo "[config] api_base=${API_BASE}" >&2
echo "[config] model_name=${MODEL_NAME}" >&2
echo "[config] priority_mode=${PRIORITY_MODE}" >&2
echo "[config] time_slots=${TIME_SLOTS_STR:-<all>}" >&2

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
run_rank_only_once "model" "${API_BASE}" "${MODEL_NAME}" "${PRIORITY_MODE}"
MODEL_RUN_DIR="${RUN_DIR_RESULT}"

echo "[eval] priority alignment (single model)"
PYTHONPATH=src "${_py[@]}" evals/eval_priority_alignment.py \
  --weights "${MODEL_RUN_DIR}/weight_configs" \
  --demands "${ALIGNMENT_DEMANDS_PATH:-${MODEL_RUN_DIR}/extracted_demands.json}" \
  --dialogues "${DIALOGUES}" \
  --ground-truth "${GROUND_TRUTH}" \
  --urgent-threshold "${URGENT_THRESHOLD}" \
  --truth-source "${TRUTH_SOURCE}" \
  "${TRUTH_ARGS[@]}" \
  --output "${OUTPUT_ROOT}/evals/alignment.json"

cat > "${OUTPUT_ROOT}/evals/eval_manifest.json" <<EOF
{
  "mode": "rank_only_single_model",
  "rank_only_mode": "${RANK_ONLY_MODE}",
  "truth_source": "${TRUTH_SOURCE}",
  "priority_mode": "${PRIORITY_MODE}",
  "api_base": "${API_BASE}",
  "model_name": "${MODEL_NAME}",
  "fixed_extracted_demands": "${FIXED_EXTRACTED_DEMANDS}",
  "selected_alignment_demands": "${ALIGNMENT_DEMANDS_PATH}",
  "selection_manifest": "${SELECTION_MANIFEST_PATH}",
  "run_dir": "${MODEL_RUN_DIR}",
  "alignment": "${OUTPUT_ROOT}/evals/alignment.json",
  "summary_json": "${OUTPUT_ROOT}/evals/summary.json",
  "summary_md": "${OUTPUT_ROOT}/evals/summary.md",
  "time_slots": "${TIME_SLOTS_STR}",
  "urgent_threshold": ${URGENT_THRESHOLD}
}
EOF

PYTHONPATH=src "${_py[@]}" evals/build_single_model_eval_summary.py \
  --manifest "${OUTPUT_ROOT}/evals/eval_manifest.json" \
  --output-json "${OUTPUT_ROOT}/evals/summary.json" \
  --output-md "${OUTPUT_ROOT}/evals/summary.md"

echo ""
echo "Finished single-model rank-only evaluation."
echo "  run_dir      : ${MODEL_RUN_DIR}"
echo "  eval outputs : ${OUTPUT_ROOT}/evals"
echo "  quick summary: ${OUTPUT_ROOT}/evals/summary.md"
