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

TIME_SLOTS_STR="${TIME_SLOTS_STR:-0 1 2 3 4 5 6 7 8 9}"
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
    --time-slots "${TIME_SLOTS[@]}" \
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

payload = {
    "pre_alignment": str(pre_path),
    "post_alignment": str(post_path),
    "delta_metrics_post_minus_pre": {
        "accuracy": diff("accuracy"),
        "macro_f1": diff("macro_f1"),
        "weighted_f1": diff("weighted_f1"),
        "spearman": diff("spearman"),
        "kendall_tau": diff("kendall_tau"),
        "top_k_hit_rate": None,
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
