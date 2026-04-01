#!/usr/bin/env bash
set -euo pipefail

# Convert a VERL FSDP SFT checkpoint into a HuggingFace model directory
# that can be used as GRPO warm-start model.path.
#
# Usage:
#   bash scripts/export_sft_ckpt_to_hf.sh \
#     data/checkpoints/llm3_sft/global_step_1 \
#     data/checkpoints/llm3_sft_merged_hf/global_step_1

SFT_CKPT_DIR="${1:-data/checkpoints/llm3_sft/global_step_1}"
TARGET_DIR="${2:-data/checkpoints/llm3_sft_merged_hf/global_step_1}"

if [[ ! -d "${SFT_CKPT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${SFT_CKPT_DIR}" >&2
  exit 1
fi

# PPO checkpoints are usually .../global_step_x/actor, while SFT checkpoints
# are often directly .../global_step_x. Support both by auto-detecting.
LOCAL_DIR="${SFT_CKPT_DIR}"
if [[ -d "${SFT_CKPT_DIR}/actor" && ! -f "${SFT_CKPT_DIR}/fsdp_config.json" ]]; then
  LOCAL_DIR="${SFT_CKPT_DIR}/actor"
fi

mkdir -p "${TARGET_DIR}"

LORA_META_PATH="${LOCAL_DIR}/lora_train_meta.json"
LORA_META_BAK_PATH=""

cleanup() {
  if [[ -n "${LORA_META_BAK_PATH}" && -f "${LORA_META_BAK_PATH}" ]]; then
    mv -f "${LORA_META_BAK_PATH}" "${LORA_META_PATH}"
  fi
}
trap cleanup EXIT

if [[ -f "${LORA_META_PATH}" ]]; then
  # Work around a verl model_merger bug in some versions:
  # task_type is loaded as string and later treated as Enum (.value), causing
  # AttributeError: 'str' object has no attribute 'value'.
  # We temporarily null task_type during merge, then restore metadata.
  export LORA_META_PATH
  NEED_SANITIZE="$(PYTHONNOUSERSITE=1 python - <<'PY'
import json
import os

path = os.environ["LORA_META_PATH"]
with open(path, encoding="utf-8") as f:
    meta = json.load(f)
print("1" if isinstance(meta.get("task_type"), str) else "0")
PY
)"
  if [[ "${NEED_SANITIZE}" == "1" ]]; then
    LORA_META_BAK_PATH="${LORA_META_PATH}.bak_for_model_merger"
    cp "${LORA_META_PATH}" "${LORA_META_BAK_PATH}"
    PYTHONNOUSERSITE=1 python - <<'PY'
import json
import os

path = os.environ["LORA_META_PATH"]
with open(path, encoding="utf-8") as f:
    meta = json.load(f)
meta["task_type"] = None
with open(path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=4)
    f.write("\n")
print("Patched lora_train_meta.json task_type -> null (temporary).")
PY
  fi
fi

echo "[1/3] Merging FSDP checkpoint to HuggingFace format"
echo "      local_dir = ${LOCAL_DIR}"
echo "      target_dir = ${TARGET_DIR}"
PYTHONNOUSERSITE=1 python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${LOCAL_DIR}" \
  --target_dir "${TARGET_DIR}"

echo "[2/3] Sanity check merged model files"
if [[ ! -f "${TARGET_DIR}/config.json" ]]; then
  echo "Missing config.json in ${TARGET_DIR}" >&2
  exit 1
fi
if [[ ! -f "${TARGET_DIR}/tokenizer.json" ]]; then
  echo "Missing tokenizer.json in ${TARGET_DIR}" >&2
  exit 1
fi
if [[ ! -f "${TARGET_DIR}/model.safetensors" && ! -f "${TARGET_DIR}/pytorch_model.bin" ]]; then
  echo "Missing model weights (.safetensors/.bin) in ${TARGET_DIR}" >&2
  exit 1
fi

echo "[3/3] Loading merged model/tokenizer"
export TARGET_DIR
PYTHONNOUSERSITE=1 python - <<'PY'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

target = os.environ["TARGET_DIR"]
tok = AutoTokenizer.from_pretrained(target, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(target, trust_remote_code=True)
print("Tokenizer vocab size:", len(tok))
print("Model class:", model.__class__.__name__)
print("Warm-start path ready:", target)
PY

echo
echo "Done. Use this in GRPO:"
echo "  actor_rollout_ref.model.path=${TARGET_DIR}"
