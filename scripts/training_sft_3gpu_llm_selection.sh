#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 3 卡 SFT（llm_selection）
# 用法：
#   bash scripts/training_sft_3gpu_llm_selection.sh
#
# 默认流程：
# 1) 从原始 llm_selection JSONL 生成 compact JSONL（4B short 配置）
# 2) 将 train/val SFT JSONL 转为 parquet
# 3) 启动 VERL SFT 训练（3 GPUs）
# 4) 自动把最新 SFT checkpoint merge 成 HuggingFace 目录
# 5) 可选地直接串行启动 GRPO

# ---------- 只改这里 ----------
CONDA_ENV="${CONDA_ENV:-verl}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/model--Qwen-Qwen3-4B-Instruct-2507}"
# /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models-Qwen-Qwen3-1.7B
# "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/model--Qwen-Qwen3-4B-Instruct-2507"
# "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3.5-9B"

RAW_JSONL="${RAW_JSONL:-data/train/llm_selection_training_jsonl_1000_qs_v1_20260413/llm_selection_train_1000_qs_v1.jsonl}"
COMPACT_DIR="${COMPACT_DIR:-data/train/llm_selection_training_jsonl_1000_qs_v1_20260413/compact_exports_4b_short}"
# 这份 llm_selection 任务的 prompt 很长；即便 short 配置也容易逼近上下文上限。
# 默认进一步压缩到 3 候选 / 6 需求卡，优先保证关键差异特征存在且不被截断。
KEEP_CANDIDATES_PER_GROUP="${KEEP_CANDIDATES_PER_GROUP:-3}"
MAX_DEMAND_CARDS="${MAX_DEMAND_CARDS:-6}"
DATA_VAL_RATIO="${DATA_VAL_RATIO:-0.1}"
DATA_SEED="${DATA_SEED:-42}"
MULTIGROUP_OVERSAMPLE="${MULTIGROUP_OVERSAMPLE:-3}"

SFT_TRAIN_JSONL="${SFT_TRAIN_JSONL:-${COMPACT_DIR}/train_sft_compact.jsonl}"
SFT_VAL_JSONL="${SFT_VAL_JSONL:-${COMPACT_DIR}/val_sft_compact.jsonl}"
SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-data/train/verl/llm_selection_4b_short_sft_train.parquet}"
SFT_VAL_FILE="${SFT_VAL_FILE:-data/train/verl/llm_selection_4b_short_sft_val.parquet}"

AUTO_PREPARE_DATA="${AUTO_PREPARE_DATA:-1}"
AUTO_CONVERT_PARQUET="${AUTO_CONVERT_PARQUET:-1}"

CKPT_DIR="${CKPT_DIR:-data/checkpoints/llm_selection_sft_3gpu}"
HYDRA_ROOT="${HYDRA_ROOT:-data/hydra_outputs}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-llm-selection-sft}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-llm-selection-sft-3gpu}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-[\"console\",\"tensorboard\"]}"
SFT_RESUME_MODE="${SFT_RESUME_MODE:-disable}"
FAIL_ON_EXISTING_CKPT_DIR="${FAIL_ON_EXISTING_CKPT_DIR:-0}"
AUTO_MERGE_AFTER_SFT="${AUTO_MERGE_AFTER_SFT:-1}"
MERGED_MODEL_DIR_NAME="${MERGED_MODEL_DIR_NAME:-huggingface_lora_merged}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-}"
FAIL_ON_EXISTING_MERGED_MODEL_DIR="${FAIL_ON_EXISTING_MERGED_MODEL_DIR:-0}"
AUTO_RUN_GRPO_AFTER_MERGE="${AUTO_RUN_GRPO_AFTER_MERGE:-0}"
GRPO_SCRIPT="${GRPO_SCRIPT:-scripts/training_grpo_3gpu_llm_selection.sh}"

# 这份数据是小规模、结构化 JSON 输出、且几乎全是 hard decision。
# 用较保守的 epoch 避免过拟合规则标签；学习率保持中等，方便模型尽快学会格式和选择逻辑。
SFT_GLOBAL_BATCH_SIZE="${SFT_GLOBAL_BATCH_SIZE:-24}"
SFT_LR="${SFT_LR:-1e-5}"
SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-2}"
SFT_MODEL_DTYPE="${SFT_MODEL_DTYPE:-bfloat16}"
SFT_TORCH_COMPILE="${SFT_TORCH_COMPILE:-false}"
SFT_DATALOADER_WORKERS="${SFT_DATALOADER_WORKERS:-2}"
SFT_LORA_RANK="${SFT_LORA_RANK:-32}"
SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-16}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-4096}"

if [[ -n "${CONDA_ENV}" ]]; then
  _py=(conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python)
else
  _py=(env PYTHONNOUSERSITE=1 python)
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "MODEL_PATH 必须是 HuggingFace 模型目录（缺少 config.json）: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ "${AUTO_RUN_GRPO_AFTER_MERGE}" == "1" && "${AUTO_MERGE_AFTER_SFT}" != "1" ]]; then
  echo "AUTO_RUN_GRPO_AFTER_MERGE=1 需要 AUTO_MERGE_AFTER_SFT=1。" >&2
  exit 1
fi

if [[ "${SFT_RESUME_MODE}" == "disable" && "${FAIL_ON_EXISTING_CKPT_DIR}" == 1 && -d "${CKPT_DIR}" ]]; then
  echo "CKPT_DIR already exists with resume disabled: ${CKPT_DIR}" >&2
  exit 1
fi

if [[ "${AUTO_PREPARE_DATA}" == "1" ]]; then
  "${_py[@]}" scripts/export_llm_selection_compact.py \
    --input-jsonl "${RAW_JSONL}" \
    --output-dir "${COMPACT_DIR}" \
    --keep-candidates-per-group "${KEEP_CANDIDATES_PER_GROUP}" \
    --max-demand-cards "${MAX_DEMAND_CARDS}" \
    --val-ratio "${DATA_VAL_RATIO}" \
    --seed "${DATA_SEED}" \
    --multigroup-oversample "${MULTIGROUP_OVERSAMPLE}"
fi

if [[ "${AUTO_CONVERT_PARQUET}" == "1" || ! -f "${SFT_TRAIN_FILE}" || ! -f "${SFT_VAL_FILE}" ]]; then
  mkdir -p "$(dirname "${SFT_TRAIN_FILE}")" "$(dirname "${SFT_VAL_FILE}")"
  "${_py[@]}" scripts/jsonl_to_parquet.py \
    --input-jsonl "${SFT_TRAIN_JSONL}" \
    --output-parquet "${SFT_TRAIN_FILE}"
  "${_py[@]}" scripts/jsonl_to_parquet.py \
    --input-jsonl "${SFT_VAL_JSONL}" \
    --output-parquet "${SFT_VAL_FILE}"
fi

IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${#_GPU_IDS[@]}"
if [[ "${NPROC_PER_NODE}" != "3" ]]; then
  echo "本脚本按 3 卡设计，请设置 CUDA_VISIBLE_DEVICES 为 3 张卡（当前: ${CUDA_VISIBLE_DEVICES}）" >&2
  exit 1
fi

echo "GPUs=${CUDA_VISIBLE_DEVICES} global_batch=${SFT_GLOBAL_BATCH_SIZE} lr=${SFT_LR} epochs=${SFT_TOTAL_EPOCHS} model_dtype=${SFT_MODEL_DTYPE} conda=${CONDA_ENV:-<none>} train=${SFT_TRAIN_FILE} val=${SFT_VAL_FILE} model=${MODEL_PATH} ckpt=${CKPT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${_py[@]}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${SFT_TRAIN_FILE}" \
  data.val_files="${SFT_VAL_FILE}" \
  data.messages_key=messages \
  data.train_batch_size="${SFT_GLOBAL_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu=1 \
  data.num_workers="${SFT_DATALOADER_WORKERS}" \
  data.max_length="${SFT_MAX_LENGTH}" \
  data.truncation=right \
  data.ignore_input_ids_mismatch=true \
  data.enable_thinking_default=false \
  optim.lr="${SFT_LR}" \
  engine.model_dtype="${SFT_MODEL_DTYPE}" \
  engine.use_torch_compile="${SFT_TORCH_COMPILE}" \
  model.path="${MODEL_PATH}" \
  model.lora_rank="${SFT_LORA_RANK}" \
  model.lora_alpha="${SFT_LORA_ALPHA}" \
  model.target_modules=all-linear \
  +model.override_config.attn_implementation=sdpa \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.project_name="${TRAINER_PROJECT_NAME}" \
  trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}" \
  trainer.logger="${TRAINER_LOGGERS}" \
  trainer.resume_mode="${SFT_RESUME_MODE}" \
  trainer.total_epochs="${SFT_TOTAL_EPOCHS}" \
  "hydra.run.dir=${HYDRA_ROOT}/"'${now:%Y-%m-%d}/${now:%H-%M-%S}'

resolve_latest_step_dir() {
  local ckpt_root="$1"
  local latest_dir=""
  local latest_step=-1
  local step_dir=""
  shopt -s nullglob
  for step_dir in "${ckpt_root}"/global_step_*; do
    local step_name
    local step_num
    step_name="$(basename "${step_dir}")"
    step_num="${step_name#global_step_}"
    [[ "${step_num}" =~ ^[0-9]+$ ]] || continue
    if (( step_num > latest_step )); then
      latest_step="${step_num}"
      latest_dir="${step_dir}"
    fi
  done
  shopt -u nullglob
  if [[ -z "${latest_dir}" ]]; then
    echo "未在 ${ckpt_root} 下找到 global_step_* checkpoint。" >&2
    return 1
  fi
  printf '%s\n' "${latest_dir}"
}

require_hf_model_dir() {
  local model_dir="$1"
  if [[ ! -f "${model_dir}/config.json" ]]; then
    echo "Merged 模型缺少 config.json: ${model_dir}" >&2
    return 1
  fi
  if [[ ! -f "${model_dir}/tokenizer.json" && ! -f "${model_dir}/tokenizer_config.json" ]]; then
    echo "Merged 模型缺少 tokenizer 文件: ${model_dir}" >&2
    return 1
  fi
  shopt -s nullglob
  local weight_files=(
    "${model_dir}"/model.safetensors
    "${model_dir}"/pytorch_model.bin
    "${model_dir}"/model-*.safetensors
    "${model_dir}"/pytorch_model-*.bin
  )
  shopt -u nullglob
  if (( ${#weight_files[@]} == 0 )); then
    echo "Merged 模型缺少权重文件: ${model_dir}" >&2
    return 1
  fi
}

if [[ "${AUTO_MERGE_AFTER_SFT}" == "1" ]]; then
  LATEST_STEP_DIR="$(resolve_latest_step_dir "${CKPT_DIR}")"
  if [[ -n "${MERGED_MODEL_DIR}" ]]; then
    MERGED_MODEL_PATH="${MERGED_MODEL_DIR}"
  else
    MERGED_MODEL_PATH="${LATEST_STEP_DIR}/${MERGED_MODEL_DIR_NAME}"
  fi

  if [[ "${FAIL_ON_EXISTING_MERGED_MODEL_DIR}" == "1" && -d "${MERGED_MODEL_PATH}" ]]; then
    echo "Merged 输出目录已存在: ${MERGED_MODEL_PATH}" >&2
    exit 1
  fi

  echo "Merging latest SFT checkpoint: ${LATEST_STEP_DIR} -> ${MERGED_MODEL_PATH}"
  "${_py[@]}" scripts/loRA_to_merged.py \
    --ckpt-dir "${LATEST_STEP_DIR}" \
    --base-path "${MODEL_PATH}" \
    --output-path "${MERGED_MODEL_PATH}" \
    --lora-rank "${SFT_LORA_RANK}" \
    --lora-alpha "${SFT_LORA_ALPHA}"

  require_hf_model_dir "${MERGED_MODEL_PATH}"
  echo "Merged HuggingFace model ready: ${MERGED_MODEL_PATH}"

  if [[ "${AUTO_RUN_GRPO_AFTER_MERGE}" == "1" ]]; then
    echo "Starting GRPO from merged model: ${MERGED_MODEL_PATH}"
    CONDA_ENV="${CONDA_ENV}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    MODEL_PATH="${MERGED_MODEL_PATH}" \
    RAW_JSONL="${RAW_JSONL}" \
    COMPACT_DIR="${COMPACT_DIR}" \
    KEEP_CANDIDATES_PER_GROUP="${KEEP_CANDIDATES_PER_GROUP}" \
    MAX_DEMAND_CARDS="${MAX_DEMAND_CARDS}" \
    DATA_VAL_RATIO="${DATA_VAL_RATIO}" \
    DATA_SEED="${DATA_SEED}" \
    MULTIGROUP_OVERSAMPLE="${MULTIGROUP_OVERSAMPLE}" \
    bash "${GRPO_SCRIPT}"
  fi
fi
