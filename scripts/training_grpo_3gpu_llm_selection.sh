#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 3 卡 GRPO（llm_selection）
# 用法：
#   bash scripts/training_grpo_3gpu_llm_selection.sh
#
# 默认流程：
# 1) 从原始 llm_selection JSONL 生成 compact JSONL（4B short 配置）
# 2) 将 train/val GRPO JSONL 转为 parquet
# 3) 用自定义 reward 启动 GRPO 训练（3 GPUs）

# ---------- 只改这里 ----------
CONDA_ENV="${CONDA_ENV:-verl}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
MODEL_PATH="${MODEL_PATH:-/path/to/sft-merged-hf-model}"

RAW_JSONL="${RAW_JSONL:-data/train/llm_selection_training_jsonl_1000_qs_v1_20260413/llm_selection_train_1000_qs_v1.jsonl}"
COMPACT_DIR="${COMPACT_DIR:-data/train/llm_selection_training_jsonl_1000_qs_v1_20260413/compact_exports_4b_short}"
KEEP_CANDIDATES_PER_GROUP="${KEEP_CANDIDATES_PER_GROUP:-4}"
MAX_DEMAND_CARDS="${MAX_DEMAND_CARDS:-8}"
DATA_VAL_RATIO="${DATA_VAL_RATIO:-0.1}"
DATA_SEED="${DATA_SEED:-42}"
MULTIGROUP_OVERSAMPLE="${MULTIGROUP_OVERSAMPLE:-3}"

GRPO_TRAIN_JSONL="${GRPO_TRAIN_JSONL:-${COMPACT_DIR}/train_grpo_compact.jsonl}"
GRPO_VAL_JSONL="${GRPO_VAL_JSONL:-${COMPACT_DIR}/val_grpo_compact.jsonl}"
GRPO_TRAIN_FILE="${GRPO_TRAIN_FILE:-data/train/verl/llm_selection_4b_short_grpo_train.parquet}"
GRPO_VAL_FILE="${GRPO_VAL_FILE:-data/train/verl/llm_selection_4b_short_grpo_val.parquet}"

AUTO_PREPARE_DATA="${AUTO_PREPARE_DATA:-1}"
AUTO_CONVERT_PARQUET="${AUTO_CONVERT_PARQUET:-1}"

CKPT_DIR="${CKPT_DIR:-data/checkpoints/llm_selection_grpo_3gpu}"
HYDRA_ROOT="${HYDRA_ROOT:-data/hydra_outputs}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-llm-selection-grpo}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-llm-selection-grpo-3gpu}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-[\"console\",\"tensorboard\"]}"
GRPO_RESUME_MODE="${GRPO_RESUME_MODE:-disable}"
FAIL_ON_EXISTING_CKPT_DIR="${FAIL_ON_EXISTING_CKPT_DIR:-0}"

# 4B + 3 卡建议起点
GRPO_TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-12}"
GRPO_PPO_MINI_BATCH_SIZE="${GRPO_PPO_MINI_BATCH_SIZE:-6}"
GRPO_ROLLOUT_N="${GRPO_ROLLOUT_N:-4}"
GRPO_ACTOR_LR="${GRPO_ACTOR_LR:-8e-7}"
GRPO_MODEL_DTYPE="${GRPO_MODEL_DTYPE:-bfloat16}"
GRPO_TORCH_COMPILE="${GRPO_TORCH_COMPILE:-false}"
GRPO_DATALOADER_WORKERS="${GRPO_DATALOADER_WORKERS:-2}"
GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-4096}"
GRPO_MAX_RESPONSE_LENGTH="${GRPO_MAX_RESPONSE_LENGTH:-384}"
GRPO_MAX_NUM_BATCHED_TOKENS="${GRPO_MAX_NUM_BATCHED_TOKENS:-8192}"
GRPO_ROLLOUT_MAX_MODEL_LEN="${GRPO_ROLLOUT_MAX_MODEL_LEN:-4096}"
GRPO_ROLLOUT_GPU_MEM_UTIL="${GRPO_ROLLOUT_GPU_MEM_UTIL:-0.45}"
GRPO_KL_LOSS_COEF="${GRPO_KL_LOSS_COEF:-0.01}"
GRPO_TOTAL_EPOCHS="${GRPO_TOTAL_EPOCHS:-2}"
GRPO_SAVE_FREQ="${GRPO_SAVE_FREQ:-50}"
GRPO_TEST_FREQ="${GRPO_TEST_FREQ:-25}"

if [[ -n "${CONDA_ENV}" ]]; then
  _py=(conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python)
else
  _py=(env PYTHONNOUSERSITE=1 python)
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "MODEL_PATH 必须是 HuggingFace 模型目录（缺少 config.json）: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ "${GRPO_RESUME_MODE}" == "disable" && "${FAIL_ON_EXISTING_CKPT_DIR}" == 1 && -d "${CKPT_DIR}" ]]; then
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

if [[ "${AUTO_CONVERT_PARQUET}" == "1" || ! -f "${GRPO_TRAIN_FILE}" || ! -f "${GRPO_VAL_FILE}" ]]; then
  mkdir -p "$(dirname "${GRPO_TRAIN_FILE}")" "$(dirname "${GRPO_VAL_FILE}")"
  "${_py[@]}" scripts/jsonl_to_parquet.py \
    --input-jsonl "${GRPO_TRAIN_JSONL}" \
    --output-parquet "${GRPO_TRAIN_FILE}"
  "${_py[@]}" scripts/jsonl_to_parquet.py \
    --input-jsonl "${GRPO_VAL_JSONL}" \
    --output-parquet "${GRPO_VAL_FILE}"
fi

IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${#_GPU_IDS[@]}"
if [[ "${NPROC_PER_NODE}" != "3" ]]; then
  echo "本脚本按 3 卡设计，请设置 CUDA_VISIBLE_DEVICES 为 3 张卡（当前: ${CUDA_VISIBLE_DEVICES}）" >&2
  exit 1
fi

if (( GRPO_TRAIN_BATCH_SIZE < GRPO_PPO_MINI_BATCH_SIZE )); then
  echo "GRPO_TRAIN_BATCH_SIZE 必须 >= GRPO_PPO_MINI_BATCH_SIZE" >&2
  exit 1
fi
if (( (GRPO_TRAIN_BATCH_SIZE * GRPO_ROLLOUT_N) % NPROC_PER_NODE != 0 )); then
  echo "(train_batch * rollout_n) 必须能被 GPU 数整除。" >&2
  echo "当前: train_batch=${GRPO_TRAIN_BATCH_SIZE}, rollout_n=${GRPO_ROLLOUT_N}, gpus=${NPROC_PER_NODE}" >&2
  exit 1
fi

echo "GPUs=${CUDA_VISIBLE_DEVICES} train_batch=${GRPO_TRAIN_BATCH_SIZE} rollout_n=${GRPO_ROLLOUT_N} ppo_mini_batch=${GRPO_PPO_MINI_BATCH_SIZE} actor_lr=${GRPO_ACTOR_LR} kl=${GRPO_KL_LOSS_COEF} epochs=${GRPO_TOTAL_EPOCHS} model_dtype=${GRPO_MODEL_DTYPE} conda=${CONDA_ENV:-<none>} train=${GRPO_TRAIN_FILE} val=${GRPO_VAL_FILE} model=${MODEL_PATH} ckpt=${CKPT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${_py[@]}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${GRPO_TRAIN_FILE}" \
  data.val_files="${GRPO_VAL_FILE}" \
  data.train_batch_size="${GRPO_TRAIN_BATCH_SIZE}" \
  data.dataloader_num_workers="${GRPO_DATALOADER_WORKERS}" \
  data.max_prompt_length="${GRPO_MAX_PROMPT_LENGTH}" \
  data.max_response_length="${GRPO_MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${GRPO_MODEL_DTYPE}" \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile="${GRPO_TORCH_COMPILE}" \
  actor_rollout_ref.ref.fsdp_config.model_dtype="${GRPO_MODEL_DTYPE}" \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile="${GRPO_TORCH_COMPILE}" \
  actor_rollout_ref.actor.optim.lr="${GRPO_ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${GRPO_PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${GRPO_KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_adapter_path=null \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GRPO_ROLLOUT_GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n="${GRPO_ROLLOUT_N}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${GRPO_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_model_len="${GRPO_ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  custom_reward_function.path=scripts/verl_llm_selection_reward.py \
  custom_reward_function.name=compute_score \
  trainer.logger="${TRAINER_LOGGERS}" \
  trainer.project_name="${TRAINER_PROJECT_NAME}" \
  trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.resume_mode="${GRPO_RESUME_MODE}" \
  "hydra.run.dir=${HYDRA_ROOT}/"'${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  "hydra.sweep.dir=${HYDRA_ROOT}/multirun/"'${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq="${GRPO_SAVE_FREQ}" \
  trainer.test_freq="${GRPO_TEST_FREQ}" \
  trainer.total_epochs="${GRPO_TOTAL_EPOCHS}"
