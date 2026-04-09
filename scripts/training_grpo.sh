#!/usr/bin/env bash
#
# 用法：
#   bash scripts/training_grpo.sh
# 先改下面「只改这里」：CONDA_ENV、GRPO parquet 路径、MODEL_PATH（必须是 HuggingFace 目录，含
# config.json）等。
#
# 模型路径：actor_rollout_ref.model.path 只接受 HF 格式。默认使用已合并好的 SFT 权重目录。
#
# 数据：若 GRPO_TRAIN_FILE / GRPO_VAL_FILE 不存在且 AUTO_EXPORT_GRPO=1，会按
# GRPO_EXPORT_TRAIN_GLOB 与 GRPO_EXPORT_VAL_GLOB 分开导出 parquet；GRPO_EXPORT_TEST_GLOB 仅预留评估集。
#
# 可选环境变量（示例）：
#   GRPO_TRAIN_BATCH_SIZE / GRPO_PPO_MINI_BATCH_SIZE — 须满足 train_batch >= mini_batch；
#     且 (GRPO_TRAIN_BATCH_SIZE * GRPO_ROLLOUT_N) % N_GPUS == 0（verl 静态 batch 校验）。
#   GRPO_ROLLOUT_N — 每 prompt 采样条数，默认与 actor_rollout_ref.rollout.n 一致。
#   GRPO_RESUME_MODE — auto | disable | resume_path；换基座或清实验时建议 disable。
#   GRPO_MODEL_DTYPE — actor/ref FSDP 加载 dtype，默认 bfloat16，减轻多进程 fp32 加载内存峰值。
#   GRPO_DATALOADER_WORKERS — data.dataloader_num_workers，默认 2。
#   GRPO_MAX_PROMPT_LENGTH / GRPO_MAX_RESPONSE_LENGTH — 与数据、显存权衡。
#
# 典型顺序：training_sft.sh → training_grpo.sh。
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------- 只改这里 ----------
CONDA_ENV="${CONDA_ENV:-verl}"
GRPO_TRAIN_FILE="${GRPO_TRAIN_FILE:-data/train/verl/expB_grpo_hard_train.parquet}"
GRPO_VAL_FILE="${GRPO_VAL_FILE:-data/train/verl/expB_grpo_hard_val.parquet}"
# GRPO 的 actor_rollout_ref.model.path 必须是 HuggingFace 目录（含 config.json 等）
# Exp B 默认沿用 Exp A 的 SFT 最终 step，并直接使用已 merge 的 HF 权重目录。
SFT_GLOBAL_STEP="${SFT_GLOBAL_STEP:-594}"
MODEL_PATH="${MODEL_PATH:-data/checkpoints/expA_sft_baseline/global_step_${SFT_GLOBAL_STEP}/huggingface_lora_merged}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-data/checkpoints/expA_sft_baseline/global_step_${SFT_GLOBAL_STEP}}"
AUTO_EXPORT_GRPO="${AUTO_EXPORT_GRPO:-1}"
FORCE_REEXPORT_GRPO="${FORCE_REEXPORT_GRPO:-1}"
GRPO_EXPORT_TRAIN_GLOB="${GRPO_EXPORT_TRAIN_GLOB:-data/train/llm3_medium_5min_v1/seed_410[1-8] data/train/llm3_5min_large_v1/seed_411[3-9] data/train/llm3_5min_large_v1/seed_4120 data/train/llm3_5min_large_v1/seed_412[5-9] data/train/llm3_5min_large_v1/seed_413[01]}"
GRPO_EXPORT_VAL_GLOB="${GRPO_EXPORT_VAL_GLOB:-data/train/llm3_medium_5min_v1/seed_4109 data/train/llm3_medium_5min_v1/seed_4110 data/train/llm3_5min_large_v1/seed_4132 data/train/llm3_5min_large_v1/seed_4133}"
GRPO_EXPORT_TEST_GLOB="${GRPO_EXPORT_TEST_GLOB:-data/train/llm3_medium_5min_v1/seed_4111 data/train/llm3_medium_5min_v1/seed_4112 data/train/llm3_5min_large_v1/seed_413[4-6]}"
GRPO_EXPORT_SEED="${GRPO_EXPORT_SEED:-42}"
CKPT_DIR="${CKPT_DIR:-data/checkpoints/expB_grpo_hard}"
HYDRA_ROOT="${HYDRA_ROOT:-data/hydra_outputs}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-llm3-grpo}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-expB-grpo-hard}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-[\"console\",\"tensorboard\"]}"
FAIL_ON_EXISTING_CKPT_DIR="${FAIL_ON_EXISTING_CKPT_DIR:-0}"
# Exp B hard 数据量更大：默认 train_batch=16、rollout.n=8、ppo mini=8，兼顾探索与稳定性。
# 仍需满足 (train_batch * rollout_n) % N_GPUS == 0，且 train_batch >= ppo_mini_batch。
GRPO_TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-16}"
GRPO_PPO_MINI_BATCH_SIZE="${GRPO_PPO_MINI_BATCH_SIZE:-8}"
GRPO_ROLLOUT_N="${GRPO_ROLLOUT_N:-8}"
GRPO_ACTOR_LR="${GRPO_ACTOR_LR:-5e-7}"
# 默认不自动恢复，避免 CKPT_DIR 里旧实验与当前合并后 HF 权重不兼容。
GRPO_RESUME_MODE="${GRPO_RESUME_MODE:-disable}"
GRPO_MODEL_DTYPE="${GRPO_MODEL_DTYPE:-bfloat16}"
GRPO_TORCH_COMPILE="${GRPO_TORCH_COMPILE:-false}"
GRPO_DATALOADER_WORKERS="${GRPO_DATALOADER_WORKERS:-2}"
GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-2048}"
GRPO_MAX_RESPONSE_LENGTH="${GRPO_MAX_RESPONSE_LENGTH:-512}"
# 按每步 token 上界粗调 vLLM 批 token（prompt+response 量级）；过长可再加大或提高 rollout.gpu_memory_utilization
GRPO_MAX_NUM_BATCHED_TOKENS="${GRPO_MAX_NUM_BATCHED_TOKENS:-8192}"
# vLLM 默认会按模型 config 的超长上下文建 KV cache；对 RL 任务应显式设 rollout.max_model_len，
# 否则像 Qwen3 的 262144 会直接把 KV cache 撑爆（4 卡更容易）。
GRPO_ROLLOUT_MAX_MODEL_LEN="${GRPO_ROLLOUT_MAX_MODEL_LEN:-4096}"
GRPO_ROLLOUT_GPU_MEM_UTIL="${GRPO_ROLLOUT_GPU_MEM_UTIL:-0.5}"
GRPO_KL_LOSS_COEF="${GRPO_KL_LOSS_COEF:-0.005}"
GRPO_TOTAL_EPOCHS="${GRPO_TOTAL_EPOCHS:-3}"
GRPO_SAVE_FREQ="${GRPO_SAVE_FREQ:-50}"
GRPO_TEST_FREQ="${GRPO_TEST_FREQ:-25}"
# CONDA_ENV 留空则用当前环境的 python

if [[ -n "${CONDA_ENV}" ]]; then
  _py=(conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python)
else
  _py=(env PYTHONNOUSERSITE=1 python)
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "GRPO 需要 HuggingFace 权重目录（缺少 ${MODEL_PATH}/config.json）。当前脚本默认不自动 merge，请确认 MODEL_PATH 指向已合并目录。" >&2
  echo "若要从 FSDP 检查点导出，可手动运行: bash scripts/export_sft_ckpt_to_hf.sh \"${SFT_CKPT_DIR}\" \"${MODEL_PATH}\"" >&2
  exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#_GPU_IDS[@]}"
else
  NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
(( NPROC_PER_NODE >= 1 )) || { echo "no GPU"; exit 1; }

if [[ "${GRPO_RESUME_MODE}" == "disable" && "${FAIL_ON_EXISTING_CKPT_DIR}" == 1 && -d "${CKPT_DIR}" ]]; then
  echo "CKPT_DIR already exists with resume disabled: ${CKPT_DIR}" >&2
  echo "Choose a new CKPT_DIR, or set FAIL_ON_EXISTING_CKPT_DIR=0 / GRPO_RESUME_MODE=auto intentionally." >&2
  exit 1
fi

if [[ "${FORCE_REEXPORT_GRPO}" == 1 || ! -f "${GRPO_TRAIN_FILE}" || ! -f "${GRPO_VAL_FILE}" ]]; then
  [[ "${AUTO_EXPORT_GRPO}" == 1 ]] || { echo "missing parquet"; exit 1; }
  shopt -s nullglob
  TRAIN_INPUT_DIRS=(${GRPO_EXPORT_TRAIN_GLOB})
  VAL_INPUT_DIRS=(${GRPO_EXPORT_VAL_GLOB})
  TEST_INPUT_DIRS=(${GRPO_EXPORT_TEST_GLOB})
  shopt -u nullglob
  ((${#TRAIN_INPUT_DIRS[@]})) || { echo "no train dirs: ${GRPO_EXPORT_TRAIN_GLOB}"; exit 1; }
  ((${#VAL_INPUT_DIRS[@]})) || { echo "no val dirs: ${GRPO_EXPORT_VAL_GLOB}"; exit 1; }
  ((${#TEST_INPUT_DIRS[@]})) || { echo "no test dirs: ${GRPO_EXPORT_TEST_GLOB}"; exit 1; }
  mkdir -p "$(dirname "${GRPO_TRAIN_FILE}")" "$(dirname "${GRPO_VAL_FILE}")"
  train_unused_val="${GRPO_TRAIN_FILE%.parquet}.unused_val.parquet"
  val_unused_val="${GRPO_VAL_FILE%.parquet}.unused_val.parquet"
  exp_train=("${_py[@]}" scripts/export_llm3_to_verl_grpo.py)
  for d in "${TRAIN_INPUT_DIRS[@]}"; do exp_train+=(--input-dir "${d}"); done
  exp_train+=(--train-out "${GRPO_TRAIN_FILE}" --val-out "${train_unused_val}" --val-ratio 0 --seed "${GRPO_EXPORT_SEED}")
  "${exp_train[@]}"
  exp_val=("${_py[@]}" scripts/export_llm3_to_verl_grpo.py)
  for d in "${VAL_INPUT_DIRS[@]}"; do exp_val+=(--input-dir "${d}"); done
  exp_val+=(--train-out "${GRPO_VAL_FILE}" --val-out "${val_unused_val}" --val-ratio 0 --seed "${GRPO_EXPORT_SEED}")
  "${exp_val[@]}"
  rm -f "${train_unused_val}" "${val_unused_val}"
  echo "reserved test dirs (${#TEST_INPUT_DIRS[@]}): ${TEST_INPUT_DIRS[*]}"
fi

echo "GPUs=${NPROC_PER_NODE} train_batch=${GRPO_TRAIN_BATCH_SIZE} rollout_n=${GRPO_ROLLOUT_N} ppo_mini_batch=${GRPO_PPO_MINI_BATCH_SIZE} actor_lr=${GRPO_ACTOR_LR} kl=${GRPO_KL_LOSS_COEF} epochs=${GRPO_TOTAL_EPOCHS} resume=${GRPO_RESUME_MODE} model_dtype=${GRPO_MODEL_DTYPE} dataloader_workers=${GRPO_DATALOADER_WORKERS} max_model_len=${GRPO_ROLLOUT_MAX_MODEL_LEN} rollout_gpu_mem=${GRPO_ROLLOUT_GPU_MEM_UTIL} conda=${CONDA_ENV:-<none>} train=${GRPO_TRAIN_FILE} val=${GRPO_VAL_FILE} model=${MODEL_PATH} ckpt=${CKPT_DIR} force_reexport=${FORCE_REEXPORT_GRPO}"

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
  custom_reward_function.path=scripts/verl_llm3_reward.py \
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
