#!/usr/bin/env bash
#
# 用法：
#   bash scripts/training_grpo.sh
# 先改下面「只改这里」：CONDA_ENV、GRPO parquet 路径、MODEL_PATH（必须是 HuggingFace 目录，含
# config.json）、SFT_CKPT_DIR（FSDP SFT 检查点，用于自动 merge）等。
#
# 模型路径：actor_rollout_ref.model.path 只接受 HF 格式。若 MODEL_PATH 下尚无 config.json 且
# AUTO_MERGE_SFT_HF=1，会调用 scripts/export_sft_ckpt_to_hf.sh，把 SFT_CKPT_DIR merge 到 MODEL_PATH。
#
# 数据：若 GRPO_TRAIN_FILE / GRPO_VAL_FILE 不存在且 AUTO_EXPORT_GRPO=1，会从 GRPO_EXPORT_INPUT_GLOB
# 下各目录读取 llm3_grpo_hard.jsonl，调用 scripts/export_llm3_to_verl_grpo.py 生成 parquet。
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
CONDA_ENV=verl
GRPO_TRAIN_FILE=data/train/verl/quality_pilot_grpo_train.parquet
GRPO_VAL_FILE=data/train/verl/quality_pilot_grpo_val.parquet
# GRPO 的 actor_rollout_ref.model.path 必须是 HuggingFace 目录（含 config.json 等）
MODEL_PATH=data/checkpoints/llm3_sft_merged_hf/global_step_8
# 若上面目录尚不存在，从 VERL FSDP SFT 检查点自动 merge（调用 scripts/export_sft_ckpt_to_hf.sh）
SFT_CKPT_DIR=data/checkpoints/llm3_sft/global_step_8
AUTO_MERGE_SFT_HF=1
AUTO_EXPORT_GRPO=1
GRPO_EXPORT_INPUT_GLOB=data/train/quality_pilot/seed_*
GRPO_EXPORT_VAL_RATIO=0.1
GRPO_EXPORT_SEED=42
CKPT_DIR=data/checkpoints/llm3_grpo
HYDRA_ROOT=data/hydra_outputs
TRAINER_PROJECT_NAME=llm3-grpo
TRAINER_EXPERIMENT_NAME=qwen-grpo
# 与 quality_pilot 规模（数百条）相比默认 1024 过大；4B+vLLM 也不宜一次过大。
# 8 卡、rollout.n=2 时：real_batch = train_batch * 2，须被 8 整除 → train_batch 取 4/8/12…
GRPO_TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-8}"
GRPO_PPO_MINI_BATCH_SIZE="${GRPO_PPO_MINI_BATCH_SIZE:-4}"
GRPO_ROLLOUT_N="${GRPO_ROLLOUT_N:-2}"
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
# CONDA_ENV 留空则用当前环境的 python

if [[ -n "${CONDA_ENV}" ]]; then
  _py=(conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python)
else
  _py=(env PYTHONNOUSERSITE=1 python)
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  if [[ "${AUTO_MERGE_SFT_HF}" != 1 ]]; then
    echo "GRPO 需要 HuggingFace 权重目录（缺少 ${MODEL_PATH}/config.json）。请先运行 scripts/export_sft_ckpt_to_hf.sh，或设 AUTO_MERGE_SFT_HF=1。" >&2
    exit 1
  fi
  if [[ ! -d "${SFT_CKPT_DIR}" ]]; then
    echo "用于 merge 的 SFT 检查点不存在: ${SFT_CKPT_DIR}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${MODEL_PATH}")"
  if [[ -n "${CONDA_ENV}" ]]; then
    conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 bash scripts/export_sft_ckpt_to_hf.sh "${SFT_CKPT_DIR}" "${MODEL_PATH}"
  else
    env PYTHONNOUSERSITE=1 bash scripts/export_sft_ckpt_to_hf.sh "${SFT_CKPT_DIR}" "${MODEL_PATH}"
  fi
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#_GPU_IDS[@]}"
else
  NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
(( NPROC_PER_NODE >= 1 )) || { echo "no GPU"; exit 1; }

if [[ ! -f "${GRPO_TRAIN_FILE}" || ! -f "${GRPO_VAL_FILE}" ]]; then
  [[ "${AUTO_EXPORT_GRPO}" == 1 ]] || { echo "missing parquet"; exit 1; }
  shopt -s nullglob
  INPUT_DIRS=(${GRPO_EXPORT_INPUT_GLOB})
  shopt -u nullglob
  ((${#INPUT_DIRS[@]})) || { echo "no dirs: ${GRPO_EXPORT_INPUT_GLOB}"; exit 1; }
  mkdir -p "$(dirname "${GRPO_TRAIN_FILE}")" "$(dirname "${GRPO_VAL_FILE}")"
  exp=("${_py[@]}" scripts/export_llm3_to_verl_grpo.py)
  for d in "${INPUT_DIRS[@]}"; do exp+=(--input-dir "${d}"); done
  exp+=(--train-out "${GRPO_TRAIN_FILE}" --val-out "${GRPO_VAL_FILE}" --val-ratio "${GRPO_EXPORT_VAL_RATIO}" --seed "${GRPO_EXPORT_SEED}")
  "${exp[@]}"
fi

echo "GPUs=${NPROC_PER_NODE} train_batch=${GRPO_TRAIN_BATCH_SIZE} rollout_n=${GRPO_ROLLOUT_N} ppo_mini_batch=${GRPO_PPO_MINI_BATCH_SIZE} resume=${GRPO_RESUME_MODE} model_dtype=${GRPO_MODEL_DTYPE} dataloader_workers=${GRPO_DATALOADER_WORKERS} max_model_len=${GRPO_ROLLOUT_MAX_MODEL_LEN} rollout_gpu_mem=${GRPO_ROLLOUT_GPU_MEM_UTIL} conda=${CONDA_ENV:-<none>} train=${GRPO_TRAIN_FILE} model=${MODEL_PATH}"

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
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${GRPO_PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
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
  trainer.logger='["console"]' \
  trainer.project_name="${TRAINER_PROJECT_NAME}" \
  trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.resume_mode="${GRPO_RESUME_MODE}" \
  "hydra.run.dir=${HYDRA_ROOT}/"'${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  "hydra.sweep.dir=${HYDRA_ROOT}/multirun/"'${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=10 \
  trainer.total_epochs=1
