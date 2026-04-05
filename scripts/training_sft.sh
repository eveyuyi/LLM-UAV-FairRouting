#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------- 只改这里 ----------
CONDA_ENV=verl
# llm3_medium_5min_v1 推荐按 seed 划分：8 train / 2 val / 2 test（test 仅保留用于离线评估）
SFT_TRAIN_FILE=data/train/verl/llm3_medium_5min_v1_sft_train.parquet
SFT_VAL_FILE=data/train/verl/llm3_medium_5min_v1_sft_val.parquet
MODEL_PATH=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/model--Qwen-Qwen3-4B-Instruct-2507
# /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models-Qwen-Qwen3-1.7B
# "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/model--Qwen-Qwen3-4B-Instruct-2507"
# "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3.5-9B"
AUTO_EXPORT_SFT=1
SFT_EXPORT_TRAIN_GLOB=data/train/llm3_medium_5min_v1/seed_410[1-8]
SFT_EXPORT_VAL_GLOB="data/train/llm3_medium_5min_v1/seed_4109 data/train/llm3_medium_5min_v1/seed_4110"
SFT_EXPORT_TEST_GLOB="data/train/llm3_medium_5min_v1/seed_4111 data/train/llm3_medium_5min_v1/seed_4112"
SFT_EXPORT_SEED=42
# llm3_medium_5min_v1（clean+pipeline）约 10368 条；8 train 约 6912，2 val 约 1728，适当增大全局 batch。
CKPT_DIR=data/checkpoints/llm3_sft_medium_v1
HYDRA_ROOT=data/hydra_outputs
TRAINER_PROJECT_NAME=llm3-sft
TRAINER_EXPERIMENT_NAME=qwen-sft-llm3-medium-v1
TRAINER_LOGGERS="${TRAINER_LOGGERS:-[\"console\",\"tensorboard\"]}"
# 8 卡下全局 128（每卡 16）对约 6.9k train 样本更合适，单 epoch 约 54 step。
SFT_GLOBAL_BATCH_SIZE="${SFT_GLOBAL_BATCH_SIZE:-128}"
# verl FSDP 默认 engine.model_dtype=fp32：8 rank 各自整模 fp32 加载，CPU/GPU 峰值极高，易被 OOM killer SIGKILL(-9)。应用 bf16 加载。
# 仍 OOM：减小 CUDA_VISIBLE_DEVICES 并发、或换更大 CPU 内存节点（每 rank 仍会各自加载一份权重到本进程内存）。
SFT_MODEL_DTYPE="${SFT_MODEL_DTYPE:-bfloat16}"
SFT_TORCH_COMPILE="${SFT_TORCH_COMPILE:-false}"
# DataLoader worker 数（默认 8）；8 卡 × 8 worker 易占满 CPU 内存。
SFT_DATALOADER_WORKERS="${SFT_DATALOADER_WORKERS:-4}"
# verl 默认 resume_mode=auto：若 CKPT_DIR 里已有上一段训练的 FSDP+LoRA，会尝试加载；基座或结构不同会报 shape mismatch（如 hidden 2048 vs 2560）。
# 续训同一模型同一目录：export SFT_RESUME_MODE=auto
SFT_RESUME_MODE="${SFT_RESUME_MODE:-disable}"
# CONDA_ENV 留空则用当前环境的 python

if [[ -n "${CONDA_ENV}" ]]; then
  _py=(conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python)
else
  _py=(env PYTHONNOUSERSITE=1 python)
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#_GPU_IDS[@]}"
else
  NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
(( NPROC_PER_NODE >= 1 )) || { echo "no GPU"; exit 1; }

if [[ ! -f "${SFT_TRAIN_FILE}" || ! -f "${SFT_VAL_FILE}" ]]; then
  [[ "${AUTO_EXPORT_SFT}" == 1 ]] || { echo "missing parquet"; exit 1; }
  shopt -s nullglob
  TRAIN_INPUT_DIRS=(${SFT_EXPORT_TRAIN_GLOB})
  VAL_INPUT_DIRS=(${SFT_EXPORT_VAL_GLOB})
  TEST_INPUT_DIRS=(${SFT_EXPORT_TEST_GLOB})
  shopt -u nullglob
  ((${#TRAIN_INPUT_DIRS[@]})) || { echo "no train dirs: ${SFT_EXPORT_TRAIN_GLOB}"; exit 1; }
  ((${#VAL_INPUT_DIRS[@]})) || { echo "no val dirs: ${SFT_EXPORT_VAL_GLOB}"; exit 1; }
  ((${#TEST_INPUT_DIRS[@]})) || { echo "no test dirs: ${SFT_EXPORT_TEST_GLOB}"; exit 1; }
  mkdir -p "$(dirname "${SFT_TRAIN_FILE}")" "$(dirname "${SFT_VAL_FILE}")"
  train_unused_val="${SFT_TRAIN_FILE%.parquet}.unused_val.parquet"
  val_unused_val="${SFT_VAL_FILE%.parquet}.unused_val.parquet"
  exp_train=("${_py[@]}" scripts/export_llm3_to_verl_sft.py)
  for d in "${TRAIN_INPUT_DIRS[@]}"; do exp_train+=(--input-dir "${d}"); done
  exp_train+=(--train-out "${SFT_TRAIN_FILE}" --val-out "${train_unused_val}" --val-ratio 0 --seed "${SFT_EXPORT_SEED}")
  "${exp_train[@]}"
  exp_val=("${_py[@]}" scripts/export_llm3_to_verl_sft.py)
  for d in "${VAL_INPUT_DIRS[@]}"; do exp_val+=(--input-dir "${d}"); done
  exp_val+=(--train-out "${SFT_VAL_FILE}" --val-out "${val_unused_val}" --val-ratio 0 --seed "${SFT_EXPORT_SEED}")
  "${exp_val[@]}"
  rm -f "${train_unused_val}" "${val_unused_val}"
  echo "reserved test dirs (${#TEST_INPUT_DIRS[@]}): ${TEST_INPUT_DIRS[*]}"
fi

echo "GPUs=${NPROC_PER_NODE} global_batch=${SFT_GLOBAL_BATCH_SIZE} resume=${SFT_RESUME_MODE} model_dtype=${SFT_MODEL_DTYPE} torch_compile=${SFT_TORCH_COMPILE} dataloader_workers=${SFT_DATALOADER_WORKERS} conda=${CONDA_ENV:-<none>} train=${SFT_TRAIN_FILE} model=${MODEL_PATH}"

"${_py[@]}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${SFT_TRAIN_FILE}" \
  data.val_files="${SFT_VAL_FILE}" \
  data.messages_key=messages \
  data.train_batch_size="${SFT_GLOBAL_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu=1 \
  data.num_workers="${SFT_DATALOADER_WORKERS}" \
  engine.model_dtype="${SFT_MODEL_DTYPE}" \
  engine.use_torch_compile="${SFT_TORCH_COMPILE}" \
  model.path="${MODEL_PATH}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.project_name="${TRAINER_PROJECT_NAME}" \
  trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}" \
  trainer.logger="${TRAINER_LOGGERS}" \
  trainer.resume_mode="${SFT_RESUME_MODE}" \
  trainer.total_epochs=1 \
  model.lora_rank=32 \
  model.lora_alpha=16 \
  model.target_modules=all-linear \
  +model.override_config.attn_implementation=sdpa \
  data.ignore_input_ids_mismatch=true \
  data.enable_thinking_default=false \
  data.max_length=4096 \
  data.truncation=right \
  "hydra.run.dir=${HYDRA_ROOT}/"'${now:%Y-%m-%d}/${now:%H-%M-%S}'
