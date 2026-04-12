#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SFT_GLOBAL_STEP=594
export MODEL_PATH=data/checkpoints/expA_sft_baseline/global_step_${SFT_GLOBAL_STEP}/huggingface_lora_merged
export SFT_CKPT_DIR=data/checkpoints/expA_sft_baseline/global_step_${SFT_GLOBAL_STEP}
export GRPO_TRAIN_FILE=data/train/verl/expC_grpo_mixed_train.parquet
export GRPO_VAL_FILE=data/train/verl/expC_grpo_mixed_val.parquet
export AUTO_EXPORT_GRPO=0
export FORCE_REEXPORT_GRPO=0
export GRPO_ROLLOUT_N=8
export GRPO_ACTOR_LR=5e-7
export GRPO_TOTAL_EPOCHS=200
export GRPO_TRAIN_BATCH_SIZE=16
export GRPO_PPO_MINI_BATCH_SIZE=8
export GRPO_KL_LOSS_COEF=0.005
export GRPO_SAVE_FREQ=1000
export GRPO_TEST_FREQ=1000
export CKPT_DIR=data/checkpoints/expC_grpo_mixed_fake
export TRAINER_EXPERIMENT_NAME=expC-grpo-mixed_fake
export FAIL_ON_EXISTING_CKPT_DIR=0

bash scripts/training_grpo.sh
