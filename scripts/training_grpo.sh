
# dry run转数据
# PYTHONNOUSERSITE=1 python scripts/export_llm3_to_verl_grpo.py \
#   --input-dir data/train/llm3_smoke/seed_102 \
#   --input-dir data/train/llm3_smoke/seed_103 \
#   --train-out data/train/verl/llm3_grpo_train.parquet \
#   --val-out data/train/verl/llm3_grpo_val.parquet \
#   --val-ratio 0.1 \
#   --dry-run

# 正式转数据

# 自动读取当前可见 GPU 数，避免和 Ray 实际资源不一致
N_GPUS=${N_GPUS:-$(python -c "import torch; print(torch.cuda.device_count())")}
if [ "${N_GPUS}" -lt 1 ]; then
  echo "No visible GPU. Please check CUDA_VISIBLE_DEVICES."
  exit 1
fi

# 训练脚本
PYTHONNOUSERSITE=1 python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=data/train/verl/llm3_grpo_train.parquet \
  data.val_files=data/train/verl/llm3_grpo_val.parquet \
  data.train_batch_size=4 \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=data/checkpoints/llm3_sft_merged_hf/global_step_1 \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  custom_reward_function.path=scripts/verl_llm3_reward.py \
  custom_reward_function.name=compute_score \
  trainer.logger='["console"]' \
  trainer.project_name=llm3-grpo \
  trainer.experiment_name=qwen25-7b-llm3-grpo-smoke \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=10 \
  trainer.total_epochs=1
