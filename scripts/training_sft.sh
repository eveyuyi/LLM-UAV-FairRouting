# 先做一轮dry-run，看样本数

# python scripts/export_llm3_to_verl_sft.py \
#     --input-dir data/train/llm3_smoke/seed_102 \
#     --input-dir data/train/llm3_smoke/seed_103 \
#     --train-out data/train/verl/llm3_sft_train.parquet \
#     --val-out data/train/verl/llm3_sft_val.parquet \
#     --val-ratio 0.1 \
#     --dry-run
# 然后正式转数据格式

# 然后训练

PYTHONNOUSERSITE=1 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m verl.trainer.sft_trainer \
  data.train_files=data/train/verl/llm3_sft_train.parquet \
  data.val_files=data/train/verl/llm3_sft_val.parquet \
  data.messages_key=messages \
  data.micro_batch_size_per_gpu=1 \
  model.path=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models-Qwen-Qwen3-1.7B \
  trainer.default_local_dir=data/checkpoints/llm3_sft \
  trainer.project_name=llm3-sft \
  trainer.experiment_name=qwen25-7b-llm3-sft-smoke \
  trainer.logger='["console"]' \
  trainer.total_epochs=1 \
  model.lora_rank=32 \
  model.lora_alpha=16 \
  model.target_modules=all-linear \
  +model.override_config.attn_implementation=sdpa \
  data.ignore_input_ids_mismatch=true \
  data.enable_thinking_default=false \
  data.max_length=4096 \
  data.truncation=right \
  'hydra.run.dir=data/hydra_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'