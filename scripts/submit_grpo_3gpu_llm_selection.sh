#!/usr/bin/env bash
# =============================================================================
# SLURM 提交脚本：GRPO 训练（llm_selection，3 × A100 GPU）
# 用法：
#   sbatch scripts/submit_grpo_3gpu_llm_selection.sh
#
# 可在提交时覆盖任意环境变量，例如：
#   MODEL_PATH=/path/to/model sbatch scripts/submit_grpo_3gpu_llm_selection.sh
# =============================================================================

# ---------- SLURM 资源申请 ----------
#SBATCH -J grpo_llm_sel_3gpu          # 作业名称
#SBATCH -p nvidia                     # GPU 分区
#SBATCH --gres=gpu:a100:3             # 3 张 A100（40 GB）
#SBATCH -c 24                         # 每 task 24 个 CPU（每卡 ~8 核，支持 dataloader 多进程）
#SBATCH -t 3-00:00:00                 # 最长 3 天（nvidia 分区上限 4 天）
#SBATCH -o logs/grpo_llm_sel_%j.out   # 标准输出（含 stderr）
#SBATCH --mail-type=END,FAIL          # 结束或失败时发邮件
# #SBATCH --mail-user=your@email.com  # 取消注释并填写邮箱

# ---------- 环境初始化 ----------
set -euo pipefail

# 切换到项目根目录（脚本位于 scripts/ 子目录下）
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 建立日志目录（若不存在）
mkdir -p logs

echo "=============================="
echo "SLURM Job ID : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Start        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Work dir     : $(pwd)"
echo "=============================="

# ---------- 模块加载 ----------
module purge
module load cuda/12.1          # 按实际集群模块名修改

# ---------- Conda 环境 ----------
# 若 conda 未自动初始化，需要先 source conda.sh
# 典型路径（按实际安装位置修改）：
# source /opt/miniconda3/etc/profile.d/conda.sh
# source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate verl

# ---------- GPU 设置 ----------
# SLURM 会自动把分配到的 GPU 映射到 CUDA_VISIBLE_DEVICES 0,1,2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

# ---------- 透传给训练脚本的参数（均可在 sbatch 前以环境变量覆盖）----------
export CONDA_ENV="verl"
export CUDA_VISIBLE_DEVICES

# 数据路径（保持与 training_grpo_3gpu_llm_selection.sh 默认值一致）
# export RAW_JSONL="data/train/llm_selection_training_jsonl_1000_qs_v1_20260413/llm_selection_train_1000_qs_v1.jsonl"
# export COMPACT_DIR="data/train/llm_selection_training_jsonl_1000_qs_v1_20260413/compact_exports_4b_short"

# 若已有 SFT merged 模型，可直接指定；否则脚本自动从 SFT checkpoint 目录推断
# export MODEL_PATH="/path/to/sft_merged_hf_model"

# 训练超参（如无需修改，注释掉即可使用训练脚本内的默认值）
# export GRPO_TOTAL_EPOCHS=2
# export GRPO_TRAIN_BATCH_SIZE=12
# export GRPO_PPO_MINI_BATCH_SIZE=6
# export GRPO_ROLLOUT_N=4
# export GRPO_ACTOR_LR=8e-7
# export GRPO_SAVE_FREQ=50
# export GRPO_TEST_FREQ=25

# ---------- 执行训练脚本 ----------
echo "Launching training script ..."
bash scripts/training_grpo_3gpu_llm_selection.sh

echo "=============================="
echo "End : $(date '+%Y-%m-%d %H:%M:%S')"
echo "DONE"
echo "=============================="
