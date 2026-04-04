# 训练备忘（个人备份）

本仓库 SFT / GRPO 训练走 VERL，入口脚本在 `scripts/`。以下为本人复现时最常用的步骤与可调参数位置。

## 1. 环境

- **CUDA 12.8**：用 `env/` 里的一套（PyTorch cu128 wheel + lock 依赖）。
- 推荐一键安装（默认环境名 `verl310_grpo_stable`，可通过第一个参数改名）：

```bash
bash env/install_verl_grpo.sh
# 或指定名字：bash env/install_verl_grpo.sh my_verl_env
```

- 相关文件：
  - `env/install_verl_grpo.sh`：创建 conda 环境并装 torch 2.9 + `env/verl-grpo-requirements-lock.txt`
  - `env/verl-grpo-conda.yaml`：说明性 yaml（pip 块对 index 支持有限，实际以 install 脚本为准）

装好后，把下面训练脚本里的 **`CONDA_ENV`** 改成你创建的环境名（脚本里若写 `verl` 而实际叫 `verl310_grpo_stable`，需一致）。

- **`CONDA_ENV` 留空**：使用当前 shell 已激活的 Python，不经过 `conda run`。

## 2. 选模型

- **SFT**：`MODEL_PATH` 指向基座 HuggingFace 模型目录（本地路径或已缓存的 hub 布局均可）。
- **GRPO**：`MODEL_PATH` 一般为 **SFT 合并后的 HF 目录**（与 SFT 的 base 不同）；可用 `scripts/export_sft_ckpt_to_hf.sh` 把 VERL SFT checkpoint 导出成 HF，再填到 `training_grpo.sh`。

## 3. 改脚本参数

只改每个脚本顶部「只改这里」区块即可。

### `scripts/training_sft.sh`

| 变量 | 含义 |
|------|------|
| `CONDA_ENV` | conda 环境名 |
| `MODEL_PATH` | 基座模型 |
| `SFT_TRAIN_FILE` / `SFT_VAL_FILE` | 训练/验证 parquet |
| `AUTO_EXPORT_SFT` | 为 1 且 parquet 不存在时，用 `SFT_EXPORT_INPUT_GLOB` 等自动生成 |
| `CKPT_DIR` / `HYDRA_ROOT` | checkpoint 与 Hydra 输出目录 |
| `TRAINER_PROJECT_NAME` / `TRAINER_EXPERIMENT_NAME` | 实验名 |
| `SFT_GLOBAL_BATCH_SIZE` | 全局 batch（可用环境变量覆盖） |

GPU 数：未设 `CUDA_VISIBLE_DEVICES` 时用本机可见 GPU 数量；若只跑部分卡，先 `export CUDA_VISIBLE_DEVICES=0,1,...`。

### `scripts/training_grpo.sh`

| 变量 | 含义 |
|------|------|
| `CONDA_ENV` | 同上 |
| `MODEL_PATH` | 一般为 SFT merge 后的 HF |
| `GRPO_TRAIN_FILE` / `GRPO_VAL_FILE` | GRPO 用 parquet |
| `AUTO_EXPORT_GRPO` | 为 1 且缺 parquet 时从 `GRPO_EXPORT_INPUT_GLOB` 导出 |
| `CKPT_DIR` / `HYDRA_ROOT` | 同上 |
| `GRPO_TRAIN_BATCH_SIZE` / `GRPO_PPO_MINI_BATCH_SIZE` | batch 相关（可用环境变量覆盖） |

## 4. 运行顺序

1. 安装环境并确认 `CONDA_ENV` 与脚本一致。  
2. 配好数据路径（或打开 `AUTO_EXPORT_*` 让脚本自动生成 parquet）。  
3. SFT：`bash scripts/training_sft.sh`  
4. （如需）SFT → HF：`bash scripts/export_sft_ckpt_to_hf.sh <sft_ckpt_dir> <hf_out_dir>`，再把 `training_grpo.sh` 的 `MODEL_PATH` 指到 `<hf_out_dir>`。  
5. GRPO：`bash scripts/training_grpo.sh`

构建原始训练数据见 `scripts/build_training_data.sh` 等（与本文「只记训练链路」分开即可）。

## 5. GRPO checkpoint 转 HuggingFace（用于部署/评测）

`data/checkpoints/llm3_grpo` 是训练输出根目录，不能直接给 vLLM。  
需先把具体 step（如 `global_step_8`）转换为 HF 目录：

```bash
bash scripts/export_sft_ckpt_to_hf.sh \
  data/checkpoints/llm3_grpo/global_step_8 \
  data/checkpoints/llm3_grpo_merged_hf/global_step_8
```

说明：
- 脚本会自动兼容 GRPO 的 `.../global_step_x/actor` 结构。
- 成功后目标目录应包含 `config.json`、tokenizer 文件和模型权重分片。
- 后续部署时，`MODEL_PATH` 请填 **转换后的 HF 目录**，例如：  
  `data/checkpoints/llm3_grpo_merged_hf/global_step_8`

## 6. vLLM 部署（OpenAI 兼容）

仓库提供 `scripts/serve_vllm_model.sh`，用于启动本地 API 服务。

### 6.1 部署训练前模型（pre）

```bash
CONDA_ENV=verl \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/model--Qwen-Qwen3-4B-Instruct-2507 \
SERVED_MODEL_NAME=qwen3-pre \
TENSOR_PARALLEL_SIZE=4 \
GPU_MEMORY_UTILIZATION=0.95 \
MAX_MODEL_LEN=16384 \
PORT=8000 \
bash scripts/serve_vllm_model.sh
```

### 6.2 部署训练后模型（post）

```bash
CONDA_ENV=verl \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MODEL_PATH=data/checkpoints/llm3_grpo_merged_hf/global_step_8 \
SERVED_MODEL_NAME=qwen3-post \
TENSOR_PARALLEL_SIZE=4 \
GPU_MEMORY_UTILIZATION=0.95 \
MAX_MODEL_LEN=16384 \
PORT=8001 \
bash scripts/serve_vllm_model.sh
```

## 7. 训练前后效果分层评测

两个分层脚本：
1) `scripts/eval_pre_post_priority_alignment.sh`（快，跳过 solver，只看排序能力）  
2) `scripts/eval_pre_post_operational_sampled.sh`（慢，分层抽样后看运营效果）

### 7.1 排序能力评测（rank-only）

脚本：`scripts/eval_pre_post_priority_alignment.sh`

特点：
- pre/post 都执行 `--skip-solver`
- 只产出 `priority_alignment` 与 pre/post 的 delta
- 适合大规模窗口快速筛模型

示例：

```bash
CONDA_ENV=verl \
OPENAI_API_KEY=xxx \
PRE_API_BASE=http://127.0.0.1:8000/v1 \
PRE_MODEL=qwen3-pre \
POST_API_BASE=http://127.0.0.1:8001/v1 \
POST_MODEL=qwen3-post \
OUTPUT_ROOT=data/eval_runs/pre_post_rank_only \
TIME_SLOTS_STR="0 1 2 3 4 5 6 7 8 9" \
bash scripts/eval_pre_post_priority_alignment.sh
```

产物（默认在 `data/eval_runs/pre_post_rank_only/evals/`）：
- `pre_alignment.json`
- `post_alignment.json`
- `post_vs_pre_alignment_delta.json`
- `eval_manifest.json`

### 7.2 小样本运营效果评测（stratified sampled）

脚本：`scripts/eval_pre_post_operational_sampled.sh`

特点：
- 在小样本 `time_slot` 上跑 solver（默认 `nsga3_heuristic`）
- 默认自动做分层抽样（低/中/高需求窗口）
- 同时产出排序能力与 `priority_operational_impact`

示例（自动抽样）：

```bash
CONDA_ENV=verl \
OPENAI_API_KEY=xxx \
PRE_API_BASE=http://127.0.0.1:8000/v1 \
PRE_MODEL=qwen3-pre \
POST_API_BASE=http://127.0.0.1:8001/v1 \
POST_MODEL=qwen3-post \
OUTPUT_ROOT=data/eval_runs/pre_post_operational_sampled \
SAMPLE_TOTAL_SLOTS=9 \
SAMPLE_SEED=42 \
SOLVER_BACKEND=nsga3_heuristic \
NSGA3_POP_SIZE=4 \
NSGA3_N_GENERATIONS=2 \
bash scripts/eval_pre_post_operational_sampled.sh
```

示例（手动指定窗口）：

```bash
CONDA_ENV=verl \
OPENAI_API_KEY=xxx \
PRE_API_BASE=http://127.0.0.1:8000/v1 \
PRE_MODEL=qwen3-pre \
POST_API_BASE=http://127.0.0.1:8001/v1 \
POST_MODEL=qwen3-post \
TIME_SLOTS_STR="1 3 7" \
SOLVER_BACKEND=nsga3_heuristic \
bash scripts/eval_pre_post_operational_sampled.sh
```

产物（默认在 `data/eval_runs/pre_post_operational_sampled/evals/`）：
- `pre_alignment.json`
- `post_alignment.json`
- `post_vs_pre_operational_impact.json`
- `slot_sampling.json`（自动抽样时）
- `eval_manifest.json`

### 7.3 核心指标解读

- 排序能力：看 `accuracy`、`macro_f1`、`spearman`、`kendall_tau`、`top_k_hit_rate`
- 高需求倾斜：看 `post_vs_pre_operational_impact.json` 里的：
  - `priority_1_service_rate_gain`
  - `urgent_service_rate_gain`
  - `priority_1_on_time_rate_gain`
  - `priority_weighted_service_gain`
  - `priority_weighted_on_time_gain`

一般这些 gain 为正，且高优先级时延改善时，可支持“训练后更偏向高需要需求”的结论。

## 8. Benchmark 数据建议

- 开发期可先用训练数据做快速 sanity check。
- 正式 benchmark 建议使用独立测试集（按 seed 隔离，避免 train/test 泄漏）。
- 若条件允许，额外生成一批未参与训练的新测试数据，结论更稳健。
