# 实验状态与指标覆盖报告

**生成时间**: 2026-05-01  
**当前实验范围**: seed=4111, norm_eval split, time_slots 0-47 (96 windows)

---

## 已完成实验

### Contribution 2 — 主实验 Operational Metrics (seed=4111)

| 方法 | 输出目录 | Windows | 状态 |
|---|---|---|---|
| M0a (random) | `formal_m0a_seed4111/run_20260430_155133_*` | 96 | ✅ |
| M0b (uniform) | `formal_m0b_seed4111/run_20260430_155133_*` | 96 | ✅ |
| M0c (rule-only) | `formal_m0c_seed4111/run_20260430_155133_*` | 96 | ✅ |
| M1_pre (qwen3-base) | `formal_m1_pre_p2_seed4111/run_20260430_155133_*` | 96 | ✅ |
| M1_ft (qwen3-grpo) | `formal_m1_ft_p2_seed4111/run_20260430_155133_*` | 96 | ✅ |
| M1_gemini | `m1_gemini_norm_eval_seed4111/...` | **2** | ❌ 不可用 |

**关键结果** (from `formal_comparison_seed4111.json`):

| 指标 | M0a | M0b | M0c | M1_pre | M1_ft |
|---|---|---|---|---|---|
| Overall service rate | 100% | 100% | 100% | 100% | 100% |
| Overall on-time rate | 35.6% | 37.1% | 29.5% | 32.1% | 31.2% |
| **P1 avg latency (min)** | 261.9 | 249.8 | 29.6 | 62.5 | **13.5** |
| P1 on-time rate | 15.9% | 22.7% | 54.5% | 60.2% | **63.6%** |
| Priority-weighted on-time | 30.4% | 32.2% | 40.9% | 42.3% | **44.6%** |
| Latency gap P1 vs P4 (min) | 1.4 | -10.3 | 350.6 | 283.8 | 368.7 |

### M1 Phase 1 (LLM3 Weight Configs)

| 方法 | 目录 | Weight configs | 状态 |
|---|---|---|---|
| M1_pre seed=4111 | `formal_m1_pre_seed4111/run_20260429_132824_*` | 96 | ✅ |
| M1_ft  seed=4111 | `formal_m1_ft_seed4111/run_20260429_132824_*` | 96 | ✅ |
| M1_pre seed=4111 (full 576w) | `m1_pre_qwen3base_norm_eval_seed4111/...` | 576 | ✅ |
| M1_pre seed=4112 (full 576w) | `m1_pre_qwen3base_norm_eval_seed4112/...` | 576 | ✅ |
| M1_pre seed=5111 (full 576w) | `m1_pre_qwen3base_norm_eval_seed5111/...` | 576 | ✅ |
| M1_pre seed=5112 (full 576w) | `m1_pre_qwen3base_norm_eval_seed5112/...` | 61 | ❌ 不完整 |
| M1_pre seed=5113 (full 576w) | `m1_pre_qwen3base_norm_eval_seed5113/...` | 40 | ❌ 不完整 |
| M1_ft  多 seed | — | — | ❌ 未跑 |

---

## 已计算指标文件

| 文件 | 内容 | 脚本 |
|---|---|---|
| `formal_comparison_seed4111.json` | Contribution 2 全套 operational metrics | `scripts/eval_formal_comparison.py` |
| `extraction_quality_seed4111.json` | Contribution 1 LLM2 字段提取准确率 | `scripts/eval_extraction_quality.py` |
| `priority_alignment_seed4111.json` | Contribution 3 LLM3 优先级对齐指标 | `scripts/eval_priority_alignment_all.py` |
| `efficiency_seed4111.json` | S2 计算效率（solver timing, 路径缓存） | `scripts/eval_efficiency.py` |

---

## Contribution 1 — LLM2 Extraction Quality (seed=4111, n=4340)

**来源**: `llm3_sft_pipeline.jsonl` vs `gold_extraction`

| 字段 | 准确率 |
|---|---|
| origin_fid | **100.0%** |
| dest_fid | **100.0%** |
| deadline_minutes | **99.9%** |
| weight_kg | 99.0% |
| cargo_type | 95.9% |
| children_involved | 86.4% |
| special_handling_match | 83.0% |
| requester_role | 78.2% |
| temperature_sensitive | 75.1% |
| elderly_involved | 61.1% |
| dest_type | 51.0% |
| vulnerable_community | 50.1% |
| demand_tier | 47.7% |
| **Critical signal recall (all 4)** | **28.5%** |
| **Schema validity rate** | **100.0%** |
| Priority chain consistent | 42.1% |

> ⚠️ Critical signal recall = 28.5% 表示只有 28.5% 的需求同时正确提取了所有4个关键字段（deadline, requester_role, elderly_involved, vulnerable_community）。这是一个如实反映的结果，说明 LLM2 在 population vulnerability 字段上存在系统性遗漏。

---

## Contribution 3 — LLM3 Priority Alignment (seed=4111, slots 0-47)

**Ground truth**: `extraction_observable_priority`

| 指标 | M0c (rule) | M1_pre | M1_ft |
|---|---|---|---|
| n aligned | 722 | 661 | 722 |
| **Accuracy** | 1.0000 | 0.4569 | **0.6191** |
| Macro-F1 | 1.0000 | 0.4018 | **0.6201** |
| **P1 Recall** | 1.0000 | 0.5000 | **0.7895** |
| P1 F1 | 1.0000 | 0.3506 | **0.6294** |
| Urgent Recall | 1.0000 | 0.6336 | **0.8505** |
| Urgent F1 | 1.0000 | 0.7186 | **0.8918** |
| **Spearman r** | 1.0000 | 0.5894 | **0.7700** |
| **Kendall τ** | 1.0000 | 0.5169 | **0.6953** |
| **Pairwise acc** | 1.0000 | 0.7448 | **0.8438** |

**M1_pre 多 seed (seeds 4111/4112/5111, full 576w)**:
- accuracy: mean=0.464 (0.448 / 0.435 / 0.510)
- Kendall τ: mean=0.528 (0.537 / 0.511 / 0.535)
- P1 recall: mean=0.614 (0.600 / 0.629 / 0.614)

> ⚠️ M0c accuracy=1.0 是因为 ground truth 用的是 `extraction_observable_priority`（由同一套规则生成），与 M0c 的 rule-only 输出完全一致，属于循环验证。真实的对比意义在于 M1_ft vs M1_pre 的提升。

---

## S2 — Computational Efficiency (seed=4111)

| 指标 | M0a | M0b | M0c | M1_pre | M1_ft |
|---|---|---|---|---|---|
| Avg solve time (s/window) | 2.58 | 0.39 | 0.37 | 0.32 | 0.41 |
| Total solver time (min) | 4.1 | 0.6 | 0.6 | 0.5 | 0.7 |
| Path cache hit rate | 99.9% | 99.9% | 100.0% | 100.0% | 100.0% |

> ⚠️ LLM2/LLM3 模块延迟未记录（当前跑 offline 模式）。需另行在线跑一个 seed 以获取完整 S2 数据。

**路径缓存**: 截至 2026-05-01 共 122,070 条缓存条目（含所有已跑实验积累）。

---

## ❌ 缺失实验清单（需补充）

### P0 — 最高优先级（影响论文核心结论可信度）

| 实验 | 描述 | 预估工作量 |
|---|---|---|
| **多 seed operational runs** | M0a/b/c + M1_pre/ft 在 seeds 4112/5111/5112/5113 上的 NSGA-III 全量跑（96 windows each） | 5方法 × 4seed × ~3h = ~60h 计算 |
| **M1_gemini 全量** | Gemini 在 seeds 4111-5113 上的完整 96 窗口实验（目前每 seed 仅 2 窗口） | 需 Vertex AI 配置 |

### P1 — 高优先级

| 实验 | 描述 |
|---|---|
| **Hard eval LLM3 outputs** | 在 43 个 hard 窗口上跑 M1_pre/ft Phase 1 + Phase 2，获取 hard eval 对齐指标 |
| **M1_ft 多 seed Phase 1** | M1_ft 在 seeds 4112/5111/5112/5113 上的 LLM3 优先级输出（Phase 1）|
| **P2: SFT-only baseline** | 使用 SFT-only checkpoint 做优先级对齐，与 SFT+GRPO 对比 |

### P2 — 中优先级

| 实验 | 描述 |
|---|---|
| **P3: GPT-4o / frontier LLM** | GPT-4o zero-shot 优先级排序 vs 微调 Qwen3-4B |
| **OR1: NSGA-III vs CPLEX** | 小规模求解器效率对比 |
| **S2: LLM 模块延迟** | 在线模式跑一个 seed，记录 LLM2/LLM3 每需求延迟 |

---

## 关键注意事项

1. **M0c accuracy=1.0 的循环验证问题**：`extraction_observable_priority` 由规则生成，与 M0c rule-only 输出相同。Priority alignment 的真实对比应与 `latent_priority` 或 human annotation 比较。当前数据可用 `latent_priority` 作替代 ground truth 重新计算。

2. **单 seed 统计局限**：当前所有主实验结论基于 seed=4111 单组，无法报置信区间。多 seed 实验是论文发表的最低要求。

3. **服务率 100% 的解释**：所有方法 service_rate=100% 说明求解器最终都能服务所有需求，差异体现在**何时**（latency/on-time rate）而非**是否**服务——这正是优先级的价值所在，需在论文中明确说明。

---

## S3 Case Study 补充（2026-05-01）

**脚本**: `scripts/analyze_s3_case_study.py`  
**输出**: `evals/results/s3_case_study_m1ft_vs_m0c.md`  
**比较**: M0c (rule) vs M1_ft，ground truth = `latent_priority`

**关键发现**：
- Rule 低估紧急需求（GT≤P2, rule≥P3, LLM≤P2）：**0 cases**（M0c 不漏掉高优先级）
- Rule 过度优先非紧急需求（GT≥P3, rule≤P2, LLM≥P3）：**41 cases**（LLM 全部纠正）
- 选取 5 个典型案例写入 case study（gap≥1，无 urgent filter）

**论文叙事**：规则系统基于关键词触发（"OTC medication" + "vulnerable population" + "120min deadline"）对普通日用品配送过度赋予 P2，LLM 通过对话语义理解正确识别为 P3/P4，避免占用紧急配送资源。

## Hard Eval 任务状态（2026-05-01 提交）

| Job | 内容 | 状态 |
|---|---|---|
| 15294610 | Hard eval M1_pre + M1_ft Phase 1（43 windows, GPU） | PENDING |
| 15294611 | Hard eval Phase 2 solver（depend on 15294610） | PENDING |

完成后运行：
```bash
PYTHONPATH=src python scripts/eval_priority_alignment_all.py \
  --seed 4111 \
  --hard-eval-wc data/eval_runs/hard_eval_m1_{pre,ft}_seed4111/run_*/weight_configs \
  --output evals/results/priority_alignment_hard_seed4111.json
```
（需先扩展 eval_priority_alignment_all.py 支持 hard eval 路径参数）

---

## Contribution 1 根因分析补充（2026-05-01）

详细分析见 `evals/results/contribution1_analysis.md`。

### Critical signal recall = 28.5% 的解释

**不是 LLM2 提取能力差，而是评估设计混淆了两类字段：**

- LLM2 可见字段（来自对话）：origin/dest/deadline/requester_role → 78–100% 准确
- 需外部数据库填充字段（地理人口统计）：elderly_involved/vulnerable_community → 对话中根本不存在

gold_extraction 的 elderly_involved=True 来自目的地区域 elderly_ratio（如 0.31），对话中完全未提及。
**修正后的 critical signal recall（仅对话可见字段 deadline + requester_role）= 78.1%**

### Priority chain consistency = 42.1% 的解释

瓶颈在第二步（dialogue_observable → extraction_observable = 43.0%），第一步（latent → dialogue = 88.5%）质量良好。

**LLM2 存在系统性过度优先化偏差**（safety bias）：
- 最大失败模式 (4,4,3)：1,131 例，对话正确表达 P4 但 LLM2 升为 P3
- 合计过度升级 1,762 例（41% of all demands）
- LLM2 对"医疗/家庭/弱势"关键词敏感，倾向保守地提高优先级
- 这正是 LLM3 存在的意义：对 LLM2 过度优先化做判别性排序纠正

### TODO

- **Pipeline latency (LLM2/LLM3)**：需在线模式跑一次（起 vLLM server），当前 offline 无法计时
