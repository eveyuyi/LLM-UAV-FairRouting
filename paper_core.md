
# 文章的 3 个 contribution：

## **Contribution 1**
**Beyond Structured Forms: Contextual Information Enrichment**

核心主张：传统表单只记录显式字段，本文利用LLM能力，从自然对话中提取 OR solver 需要的结构化需求，并保留上下文里的关键信号，比如 urgency、requester role、special handling、population vulnerability、receiver readiness。（证明LLM+OR框架的可行性）

对应指标：
- `Field extraction accuracy / F1`：看 `origin / destination / cargo / weight / deadline / requester_role / special_handling / vulnerability / readiness` 是否提取正确。
- `Critical signal recall`：重点看 `deadline`、`requester_role`、`special_handling`、`population_vulnerability` 这些影响 priority 的字段有没有被提出来。
- `Schema validity rate`：LLM2 输出是否符合 solver schema，能否直接进入后端。
- `solver feasible rate`：LLM2 输出进入 OR 后能否生成可行解。
- `dialogue_observable_priority -> extraction_observable_priority consistency`：看 LLM2 提取后，priority 相关信息是否还保留。
- `pipeline latency：LLM2、LLM3、solver` 各模块耗时。


解释：这一组指标证明 LLM2 不是在做泛化摘要，而是在把人类对话里的上下文信息转成可计算、可调度的结构化需求。LLM+OR 能跑通，而且 LLM 提取的信息真的能被 OR 使用。



## **Contribution 2**
**Human-Aligned Priority Learning for Fair Allocation**

核心主张：LLM3 学到的不是表面 urgency，而是基于对话中提取的需求进行 human-aligned priority ranking，把稀缺资源优先分配给更高需要的人。（证明LLM+OR框架的优越性）
核心问题：相比纯 OR 依赖 random / uniform / rule-based priority，LLM-generated priority 是否能带来更好的资源分配结果？

这里主实验：

M0a: Random Priority + NSGA-III
M0b: Uniform Priority + NSGA-III
M0c: Rule-based Priority + NSGA-III
M1: LLM Priority + NSGA-III
M1_pre: Qwen3-4b-base + NSGA-III
M1_ft: Qwen3-ft + NSGA-III
M1_api: gemini-3.1-flash-lite + NSGA-III
这里的主指标应该是 operational metrics：

overall service rate：总服务率，证明整体效率没有牺牲太多。
overall on-time rate：总体准时率。
priority_1 service rate：最高需要需求是否更容易被服务。
priority_1 on-time rate：最高需要需求是否更及时得到服务。
urgent service / on-time rate：priority 1/2 的服务和准时情况。
priority-weighted service score：按 priority 加权后的服务质量。
priority-weighted on-time score：按 priority 加权后的准时质量。
average latency by priority tier：高优先级需求是否平均延迟更低。
vulnerable group service / on-time rate：老人、儿童、脆弱社区相关需求是否得到更公平照顾。（目前似乎还没有体现）
Pareto / objective metrics：如果 NSGA-III 输出 Pareto 解，可以报告 hypervolume、distance、risk、time objective tradeoff。



-------

## 待讨论

**Contribution 3** (待讨论)

### 第一个方向：**Contribution 3: Preference Alignment and Trainable Priority Policy**

你之前那些 LLM3 ranking metrics 仍然很有价值，但我建议放到 Contribution 3 或 5.4，而不是 Contribution 2 主证据。

对应实验：
- `P1`: LLM3 训练前后对比，Normal + Hard
- `P2`: SFT vs SFT+GRPO
- `P3`: GPT-4o zero-shot vs fine-tuned Qwen3-4B

指标：
- `priority accuracy / macro-F1 / weighted-F1`
- `priority_1 recall / F1`
- `urgent recall / F1`
- `top_k_hit_rate`
- `pairwise accuracy`
- `Spearman / Kendall tau`
- `hard subtype metrics`
  - `counterfactual`
  - `surface_contradiction`
  - `near_tie`
  - `vulnerable_population`




### 第二个方向**Observability-Aware Training and Evaluation Pipeline**

核心主张：你的数据生成和训练流程显式区分 `latent_priority`、`dialogue_observable_priority`、`extraction_observable_priority`，避免“标签存在但模型看不到”的训练噪声，并构造 hard cases 来评估模型是否真的学到 priority reasoning。

对应指标：
- `audit pass rate`：LLM1 对话是否真的表达了 must-mention priority signals。
- `observability_score`：对话中 priority 线索的可观察程度。
- `latent_priority -> dialogue_observable_priority agreement`：原始优先级是否能从对话中恢复。
- `dialogue_observable_priority -> extraction_observable_priority agreement`：LLM2 提取后，priority 信息是否仍可恢复。
- `hard case coverage`：`surface_contradiction / near_tie / counterfactual` 的数量和比例。
- `standard vs hard eval gap`：模型在普通测试集和 hard eval 上的性能差距。
- `operational impact metrics`：包括 `priority_1_on_time_rate_gain`、`urgent_on_time_rate_gain`、`priority_weighted_on_time_gain`。

解释：这一贡献强调你的方法不只是“生成一些数据训练模型”，而是建立了一套可审计的数据链路，保证训练标签和模型实际可见信息一致，并用 hard eval 检查模型是否真正具备鲁棒的 priority reasoning。

我会把三者的关系写成一句主线：

**LLM2 enriches structured demands from human dialogue; LLM3 learns human-aligned priority from observable structured needs; the observability-aware data pipeline ensures that training and evaluation are faithful to what each model can actually see.**




# 实验清单


| # | 实验 | 备注 |
|---|------|------|
| M0c | Rule-based Priority + NSGA-III | 主对比 baseline |
| M1 | Frontier LLM Priority + NSGA-III | 主对比方法 |
| P1 | LLM3 训练前后对比（Normal + Hard） | 偏好对齐核心证据 |
| P2 | SFT vs SFT+GRPO 分解 | 训练方法贡献 |
| — | CPLEX baseline 实现 | OR1 可直接用 |

### 待补充（按优先级排序）

| 优先级 | # | 实验 | 工作量 | 关键说明 |
|:------:|---|------|:------:|---------|
| **P0** | M0a | Random Priority + NSGA-III | 小 | 与 M0c/M1 共用需求场景，仅替换优先级输入 |
| **P0** | M0b | Uniform Priority + NSGA-III | 小 | 同上 |
| **P0** | OR1 | NSGA-III vs CPLEX 效率对比 | 小 | 小规模需求集，独立子实验，不跑全流程 |
| **P0** | P3 | GPT-4o 零样本 vs Fine-tuned Qwen3-4B | 中 | 用相同 eval 集（含 Normal + Hard） |
| **P1** | S2 | 计算效率分析 | 小 | 各模块（LLM2/LLM3/求解器）时延统计 |
| **P1** | S3 | Case Study | 小 | 挑 2–3 个 LLM 处理优于规则的语义场景 |
| **P2** | P4b | 偏好定制化（修改标签微调） | 中 | 时间不足则降级为 discussion |

---

# 论文实验章节结构建议

```
5. Experiments
├── 5.1 Experimental Setup
│   ├── 数据集 / 场景设置
│   └── 评估指标（服务率、公平性指数、延误率）
│
├── 5.2 Main Comparison: Priority Strategies
│   └── M0a / M0b / M0c / M1 对比表 + 图
│        → 证明 LLM 优先级的优越性
│
├── 5.3 Solver Efficiency: NSGA-III vs CPLEX
│   └── OR1
│        → 证明 OR 方法选型合理
│
├── 5.4 LLM3 Preference Alignment Study
│   ├── P1 训练前后（Normal + Hard）
│   ├── P2 SFT vs GRPO 分解
│   └── P3 vs Frontier LLM
│        → 证明小模型微调可达 frontier 水平 + 偏好可对齐
│
├── 5.5 Computational Efficiency
│   └── S2
│
├── 5.6 Case Study
│   └── S3
│
└── 5.7 Discussion
    └── P4b（若已完成则作为定制化能力实证；否则作为 future work 提及）

---

每一个实验都对应一个潜在的审稿人质疑：

| 审稿人可能的质疑 | 由哪个实验回应 |
|----------------|---------------|
| "随便给优先级也能工作吧？" | M0a (random) |
| "公平的话不就该所有人一样优先级吗？" | M0b (uniform) |
| "传统规则就够了，为什么要 LLM？" | M0c vs M1 |
| "为什么用 NSGA 不用 CPLEX？" | OR1 |
| "直接用 GPT-4o 不就行了，为什么要 fine-tune？" | P3（部署成本 + 偏好对齐） |
| "Fine-tune 真的有用吗？还是 SFT 就够？" | P2 |
| "你的方法在歧义场景下能 work 吗？" | P1 Hard task + S3 |
| "LLM 算得起吗？实用吗？" | S2 |
| "号称可定制，但怎么证明？" | P4b（或 discussion） |


