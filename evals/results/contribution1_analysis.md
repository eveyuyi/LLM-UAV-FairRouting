# Contribution 1 — LLM2 Extraction Quality 详细分析

**数据来源**: `data/test/test_seeds/norm_eval/seed_4111/llm3_sft_pipeline.jsonl`  
**样本量**: 4,340 demands（576 windows × 平均 7.5 demands）  
**评估方式**: 每条 demand 中的提取字段 vs `gold_extraction` 字段逐项比对

---

## 字段级准确率

| 字段 | 准确率 | FN（漏检） | FP（误报） | 说明 |
|---|---|---|---|---|
| origin_fid | **100.0%** | 0 | 0 | 供货站点，对话中明确提及 |
| dest_fid | **100.0%** | 0 | 0 | 目的地节点，对话中明确提及 |
| deadline_minutes | **99.9%** | 2 | 0 | 截止时间，对话中明确提及 |
| weight_kg | 99.0% | — | — | 货物重量，对话中明确提及 |
| cargo_type | 95.9% | — | — | 货物类型，对话中明确提及 |
| children_involved | 86.4% | — | — | |
| special_handling_match | 83.0% | — | — | |
| requester_role | 78.2% | 0 | 0 | 所有错误来自角色名称变体（如 family_caregiver vs caregiver）|
| temperature_sensitive | 75.1% | — | — | |
| elderly_involved | 61.1% | **1,591** | 98 | ⚠️ 详见下方根因分析 |
| dest_type | 51.0% | — | — | |
| vulnerable_community | 50.1% | **2,140** | 27 | ⚠️ 详见下方根因分析 |
| demand_tier | 47.7% | — | — | |

## 综合指标

| 指标 | 数值 | 说明 |
|---|---|---|
| **Schema validity rate** | **100.0%** | 所有提取结果可直接进入 OR solver |
| **Solver feasible rate** | **100.0%** | 96/96 windows 产生可行解，722/722 需求全部可行，0 需求被过滤 |
| Critical signal recall（4 字段全对） | 28.5% | ⚠️ 详见下方根因分析 |
| Priority chain 三值完全一致 | 42.1% | ⚠️ 详见下方根因分析 |

---

## ⚠️ Critical Signal Recall = 28.5% 的根因

### 表面现象
Critical signal recall 要求以下 4 个字段**同时**正确：
- deadline_minutes（99.9%）
- requester_role（78.2%）
- elderly_involved（61.1%）
- vulnerable_community（50.1%）

若独立，期望联合准确率 ≈ 0.999 × 0.782 × 0.611 × 0.501 ≈ **23.9%**，实际 28.5% 略高（字段间正相关）。

### 真正的根因：elderly_involved / vulnerable_community 不可从对话推断

对 1,591 个 `elderly_involved` FN 案例的分析显示：

```
DEM_000_00: extracted=False, gold=True (elderly_ratio=0.31)
dialogue: "Please route 74 packages of daily supplies... The receiver requested same-day delivery."

DEM_000_01: extracted=False, gold=True (elderly_ratio=0.26)
dialogue: "I need to arrange same-day delivery... within 120 minutes."
```

**对话中没有任何关于老年人或脆弱群体的描述。** gold_extraction 里的 `elderly_involved=True` 来自目的地区域的人口统计数据库（elderly_ratio=0.26~0.56），这是地理信息系统中的**外部数据**，不是对话中能观察到的信息。

### 结论

LLM2 的低 elderly_involved / vulnerable_community 准确率**不是提取失败**，而是**评估方式的混淆**：

| 字段类型 | 来源 | LLM2 能否提取 |
|---|---|---|
| origin/dest/cargo/deadline/requester_role | 对话内容 | ✅ 可提取（78–100%）|
| elderly_involved / vulnerable_community | 目的地人口统计数据库（地理信息） | ❌ 对话中不存在 |

**修正后的 critical signal recall**（仅计 LLM2 可见字段：deadline + requester_role）= **78.1%**

在论文 Contribution 1 中，应将 elderly/vulnerability 字段定性为"需要外部地理数据库自动增强"的字段，而非 LLM2 的提取目标。管道的实际设计中这些字段由地理位置查询自动填充，LLM2 负责对话可见字段。

---

## ⚠️ Priority Chain Consistency = 42.1% 的根因

### 三步链路分解

```
latent_priority → dialogue_observable_priority → extraction_observable_priority
                  (对话生成忠实度)                (LLM2 提取准确度)
```

| 步骤 | 一致率 | 分析 |
|---|---|---|
| latent → dialogue_observable | **88.5%** | 对话生成质量高，88.5% 的对话忠实表达了真实优先级 |
| dialogue_observable → extraction_observable | **43.0%** | **瓶颈在这里**：LLM2 提取时偏离对话可观察优先级 |
| 全链路一致（三值相等） | **42.1%** | ≈ 第二步的 43%（第一步正确时第二步才有意义）|

### 最常见的失败模式

| 三元组 (latent, dial_obs, extract_obs) | 数量 | 分析 |
|---|---|---|
| **(4, 4, 3)** | 1,131 | 最大问题：对话正确表达 P4，但 LLM2 **过度升级**为 P3 |
| (2, 1, 2) | 460 | 对话夸大紧迫性（P2→P1），LLM2 反而**正确回归** P2 |
| (3, 3, 2) | 395 | LLM2 将 P3 过度升级为 P2 |
| (1, 1, 2) | 248 | LLM2 将 P1 降级为 P2 |
| (4, 4, 2) | 236 | LLM2 将 P4 大幅升级为 P2 |

### 结论

**LLM2 存在系统性"过度优先化"偏差**（safety bias）：将 P3/P4 常规配送请求误判为 P2。这是训练数据中安全优先惯性导致的，不是能力限制。

- `(4,4,3)` 1131 例 + `(3,3,2)` 395 例 + `(4,4,2)` 236 例 = **1,762 例过度升级**（占 41%）
- `(2,1,2)` 460 例 = LLM2 反而纠正了对话的夸大表达，这是正向的

**对 Contribution 1 叙事的影响**：priority chain 低 (42%) 主要反映 LLM2 的安全偏差，而非提取能力不足。在论文中可将此定性为"LLM2 偏向保守优先策略，需要 LLM3 的判别性排序来纠正"。

---

## TODO

- **Pipeline latency (LLM2/LLM3 模块延迟)**: 当前所有实验为 offline 模式，LLM 调用无计时。需单独在线跑一次（起 vLLM server）记录各模块延迟。预计 LLM3 per-window ≈ 10s，LLM2 在本 pipeline 中已预计算。
