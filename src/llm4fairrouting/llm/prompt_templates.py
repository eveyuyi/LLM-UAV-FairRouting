"""
无人机医疗配送 workflow 的 prompt 模板。

- dialogue_generation_prompt()   — Module 1 (对话生成)
- context_extraction_prompt()    — Module 2 (信息提取 + 优先级评估信号，不赋值优先级)
- weight_adjustment_prompt()     — Module 3 (优先级排序 + 约束建议)
"""

import json
from typing import List, Dict, Optional


# ============================================================================
# System prompt
# ============================================================================

DRONE_SYSTEM_PROMPT = (
    "你是深圳市无人机医疗配送调度系统的 AI 助手。"
    "你需要从自然语言对话中提取结构化的配送需求信息，"
    "并在 Module 3 中根据需求特征进行优先级排序和补充约束建议。"
    "请始终以有效的 JSON 格式返回结果。"
)


# ============================================================================
# Module 1: Dialogue Generation
# ============================================================================

def dialogue_generation_prompt(batch_context: List[Dict]) -> str:
    """Module 1 LLM prompt — generate English dispatch dialogues in batch.

    Demand tiers (demand_tier):
      life_support — life-critical supplies (medication, ventilator, etc. at priority 1)
      critical     — urgent medical supplies (priority 2)
      regular      — routine medical supplies (priority 3)
      consumer     — consumer on-demand delivery (priority 4/5)
    """
    context_json = json.dumps(batch_context, ensure_ascii=False, indent=2)

    return f"""You are a simulation engine for a drone medical delivery dispatch system in Shenzhen.
For each of the {len(batch_context)} delivery events below, generate one realistic English multi-turn dispatch dialogue.

## Delivery Event List
{context_json}

## Demand Tier Guidelines
- **life_support**: Life-threatening emergencies (cardiac arrest, hemorrhage, respiratory failure).
  Dialogue must convey extreme urgency. Roles: ER physician, paramedic, emergency dispatch.
  Language: terse, precise, urgent — "Every second counts", "CODE RED", "immediately".
- **critical**: Serious patients requiring urgent hospital supplies (ICU drugs, ventilators).
  Roles: ICU nurse, ward coordinator, clinical pharmacist. Clear patient context required.
- **regular**: Routine restocking at clinics, community health centers, isolation facilities.
  Relaxed tone, flexible deadlines. Roles: clinic manager, community health worker, resident.
- **consumer**: App-style on-demand delivery (OTC meds, personal supplies).
  Casual, conversational, may include emojis. Roles: app user, customer.

## Dialogue Generation Requirements
1. Multi-turn format: `[HH:MM] Role: content` — 2–4 turns per dialogue.
2. Requester role must match the `requester_role` field in the event metadata.
3. Must naturally mention: material type (in English), quantity/weight, origin station name,
   destination node ID, and time constraint.
4. life_support dialogues must convey life urgency ("patient is waiting", "deploy immediately").
5. consumer dialogues should be casual and app-like.
6. Keep each dialogue concise: 60–180 words.
7. All dialogue text must be in English.

## Output Format (strict JSON, no extra text)
```json
{{
  "dialogues": [
    {{
      "dialogue_id": "D0001",
      "conversation": "multi-turn dialogue text in English..."
    }}
  ]
}}
```

Generate one dialogue per event. dialogue_id must match the input exactly."""


# ============================================================================
# Module 2: Context Extraction
# ============================================================================

def context_extraction_prompt(dialogues: List[Dict], time_window: str) -> str:
    """Module 2：从对话中提取结构化信息及优先级评估所需信号。

    注意：Module 2 **不**做优先级排序，只提取客观信息和信号，
    优先级排序由 Module 3（priority_inference）完成。

    Parameters
    ----------
    dialogues : list[dict]
        当前时间窗口内的对话列表。
    time_window : str
        时间窗口标识，如 "2024-03-15T09:00-09:30"。
    """
    # Module 2 只看对话文本，不接收坐标/人口统计等 metadata
    # 坐标/fid 由 demand_extraction 在 LLM 返回后从原始 metadata 回填
    dialogue_text = ""
    for d in dialogues:
        dialogue_text += (
            f"--- 对话 {d['dialogue_id']} [{d['timestamp']}] ---\n"
            f"{d['conversation']}\n\n"
        )

    return f"""你是深圳市无人机医疗配送调度系统的信息提取模块（Module 2）。
请仅根据以下调度对话的**文本内容**，提取所有配送需求的结构化信息和优先级评估所需信号。

**重要规则：**
- 只能从对话文本中提取信息，不依赖任何外部数据
- 只提取，不排序，不分配优先级数值（排序由 Module 3 完成）

## 时间窗口
{time_window}

## 调度对话（仅文本）
{dialogue_text}

## 输出格式
```json
{{
  "time_window": "{time_window}",
  "demands": [
    {{
      "demand_id": "REQ001",
      "source_dialogue_id": "D0001",
      "origin": {{
        "station_name": "丰翼无人机石岩集散中心航站",
        "type": "supply_station"
      }},
      "destination": {{
        "node_id": "D9675",
        "type": "hospital | clinic | residential_area | public_space | office"
      }},
      "cargo": {{
        "type": "aed",
        "type_cn": "自动除颤仪",
        "demand_tier": "life_support",
        "weight_kg": 2.1,
        "quantity": 1,
        "quantity_unit": "台",
        "temperature_sensitive": false
      }},
      "demand_tier": "life_support | critical | regular | consumer",
      "time_constraint": {{
        "type": "hard | soft",
        "description": "15分钟内必须到位",
        "deadline_minutes": 15
      }},
      "priority_evaluation_signals": {{
        "patient_condition": "心脏骤停，CPR进行中（从对话推断）",
        "time_sensitivity": "黄金抢救窗口，每分钟都至关重要",
        "population_vulnerability": {{
          "elderly_involved": true,
          "children_involved": false,
          "vulnerable_community": false
        }},
        "medical_urgency_self_report": "极端紧急",
        "requester_role": "emergency_doctor | icu_nurse | community_health_worker | consumer",
        "scenario_context": "心脏骤停，现场 CPR，等待 AED",
        "nearby_critical_facility": "public_space"
      }},
      "context_signals": [
        "心脏骤停，CPR 进行中",
        "降落区已清空，急救员现场等候"
      ]
    }}
  ]
}}
```

字段说明（所有信息均须从对话文本推断）：
- `demand_tier`：life_support > critical > regular > consumer，根据对话中的物资类型和紧急程度判断
- `origin.station_name`：对话中提到的起点站点名称
- `destination.node_id`：对话中提到的目的地编号（如 D9675）
- `priority_evaluation_signals`：从对话语气、角色、患者描述中提取的客观信号
- `time_constraint.type`：hard=对话中有明确截止时间，soft=弹性/今日内即可

请严格按照 JSON schema 输出，为每条对话都生成需求记录，不要遗漏。"""


# ============================================================================
# Module 3: Weight Adjustment + Priority Ranking
# ============================================================================

def weight_adjustment_prompt(demands: List[Dict], city_context: Optional[Dict] = None) -> str:
    """Module 3：根据需求层级和优先级评估信号，做排序并分配优先级。

    Module 3 负责：
    1. 在时间窗口内对所有需求进行优先级排序（考虑层级 + 信号）
    2. 为每条需求分配 priority，并给出可选的 supplementary_constraints

    Parameters
    ----------
    demands : list[dict]
        Module 2 输出的结构化需求列表（含 demand_tier 和 priority_evaluation_signals）。
    city_context : dict, optional
        城市上下文信息（交通状况、天气等）。
    """
    demands_json = json.dumps(demands, ensure_ascii=False, indent=2)
    city_json = json.dumps(city_context or {}, ensure_ascii=False, indent=2)

    return f"""你是深圳市无人机医疗配送优化系统的优先级排序模块（Module 3）。

请根据以下配送需求的层级信息和优先级评估信号，完成两步工作：
**Step 1：在时间窗口内对所有需求进行优先级排序**
**Step 2：为每条需求分配求解优先级（priority）并补充必要约束建议**

## 配送需求（含 Module 2 提取的评估信号）
{demands_json}

## 城市上下文
{city_json}

## 四层需求体系（优先级由高到低）
| 层级 | demand_tier | 典型场景 | priority |
|------|-------------|---------|----------|
| 生命支持 | life_support | 心脏骤停、大出血、AED 投送 | 1 |
| 重症物资 | critical | ICU 药物、呼吸机调配 | 2 |
| 常规物资 | regular | 社区补货、居民用药 | 3 |
| 消费配送 | consumer | 外卖、OTC 药品、日用品 | 4 |

## 参数说明
- **priority** (int, 1–4)：1=最高优先级，4=最低优先级
- supplementary_constraints 仅在确有必要时输出；没有就返回空列表

## 排序调整规则（在层级基础上微调）
- 涉及心脏骤停/脑梗/大出血等 → 同层级内上调
- 老年人 (elderly_ratio > 0.45) 或儿童受益 → 同层级内上调 0.5
- 弱势社区 (vulnerable_community=true) → 同层级内上调
- consumer 级别中的急症OTC（如退烧药+儿童）→ 可上调至 regular

## 输出格式
```json
{{
  "global_weights": {{
    "w_distance": 1.0,
    "w_time": 1.0,
    "w_risk": 1.0
  }},
  "demand_configs": [
    {{
      "demand_id": "REQ001",
      "demand_tier": "life_support",
      "priority": 1,
      "window_rank": 1,
      "reasoning": "心脏骤停，CPR 进行中，黄金抢救窗口 < 4 分钟，老年患者 62 岁"
    }}
  ],
  "supplementary_constraints": [
    {{
      "type": "noise_avoidance | speed_override | no_fly_zone",
      "description": "说明",
      "affected_zone": {{"center": [113.88, 22.65], "radius_m": 300}},
      "time_window": ["09:00", "09:30"]
    }}
  ]
}}
```

字段说明：
- `window_rank`：该需求在当前时间窗口内的调度优先级排名（1=最优先）
- `reasoning`：简要说明排序依据（≤50字）

请严格按照 JSON 格式输出，不要添加额外文字。"""
