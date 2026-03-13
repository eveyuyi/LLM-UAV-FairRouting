"""
Prompt templates for the drone-delivery workflow.

- dialogue_generation_prompt()   — Module 1 (dialogue generation)
- context_extraction_prompt()    — Module 2 (structured extraction + priority evidence)
- weight_adjustment_prompt()     — Module 3 (ranking + constraint suggestions)
"""

import json
from typing import List, Dict, Optional


# ============================================================================
# System prompt
# ============================================================================

DRONE_SYSTEM_PROMPT = (
    "You are the AI assistant for a drone-based delivery dispatch system in Shenzhen. "
    "You extract structured delivery demands from natural-language dispatch dialogues and, "
    "in Module 3, rank them by operational urgency. Always return valid JSON only."
)


# ============================================================================
# Module 1: Dialogue Generation
# ============================================================================

def dialogue_generation_prompt(batch_context: List[Dict]) -> str:
    """Module 1 prompt: generate realistic English dispatch dialogues in batch."""
    context_json = json.dumps(batch_context, ensure_ascii=False, indent=2)

    return f"""You are a simulation engine for a drone medical delivery dispatch system in Shenzhen.
For each of the {len(batch_context)} delivery events below, generate one realistic English multi-turn dispatch dialogue.

## Delivery Event List
{context_json}

## Tier Guidance
- `life_support`: life-threatening emergency; concise, urgent, operational.
- `critical`: serious hospital request; urgent but controlled.
- `regular`: routine clinical replenishment; professional and calm.
- `consumer`: app-like same-day request; casual but still specific.

## Requirements
1. Use the format `[HH:MM] Role: message`.
2. Write 2-4 turns per dialogue.
3. Match the requester role, request channel, handling notes, and receiver notes in the input.
4. Naturally mention the item, quantity or weight, origin station, destination node, and delivery target.
5. Do not mention internal labels such as numeric priority.
6. Keep the tone operational and realistic. Avoid generic filler.
7. Keep each dialogue between 60 and 180 words.
8. Output English only.

## Output Format
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

Return one dialogue per event. `dialogue_id` must match the input exactly."""


# ============================================================================
# Module 2: Context Extraction
# ============================================================================

def context_extraction_prompt(dialogues: List[Dict], time_window: str) -> str:
    """Module 2 prompt: extract structured demand data and ranking evidence."""
    dialogue_text = ""
    for d in dialogues:
        dialogue_text += (
            f"--- Dialogue {d['dialogue_id']} [{d['timestamp']}] ---\n"
            f"{d['conversation']}\n\n"
        )

    return f"""You are Module 2 of the drone-delivery workflow.
Extract structured delivery demands and ranking evidence from the dialogue text only.

## Rules
- Use only the dialogue text below.
- Do not assign numeric priority.
- Do not rank demands.
- Extract one demand record for each dialogue.

## Time Window
{time_window}

## Dialogue Text
{dialogue_text}

## Output Format
```json
{{
  "time_window": "{time_window}",
  "demands": [
    {{
      "demand_id": "REQ001",
      "source_dialogue_id": "D0001",
      "origin": {{
        "station_name": "Fenyi Shiyan Dispatch Hub",
        "type": "supply_station"
      }},
      "destination": {{
        "node_id": "D9675",
        "type": "hospital | clinic | residential_area | public_space | office"
      }},
      "cargo": {{
        "type": "aed",
        "type_cn": "AED defibrillator",
        "demand_tier": "life_support",
        "weight_kg": 2.1,
        "quantity": 1,
        "quantity_unit": "unit",
        "temperature_sensitive": false
      }},
      "demand_tier": "life_support | critical | regular | consumer",
      "time_constraint": {{
        "type": "hard | soft",
        "description": "Must arrive within 15 minutes",
        "deadline_minutes": 15
      }},
      "priority_evaluation_signals": {{
        "patient_condition": "Cardiac arrest; CPR in progress",
        "time_sensitivity": "Immediate action required",
        "population_vulnerability": {{
          "elderly_involved": true,
          "children_involved": false,
          "vulnerable_community": false
        }},
        "medical_urgency_self_report": "Immediate action required",
        "requester_role": "emergency_doctor | icu_nurse | community_health_worker | consumer",
        "scenario_context": "Cardiac arrest response; AED requested",
        "nearby_critical_facility": "public_space",
        "operational_readiness": "Landing zone cleared; team waiting",
        "special_handling": ["shock_protection"]
      }},
      "context_signals": [
        "Cardiac arrest response in progress",
        "Landing zone cleared and receiver waiting"
      ]
    }}
  ]
}}
```

## Extraction Notes
- `demand_tier`: infer from item type and urgency in the dialogue.
- `time_constraint.type`: use `hard` when the dialogue states a strict delivery target; otherwise use `soft`.
- `priority_evaluation_signals`: capture evidence, not conclusions.
- `context_signals`: short English evidence phrases copied or inferred from the dialogue.

Return valid JSON only."""


# ============================================================================
# Module 3: Weight Adjustment + Priority Ranking
# ============================================================================

def weight_adjustment_prompt(demands: List[Dict], city_context: Optional[Dict] = None) -> str:
    """Module 3 prompt: rank demands and assign solver priorities."""
    demands_json = json.dumps(demands, ensure_ascii=False, indent=2)
    city_json = json.dumps(city_context or {}, ensure_ascii=False, indent=2)

    return f"""You are Module 3 of the drone-delivery workflow.
Rank the demands within this time window and assign solver priorities.

## Demand List
{demands_json}

## City Context
{city_json}

## Priority Semantics
- `priority=1`: life-support or immediately life-threatening demand
- `priority=2`: urgent clinical demand with strong time pressure
- `priority=3`: routine but time-bound operational demand
- `priority=4`: flexible same-day demand

## Ranking Guidance
- Use `demand_tier`, `time_constraint`, `patient_condition`, requester role, special handling, and vulnerability signals together.
- Life-threatening cases, strict deadlines, and ready receivers should rank higher.
- Consumer OTC medication may move up to `priority=3` when the dialogue shows real urgency.
- Suggest supplementary constraints only when they materially affect routing.

## Output Format
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
      "reasoning": "Cardiac arrest, CPR in progress, rescue window under 4 minutes"
    }}
  ],
  "supplementary_constraints": [
    {{
      "type": "noise_avoidance | speed_override | no_fly_zone",
      "description": "Short routing note",
      "affected_zone": {{"center": [113.88, 22.65], "radius_m": 300}},
      "time_window": ["09:00", "09:30"]
    }}
  ]
}}
```

## Notes
- `window_rank=1` means the most urgent demand in the current window.
- Keep `reasoning` short and concrete.
- Return valid JSON only."""
