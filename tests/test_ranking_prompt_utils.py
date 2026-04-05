from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from llm4fairrouting.llm.prompt_templates import weight_adjustment_prompt
from llm4fairrouting.llm.ranking_prompt_utils import (
    build_compact_ranking_payload,
    compact_ranking_demand,
    render_priority_ranking_prompt,
)

_UTILS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "llm3_verl_utils.py"
_SPEC = importlib.util.spec_from_file_location("llm3_verl_utils_for_test", _UTILS_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_prompt_text = _MODULE.build_prompt_text


def _sample_demand() -> dict:
    return {
        "demand_id": "REQ001",
        "source_dialogue_id": "D0001",
        "source_event_id": "EV001",
        "request_timestamp": "2024-03-15T07:40:00",
        "origin": {
            "station_name": "Commercial Distribution Hub COM_54",
            "type": "supply_station",
            "fid": "COM_54",
            "coords": [113.80, 22.66],
        },
        "destination": {
            "node_id": "DEM_7427",
            "type": "residential_area",
            "fid": "DEM_7427",
            "coords": [113.94, 22.67],
        },
        "cargo": {
            "type": "daily supplies",
            "type_cn": "daily supplies",
            "demand_tier": "regular",
            "weight_kg": 3.4,
            "quantity": 11,
            "quantity_unit": "packages",
            "temperature_sensitive": False,
        },
        "demand_tier": "regular",
        "time_constraint": {
            "type": "hard",
            "description": "Delivery is needed within 120 minutes",
            "deadline_minutes": 120,
        },
        "receiver_ready": True,
        "priority_evaluation_signals": {
            "patient_condition": "Routine supply request for a campus clinic that still needs timely restock.",
            "time_sensitivity": "Flexible same-day delivery",
            "population_vulnerability": {
                "elderly_involved": False,
                "children_involved": True,
                "vulnerable_community": False,
                "elderly_ratio": 0.32,
                "population": 8842,
            },
            "medical_urgency_self_report": "Flexible same-day delivery",
            "requester_role": "family caregiver",
            "scenario_context": "same-day order request with clinic handoff coordination",
            "nearby_critical_facility": "hospital",
            "operational_readiness": "morning intake wave has consumed today's first stock batch",
            "special_handling": [],
        },
        "context_signals": [
            "Tier inferred from dialogue: consumer",
            "Patient or service context: Routine supply request",
            "Delivery target mentioned in dialogue: 120 min",
            "Requester role inferred from dialogue: family_caregiver",
        ],
        "labels": {
            "latent_priority": 4,
            "dialogue_observable_priority": 4,
            "extraction_observable_priority": 3,
        },
        "gold_extraction": {
            "destination": {"coords": [113.94, 22.67]},
        },
    }


def test_compact_ranking_demand_omits_heavy_and_answer_leak_fields():
    compact = compact_ranking_demand(_sample_demand())

    assert compact["demand_id"] == "REQ001"
    assert compact["tier"] == "regular"
    assert compact["cargo"] == "daily supplies"
    assert compact["dest"] == "residential_area"
    assert "labels" not in compact
    assert "gold_extraction" not in compact
    assert "origin" not in compact
    assert "destination" not in compact
    assert "coords" not in json.dumps(compact, ensure_ascii=False)


def test_render_priority_ranking_prompt_uses_priority_labels_schema():
    prompt = render_priority_ranking_prompt([_sample_demand()], time_window="2024-03-15T07:40-07:45")

    assert '"priority_labels"' in prompt
    assert '"demand_configs"' not in prompt
    assert '"labels"' not in prompt
    assert '"gold_extraction"' not in prompt


def test_training_and_inference_prompts_use_compact_payload():
    demands = [_sample_demand() for _ in range(3)]
    full_payload = json.dumps({"time_window": "2024-03-15T07:40-07:45", "demands": demands}, ensure_ascii=False)
    compact_payload = json.dumps(
        build_compact_ranking_payload(demands, time_window="2024-03-15T07:40-07:45"),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    training_prompt = build_prompt_text("2024-03-15T07:40-07:45", demands)
    inference_prompt = weight_adjustment_prompt(demands)

    assert len(compact_payload) < len(full_payload)
    assert '"labels"' not in training_prompt
    assert '"gold_extraction"' not in training_prompt
    assert '"demand_configs"' not in inference_prompt
