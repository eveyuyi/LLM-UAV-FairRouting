"""Structured-demand views derived from canonical event manifests."""

from __future__ import annotations

from typing import Mapping

from llm4fairrouting.data.event_semantics import (
    MATERIAL_LABELS,
    PRIORITY_TO_DEADLINE,
    PRIORITY_TO_TIER,
    TEMPERATURE_SENSITIVE_MATERIALS,
    UNIT_BY_MATERIAL,
    operational_readiness,
    scenario_context,
)
from llm4fairrouting.data.priority_labels import derive_priority_labels
from llm4fairrouting.data.priority_policy import PRIORITY_POLICY_VERSION


def build_gold_structured_demand(event: Mapping[str, object]) -> dict[str, object]:
    event_id = str(event.get("event_id", ""))
    priority = int(event.get("latent_priority", 4))
    cargo = dict(event.get("cargo", {}) or {})
    origin = dict(event.get("origin", {}) or {})
    destination = dict(event.get("destination", {}) or {})
    weight_kg = float(event.get("weight_kg", cargo.get("weight_kg", 1.0)) or 1.0)
    deadline_minutes = int(event.get("deadline_minutes", PRIORITY_TO_DEADLINE.get(priority, 120)))
    demand_tier = str(event.get("demand_tier") or cargo.get("demand_tier") or PRIORITY_TO_TIER.get(priority, "consumer"))
    requester_role = str(event.get("requester_role", "consumer"))
    special_handling = list(event.get("special_handling", []) or [])
    vulnerability = dict(event.get("population_vulnerability", {}) or {})
    receiver_ready = bool(event.get("receiver_ready", False))
    material_type = str(cargo.get("type", "medicine"))
    scenario = str(
        event.get("scenario_context")
        or dict(event.get("priority_factors", {}) or {}).get("scenario_context")
        or scenario_context(priority, material_type, str(destination.get("type", "residential_area")))
    )
    readiness = str(
        event.get("operational_readiness")
        or operational_readiness(receiver_ready, priority)
    )
    quantity = max(
        1,
        round(
            weight_kg
            / {
                "aed": 2.0,
                "blood_product": 0.25,
                "cardiac_drug": 0.05,
                "thrombolytic": 0.05,
                "ventilator": 8.0,
                "icu_drug": 0.1,
                "vaccine": 0.3,
                "medicine": 0.5,
                "protective_suit": 0.8,
                "mask": 0.05,
                "disinfectant": 1.0,
                "food": 0.5,
                "otc_drug": 0.1,
                "daily_supply": 0.3,
            }.get(material_type, 0.5)
        ),
    )

    structured = {
        "demand_id": event_id,
        "source_event_id": event_id,
        "request_timestamp": event.get("request_timestamp"),
        "origin": {
            "station_name": str(origin.get("station_name", origin.get("fid", ""))),
            "fid": str(origin.get("fid", "")),
            "coords": list(origin.get("coords", [0.0, 0.0])),
            "type": "supply_station",
        },
        "destination": {
            "node_id": str(destination.get("node_id", destination.get("fid", ""))),
            "fid": str(destination.get("fid", "")),
            "coords": list(destination.get("coords", [0.0, 0.0])),
            "type": str(destination.get("type", "residential_area")),
        },
        "cargo": {
            "type": material_type,
            "type_cn": str(cargo.get("type_cn", MATERIAL_LABELS.get(material_type, material_type))),
            "weight_kg": weight_kg,
            "quantity": quantity,
            "quantity_unit": UNIT_BY_MATERIAL.get(material_type, "unit"),
            "temperature_sensitive": bool(
                cargo.get("temperature_sensitive", material_type in TEMPERATURE_SENSITIVE_MATERIALS)
            ),
            "demand_tier": demand_tier,
        },
        "demand_tier": demand_tier,
        "time_constraint": {
            "type": "hard" if deadline_minutes <= 30 or demand_tier in {"life_support", "critical"} else "soft",
            "description": f"Delivery target within {deadline_minutes} minutes",
            "deadline_minutes": deadline_minutes,
        },
        "requester_role": requester_role,
        "special_handling": special_handling,
        "population_vulnerability": vulnerability,
        "operational_readiness": readiness,
        "receiver_ready": receiver_ready,
        "priority_policy_version": PRIORITY_POLICY_VERSION,
        "priority_evaluation_signals": {
            "patient_condition": scenario,
            "time_sensitivity": (
                "Immediate action required"
                if deadline_minutes <= 15
                else "Urgent same-window delivery required"
                if deadline_minutes <= 30
                else "Timely delivery needed within the service window"
                if deadline_minutes <= 90
                else "Flexible same-day delivery"
            ),
            "population_vulnerability": vulnerability,
            "medical_urgency_self_report": scenario,
            "requester_role": requester_role,
            "scenario_context": scenario,
            "nearby_critical_facility": str(destination.get("type", "")),
            "operational_readiness": readiness,
            "special_handling": special_handling,
        },
        "context_signals": [
            scenario,
            f"Structured deadline: {deadline_minutes} minutes",
            f"Requester role: {requester_role}",
            readiness,
        ],
    }
    labels = derive_priority_labels(
        structured,
        latent_priority=priority,
        dialogue_observable_priority=event.get("dialogue_observable_priority"),
    )
    structured["labels"] = labels
    structured["latent_priority"] = labels.get("latent_priority", priority)
    structured["extraction_observable_priority"] = labels["extraction_observable_priority"]
    structured["solver_useful_priority"] = labels["solver_useful_priority"]
    return structured
