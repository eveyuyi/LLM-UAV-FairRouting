"""Canonical event-core record shared by manifest generation and downstream views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping


@dataclass(frozen=True)
class EventCore:
    event_id: str
    time_slot: int
    time_hour: float
    origin: Dict[str, object]
    destination: Dict[str, object]
    cargo: Dict[str, object]
    weight_kg: float
    deadline_minutes: int
    demand_tier: str
    requester_role: str
    special_handling: List[str]
    population_vulnerability: Dict[str, object]
    receiver_ready: bool
    latent_priority: int
    scenario_context: str
    dialogue_styles: List[str]
    request_timestamp: str | None = None


def event_core_to_manifest_record(
    core: EventCore,
    *,
    priority_factors: Mapping[str, object],
    must_mention_factors: List[Dict[str, object]],
    optional_factors: List[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "schema_version": "priority_observability_v2",
        "event_id": core.event_id,
        "time_slot": int(core.time_slot),
        "time_hour": round(float(core.time_hour), 4),
        "request_timestamp": core.request_timestamp,
        "origin": dict(core.origin),
        "destination": dict(core.destination),
        "cargo": dict(core.cargo),
        "weight_kg": round(float(core.weight_kg), 1),
        "deadline_minutes": int(core.deadline_minutes),
        "demand_tier": str(core.demand_tier),
        "requester_role": str(core.requester_role),
        "special_handling": list(core.special_handling),
        "population_vulnerability": dict(core.population_vulnerability),
        "receiver_ready": bool(core.receiver_ready),
        "latent_priority": int(core.latent_priority),
        "scenario_context": str(core.scenario_context),
        "dialogue_styles": list(core.dialogue_styles),
        "priority_factors": dict(priority_factors),
        "must_mention_factors": list(must_mention_factors),
        "optional_factors": list(optional_factors),
    }
