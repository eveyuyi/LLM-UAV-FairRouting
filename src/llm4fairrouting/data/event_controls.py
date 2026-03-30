"""Dialogue-control views derived from canonical event-core records."""

from __future__ import annotations

from typing import Dict, List, Tuple

from llm4fairrouting.data.event_core import EventCore
from llm4fairrouting.data.priority_policy import PRIORITY_POLICY_VERSION
from llm4fairrouting.data.event_semantics import unique_keywords

_EMERGENCY_ROLES = {"emergency_doctor", "paramedic", "triage_nurse"}
_CRITICAL_ROLES = {"icu_nurse", "clinical_pharmacist", "ward_coordinator"}


def _factor_spec(
    name: str,
    value: object,
    description: str,
    keywords: List[str],
    *,
    source_field: str,
) -> Dict[str, object]:
    return {
        "name": name,
        "value": value,
        "description": description,
        "keywords": unique_keywords(keywords),
        "source_field": source_field,
    }


def _priority_reason_codes(core: EventCore) -> tuple[List[str], List[str]]:
    reason_codes = [f"tier_{core.demand_tier}"]
    source_fields = ["demand_tier", "scenario_context"]

    if core.deadline_minutes <= 15:
        reason_codes.append("deadline_le_15m")
        source_fields.append("deadline_minutes")
    elif core.deadline_minutes <= 30:
        reason_codes.append("deadline_le_30m")
        source_fields.append("deadline_minutes")
    elif core.deadline_minutes <= 60:
        reason_codes.append("deadline_le_60m")
        source_fields.append("deadline_minutes")

    if core.requester_role in _EMERGENCY_ROLES:
        reason_codes.append("emergency_requester")
        source_fields.append("requester_role")
    elif core.requester_role in _CRITICAL_ROLES:
        reason_codes.append("critical_requester")
        source_fields.append("requester_role")

    if "cold_chain" in core.special_handling:
        reason_codes.append("special_handling_cold_chain")
        source_fields.append("special_handling")
    if "shock_protection" in core.special_handling:
        reason_codes.append("special_handling_shock_protection")
        source_fields.append("special_handling")

    vulnerability = core.population_vulnerability
    if vulnerability.get("vulnerable_community"):
        reason_codes.append("vulnerable_population")
        source_fields.append("population_vulnerability")
    if vulnerability.get("children_involved"):
        reason_codes.append("children_involved")
        source_fields.append("population_vulnerability")
    if vulnerability.get("elderly_involved"):
        reason_codes.append("elderly_involved")
        source_fields.append("population_vulnerability")

    if core.receiver_ready:
        reason_codes.append("receiver_ready")
        source_fields.append("receiver_ready")

    if str(core.destination.get("type", "")) in {"hospital", "clinic", "community_health_center"}:
        reason_codes.append("destination_hospital_like")
        source_fields.append("destination.type")

    if core.scenario_context:
        reason_codes.append("scenario_context_present")

    return list(dict.fromkeys(reason_codes)), list(dict.fromkeys(source_fields))


def build_priority_factors(core: EventCore) -> Dict[str, object]:
    reason_codes, source_fields = _priority_reason_codes(core)
    return {
        "policy_version": PRIORITY_POLICY_VERSION,
        "scenario_context": core.scenario_context,
        "reason_codes": reason_codes,
        "source_fields": source_fields,
    }


def build_must_mention_factors(core: EventCore) -> List[Dict[str, object]]:
    factors = [
        _factor_spec(
            "scenario_context",
            core.scenario_context,
            core.scenario_context,
            ["cpr", "cardiac arrest", "transfusion", "stroke", "backup ventilator", "vaccination", "same-day"],
            source_field="scenario_context",
        ),
        _factor_spec(
            "deadline_minutes",
            core.deadline_minutes,
            f"Delivery is needed within {core.deadline_minutes} minutes.",
            [
                f"{core.deadline_minutes} min",
                f"{core.deadline_minutes}-minute",
                f"within {core.deadline_minutes} minutes",
            ],
            source_field="deadline_minutes",
        ),
        _factor_spec(
            "requester_role",
            core.requester_role,
            f"The request comes from the {core.requester_role.replace('_', ' ')}.",
            [
                core.requester_role.replace("_", " "),
                core.requester_role.replace("_", " ").title(),
            ],
            source_field="requester_role",
        ),
    ]
    if core.special_handling:
        factors.append(
            _factor_spec(
                "special_handling",
                list(core.special_handling),
                f"Special handling is required: {', '.join(core.special_handling)}.",
                list(core.special_handling) + ["cold-chain", "shock-proof", "insulated"],
                source_field="special_handling",
            )
        )
    if core.receiver_ready:
        factors.append(
            _factor_spec(
                "receiver_ready",
                True,
                "Landing zone cleared; team waiting for immediate handoff",
                ["landing zone", "standing by", "ready for handoff", "team waiting"],
                source_field="receiver_ready",
            )
        )
    return factors


def build_optional_factors(core: EventCore) -> List[Dict[str, object]]:
    factors = [
        _factor_spec(
            "destination_type",
            str(core.destination.get("type", "residential_area")),
            f"The receiving point is a {str(core.destination.get('type', 'residential_area')).replace('_', ' ')}.",
            [
                str(core.destination.get("type", "residential_area")).replace("_", " "),
                "receiving point",
            ],
            source_field="destination.type",
        )
    ]
    vulnerability = core.population_vulnerability
    if vulnerability.get("children_involved") or vulnerability.get("elderly_involved"):
        factors.append(
            _factor_spec(
                "population_vulnerability",
                dict(vulnerability),
                "The receiver serves a vulnerable population.",
                ["child", "children", "elderly", "senior", "vulnerable community"],
                source_field="population_vulnerability",
            )
        )
    return factors


def build_dialogue_control_views(core: EventCore) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    priority_factors = build_priority_factors(core)
    must_mention = build_must_mention_factors(core)
    optional = build_optional_factors(core)
    return priority_factors, must_mention, optional
