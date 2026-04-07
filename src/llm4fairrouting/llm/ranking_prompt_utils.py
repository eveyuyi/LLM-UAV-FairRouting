"""Shared compact prompt builders for LLM3 ranking tasks."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence


def _as_dict(value: object) -> Dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _clean_text(value: object, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _clean_str_list(values: object, *, max_items: int, max_chars: int) -> List[str]:
    cleaned: List[str] = []
    for item in _as_list(values):
        text = _clean_text(item, max_chars=max_chars)
        if text:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _compact_vulnerability(vulnerability: object) -> Dict:
    vuln = _as_dict(vulnerability)
    compact: Dict[str, object] = {}
    for key in ("elderly_involved", "children_involved", "vulnerable_community"):
        if key in vuln:
            compact[key] = bool(vuln.get(key))
    elderly_ratio = vuln.get("elderly_ratio")
    if elderly_ratio not in (None, ""):
        try:
            compact["elderly_ratio"] = round(float(elderly_ratio), 2)
        except (TypeError, ValueError):
            pass
    population = vuln.get("population")
    if population not in (None, ""):
        try:
            compact["population"] = int(population)
        except (TypeError, ValueError):
            pass
    return compact


def compact_ranking_demand(demand: Dict) -> Dict:
    destination = _as_dict(demand.get("destination", {}))
    cargo = _as_dict(demand.get("cargo", {}))
    time_constraint = _as_dict(demand.get("time_constraint", {}))
    signals = _as_dict(demand.get("priority_evaluation_signals", {}))

    record: Dict[str, object] = {
        "demand_id": str(demand.get("demand_id", "")),
        "tier": str(demand.get("demand_tier") or cargo.get("demand_tier") or ""),
        "cargo": str(cargo.get("type", "") or ""),
        "dest": str(destination.get("type", "") or ""),
    }

    deadline = time_constraint.get("deadline_minutes")
    if deadline not in (None, ""):
        try:
            record["deadline_min"] = int(deadline)
        except (TypeError, ValueError):
            pass

    constraint_type = str(time_constraint.get("type", "") or "")
    if constraint_type:
        record["deadline_type"] = constraint_type

    requester = str(signals.get("requester_role") or demand.get("requester_role") or "")
    if requester:
        record["requester"] = requester

    if demand.get("receiver_ready") is not None:
        record["receiver_ready"] = bool(demand.get("receiver_ready"))

    quantity = cargo.get("quantity")
    if quantity not in (None, ""):
        record["qty"] = quantity
    quantity_unit = str(cargo.get("quantity_unit", "") or "")
    if quantity_unit:
        record["qty_unit"] = quantity_unit

    weight_kg = cargo.get("weight_kg", demand.get("weight_kg"))
    if weight_kg not in (None, ""):
        try:
            record["weight_kg"] = round(float(weight_kg), 1)
        except (TypeError, ValueError):
            pass

    if bool(cargo.get("temperature_sensitive")):
        record["temperature_sensitive"] = True

    patient = _clean_text(signals.get("patient_condition", ""), max_chars=120)
    if patient:
        record["patient"] = patient
    urgency = _clean_text(
        signals.get("time_sensitivity") or signals.get("medical_urgency_self_report") or "",
        max_chars=120,
    )
    if urgency:
        record["urgency"] = urgency
    scenario = _clean_text(signals.get("scenario_context", ""), max_chars=120)
    if scenario:
        record["scenario"] = scenario
    readiness = _clean_text(signals.get("operational_readiness", ""), max_chars=120)
    if readiness:
        record["readiness"] = readiness
    nearby_facility = _clean_text(signals.get("nearby_critical_facility", ""), max_chars=48)
    if nearby_facility:
        record["near_facility"] = nearby_facility

    handling = _clean_str_list(signals.get("special_handling", []), max_items=4, max_chars=32)
    if handling:
        record["handling"] = handling

    vulnerability = _compact_vulnerability(signals.get("population_vulnerability", {}))
    if vulnerability:
        record["vulnerability"] = vulnerability

    notes = _clean_str_list(demand.get("context_signals", []), max_items=3, max_chars=96)
    if notes:
        record["notes"] = notes

    return {key: value for key, value in record.items() if value not in ("", [], {})}


def build_compact_ranking_payload(
    demands: Sequence[Dict],
    *,
    time_window: Optional[str] = None,
    city_context: Optional[Dict] = None,
) -> Dict:
    payload: Dict[str, object] = {
        "demands": [compact_ranking_demand(demand) for demand in demands],
    }
    if time_window:
        payload["time_window"] = str(time_window)
    city = _as_dict(city_context)
    if city:
        payload["city_context"] = city
    return payload


def render_priority_ranking_prompt(
    demands: Sequence[Dict],
    *,
    time_window: Optional[str] = None,
    city_context: Optional[Dict] = None,
) -> str:
    payload = build_compact_ranking_payload(
        demands,
        time_window=time_window,
        city_context=city_context,
    )
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    return (
        "Rank the delivery demands in this window.\n"
        "Return JSON only with:\n"
        '{"priority_labels":[{"demand_id":"string","priority":1,"window_rank":1,"reasoning":"short string"}]}\n'
        "Rules:\n"
        "- priority is 1-4, where 1 is highest.\n"
        "- window_rank must be unique and cover every demand exactly once.\n"
        "- Use tier, deadline, requester, patient condition, scenario, handling, vulnerability, and receiver readiness.\n"
        "- Keep reasoning short and concrete.\n"
        "Input:\n"
        f"{payload_json}"
    )
