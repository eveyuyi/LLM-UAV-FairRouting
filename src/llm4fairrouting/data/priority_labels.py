"""Canonical observable-priority labels derived from structured demands.

The rules in this module implement the human-aligned allocation policy behind
LLM3: prioritize scarce delivery capacity for requests with the strongest
evidence of severity, urgency, vulnerability, and operational actionability.
"""

from __future__ import annotations

import copy
from typing import Dict, List

from llm4fairrouting.data.priority_policy import PRIORITY_POLICY_VERSION

TIER_PRIORITIES = {
    "life_support": 1,
    "critical": 2,
    "regular": 3,
    "consumer": 4,
}

_TIER_BASE_SCORE = {
    "life_support": 100,
    "critical": 72,
    "regular": 42,
    "consumer": 12,
}

_URGENCY_TIER_MAP = {
    "extreme": "life_support",
    "urgent": "critical",
    "express": "regular",
    "normal": "regular",
}

_EXTREME_EVIDENCE = (
    "cardiac arrest",
    "cpr",
    "life-threatening",
    "code red",
    "stroke",
    "hemorrhage",
    "active transfusion",
    "golden rescue window",
)

_CRITICAL_EVIDENCE = (
    "icu",
    "ventilator",
    "urgent",
    "serious clinical",
    "post-exposure",
    "backup unit",
    "urgent same-window",
)

_EMERGENCY_ROLES = {"emergency_doctor", "paramedic", "triage_nurse"}
_CRITICAL_ROLES = {"icu_nurse", "clinical_pharmacist", "ward_coordinator"}
_LIFE_CARGO = {"aed", "blood_product", "cardiac_drug", "thrombolytic"}
_CRITICAL_CARGO = {"ventilator", "icu_drug"}

_SCORE_BONUSES = {
    "deadline_le_15m": 30,
    "deadline_le_30m": 20,
    "deadline_le_60m": 10,
    "hard_deadline": 6,
    "life_cargo": 24,
    "critical_cargo": 14,
    "life_threatening_context": 34,
    "critical_clinical_context": 18,
    "emergency_requester": 12,
    "critical_requester": 8,
    "children_involved": 10,
    "elderly_involved": 8,
    "vulnerable_community": 6,
    "handoff_ready": 4,
    "special_handling": 4,
    "destination_hospital_like": 4,
    "solver_ready_receiver": 6,
    "solver_routeable": 2,
}

_PRIORITY_THRESHOLDS = (
    (125, 1),
    (82, 2),
    (42, 3),
)


def normalize_priority(priority: object, default: int = 4) -> int:
    try:
        value = int(priority)
    except (TypeError, ValueError):
        value = default
    return min(max(value, 1), 4)


def get_demand_tier(demand: Dict) -> str:
    tier = demand.get("demand_tier") or demand.get("cargo", {}).get("demand_tier")
    if tier and tier in TIER_PRIORITIES:
        return tier
    urgency = demand.get("urgency", "normal")
    return _URGENCY_TIER_MAP.get(urgency, "regular")


def _as_dict(value: object) -> Dict:
    return value if isinstance(value, dict) else {}


def _extract_vulnerability(demand: Dict) -> Dict:
    signals = _as_dict(demand.get("priority_evaluation_signals", {}))
    vuln = _as_dict(signals.get("population_vulnerability", {}))
    elderly_ratio_raw = vuln.get("elderly_ratio", 0.0) or 0.0
    try:
        elderly_ratio = float(elderly_ratio_raw)
    except (TypeError, ValueError):
        elderly_ratio = 0.0
    return {
        "elderly_ratio": elderly_ratio,
        "elderly_involved": bool(vuln.get("elderly_involved", False)),
        "vulnerable_community": bool(vuln.get("vulnerable_community", False)),
        "children_involved": bool(vuln.get("children_involved", False)),
    }


def _collect_text_evidence(demand: Dict) -> str:
    signals = _as_dict(demand.get("priority_evaluation_signals", {}))
    destination = _as_dict(demand.get("destination", {}))
    cargo = _as_dict(demand.get("cargo", {}))
    evidence_parts = [
        signals.get("patient_condition", ""),
        signals.get("time_sensitivity", ""),
        signals.get("medical_urgency_self_report", ""),
        signals.get("scenario_context", ""),
        signals.get("nearby_critical_facility", ""),
        signals.get("requester_role", ""),
        signals.get("operational_readiness", ""),
        demand.get("operational_readiness", ""),
        " ".join(str(item) for item in (signals.get("special_handling") or [])),
        " ".join(str(item) for item in (demand.get("special_handling") or [])),
        " ".join(str(item) for item in (demand.get("context_signals") or [])),
        destination.get("type", ""),
        cargo.get("type", ""),
    ]
    return " ".join(str(part or "") for part in evidence_parts).lower()


def _extract_deadline_minutes(demand: Dict) -> int:
    time_constraint = demand.get("time_constraint", {})
    raw = (
        time_constraint.get("deadline_minutes")
        or demand.get("deadline_minutes")
        or demand.get("deadline")
        or 0
    )
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def _priority_from_score(
    score: int,
    tier: str,
    demand: Dict,
    evidence_text: str,
) -> int:
    priority = 4
    for threshold, mapped_priority in _PRIORITY_THRESHOLDS:
        if score >= threshold:
            priority = mapped_priority
            break

    cargo_type = str(demand.get("cargo", {}).get("type", "") or "")
    if tier == "life_support":
        return 1
    if tier == "critical":
        return min(priority, 2)
    if tier == "regular":
        if "post-exposure" in evidence_text or _extract_deadline_minutes(demand) <= 30:
            return min(priority, 2)
        return max(priority, 3)
    if tier == "consumer":
        if cargo_type == "otc_drug" and any(
            keyword in evidence_text for keyword in ("fever", "child", "pediatric", "elderly")
        ):
            return min(priority, 3)
        return max(priority, 4)
    return normalize_priority(priority)


def derive_priority_assessment(demand: Dict, *, solver_mode: bool = False) -> Dict:
    tier = get_demand_tier(demand)
    score = _TIER_BASE_SCORE.get(tier, 42)
    reasons = [f"tier={tier}"]
    signals = demand.get("priority_evaluation_signals", {})
    evidence_text = _collect_text_evidence(demand)
    vuln = _extract_vulnerability(demand)
    deadline_minutes = _extract_deadline_minutes(demand)
    requester_role = str(
        signals.get("requester_role")
        or demand.get("requester_role")
        or ""
    )
    cargo_type = str(demand.get("cargo", {}).get("type", "") or "")
    destination_type = str(demand.get("destination", {}).get("type", "") or "")
    special_handling = list(
        signals.get("special_handling")
        or demand.get("special_handling")
        or []
    )

    if deadline_minutes:
        if deadline_minutes <= 15:
            score += _SCORE_BONUSES["deadline_le_15m"]
            reasons.append("deadline<=15m")
        elif deadline_minutes <= 30:
            score += _SCORE_BONUSES["deadline_le_30m"]
            reasons.append("deadline<=30m")
        elif deadline_minutes <= 60:
            score += _SCORE_BONUSES["deadline_le_60m"]
            reasons.append("deadline<=60m")
    if str(demand.get("time_constraint", {}).get("type", "")).lower() == "hard":
        score += _SCORE_BONUSES["hard_deadline"]
        reasons.append("hard_deadline")

    if cargo_type in _LIFE_CARGO:
        score += _SCORE_BONUSES["life_cargo"]
        reasons.append(f"cargo={cargo_type}")
    elif cargo_type in _CRITICAL_CARGO:
        score += _SCORE_BONUSES["critical_cargo"]
        reasons.append(f"cargo={cargo_type}")

    if any(keyword in evidence_text for keyword in _EXTREME_EVIDENCE):
        score += _SCORE_BONUSES["life_threatening_context"]
        reasons.append("life_threatening_context")
    elif any(keyword in evidence_text for keyword in _CRITICAL_EVIDENCE):
        score += _SCORE_BONUSES["critical_clinical_context"]
        reasons.append("critical_clinical_context")

    if requester_role in _EMERGENCY_ROLES:
        score += _SCORE_BONUSES["emergency_requester"]
        reasons.append(f"role={requester_role}")
    elif requester_role in _CRITICAL_ROLES:
        score += _SCORE_BONUSES["critical_requester"]
        reasons.append(f"role={requester_role}")

    if vuln["children_involved"]:
        score += _SCORE_BONUSES["children_involved"]
        reasons.append("children_involved")
    if vuln["elderly_involved"]:
        score += _SCORE_BONUSES["elderly_involved"]
        reasons.append("elderly_involved")
    if vuln["vulnerable_community"]:
        score += _SCORE_BONUSES["vulnerable_community"]
        reasons.append("vulnerable_community")

    if "ready" in evidence_text or "standing by" in evidence_text or "handoff" in evidence_text:
        score += _SCORE_BONUSES["handoff_ready"]
        reasons.append("handoff_ready")
    if any(item in {"cold_chain", "shock_protection"} for item in special_handling):
        score += _SCORE_BONUSES["special_handling"]
        reasons.append("special_handling")
    if destination_type in {"hospital", "clinic"} and tier in {"life_support", "critical"}:
        score += _SCORE_BONUSES["destination_hospital_like"]
        reasons.append(f"destination={destination_type}")

    if solver_mode:
        if demand.get("receiver_ready") or "landing zone cleared" in evidence_text:
            score += _SCORE_BONUSES["solver_ready_receiver"]
            reasons.append("solver_ready_receiver")
        if demand.get("origin", {}).get("fid") and demand.get("destination", {}).get("fid"):
            score += _SCORE_BONUSES["solver_routeable"]
            reasons.append("solver_routeable")

    priority = _priority_from_score(score, tier, demand, evidence_text)
    reasoning = ", ".join(reasons[:5])
    mode = "solver_useful" if solver_mode else "extraction_observable"
    return {
        "demand_id": str(demand.get("demand_id", "")),
        "demand_tier": tier,
        "priority": priority,
        "score": score,
        "reasoning": reasoning,
        "mode": mode,
        "factor_hits": reasons,
    }


def derive_priority_labels(
    demand: Dict,
    *,
    latent_priority: int | None = None,
    dialogue_observable_priority: int | None = None,
) -> Dict:
    existing_labels = demand.get("labels", {})
    latent = latent_priority
    if latent is None:
        latent = existing_labels.get("latent_priority", demand.get("latent_priority"))
    dialogue_observable = dialogue_observable_priority
    if dialogue_observable is None:
        dialogue_observable = existing_labels.get(
            "dialogue_observable_priority",
            demand.get("dialogue_observable_priority"),
        )

    extraction = derive_priority_assessment(demand, solver_mode=False)
    solver = derive_priority_assessment(demand, solver_mode=True)
    labels = {
        "priority_policy_version": PRIORITY_POLICY_VERSION,
        "extraction_observable_priority": extraction["priority"],
        "extraction_observable_score": extraction["score"],
        "extraction_observable_reasoning": extraction["reasoning"],
        "solver_useful_priority": solver["priority"],
        "solver_useful_score": solver["score"],
        "solver_useful_reasoning": solver["reasoning"],
    }
    if latent is not None:
        labels["latent_priority"] = normalize_priority(latent)
    if dialogue_observable is not None:
        labels["dialogue_observable_priority"] = normalize_priority(dialogue_observable)
    return labels


def attach_priority_labels(
    demand: Dict,
    *,
    latent_priority: int | None = None,
    dialogue_observable_priority: int | None = None,
) -> Dict:
    labels = dict(demand.get("labels", {}))
    labels.update(
        derive_priority_labels(
            demand,
            latent_priority=latent_priority,
            dialogue_observable_priority=dialogue_observable_priority,
        )
    )
    demand["labels"] = labels
    demand["extraction_observable_priority"] = labels["extraction_observable_priority"]
    demand["solver_useful_priority"] = labels["solver_useful_priority"]
    return demand


def _resolve_priority_value(demand: Dict, priority_field: str) -> tuple[int, int, str, Dict]:
    working = copy.deepcopy(demand)
    attach_priority_labels(working)
    labels = working.get("labels", {})
    priority = normalize_priority(
        labels.get(priority_field, working.get(priority_field)),
        default=4,
    )
    score_field = priority_field.replace("_priority", "_score")
    reasoning_field = priority_field.replace("_priority", "_reasoning")
    score = int(labels.get(score_field, 0) or 0)
    reasoning = str(labels.get(reasoning_field, "") or "")
    return priority, score, reasoning, working


def rank_demands_for_window(
    demands: List[Dict],
    *,
    priority_field: str = "extraction_observable_priority",
) -> List[Dict]:
    ranked_rows = []
    for demand in demands:
        priority, score, reasoning, working = _resolve_priority_value(demand, priority_field)
        labels = working.get("labels", {})
        ranked_rows.append({
            "demand_id": str(working.get("demand_id", "")),
            "priority": priority,
            "score": score,
            "reasoning": reasoning,
            "demand_tier": get_demand_tier(working),
            "labels": labels,
        })

    ranked_rows.sort(
        key=lambda item: (
            int(item["priority"]),
            -int(item["score"]),
            str(item["demand_id"]),
        )
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["window_rank"] = rank
    return ranked_rows


def build_window_priority_targets(
    demands: List[Dict],
    *,
    priority_field: str = "extraction_observable_priority",
    urgent_threshold: int = 2,
) -> Dict:
    ranked_rows = rank_demands_for_window(demands, priority_field=priority_field)
    demand_configs = [
        {
            "demand_id": row["demand_id"],
            "demand_tier": row["demand_tier"],
            "priority": row["priority"],
            "window_rank": row["window_rank"],
            "reasoning": row["reasoning"],
        }
        for row in ranked_rows
    ]

    pairwise_preferences = []
    for higher_idx, higher in enumerate(ranked_rows):
        for lower in ranked_rows[higher_idx + 1:]:
            if higher["priority"] == lower["priority"]:
                continue
            pairwise_preferences.append({
                "higher_priority_demand_id": higher["demand_id"],
                "lower_priority_demand_id": lower["demand_id"],
                "priority_gap": int(lower["priority"]) - int(higher["priority"]),
            })

    critical_topk_targets = [
        row["demand_id"]
        for row in ranked_rows
        if int(row["priority"]) <= urgent_threshold
    ]

    return {
        "demand_configs": demand_configs,
        "pairwise_preferences": pairwise_preferences,
        "critical_topk_targets": critical_topk_targets,
    }
