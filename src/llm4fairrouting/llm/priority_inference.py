"""
Module 3a: Priority Ranking — 根据结构化需求，由 LLM 分配优先级并生成补充约束建议。
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from openai import OpenAI

from llm4fairrouting.config.runtime_env import env_text, prepare_env_file
from llm4fairrouting.llm.client_utils import (
    call_llm,
    create_openai_client,
    parse_json_response,
)


# ============================================================================
# Priority adjustment
# ============================================================================

# Base tier priorities (1 = highest)
TIER_PRIORITIES = {
    "life_support": 1,
    "critical": 2,
    "regular": 3,
    "consumer": 4,
}

# Score anchor for within-window ranking
_TIER_BASE_SCORE = {
    "life_support": 100,
    "critical": 72,
    "regular": 42,
    "consumer": 12,
}

# Compatibility with the older urgency field
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


def _get_tier(demand: Dict) -> str:
    """从需求字典中获取 demand_tier，兼容旧版 urgency 字段。"""
    tier = demand.get("demand_tier") or demand.get("cargo", {}).get("demand_tier")
    if tier and tier in TIER_PRIORITIES:
        return tier
    urgency = demand.get("urgency", "normal")
    return _URGENCY_TIER_MAP.get(urgency, "regular")


def _normalize_priority(priority: object, default: int = 4) -> int:
    try:
        value = int(priority)
    except (TypeError, ValueError):
        value = default
    return min(max(value, 1), 4)


def _normalize_weight_config(result: Dict) -> Dict:
    demand_configs = []
    for rank, config in enumerate(result.get("demand_configs", []), start=1):
        demand_config = {
            "demand_id": str(config.get("demand_id", "")).strip(),
            "priority": _normalize_priority(config.get("priority"), default=4),
            "window_rank": int(config.get("window_rank", rank)),
            "reasoning": str(config.get("reasoning", "")).strip(),
        }
        demand_tier = str(config.get("demand_tier", "")).strip()
        if demand_tier:
            demand_config["demand_tier"] = demand_tier
        demand_configs.append(demand_config)

    return {
        "global_weights": result.get(
            "global_weights",
            {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
        ),
        "demand_configs": demand_configs,
        "supplementary_constraints": list(result.get("supplementary_constraints", [])),
    }


def _extract_vulnerability(demand: Dict) -> Dict:
    """Extract vulnerability signals from structured fields and text fallbacks."""
    signals = demand.get("priority_evaluation_signals", {})
    vuln = signals.get("population_vulnerability", {})

    elderly_ratio = vuln.get("elderly_ratio", 0.0)
    elderly_involved = vuln.get("elderly_involved", False)
    vulnerable_community = vuln.get("vulnerable_community", False)
    children_involved = vuln.get("children_involved", False)

    # fallback: parse context_signals text
    if not elderly_ratio:
        for sig in demand.get("context_signals", []):
            if "老年" in sig or "elderly_ratio" in sig:
                try:
                    ratio = float(sig.split("老年比例")[-1].strip())
                    elderly_ratio = ratio
                    elderly_involved = ratio > 0.40
                    vulnerable_community = ratio > 0.50
                except (ValueError, IndexError):
                    pass

    return {
        "elderly_ratio": elderly_ratio,
        "elderly_involved": elderly_involved,
        "vulnerable_community": vulnerable_community,
        "children_involved": children_involved,
    }


def _collect_text_evidence(demand: Dict) -> str:
    signals = demand.get("priority_evaluation_signals", {})
    evidence_parts = [
        signals.get("patient_condition", ""),
        signals.get("time_sensitivity", ""),
        signals.get("medical_urgency_self_report", ""),
        signals.get("scenario_context", ""),
        signals.get("nearby_critical_facility", ""),
        signals.get("requester_role", ""),
        signals.get("operational_readiness", ""),
        " ".join(str(item) for item in signals.get("special_handling", [])),
        " ".join(str(item) for item in demand.get("context_signals", [])),
        demand.get("destination", {}).get("type", ""),
        demand.get("cargo", {}).get("type", ""),
    ]
    return " ".join(str(part or "") for part in evidence_parts).lower()


def _extract_deadline_minutes(demand: Dict) -> int:
    time_constraint = demand.get("time_constraint", {})
    try:
        return max(0, int(time_constraint.get("deadline_minutes", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _priority_from_score(
    score: int,
    tier: str,
    demand: Dict,
    evidence_text: str,
) -> int:
    priority = 4
    if score >= 125:
        priority = 1
    elif score >= 82:
        priority = 2
    elif score >= 42:
        priority = 3

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
    return min(max(priority, 1), 4)


def _assess_demand_priority(demand: Dict) -> Dict:
    tier = _get_tier(demand)
    score = _TIER_BASE_SCORE.get(tier, 42)
    reasons = [f"tier={tier}"]
    signals = demand.get("priority_evaluation_signals", {})
    evidence_text = _collect_text_evidence(demand)
    vuln = _extract_vulnerability(demand)
    deadline_minutes = _extract_deadline_minutes(demand)
    requester_role = str(signals.get("requester_role", "") or "")
    cargo_type = str(demand.get("cargo", {}).get("type", "") or "")
    destination_type = str(demand.get("destination", {}).get("type", "") or "")
    special_handling = signals.get("special_handling", [])

    if deadline_minutes:
        if deadline_minutes <= 15:
            score += 30
            reasons.append("deadline<=15m")
        elif deadline_minutes <= 30:
            score += 20
            reasons.append("deadline<=30m")
        elif deadline_minutes <= 60:
            score += 10
            reasons.append("deadline<=60m")
    if str(demand.get("time_constraint", {}).get("type", "")).lower() == "hard":
        score += 6
        reasons.append("hard_deadline")

    if cargo_type in _LIFE_CARGO:
        score += 24
        reasons.append(f"cargo={cargo_type}")
    elif cargo_type in _CRITICAL_CARGO:
        score += 14
        reasons.append(f"cargo={cargo_type}")

    if any(keyword in evidence_text for keyword in _EXTREME_EVIDENCE):
        score += 34
        reasons.append("life_threatening_context")
    elif any(keyword in evidence_text for keyword in _CRITICAL_EVIDENCE):
        score += 18
        reasons.append("critical_clinical_context")

    if requester_role in _EMERGENCY_ROLES:
        score += 12
        reasons.append(f"role={requester_role}")
    elif requester_role in _CRITICAL_ROLES:
        score += 8
        reasons.append(f"role={requester_role}")

    if vuln["children_involved"]:
        score += 10
        reasons.append("children_involved")
    if vuln["elderly_involved"]:
        score += 8
        reasons.append("elderly_involved")
    if vuln["vulnerable_community"]:
        score += 6
        reasons.append("vulnerable_community")

    if "ready" in evidence_text or "standing by" in evidence_text or "handoff" in evidence_text:
        score += 4
        reasons.append("handoff_ready")
    if any(item in {"cold_chain", "shock_protection"} for item in special_handling):
        score += 4
        reasons.append("special_handling")
    if destination_type in {"hospital", "clinic"} and tier in {"life_support", "critical"}:
        score += 4
        reasons.append(f"destination={destination_type}")

    priority = _priority_from_score(score, tier, demand, evidence_text)
    reasoning = ", ".join(reasons[:4])
    return {
        "demand_id": demand["demand_id"],
        "demand_tier": tier,
        "priority": priority,
        "score": score,
        "reasoning": reasoning,
    }


def _build_supplementary_constraints(demands: List[Dict]) -> List[Dict]:
    constraints: List[Dict] = []
    for demand in demands:
        signals = demand.get("priority_evaluation_signals", {})
        evidence_text = _collect_text_evidence(demand)
        if any(keyword in evidence_text for keyword in ("school", "kindergarten")):
            destination = demand.get("destination", {})
            constraints.append({
                "type": "noise_avoidance",
                "description": f"Avoid low-altitude flight near {destination.get('fid', '')} because of a nearby school zone.",
                "affected_zone": {
                    "center": destination.get("coords", [0, 0]),
                    "radius_m": 300,
                },
            })
        if "cold_chain" in signals.get("special_handling", []):
            constraints.append({
                "type": "speed_override",
                "description": f"Prioritize a direct path for {demand.get('demand_id')} to reduce thermal exposure.",
            })
    deduped = []
    seen = set()
    for constraint in constraints:
        key = json.dumps(constraint, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            deduped.append(constraint)
    return deduped


def _build_rule_weight_config(demands: List[Dict]) -> Dict:
    assessments = [_assess_demand_priority(demand) for demand in demands]
    assessments.sort(key=lambda item: (-item["score"], item["priority"], item["demand_id"]))

    demand_configs = []
    for rank, assessment in enumerate(assessments, start=1):
        demand_configs.append({
            "demand_id": assessment["demand_id"],
            "demand_tier": assessment["demand_tier"],
            "priority": assessment["priority"],
            "window_rank": rank,
            "reasoning": assessment["reasoning"],
        })

    return {
        "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
        "demand_configs": demand_configs,
        "supplementary_constraints": _build_supplementary_constraints(demands),
    }


def _merge_weight_configs(
    demands: List[Dict],
    rule_result: Dict,
    llm_result: Dict,
) -> Dict:
    rule_map = {item["demand_id"]: item for item in rule_result.get("demand_configs", [])}
    llm_map = {item["demand_id"]: item for item in llm_result.get("demand_configs", []) if item.get("demand_id")}
    combined_rows = []

    for demand in demands:
        demand_id = demand["demand_id"]
        rule_cfg = rule_map.get(demand_id, {
            "demand_id": demand_id,
            "demand_tier": _get_tier(demand),
            "priority": 4,
            "reasoning": "",
            "window_rank": len(combined_rows) + 1,
        })
        llm_cfg = llm_map.get(demand_id)
        final_priority = int(rule_cfg["priority"])
        combined_score = (5 - final_priority) * 20
        reasoning_parts = [str(rule_cfg.get("reasoning", "")).strip()]

        if llm_cfg:
            llm_priority = _normalize_priority(llm_cfg.get("priority"), default=final_priority)
            strong_rule = final_priority == 1 or abs(final_priority - llm_priority) <= 1
            if not strong_rule:
                final_priority = llm_priority
            combined_score += (5 - llm_priority) * 8
            llm_reason = str(llm_cfg.get("reasoning", "")).strip()
            if llm_reason:
                reasoning_parts.append(llm_reason)

        combined_rows.append({
            "demand_id": demand_id,
            "demand_tier": rule_cfg.get("demand_tier", _get_tier(demand)),
            "priority": final_priority,
            "score": combined_score,
            "reasoning": " | ".join(part for part in reasoning_parts if part),
        })

    combined_rows.sort(key=lambda item: (-item["score"], item["priority"], item["demand_id"]))
    demand_configs = []
    for rank, row in enumerate(combined_rows, start=1):
        demand_configs.append({
            "demand_id": row["demand_id"],
            "demand_tier": row["demand_tier"],
            "priority": row["priority"],
            "window_rank": rank,
            "reasoning": row["reasoning"],
        })

    supplementary = []
    seen_constraints = set()
    for constraint in rule_result.get("supplementary_constraints", []) + llm_result.get("supplementary_constraints", []):
        key = json.dumps(constraint, sort_keys=True, ensure_ascii=False)
        if key not in seen_constraints:
            seen_constraints.add(key)
            supplementary.append(constraint)

    return {
        "global_weights": llm_result.get(
            "global_weights",
            rule_result.get("global_weights", {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0}),
        ),
        "demand_configs": demand_configs,
        "supplementary_constraints": supplementary,
    }


def adjust_weights(
    demands: List[Dict],
    client: "OpenAI",
    model: str,
    city_context: Optional[Dict] = None,
    temperature: float = 0.0,
) -> Dict:
    """Run LLM-based priority inference and reconcile it with deterministic evidence scoring."""
    from llm4fairrouting.llm.prompt_templates import (
        DRONE_SYSTEM_PROMPT,
        weight_adjustment_prompt,
    )

    prompt = weight_adjustment_prompt(demands, city_context)
    print(f"  [Module 3a] Ranking {len(demands)} demands with LLM + evidence scorer")

    raw = call_llm(client, model, DRONE_SYSTEM_PROMPT, prompt, temperature)
    llm_result = _normalize_weight_config(parse_json_response(raw))
    rule_result = _build_rule_weight_config(demands)
    merged = _merge_weight_configs(demands, rule_result, llm_result)

    n_configs = len(merged.get("demand_configs", []))
    n_supp = len(merged.get("supplementary_constraints", []))
    print(f"  [Module 3a] Produced {n_configs} demand configs and {n_supp} supplementary constraints")
    return merged


def adjust_weights_offline(demands: List[Dict]) -> Dict:
    """Offline Module 3a path using deterministic evidence scoring."""
    return _build_rule_weight_config(demands)


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    active_env_file = prepare_env_file(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Module 3a: Weight Adjustment")
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(active_env_file) if active_env_file else None,
        help="Environment file path; defaults to the project .env when present",
    )
    parser.add_argument(
        "--input", type=str,
        default=str(PROJECT_ROOT / "data" / "drone" / "extracted_demands.json"),
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--api-base", type=str, default=env_text("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", type=str, default=env_text("OPENAI_API_KEY"))
    parser.add_argument("--model", type=str, default=env_text("LLM4FAIRROUTING_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        windows_data = json.load(f)

    all_results = []
    for window in windows_data:
        demands = window.get("demands", [])
        tw = window.get("time_window", "")
        print(f"[Module 3a] Window {tw}: {len(demands)} demands")

        if args.offline:
            result = adjust_weights_offline(demands)
        else:
            client = create_openai_client(args.api_base, args.api_key)
            result = adjust_weights(demands, client, args.model)

        result["time_window"] = tw
        all_results.append(result)

    out_path = args.output or str(PROJECT_ROOT / "data" / "drone" / "weight_configs.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
