"""
Module 2: Context Extraction — 按时间窗口聚合对话，调用 LLM 提取结构化需求。
"""

import json
import re
from collections import defaultdict
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


_CARGO_TEXT_MAP = {
    "aed defibrillator": ("aed", "AED defibrillator", False),
    "aed": ("aed", "AED defibrillator", False),
    "blood product": ("blood_product", "blood product", True),
    "cardiac emergency drug": ("cardiac_drug", "cardiac emergency drug", False),
    "thrombolytic agent": ("thrombolytic", "thrombolytic agent", True),
    "ventilator": ("ventilator", "ventilator", False),
    "icu medication": ("icu_drug", "ICU medication", False),
    "vaccine": ("vaccine", "vaccine", True),
    "medication": ("medicine", "medication", False),
    "medicine": ("medicine", "medication", False),
    "protective suit": ("protective_suit", "protective suit", False),
    "face mask": ("mask", "face mask", False),
    "mask": ("mask", "face mask", False),
    "disinfectant solution": ("disinfectant", "disinfectant solution", False),
    "disinfectant": ("disinfectant", "disinfectant solution", False),
    "food/meal": ("food", "food/meal", False),
    "meal": ("food", "food/meal", False),
    "otc medication": ("otc_drug", "OTC medication", False),
    "daily supplies": ("daily_supply", "daily supplies", False),
}

_ROLE_HINTS = {
    "er physician": "emergency_doctor",
    "emergency": "emergency_doctor",
    "paramedic": "paramedic",
    "triage nurse": "triage_nurse",
    "icu nurse": "icu_nurse",
    "clinical pharmacist": "clinical_pharmacist",
    "ward coordinator": "ward_coordinator",
    "community health worker": "community_health_worker",
    "clinic manager": "clinic_manager",
    "pharmacy staff": "pharmacy_staff",
    "customer": "consumer",
    "family caregiver": "family_caregiver",
    "office administrator": "office_administrator",
    "user": "consumer",
}

_EXTREME_URGENCY_KEYWORDS = (
    "code red",
    "cardiac arrest",
    "cpr",
    "resuscitation",
    "life-threatening",
    "immediate support",
    "golden rescue window",
    "stroke window",
    "active transfusion",
)

_CRITICAL_URGENCY_KEYWORDS = (
    "urgent",
    "icu",
    "critical clinical",
    "post-exposure",
    "biomedical",
    "serious",
    "backup unit",
)

_CHILD_KEYWORDS = ("child", "children", "pediatric", "infant", "school")
_ELDERLY_KEYWORDS = ("elderly", "senior", "older", "72-year-old", "80-year-old", "grandparent")
_READYNESS_KEYWORDS = (
    "landing zone",
    "standing by",
    "ready for handoff",
    "receive the payload",
    "receive it at",
    "sign for it",
    "notification",
)
_COLD_CHAIN_KEYWORDS = ("cold-chain", "insulated", "temperature", "vaccine box")
_SHOCK_PROTECTION_KEYWORDS = ("shock-proof", "reinforced brackets", "secure the unit")


# ============================================================================
# 时间窗口分组
# ============================================================================

def group_by_time_window(
    dialogues: List[Dict],
    window_minutes: int = 5,
) -> Dict[str, List[Dict]]:
    """将对话按 ``window_minutes`` 分钟的时间窗口分组。

    返回 ``{window_label: [dialogues]}``，window_label 形如
    ``"2024-03-15T00:00-00:05"``。

    正确处理小时边界（如 00:55-01:00）。
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)

    for d in dialogues:
        ts = d["timestamp"]  # ISO format: "2024-03-15T00:05:00"
        hour = int(ts[11:13])
        minute = int(ts[14:16])

        # 从午夜起的绝对分钟数，方便处理跨小时边界
        abs_start = (hour * 60 + minute) // window_minutes * window_minutes
        abs_end = abs_start + window_minutes

        h_start, m_start = divmod(abs_start, 60)
        h_end, m_end = divmod(abs_end, 60)

        date_part = ts[:10]
        label = (
            f"{date_part}T{h_start:02d}:{m_start:02d}"
            f"-{h_end:02d}:{m_end:02d}"
        )
        groups[label].append(d)

    for label in groups:
        groups[label].sort(key=lambda d: d["timestamp"])

    return dict(sorted(groups.items()))


# ============================================================================
# 坐标回填：LLM 只看文本，结构化坐标/fid 事后从 metadata 注入
# ============================================================================

def _enrich_demands_with_metadata(demands: List[Dict], dialogues: List[Dict]) -> List[Dict]:
    """LLM 提取结束后，将原始 dialogue metadata 中的坐标/fid 回填到需求记录。

    Module 2 的 LLM 只看对话文本，不知道坐标。但 solver 需要精确坐标，
    因此在 LLM 返回后由代码把坐标从 metadata 注入，不影响 LLM 的语义理解。
    """
    dlg_lookup = {d["dialogue_id"]: d for d in dialogues}

    for demand in demands:
        src_id = demand.get("source_dialogue_id")
        if not src_id or src_id not in dlg_lookup:
            continue
        dialogue = dlg_lookup[src_id]
        meta = dialogue["metadata"]

        origin = demand.setdefault("origin", {})
        origin["fid"]    = meta.get("origin_fid", "")
        origin["coords"] = meta.get("origin_coords", [0.0, 0.0])

        dest = demand.setdefault("destination", {})
        dest["fid"]    = meta.get("destination_fid", 0)
        dest["coords"] = meta.get("dest_coords", [0.0, 0.0])

        demand["request_timestamp"] = dialogue.get("timestamp")
        demand["source_event_id"] = meta.get("event_id", demand.get("source_event_id"))

    return demands


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _extract_first_role(conversation: str) -> str:
    for line in str(conversation or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^\[\d{2}:\d{2}\]\s*([^:]+):", stripped)
        if match:
            return match.group(1).strip()
    return ""


def _infer_requester_role(conversation: str, metadata: Dict) -> str:
    role_text = _normalize_text(_extract_first_role(conversation))
    for hint, label in _ROLE_HINTS.items():
        if hint in role_text:
            return label
    return str(metadata.get("requester_role", "community_health_worker"))


def _infer_cargo(conversation: str, metadata: Dict) -> Dict:
    text = _normalize_text(conversation)
    for needle, (cargo_type, type_cn, temp_sensitive) in _CARGO_TEXT_MAP.items():
        if needle in text:
            return {
                "type": cargo_type,
                "type_cn": type_cn,
                "temperature_sensitive": temp_sensitive,
            }
    cargo_type = str(metadata.get("material_type", "medicine"))
    return {
        "type": cargo_type,
        "type_cn": cargo_type.replace("_", " "),
        "temperature_sensitive": cargo_type in {"vaccine", "blood_product", "thrombolytic"},
    }


def _extract_deadline_minutes(conversation: str, default: int = 90) -> int:
    patterns = (
        r"(?:eta|arrival(?:\s+within)?|within|under|in)\s*(?:about\s*)?(?:<=\s*)?(\d{1,3})\s*min",
        r"(\d{1,3})\s*minute(?:s)?",
    )
    text = _normalize_text(conversation)
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return max(5, min(int(match.group(1)), 240))
            except ValueError:
                continue
    return max(5, min(int(default or 90), 240))


def _infer_patient_condition(conversation: str, cargo_type: str) -> str:
    text = _normalize_text(conversation)
    if any(keyword in text for keyword in _EXTREME_URGENCY_KEYWORDS):
        if "cpr" in text or "cardiac arrest" in text:
            return "Cardiac arrest response in progress"
        if "stroke window" in text:
            return "Acute stroke treatment window is closing"
        return "Life-threatening condition with immediate intervention needed"
    if any(keyword in text for keyword in _CRITICAL_URGENCY_KEYWORDS):
        if cargo_type == "ventilator":
            return "Respiratory support risk; backup ventilator needed"
        if "post-exposure" in text:
            return "Post-exposure prophylaxis window is narrow"
        return "Serious clinical condition requiring urgent support"
    if cargo_type == "otc_drug" and "fever" in text:
        return "Home symptom relief request with elevated urgency"
    return "Routine supply request"


def _infer_demand_tier(
    conversation: str,
    cargo_type: str,
    requester_role: str,
    deadline_minutes: int,
) -> str:
    text = _normalize_text(conversation)
    if cargo_type in {"aed", "blood_product", "cardiac_drug", "thrombolytic"}:
        return "life_support"
    if any(keyword in text for keyword in _EXTREME_URGENCY_KEYWORDS):
        return "life_support"
    if cargo_type in {"ventilator", "icu_drug"}:
        return "critical"
    if any(keyword in text for keyword in _CRITICAL_URGENCY_KEYWORDS) or deadline_minutes <= 30:
        return "critical"
    if requester_role in {"consumer", "family_caregiver", "office_administrator"} or "locker" in text:
        if cargo_type == "otc_drug" and any(keyword in text for keyword in ("fever", "pain spike", "child")):
            return "regular"
        return "consumer"
    if cargo_type in {"food", "daily_supply", "otc_drug"}:
        return "consumer"
    return "regular"


def _infer_destination_type(conversation: str, requester_role: str) -> str:
    text = _normalize_text(conversation)
    if any(keyword in text for keyword in ("icu", "er ", "emergency", "ward", "hospital")):
        return "hospital"
    if any(keyword in text for keyword in ("clinic", "community health", "vaccination")):
        return "clinic"
    if "pharmacy" in text:
        return "pharmacy"
    if requester_role in {"consumer", "family_caregiver", "office_administrator"}:
        if "office" in text:
            return "office"
        return "residential_area"
    return "public_space"


def _infer_facility_context(conversation: str, destination_type: str) -> Optional[str]:
    text = _normalize_text(conversation)
    if "school" in text:
        return "school"
    if "community locker" in text or destination_type == "residential_area":
        return "residential_area"
    if destination_type == "hospital":
        return "hospital"
    if destination_type == "clinic":
        return "clinic"
    if destination_type == "pharmacy":
        return "pharmacy"
    return None


def _infer_population_vulnerability(conversation: str) -> Dict:
    text = _normalize_text(conversation)
    children = any(keyword in text for keyword in _CHILD_KEYWORDS)
    elderly = any(keyword in text for keyword in _ELDERLY_KEYWORDS)
    return {
        "elderly_involved": elderly,
        "children_involved": children,
        "vulnerable_community": elderly or children,
    }


def _infer_time_sensitivity(deadline_minutes: int, demand_tier: str) -> str:
    if deadline_minutes <= 15 or demand_tier == "life_support":
        return "Immediate action required"
    if deadline_minutes <= 30 or demand_tier == "critical":
        return "Urgent same-window delivery required"
    if deadline_minutes <= 90:
        return "Timely delivery needed within the service window"
    return "Flexible same-day delivery"


def _infer_operational_readiness(conversation: str) -> str:
    text = _normalize_text(conversation)
    if any(keyword in text for keyword in _READYNESS_KEYWORDS):
        return "Receiver is ready and handoff can happen immediately"
    return "Standard handoff readiness"


def _infer_special_handling(conversation: str) -> List[str]:
    text = _normalize_text(conversation)
    handling: List[str] = []
    if any(keyword in text for keyword in _COLD_CHAIN_KEYWORDS):
        handling.append("cold_chain")
    if any(keyword in text for keyword in _SHOCK_PROTECTION_KEYWORDS):
        handling.append("shock_protection")
    return handling


def _build_context_signals(
    demand_tier: str,
    patient_condition: str,
    deadline_minutes: int,
    requester_role: str,
    operational_readiness: str,
    special_handling: List[str],
    vulnerability: Dict,
) -> List[str]:
    signals = [
        f"Tier inferred from dialogue: {demand_tier}",
        f"Patient or service context: {patient_condition}",
        f"Delivery target mentioned in dialogue: {deadline_minutes} min",
        f"Requester role inferred from dialogue: {requester_role}",
        operational_readiness,
    ]
    if special_handling:
        signals.append(f"Special handling: {', '.join(special_handling)}")
    if vulnerability.get("children_involved"):
        signals.append("Children are explicitly mentioned in the dialogue")
    if vulnerability.get("elderly_involved"):
        signals.append("An elderly beneficiary is explicitly mentioned in the dialogue")
    return signals


def _build_heuristic_demand(dialogue: Dict, demand_id: str) -> Dict:
    metadata = dialogue.get("metadata", {})
    conversation = str(dialogue.get("conversation", ""))
    requester_role = _infer_requester_role(conversation, metadata)
    cargo = _infer_cargo(conversation, metadata)
    deadline_minutes = _extract_deadline_minutes(
        conversation,
        default=int(metadata.get("delivery_deadline_minutes", 90)),
    )
    demand_tier = _infer_demand_tier(
        conversation,
        cargo_type=cargo["type"],
        requester_role=requester_role,
        deadline_minutes=deadline_minutes,
    )
    destination_type = _infer_destination_type(conversation, requester_role)
    patient_condition = _infer_patient_condition(conversation, cargo["type"])
    vulnerability = _infer_population_vulnerability(conversation)
    operational_readiness = _infer_operational_readiness(conversation)
    special_handling = _infer_special_handling(conversation)
    facility = _infer_facility_context(conversation, destination_type)

    return {
        "demand_id": demand_id,
        "source_dialogue_id": dialogue.get("dialogue_id"),
        "source_event_id": metadata.get("event_id"),
        "request_timestamp": dialogue.get("timestamp"),
        "origin": {
            "station_name": metadata.get("supply_station_name", ""),
            "type": "supply_station",
        },
        "destination": {
            "node_id": str(metadata.get("destination_fid", "")),
            "type": destination_type,
        },
        "cargo": {
            "type": cargo["type"],
            "type_cn": cargo["type_cn"],
            "demand_tier": demand_tier,
            "weight_kg": float(metadata.get("quantity_kg", 2.0)),
            "quantity": max(1, round(float(metadata.get("quantity_kg", 2.0)))),
            "quantity_unit": "units",
            "temperature_sensitive": cargo["temperature_sensitive"],
        },
        "demand_tier": demand_tier,
        "time_constraint": {
            "type": "hard" if deadline_minutes <= 30 or demand_tier in {"life_support", "critical"} else "soft",
            "description": f"Delivery target within {deadline_minutes} minutes",
            "deadline_minutes": deadline_minutes,
        },
        "priority_evaluation_signals": {
            "patient_condition": patient_condition,
            "time_sensitivity": _infer_time_sensitivity(deadline_minutes, demand_tier),
            "population_vulnerability": vulnerability,
            "medical_urgency_self_report": _infer_time_sensitivity(deadline_minutes, demand_tier),
            "requester_role": requester_role,
            "scenario_context": metadata.get("scenario_summary", patient_condition),
            "nearby_critical_facility": facility,
            "operational_readiness": operational_readiness,
            "special_handling": special_handling,
        },
        "context_signals": _build_context_signals(
            demand_tier=demand_tier,
            patient_condition=patient_condition,
            deadline_minutes=deadline_minutes,
            requester_role=requester_role,
            operational_readiness=operational_readiness,
            special_handling=special_handling,
            vulnerability=vulnerability,
        ),
    }


def _prefer_value(primary, fallback):
    if primary is None:
        return fallback
    if isinstance(primary, str) and not primary.strip():
        return fallback
    if isinstance(primary, list) and not primary:
        return fallback
    if isinstance(primary, dict) and not primary:
        return fallback
    return primary


def _merge_demand_records(heuristic: Dict, extracted: Dict) -> Dict:
    merged = {
        "demand_id": _prefer_value(extracted.get("demand_id"), heuristic["demand_id"]),
        "source_dialogue_id": _prefer_value(
            extracted.get("source_dialogue_id"), heuristic["source_dialogue_id"]
        ),
        "source_event_id": _prefer_value(
            extracted.get("source_event_id"), heuristic.get("source_event_id")
        ),
        "request_timestamp": _prefer_value(
            extracted.get("request_timestamp"), heuristic["request_timestamp"]
        ),
    }

    merged["origin"] = {
        **heuristic.get("origin", {}),
        **extracted.get("origin", {}),
    }
    merged["destination"] = {
        **heuristic.get("destination", {}),
        **extracted.get("destination", {}),
    }
    merged["cargo"] = {
        **heuristic.get("cargo", {}),
        **extracted.get("cargo", {}),
    }
    merged["demand_tier"] = _prefer_value(
        extracted.get("demand_tier"), heuristic.get("demand_tier")
    )
    merged["time_constraint"] = {
        **heuristic.get("time_constraint", {}),
        **extracted.get("time_constraint", {}),
    }
    merged["priority_evaluation_signals"] = {
        **heuristic.get("priority_evaluation_signals", {}),
        **extracted.get("priority_evaluation_signals", {}),
    }
    merged["context_signals"] = list(
        dict.fromkeys(
            heuristic.get("context_signals", []) + extracted.get("context_signals", [])
        )
    )
    return merged


def _normalize_extracted_window(result: Dict, dialogues: List[Dict]) -> Dict:
    extracted_demands = result.get("demands", [])
    by_dialogue_id = {
        str(item.get("source_dialogue_id")): item
        for item in extracted_demands
        if item.get("source_dialogue_id")
    }
    normalized_demands: List[Dict] = []
    for index, dialogue in enumerate(dialogues, start=1):
        heuristic = _build_heuristic_demand(dialogue, demand_id=f"REQ{index:03d}")
        extracted = by_dialogue_id.get(str(dialogue.get("dialogue_id")))
        if extracted is None and index - 1 < len(extracted_demands):
            extracted = extracted_demands[index - 1]
            extracted.setdefault("source_dialogue_id", dialogue.get("dialogue_id"))
        normalized_demands.append(_merge_demand_records(heuristic, extracted or {}))
    result["demands"] = _enrich_demands_with_metadata(normalized_demands, dialogues)
    return result


# ============================================================================
# 提取入口
# ============================================================================

def extract_demands_for_window(
    dialogues: List[Dict],
    time_window: str,
    client: "OpenAI",
    model: str,
    temperature: float = 0.0,
) -> Dict:
    """对单个时间窗口的对话调用 LLM，提取结构化需求，再回填坐标。

    LLM 只接收对话文本；坐标/fid 由 _enrich_demands_with_metadata 从
    原始 dialogue metadata 注入，保证 solver 可用。
    """
    from llm4fairrouting.llm.prompt_templates import (
        DRONE_SYSTEM_PROMPT,
        context_extraction_prompt,
    )
    prompt = context_extraction_prompt(dialogues, time_window)
    print(f"  [Module 2] Window {time_window}: extracting {len(dialogues)} dialogues with the LLM")

    raw = call_llm(client, model, DRONE_SYSTEM_PROMPT, prompt, temperature)
    result = _normalize_extracted_window(parse_json_response(raw), dialogues)

    n_demands = len(result.get("demands", []))
    print(f"  [Module 2] Extracted {n_demands} normalized demands")
    return result


def extract_all_demands(
    dialogues: List[Dict],
    client: "OpenAI",
    model: str,
    window_minutes: int = 5,
    temperature: float = 0.0,
) -> List[Dict]:
    """对所有对话按时间窗口分组，逐窗口提取需求。

    Returns
    -------
    list[dict]
        每个元素是一个时间窗口的提取结果::

            {"time_window": "...", "demands": [...]}
    """
    windows = group_by_time_window(dialogues, window_minutes)
    print(f"[Module 2] Grouped dialogues into {len(windows)} time windows")

    results = []
    for label, group in windows.items():
        result = extract_demands_for_window(group, label, client, model, temperature)
        results.append(result)

    return results


# ============================================================================
# 离线 / Mock 模式 — 不调用 LLM，直接从对话 metadata 构造需求
# ============================================================================

# 目的地类型映射（基于 nearby_poi 推断）
_POI_DEST_TYPE = {
    "hospital":                "hospital",
    "icu_unit":                "hospital",
    "emergency_room":          "hospital",
    "trauma_center":           "hospital",
    "surgery_room":            "hospital",
    "clinic":                  "clinic",
    "community_health_center": "clinic",
    "pharmacy":                "pharmacy",
    "residential":             "residential_area",
    "public_space":            "public_space",
    "office_building":         "office",
    "shopping_mall":           "commercial",
}


def _infer_dest_type(nearby_poi: List[str]) -> str:
    for poi in nearby_poi:
        if poi in _POI_DEST_TYPE:
            return _POI_DEST_TYPE[poi]
    return "residential_area"


def extract_demands_offline(dialogues: List[Dict], window_minutes: int = 5) -> List[Dict]:
    """Offline Module 2 path based on dialogue text plus safe metadata enrichment."""
    windows = group_by_time_window(dialogues, window_minutes)
    results = []

    for label, group in windows.items():
        normalized = _normalize_extracted_window(
            {"time_window": label, "demands": []},
            group,
        )

        for demand, dialogue in zip(normalized["demands"], group):
            meta = dialogue.get("metadata", {})
            demographics = meta.get("dest_demographics", {})
            elderly_ratio = float(demographics.get("elderly_ratio", 0.0) or 0.0)
            population = int(demographics.get("population", 0) or 0)
            nearby_poi = meta.get("nearby_poi", [])

            demand["destination"]["type"] = _infer_dest_type(nearby_poi) or demand["destination"]["type"]
            vulnerability = demand["priority_evaluation_signals"].setdefault(
                "population_vulnerability",
                {},
            )
            vulnerability["elderly_ratio"] = elderly_ratio
            vulnerability["population"] = population
            vulnerability["elderly_involved"] = vulnerability.get("elderly_involved", False) or elderly_ratio > 0.40
            vulnerability["vulnerable_community"] = vulnerability.get("vulnerable_community", False) or elderly_ratio > 0.50
            if nearby_poi and not demand["priority_evaluation_signals"].get("nearby_critical_facility"):
                demand["priority_evaluation_signals"]["nearby_critical_facility"] = nearby_poi[0]
            demand["context_signals"] = list(
                dict.fromkeys(
                    demand["context_signals"]
                    + [
                        f"Population near destination: {population}",
                        f"Elderly ratio near destination: {elderly_ratio:.0%}",
                    ]
                )
            )

        results.append(normalized)

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    active_env_file = prepare_env_file(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Module 2: Context Extraction")
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(active_env_file) if active_env_file else None,
        help="Environment file path; defaults to the project .env when present",
    )
    parser.add_argument(
        "--input", type=str,
        default=str(PROJECT_ROOT / "data" / "drone" / "mock_dialogues.jsonl"),
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--offline", action="store_true", help="Run without calling an LLM")
    parser.add_argument("--api-base", type=str, default=env_text("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", type=str, default=env_text("OPENAI_API_KEY"))
    parser.add_argument("--model", type=str, default=env_text("LLM4FAIRROUTING_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dialogues = [json.loads(l.strip()) for l in f if l.strip()]

    print(f"Loaded {len(dialogues)} dialogues")

    if args.offline:
        results = extract_demands_offline(dialogues, args.window)
    else:
        client = create_openai_client(args.api_base, args.api_key)
        results = extract_all_demands(dialogues, client, args.model, args.window)

    out_path = args.output or str(PROJECT_ROOT / "data" / "drone" / "extracted_demands.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
