"""
Module 1: Dialogue Generator — 从 rich event manifest 读取结构化需求事件，
生成 Module 2 所需的对话 JSON 格式数据。

需求层级（优先级由高到低）：
  Tier 1 — 生命支持物资  (life_support)：心脏骤停急救药、血液制品、AED、溶栓药物
  Tier 2 — 重症物资      (critical)：呼吸机、ICU 药物、手术急需
  Tier 3 — 常规物资      (regular)：疫苗、常规药品、防护物资
  Tier 4 — 消费类即时配送 (consumer)：餐食、OTC 药品、日用品

支持两种模式：
  offline  — 基于规则/模板生成中文对话文本，无需 LLM
  online   — 调用 LLM 生成更自然的对话文本
"""

import json
import hashlib
import math
import random
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from openai import OpenAI

from llm4fairrouting.config.runtime_env import env_text, prepare_env_file
from llm4fairrouting.data.demand_event_generation import (
    DIALOGUE_STYLE_VARIANTS,
    build_gold_structured_demand,
)
from llm4fairrouting.data.priority_labels import derive_priority_labels
from llm4fairrouting.data.seed_paths import (
    DEMAND_DIALOGUES_PATH,
    DEMAND_EVENTS_FILENAME,
    PRIMARY_EVENT_DATA_PATH,
    STATION_DATA_FILENAME,
    STATION_DATA_PATH,
)
from llm4fairrouting.data.stations import load_station_data
from llm4fairrouting.llm.client_utils import (
    call_llm,
    create_openai_client,
    parse_json_response,
)


# ============================================================================
# Demand tier system  (priority from CSV → tier mapping)
# ============================================================================
# CSV priority: 1 = most urgent, 5 = least urgent
# Tier mapping: 1 → life_support, 2 → critical, 3 → regular, 4/5 → consumer

DEMAND_TIERS = {
    "life_support": "Life-Support Supply",
    "critical":     "Critical Medical Supply",
    "regular":      "Routine Medical Supply",
    "consumer":     "Consumer On-Demand Delivery",
}

SUPPLY_TYPE_ALIASES = {
    "medical": "medical",
    "medical_supply": "medical",
    "医疗": "medical",
    "commercial": "commercial",
    "commercial_supply": "commercial",
    "商业": "commercial",
}

# English material names (used in conversation templates)
MATERIAL_EN = {
    "vaccine":         "vaccine",
    "medicine":        "medication",
    "protective_suit": "protective suit",
    "mask":            "face mask",
    "disinfectant":    "disinfectant solution",
    "ventilator":      "ventilator",
    "aed":             "AED defibrillator",
    "cardiac_drug":    "cardiac emergency drug",
    "blood_product":   "blood product",
    "thrombolytic":    "thrombolytic agent",
    "icu_drug":        "ICU medication",
    "food":            "food/meal",
    "otc_drug":        "OTC medication",
    "daily_supply":    "daily supplies",
}

PRIORITY_DEADLINE = {1: 15, 2: 30, 3: 60, 4: 120, 5: 240}
DIALOGUE_PROFILE_VERSION = "v2"

_SCENARIO_LABEL: Dict[Tuple[str, str], str] = {
    ("life_support", "vaccine"):         "Life-critical vaccine dispatch",
    ("life_support", "medicine"):        "Emergency medication — life-critical",
    ("life_support", "ventilator"):      "Emergency ventilator deployment",
    ("life_support", "protective_suit"): "Emergency PPE — life-critical",
    ("life_support", "mask"):            "Emergency face mask — life-critical",
    ("life_support", "disinfectant"):    "Emergency disinfectant — life-critical",
    ("critical",     "vaccine"):         "Urgent post-exposure vaccine",
    ("critical",     "medicine"):        "Urgent hospital medication",
    ("critical",     "ventilator"):      "Critical ventilator supply",
    ("critical",     "protective_suit"): "Urgent PPE restocking",
    ("critical",     "mask"):            "Urgent face mask supply",
    ("critical",     "disinfectant"):    "Urgent disinfectant supply",
    ("regular",      "vaccine"):         "Routine vaccine restocking",
    ("regular",      "medicine"):        "Routine medication delivery",
    ("regular",      "protective_suit"): "Standard PPE restocking",
    ("regular",      "mask"):            "Standard face mask restocking",
    ("regular",      "disinfectant"):    "Routine disinfectant restocking",
    ("consumer",     "vaccine"):         "Consumer vaccine order",
    ("consumer",     "medicine"):        "Consumer medication order",
    ("consumer",     "protective_suit"): "Consumer PPE order",
    ("consumer",     "mask"):            "Consumer face mask order",
    ("consumer",     "disinfectant"):    "Consumer disinfectant order",
}

_REQUESTER_ROLE: Dict[str, str] = {
    "life_support": "emergency_doctor",
    "critical":     "icu_nurse",
    "regular":      "community_health_worker",
    "consumer":     "consumer",
}

_REQUESTER_ROLE_OPTIONS: Dict[str, List[str]] = {
    "life_support": ["emergency_doctor", "paramedic", "triage_nurse"],
    "critical": ["icu_nurse", "clinical_pharmacist", "ward_coordinator"],
    "regular": ["community_health_worker", "clinic_manager", "pharmacy_staff"],
    "consumer": ["consumer", "family_caregiver", "office_administrator"],
}

_REQUESTER_TITLES: Dict[str, str] = {
    "emergency_doctor": "ER Physician",
    "paramedic": "Paramedic",
    "triage_nurse": "Triage Nurse",
    "icu_nurse": "ICU Nurse",
    "clinical_pharmacist": "Clinical Pharmacist",
    "ward_coordinator": "Ward Coordinator",
    "community_health_worker": "Community Health Worker",
    "clinic_manager": "Clinic Manager",
    "pharmacy_staff": "Pharmacy Staff",
    "consumer": "Customer",
    "family_caregiver": "Family Caregiver",
    "office_administrator": "Office Administrator",
}

_DISPATCHER_TITLES: Dict[str, str] = {
    "life_support": "Emergency Dispatch",
    "critical": "Medical Logistics",
    "regular": "Dispatch Coordinator",
    "consumer": "Delivery Platform",
}

_TIME_OF_DAY_CONTEXT: Dict[str, List[str]] = {
    "overnight": [
        "This is the overnight shift and local stock is already depleted.",
        "It is an overnight request, so the on-site team is working with a reduced backup stock.",
    ],
    "morning": [
        "The morning intake wave has already consumed today's first stock batch.",
        "This request is landing during the morning surge, so the usual shelf stock is gone.",
    ],
    "afternoon": [
        "The afternoon care cycle pushed this item to the top of the local backlog.",
        "This came up during the afternoon treatment block and cannot wait for the next van route.",
    ],
    "evening": [
        "The evening handover is underway, so we need a clean and fast replenishment.",
        "This is an evening request and the team needs the item before the next bedside round.",
    ],
}

_BENEFICIARY_HINTS: Dict[str, List[str]] = {
    "life_support": [
        "A resuscitation team is already at the receiving point.",
        "The bedside team is waiting at the landing zone right now.",
    ],
    "critical": [
        "The care team will use it immediately on arrival.",
        "Receiving staff and biomedical support are already on standby.",
    ],
    "regular": [
        "The receiving desk can sign for it as soon as the drone lands.",
        "Front-desk staff will take handoff once the drone arrives.",
    ],
    "consumer": [
        "Please drop it at the community locker if possible.",
        "A household member will collect it right after notification.",
    ],
}

_HANDLING_NOTES: Dict[str, List[str]] = {
    "aed": ["Shock-proof case is required for this flight."],
    "blood_product": ["Keep the payload in validated cold-chain packaging."],
    "thrombolytic": ["Cold-chain handling is mandatory for this dispatch."],
    "ventilator": ["Secure the unit with reinforced brackets before takeoff."],
    "icu_drug": ["Seal the medication kit and keep the chain-of-custody note attached."],
    "vaccine": ["Use the insulated vaccine box and keep the cold-chain monitor active."],
    "medicine": ["Package the medication kit for direct bedside handoff."],
    "protective_suit": ["Bundle and seal the PPE cartons to speed up unloading."],
    "mask": ["Standard sealed cartons are sufficient for this drop."],
    "disinfectant": ["Use the leak-proof chemical container for transport."],
    "food": ["Keep the thermal bag closed until handoff."],
    "otc_drug": ["Pack the order in a standard tamper-evident pharmacy bag."],
    "daily_supply": ["Standard parcel handling is enough for this order."],
}

_RECEIVER_NOTES: Dict[str, List[str]] = {
    "life_support": [
        "Please route straight to the emergency receiving pad.",
        "Please use the closest emergency drop point for handoff.",
    ],
    "critical": [
        "The nurse station will receive it at the clinical loading zone.",
        "Please notify the ICU desk once final approach begins.",
    ],
    "regular": [
        "Reception will collect it from the standard clinic landing point.",
        "Please follow normal handoff at the community pickup spot.",
    ],
    "consumer": [
        "A locker drop-off is preferred.",
        "Please use the standard residential delivery point.",
    ],
}

_SCENARIO_HINTS: Dict[Tuple[str, str], List[str]] = {
    ("life_support", "aed"): [
        "CPR is in progress and the backup AED cabinet is empty.",
    ],
    ("life_support", "blood_product"): [
        "An active transfusion case needs immediate blood product support.",
    ],
    ("life_support", "cardiac_drug"): [
        "The emergency cart is short on the cardiac rescue dose.",
    ],
    ("life_support", "thrombolytic"): [
        "The stroke window is closing and the thrombolytic dose must arrive fast.",
    ],
    ("critical", "ventilator"): [
        "A patient transfer requires a standby ventilator before the next procedure starts.",
    ],
    ("critical", "icu_drug"): [
        "The ICU team needs a drug refill before the next administration round.",
    ],
    ("critical", "vaccine"): [
        "The post-exposure window is narrow, so the vaccine must arrive promptly.",
    ],
    ("regular", "vaccine"): [
        "The afternoon vaccination block will start soon.",
    ],
    ("regular", "medicine"): [
        "Today's clinic queue is longer than expected and local stock ran low.",
    ],
    ("consumer", "food"): [
        "This is part of a normal same-day household order.",
    ],
    ("consumer", "otc_drug"): [
        "The order is for same-day symptom relief at home.",
    ],
}


# ============================================================================
# English multi-turn dialogue templates  (tier × material)
# ============================================================================
# Placeholders: {t} HH:MM, {dest_id}, {origin_name},
#               {mat_en}, {qty}, {kg}, {deadline}

_TPL_LIFE_SUPPORT: Dict[str, List[str]] = {
    "medicine": [
        "[{t}] 🚨 Emergency Dispatch → {origin_name}: LIFE-CRITICAL alert from {dest_id}. "
        "Patient in cardiac arrest — need {qty}x {mat_en} ({kg} kg) deployed IMMEDIATELY. "
        "Golden rescue window closing!\n"
        "[{t}] Station Operator: CODE RED confirmed. Drone loaded and cleared for takeoff.\n"
        "[{t}] Dispatch: ETA ≤{deadline} min. Medical team at {dest_id} is standing by. Launch now.\n"
        "[{t}] Station Operator: Drone airborne. Live tracking enabled.",

        "[{t}] ER Physician ({dest_id}): This is the ER — patient in critical condition. "
        "We urgently need {qty}x {mat_en} ({kg} kg) from {origin_name} within {deadline} min. "
        "Treat this as PRIORITY ONE — every second counts.\n"
        "[{t}] Dispatch System: PRIORITY ONE acknowledged. Drone launched from {origin_name}. "
        "ETA ≤{deadline} min.\n"
        "[{t}] ER Physician: Thank you. My team is waiting at the emergency bay.",
    ],
    "ventilator": [
        "[{t}] ICU Charge Nurse ({dest_id}): EMERGENCY — patient on life support, "
        "primary {mat_en} failing. Need backup unit ({qty} pc, {kg} kg) from {origin_name} NOW.\n"
        "[{t}] Logistics: Critical equipment confirmed in stock. Shock-proof packaging ready. "
        "Drone launching — ETA {deadline} min.\n"
        "[{t}] ICU Nurse: Copy. Biomedical engineer is on standby to receive and install.",
    ],
    "_default": [
        "[{t}] 🚨 LIFE-CRITICAL REQUEST — {dest_id}: Need {qty}x {mat_en} ({kg} kg) "
        "from {origin_name} within {deadline} min. Patient status critical. "
        "Mobilize all available resources.\n"
        "[{t}] Emergency Dispatch: Confirmed. Drone from {origin_name} launched immediately. "
        "ETA ≤{deadline} min. Please keep this channel open.\n"
        "[{t}] Requestor: Understood. Landing zone cleared. Waiting at {dest_id}.",

        "[{t}] 120 Command Center ({dest_id}): Patient in life-threatening condition — "
        "{qty}x {mat_en} ({kg} kg) required urgently from {origin_name}. "
        "Stock depleted here. Time critical.\n"
        "[{t}] Fenyi UAV Dispatch: Life-critical task received. {origin_name} drone airborne. "
        "ETA ≤{deadline} min. Cold-chain maintained where applicable.\n"
        "[{t}] Medical Team: Standing by at drop-off point. Please hurry.",
    ],
}

_TPL_CRITICAL: Dict[str, List[str]] = {
    "medicine": [
        "[{t}] Clinical Pharmacist ({dest_id}): Urgent — patient requires {qty}x {mat_en} "
        "({kg} kg). Our in-house stock is depleted; oral substitute not suitable. "
        "Please expedite from {origin_name}.\n"
        "[{t}] Drone Dispatch: Confirmed. Loading and dispatching from {origin_name}. "
        "Estimated arrival: {deadline} min.\n"
        "[{t}] Pharmacist: Understood. Will administer immediately upon receipt.",

        "[{t}] Emergency Dept ({dest_id}): Urgent medication request — {qty}x {mat_en} "
        "({kg} kg), our supply just ran out. Please dispatch from {origin_name} ASAP.\n"
        "[{t}] Medical Logistics: Prioritized. Drone en route from {origin_name}. "
        "ETA {deadline} min. Nurse station please stand by.",
    ],
    "ventilator": [
        "[{t}] ICU Nurse Manager ({dest_id}): Bed 36 — patient on ventilator, backup unit "
        "needed urgently. Please dispatch {mat_en} ({qty} unit, {kg} kg) from {origin_name}.\n"
        "[{t}] Equipment Dept: Stock confirmed at {origin_name}. Protective packaging complete. "
        "Drone dispatched — ETA {deadline} min. Please arrange engineer on-site.\n"
        "[{t}] Nurse Manager: Copy. Team is ready to receive and set up.",
    ],
    "vaccine": [
        "[{t}] Infection Control Officer ({dest_id}): Urgent post-exposure prophylaxis — "
        "need {mat_en} ({qty} doses, {kg} kg) from {origin_name} within {deadline} min. "
        "Time window for effectiveness is very narrow.\n"
        "[{t}] Dispatch: Urgent vaccine shipment confirmed. Drone en route from {origin_name}. "
        "ETA {deadline} min, cold-chain maintained throughout.\n"
        "[{t}] Officer: Good. Patient is being prepped for injection.",
    ],
    "protective_suit": [
        "[{t}] Isolation Unit ({dest_id}): New confirmed cases today — PPE stock critically low. "
        "Urgent request: {qty}x {mat_en} ({kg} kg) from {origin_name} within {deadline} min.\n"
        "[{t}] Logistics: Request received. {origin_name} dispatching now. "
        "ETA {deadline} min. Please designate a receive point.",
    ],
    "_default": [
        "[{t}] Ward Coordinator ({dest_id}): Urgent supply request — {qty}x {mat_en} ({kg} kg). "
        "Patient condition is serious. Please dispatch from {origin_name} within {deadline} min.\n"
        "[{t}] Logistics System: Request received. Stock confirmed at {origin_name}. "
        "Drone dispatching — ETA {deadline} min. Please have someone at the pickup point.\n"
        "[{t}] Ward Coordinator: Understood. Nurse will be there to receive.",
    ],
}

_TPL_REGULAR: Dict[str, List[str]] = {
    "vaccine": [
        "[{t}] Community Health Center ({dest_id}): Good morning — routine vaccine restock. "
        "We need {qty} dose(s) of {mat_en} ({kg} kg) for this afternoon's vaccination session. "
        "Please send from {origin_name}; within {deadline} min works fine.\n"
        "[{t}] Dispatch: Received. {origin_name} preparing your order. "
        "ETA {deadline} min, cold-chain packaging confirmed.\n"
        "[{t}] Health Center: Great, Nurse Li will sign for it at reception. Thank you!",

        "[{t}] Maternal & Child Health Clinic ({dest_id}): Hi, we're running low on {mat_en} "
        "({qty} doses, {kg} kg) — peak vaccination month. Please restock from {origin_name} "
        "before 3 PM today.\n"
        "[{t}] Fenyi Delivery: Added to priority queue. {origin_name} departing now. "
        "ETA {deadline} min. Please have cold-chain reception ready.",
    ],
    "medicine": [
        "[{t}] Resident ({dest_id}): Hi, my father needs {mat_en} ({qty} boxes, {kg} kg). "
        "He can't go out today. Could you send it to the package locker downstairs?\n"
        "[{t}] Delivery Platform: Received. {origin_name} dispatching now. "
        "ETA {deadline} min — you'll get a push notification when it arrives.\n"
        "[{t}] Resident: Perfect, I'll pick it up right away. Thank you!",

        "[{t}] General Practice Clinic ({dest_id}): Afternoon clinic is running short on "
        "{mat_en} ({qty} units, {kg} kg). No emergency — please restock from {origin_name} "
        "at your convenience today.\n"
        "[{t}] Fenyi Dispatch: Added to today's delivery plan. {origin_name} en route. "
        "ETA ~{deadline} min. Reception desk please watch for arrival notification.\n"
        "[{t}] Clinic: Got it, front desk is notified. Thanks!",
    ],
    "protective_suit": [
        "[{t}] Isolation Facility Manager ({dest_id}): We admitted new close contacts today — "
        "protective suit stock is getting low. Please send {qty}x {mat_en} ({kg} kg) "
        "from {origin_name}. No immediate rush, within {deadline} min is fine.\n"
        "[{t}] Dispatch: Noted. {origin_name} will dispatch shortly. "
        "ETA ~{deadline} min. Please assign a receiving staff member.",

        "[{t}] Fever Clinic Nurse Manager ({dest_id}): Higher patient volume today — "
        "{mat_en} consumption doubled. Requesting restock: {qty} units ({kg} kg). "
        "Within {deadline} min would be great.\n"
        "[{t}] Logistics: Confirmed with {origin_name}. On-schedule delivery. "
        "ETA {deadline} min. Clinic nurse to sign for receipt.",
    ],
    "mask": [
        "[{t}] Community Committee ({dest_id}): Hi, we're organizing a health fair this weekend "
        "and need {qty} packs of {mat_en} ({kg} kg) for distribution. "
        "No rush — today is fine. Source: {origin_name}.\n"
        "[{t}] Platform: Order confirmed. {origin_name} arranging standard delivery. "
        "ETA ~{deadline} min. Please watch for the arrival notification.",

        "[{t}] Pharmacy ({dest_id}): {mat_en} shelf running low — need restock: "
        "{qty} packs ({kg} kg) from {origin_name}. Part of today's replenishment plan.\n"
        "[{t}] Fenyi: Arranged. {origin_name} dispatching. "
        "ETA {deadline} min. Rear entrance receiving.",
    ],
    "disinfectant": [
        "[{t}] Market Manager ({dest_id}): Additional {mat_en} needed for today's "
        "sanitation round — {qty} bottles ({kg} kg). Please deliver from {origin_name} "
        "within {deadline} min.\n"
        "[{t}] Dispatch: Confirmed. {origin_name} preparing standard chemical packaging. "
        "ETA {deadline} min. Please confirm unloading area.",

        "[{t}] School Facilities ({dest_id}): Planning full-campus disinfection next week — "
        "advance stock request: {qty} bottles of {mat_en} ({kg} kg). "
        "End of day delivery from {origin_name} is fine.\n"
        "[{t}] {origin_name}: Added to today's last run. ETA {deadline} min. "
        "Please have someone at the school gate to sign.",
    ],
    "_default": [
        "[{t}] Supply Request ({dest_id}): Standard restock order — {mat_en} x{qty} ({kg} kg). "
        "Source: {origin_name}. Delivery within {deadline} min. Non-urgent.\n"
        "[{t}] System: Order confirmed. Standard delivery scheduled from {origin_name}. "
        "ETA: {deadline} min. You'll receive a notification on arrival.",
    ],
}

_TPL_CONSUMER: Dict[str, List[str]] = {
    "medicine": [
        "[{t}] User (Pharmacy App • {dest_id}): Hi! I need {mat_en} x{qty} ({kg} kg) "
        "delivered — I can't go out today. Drone delivery would be perfect!\n"
        "[{t}] System: Order placed! {origin_name} station is preparing your order. "
        "Drone en route — ETA {deadline} min. Check your locker notification.\n"
        "[{t}] User: Thanks! Just drop it at my building's delivery locker 😊",

        "[{t}] Customer ({dest_id}): Hey, can I get {qty}x {mat_en} ({kg} kg) "
        "delivered sometime today? No hurry at all.\n"
        "[{t}] Platform: Got it! {origin_name} will handle your order. "
        "ETA ~{deadline} min. App will ping you when it's at the locker.",
    ],
    "mask": [
        "[{t}] User (App • {dest_id}): Ordering {qty} pack(s) of {mat_en} ({kg} kg) — "
        "for personal use, no rush. Drone delivery please!\n"
        "[{t}] Delivery System: Order accepted. {origin_name} drone en route. "
        "ETA ~{deadline} min. You'll get a push when it's dropped off.",
    ],
    "disinfectant": [
        "[{t}] User ({dest_id}): Hi, need {qty}x {mat_en} ({kg} kg) for home cleaning. "
        "Drone delivery anytime today is fine!\n"
        "[{t}] {origin_name} Platform: Order in queue. Standard delivery scheduled. "
        "ETA {deadline} min. Check app for status updates.",
    ],
    "vaccine": [
        "[{t}] User (Health App • {dest_id}): Hi, I'd like to order {qty} dose(s) of "
        "{mat_en} ({kg} kg) for home delivery. No specific time pressure.\n"
        "[{t}] System: Noted! {origin_name} will arrange delivery with proper cold-chain packaging. "
        "ETA ~{deadline} min. A notification will be sent on arrival.",
    ],
    "protective_suit": [
        "[{t}] User ({dest_id}): Could I get {qty}x {mat_en} ({kg} kg) delivered to my "
        "door? Personal order, no urgency 😊\n"
        "[{t}] {origin_name} Platform: Order received. Drone delivery queued. "
        "ETA ~{deadline} min. Please watch for the locker notification.",
    ],
    "_default": [
        "[{t}] User (App • {dest_id}): Just ordered {mat_en} x{qty} ({kg} kg). "
        "Drone delivery please! No rush, anytime within {deadline} min is totally fine 😊\n"
        "[{t}] Delivery System: Order accepted! {origin_name} drone station is on it. "
        "Est. arrival: {deadline} min. We'll ping you when it's dropped off at your locker.\n"
        "[{t}] User: Sweet, thanks! Will grab it when I get the alert.",

        "[{t}] Customer ({dest_id}): Hi, can I get {qty}x {mat_en} ({kg} kg) delivered? "
        "Not urgent at all — whenever is convenient. From {origin_name} if possible.\n"
        "[{t}] Platform: Got your order! {origin_name} will dispatch when slot opens. "
        "ETA ~{deadline} min. Check the app for real-time updates!",
    ],
}

_ALL_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "life_support": _TPL_LIFE_SUPPORT,
    "critical":     _TPL_CRITICAL,
    "regular":      _TPL_REGULAR,
    "consumer":     _TPL_CONSUMER,
}


def _map_priority_to_tier(priority: int) -> str:
    """Map CSV priority (1–5) directly to demand tier.

    1 → life_support  (life-critical, ≤15 min)
    2 → critical      (urgent medical, ≤30 min)
    3 → regular       (routine medical, ≤60 min)
    4, 5 → consumer   (on-demand / non-urgent)
    """
    if priority == 1:
        return "life_support"
    elif priority == 2:
        return "critical"
    elif priority == 3:
        return "regular"
    else:
        return "consumer"


def _pick_tier_template(tier: str, material: str, rng: random.Random) -> str:
    """从对应层级和物资类型中随机选择一条对话模板。"""
    tier_templates = _ALL_TEMPLATES.get(tier, _ALL_TEMPLATES["regular"])
    templates = tier_templates.get(material, tier_templates.get("_default", []))
    if not templates:
        templates = _ALL_TEMPLATES["regular"]["_default"]
    return rng.choice(templates)


def _stable_seed(*parts: object) -> int:
    payload = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _choose(options: List[str], rng: random.Random, fallback: str = "") -> str:
    if not options:
        return fallback
    return rng.choice(options)


def _time_of_day_bucket(ts_hhmm: str) -> str:
    hour = int(ts_hhmm.split(":", 1)[0])
    if 0 <= hour < 6:
        return "overnight"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening"


def _normalize_dialogue_style(style: str) -> str:
    normalized = str(style or "direct").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "plain": "direct",
        "direct": "direct",
        "technical": "technical",
        "term_heavy": "technical",
        "colloquial": "conversational",
        "spoken": "conversational",
        "conversational": "conversational",
        "dispatch": "dispatch_brief",
        "dispatch_brief": "dispatch_brief",
        "dispatch_style": "dispatch_brief",
    }
    canonical = aliases.get(normalized, normalized)
    if canonical not in DIALOGUE_STYLE_VARIANTS:
        return "direct"
    return canonical


def _normalize_dialogue_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _event_must_mention_factors(event: Dict) -> List[Dict]:
    return list(event.get("must_mention_factors", []) or [])


def _event_optional_factors(event: Dict) -> List[Dict]:
    return list(event.get("optional_factors", []) or [])


def _matches_factor(text: str, factor: Dict) -> bool:
    keywords = [
        str(keyword).strip().lower()
        for keyword in factor.get("keywords", [])
        if str(keyword).strip()
    ]
    if not keywords:
        description = str(factor.get("description", "")).strip().lower()
        keywords = [description] if description else []
    return any(keyword in text for keyword in keywords)


def _mask_gold_demand_for_unobserved_factors(gold: Dict, observed_factor_names: set[str]) -> Dict:
    masked = deepcopy(gold)
    if "scenario_context" not in observed_factor_names:
        masked["priority_evaluation_signals"]["patient_condition"] = "Routine supply request"
        masked["priority_evaluation_signals"]["medical_urgency_self_report"] = "Routine supply request"
        masked["priority_evaluation_signals"]["scenario_context"] = "Routine supply request"
        masked["context_signals"] = [
            signal for signal in masked.get("context_signals", [])
            if "Structured deadline" in signal or "Requester role" in signal
        ]
    if "deadline_minutes" not in observed_factor_names:
        masked["time_constraint"]["deadline_minutes"] = max(
            120,
            int(masked.get("time_constraint", {}).get("deadline_minutes", 120) or 120),
        )
        masked["time_constraint"]["type"] = "soft"
        masked["time_constraint"]["description"] = "No explicit hard deadline observed in the dialogue"
    if "requester_role" not in observed_factor_names:
        default_role = "consumer" if masked.get("demand_tier") == "consumer" else "community_health_worker"
        masked["requester_role"] = default_role
        masked["priority_evaluation_signals"]["requester_role"] = default_role
    if "special_handling" not in observed_factor_names:
        masked["special_handling"] = []
        masked["priority_evaluation_signals"]["special_handling"] = []
    if "population_vulnerability" not in observed_factor_names:
        masked["population_vulnerability"] = {
            "elderly_ratio": 0.0,
            "population": 0,
            "elderly_involved": False,
            "children_involved": False,
            "vulnerable_community": False,
        }
        masked["priority_evaluation_signals"]["population_vulnerability"] = dict(masked["population_vulnerability"])
    if "receiver_ready" not in observed_factor_names:
        masked["receiver_ready"] = False
        masked["operational_readiness"] = "Standard handoff readiness"
        masked["priority_evaluation_signals"]["operational_readiness"] = "Standard handoff readiness"
    return masked


def audit_dialogue(dialogue: Dict) -> Dict[str, object]:
    """Check whether the dialogue exposes the factors needed to recover its priority."""
    annotations = dialogue.get("annotations", {})
    gold_demand = annotations.get("gold_structured_demand")
    latent_priority = annotations.get("latent_priority")
    must_mention = list(annotations.get("must_mention_factors", []) or [])
    optional = list(annotations.get("optional_factors", []) or [])
    if not must_mention and not optional:
        return {
            "dialogue_observable_priority": latent_priority,
            "observed_factors_in_dialogue": [],
            "missing_must_mention_factors": [],
            "observability_score": 1.0,
            "passed": True,
        }
    text = _normalize_dialogue_text(dialogue.get("conversation", ""))

    observed_factors = []
    missing_factors = []
    observed_names: set[str] = set()

    for factor in must_mention + optional:
        if _matches_factor(text, factor):
            observed_factors.append({
                "name": factor.get("name"),
                "value": factor.get("value"),
            })
            observed_names.add(str(factor.get("name", "")))
        elif factor in must_mention:
            missing_factors.append(str(factor.get("name", "")))

    matched_required = len(must_mention) - len(missing_factors)
    optional_matches = max(0, len(observed_factors) - matched_required)
    denominator = max(len(must_mention) + max(len(optional), 1) * 0.5, 1.0)
    observability_score = min(
        1.0,
        (matched_required + optional_matches * 0.5) / denominator,
    )

    dialogue_observable_priority = None
    if gold_demand:
        masked = _mask_gold_demand_for_unobserved_factors(gold_demand, observed_names)
        derived = derive_priority_labels(masked, latent_priority=latent_priority)
        dialogue_observable_priority = int(
            derived.get("extraction_observable_priority", latent_priority or 4)
        )
    elif latent_priority is not None and not missing_factors:
        dialogue_observable_priority = int(latent_priority)

    passed = not missing_factors
    if latent_priority is not None and dialogue_observable_priority is not None:
        passed = passed and int(dialogue_observable_priority) == int(latent_priority)

    return {
        "dialogue_observable_priority": dialogue_observable_priority,
        "observed_factors_in_dialogue": observed_factors,
        "missing_must_mention_factors": missing_factors,
        "observability_score": round(float(observability_score), 4),
        "passed": passed,
    }


def _expand_event_style_specs(
    demand_events: List[Dict],
    styles: Optional[List[str]] = None,
) -> List[tuple[Dict, str]]:
    expanded: List[tuple[Dict, str]] = []
    requested_styles = [_normalize_dialogue_style(style) for style in (styles or [])]
    for event in demand_events:
        event_style = _normalize_dialogue_style(str(event.get("dialogue_style", "direct")))
        style_list = requested_styles or [event_style]
        for style in style_list:
            expanded.append((event, style))
    return expanded


def _infer_quantity_units(material: str) -> str:
    unit_map = {
        "aed": "unit",
        "ventilator": "unit",
        "vaccine": "dose",
        "blood_product": "pack",
        "cardiac_drug": "dose",
        "thrombolytic": "dose",
        "icu_drug": "kit",
        "medicine": "box",
        "mask": "pack",
        "protective_suit": "set",
        "disinfectant": "bottle",
        "food": "meal",
        "otc_drug": "box",
        "daily_supply": "package",
    }
    return unit_map.get(material, "unit")


def _build_dialogue_profile(
    event: Dict,
    material: str,
    tier: str,
    station_name: str,
    dest_id: str,
    ts_hhmm: str,
    deadline_minutes: int,
) -> Dict[str, str]:
    event_id = str(event.get("event_id", event.get("unique_id", dest_id)))
    rng = random.Random(
        _stable_seed(
            DIALOGUE_PROFILE_VERSION,
            event_id,
            material,
            tier,
            station_name,
            ts_hhmm,
        )
    )
    requester_role = str(event.get("requester_role", "")).strip() or _choose(
        _REQUESTER_ROLE_OPTIONS.get(tier, [_REQUESTER_ROLE.get(tier, "consumer")]),
        rng,
        fallback=_REQUESTER_ROLE.get(tier, "consumer"),
    )
    requester_title = _REQUESTER_TITLES.get(requester_role, "Requester")
    dispatcher_title = _DISPATCHER_TITLES.get(tier, "Dispatch Coordinator")
    time_context = str(event.get("time_context", "")).strip() or _choose(
        _TIME_OF_DAY_CONTEXT.get(_time_of_day_bucket(ts_hhmm), []),
        rng,
        fallback="The local team needs a fast drone handoff.",
    )
    scenario_summary = str(event.get("scenario_hint", "")).strip() or _choose(
        _SCENARIO_HINTS.get((tier, material), []),
        rng,
        fallback=f"Please move the {MATERIAL_EN.get(material, material)} request without delay.",
    )
    beneficiary_hint = str(event.get("beneficiary_hint", "")).strip() or _choose(
        _BENEFICIARY_HINTS.get(tier, []),
        rng,
        fallback="Receiving staff are ready for handoff.",
    )
    handling_note = str(event.get("handling_notes", "")).strip() or _choose(
        _HANDLING_NOTES.get(material, []),
        rng,
        fallback="Please follow standard packaging and handoff checks.",
    )
    receiver_note = str(event.get("receiver_notes", "")).strip() or _choose(
        _RECEIVER_NOTES.get(tier, []),
        rng,
        fallback="Please use the standard receiving point.",
    )
    request_channel = str(event.get("request_channel", "")).strip() or {
        "life_support": "emergency dispatch line",
        "critical": "hospital logistics desk",
        "regular": "clinic replenishment queue",
        "consumer": "consumer delivery app",
    }.get(tier, "dispatch channel")

    if tier == "life_support":
        opening = "Immediate support needed."
        followup = "Team is in position and will receive the payload on touchdown."
    elif tier == "critical":
        opening = "Urgent clinical support needed."
        followup = "The care team will take over the item as soon as it lands."
    elif tier == "regular":
        opening = "Routine replenishment request."
        followup = "Reception will complete handoff once the drone arrives."
    else:
        opening = "Same-day order request."
        followup = "Please send the app notification once the drop is complete."

    return {
        "requester_role": requester_role,
        "requester_title": requester_title,
        "dispatcher_title": dispatcher_title,
        "time_context": time_context,
        "scenario_summary": scenario_summary,
        "beneficiary_hint": beneficiary_hint,
        "handling_note": handling_note,
        "receiver_note": receiver_note,
        "request_channel": request_channel,
        "opening": opening,
        "followup": followup,
        "station_name": station_name,
        "dest_id": dest_id,
        "deadline_minutes": str(deadline_minutes),
    }


# ============================================================================
# 数据加载
# ============================================================================

def load_stations(station_path: str) -> List[Dict]:
    """从规范化站点数据读取深圳站点列表。

    每个站点格式::

        {"station_id": "ST001", "name": "丰翼无人机燕罗航站",
         "lon": 113.879911, "lat": 22.799354}
    """
    df = load_station_data(station_path)
    df = df[df["city"].astype(str).str.contains("Shenzhen", na=False)]

    stations = []
    for i, (_, row) in enumerate(df.iterrows()):
        lat_val = row["latitude"]
        lon_val = row["longitude"]
        try:
            lat = float(lat_val)
            lon = float(lon_val)
        except (TypeError, ValueError):
            continue
        if math.isnan(lat) or math.isnan(lon):
            continue
        name_val = row["station_name"] if "station_name" in row.index else f"S{i+1}"
        name = str(name_val).strip()
        stations.append({
            "station_id": f"ST{i+1:03d}",
            "name": name,
            "lon": lon,
            "lat": lat,
        })

    return stations


def _normalize_supply_type(supply_type: str) -> str:
    raw = str(supply_type).strip()
    if not raw:
        return ""
    normalized = raw.lower().replace("-", "_").replace(" ", "_")
    return SUPPLY_TYPE_ALIASES.get(raw, SUPPLY_TYPE_ALIASES.get(normalized, normalized))


def _infer_material_type(supply_type: str, priority: int, unique_id: str = "") -> str:
    """Infer material type from supply_type and priority (for CSV without material_type)."""
    rng = random.Random(_stable_seed("material", unique_id, supply_type, priority))
    normalized_supply_type = _normalize_supply_type(supply_type)
    if normalized_supply_type == "medical":
        if priority == 1:
            return rng.choice(["cardiac_drug", "blood_product", "aed", "thrombolytic"])
        elif priority == 2:
            return rng.choice(["ventilator", "icu_drug"])
        elif priority == 3:
            return rng.choice(["vaccine", "medicine", "protective_suit"])
        else:
            return rng.choice(["medicine", "vaccine"])
    else:
        if priority == 1:
            return rng.choice(["medicine", "protective_suit", "disinfectant"])
        elif priority == 2:
            return rng.choice(["medicine", "mask", "disinfectant"])
        elif priority == 3:
            return rng.choice(["food", "mask", "daily_supply"])
        else:
            return rng.choice(["food", "daily_supply", "otc_drug"])


def _normalize_manifest_event(record: Dict) -> Dict:
    normalized = dict(record)
    origin = dict(record.get("origin", {}) or {})
    destination = dict(record.get("destination", {}) or {})
    cargo = dict(record.get("cargo", {}) or {})
    if origin:
        normalized.setdefault("supply_fid", origin.get("fid", ""))
        normalized.setdefault("supply_lon", (origin.get("coords") or [0.0, 0.0])[0])
        normalized.setdefault("supply_lat", (origin.get("coords") or [0.0, 0.0])[1])
        normalized.setdefault("supply_type", origin.get("supply_type", ""))
        normalized.setdefault("supply_station_name", origin.get("station_name", origin.get("fid", "")))
    if destination:
        normalized.setdefault("demand_node_id", destination.get("node_id", destination.get("fid", "")))
        normalized.setdefault("demand_fid", destination.get("fid", ""))
        normalized.setdefault("demand_lon", (destination.get("coords") or [0.0, 0.0])[0])
        normalized.setdefault("demand_lat", (destination.get("coords") or [0.0, 0.0])[1])
        normalized.setdefault("destination_type", destination.get("type", "residential_area"))
    if cargo:
        normalized.setdefault("material_type", cargo.get("type", "medicine"))
        normalized.setdefault("quantity_kg", cargo.get("weight_kg", record.get("weight_kg", 1.0)))
        normalized.setdefault("demand_tier", cargo.get("demand_tier"))
    if "event_id" not in normalized and "unique_id" in normalized:
        normalized["event_id"] = normalized["unique_id"]
    if "unique_id" not in normalized and "event_id" in normalized:
        normalized["unique_id"] = normalized["event_id"]
    if "priority" not in normalized and "latent_priority" in normalized:
        normalized["priority"] = normalized["latent_priority"]
    if "time_slot" not in normalized and "time_hour" in normalized:
        normalized["time_slot"] = int(round(float(normalized["time_hour"]) * 12))
    return normalized


def load_demand_events(
    csv_path: str,
    n_events: Optional[int] = None,
    time_slots: Optional[List[int]] = None,
) -> List[Dict]:
    """读取需求事件数据，主路径是 rich event manifest，CSV 仅作为兼容回退。

    Parameters
    ----------
    csv_path : str
        CSV 路径。
    n_events : int, optional
        若指定，随机采样不超过该数量的事件。
    time_slots : list[int], optional
        若指定，仅加载这些 time_slot 的事件。
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("需要 pandas: pip install pandas")

    path = Path(csv_path)
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            events = [
                _normalize_manifest_event(json.loads(line))
                for line in handle
                if line.strip()
            ]
        if time_slots is not None:
            allowed = {int(slot) for slot in time_slots}
            events = [
                event for event in events
                if int(event.get("time_slot", -1)) in allowed
            ]
        if n_events is not None and len(events) > n_events:
            rng = random.Random(42)
            events = sorted(
                rng.sample(events, k=n_events),
                key=lambda item: (int(item.get("time_slot", 0)), str(item.get("event_id", ""))),
            )
        return events

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "supply_type" in df.columns:
        df["supply_type"] = df["supply_type"].map(_normalize_supply_type)

    # Auto-detect new CSV format: has "time" (float hours) instead of "time_slot" (int)
    if "time" in df.columns and "time_slot" not in df.columns:
        df["time_slot"] = (df["time"] * 12).round().astype(int)

    # Rename new CSV columns to match expected schema
    rename_map = {}
    if "unique_id" in df.columns and "event_id" not in df.columns:
        rename_map["unique_id"] = "event_id"
    if "demand_fid" in df.columns and "demand_node_id" not in df.columns:
        rename_map["demand_fid"] = "demand_node_id"
    if "material_weight" in df.columns and "quantity_kg" not in df.columns:
        rename_map["material_weight"] = "quantity_kg"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Infer material_type from supply_type if not present
    if "material_type" not in df.columns:
        if "supply_type" in df.columns:
            df["material_type"] = df.apply(
                lambda row: _infer_material_type(
                    str(row.get("supply_type", "")),
                    int(row.get("priority", 3)),
                    str(row.get("event_id", "")),
                ),
                axis=1,
            )
        else:
            df["material_type"] = "medicine"

    if time_slots is not None:
        df = df[df["time_slot"].isin(time_slots)]

    if n_events is not None and len(df) > n_events:
        df = df.sample(n=n_events, random_state=42).sort_values(
            ["time_slot", "event_id"]
        )

    return df.to_dict(orient="records")


# ============================================================================
# 辅助：最近站点
# ============================================================================

def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _find_nearest_station(lon: float, lat: float, stations: List[Dict]) -> Dict:
    return min(stations, key=lambda s: _haversine(lon, lat, s["lon"], s["lat"]))


# ============================================================================
# 事件 → dialogue record
# ============================================================================

def _supply_to_station(event: Dict) -> Dict:
    """Build a station-like dict from CSV supply point info."""
    supply_fid = str(event.get("supply_fid", ""))
    supply_type = _normalize_supply_type(str(event.get("supply_type", "")))
    if supply_type == "medical":
        name = f"Medical Supply Center {supply_fid}"
    else:
        name = f"Commercial Distribution Hub {supply_fid}"
    return {
        "station_id": supply_fid,
        "name": name,
        "lon": float(event["supply_lon"]),
        "lat": float(event["supply_lat"]),
    }


def _event_to_dialogue(
    event: Dict,
    stations: List[Dict],
    base_date: str,
    dialogue_idx: int,
    conversation: Optional[str] = None,
    style: str = "direct",
) -> Dict:
    """将单个 demand_event 行转换为 Module 2 所需的 dialogue JSON。

    Parameters
    ----------
    event : dict
        需求事件的一行（支持新旧 CSV 格式）。
    stations : list[dict]
        站点列表（已按 load_stations() 格式化）。可为空列表。
    base_date : str
        日期字符串，如 "2024-03-15"，time_slot=0 对应 00:00:00。
    dialogue_idx : int
        全局序号，用于生成 dialogue_id。
    conversation : str, optional
        若提供则使用此文本；否则自动生成。
    """
    normalized_event = _normalize_manifest_event(event)
    time_slot = int(normalized_event["time_slot"])
    base_dt = datetime.strptime(base_date, "%Y-%m-%d")
    ts = base_dt + timedelta(minutes=time_slot * 5)
    timestamp = ts.strftime("%Y-%m-%dT%H:%M:%S")
    ts_hhmm = ts.strftime("%H:%M")
    style = _normalize_dialogue_style(style)

    dest_lon = float(normalized_event["demand_lon"])
    dest_lat = float(normalized_event["demand_lat"])
    original_material = str(normalized_event.get("material_type", "medicine"))
    qty_kg = float(normalized_event.get("quantity_kg", normalized_event.get("weight_kg", 1.0)))
    priority = int(normalized_event.get("latent_priority", normalized_event.get("priority", 3)))
    demand_node_id = str(normalized_event.get("demand_node_id", f"D{dialogue_idx}"))
    dest_fid = normalized_event.get(
        "demand_node_index",
        normalized_event.get("demand_fid", normalized_event.get("demand_node_id", f"D{dialogue_idx}")),
    )

    tier = _map_priority_to_tier(priority)
    deadline_minutes = int(
        normalized_event.get(
            "deadline_minutes",
            normalized_event.get("delivery_deadline_minutes", PRIORITY_DEADLINE.get(priority, 60)),
        )
    )

    # Use supply point from CSV if available, otherwise find nearest station
    if normalized_event.get("supply_lon") is not None and normalized_event.get("supply_lat") is not None:
        station = _supply_to_station(normalized_event)
    elif stations:
        station = _find_nearest_station(dest_lon, dest_lat, stations)
    else:
        station = {
            "station_id": f"ST{dialogue_idx:03d}",
            "name": f"Station-{dialogue_idx}",
            "lon": dest_lon,
            "lat": dest_lat,
        }

    profile = _build_dialogue_profile(
        event=normalized_event,
        material=original_material,
        tier=tier,
        station_name=station["name"],
        dest_id=demand_node_id,
        ts_hhmm=ts_hhmm,
        deadline_minutes=deadline_minutes,
    )

    if conversation is None:
        conversation = _generate_rule_conversation(
            demand_node_id,
            station,
            original_material,
            qty_kg,
            priority,
            tier,
            ts_hhmm,
            profile=profile,
            event=normalized_event,
            style=style,
        )

    rng = random.Random(_stable_seed("demographics", dest_lon, dest_lat))
    elderly_ratio = round(rng.uniform(0.15, 0.60), 2)
    population = rng.randint(500, 8000)
    gold_structured_demand = normalized_event.get("gold_structured_demand")
    if not gold_structured_demand:
        normalized_event.setdefault("event_id", str(normalized_event.get("unique_id", "")))
        normalized_event.setdefault("origin", {
            "fid": station["station_id"],
            "coords": [station["lon"], station["lat"]],
            "type": "supply_station",
            "station_name": station["name"],
            "supply_type": _normalize_supply_type(str(normalized_event.get("supply_type", ""))),
        })
        normalized_event.setdefault("destination", {
            "fid": str(dest_fid),
            "node_id": demand_node_id,
            "coords": [dest_lon, dest_lat],
            "type": str(normalized_event.get("destination_type", "residential_area")),
        })
        normalized_event.setdefault("cargo", {
            "type": original_material,
            "weight_kg": qty_kg,
            "demand_tier": tier,
            "temperature_sensitive": original_material in {"vaccine", "blood_product", "thrombolytic"},
        })
        normalized_event.setdefault("requester_role", profile["requester_role"])
        normalized_event.setdefault("special_handling", [])
        normalized_event.setdefault("population_vulnerability", {
            "elderly_ratio": elderly_ratio,
            "population": population,
            "elderly_involved": elderly_ratio >= 0.45,
            "children_involved": False,
            "vulnerable_community": elderly_ratio >= 0.50,
        })
        normalized_event.setdefault("receiver_ready", False)
        normalized_event.setdefault("operational_readiness", "Standard handoff readiness")
        normalized_event.setdefault("latent_priority", priority)
        normalized_event.setdefault("priority", priority)
        gold_structured_demand = build_gold_structured_demand(normalized_event)

    dialogue = {
        "dialogue_id": f"D{dialogue_idx:04d}",
        "timestamp": timestamp,
        "conversation": conversation,
        "metadata": {
            "event_id": str(normalized_event.get("event_id", "")),
            "time_slot": time_slot,
            "origin_fid": station["station_id"],
            "destination_fid": dest_fid,
            "origin_coords": [station["lon"], station["lat"]],
            "dest_coords": [dest_lon, dest_lat],
            "dest_demographics": {
                "elderly_ratio": elderly_ratio,
                "population": population,
            },
            "nearby_poi": _infer_poi(original_material, priority, tier),
            "material_type": original_material,
            "quantity_kg": qty_kg,
            "requester_role": profile["requester_role"],
            "requester_title": profile["requester_title"],
            "request_channel": profile["request_channel"],
            "delivery_deadline_minutes": deadline_minutes,
            "scenario_summary": profile["scenario_summary"],
            "beneficiary_hint": profile["beneficiary_hint"],
            "handling_notes": profile["handling_note"],
            "receiver_notes": profile["receiver_note"],
            "dialogue_profile_version": DIALOGUE_PROFILE_VERSION,
            "demand_type": str(normalized_event.get("demand_type", "")),
            "supply_station_name": station["name"],
            "supply_type": _normalize_supply_type(str(normalized_event.get("supply_type", ""))),
        },
        "annotations": {
            "dialogue_style": style,
            "latent_priority": priority,
            "priority_factors": dict(normalized_event.get("priority_factors", {})),
            "must_mention_factors": _event_must_mention_factors(normalized_event),
            "optional_factors": _event_optional_factors(normalized_event),
            "gold_structured_demand": gold_structured_demand,
            "dialogue_styles": list(normalized_event.get("dialogue_styles", [])),
        },
    }
    dialogue["audit"] = audit_dialogue(dialogue)
    labels = derive_priority_labels(
        gold_structured_demand,
        latent_priority=priority,
        dialogue_observable_priority=dialogue["audit"].get("dialogue_observable_priority"),
    )
    dialogue["annotations"]["gold_structured_demand"]["labels"] = {
        **gold_structured_demand.get("labels", {}),
        **labels,
    }
    return dialogue


def _generate_rule_conversation(
    dest_id: str,
    station: Dict,
    material: str,
    qty_kg: float,
    priority: int,
    tier: Optional[str] = None,
    ts_hhmm: str = "09:00",
    profile: Optional[Dict[str, str]] = None,
    event: Optional[Dict] = None,
    style: str = "direct",
) -> str:
    """Generate a realistic multi-turn English dispatch dialogue."""
    if tier is None:
        tier = _map_priority_to_tier(priority)
    mat_en = MATERIAL_EN.get(material, material)
    deadline = PRIORITY_DEADLINE.get(priority, 60)

    unit_weight = {
        "cardiac_drug": 0.05, "blood_product": 0.25, "aed": 2.0, "thrombolytic": 0.05,
        "ventilator": 8.0, "icu_drug": 0.1,
        "vaccine": 0.3, "medicine": 0.5, "mask": 0.05,
        "protective_suit": 0.8, "disinfectant": 1.0,
        "food": 0.5, "otc_drug": 0.1, "daily_supply": 0.3,
    }
    uw = unit_weight.get(material, 0.5)
    qty = max(1, round(qty_kg / uw))
    qty_unit = _infer_quantity_units(material)
    qty_phrase = f"{qty} {qty_unit}{'' if qty == 1 else 's'}"
    profile = profile or _build_dialogue_profile(
        event={},
        material=material,
        tier=tier,
        station_name=station["name"],
        dest_id=dest_id,
        ts_hhmm=ts_hhmm,
        deadline_minutes=deadline,
    )
    deadline = int(profile.get("deadline_minutes", deadline))
    kg = round(qty_kg, 1)
    style = _normalize_dialogue_style(style)
    event = event or {}
    must_mention = " ".join(
        str(factor.get("description", "")).strip()
        for factor in _event_must_mention_factors(event)
        if str(factor.get("description", "")).strip()
    )
    optional_factors = _event_optional_factors(event)
    optional_snippets = []
    for factor in optional_factors[:2]:
        description = str(factor.get("description", "")).strip()
        if description:
            optional_snippets.append(description)
    optional_text = " ".join(optional_snippets)

    if style == "technical":
        requester_line = (
            f"[{ts_hhmm}] {profile['requester_title']} ({dest_id}): Clinical dispatch request. "
            f"Please route {qty_phrase} of {mat_en} ({kg} kg) from {station['name']}. "
            f"{must_mention} {profile['time_context']} {optional_text}"
        )
        dispatcher_line = (
            f"[{ts_hhmm}] {profile['dispatcher_title']}: Request acknowledged. "
            f"{profile['handling_note']} Departure from {station['name']} is being prioritized. "
            f"Planned ETA is {deadline} min. {profile['receiver_note']}"
        )
        ack_line = (
            f"[{ts_hhmm}] {profile['requester_title']}: Confirmed. "
            f"{profile['beneficiary_hint']} {profile['followup']}"
        )
    elif style == "conversational":
        requester_line = (
            f"[{ts_hhmm}] {profile['requester_title']} ({dest_id}): Hi, we need {qty_phrase} of "
            f"{mat_en} ({kg} kg) from {station['name']}. {must_mention} {optional_text}"
        )
        dispatcher_line = (
            f"[{ts_hhmm}] {profile['dispatcher_title']}: Got it. {profile['handling_note']} "
            f"We're sending it from {station['name']} now and aiming for about {deadline} min. "
            f"{profile['receiver_note']}"
        )
        ack_line = (
            f"[{ts_hhmm}] {profile['requester_title']}: Thanks. "
            f"{profile['beneficiary_hint']} {profile['followup']}"
        )
    elif style == "dispatch_brief":
        requester_line = (
            f"[{ts_hhmm}] {profile['requester_title']} ({dest_id}): Dispatch brief. "
            f"{qty_phrase} of {mat_en} ({kg} kg) from {station['name']}. {must_mention}"
        )
        dispatcher_line = (
            f"[{ts_hhmm}] {profile['dispatcher_title']}: Copy. "
            f"ETA {deadline} min. {profile['handling_note']} {profile['receiver_note']}"
        )
        ack_line = (
            f"[{ts_hhmm}] {profile['requester_title']}: Roger. "
            f"{profile['beneficiary_hint']}"
        )
    else:
        requester_line = (
            f"[{ts_hhmm}] {profile['requester_title']} ({dest_id}): {profile['opening']} "
            f"We need {qty_phrase} of {mat_en} ({kg} kg) from {station['name']}. "
            f"{must_mention} {profile['time_context']} {optional_text}"
        )
        dispatcher_line = (
            f"[{ts_hhmm}] {profile['dispatcher_title']}: Confirmed. {profile['handling_note']} "
            f"{station['name']} is preparing the payload now. ETA {deadline} min. {profile['receiver_note']}"
        )
        ack_line = (
            f"[{ts_hhmm}] {profile['requester_title']}: Understood. "
            f"{profile['beneficiary_hint']} {profile['followup']}"
        )

    return "\n".join([requester_line.strip(), dispatcher_line.strip(), ack_line.strip()])


def _infer_poi(material: str, priority: int, tier: Optional[str] = None) -> List[str]:
    """根据物资类型和需求层级推断附近兴趣点。"""
    if tier is None:
        tier = _map_priority_to_tier(priority)
    poi_map = {
        "aed":              ["public_space", "residential", "community_center"],
        "cardiac_drug":     ["hospital", "emergency_room", "icu_unit"],
        "blood_product":    ["hospital", "surgery_room", "trauma_center"],
        "thrombolytic":     ["hospital", "neurology_department"],
        "ventilator":       ["hospital", "icu_unit"],
        "icu_drug":         ["hospital", "icu_unit", "pharmacy"],
        "vaccine":          ["clinic", "community_health_center"],
        "medicine":         ["hospital", "pharmacy"],
        "protective_suit":  ["hospital", "quarantine_point"],
        "mask":             ["residential", "shopping_mall", "pharmacy"],
        "disinfectant":     ["market", "residential", "school"],
        "food":             ["residential", "office_building", "shopping_mall"],
        "otc_drug":         ["residential", "pharmacy", "community_center"],
        "daily_supply":     ["residential", "shopping_mall"],
    }
    pois = poi_map.get(material, ["residential"])
    if tier == "life_support" or priority == 1:
        pois = pois + ["emergency_response_center"]
    return pois


# ============================================================================
# 主入口：离线生成
# ============================================================================

def generate_dialogues_offline(
    demand_events: List[Dict],
    stations: List[Dict],
    base_date: str = "2024-03-15",
    styles: Optional[List[str]] = None,
) -> List[Dict]:
    """将 demand_events 批量转换为对话格式（不调用 LLM）。"""
    dialogues = []
    tier_counts: Dict[str, int] = {}
    for idx, (event, style) in enumerate(_expand_event_style_specs(demand_events, styles), start=1):
        dlg = _event_to_dialogue(event, stations, base_date, idx, style=style)
        dialogues.append(dlg)
        tier = _map_priority_to_tier(int(event.get("latent_priority", event.get("priority", 4))))
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    print(
        f"[Module 1] Offline generation complete: {len(dialogues)} dialogues, "
        f"tier distribution {tier_counts}"
    )
    return dialogues


def generate_dialogues_online(
    demand_events: List[Dict],
    stations: List[Dict],
    client: "OpenAI",
    model: str = "gpt-4o-mini",
    base_date: str = "2024-03-15",
    temperature: float = 0.7,
    batch_size: int = 5,
    styles: Optional[List[str]] = None,
) -> List[Dict]:
    """调用 LLM 批量生成自然语言对话，替换规则模板文本。

    每次将 batch_size 条事件打包成一个 LLM 请求，生成对话文本后
    替换离线规则文本，其余字段保持不变。
    """
    from llm4fairrouting.llm.prompt_templates import (
        DRONE_SYSTEM_PROMPT,
        dialogue_generation_prompt,
    )

    event_specs = _expand_event_style_specs(demand_events, styles)
    dialogues = [
        _event_to_dialogue(event, stations, base_date, idx + 1, style=style)
        for idx, (event, style) in enumerate(event_specs)
    ]
    spec_by_id = {
        dialogue["dialogue_id"]: (event, style)
        for dialogue, (event, style) in zip(dialogues, event_specs)
    }

    total = len(dialogues)
    total_batches = math.ceil(total / batch_size) if total else 0
    started_at = time.perf_counter()
    print(
        f"[Module 1] Starting LLM dialogue generation: "
        f"{total} dialogues across {total_batches} batches (batch size {batch_size})",
        flush=True,
    )

    for start in range(0, total, batch_size):
        batch_started_at = time.perf_counter()
        batch_index = start // batch_size + 1
        batch = dialogues[start: start + batch_size]
        batch_context = []
        for dlg in batch:
            event, style = spec_by_id[dlg["dialogue_id"]]
            meta = dlg["metadata"]
            annotations = dlg.get("annotations", {})
            seed_tier = _map_priority_to_tier(int(event.get("latent_priority", event.get("priority", 4))))
            batch_context.append({
                "dialogue_id": dlg["dialogue_id"],
                "timestamp": dlg["timestamp"],
                "dialogue_style": style,
                "demand_tier": seed_tier,
                "demand_tier_label": DEMAND_TIERS.get(seed_tier, ""),
                "scenario_summary": meta.get("scenario_summary", ""),
                "material_type": meta["material_type"],
                "material_en": MATERIAL_EN.get(meta["material_type"], meta["material_type"]),
                "quantity_kg": meta["quantity_kg"],
                "delivery_deadline_minutes": meta.get("delivery_deadline_minutes", 60),
                "origin_station": meta["supply_station_name"],
                "dest_node": str(meta["destination_fid"]),
                "dest_coords": meta["dest_coords"],
                "dest_demographics": meta["dest_demographics"],
                "nearby_poi": meta["nearby_poi"],
                "requester_role": meta["requester_role"],
                "request_channel": meta.get("request_channel", ""),
                "beneficiary_hint": meta.get("beneficiary_hint", ""),
                "handling_notes": meta.get("handling_notes", ""),
                "receiver_notes": meta.get("receiver_notes", ""),
                "must_mention_factors": annotations.get("must_mention_factors", []),
                "optional_factors": annotations.get("optional_factors", []),
            })

        prompt = dialogue_generation_prompt(batch_context)
        print(
            f"  [Module 1] Batch {batch_index}/{total_batches}: "
            f"sending {len(batch)} dialogues to the LLM "
            f"({start}/{total} completed)",
            flush=True,
        )

        try:
            raw = call_llm(client, model, DRONE_SYSTEM_PROMPT, prompt, temperature)
            llm_texts = _parse_llm_batch_response(raw, batch)
        except Exception as e:
            print(f"  [Module 1] LLM failed, falling back to rule-based text: {e}", flush=True)
            llm_texts = {dlg["dialogue_id"]: dlg["conversation"] for dlg in batch}

        for dlg in batch:
            if dlg["dialogue_id"] in llm_texts:
                dlg["conversation"] = llm_texts[dlg["dialogue_id"]]
            dlg["audit"] = audit_dialogue(dlg)
            if not dlg["audit"].get("passed", True):
                event, style = spec_by_id[dlg["dialogue_id"]]
                regenerated = _event_to_dialogue(
                    event,
                    stations,
                    base_date,
                    int(dlg["dialogue_id"][1:]),
                    style=style,
                )
                dlg["conversation"] = regenerated["conversation"]
                dlg["audit"] = regenerated["audit"]

        completed = min(start + len(batch), total)
        print(
            f"  [Module 1] Batch {batch_index}/{total_batches} complete: "
            f"{completed}/{total} dialogues ready "
            f"({completed / total * 100:.1f}%), "
            f"batch {time.perf_counter() - batch_started_at:.1f}s, "
            f"elapsed {time.perf_counter() - started_at:.1f}s",
            flush=True,
        )

    print(
        f"[Module 1] LLM dialogue generation complete in "
        f"{time.perf_counter() - started_at:.1f}s",
        flush=True,
    )
    return dialogues


def _parse_llm_batch_response(raw: str, batch: List[Dict]) -> Dict[str, str]:
    """从 LLM 返回中解析批量对话文本。

    期望格式::

        {"dialogues": [{"dialogue_id": "D0001", "conversation": "..."}, ...]}
    """
    cleaned = raw.strip()
    try:
        obj = parse_json_response(cleaned)
        items = obj.get("dialogues", [])
        return {item["dialogue_id"]: item["conversation"] for item in items}
    except Exception:
        return {}


def load_dialogues_from_file(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================================
# 统一生成入口
# ============================================================================

def generate_dialogues(
    csv_path: str,
    xlsx_path: Optional[str] = None,
    offline: bool = True,
    client: Optional["OpenAI"] = None,
    model: str = "gpt-4o-mini",
    base_date: str = "2024-03-15",
    n_events: Optional[int] = None,
    time_slots: Optional[List[int]] = None,
    temperature: float = 0.7,
    batch_size: int = 5,
    styles: Optional[List[str]] = None,
) -> List[Dict]:
    """Module 1 统一入口：加载数据并生成对话。

    Parameters
    ----------
    csv_path : str
        需求事件路径，优先使用 rich event manifest JSONL。
    xlsx_path : str, optional
        drone_station_locations.csv（站点数据）路径。当 CSV 内含供给点信息时可省略。
    offline : bool
        True=规则模式，False=调用 LLM。
    client : OpenAI, optional
        online 模式必须提供。
    model : str
        LLM 模型名称。
    base_date : str
        基准日期，time_slot=0 对应该日 00:00。
    n_events : int, optional
        最多处理的事件数量，None=全量。
    time_slots : list[int], optional
        仅处理指定 time_slot 的事件。
    temperature : float
        LLM 温度参数（仅 online 模式生效）。
    batch_size : int
        LLM 批处理大小（仅 online 模式生效）。
    """
    stations: List[Dict] = []
    if xlsx_path:
        try:
            stations = load_stations(xlsx_path)
            print(f"[Module 1] Loaded {len(stations)} stations")
        except Exception as e:
            print(f"[Module 1] Station load failed ({e}); using supply data from the CSV")

    events = load_demand_events(csv_path, n_events=n_events, time_slots=time_slots)
    if not events:
        raise ValueError(f"No demand events could be loaded from {csv_path}")
    print(
        f"[Module 1] Loaded {len(events)} demand events "
        f"(time_slots={time_slots}, n_events={n_events})"
    )

    if offline:
        dialogues = generate_dialogues_offline(events, stations, base_date, styles=styles)
    else:
        if client is None:
            raise ValueError("online mode requires an OpenAI client")
        dialogues = generate_dialogues_online(
            events, stations, client, model, base_date, temperature, batch_size, styles=styles
        )
    return dialogues


# ============================================================================
# 保存对话
# ============================================================================

def save_dialogues(dialogues: List[Dict], output_path: str) -> None:
    """将对话列表保存为 JSONL 格式（每行一条 JSON）。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for dlg in dialogues:
            f.write(json.dumps(dlg, ensure_ascii=False) + "\n")
    print(f"[Module 1] Saved {len(dialogues)} dialogues to {output_path}", flush=True)


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    active_env_file = prepare_env_file(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Module 1: Dialogue Generator")
    parser.add_argument(
        "--env-file", type=str,
        default=str(active_env_file) if active_env_file else None,
        help="Environment file path; defaults to the project .env when present",
    )
    parser.add_argument(
        "--csv", type=str,
        default=str(PRIMARY_EVENT_DATA_PATH),
        help="Rich event manifest JSONL path",
    )
    parser.add_argument(
        "--stations", type=str,
        default=str(STATION_DATA_PATH),
        help=f"{STATION_DATA_FILENAME} 路径（默认使用项目 seed 数据）",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(DEMAND_DIALOGUES_PATH),
        help="Output JSONL path",
    )
    parser.add_argument("--offline", action="store_true", help="离线模式，不调用 LLM")
    parser.add_argument("--n-events", type=int, default=None, help="最多处理的事件数")
    parser.add_argument("--time-slots", type=int, nargs="+", default=None,
                        help="仅处理指定 time_slot（空格分隔）")
    parser.add_argument("--base-date", type=str, default="2024-03-15")
    parser.add_argument("--api-base", type=str, default=env_text("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", type=str, default=env_text("OPENAI_API_KEY"))
    parser.add_argument("--model", type=str, default=env_text("LLM4FAIRROUTING_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--styles",
        type=str,
        nargs="+",
        default=None,
        help="Optional dialogue styles to generate, e.g. direct technical conversational dispatch_brief",
    )
    args = parser.parse_args()

    client = None
    if not args.offline:
        client = create_openai_client(args.api_base, args.api_key)

    dialogues = generate_dialogues(
        csv_path=args.csv,
        xlsx_path=args.stations,
        offline=args.offline,
        client=client,
        model=args.model,
        base_date=args.base_date,
        n_events=args.n_events,
        time_slots=args.time_slots,
        temperature=args.temperature,
        batch_size=args.batch_size,
        styles=args.styles,
    )

    save_dialogues(dialogues, args.output)


if __name__ == "__main__":
    main()
