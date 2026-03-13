"""
Module 1: Dialogue Generator — 从 daily_demand_events.csv 读取结构化需求事件，
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
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from openai import OpenAI

from llm4fairrouting.data.seed_paths import (
    DEMAND_EVENTS_FILENAME,
    DEMAND_EVENTS_PATH,
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


def _pick_template(priority: int, material: str) -> str:
    """兼容旧测试/调用方的模板选择入口。"""
    tier = _map_priority_to_tier(priority)
    rng = random.Random(hash((priority, material)))
    return _pick_tier_template(tier, material, rng)


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


def _infer_material_type(supply_type: str, priority: int, unique_id: str = "") -> str:
    """Infer material type from supply_type and priority (for CSV without material_type).

    Medical (医疗) supply types map to clinical materials;
    Commercial (商业) supply types map to consumer goods.
    Priority determines urgency tier but not material category.
    """
    rng = random.Random(hash(unique_id))
    if supply_type == "医疗":
        if priority == 1:
            return rng.choice(["cardiac_drug", "blood_product", "aed", "thrombolytic"])
        elif priority == 2:
            return rng.choice(["ventilator", "icu_drug"])
        elif priority == 3:
            return rng.choice(["vaccine", "medicine", "protective_suit"])
        else:
            return rng.choice(["medicine", "vaccine"])
    else:  # 商业 or other
        if priority == 1:
            return rng.choice(["medicine", "protective_suit", "disinfectant"])
        elif priority == 2:
            return rng.choice(["medicine", "mask", "disinfectant"])
        elif priority == 3:
            return rng.choice(["food", "mask", "daily_supply"])
        else:
            return rng.choice(["food", "daily_supply", "otc_drug"])


def load_demand_events(
    csv_path: str,
    n_events: Optional[int] = None,
    time_slots: Optional[List[int]] = None,
) -> List[Dict]:
    """读取需求事件 CSV（支持旧 demand_events_5min.csv 和新 daily_demand_events.csv）。

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

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

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
    supply_type = str(event.get("supply_type", ""))
    if supply_type == "医疗":
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
    time_slot = int(event["time_slot"])
    base_dt = datetime.strptime(base_date, "%Y-%m-%d")
    ts = base_dt + timedelta(minutes=time_slot * 5)
    timestamp = ts.strftime("%Y-%m-%dT%H:%M:%S")
    ts_hhmm = ts.strftime("%H:%M")

    dest_lon = float(event["demand_lon"])
    dest_lat = float(event["demand_lat"])
    original_material = str(event.get("material_type", "medicine"))
    qty_kg = float(event.get("quantity_kg", 1.0))
    priority = int(event.get("priority", 3))
    demand_node_id = str(event.get("demand_node_id", f"D{dialogue_idx}"))
    dest_fid = event.get("demand_node_index", event.get("demand_node_id", f"D{dialogue_idx}"))

    tier = _map_priority_to_tier(priority)

    # Use supply point from CSV if available, otherwise find nearest station
    if event.get("supply_lon") is not None and event.get("supply_lat") is not None:
        station = _supply_to_station(event)
    elif stations:
        station = _find_nearest_station(dest_lon, dest_lat, stations)
    else:
        station = {
            "station_id": f"ST{dialogue_idx:03d}",
            "name": f"Station-{dialogue_idx}",
            "lon": dest_lon,
            "lat": dest_lat,
        }

    if conversation is None:
        conversation = _generate_rule_conversation(
            demand_node_id, station, original_material, qty_kg, priority, tier, ts_hhmm
        )

    rng = random.Random(hash((dest_lon, dest_lat)))
    elderly_ratio = round(rng.uniform(0.15, 0.60), 2)
    population = rng.randint(500, 8000)

    scenario = _SCENARIO_LABEL.get(
        (tier, original_material),
        f"{DEMAND_TIERS[tier]} — {MATERIAL_EN.get(original_material, original_material)}",
    )
    requester_role = _REQUESTER_ROLE.get(tier, "community_health_worker")

    return {
        "dialogue_id": f"D{dialogue_idx:04d}",
        "timestamp": timestamp,
        "conversation": conversation,
        "metadata": {
            "event_id": str(event.get("event_id", "")),
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
            "priority": priority,
            "demand_tier": tier,
            "scenario": scenario,
            "requester_role": requester_role,
            "demand_type": str(event.get("demand_type", "")),
            "supply_station_name": station["name"],
            "supply_type": str(event.get("supply_type", "")),
        },
    }


def _generate_rule_conversation(
    dest_id: str,
    station: Dict,
    material: str,
    qty_kg: float,
    priority: int,
    tier: Optional[str] = None,
    ts_hhmm: str = "09:00",
) -> str:
    """Generate a multi-turn English dialogue from tier × material templates."""
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

    rng = random.Random(hash((dest_id, material, priority)))
    tpl = _pick_tier_template(tier, material, rng)

    return tpl.format(
        t=ts_hhmm,
        dest_id=dest_id,
        origin_name=station["name"],
        mat_en=mat_en,
        qty=qty,
        kg=round(qty_kg, 1),
        deadline=deadline,
    )


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
) -> List[Dict]:
    """将 demand_events 批量转换为对话格式（不调用 LLM）。"""
    dialogues = []
    for idx, event in enumerate(demand_events):
        dlg = _event_to_dialogue(event, stations, base_date, idx + 1)
        dialogues.append(dlg)
    # summarize tier distribution
    from collections import Counter
    tier_counts = Counter(d["metadata"]["demand_tier"] for d in dialogues)
    print(f"[Module 1] Offline generation: {len(dialogues)} dialogues, tier distribution: {dict(tier_counts)}")
    return dialogues


def generate_dialogues_online(
    demand_events: List[Dict],
    stations: List[Dict],
    client: "OpenAI",
    model: str = "gpt-4o-mini",
    base_date: str = "2024-03-15",
    temperature: float = 0.7,
    batch_size: int = 5,
) -> List[Dict]:
    """调用 LLM 批量生成自然语言对话，替换规则模板文本。

    每次将 batch_size 条事件打包成一个 LLM 请求，生成对话文本后
    替换离线规则文本，其余字段保持不变。
    """
    from llm4fairrouting.llm.prompt_templates import (
        DRONE_SYSTEM_PROMPT,
        dialogue_generation_prompt,
    )

    dialogues = generate_dialogues_offline(demand_events, stations, base_date)

    total = len(dialogues)
    print(f"[Module 1] Starting LLM dialogue generation: {total} total, batch size {batch_size}")

    for start in range(0, total, batch_size):
        batch = dialogues[start: start + batch_size]
        batch_context = []
        for dlg in batch:
            meta = dlg["metadata"]
            batch_context.append({
                "dialogue_id": dlg["dialogue_id"],
                "timestamp": dlg["timestamp"],
                "demand_tier": meta["demand_tier"],
                "demand_tier_label": DEMAND_TIERS.get(meta["demand_tier"], ""),
                "scenario": meta["scenario"],
                "material_type": meta["material_type"],
                "material_en": MATERIAL_EN.get(meta["material_type"], meta["material_type"]),
                "quantity_kg": meta["quantity_kg"],
                "priority": meta["priority"],
                "origin_station": meta["supply_station_name"],
                "dest_node": str(meta["destination_fid"]),
                "dest_coords": meta["dest_coords"],
                "dest_demographics": meta["dest_demographics"],
                "nearby_poi": meta["nearby_poi"],
                "requester_role": meta["requester_role"],
            })

        prompt = dialogue_generation_prompt(batch_context)
        print(f"  [Module 1] batch {start // batch_size + 1}: {len(batch)} items, calling LLM ...")

        try:
            raw = call_llm(client, model, DRONE_SYSTEM_PROMPT, prompt, temperature)
            llm_texts = _parse_llm_batch_response(raw, batch)
        except Exception as e:
            print(f"  [Module 1] LLM failed, falling back to rule-based text: {e}")
            llm_texts = {dlg["dialogue_id"]: dlg["conversation"] for dlg in batch}

        for dlg in batch:
            if dlg["dialogue_id"] in llm_texts:
                dlg["conversation"] = llm_texts[dlg["dialogue_id"]]

    print(f"[Module 1] LLM dialogue generation complete")
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
) -> List[Dict]:
    """Module 1 统一入口：加载数据并生成对话。

    Parameters
    ----------
    csv_path : str
        需求事件 CSV 路径（支持旧 demand_events_5min.csv 和新
        daily_demand_events.csv 格式）。
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
            print(f"[Module 1] 加载 {len(stations)} 个站点")
        except Exception as e:
            print(f"[Module 1] 站点数据加载失败 ({e})，将使用 CSV 中的供给点信息")

    events = load_demand_events(csv_path, n_events=n_events, time_slots=time_slots)
    if not events:
        raise ValueError(f"未能从 {csv_path} 加载任何需求事件")
    print(f"[Module 1] 加载 {len(events)} 条需求事件 (time_slots={time_slots}, n_events={n_events})")

    if offline:
        return generate_dialogues_offline(events, stations, base_date)
    else:
        if client is None:
            raise ValueError("online 模式需要提供 OpenAI client")
        return generate_dialogues_online(
            events, stations, client, model, base_date, temperature, batch_size
        )


# ============================================================================
# 保存对话
# ============================================================================

def save_dialogues(dialogues: List[Dict], output_path: str) -> None:
    """将对话列表保存为 JSONL 格式（每行一条 JSON）。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for dlg in dialogues:
            f.write(json.dumps(dlg, ensure_ascii=False) + "\n")
    print(f"[Module 1] 对话保存至 {output_path} ({len(dialogues)} 条)")


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Module 1: Dialogue Generator")
    parser.add_argument(
        "--csv", type=str,
        default=str(DEMAND_EVENTS_PATH),
        help=f"需求事件 CSV 路径（默认 {DEMAND_EVENTS_FILENAME}）",
    )
    parser.add_argument(
        "--stations", type=str,
        default=str(STATION_DATA_PATH),
        help=f"{STATION_DATA_FILENAME} 路径（默认使用项目 seed 数据）",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "data" / "drone" / "generated_dialogues.jsonl"),
        help="输出 JSONL 路径",
    )
    parser.add_argument("--offline", action="store_true", help="离线模式，不调用 LLM")
    parser.add_argument("--n-events", type=int, default=None, help="最多处理的事件数")
    parser.add_argument("--time-slots", type=int, nargs="+", default=None,
                        help="仅处理指定 time_slot（空格分隔）")
    parser.add_argument("--base-date", type=str, default="2024-03-15")
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=5)
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
    )

    save_dialogues(dialogues, args.output)


if __name__ == "__main__":
    main()
