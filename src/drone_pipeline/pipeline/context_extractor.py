"""
Module 2: Context Extraction — 按时间窗口聚合对话，调用 LLM 提取结构化需求。
"""

import json
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from openai import OpenAI


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
# LLM 调用
# ============================================================================

def _call_llm(
    client: "OpenAI",
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            print(f"  [LLM] attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2.0)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def _parse_json_response(text: str) -> Dict:
    """从 LLM 返回中提取 JSON（兼容 markdown code fence）。"""
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1]
    if "```" in cleaned:
        cleaned = cleaned.split("```", 1)[0]
    return json.loads(cleaned.strip())


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

        # Also surface demand_tier from metadata as a cross-check signal
        if not demand.get("demand_tier") and meta.get("demand_tier"):
            demand.setdefault("cargo", {})["demand_tier_hint"] = meta["demand_tier"]

        demand["request_timestamp"] = dialogue.get("timestamp")

    return demands


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
    from drone_pipeline.prompts.drone_prompt import DRONE_SYSTEM_PROMPT, context_extraction_prompt
    prompt = context_extraction_prompt(dialogues, time_window)
    print(f"  [Module 2] 窗口 {time_window}: {len(dialogues)} 条对话，调用 LLM ...")

    raw = _call_llm(client, model, DRONE_SYSTEM_PROMPT, prompt, temperature)
    result = _parse_json_response(raw)

    # Enrich with coords from metadata (LLM cannot know these from text alone)
    result["demands"] = _enrich_demands_with_metadata(
        result.get("demands", []), dialogues
    )

    n_demands = len(result.get("demands", []))
    print(f"  [Module 2] 提取到 {n_demands} 条需求")
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
    print(f"[Module 2] 共 {len(windows)} 个时间窗口")

    results = []
    for label, group in windows.items():
        result = extract_demands_for_window(group, label, client, model, temperature)
        results.append(result)

    return results


# ============================================================================
# 离线 / Mock 模式 — 不调用 LLM，直接从对话 metadata 构造需求
# ============================================================================

# 层级 → 硬/软时限和截止时间
_TIER_DEADLINE = {
    "life_support": (15,  "hard"),
    "critical":     (30,  "hard"),
    "regular":      (90,  "soft"),
    "consumer":     (120, "soft"),
}

# 层级 → 自报紧急程度（供 Module 3 参考）
_TIER_URGENCY_LABEL = {
    "life_support": "极端紧急",
    "critical":     "紧急",
    "regular":      "加急/常规",
    "consumer":     "常规",
}

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
    """不经过 LLM，直接从对话的 metadata 生成结构化需求（用于测试 Module 3）。

    Module 2 职责：提取信息 + 优先级评估信号，不做优先级排序。
    优先级排序由 Module 3 完成。
    """
    windows = group_by_time_window(dialogues, window_minutes)
    results = []

    for label, group in windows.items():
        demands = []
        for d in group:
            meta = d["metadata"]
            tier = meta.get("demand_tier", "regular")
            deadline_min, constraint_type = _TIER_DEADLINE.get(tier, (90, "soft"))

            elderly_ratio = meta["dest_demographics"].get("elderly_ratio", 0.0)
            population = meta["dest_demographics"].get("population", 0)
            nearby_poi = meta.get("nearby_poi", [])

            demands.append({
                "demand_id": f"REQ{len(demands)+1:03d}",
                "source_dialogue_id": d["dialogue_id"],
                "request_timestamp": d.get("timestamp"),
                "origin": {
                    "fid": meta["origin_fid"],
                    "coords": meta["origin_coords"],
                    "type": "supply_station",
                },
                "destination": {
                    "fid": meta["destination_fid"],
                    "coords": meta["dest_coords"],
                    "type": _infer_dest_type(nearby_poi),
                },
                "cargo": {
                    "type": meta.get("material_type", "medicine"),
                    "type_cn": meta.get("material_type", "医疗物资"),
                    "demand_tier": tier,
                    "weight_kg": meta.get("quantity_kg", 2.0),
                    "quantity": max(1, round(meta.get("quantity_kg", 2.0))),
                    "quantity_unit": "units",
                    "temperature_sensitive": tier in ("life_support", "critical"),
                },
                "demand_tier": tier,
                "time_constraint": {
                    "type": constraint_type,
                    "description": f"{deadline_min}分钟内配送",
                    "deadline_minutes": deadline_min,
                },
                "priority_evaluation_signals": {
                    "patient_condition": meta.get("scenario", None),
                    "time_sensitivity": _TIER_URGENCY_LABEL.get(tier, "常规"),
                    "population_vulnerability": {
                        "elderly_involved": elderly_ratio > 0.40,
                        "children_involved": False,
                        "elderly_ratio": elderly_ratio,
                        "population": population,
                        "vulnerable_community": elderly_ratio > 0.50,
                    },
                    "medical_urgency_self_report": _TIER_URGENCY_LABEL.get(tier, "常规"),
                    "requester_role": meta.get("requester_role", "community_health_worker"),
                    "scenario_context": meta.get("scenario", ""),
                    "nearby_critical_facility": nearby_poi[0] if nearby_poi else None,
                },
                "context_signals": [
                    f"需求层级: {tier} ({meta.get('scenario', '')})",
                    f"目的地人口 {population}，老年比例 {elderly_ratio}",
                ],
            })

        results.append({"time_window": label, "demands": demands})

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Module 2: Context Extraction")
    parser.add_argument(
        "--input", type=str,
        default=str(PROJECT_ROOT / "data" / "drone" / "mock_dialogues.jsonl"),
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--offline", action="store_true", help="离线模式，不调用 LLM")
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dialogues = [json.loads(l.strip()) for l in f if l.strip()]

    print(f"读取 {len(dialogues)} 条对话")

    if args.offline:
        results = extract_demands_offline(dialogues, args.window)
    else:
        import os
        base = args.api_base or os.getenv("LLMOPT_API_BASE_URL", "http://35.220.164.252:3888/v1/")
        key = args.api_key or os.getenv("LLMOPT_API_KEY")
        if not key:
            raise ValueError("需要 API key: 设置 LLMOPT_API_KEY 或 --api-key")
        client = OpenAI(base_url=base, api_key=key)
        results = extract_all_demands(dialogues, client, args.model, args.window)

    out_path = args.output or str(PROJECT_ROOT / "data" / "drone" / "extracted_demands.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"结果保存至 {out_path}")


if __name__ == "__main__":
    main()
