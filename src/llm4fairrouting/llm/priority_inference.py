"""
Module 3a: Priority Ranking — 根据结构化需求，由 LLM 分配优先级并生成补充约束建议。
"""

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from openai import OpenAI


# ============================================================================
# LLM 调用 (复用 demand_extraction 中的模式)
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
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1]
    if "```" in cleaned:
        cleaned = cleaned.split("```", 1)[0]
    return json.loads(cleaned.strip())


# ============================================================================
# 优先级调整
# ============================================================================

def adjust_weights(
    demands: List[Dict],
    client: "OpenAI",
    model: str,
    city_context: Optional[Dict] = None,
    temperature: float = 0.0,
) -> Dict:
    """调用 LLM 为一组需求分配优先级。"""
    from llm4fairrouting.llm.prompt_templates import (
        DRONE_SYSTEM_PROMPT,
        weight_adjustment_prompt,
    )
    prompt = weight_adjustment_prompt(demands, city_context)
    print(f"  [Module 3a] {len(demands)} 条需求，调用 LLM 分配权重 ...")

    raw = _call_llm(client, model, DRONE_SYSTEM_PROMPT, prompt, temperature)
    result = _normalize_weight_config(_parse_json_response(raw))

    n_configs = len(result.get("demand_configs", []))
    n_supp = len(result.get("supplementary_constraints", []))
    print(f"  [Module 3a] 返回 {n_configs} 个需求配置, {n_supp} 个补充约束")
    return result


# ============================================================================
# 离线 / Mock 模式 — 基于需求层级的规则排序
# ============================================================================

# 四层需求体系基础优先级（1=最高）
TIER_PRIORITIES = {
    "life_support": 1,
    "critical": 2,
    "regular": 3,
    "consumer": 4,
}

# 层级排序权重（数值越小排名越靠前）
_TIER_RANK_BASE = {
    "life_support": 0,
    "critical":     100,
    "regular":      200,
    "consumer":     300,
}

# 兼容旧版 urgency 字段（部分旧数据可能仍使用）
_URGENCY_TIER_MAP = {
    "extreme": "life_support",
    "urgent":  "critical",
    "express": "regular",
    "normal":  "regular",
}


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
    """从 priority_evaluation_signals 或 context_signals 中提取脆弱性信号。"""
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


def adjust_weights_offline(demands: List[Dict]) -> Dict:
    """不调用 LLM，使用四层需求体系规则排序并分配优先级。

    Module 3 职责：
    1. 在时间窗口内对所有需求按层级 + 脆弱性信号排序 (window_rank)
    2. 按层级分配 priority，并根据信号微调
    """
    configs = []
    supplementary = []

    # Step 1: 计算每条需求的排序得分
    scored = []
    for d in demands:
        tier = _get_tier(d)
        vuln = _extract_vulnerability(d)
        signals = d.get("priority_evaluation_signals", {})

        rank_score = _TIER_RANK_BASE.get(tier, 200)

        # Within-tier adjustments (lower score = higher rank)
        if vuln["elderly_involved"] or vuln["vulnerable_community"]:
            rank_score -= 15
        if vuln["children_involved"]:
            rank_score -= 10
        patient_cond = str(signals.get("patient_condition", "") or "").lower()
        if any(kw in patient_cond for kw in ("骤停", "心梗", "大出血", "脑梗", "溶栓")):
            rank_score -= 20
        if any(kw in patient_cond for kw in ("重症", "icu", "术后")):
            rank_score -= 10

        scored.append((rank_score, d, tier, vuln, signals))

    # Step 2: 按排序得分排序（分数低=优先级高）
    scored.sort(key=lambda x: x[0])

    for rank, (rank_score, d, tier, vuln, signals) in enumerate(scored, start=1):
        priority = TIER_PRIORITIES.get(tier, TIER_PRIORITIES["regular"])

        reasoning_parts = [f"层级: {tier}"]

        patient_cond = str(signals.get("patient_condition", "") or "").lower()
        if any(kw in patient_cond for kw in ("骤停", "心梗", "大出血", "脑梗")):
            priority = max(1, priority - 1)
            reasoning_parts.append("危急患者状态")

        if vuln["elderly_involved"] or vuln["vulnerable_community"]:
            priority = max(1, priority - 1)
            reasoning_parts.append(f"老年/弱势社区 (比例 {vuln['elderly_ratio']:.0%})")

        if vuln["children_involved"]:
            priority = max(1, priority - 1)
            reasoning_parts.append("儿童受益")

        # Consumer OTC with urgency → upgrade
        if tier == "consumer" and "otc_drug" in str(d.get("cargo", {}).get("type", "")):
            if "发烧" in patient_cond or "儿童" in patient_cond:
                priority = min(priority, 3)
                reasoning_parts.append("OTC急需(儿童发烧)，提升至regular级别")

        configs.append({
            "demand_id": d["demand_id"],
            "demand_tier": tier,
            "priority": priority,
            "window_rank": rank,
            "reasoning": "；".join(reasoning_parts),
        })

        # Supplementary constraints
        for sig in d.get("context_signals", []) + [signals.get("nearby_critical_facility", "")]:
            sig_str = str(sig or "").lower()
            if "学校" in sig_str or "kindergarten" in sig_str or "school" in sig_str:
                dest = d.get("destination", {})
                coords = dest.get("coords", [0, 0])
                supplementary.append({
                    "type": "noise_avoidance",
                    "description": f"规避 {dest.get('fid', '')} 附近学校/幼儿园区域上空",
                    "affected_zone": {"center": coords, "radius_m": 300},
                })
                break

    return {
        "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
        "demand_configs": configs,
        "supplementary_constraints": supplementary,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Module 3a: Weight Adjustment")
    parser.add_argument(
        "--input", type=str,
        default=str(PROJECT_ROOT / "data" / "drone" / "extracted_demands.json"),
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        windows_data = json.load(f)

    all_results = []
    for window in windows_data:
        demands = window.get("demands", [])
        tw = window.get("time_window", "")
        print(f"[Module 3a] 窗口 {tw}: {len(demands)} 条需求")

        if args.offline:
            result = adjust_weights_offline(demands)
        else:
            import os
            base = args.api_base or os.getenv("LLMOPT_API_BASE_URL", "http://35.220.164.252:3888/v1/")
            key = args.api_key or os.getenv("LLMOPT_API_KEY")
            if not key:
                raise ValueError("需要 API key")
            client = OpenAI(base_url=base, api_key=key)
            result = adjust_weights(demands, client, args.model)

        result["time_window"] = tw
        all_results.append(result)

    out_path = args.output or str(PROJECT_ROOT / "data" / "drone" / "weight_configs.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"结果保存至 {out_path}")


if __name__ == "__main__":
    main()
