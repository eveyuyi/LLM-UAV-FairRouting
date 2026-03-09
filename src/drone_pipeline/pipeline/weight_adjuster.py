"""
Module 3a: Weight Adjustment — 根据结构化需求，由 LLM 分配优先级和求解器权重。
"""

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from openai import OpenAI


# ============================================================================
# LLM 调用 (复用 context_extractor 中的模式)
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
# 权重调整
# ============================================================================

def adjust_weights(
    demands: List[Dict],
    client: "OpenAI",
    model: str,
    city_context: Optional[Dict] = None,
    temperature: float = 0.0,
) -> Dict:
    """调用 LLM 为一组需求分配优先级和求解器权重。"""
    from drone_pipeline.prompts.drone_prompt import DRONE_SYSTEM_PROMPT, weight_adjustment_prompt
    prompt = weight_adjustment_prompt(demands, city_context)
    print(f"  [Module 3a] {len(demands)} 条需求，调用 LLM 分配权重 ...")

    raw = _call_llm(client, model, DRONE_SYSTEM_PROMPT, prompt, temperature)
    result = _parse_json_response(raw)

    n_configs = len(result.get("demand_configs", []))
    n_supp = len(result.get("supplementary_constraints", []))
    print(f"  [Module 3a] 返回 {n_configs} 个需求配置, {n_supp} 个补充约束")
    return result


# ============================================================================
# 离线 / Mock 模式 — 基于需求层级的规则排序与权重映射
# ============================================================================

# 四层需求体系基础参数
TIER_PARAMS = {
    "life_support": {"alpha": 20.0, "beta": 0.1,  "priority": 1},
    "critical":     {"alpha": 10.0, "beta": 0.3,  "priority": 1},
    "regular":      {"alpha":  4.0, "beta": 1.0,  "priority": 2},
    "consumer":     {"alpha":  1.0, "beta": 2.0,  "priority": 3},
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
    if tier and tier in TIER_PARAMS:
        return tier
    urgency = demand.get("urgency", "normal")
    return _URGENCY_TIER_MAP.get(urgency, "regular")


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
    """不调用 LLM，使用四层需求体系规则排序并分配权重。

    Module 3 职责：
    1. 在时间窗口内对所有需求按层级 + 脆弱性信号排序 (window_rank)
    2. 按层级分配基础 alpha/beta/priority，并根据信号微调
    """
    import copy

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
        params = copy.deepcopy(TIER_PARAMS.get(tier, TIER_PARAMS["regular"]))

        reasoning_parts = [f"层级: {tier}"]

        # Alpha uplift for extreme patient conditions
        patient_cond = str(signals.get("patient_condition", "") or "").lower()
        if any(kw in patient_cond for kw in ("骤停", "心梗", "大出血", "脑梗")):
            params["alpha"] = min(20.0, params["alpha"] + 3.0)
            reasoning_parts.append("危急患者状态")

        # Priority/beta adjustment for vulnerable populations
        if vuln["elderly_involved"] or vuln["vulnerable_community"]:
            params["priority"] = max(1, params["priority"] - 1)
            params["alpha"] = min(20.0, params["alpha"] + 1.5)
            reasoning_parts.append(f"老年/弱势社区 (比例 {vuln['elderly_ratio']:.0%})")

        if vuln["children_involved"]:
            params["priority"] = max(1, params["priority"] - 1)
            reasoning_parts.append("儿童受益")

        # Consumer OTC with urgency → upgrade
        if tier == "consumer" and "otc_drug" in str(d.get("cargo", {}).get("type", "")):
            if "发烧" in patient_cond or "儿童" in patient_cond:
                params["alpha"] = max(params["alpha"], 3.0)
                params["priority"] = min(params["priority"], 2)
                reasoning_parts.append("OTC急需(儿童发烧)，提升至regular级别")

        configs.append({
            "demand_id": d["demand_id"],
            "demand_tier": tier,
            "alpha": round(params["alpha"], 1),
            "beta": round(params["beta"], 2),
            "priority": params["priority"],
            "window_rank": rank,
            "reasoning": "；".join(reasoning_parts),
        })

        # Supplementary constraints
        nearby_poi = d.get("destination", {}).get("type", "")
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
