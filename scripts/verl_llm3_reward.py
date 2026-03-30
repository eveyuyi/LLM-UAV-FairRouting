"""Custom reward function for LLM3 priority-ranking GRPO in VERL."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

JSON_WEIGHT = 0.10
PRIORITY_WEIGHT = 0.55
PAIRWISE_WEIGHT = 0.25
TOPK_WEIGHT = 0.10


def _maybe_parse_json(value):
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def _normalize_priority(value) -> Optional[int]:
    try:
        priority = int(value)
    except (TypeError, ValueError):
        return None
    return min(max(priority, 1), 4)


def _normalize_rank(value) -> Optional[int]:
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return None
    return rank if rank >= 1 else None


def _extract_priority_labels(payload) -> List[Dict]:
    data = _maybe_parse_json(payload)
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        for key in ("priority_labels", "ranked_demands", "demands", "labels"):
            if isinstance(data.get(key), list):
                candidates = data[key]
                break
        else:
            return []
    else:
        return []

    labels: List[Dict] = []
    seen_ids = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        demand_id = str(item.get("demand_id", "")).strip()
        if not demand_id or demand_id in seen_ids:
            continue
        seen_ids.add(demand_id)
        labels.append(
            {
                "demand_id": demand_id,
                "priority": _normalize_priority(item.get("priority")),
                "window_rank": _normalize_rank(item.get("window_rank")),
            }
        )
    return labels


def _prediction_map(labels: Iterable[Dict]) -> Dict[str, Dict]:
    return {
        str(label["demand_id"]): label
        for label in labels
        if label.get("demand_id")
    }


def _priority_match_score(pred_priority: Optional[int], gt_priority: int) -> float:
    if pred_priority is None:
        return 0.0
    return max(0.0, 1.0 - (abs(int(pred_priority) - int(gt_priority)) / 3.0))


def _compare_pair(pred_map: Dict[str, Dict], higher_id: str, lower_id: str) -> bool:
    higher = pred_map.get(higher_id)
    lower = pred_map.get(lower_id)
    if not higher or not lower:
        return False

    higher_priority = higher.get("priority")
    lower_priority = lower.get("priority")
    if higher_priority is None or lower_priority is None:
        return False
    if higher_priority != lower_priority:
        return int(higher_priority) < int(lower_priority)

    higher_rank = higher.get("window_rank")
    lower_rank = lower.get("window_rank")
    if higher_rank is None or lower_rank is None:
        return False
    return int(higher_rank) < int(lower_rank)


def _ordered_prediction_ids(pred_labels: List[Dict]) -> List[str]:
    sortable = []
    for label in pred_labels:
        demand_id = str(label.get("demand_id", ""))
        if not demand_id:
            continue
        priority = label.get("priority")
        rank = label.get("window_rank")
        sortable.append(
            (
                99 if priority is None else int(priority),
                99 if rank is None else int(rank),
                demand_id,
            )
        )
    sortable.sort()
    return [item[2] for item in sortable]


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if str(data_source or "") != "llm3_priority_window":
        return 0.0

    gt = _maybe_parse_json(ground_truth) or {}
    if isinstance(gt, dict) and "ground_truth" in gt:
        gt = _maybe_parse_json(gt.get("ground_truth")) or {}
    if not isinstance(gt, dict):
        return 0.0

    pred_labels = _extract_priority_labels(solution_str)
    pred_map = _prediction_map(pred_labels)
    json_score = 1.0 if pred_labels else 0.0

    gt_labels = _extract_priority_labels(gt.get("priority_labels", []))
    if not gt_labels:
        return JSON_WEIGHT * json_score

    priority_scores = []
    for gt_label in gt_labels:
        gt_priority = _normalize_priority(gt_label.get("priority"))
        if gt_priority is None:
            continue
        pred_priority = pred_map.get(gt_label["demand_id"], {}).get("priority")
        priority_scores.append(_priority_match_score(pred_priority, gt_priority))
    priority_score = sum(priority_scores) / len(priority_scores) if priority_scores else 0.0

    pairwise_preferences = gt.get("pairwise_preferences", [])
    if pairwise_preferences:
        pairwise_hits = 0
        for pair in pairwise_preferences:
            if not isinstance(pair, dict):
                continue
            if _compare_pair(
                pred_map,
                str(pair.get("higher_priority_demand_id", "")),
                str(pair.get("lower_priority_demand_id", "")),
            ):
                pairwise_hits += 1
        pairwise_score = pairwise_hits / len(pairwise_preferences)
    else:
        pairwise_score = 1.0

    critical_topk_targets = [
        str(item)
        for item in (gt.get("critical_topk_targets", []) or [])
        if str(item)
    ]
    if critical_topk_targets:
        predicted_topk = _ordered_prediction_ids(pred_labels)[: len(critical_topk_targets)]
        overlap = len(set(predicted_topk) & set(critical_topk_targets))
        topk_score = overlap / len(critical_topk_targets)
    else:
        topk_score = 1.0

    total_score = (
        JSON_WEIGHT * json_score
        + PRIORITY_WEIGHT * priority_score
        + PAIRWISE_WEIGHT * pairwise_score
        + TOPK_WEIGHT * topk_score
    )
    return max(0.0, min(1.0, float(total_score)))
