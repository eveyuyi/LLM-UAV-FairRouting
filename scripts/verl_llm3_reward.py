"""Custom reward function for LLM3 priority-ranking GRPO in VERL.

The reward is intentionally aligned with the downstream rank-only evaluation:
- exact priority assignment is the primary objective
- urgent-item identification (P1/P2) is explicitly rewarded
- pairwise and top-k ordering remain important, but secondary

This avoids a failure mode where the model can earn a decent reward by
outputting approximately-correct rankings while collapsing exact priorities.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

JSON_WEIGHT = 0.05
BALANCED_PRIORITY_WEIGHT = 0.25
EXACT_PRIORITY_WEIGHT = 0.25
DISTANCE_PRIORITY_WEIGHT = 0.05
PAIRWISE_WEIGHT = 0.10
TOPK_WEIGHT = 0.05
URGENT_F1_WEIGHT = 0.10
PRIORITY1_RECALL_WEIGHT = 0.10
DISTRIBUTION_WEIGHT = 0.05

URGENT_THRESHOLD = 2
PRIORITY_IMPORTANCE = {
    1: 4.0,
    2: 3.0,
    3: 2.0,
    4: 1.0,
}


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


def _safe_positive_int(value, default: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed >= 1 else int(default)


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


def _priority_weight(priority: int) -> float:
    return float(PRIORITY_IMPORTANCE.get(int(priority), 1.0))


def _weighted_exact_priority_score(pred_map: Dict[str, Dict], gt_labels: List[Dict]) -> float:
    numerator = 0.0
    denominator = 0.0
    for gt_label in gt_labels:
        gt_priority = _normalize_priority(gt_label.get("priority"))
        if gt_priority is None:
            continue
        weight = _priority_weight(gt_priority)
        denominator += weight
        pred_priority = pred_map.get(gt_label["demand_id"], {}).get("priority")
        if pred_priority is not None and int(pred_priority) == int(gt_priority):
            numerator += weight
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _weighted_distance_priority_score(pred_map: Dict[str, Dict], gt_labels: List[Dict]) -> float:
    numerator = 0.0
    denominator = 0.0
    for gt_label in gt_labels:
        gt_priority = _normalize_priority(gt_label.get("priority"))
        if gt_priority is None:
            continue
        weight = _priority_weight(gt_priority)
        denominator += weight
        pred_priority = pred_map.get(gt_label["demand_id"], {}).get("priority")
        numerator += weight * _priority_match_score(pred_priority, gt_priority)
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _balanced_priority_recall_score(pred_map: Dict[str, Dict], gt_labels: List[Dict]) -> float:
    hits_by_priority: Dict[int, int] = {}
    totals_by_priority: Dict[int, int] = {}
    for gt_label in gt_labels:
        gt_priority = _normalize_priority(gt_label.get("priority"))
        if gt_priority is None:
            continue
        totals_by_priority[gt_priority] = totals_by_priority.get(gt_priority, 0) + 1
        pred_priority = pred_map.get(gt_label["demand_id"], {}).get("priority")
        if pred_priority is not None and int(pred_priority) == int(gt_priority):
            hits_by_priority[gt_priority] = hits_by_priority.get(gt_priority, 0) + 1

    per_class_recalls: List[float] = []
    for priority, total in sorted(totals_by_priority.items()):
        if total <= 0:
            continue
        per_class_recalls.append(float(hits_by_priority.get(priority, 0)) / float(total))
    if not per_class_recalls:
        return 0.0
    return sum(per_class_recalls) / len(per_class_recalls)


def _priority_lookup(gt_labels: List[Dict]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for gt_label in gt_labels:
        demand_id = str(gt_label.get("demand_id", "")).strip()
        gt_priority = _normalize_priority(gt_label.get("priority"))
        if demand_id and gt_priority is not None:
            lookup[demand_id] = int(gt_priority)
    return lookup


def _distribution_score(pred_map: Dict[str, Dict], gt_labels: List[Dict]) -> float:
    gt_counts = {priority: 0 for priority in range(1, 5)}
    pred_counts = {priority: 0 for priority in range(1, 5)}

    for gt_label in gt_labels:
        gt_priority = _normalize_priority(gt_label.get("priority"))
        if gt_priority is not None:
            gt_counts[int(gt_priority)] += 1
        pred_priority = pred_map.get(str(gt_label.get("demand_id", "")), {}).get("priority")
        pred_priority = _normalize_priority(pred_priority)
        if pred_priority is not None:
            pred_counts[int(pred_priority)] += 1

    total_gt = sum(gt_counts.values())
    total_pred = sum(pred_counts.values())
    if total_gt <= 0 or total_pred <= 0:
        return 0.0

    l1 = 0.0
    for priority in range(1, 5):
        gt_ratio = float(gt_counts[priority]) / float(total_gt)
        pred_ratio = float(pred_counts[priority]) / float(total_pred)
        l1 += abs(gt_ratio - pred_ratio)
    return max(0.0, 1.0 - (l1 / 2.0))


def _safe_f1(tp: float, fp: float, fn: float) -> float:
    precision = 0.0 if tp + fp <= 0.0 else tp / (tp + fp)
    recall = 0.0 if tp + fn <= 0.0 else tp / (tp + fn)
    if precision + recall <= 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _urgent_f1_score(pred_map: Dict[str, Dict], gt_labels: List[Dict]) -> float:
    tp = 0.0
    fp = 0.0
    fn = 0.0
    for gt_label in gt_labels:
        gt_priority = _normalize_priority(gt_label.get("priority"))
        if gt_priority is None:
            continue
        weight = _priority_weight(gt_priority)
        pred_priority = pred_map.get(gt_label["demand_id"], {}).get("priority")
        gt_urgent = int(gt_priority) <= URGENT_THRESHOLD
        pred_urgent = pred_priority is not None and int(pred_priority) <= URGENT_THRESHOLD
        if gt_urgent and pred_urgent:
            tp += weight
        elif (not gt_urgent) and pred_urgent:
            fp += weight
        elif gt_urgent and (not pred_urgent):
            fn += weight
    return _safe_f1(tp, fp, fn)


def _priority1_recall_score(pred_map: Dict[str, Dict], gt_labels: List[Dict]) -> float:
    numerator = 0.0
    denominator = 0.0
    for gt_label in gt_labels:
        gt_priority = _normalize_priority(gt_label.get("priority"))
        if gt_priority != 1:
            continue
        weight = _priority_weight(1)
        denominator += weight
        pred_priority = pred_map.get(gt_label["demand_id"], {}).get("priority")
        if pred_priority is not None and int(pred_priority) == 1:
            numerator += weight
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _weighted_pairwise_score(
    pred_map: Dict[str, Dict],
    pairwise_preferences: List[Dict],
    gt_priority_by_id: Dict[str, int],
) -> Optional[float]:
    if not pairwise_preferences:
        return None

    numerator = 0.0
    denominator = 0.0
    for pair in pairwise_preferences:
        if not isinstance(pair, dict):
            continue
        higher_id = str(pair.get("higher_priority_demand_id", ""))
        lower_id = str(pair.get("lower_priority_demand_id", ""))
        higher_priority = gt_priority_by_id.get(higher_id)
        lower_priority = gt_priority_by_id.get(lower_id)
        gap = _safe_positive_int(pair.get("priority_gap"), default=1)
        weight = float(gap)
        if higher_priority is not None and int(higher_priority) <= URGENT_THRESHOLD:
            weight += 1.0
        if lower_priority is not None and int(lower_priority) <= URGENT_THRESHOLD:
            weight += 0.5
        denominator += weight
        if _compare_pair(pred_map, higher_id, lower_id):
            numerator += weight

    if denominator <= 0.0:
        return None
    return numerator / denominator


def _compare_pair(pred_map: Dict[str, Dict], higher_id: str, lower_id: str) -> bool:
    higher = pred_map.get(higher_id)
    lower = pred_map.get(lower_id)
    if not higher or not lower:
        return False

    higher_priority = higher.get("priority")
    lower_priority = lower.get("priority")
    if higher_priority is None or lower_priority is None:
        return False
    # Ground-truth pairwise preferences are only emitted when the two items have
    # different gold priorities. Reward should therefore require a strict
    # predicted priority separation; merely ordering two equal-priority
    # predictions by rank is not enough.
    return int(higher_priority) < int(lower_priority)


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

    gt_priority_by_id = _priority_lookup(gt_labels)
    balanced_priority_score = _balanced_priority_recall_score(pred_map, gt_labels)
    exact_priority_score = _weighted_exact_priority_score(pred_map, gt_labels)
    distance_priority_score = _weighted_distance_priority_score(pred_map, gt_labels)
    distribution_score = _distribution_score(pred_map, gt_labels)
    urgent_f1_score = _urgent_f1_score(pred_map, gt_labels)
    priority1_recall_score = _priority1_recall_score(pred_map, gt_labels)

    pairwise_preferences = gt.get("pairwise_preferences", [])
    pairwise_score = _weighted_pairwise_score(pred_map, pairwise_preferences, gt_priority_by_id)

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
        topk_score = None

    components = [
        (JSON_WEIGHT, json_score),
        (BALANCED_PRIORITY_WEIGHT, balanced_priority_score),
        (EXACT_PRIORITY_WEIGHT, exact_priority_score),
        (DISTANCE_PRIORITY_WEIGHT, distance_priority_score),
        (PAIRWISE_WEIGHT, pairwise_score),
        (TOPK_WEIGHT, topk_score),
        (URGENT_F1_WEIGHT, urgent_f1_score),
        (
            PRIORITY1_RECALL_WEIGHT,
            priority1_recall_score if any(priority == 1 for priority in gt_priority_by_id.values()) else None,
        ),
        (DISTRIBUTION_WEIGHT, distribution_score),
    ]
    active_components = [(weight, score) for weight, score in components if score is not None]
    if not active_components:
        return 0.0
    total_weight = sum(weight for weight, _ in active_components)
    total_score = sum(weight * float(score) for weight, score in active_components) / total_weight
    return max(0.0, min(1.0, float(total_score)))
