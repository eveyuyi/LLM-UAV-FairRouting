"""Custom reward for llm_selection GRPO training."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Set

JSON_VALID_WEIGHT = 0.10
CANDIDATE_VALID_WEIGHT = 0.10
GROUP_MATCH_WEIGHT = 0.20
SOLUTION_MATCH_WEIGHT = 0.45
REASON_F1_WEIGHT = 0.10
SCENE_MATCH_WEIGHT = 0.05


def _parse_json(value):
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def _str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_set(values: Iterable) -> Set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        single = _str(values)
        return {single} if single else set()
    if not isinstance(values, Iterable):
        single = _str(values)
        return {single} if single else set()
    return {_str(v) for v in values if _str(v)}


def _extract_scene_type(value) -> str:
    """Robustly parse scene_type from training_labels-like payload."""
    if isinstance(value, dict):
        return _str(value.get("scene_type"))
    if isinstance(value, list):
        # Some model outputs return training_labels as a list of objects/strings.
        for item in value:
            if isinstance(item, dict):
                scene_type = _str(item.get("scene_type"))
                if scene_type:
                    return scene_type
            else:
                item_text = _str(item)
                if item_text:
                    return item_text
    if isinstance(value, str):
        parsed = _parse_json(value)
        if parsed is not None and parsed is not value:
            return _extract_scene_type(parsed)
    return ""


def _f1(pred: Set[str], truth: Set[str]) -> Optional[float]:
    if not truth:
        return None
    if not pred:
        return 0.0
    inter = len(pred & truth)
    if inter <= 0:
        return 0.0
    precision = inter / len(pred)
    recall = inter / len(truth)
    denom = precision + recall
    if denom <= 0.0:
        return 0.0
    return 2.0 * precision * recall / denom


def _extract_prediction(solution_str) -> Optional[Dict]:
    payload = _parse_json(solution_str)
    if isinstance(payload, dict):
        return payload
    return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if str(data_source or "") != "llm_selection_pareto_window":
        return 0.0

    gt = _parse_json(ground_truth)
    if isinstance(gt, dict) and "ground_truth" in gt:
        gt = _parse_json(gt.get("ground_truth"))
    if not isinstance(gt, dict):
        return 0.0

    pred = _extract_prediction(solution_str)
    json_valid = 1.0 if isinstance(pred, dict) else 0.0
    if pred is None:
        return JSON_VALID_WEIGHT * json_valid

    pred_solution_id = _str(pred.get("selected_solution_id"))
    pred_group_id = _str(pred.get("selected_group_id"))
    pred_reason_codes = _safe_set(pred.get("primary_reason_codes") or [])
    pred_scene_type = _extract_scene_type(pred.get("training_labels"))

    gt_solution_id = _str(gt.get("selected_solution_id"))
    gt_group_id = _str(gt.get("selected_group_id"))
    gt_reason_codes = _safe_set(gt.get("primary_reason_codes") or [])
    gt_scene_type = _extract_scene_type(gt.get("training_labels"))
    candidate_solution_ids = _safe_set(gt.get("candidate_solution_ids") or [])

    candidate_valid = 1.0
    if candidate_solution_ids:
        candidate_valid = 1.0 if pred_solution_id in candidate_solution_ids else 0.0

    group_match = 1.0 if pred_group_id and pred_group_id == gt_group_id else 0.0
    solution_match = 1.0 if pred_solution_id and pred_solution_id == gt_solution_id else 0.0
    reason_f1 = _f1(pred_reason_codes, gt_reason_codes)
    scene_match = None
    if gt_scene_type:
        scene_match = 1.0 if pred_scene_type == gt_scene_type else 0.0

    components: List[tuple[float, Optional[float]]] = [
        (JSON_VALID_WEIGHT, json_valid),
        (CANDIDATE_VALID_WEIGHT, candidate_valid),
        (GROUP_MATCH_WEIGHT, group_match),
        (SOLUTION_MATCH_WEIGHT, solution_match),
        (REASON_F1_WEIGHT, reason_f1),
        (SCENE_MATCH_WEIGHT, scene_match),
    ]
    active = [(weight, score) for weight, score in components if score is not None]
    if not active:
        return 0.0

    total_weight = sum(weight for weight, _ in active)
    final_score = sum(weight * float(score) for weight, score in active) / total_weight
    return max(0.0, min(1.0, float(final_score)))
