from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_reward_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "verl_llm3_reward.py"
    spec = importlib.util.spec_from_file_location("verl_llm3_reward", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_compute_score_is_one_for_perfect_prediction():
    reward = _load_reward_module()
    ground_truth = {
        "priority_labels": [
            {"demand_id": "A", "priority": 1, "window_rank": 1},
            {"demand_id": "B", "priority": 3, "window_rank": 2},
        ],
        "pairwise_preferences": [
            {"higher_priority_demand_id": "A", "lower_priority_demand_id": "B", "priority_gap": 2},
        ],
        "critical_topk_targets": ["A"],
    }
    solution = {
        "priority_labels": [
            {"demand_id": "A", "priority": 1, "window_rank": 1, "reasoning": "highest urgency"},
            {"demand_id": "B", "priority": 3, "window_rank": 2, "reasoning": "lower urgency"},
        ]
    }
    score = reward.compute_score("llm3_priority_window", solution, ground_truth)
    assert score == 1.0


def test_compute_score_penalizes_invalid_json():
    reward = _load_reward_module()
    ground_truth = {
        "priority_labels": [{"demand_id": "A", "priority": 1, "window_rank": 1}],
        "pairwise_preferences": [],
        "critical_topk_targets": ["A"],
    }
    score = reward.compute_score("llm3_priority_window", "not-json", ground_truth)
    assert score < 0.5
