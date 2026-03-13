import heapq

from llm4fairrouting.llm.priority_inference import adjust_weights_offline
from llm4fairrouting.routing.domain import DemandEvent, priority_service_score


def test_priority_service_score_rewards_smaller_priority_values():
    assert priority_service_score(1, 4) == 4
    assert priority_service_score(2, 4) == 3
    assert priority_service_score(3, 4) == 2
    assert priority_service_score(4, 4) == 1


def test_demand_event_heap_prefers_priority_one_when_times_match():
    events = [
        (0.0, DemandEvent(time=0.0, node_idx=0, weight=1.0, unique_id="low", priority=4)),
        (0.0, DemandEvent(time=0.0, node_idx=0, weight=1.0, unique_id="high", priority=1)),
    ]
    heapq.heapify(events)

    _, first = heapq.heappop(events)
    assert first.unique_id == "high"


def test_adjust_weights_offline_omits_alpha_beta_fields():
    result = adjust_weights_offline(
        [
            {
                "demand_id": "REQ001",
                "demand_tier": "life_support",
                "cargo": {"type": "aed"},
                "destination": {"fid": "DEST1", "coords": [113.88, 22.8]},
                "priority_evaluation_signals": {
                    "patient_condition": "心脏骤停",
                    "population_vulnerability": {
                        "elderly_ratio": 0.6,
                        "elderly_involved": True,
                        "vulnerable_community": True,
                        "children_involved": False,
                    },
                },
                "context_signals": [],
            }
        ]
    )

    config = result["demand_configs"][0]
    assert "alpha" not in config
    assert "beta" not in config
    assert config["priority"] == 1
