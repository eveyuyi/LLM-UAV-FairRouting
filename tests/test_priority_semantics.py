import heapq

from llm4fairrouting.llm import priority_inference as priority_inference_module
from llm4fairrouting.llm.priority_inference import adjust_weights_offline, _normalize_weight_config
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


def test_adjust_weights_offline_upgrades_urgent_consumer_otc_case():
    result = adjust_weights_offline(
        [
            {
                "demand_id": "REQ010",
                "demand_tier": "consumer",
                "destination": {"fid": "DEST2", "coords": [113.88, 22.8], "type": "residential_area"},
                "cargo": {"type": "otc_drug"},
                "time_constraint": {"type": "soft", "deadline_minutes": 90},
                "priority_evaluation_signals": {
                    "patient_condition": "Child with persistent fever at home",
                    "time_sensitivity": "Timely delivery needed within the service window",
                    "requester_role": "consumer",
                    "population_vulnerability": {
                        "elderly_ratio": 0.0,
                        "elderly_involved": False,
                        "vulnerable_community": False,
                        "children_involved": True,
                    },
                    "special_handling": [],
                },
                "context_signals": ["Child is waiting for fever relief medication at home"],
            }
        ]
    )

    config = result["demand_configs"][0]
    assert config["priority"] == 3


def test_adjust_weights_offline_ranks_short_deadline_critical_case_ahead_of_regular():
    result = adjust_weights_offline(
        [
            {
                "demand_id": "REQ100",
                "demand_tier": "regular",
                "destination": {"fid": "DEST3", "coords": [113.88, 22.8], "type": "clinic"},
                "cargo": {"type": "vaccine"},
                "time_constraint": {"type": "soft", "deadline_minutes": 120},
                "priority_evaluation_signals": {
                    "patient_condition": "Routine vaccine restock",
                    "requester_role": "community_health_worker",
                    "population_vulnerability": {
                        "elderly_ratio": 0.0,
                        "elderly_involved": False,
                        "vulnerable_community": False,
                        "children_involved": False,
                    },
                    "special_handling": ["cold_chain"],
                },
                "context_signals": [],
            },
            {
                "demand_id": "REQ101",
                "demand_tier": "critical",
                "destination": {"fid": "DEST4", "coords": [113.89, 22.81], "type": "hospital"},
                "cargo": {"type": "ventilator"},
                "time_constraint": {"type": "hard", "deadline_minutes": 20},
                "priority_evaluation_signals": {
                    "patient_condition": "ICU patient needs a backup ventilator immediately",
                    "time_sensitivity": "Urgent same-window delivery required",
                    "requester_role": "icu_nurse",
                    "population_vulnerability": {
                        "elderly_ratio": 0.0,
                        "elderly_involved": False,
                        "vulnerable_community": False,
                        "children_involved": False,
                    },
                    "special_handling": ["shock_protection"],
                },
                "context_signals": ["Receiving staff and biomedical support are on standby"],
            },
        ]
    )

    configs = result["demand_configs"]
    assert configs[0]["demand_id"] == "REQ101"
    assert configs[0]["priority"] <= configs[1]["priority"]


def test_normalize_weight_config_accepts_priority_labels_schema():
    result = _normalize_weight_config(
        {
            "priority_labels": [
                {"demand_id": "DEM_001", "priority": 1, "window_rank": 1, "reasoning": "highest need"},
                {"demand_id": "DEM_002", "priority": 3, "window_rank": 2, "reasoning": "lower need"},
            ]
        }
    )

    assert [item["demand_id"] for item in result["demand_configs"]] == ["DEM_001", "DEM_002"]
    assert [item["priority"] for item in result["demand_configs"]] == [1, 3]


def test_adjust_weights_offline_tolerates_null_optional_list_fields():
    result = adjust_weights_offline(
        [
            {
                "demand_id": "REQ_NULLS",
                "demand_tier": "regular",
                "destination": {"fid": "DEST5", "coords": [113.88, 22.8], "type": "school"},
                "cargo": {"type": "vaccine"},
                "priority_evaluation_signals": {
                    "patient_condition": "Routine campus clinic refill",
                    "requester_role": "community_health_worker",
                    "special_handling": None,
                    "population_vulnerability": None,
                },
                "context_signals": None,
            }
        ]
    )

    assert result["demand_configs"][0]["demand_id"] == "REQ_NULLS"
    assert isinstance(result["supplementary_constraints"], list)


def test_chunked_priority_ranking_recursively_splits_oversized_batches(monkeypatch):
    def fake_call_llm_rank(*, demands, client, model, city_context, temperature):
        if len(demands) > 2:
            raise RuntimeError("maximum context length is 16384 tokens")
        return {
            "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
            "demand_configs": [
                {
                    "demand_id": demand["demand_id"],
                    "priority": 2,
                    "window_rank": idx,
                    "reasoning": f"ranked {demand['demand_id']}",
                }
                for idx, demand in enumerate(demands, start=1)
            ],
            "supplementary_constraints": [],
        }

    monkeypatch.setattr(priority_inference_module, "_call_llm_rank", fake_call_llm_rank)
    monkeypatch.setattr(priority_inference_module, "env_int", lambda *_args, **_kwargs: 3)

    result = priority_inference_module._call_llm_rank_with_chunk_fallback(
        demands=[{"demand_id": f"REQ{i}"} for i in range(5)],
        client=None,
        model="demo-model",
        city_context=None,
        temperature=0.0,
    )

    assert len(result["demand_configs"]) == 5
    assert {item["demand_id"] for item in result["demand_configs"]} == {
        "REQ0", "REQ1", "REQ2", "REQ3", "REQ4"
    }
