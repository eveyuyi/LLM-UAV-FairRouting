from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from evals.eval_priority_alignment import evaluate_priority_alignment


def _make_case_dir(prefix: str) -> Path:
    path = Path.cwd() / ".test_artifacts" / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_evaluate_priority_alignment_reports_ranking_and_urgent_metrics():
    base = _make_case_dir("priority_alignment_eval")
    weights_dir = base / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    with open(weights_dir / "w1.json", "w", encoding="utf-8") as handle:
        json.dump({
            "time_window": "W1",
            "demand_configs": [
                {"demand_id": "REQ001", "priority": 1, "window_rank": 1, "reasoning": "life support"},
                {"demand_id": "REQ002", "priority": 4, "window_rank": 2, "reasoning": "routine"},
                {"demand_id": "REQ003", "priority": 2, "window_rank": 3, "reasoning": "urgent"},
            ]
        }, handle, ensure_ascii=False, indent=2)

    demands_path = base / "demands.json"
    with open(demands_path, "w", encoding="utf-8") as handle:
        json.dump([
            {
                "time_window": "W1",
                "demands": [
                    {"demand_id": "REQ001", "source_dialogue_id": "D1", "source_event_id": "E1"},
                    {"demand_id": "REQ002", "source_dialogue_id": "D2", "source_event_id": "E2"},
                    {"demand_id": "REQ003", "source_dialogue_id": "D3", "source_event_id": "E3"},
                ],
            }
        ], handle, ensure_ascii=False, indent=2)

    dialogues_path = base / "dialogues.jsonl"
    dialogues_path.write_text("\n".join([
        json.dumps({"dialogue_id": "D1", "metadata": {"event_id": "E1"}}, ensure_ascii=False),
        json.dumps({"dialogue_id": "D2", "metadata": {"event_id": "E2"}}, ensure_ascii=False),
        json.dumps({"dialogue_id": "D3", "metadata": {"event_id": "E3"}}, ensure_ascii=False),
    ]), encoding="utf-8")

    ground_truth = base / "ground_truth.jsonl"
    ground_truth.write_text(
        "\n".join(
            [
                json.dumps({
                    "event_id": "E1",
                    "time_slot": 0,
                    "origin": {"fid": "SUP1", "coords": [0.0, 0.0], "type": "supply_station", "station_name": "SUP1", "supply_type": "medical"},
                    "destination": {"fid": "DEM1", "node_id": "DEM1", "coords": [0.0, 0.0], "type": "hospital"},
                    "cargo": {"type": "aed", "type_cn": "AED", "temperature_sensitive": False},
                    "weight_kg": 5.0,
                    "deadline_minutes": 15,
                    "demand_tier": "life_support",
                    "requester_role": "emergency_doctor",
                    "special_handling": [],
                    "population_vulnerability": {},
                    "receiver_ready": False,
                    "latent_priority": 1,
                    "scenario_context": "CPR in progress",
                    "dialogue_styles": ["direct"],
                    "priority_factors": {"policy_version": "human_aligned_priority_v1", "scenario_context": "CPR in progress", "reason_codes": ["tier_life_support"], "source_fields": ["demand_tier"]},
                    "must_mention_factors": [],
                    "optional_factors": [],
                }, ensure_ascii=False),
                json.dumps({
                    "event_id": "E2",
                    "time_slot": 1,
                    "origin": {"fid": "SUP2", "coords": [0.0, 0.0], "type": "supply_station", "station_name": "SUP2", "supply_type": "commercial"},
                    "destination": {"fid": "DEM2", "node_id": "DEM2", "coords": [0.0, 0.0], "type": "residential_area"},
                    "cargo": {"type": "otc_drug", "type_cn": "OTC", "temperature_sensitive": False},
                    "weight_kg": 8.0,
                    "deadline_minutes": 120,
                    "demand_tier": "consumer",
                    "requester_role": "consumer",
                    "special_handling": [],
                    "population_vulnerability": {},
                    "receiver_ready": False,
                    "latent_priority": 4,
                    "scenario_context": "Same-day delivery",
                    "dialogue_styles": ["direct"],
                    "priority_factors": {"policy_version": "human_aligned_priority_v1", "scenario_context": "Same-day delivery", "reason_codes": ["tier_consumer"], "source_fields": ["demand_tier"]},
                    "must_mention_factors": [],
                    "optional_factors": [],
                }, ensure_ascii=False),
                json.dumps({
                    "event_id": "E3",
                    "time_slot": 2,
                    "origin": {"fid": "SUP3", "coords": [0.0, 0.0], "type": "supply_station", "station_name": "SUP3", "supply_type": "medical"},
                    "destination": {"fid": "DEM3", "node_id": "DEM3", "coords": [0.0, 0.0], "type": "clinic"},
                    "cargo": {"type": "icu_drug", "type_cn": "ICU", "temperature_sensitive": False},
                    "weight_kg": 3.0,
                    "deadline_minutes": 30,
                    "demand_tier": "critical",
                    "requester_role": "icu_nurse",
                    "special_handling": [],
                    "population_vulnerability": {},
                    "receiver_ready": False,
                    "latent_priority": 2,
                    "scenario_context": "ICU refill",
                    "dialogue_styles": ["direct"],
                    "priority_factors": {"policy_version": "human_aligned_priority_v1", "scenario_context": "ICU refill", "reason_codes": ["tier_critical"], "source_fields": ["demand_tier"]},
                    "must_mention_factors": [],
                    "optional_factors": [],
                }, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    payload = evaluate_priority_alignment(
        weights_path=str(weights_dir),
        demands_path=str(demands_path),
        dialogues_path=str(dialogues_path),
        ground_truth_path=str(ground_truth),
        urgent_threshold=2,
    )

    assert payload["accuracy"] == 1.0
    assert payload["spearman"] == 1.0
    assert payload["priority_1_metrics"]["recall"] == 1.0
    assert payload["urgent_metrics"]["recall"] == 1.0
    assert payload["top_k_hit_rate"]["hit_rate"] == 1.0


def test_evaluate_priority_alignment_prefers_extraction_observable_priority_when_present():
    base = _make_case_dir("priority_alignment_eval_labels")
    weights_dir = base / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    with open(weights_dir / "w1.json", "w", encoding="utf-8") as handle:
        json.dump({
            "time_window": "W1",
            "demand_configs": [
                {"demand_id": "REQ001", "priority": 2, "window_rank": 1, "reasoning": "matches extraction label"},
            ],
        }, handle, ensure_ascii=False, indent=2)

    demands_path = base / "demands.json"
    with open(demands_path, "w", encoding="utf-8") as handle:
        json.dump([
            {
                "time_window": "W1",
                "demands": [
                    {
                        "demand_id": "REQ001",
                        "source_dialogue_id": "D1",
                        "source_event_id": "E1",
                        "labels": {"extraction_observable_priority": 2},
                    },
                ],
            }
        ], handle, ensure_ascii=False, indent=2)

    dialogues_path = base / "dialogues.jsonl"
    dialogues_path.write_text(
        json.dumps({"dialogue_id": "D1", "metadata": {"event_id": "E1"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    ground_truth = base / "ground_truth.jsonl"
    ground_truth.write_text(
        json.dumps({
            "event_id": "E1",
            "time_slot": 0,
            "origin": {"fid": "SUP1", "coords": [0.0, 0.0], "type": "supply_station", "station_name": "SUP1", "supply_type": "commercial"},
            "destination": {"fid": "DEM1", "node_id": "DEM1", "coords": [0.0, 0.0], "type": "residential_area"},
            "cargo": {"type": "food", "type_cn": "food", "temperature_sensitive": False},
            "weight_kg": 1.0,
            "deadline_minutes": 120,
            "demand_tier": "consumer",
            "requester_role": "consumer",
            "special_handling": [],
            "population_vulnerability": {},
            "receiver_ready": False,
            "latent_priority": 4,
            "scenario_context": "Same-day delivery",
            "dialogue_styles": ["direct"],
            "priority_factors": {"policy_version": "human_aligned_priority_v1", "scenario_context": "Same-day delivery", "reason_codes": ["tier_consumer"], "source_fields": ["demand_tier"]},
            "must_mention_factors": [],
            "optional_factors": [],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    payload = evaluate_priority_alignment(
        weights_path=str(weights_dir),
        demands_path=str(demands_path),
        dialogues_path=str(dialogues_path),
        ground_truth_path=str(ground_truth),
        urgent_threshold=2,
    )

    assert payload["accuracy"] == 1.0
    assert payload["per_item"][0]["true_priority"] == 2


def test_evaluate_priority_alignment_can_use_fixed_truth_demands():
    base = _make_case_dir("priority_alignment_eval_fixed_truth")
    weights_dir = base / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    with open(weights_dir / "w1.json", "w", encoding="utf-8") as handle:
        json.dump({
            "time_window": "W1",
            "demand_configs": [
                {"demand_id": "REQ001", "priority": 2, "window_rank": 1, "reasoning": "post model"},
            ],
        }, handle, ensure_ascii=False, indent=2)

    run_demands_path = base / "run_demands.json"
    with open(run_demands_path, "w", encoding="utf-8") as handle:
        json.dump([
            {
                "time_window": "W1",
                "demands": [
                    {
                        "demand_id": "REQ001",
                        "source_dialogue_id": "D1",
                        "source_event_id": "E1",
                        "labels": {"extraction_observable_priority": 4},
                    },
                ],
            }
        ], handle, ensure_ascii=False, indent=2)

    truth_demands_path = base / "truth_demands.json"
    with open(truth_demands_path, "w", encoding="utf-8") as handle:
        json.dump([
            {
                "time_window": "W1",
                "demands": [
                    {
                        "demand_id": "REQ001",
                        "source_dialogue_id": "D1",
                        "source_event_id": "E1",
                        "labels": {"extraction_observable_priority": 2},
                    },
                ],
            }
        ], handle, ensure_ascii=False, indent=2)

    dialogues_path = base / "dialogues.jsonl"
    dialogues_path.write_text(
        json.dumps({"dialogue_id": "D1", "metadata": {"event_id": "E1"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    ground_truth = base / "ground_truth.jsonl"
    ground_truth.write_text(
        json.dumps({
            "event_id": "E1",
            "time_slot": 0,
            "origin": {"fid": "SUP1", "coords": [0.0, 0.0], "type": "supply_station", "station_name": "SUP1", "supply_type": "commercial"},
            "destination": {"fid": "DEM1", "node_id": "DEM1", "coords": [0.0, 0.0], "type": "residential_area"},
            "cargo": {"type": "food", "type_cn": "food", "temperature_sensitive": False},
            "weight_kg": 1.0,
            "deadline_minutes": 120,
            "demand_tier": "consumer",
            "requester_role": "consumer",
            "special_handling": [],
            "population_vulnerability": {},
            "receiver_ready": False,
            "latent_priority": 4,
            "scenario_context": "Same-day delivery",
            "dialogue_styles": ["direct"],
            "priority_factors": {"policy_version": "human_aligned_priority_v1", "scenario_context": "Same-day delivery", "reason_codes": ["tier_consumer"], "source_fields": ["demand_tier"]},
            "must_mention_factors": [],
            "optional_factors": [],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    payload = evaluate_priority_alignment(
        weights_path=str(weights_dir),
        demands_path=str(run_demands_path),
        dialogues_path=str(dialogues_path),
        ground_truth_path=str(ground_truth),
        urgent_threshold=2,
        truth_source="fixed_demands",
        truth_demands_path=str(truth_demands_path),
    )

    assert payload["accuracy"] == 1.0
    assert payload["truth_source"] == "fixed_demands"
    assert payload["per_item"][0]["true_priority"] == 2
