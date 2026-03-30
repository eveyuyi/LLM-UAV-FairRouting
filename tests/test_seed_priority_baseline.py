import json

from llm4fairrouting.baselines.cplex_with_seed_priorities import build_seed_priority_inputs


def test_build_seed_priority_inputs_groups_events_and_preserves_manifest_priority(tmp_path):
    manifest_path = tmp_path / "events_manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_id": "DEM_000_00",
                        "time_slot": 0,
                        "time_hour": 0.0,
                        "origin": {"fid": "MED_1", "coords": [113.80, 22.70], "type": "supply_station", "station_name": "MED_1", "supply_type": "medical"},
                        "destination": {"fid": "DEM_1", "node_id": "DEM_1", "coords": [113.90, 22.80], "type": "hospital"},
                        "cargo": {"type": "medicine", "type_cn": "medication", "temperature_sensitive": False},
                        "weight_kg": 3.2,
                        "deadline_minutes": 30,
                        "demand_tier": "critical",
                        "requester_role": "icu_nurse",
                        "special_handling": [],
                        "population_vulnerability": {"elderly_ratio": 0.2, "population": 1000, "elderly_involved": False, "children_involved": False, "vulnerable_community": False},
                        "receiver_ready": False,
                        "latent_priority": 2,
                        "scenario_context": "The ICU team needs a refill before the next administration round.",
                        "dialogue_styles": ["direct"],
                        "priority_factors": {"policy_version": "human_aligned_priority_v1", "scenario_context": "The ICU team needs a refill before the next administration round.", "reason_codes": ["tier_critical"], "source_fields": ["demand_tier", "scenario_context"]},
                        "must_mention_factors": [],
                        "optional_factors": [],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "event_id": "DEM_001_00",
                        "time_slot": 1,
                        "time_hour": 0.0833,
                        "origin": {"fid": "COM_1", "coords": [113.81, 22.71], "type": "supply_station", "station_name": "COM_1", "supply_type": "commercial"},
                        "destination": {"fid": "DEM_2", "node_id": "DEM_2", "coords": [113.91, 22.81], "type": "residential_area"},
                        "cargo": {"type": "otc_drug", "type_cn": "OTC medication", "temperature_sensitive": False},
                        "weight_kg": 1.8,
                        "deadline_minutes": 120,
                        "demand_tier": "consumer",
                        "requester_role": "consumer",
                        "special_handling": [],
                        "population_vulnerability": {"elderly_ratio": 0.1, "population": 800, "elderly_involved": False, "children_involved": False, "vulnerable_community": False},
                        "receiver_ready": False,
                        "latent_priority": 4,
                        "scenario_context": "The receiver requested a same-day drone delivery.",
                        "dialogue_styles": ["direct"],
                        "priority_factors": {"policy_version": "human_aligned_priority_v1", "scenario_context": "The receiver requested a same-day drone delivery.", "reason_codes": ["tier_consumer"], "source_fields": ["demand_tier", "scenario_context"]},
                        "must_mention_factors": [],
                        "optional_factors": [],
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    windows, weight_configs = build_seed_priority_inputs(
        events_path=str(manifest_path),
        base_date="2024-03-15",
        window_minutes=5,
    )

    assert len(windows) == 2
    first_window = windows[0]["time_window"]
    second_window = windows[1]["time_window"]
    assert weight_configs[first_window]["demand_configs"][0]["priority"] == 2
    assert weight_configs[second_window]["demand_configs"][0]["priority"] == 4
