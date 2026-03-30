from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from evals.eval_priority_operational_impact import summarize_operational_impact


def _make_case_dir(prefix: str) -> Path:
    path = Path.cwd() / ".test_artifacts" / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_run(run_dir: Path, per_demand_results):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "workflow_results.json", "w", encoding="utf-8") as handle:
        json.dump([{"time_window": "W1", "per_demand_results": per_demand_results}], handle, ensure_ascii=False, indent=2)


def test_operational_impact_reports_urgent_efficiency_gains():
    base = _make_case_dir("priority_operational_impact")
    dialogues = base / "dialogues.jsonl"
    dialogues.write_text("\n".join([
        json.dumps({"dialogue_id": "D1", "metadata": {"event_id": "E1", "delivery_deadline_minutes": 20}}, ensure_ascii=False),
        json.dumps({"dialogue_id": "D2", "metadata": {"event_id": "E2", "delivery_deadline_minutes": 60}}, ensure_ascii=False),
    ]), encoding="utf-8")

    gt = base / "ground_truth.jsonl"
    gt.write_text(
        "\n".join(
            [
                json.dumps({
                    "event_id": "E1",
                    "time_slot": 0,
                    "origin": {"fid": "SUP1", "coords": [0.0, 0.0], "type": "supply_station", "station_name": "SUP1", "supply_type": "medical"},
                    "destination": {"fid": "DEM1", "node_id": "DEM1", "coords": [0.0, 0.0], "type": "hospital"},
                    "cargo": {"type": "aed", "type_cn": "AED", "temperature_sensitive": False},
                    "weight_kg": 1.0,
                    "deadline_minutes": 20,
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
                    "cargo": {"type": "food", "type_cn": "food", "temperature_sensitive": False},
                    "weight_kg": 1.0,
                    "deadline_minutes": 60,
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
            ]
        ),
        encoding="utf-8",
    )

    baseline_run = base / "baseline"
    llm_run = base / "llm"
    _write_run(baseline_run, [
        {"demand_id": "REQ1", "source_dialogue_id": "D1", "source_event_id": "E1", "is_served": True, "delivery_latency_min": 30.0, "deadline_minutes": 20},
        {"demand_id": "REQ2", "source_dialogue_id": "D2", "source_event_id": "E2", "is_served": True, "delivery_latency_min": 40.0, "deadline_minutes": 60},
    ])
    _write_run(llm_run, [
        {"demand_id": "REQ1", "source_dialogue_id": "D1", "source_event_id": "E1", "is_served": True, "delivery_latency_min": 10.0, "deadline_minutes": 20},
        {"demand_id": "REQ2", "source_dialogue_id": "D2", "source_event_id": "E2", "is_served": True, "delivery_latency_min": 45.0, "deadline_minutes": 60},
    ])

    payload = summarize_operational_impact(
        run_dirs={"baseline": baseline_run, "llm": llm_run},
        dialogues_path=str(dialogues),
        ground_truth_path=str(gt),
        urgent_threshold=2,
        reference_method="baseline",
    )

    assert payload["methods"]["baseline"]["priority_1"]["on_time_rate"] == 0.0
    assert payload["methods"]["llm"]["priority_1"]["on_time_rate"] == 1.0
    assert payload["comparisons"]["llm_vs_baseline"]["priority_1_on_time_rate_gain"] == 1.0
    assert payload["comparisons"]["llm_vs_baseline"]["urgent_latency_improvement"] > 0
