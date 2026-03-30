from __future__ import annotations

import json
import random

import pandas as pd

from llm4fairrouting.baselines.cplex_with_priority_noise import generate_demand_events as baseline_generate_demand_events
from llm4fairrouting.data.building_information import BUILDING_DATA_COLUMNS
from llm4fairrouting.data.demand_event_generation import (
    generate_daily_demand_dataset,
    generate_daily_demand_records,
)
from llm4fairrouting.routing.domain import Point


def _make_building_row(feature_id: int, land_use_type: str, lon: float, lat: float) -> dict[str, object]:
    return {
        "wkt_geometry": "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
        "feature_id": feature_id,
        "building_height_m": 20.0,
        "roof_area_sqm": 100.0,
        "land_use_type": land_use_type,
        "province": "Guangdong",
        "city": "Shenzhen",
        "ground_elevation_m": 5.0,
        "top_elevation_m": 25.0,
        "longitude": lon,
        "latitude": lat,
    }


def test_baseline_generate_demand_events_preserves_original_shape():
    demand_points = [
        Point(id="D1", lon=113.0, lat=22.0, alt=0.0, type="demand"),
        Point(id="D2", lon=113.1, lat=22.1, alt=0.0, type="demand"),
    ]
    supply_points = [
        Point(id="S1", lon=113.2, lat=22.2, alt=0.0, type="supply"),
        Point(id="S2", lon=113.3, lat=22.3, alt=0.0, type="supply"),
    ]

    random.seed(7)
    events = baseline_generate_demand_events(demand_points, supply_points, num_events=3, sim_duration=2.0)

    assert len(events) == 3
    assert {event.priority for event in events} == {1, 2, 3}
    assert all(0.1 <= event.time <= 1.5 for event in events)
    assert all(event.unique_id.startswith("DEM_D") for event in events)
    assert all(event.required_supply_idx in {0, 1} for event in events)


def test_generate_daily_demand_dataset_writes_manifest(tmp_path):
    rows = [
        _make_building_row(1, "medical_and_healthcare_land", 113.80, 22.70),
        _make_building_row(2, "medical_and_healthcare_land", 113.81, 22.71),
        _make_building_row(3, "commercial_service_land", 113.82, 22.72),
        _make_building_row(4, "commercial_service_land", 113.83, 22.73),
        _make_building_row(5, "residential_land", 113.90, 22.80),
        _make_building_row(6, "residential_land", 113.91, 22.81),
        _make_building_row(7, "residential_land", 113.92, 22.82),
    ]
    df = pd.DataFrame(rows, columns=BUILDING_DATA_COLUMNS)
    input_path = tmp_path / "building_information.csv"
    manifest_path = tmp_path / "events_manifest.jsonl"
    df.to_csv(input_path, index=False)

    records = generate_daily_demand_dataset(
        building_file=str(input_path),
        manifest_file=str(manifest_path),
        seed=123,
        time_window_minutes=720,
        demands_per_window_min=1,
        demands_per_window_max=1,
        medical_ratio=1.0,
        num_supply_medical=1,
        num_supply_commercial=1,
    )

    written = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert manifest_path.exists()
    assert len(records) == 2
    assert len(written) == 2
    assert [record["time_hour"] for record in written] == [0.0, 12.0]
    assert set(record["latent_priority"] for record in written).issubset({1, 2, 3})
    assert all(record["origin"]["supply_type"] == "medical" for record in written)
    assert [record["event_id"] for record in written] == ["DEM_000_00", "DEM_001_00"]


def test_generate_daily_demand_records_include_control_and_policy_fields(tmp_path):
    rows = [
        _make_building_row(1, "medical_and_healthcare_land", 113.80, 22.70),
        _make_building_row(2, "commercial_service_land", 113.82, 22.72),
        _make_building_row(3, "residential_land", 113.90, 22.80),
        _make_building_row(4, "residential_land", 113.91, 22.81),
    ]
    df = pd.DataFrame(rows, columns=BUILDING_DATA_COLUMNS)
    input_path = tmp_path / "building_information.csv"
    df.to_csv(input_path, index=False)

    records = generate_daily_demand_records(
        building_file=str(input_path),
        seed=7,
        time_window_minutes=720,
        demands_per_window_min=1,
        demands_per_window_max=1,
        medical_ratio=1.0,
        num_supply_medical=1,
        num_supply_commercial=1,
    )

    assert len(records) == 2
    record = records[0]
    for field in [
        "origin",
        "destination",
        "cargo",
        "weight_kg",
        "deadline_minutes",
        "demand_tier",
        "requester_role",
        "special_handling",
        "population_vulnerability",
        "receiver_ready",
        "latent_priority",
        "scenario_context",
        "priority_factors",
        "must_mention_factors",
        "optional_factors",
    ]:
        assert field in record
    assert "gold_structured_demand" not in record
    assert "solver_useful_priority" not in record
    assert "priority" not in record
    assert record["priority_factors"]["policy_version"] == "human_aligned_priority_v1"
    assert record["priority_factors"]["reason_codes"]
    assert record["must_mention_factors"][0]["source_field"] == "scenario_context"
