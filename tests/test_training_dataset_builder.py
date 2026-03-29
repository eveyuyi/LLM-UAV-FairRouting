from __future__ import annotations

import pandas as pd

from llm4fairrouting.data.building_information import BUILDING_DATA_COLUMNS
from llm4fairrouting.data.demand_event_generation import generate_daily_demand_records
from llm4fairrouting.data.training_dataset_builder import build_priority_training_dataset


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


def test_build_priority_training_dataset_creates_three_training_layers(tmp_path):
    rows = [
        _make_building_row(1, "medical_and_healthcare_land", 113.80, 22.70),
        _make_building_row(2, "medical_and_healthcare_land", 113.81, 22.71),
        _make_building_row(3, "commercial_service_land", 113.82, 22.72),
        _make_building_row(4, "residential_land", 113.90, 22.80),
        _make_building_row(5, "residential_land", 113.91, 22.81),
    ]
    df = pd.DataFrame(rows, columns=BUILDING_DATA_COLUMNS)
    input_path = tmp_path / "building_information.csv"
    df.to_csv(input_path, index=False)

    event_records = generate_daily_demand_records(
        building_file=str(input_path),
        seed=11,
        time_window_minutes=720,
        demands_per_window_min=1,
        demands_per_window_max=1,
        medical_ratio=1.0,
        num_supply_medical=1,
        num_supply_commercial=1,
    )

    dataset = build_priority_training_dataset(
        event_records=event_records,
        event_manifest_path=str(tmp_path / "events_manifest.jsonl"),
        dialogue_output_path=str(tmp_path / "dialogues.jsonl"),
        dataset_output_path=str(tmp_path / "priority_training_dataset.json"),
        stations_path=None,
        base_date="2024-03-15",
        styles=["direct", "technical"],
        offline=True,
        window_minutes=5,
    )

    assert len(dataset["event_manifest"]) == len(event_records)
    assert len(dataset["llm2_sft"]) == len(event_records) * 2
    assert dataset["llm3_sft"]["clean_structured"]
    assert dataset["llm3_sft"]["pipeline_structured"]
    assert dataset["llm3_sft"]["hard_contrastive"]
    first_pipeline_window = dataset["llm3_sft"]["pipeline_structured"][0]
    assert "priority_labels" in first_pipeline_window
    assert "pairwise_preferences" in first_pipeline_window
    assert "critical_topk_targets" in first_pipeline_window
