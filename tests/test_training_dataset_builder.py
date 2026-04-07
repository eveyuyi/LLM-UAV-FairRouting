from __future__ import annotations

import json

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
        output_dir=str(tmp_path / "training_dataset"),
        stations_path=None,
        base_date="2024-03-15",
        styles=["direct", "technical"],
        offline=True,
        window_minutes=5,
    )

    output_dir = tmp_path / "training_dataset"
    manifest_path = output_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert dataset["counts"]["events_manifest"] == len(event_records)
    assert dataset["counts"]["llm2_sft"] == len(event_records) * 2
    assert manifest["counts"] == dataset["counts"]
    assert manifest["files"] == {
        "events_manifest": "events_manifest.jsonl",
        "dialogues": "dialogues.jsonl",
        "llm2_sft": "llm2_sft.jsonl",
        "llm3_sft_clean": "llm3_sft_clean.jsonl",
        "llm3_sft_pipeline": "llm3_sft_pipeline.jsonl",
        "llm3_grpo_hard": "llm3_grpo_hard.jsonl",
        "quality_report": "quality_report.json",
        "release_manifest": "release_manifest.json",
    }

    for relative_path in manifest["files"].values():
        assert (output_dir / relative_path).exists()

    assert dataset["artifacts"]["llm3_sft_clean"]
    assert dataset["artifacts"]["llm3_sft_pipeline"]
    assert dataset["artifacts"]["llm3_grpo_hard"]
    hard_window_labels = {
        window["time_window"] for window in dataset["artifacts"]["llm3_grpo_hard"]
    }
    assert any("counterfactual" in label for label in hard_window_labels)
    assert any("surface_contradiction" in label for label in hard_window_labels)
    first_pipeline_window = dataset["artifacts"]["llm3_sft_pipeline"][0]
    assert "priority_labels" in first_pipeline_window
    assert "pairwise_preferences" in first_pipeline_window
    assert "critical_topk_targets" in first_pipeline_window
    assert dataset["quality_report"]["counts"]["dialogues_raw"] == len(dataset["artifacts"]["dialogues"])
    assert dataset["release_manifest"]["release_status"] in {"accepted", "needs_regen", "debug_only"}
