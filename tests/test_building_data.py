"""Tests for normalized building seed data."""

from pathlib import Path

from drone_pipeline.seed_data import BUILDING_DATA_PATH, BUILDING_DATA_SOURCE_PATH
from drone_pipeline.utils.building_data import (
    CANONICAL_BUILDING_COLUMNS,
    HEALTHCARE_LAND_USE,
    RESIDENTIAL_LAND_USE,
    load_building_data,
)


def test_normalized_building_csv_exists():
    assert Path(BUILDING_DATA_PATH).exists()


def test_normalized_building_data_uses_english_schema():
    df = load_building_data(str(BUILDING_DATA_PATH))

    assert list(df.columns) == CANONICAL_BUILDING_COLUMNS
    assert HEALTHCARE_LAND_USE in set(df["land_use_type"])
    assert RESIDENTIAL_LAND_USE in set(df["land_use_type"])

    for column in ["land_use_type", "province", "city"]:
        assert not df[column].astype(str).str.contains(r"[\u4e00-\u9fff]", na=False).any()


def test_normalized_csv_matches_source_row_count_and_ids():
    source_df = load_building_data(str(BUILDING_DATA_SOURCE_PATH))
    csv_df = load_building_data(str(BUILDING_DATA_PATH))

    assert len(csv_df) == len(source_df)
    assert csv_df["feature_id"].tolist() == source_df["feature_id"].tolist()
