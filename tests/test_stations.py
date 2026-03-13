"""Tests for normalized station seed data."""

from pathlib import Path

from llm4fairrouting.data.seed_paths import STATION_DATA_PATH, STATION_DATA_SOURCE_PATH
from llm4fairrouting.data.stations import CANONICAL_STATION_COLUMNS, load_station_data
from llm4fairrouting.llm.dialogue_generation import load_stations


def test_normalized_station_csv_exists():
    assert Path(STATION_DATA_PATH).exists()


def test_normalized_station_data_uses_english_schema():
    df = load_station_data(str(STATION_DATA_PATH))

    assert list(df.columns) == CANONICAL_STATION_COLUMNS
    assert set(df["source_sheet"]) == {"sf_express", "meituan"}

    for column in [
        "source_sheet",
        "station_name",
        "city",
        "platform",
        "notes",
        "additional_notes",
    ]:
        assert not df[column].astype(str).str.contains(r"[\u4e00-\u9fff]", na=False).any()


def test_normalized_station_csv_matches_source_rows_and_coordinates():
    source_df = load_station_data(str(STATION_DATA_SOURCE_PATH))
    csv_df = load_station_data(str(STATION_DATA_PATH))

    assert len(csv_df) == len(source_df) == 89
    assert list(zip(csv_df["source_sheet"], csv_df["latitude"], csv_df["longitude"])) == list(
        zip(source_df["source_sheet"], source_df["latitude"], source_df["longitude"])
    )


def test_load_stations_uses_both_station_sources_for_shenzhen():
    stations = load_stations(str(STATION_DATA_PATH))

    assert len(stations) == 72
    assert any("Meituan" in station["name"] for station in stations)
    assert any("Fengyi" in station["name"] for station in stations)
