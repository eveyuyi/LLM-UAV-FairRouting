"""Shared seed dataset filenames and default paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_DATA_DIR = PROJECT_ROOT / "data" / "seed"

BUILDING_DATA_FILENAME = "building_information.xlsx"
DEMAND_EVENTS_FILENAME = "daily_demand_events.csv"
STATION_DATA_FILENAME = "drone_station_locations.xlsx"

BUILDING_DATA_PATH = SEED_DATA_DIR / BUILDING_DATA_FILENAME
DEMAND_EVENTS_PATH = SEED_DATA_DIR / DEMAND_EVENTS_FILENAME
STATION_DATA_PATH = SEED_DATA_DIR / STATION_DATA_FILENAME
