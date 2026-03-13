"""Helpers for normalizing the building seed dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

BUILDING_DATA_COLUMNS = [
    "wkt_geometry",
    "feature_id",
    "building_height_m",
    "roof_area_sqm",
    "land_use_type",
    "province",
    "city",
    "ground_elevation_m",
    "top_elevation_m",
    "longitude",
    "latitude",
]

HEALTHCARE_LAND_USE = "medical_and_healthcare_land"
RESIDENTIAL_LAND_USE = "residential_land"

_COLUMN_ALIASES = {
    "wkt_geom": "wkt_geometry",
    "wkt_geometry": "wkt_geometry",
    "fid": "feature_id",
    "feature_id": "feature_id",
    "Height": "building_height_m",
    "height": "building_height_m",
    "building_height_m": "building_height_m",
    "roof_area": "roof_area_sqm",
    "roof_area_sqm": "roof_area_sqm",
    "type": "land_use_type",
    "land_use_type": "land_use_type",
    "province": "province",
    "city": "city",
    "DEM高度": "ground_elevation_m",
    "ground_elevation_m": "ground_elevation_m",
    "求和高度": "top_elevation_m",
    "top_elevation_m": "top_elevation_m",
    "经度": "longitude",
    "longitude": "longitude",
    "lon": "longitude",
    "纬度": "latitude",
    "latitude": "latitude",
    "lat": "latitude",
}

_LAND_USE_TRANSLATIONS = {
    "交通场站用地": "transportation_station_land",
    "体育与文化用地": "sports_and_cultural_land",
    "公园与绿地用地": "park_and_green_space_land",
    "医疗卫生用地": HEALTHCARE_LAND_USE,
    "商业服务用地": "commercial_service_land",
    "商务办公用地": "business_office_land",
    "居住用地": RESIDENTIAL_LAND_USE,
    "工业用地": "industrial_land",
    "教育科研用地": "education_and_research_land",
    "行政办公用地": "administrative_office_land",
}

_PROVINCE_TRANSLATIONS = {
    "广东省": "Guangdong",
    "Guangdong Province": "Guangdong",
}

_CITY_TRANSLATIONS = {
    "深圳市": "Shenzhen",
    "Shenzhen City": "Shenzhen",
}

_CJK_PATTERN = r"[\u4e00-\u9fff]"


def _read_table(file_path: str) -> pd.DataFrame:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported building data format: {file_path}")


def _contains_cjk(values: Iterable[object]) -> bool:
    return pd.Series(list(values), dtype="object").astype(str).str.contains(_CJK_PATTERN, na=False).any()


def normalize_building_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        column: _COLUMN_ALIASES[column]
        for column in df.columns
        if column in _COLUMN_ALIASES and _COLUMN_ALIASES[column] != column
    }
    normalized = df.rename(columns=rename_map).copy()

    missing_columns = [column for column in BUILDING_DATA_COLUMNS if column not in normalized.columns]
    if missing_columns:
        raise ValueError(f"Missing required building columns: {missing_columns}")

    normalized = normalized[BUILDING_DATA_COLUMNS]
    normalized["land_use_type"] = normalized["land_use_type"].replace(_LAND_USE_TRANSLATIONS)
    normalized["province"] = normalized["province"].replace(_PROVINCE_TRANSLATIONS)
    normalized["city"] = normalized["city"].replace(_CITY_TRANSLATIONS)

    for column in [
        "feature_id",
        "building_height_m",
        "roof_area_sqm",
        "ground_elevation_m",
        "top_elevation_m",
        "longitude",
        "latitude",
    ]:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")

    object_columns = normalized.select_dtypes(include="object").columns.tolist()
    remaining_cjk_columns = [
        column for column in object_columns if _contains_cjk(normalized[column].tolist())
    ]
    if remaining_cjk_columns:
        raise ValueError(f"Untranslated Chinese text remains in columns: {remaining_cjk_columns}")

    return normalized


def load_building_data(file_path: str) -> pd.DataFrame:
    return normalize_building_dataframe(_read_table(file_path))


def load_building_partitions(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_building_data(file_path)
    hospitals = df[df["land_use_type"] == HEALTHCARE_LAND_USE].copy()
    residences = df[df["land_use_type"] == RESIDENTIAL_LAND_USE].copy()
    return hospitals, residences, df


def export_normalized_building_csv(source_path: str, output_path: str) -> pd.DataFrame:
    normalized = load_building_data(source_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False, encoding="utf-8", float_format="%.15f")
    return normalized
