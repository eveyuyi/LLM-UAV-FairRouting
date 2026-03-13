"""Helpers for normalizing station seed data across all workbook sheets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from unidecode import unidecode as _unidecode
except ImportError:
    def _unidecode(value: str) -> str:
        return value


CANONICAL_STATION_COLUMNS = [
    "source_sheet",
    "latitude",
    "longitude",
    "station_name",
    "city",
    "platform",
    "notes",
    "additional_notes",
    "service_start_date",
    "service_end_date",
    "service_start_date_is_estimated",
    "service_end_date_is_estimated",
]

_SHEET_TRANSLATIONS = {
    "顺丰": "sf_express",
    "美团": "meituan",
    "sf_express": "sf_express",
    "meituan": "meituan",
}

_CITY_TRANSLATIONS = {
    "深圳": "Shenzhen",
    "北京": "Beijing",
    "上海": "Shanghai",
    "东莞": "Dongguan",
    "中山": "Zhongshan",
    "南京": "Nanjing",
    "呼和浩特": "Hohhot",
    "海口": "Haikou",
    "湛江": "Zhanjiang",
    "珠海": "Zhuhai",
    "黄石": "Huangshi",
    "Shenzhen": "Shenzhen",
    "Beijing": "Beijing",
    "Shanghai": "Shanghai",
    "Dongguan": "Dongguan",
    "Zhongshan": "Zhongshan",
    "Nanjing": "Nanjing",
    "Hohhot": "Hohhot",
    "Haikou": "Haikou",
    "Zhanjiang": "Zhanjiang",
    "Zhuhai": "Zhuhai",
    "Huangshi": "Huangshi",
}

_PLATFORM_TRANSLATIONS = {
    "顺丰丰翼": "SF Express Fengyi",
    "美团": "Meituan",
    "SF Express Fengyi": "SF Express Fengyi",
    "Meituan": "Meituan",
}

_NOTES_TRANSLATIONS = {
    "测试站": "test_site",
    "新开站": "new_site",
    "已关闭？": "possibly_closed",
    "": "",
}

_ADDITIONAL_NOTES_TRANSLATIONS = {
    "10.24时搜不到了": "not_found_when_checked_on_oct_24",
    "": "",
}

_STATION_NAME_REPLACEMENTS = [
    ("无人机空投柜", " Drone Drop Cabinet "),
    ("无人机收餐点", " Drone Meal Pickup Point "),
    ("无人机", " Drone "),
    ("网点航站", " Outlet Air Station "),
    ("航站", " Air Station "),
    ("中转场", " Transfer Hub "),
    ("集散中心", " Distribution Center "),
    ("集散", " Distribution Hub "),
    ("工业园", " Industrial Park "),
    ("工业区", " Industrial Area "),
    ("体育公园", " Sports Park "),
    ("运动公园", " Sports Park "),
    ("中心公园", " Central Park "),
    ("公园店", " Park Store "),
    ("公园", " Park "),
    ("商业中心", " Commercial Center "),
    ("街道办", " Subdistrict Office "),
    ("社区", " Community "),
    ("图书馆北馆", " Library North Branch "),
    ("图书馆中心馆", " Library Central Branch "),
    ("图书馆", " Library "),
    ("口岸前广场", " Port Front Plaza "),
    ("前广场", " Front Plaza "),
    ("口岸", " Port "),
    ("国际研究生院", " International Graduate School "),
    ("研究生院", " Graduate School "),
    ("研究院", " Research Institute "),
    ("先进院", " Institute of Advanced Technology "),
    ("校区", " Campus "),
    ("东区", " East Area "),
    ("宿舍", " Dormitories "),
    ("长城", " Great Wall "),
    ("商场", " Mall "),
    ("广场", " Plaza "),
    ("大道", " Avenue "),
    ("园区", " Campus "),
    ("篮球场", " Basketball Court "),
    ("测试场", " Test Field "),
    ("生产测试", " Production Test "),
    ("测试", " Test "),
    ("研发", " R&D "),
    ("水泥地", " Concrete Pad "),
    ("丰翼配送", "Fengyi Delivery "),
    ("丰翼", "Fengyi "),
    ("美团", "Meituan "),
    ("丰巢", "Fengchao "),
    ("北京大学", "Peking University "),
    ("清华大学", "Tsinghua University "),
    ("复旦大学", "Fudan University "),
    ("中科院", "CAS "),
]

_DATE_PATTERN = re.compile(r"^\s*(\d{4})年(\d{1,2})月(\d{1,2})日(？)?\s*$")
_CJK_PATTERN = r"[\u4e00-\u9fff]"


def _contains_cjk(values: Iterable[object]) -> bool:
    return pd.Series(list(values), dtype="object").astype(str).str.contains(_CJK_PATTERN, na=False).any()


def _translate_station_name(name: object) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip()
    for source, target in _STATION_NAME_REPLACEMENTS:
        text = text.replace(source, target)
    text = re.sub(r"(\d+)\s*期", r"Phase \1 ", text)
    text = text.replace("区-", " District - ")
    text = text.replace("区", " District ")
    text = _unidecode(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    return text.strip(" -")


def _translate_mapping(value: object, mapping: dict[str, str]) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return mapping.get(text, _unidecode(text).strip())


def _format_date(year: str, month: str, day: str) -> str:
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _parse_date_token(token: str) -> tuple[str, bool]:
    token = token.strip()
    if not token:
        return "", False

    if re.fullmatch(r"\d+(\.\d+)?", token):
        date_value = pd.to_datetime(float(token), unit="D", origin="1899-12-30")
        return date_value.strftime("%Y-%m-%d"), False

    match = _DATE_PATTERN.match(token)
    if not match:
        raise ValueError(f"Unrecognized station service date token: {token}")
    year, month, day, estimated = match.groups()
    return _format_date(year, month, day), bool(estimated)


def _parse_service_period(value: object) -> dict[str, object]:
    if pd.isna(value):
        return {
            "service_start_date": "",
            "service_end_date": "",
            "service_start_date_is_estimated": False,
            "service_end_date_is_estimated": False,
        }

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        start_date, start_estimated = _parse_date_token(str(value))
        return {
            "service_start_date": start_date,
            "service_end_date": "",
            "service_start_date_is_estimated": start_estimated,
            "service_end_date_is_estimated": False,
        }

    text = str(value).strip()
    if not text:
        return {
            "service_start_date": "",
            "service_end_date": "",
            "service_start_date_is_estimated": False,
            "service_end_date_is_estimated": False,
        }

    if "-" in text:
        start_token, end_token = text.split("-", 1)
        start_date, start_estimated = _parse_date_token(start_token) if start_token.strip() else ("", False)
        end_date, end_estimated = _parse_date_token(end_token) if end_token.strip() else ("", False)
        return {
            "service_start_date": start_date,
            "service_end_date": end_date,
            "service_start_date_is_estimated": start_estimated,
            "service_end_date_is_estimated": end_estimated,
        }

    start_date, start_estimated = _parse_date_token(text)
    return {
        "service_start_date": start_date,
        "service_end_date": "",
        "service_start_date_is_estimated": start_estimated,
        "service_end_date_is_estimated": False,
    }


def _normalize_station_sheet(sheet_name: str, df: pd.DataFrame) -> pd.DataFrame:
    station_name_series = df["航站名称"] if "航站名称" in df.columns else df["站点名"]
    platform_series = df["运营方"] if "运营方" in df.columns else df["平台"]
    service_period_series = df["服务时间"] if "服务时间" in df.columns else df["开通时间"]
    extra_notes_series = df.get("Unnamed: 7", pd.Series([""] * len(df), index=df.index))
    normalized = pd.DataFrame({
        "source_sheet": [_translate_mapping(sheet_name, _SHEET_TRANSLATIONS)] * len(df),
        "latitude": pd.to_numeric(df["纬度"], errors="raise"),
        "longitude": pd.to_numeric(df["经度"], errors="raise"),
        "station_name": station_name_series.map(_translate_station_name),
        "city": df["城市"].map(lambda value: _translate_mapping(value, _CITY_TRANSLATIONS)),
        "platform": platform_series.map(lambda value: _translate_mapping(value, _PLATFORM_TRANSLATIONS)),
        "notes": df["备注"].map(lambda value: _translate_mapping(value, _NOTES_TRANSLATIONS)),
        "additional_notes": extra_notes_series.map(
            lambda value: _translate_mapping(value, _ADDITIONAL_NOTES_TRANSLATIONS)
        ),
    })

    service_period = service_period_series.map(_parse_service_period)
    for key in [
        "service_start_date",
        "service_end_date",
        "service_start_date_is_estimated",
        "service_end_date_is_estimated",
    ]:
        normalized[key] = service_period.map(lambda item: item[key])

    return normalized[CANONICAL_STATION_COLUMNS]


def normalize_station_workbook(file_path: str) -> pd.DataFrame:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
        missing_columns = [column for column in CANONICAL_STATION_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required station columns: {missing_columns}")
        normalized = df[CANONICAL_STATION_COLUMNS].copy()
    elif suffix in {".xlsx", ".xls"}:
        workbook = pd.read_excel(file_path, sheet_name=None)
        sheets = [
            _normalize_station_sheet(sheet_name, sheet_df)
            for sheet_name, sheet_df in workbook.items()
            if not sheet_df.empty
        ]
        if not sheets:
            raise ValueError(f"No station sheets found in workbook: {file_path}")
        normalized = pd.concat(sheets, ignore_index=True)
    else:
        raise ValueError(f"Unsupported station data format: {file_path}")

    object_columns = normalized.select_dtypes(include="object").columns.tolist()
    remaining_cjk_columns = [
        column for column in object_columns if _contains_cjk(normalized[column].tolist())
    ]
    if remaining_cjk_columns:
        raise ValueError(f"Untranslated Chinese text remains in columns: {remaining_cjk_columns}")

    return normalized


def load_station_data(file_path: str) -> pd.DataFrame:
    return normalize_station_workbook(file_path)
