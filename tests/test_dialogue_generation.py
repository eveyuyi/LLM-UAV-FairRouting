"""
Module 1 单元测试 — llm/dialogue_generation.py

运行方式（在项目根目录）:
    /Users/eveyu/opt/anaconda3/envs/gpt_academic/bin/python -m pytest tests/test_dialogue_generation.py -v
"""

import json
import math
import random
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pytest

from llm4fairrouting.llm.dialogue_generation import (
    _event_to_dialogue,
    _find_nearest_station,
    _generate_rule_conversation,
    _haversine,
    _infer_poi,
    _map_priority_to_tier,
    _pick_tier_template,
    generate_dialogues_offline,
    load_demand_events,
    load_stations,
    save_dialogues,
)
from llm4fairrouting.data.seed_paths import DEMAND_EVENTS_PATH, STATION_DATA_PATH

# ============================================================================
# 测试固件
# ============================================================================

SAMPLE_STATIONS = [
    {"station_id": "ST001", "name": "燕罗航站", "lon": 113.879911, "lat": 22.799354},
    {"station_id": "ST002", "name": "铁甲航站", "lon": 113.885260, "lat": 22.786042},
    {"station_id": "ST003", "name": "福城航站", "lon": 114.024485, "lat": 22.741762},
]

SAMPLE_EVENT = {
    "time_slot": 12,
    "time_hour": 1.0,
    "event_id": "DEM_012_0050",
    "demand_node_id": "D410",
    "demand_node_index": 2200,
    "demand_lon": 113.868851,
    "demand_lat": 22.665646,
    "material_type": "vaccine",
    "quantity_kg": 2.1,
    "priority": 2,
    "supply_options": '["S1","S2"]',
    "demand_type": "居住用地",
}

SAMPLE_EVENTS = [
    {**SAMPLE_EVENT, "event_id": f"DEM_00{i}_{i}", "demand_node_id": f"D{i}",
     "demand_node_index": i * 100,
     "time_slot": i * 6,
     "material_type": mat,
     "priority": pri}
    for i, (mat, pri) in enumerate([
        ("vaccine", 1), ("medicine", 2), ("mask", 3),
        ("protective_suit", 4), ("disinfectant", 5), ("ventilator", 1),
    ])
]


# ============================================================================
# 工具函数测试
# ============================================================================

class TestHaversine:
    def test_same_point_is_zero(self):
        assert _haversine(113.0, 22.0, 113.0, 22.0) == pytest.approx(0.0)

    def test_known_distance(self):
        # 深圳北站 (114.029, 22.609) 到罗湖口岸 (114.117, 22.537) 约 11-13km
        d = _haversine(114.029, 22.609, 114.117, 22.537)
        assert 8000 < d < 16000

    def test_symmetry(self):
        d1 = _haversine(113.0, 22.0, 114.0, 23.0)
        d2 = _haversine(114.0, 23.0, 113.0, 22.0)
        assert d1 == pytest.approx(d2)


class TestFindNearestStation:
    def test_returns_nearest(self):
        # 查询坐标靠近 ST001（燕罗航站）
        result = _find_nearest_station(113.88, 22.80, SAMPLE_STATIONS)
        assert result["station_id"] == "ST001"

    def test_returns_nearest_east(self):
        # 查询坐标靠近 ST003（福城航站）
        result = _find_nearest_station(114.02, 22.74, SAMPLE_STATIONS)
        assert result["station_id"] == "ST003"

    def test_single_station(self):
        result = _find_nearest_station(0.0, 0.0, [SAMPLE_STATIONS[0]])
        assert result["station_id"] == "ST001"


class TestPickTemplate:
    def test_priority1_ventilator(self):
        tpl = _pick_tier_template(_map_priority_to_tier(1), "ventilator", random.Random(1))
        assert "ICU" in tpl or "min" in tpl

    def test_priority2_vaccine(self):
        tpl = _pick_tier_template(_map_priority_to_tier(2), "vaccine", random.Random(2))
        assert "{dest_id}" in tpl

    def test_fallback_for_unknown(self):
        tpl = _pick_tier_template(_map_priority_to_tier(99), "unknown_material", random.Random(3))
        assert "{origin_name}" in tpl


class TestGenerateRuleConversation:
    def test_returns_string(self):
        station = SAMPLE_STATIONS[0]
        conv = _generate_rule_conversation("D001", station, "vaccine", 2.1, 2)
        assert isinstance(conv, str)
        assert len(conv) > 10

    def test_contains_station_name(self):
        station = SAMPLE_STATIONS[0]
        conv = _generate_rule_conversation("D001", station, "medicine", 5.0, 1)
        assert station["name"] in conv

    def test_all_materials(self):
        materials = ["vaccine", "medicine", "protective_suit", "mask", "disinfectant", "ventilator"]
        for mat in materials:
            conv = _generate_rule_conversation("D001", SAMPLE_STATIONS[0], mat, 1.0, 3)
            assert isinstance(conv, str) and len(conv) > 5

    def test_all_priorities(self):
        for pri in range(1, 6):
            conv = _generate_rule_conversation("D001", SAMPLE_STATIONS[0], "medicine", 1.0, pri)
            assert isinstance(conv, str) and len(conv) > 5


class TestInferPoi:
    def test_vaccine_returns_clinic(self):
        poi = _infer_poi("vaccine", 3)
        assert "clinic" in poi or "community_health_center" in poi

    def test_priority1_adds_emergency(self):
        poi = _infer_poi("medicine", 1)
        assert "emergency_response_center" in poi

    def test_unknown_material_returns_residential(self):
        poi = _infer_poi("unknown", 3)
        assert "residential" in poi


# ============================================================================
# 核心转换测试
# ============================================================================

class TestEventToDialogue:
    def test_output_structure(self):
        dlg = _event_to_dialogue(SAMPLE_EVENT, SAMPLE_STATIONS, "2024-03-15", 1)
        assert "dialogue_id" in dlg
        assert "timestamp" in dlg
        assert "conversation" in dlg
        assert "metadata" in dlg

    def test_dialogue_id_format(self):
        dlg = _event_to_dialogue(SAMPLE_EVENT, SAMPLE_STATIONS, "2024-03-15", 42)
        assert dlg["dialogue_id"] == "D0042"

    def test_timestamp_from_time_slot(self):
        # time_slot=12 → 12*5=60 min after midnight → 01:00:00
        dlg = _event_to_dialogue(SAMPLE_EVENT, SAMPLE_STATIONS, "2024-03-15", 1)
        assert dlg["timestamp"] == "2024-03-15T01:00:00"

    def test_timestamp_slot0(self):
        event = {**SAMPLE_EVENT, "time_slot": 0}
        dlg = _event_to_dialogue(event, SAMPLE_STATIONS, "2024-03-15", 1)
        assert dlg["timestamp"] == "2024-03-15T00:00:00"

    def test_nearest_station_used_as_origin(self):
        # demand coords (113.868851, 22.665646) 最近的是 ST002（铁甲航站）
        dlg = _event_to_dialogue(SAMPLE_EVENT, SAMPLE_STATIONS, "2024-03-15", 1)
        meta = dlg["metadata"]
        assert meta["origin_fid"] in {"ST001", "ST002", "ST003"}

    def test_destination_fid_from_index(self):
        dlg = _event_to_dialogue(SAMPLE_EVENT, SAMPLE_STATIONS, "2024-03-15", 1)
        assert dlg["metadata"]["destination_fid"] == 2200

    def test_metadata_has_required_fields(self):
        dlg = _event_to_dialogue(SAMPLE_EVENT, SAMPLE_STATIONS, "2024-03-15", 1)
        meta = dlg["metadata"]
        required = [
            "origin_fid", "destination_fid", "origin_coords", "dest_coords",
            "dest_demographics", "nearby_poi", "material_type", "quantity_kg",
            "priority",
        ]
        for field in required:
            assert field in meta, f"缺少字段: {field}"

    def test_custom_conversation(self):
        dlg = _event_to_dialogue(
            SAMPLE_EVENT, SAMPLE_STATIONS, "2024-03-15", 1,
            conversation="自定义对话文本"
        )
        assert dlg["conversation"] == "自定义对话文本"

    def test_demographics_is_dict(self):
        dlg = _event_to_dialogue(SAMPLE_EVENT, SAMPLE_STATIONS, "2024-03-15", 1)
        demo = dlg["metadata"]["dest_demographics"]
        assert "elderly_ratio" in demo and "population" in demo
        assert 0.0 <= demo["elderly_ratio"] <= 1.0
        assert demo["population"] > 0

    def test_origin_coords_is_list(self):
        dlg = _event_to_dialogue(SAMPLE_EVENT, SAMPLE_STATIONS, "2024-03-15", 1)
        coords = dlg["metadata"]["origin_coords"]
        assert isinstance(coords, list) and len(coords) == 2

    def test_dest_coords_match_event(self):
        dlg = _event_to_dialogue(SAMPLE_EVENT, SAMPLE_STATIONS, "2024-03-15", 1)
        meta = dlg["metadata"]
        assert meta["dest_coords"][0] == pytest.approx(float(SAMPLE_EVENT["demand_lon"]))
        assert meta["dest_coords"][1] == pytest.approx(float(SAMPLE_EVENT["demand_lat"]))


# ============================================================================
# 批量生成测试
# ============================================================================

class TestGenerateDialoguesOffline:
    def test_count_matches_events(self):
        dialogues = generate_dialogues_offline(SAMPLE_EVENTS, SAMPLE_STATIONS, "2024-03-15")
        assert len(dialogues) == len(SAMPLE_EVENTS)

    def test_dialogue_ids_are_unique(self):
        dialogues = generate_dialogues_offline(SAMPLE_EVENTS, SAMPLE_STATIONS)
        ids = [d["dialogue_id"] for d in dialogues]
        assert len(ids) == len(set(ids))

    def test_timestamps_ordered_by_time_slot(self):
        dialogues = generate_dialogues_offline(SAMPLE_EVENTS, SAMPLE_STATIONS)
        ts_list = [d["timestamp"] for d in dialogues]
        # time_slots are 0, 6, 12, 18, 24, 30 → timestamps should be ascending
        assert ts_list == sorted(ts_list)

    def test_all_have_conversation(self):
        dialogues = generate_dialogues_offline(SAMPLE_EVENTS, SAMPLE_STATIONS)
        for d in dialogues:
            assert isinstance(d["conversation"], str) and len(d["conversation"]) > 0

    def test_empty_events_returns_empty(self):
        dialogues = generate_dialogues_offline([], SAMPLE_STATIONS)
        assert dialogues == []


# ============================================================================
# 文件 I/O 测试
# ============================================================================

class TestSaveDialogues:
    def test_save_and_reload(self):
        dialogues = generate_dialogues_offline(SAMPLE_EVENTS[:3], SAMPLE_STATIONS)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            tmp_path = f.name
        save_dialogues(dialogues, tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            loaded = [json.loads(l.strip()) for l in f if l.strip()]
        assert len(loaded) == 3
        assert loaded[0]["dialogue_id"] == dialogues[0]["dialogue_id"]

    def test_save_creates_parent_dir(self):
        dialogues = generate_dialogues_offline(SAMPLE_EVENTS[:1], SAMPLE_STATIONS)
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = str(Path(tmpdir) / "subdir" / "output.jsonl")
            save_dialogues(dialogues, nested)
            assert Path(nested).exists()


# ============================================================================
# 真实文件测试（integration）
# ============================================================================

CSV_PATH = str(DEMAND_EVENTS_PATH)
XLSX_PATH = str(STATION_DATA_PATH)


@pytest.mark.skipif(
    not Path(CSV_PATH).exists() or not Path(XLSX_PATH).exists(),
    reason="真实数据文件不存在，跳过集成测试"
)
class TestWithRealData:
    def test_load_stations(self):
        stations = load_stations(XLSX_PATH)
        assert len(stations) > 0
        for s in stations:
            assert "station_id" in s
            assert "lon" in s and "lat" in s
            assert -180 <= s["lon"] <= 180
            assert -90 <= s["lat"] <= 90

    def test_load_demand_events_sample(self):
        events = load_demand_events(CSV_PATH, n_events=10)
        assert 1 <= len(events) <= 10
        event = events[0]
        assert "time_slot" in event
        assert "demand_lon" in event
        assert "demand_lat" in event
        assert "material_type" in event
        assert "quantity_kg" in event
        assert "priority" in event

    def test_load_demand_events_time_slots(self):
        events = load_demand_events(CSV_PATH, time_slots=[0, 1, 2])
        for e in events:
            assert e["time_slot"] in [0, 1, 2]

    def test_offline_generate_10_events(self):
        stations = load_stations(XLSX_PATH)
        events = load_demand_events(CSV_PATH, n_events=10)
        dialogues = generate_dialogues_offline(events, stations)
        assert len(dialogues) == len(events)
        for d in dialogues:
            assert d["dialogue_id"].startswith("D")
            assert "T" in d["timestamp"]
            assert len(d["conversation"]) > 10
            meta = d["metadata"]
            assert len(meta["origin_coords"]) == 2
            assert len(meta["dest_coords"]) == 2

    def test_dialogue_json_serializable(self):
        stations = load_stations(XLSX_PATH)
        events = load_demand_events(CSV_PATH, n_events=5)
        dialogues = generate_dialogues_offline(events, stations)
        # 确保可以序列化为 JSON（无 NaN/Inf 等）
        json_str = json.dumps(dialogues, ensure_ascii=False)
        reloaded = json.loads(json_str)
        assert len(reloaded) == len(dialogues)
