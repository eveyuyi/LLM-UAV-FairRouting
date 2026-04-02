import time

import llm4fairrouting.llm.demand_extraction as demand_extraction_module
from llm4fairrouting.llm.demand_extraction import extract_all_demands, extract_demands_offline
from llm4fairrouting.llm.dialogue_generation import _event_to_dialogue


SAMPLE_STATION = [
    {"station_id": "ST001", "name": "Yanluo Hub", "lon": 113.879911, "lat": 22.799354},
]


def test_extract_demands_offline_infers_life_support_from_dialogue_text():
    event = {
        "time_slot": 0,
        "event_id": "DEM_LIFE_001",
        "demand_node_id": "D100",
        "demand_lon": 113.868851,
        "demand_lat": 22.665646,
        "material_type": "aed",
        "quantity_kg": 2.0,
        "priority": 1,
        "supply_fid": "MED_01",
        "supply_lon": 113.8076,
        "supply_lat": 22.6903,
        "supply_type": "medical",
    }
    dialogue = _event_to_dialogue(event, SAMPLE_STATION, "2024-03-15", 1)

    result = extract_demands_offline([dialogue], window_minutes=5)
    demand = result[0]["demands"][0]

    assert demand["demand_tier"] == "life_support"
    assert demand["source_event_id"] == "DEM_LIFE_001"
    assert demand["time_constraint"]["deadline_minutes"] <= 15
    assert demand["priority_evaluation_signals"]["patient_condition"]
    assert demand["labels"]["extraction_observable_priority"] == 1
    assert "priority" not in dialogue["metadata"]


def test_extract_demands_offline_infers_consumer_request_from_app_style_dialogue():
    dialogue = {
        "dialogue_id": "D0002",
        "timestamp": "2024-03-15T09:00:00",
        "conversation": (
            "[09:00] Customer (consumer delivery app • D200): Same-day order request. "
            "Please send 1 box of OTC medication (0.6 kg) from Community Hub 3. "
            "Same-day delivery within 120 min is fine. Please drop it at the community locker.\n"
            "[09:00] Delivery Platform: Order confirmed. Community Hub 3 has queued the parcel for drone dispatch. "
            "ETA about 120 min. A locker drop-off is preferred.\n"
            "[09:00] Customer: Perfect. Please send the app notification once the drop is complete."
        ),
        "metadata": {
            "origin_fid": "COM_3",
            "destination_fid": "D200",
            "origin_coords": [113.80, 22.70],
            "dest_coords": [113.90, 22.80],
            "material_type": "otc_drug",
            "quantity_kg": 0.6,
            "delivery_deadline_minutes": 120,
            "supply_station_name": "Community Hub 3",
            "requester_role": "consumer",
            "dest_demographics": {"elderly_ratio": 0.15, "population": 1500},
            "nearby_poi": ["residential"],
        },
    }

    result = extract_demands_offline([dialogue], window_minutes=5)
    demand = result[0]["demands"][0]

    assert demand["demand_tier"] == "consumer"
    assert demand["source_event_id"] is None
    assert demand["destination"]["type"] == "residential_area"
    assert demand["priority_evaluation_signals"]["requester_role"] == "consumer"
    assert demand["labels"]["extraction_observable_priority"] == 4


def test_extract_all_demands_parallel_preserves_window_order(monkeypatch):
    dialogues = [
        {"dialogue_id": "D1", "timestamp": "2024-03-15T00:00:00", "conversation": "a", "metadata": {}},
        {"dialogue_id": "D2", "timestamp": "2024-03-15T00:30:00", "conversation": "b", "metadata": {}},
        {"dialogue_id": "D3", "timestamp": "2024-03-15T01:00:00", "conversation": "c", "metadata": {}},
    ]

    monkeypatch.setattr(
        demand_extraction_module,
        "group_by_time_window",
        lambda _dialogues, _window: {
            "2024-03-15T00:00-00:30": [dialogues[0]],
            "2024-03-15T00:30-01:00": [dialogues[1]],
            "2024-03-15T01:00-01:30": [dialogues[2]],
        },
    )

    def fake_extract(group, label, client, model, temperature=0.0):
        if label.endswith("00:30-01:00"):
            time.sleep(0.03)
        return {"time_window": label, "demands": [{"source_dialogue_id": group[0]["dialogue_id"]}]}

    monkeypatch.setattr(demand_extraction_module, "extract_demands_for_window", fake_extract)

    results = extract_all_demands(dialogues, client=object(), model="fake", window_minutes=30, max_concurrency=3)

    assert [row["time_window"] for row in results] == [
        "2024-03-15T00:00-00:30",
        "2024-03-15T00:30-01:00",
        "2024-03-15T01:00-01:30",
    ]
