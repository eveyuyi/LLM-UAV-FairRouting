import pandas as pd

from llm4fairrouting.baselines.cplex_with_seed_priorities import build_seed_priority_inputs


def test_build_seed_priority_inputs_groups_events_and_preserves_csv_priority(tmp_path):
    csv_path = tmp_path / "daily_demand_events.csv"
    pd.DataFrame(
        [
            {
                "time": 0.0,
                "demand_fid": "DEM_1",
                "demand_lon": 113.90,
                "demand_lat": 22.80,
                "priority": 2,
                "supply_fid": "MED_1",
                "supply_lon": 113.80,
                "supply_lat": 22.70,
                "supply_type": "medical",
                "material_weight": 3.2,
                "unique_id": "DEM_000_00",
            },
            {
                "time": 0.0833,
                "demand_fid": "DEM_2",
                "demand_lon": 113.91,
                "demand_lat": 22.81,
                "priority": 4,
                "supply_fid": "COM_1",
                "supply_lon": 113.81,
                "supply_lat": 22.71,
                "supply_type": "commercial",
                "material_weight": 1.8,
                "unique_id": "DEM_001_00",
            },
        ]
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    windows, weight_configs = build_seed_priority_inputs(
        csv_path=str(csv_path),
        base_date="2024-03-15",
        window_minutes=5,
    )

    assert len(windows) == 2
    first_window = windows[0]["time_window"]
    second_window = windows[1]["time_window"]
    assert weight_configs[first_window]["demand_configs"][0]["priority"] == 2
    assert weight_configs[second_window]["demand_configs"][0]["priority"] == 4
