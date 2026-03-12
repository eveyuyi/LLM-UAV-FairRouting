from drone_pipeline.pipeline.solver_runner import (
    _parse_window_bounds,
    _select_station_dicts,
    demands_to_solver_inputs,
    solve_window_demands,
)


def _sample_demands():
    return [
        {
            "demand_id": "REQ001",
            "origin": {"fid": "COM_A", "coords": [113.8800, 22.8000]},
            "destination": {"fid": "DEM_1", "coords": [113.8810, 22.8010], "type": "residential_area"},
            "cargo": {"weight_kg": 5.0, "type": "medicine"},
            "priority_evaluation_signals": {},
        },
        {
            "demand_id": "REQ002",
            "origin": {"fid": "COM_A", "coords": [113.8800, 22.8000]},
            "destination": {"fid": "DEM_2", "coords": [113.8820, 22.8020], "type": "clinic"},
            "cargo": {"weight_kg": 3.5, "type": "vaccine"},
            "priority_evaluation_signals": {},
        },
    ]


def test_demands_to_solver_inputs_deduplicates_supply_points():
    supply_points, demand_points, demand_weights, required_supply = demands_to_solver_inputs(
        _sample_demands()
    )

    assert len(supply_points) == 1
    assert len(demand_points) == 2
    assert demand_weights == [5.0, 3.5]
    assert required_supply == [0, 0]


def test_select_station_dicts_preserves_source_order():
    stations = [
        {"station_id": "ST010", "name": "first", "lon": 114.1000, "lat": 22.9500},
        {"station_id": "ST001", "name": "second", "lon": 113.8795, "lat": 22.7995},
        {"station_id": "ST002", "name": "third", "lon": 113.9300, "lat": 22.8200},
    ]

    selected = _select_station_dicts(stations, _sample_demands(), max_stations=2)

    assert [station["station_id"] for station in selected] == ["ST010", "ST001"]


def test_parse_window_bounds_handles_24_hour_end():
    start_dt, end_dt = _parse_window_bounds("2024-03-15T23:55-24:00")

    assert start_dt.isoformat() == "2024-03-15T23:55:00"
    assert end_dt.isoformat() == "2024-03-16T00:00:00"
    assert (end_dt - start_dt).total_seconds() == 300


def test_solve_window_demands_returns_early_when_all_demands_overweight():
    overweight_demands = [
        {
            "demand_id": "REQ999",
            "origin": {"fid": "COM_A", "coords": [113.8800, 22.8000]},
            "destination": {"fid": "DEM_9", "coords": [113.8810, 22.8010], "type": "residential_area"},
            "cargo": {"weight_kg": 40.0, "type": "medicine"},
            "priority_evaluation_signals": {},
        }
    ]
    weight_config = {
        "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
        "demand_configs": [{"demand_id": "REQ999", "alpha": 1.0, "beta": 1.0, "priority": 3}],
        "supplementary_constraints": [],
    }

    result = solve_window_demands(
        time_window="2024-03-15T09:00-09:05",
        demands=overweight_demands,
        weight_config=weight_config,
        stations_path=None,
        max_solver_stations=3,
        time_limit=1,
        max_drones_per_station=1,
        max_payload=25.0,
        max_range=1000.0,
        noise_weight=0.0,
    )

    assert result["solution"] is None
    assert result["n_demands_filtered"] == 1
    assert result["feasible_demands"] == []
