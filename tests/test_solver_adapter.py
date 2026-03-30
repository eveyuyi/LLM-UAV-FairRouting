from llm4fairrouting.workflow.solver_adapter import (
    _parse_window_bounds,
    _select_station_dicts,
    demands_to_solver_inputs,
    serialize_workflow_results,
    solve_windows_dynamically,
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


def test_solve_windows_dynamically_returns_early_when_all_demands_overweight():
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
        "demand_configs": [{"demand_id": "REQ999", "priority": 4}],
        "supplementary_constraints": [],
    }

    results = solve_windows_dynamically(
        windows=[{"time_window": "2024-03-15T09:00-09:05", "demands": overweight_demands}],
        weight_configs={"2024-03-15T09:00-09:05": weight_config},
        stations_path=None,
        building_path=None,
        max_solver_stations=3,
        time_limit=1,
        max_drones_per_station=1,
        max_payload=25.0,
        max_range=1000.0,
        noise_weight=0.0,
    )
    result = results[0]

    assert result["solution"] is None
    assert result["n_demands_filtered"] == 1
    assert result["feasible_demands"] == []


def test_serialize_workflow_results_keeps_dynamic_drone_path_details():
    all_solutions = [
        {
            "time_window": "2024-03-15T09:00-09:05",
            "weight_config": {
                "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
                "demand_configs": [],
                "supplementary_constraints": [],
            },
            "feasible_demands": [],
            "n_demands_total": 0,
            "n_demands_filtered": 0,
            "solution": {
                "solve_mode": "dynamic_periodic",
                "solve_status": "completed",
                "solve_time_s": 0.1,
                "drone_speed_ms": 60.0,
                "snapshot_time_h": 0.0,
                "snapshot_time_window_end": "2024-03-15T09:05:00",
                "busy_drones": [],
                "total_distance": 1200.0,
                "total_noise_impact": 5.0,
                "objective_value": None,
                "run_summary": {
                    "final_total_distance_m": 1400.0,
                    "final_total_noise_impact": 6.0,
                    "average_delivery_time_h": 0.25,
                },
                "analytics_artifacts": {
                    "analytics_json": "/tmp/solver_analytics.json",
                    "charts": [
                        {"chart_type": "gantt", "path": "/tmp/drone_schedule_gantt.png"}
                    ],
                },
                "demand_event_results": {},
                "drone_path_details": [
                    {
                        "drone_id": "U11",
                        "path_node_ids": ["L1", "S_COM_A", "D_DEM_1", "L1"],
                        "path_str": "L1 -> S_COM_A -> D_DEM_1 -> L1",
                        "n_nodes_visited": 4,
                    }
                ],
            },
            "n_supply": 1,
        }
    ]

    serialized = serialize_workflow_results(all_solutions)

    assert serialized[0]["drone_path_details"][0]["drone_id"] == "U11"
    assert serialized[0]["drone_path_details"][0]["path_str"] == "L1 -> S_COM_A -> D_DEM_1 -> L1"
    assert serialized[0]["run_summary"]["final_total_distance_m"] == 1400.0
    assert serialized[0]["analytics_artifacts"]["charts"][0]["chart_type"] == "gantt"


def test_serialize_workflow_results_includes_source_event_and_latency_fields():
    all_solutions = [
        {
            "time_window": "2024-03-15T09:00-09:05",
            "weight_config": {
                "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
                "demand_configs": [
                    {"demand_id": "REQ001", "priority": 1, "window_rank": 1, "reasoning": "urgent"}
                ],
                "supplementary_constraints": [],
            },
            "feasible_demands": [
                {
                    "demand_id": "REQ001",
                    "solver_event_id": "W::REQ001::0",
                    "source_event_id": "EV001",
                    "source_dialogue_id": "D001",
                    "request_timestamp": "2024-03-15T09:00:00",
                    "demand_tier": "critical",
                    "origin": {"fid": "COM_A", "coords": [113.88, 22.80]},
                    "destination": {"fid": "DEM_1", "coords": [113.89, 22.81], "type": "clinic"},
                    "cargo": {"weight_kg": 2.0, "type": "medicine", "type_cn": "medicine", "temperature_sensitive": False},
                    "priority_evaluation_signals": {"population_vulnerability": {}},
                    "time_constraint": {"deadline_minutes": 20},
                }
            ],
            "n_demands_total": 1,
            "n_demands_filtered": 0,
            "solution": {
                "solve_mode": "dynamic_periodic",
                "solve_status": "completed",
                "solve_time_s": 0.1,
                "drone_speed_ms": 60.0,
                "snapshot_time_h": 0.0,
                "snapshot_time_window_end": "2024-03-15T09:05:00",
                "busy_drones": [],
                "total_distance": 100.0,
                "total_noise_impact": 1.0,
                "objective_value": None,
                "run_summary": {},
                "analytics_artifacts": {},
                "demand_event_results": {
                    "W::REQ001::0": {
                        "event_id": "W::REQ001::0",
                        "source_event_id": "EV001",
                        "assigned_drone": "U11",
                        "assigned_time_h": 0.02,
                        "served_time_h": 0.1,
                        "served_time_s": 360.0,
                        "delivery_latency_h": 0.08,
                        "delivery_latency_s": 288.0,
                        "is_assigned_by_snapshot": True,
                        "is_served_by_snapshot": True,
                        "is_served_eventually": True,
                    }
                },
                "drone_path_details": [],
            },
            "n_supply": 1,
        }
    ]

    serialized = serialize_workflow_results(all_solutions)
    per_demand = serialized[0]["per_demand_results"][0]

    assert per_demand["source_event_id"] == "EV001"
    assert per_demand["window_rank"] == 1
    assert per_demand["deadline_minutes"] == 20
    assert per_demand["delivery_latency_s"] == 288.0
    assert per_demand["is_deadline_met"] is True
