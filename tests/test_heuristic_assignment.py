from __future__ import annotations

import numpy as np

from llm4fairrouting.routing.domain import DemandEvent, Drone, DroneState, DroneStatus, Point
from llm4fairrouting.routing.heuristic_assignment import HeuristicAssignmentSolver



def test_heuristic_assignment_batches_multiple_demands_and_applies_local_search():
    all_points = [
        Point(id="S1", lon=0.0, lat=0.0, alt=0.0, type="supply"),
        Point(id="S2", lon=0.0, lat=0.0, alt=0.0, type="supply"),
        Point(id="D1", lon=0.0, lat=0.0, alt=0.0, type="demand"),
        Point(id="D2", lon=0.0, lat=0.0, alt=0.0, type="demand"),
        Point(id="L1", lon=0.0, lat=0.0, alt=0.0, type="station"),
    ]
    dist_matrix = np.array(
        [
            [0.0, 5.0, 12.0, 14.0, 6.0],
            [5.0, 0.0, 7.0, 6.0, 7.0],
            [12.0, 7.0, 0.0, 4.0, 10.0],
            [14.0, 6.0, 4.0, 0.0, 9.0],
            [6.0, 7.0, 10.0, 9.0, 0.0],
        ]
    )
    noise_matrix = np.zeros_like(dist_matrix)
    solver = HeuristicAssignmentSolver(
        drones=[
            Drone(id="U11", station_id=0, station_name="L1", max_payload=8.0, max_range=1000.0, speed=10.0),
        ],
        supply_indices=[0, 1],
        station_indices=[4],
        dist_matrix=dist_matrix,
        all_points=all_points,
        noise_cost_matrix=noise_matrix,
        noise_weight=0.5,
        drone_activation_cost=100.0,
    )
    drone_states = [
        DroneState(
            drone_id="U11",
            station_id=0,
            current_node=4,
            remaining_range=1000.0,
            remaining_payload=8.0,
            status=DroneStatus.IDLE,
        )
    ]
    demands = [
        DemandEvent(time=0.0, node_idx=2, weight=2.0, unique_id="REQ1", priority=1, required_supply_idx=0),
        DemandEvent(time=0.0, node_idx=3, weight=3.0, unique_id="REQ2", priority=2, required_supply_idx=1),
    ]

    assignments = solver.solve_assignment(
        drone_states,
        demands,
        current_time=0.0,
        objective_weights={"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
        solve_context={"time_window": "W0"},
    )

    assert len(assignments) == 1
    route = assignments[0]
    assert set(route["served_demand_ids"]) == {"REQ1", "REQ2"}
    stop_types = [stop["type"] for stop in route["route_stops"]]
    assert stop_types.count("pickup") == 2
    assert stop_types.count("delivery") == 2
    assert stop_types[-1] == "station"
    assert solver.last_solve_details["model_size"]["local_search_moves_applied"] >= 0
    assert solver.last_solve_details["model_size"]["heuristic_selection_attempts"] >= 2


def test_heuristic_assignment_does_not_duplicate_demands_across_idle_drones():
    all_points = [
        Point(id="S1", lon=0.0, lat=0.0, alt=0.0, type="supply"),
        Point(id="D1", lon=0.0, lat=0.0, alt=0.0, type="demand"),
        Point(id="D2", lon=0.0, lat=0.0, alt=0.0, type="demand"),
        Point(id="L1", lon=0.0, lat=0.0, alt=0.0, type="station"),
    ]
    dist_matrix = np.array(
        [
            [0.0, 4.0, 5.0, 2.0],
            [4.0, 0.0, 1.0, 3.0],
            [5.0, 1.0, 0.0, 4.0],
            [2.0, 3.0, 4.0, 0.0],
        ]
    )
    solver = HeuristicAssignmentSolver(
        drones=[
            Drone(id="U11", station_id=0, station_name="L1", max_payload=10.0, max_range=1000.0, speed=10.0),
            Drone(id="U12", station_id=0, station_name="L1", max_payload=10.0, max_range=1000.0, speed=10.0),
        ],
        supply_indices=[0],
        station_indices=[3],
        dist_matrix=dist_matrix,
        all_points=all_points,
        noise_cost_matrix=np.zeros_like(dist_matrix),
        noise_weight=0.5,
        drone_activation_cost=100.0,
    )
    drone_states = [
        DroneState(
            drone_id="U11",
            station_id=0,
            current_node=3,
            remaining_range=1000.0,
            remaining_payload=10.0,
            status=DroneStatus.IDLE,
        ),
        DroneState(
            drone_id="U12",
            station_id=0,
            current_node=3,
            remaining_range=1000.0,
            remaining_payload=10.0,
            status=DroneStatus.IDLE,
        ),
    ]
    demands = [
        DemandEvent(time=0.0, node_idx=1, weight=2.0, unique_id="REQ1", priority=1, required_supply_idx=0),
        DemandEvent(time=0.0, node_idx=2, weight=3.0, unique_id="REQ2", priority=2, required_supply_idx=0),
    ]

    assignments = solver.solve_assignment(
        drone_states,
        demands,
        current_time=0.0,
        objective_weights={"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
        solve_context={"time_window": "W0"},
    )

    served_ids = [
        demand_id
        for assignment in assignments
        for demand_id in assignment["served_demand_ids"]
    ]
    assert sorted(served_ids) == ["REQ1", "REQ2"]
    assert len(served_ids) == len(set(served_ids))
