import types

import numpy as np
import pytest

pytest.importorskip("pyomo.environ")

from llm4fairrouting.routing.assignment_model import CplexSolver, TerminationCondition
from llm4fairrouting.routing.domain import DemandEvent, Drone, DroneState, DroneStatus, Point
from llm4fairrouting.routing.simulator import FinalDroneSimulator


class RecordingMatrix:
    def __init__(self, values):
        self.values = dict(values)
        self.queries = []

    def __getitem__(self, key):
        i, j = int(key[0]), int(key[1])
        self.queries.append((i, j))
        if (i, j) in self.values:
            return self.values[(i, j)]
        if (j, i) in self.values:
            return self.values[(j, i)]
        return float(abs(i - j) + 1)


def _build_solver_fixture():
    all_points = [
        Point(id="S1", lon=0.0, lat=0.0, alt=0.0, type="supply"),
        Point(id="S2", lon=0.0, lat=0.0, alt=0.0, type="supply"),
        Point(id="D1", lon=0.0, lat=0.0, alt=0.0, type="demand"),
        Point(id="D2", lon=0.0, lat=0.0, alt=0.0, type="demand"),
        Point(id="L1", lon=0.0, lat=0.0, alt=0.0, type="station"),
    ]
    dist_values = {
        (0, 1): 8.0,
        (0, 2): 12.0,
        (0, 3): 18.0,
        (0, 4): 10.0,
        (1, 2): 9.0,
        (1, 3): 11.0,
        (1, 4): 13.0,
        (2, 3): 7.0,
        (2, 4): 14.0,
        (3, 4): 15.0,
    }
    dist_matrix = RecordingMatrix(dist_values)
    noise_matrix = RecordingMatrix({key: 0.0 for key in dist_values})
    solver = CplexSolver(
        drones=[
            Drone(id="U11", station_id=0, station_name="L1", max_payload=10.0, max_range=500.0, speed=60.0),
            Drone(id="U12", station_id=0, station_name="L1", max_payload=10.0, max_range=500.0, speed=60.0),
        ],
        supply_indices=[0, 1],
        station_indices=[4],
        dist_matrix=dist_matrix,
        all_points=all_points,
        noise_cost_matrix=noise_matrix,
        noise_weight=0.5,
        time_limit=1,
    )
    drone_states = [
        DroneState(
            drone_id="U11",
            station_id=0,
            current_node=4,
            remaining_range=500.0,
            remaining_payload=10.0,
            status=DroneStatus.IDLE,
        ),
        DroneState(
            drone_id="U12",
            station_id=0,
            current_node=4,
            remaining_range=500.0,
            remaining_payload=10.0,
            status=DroneStatus.IDLE,
        ),
    ]
    demands = [
        DemandEvent(time=0.0, node_idx=2, weight=2.0, unique_id="REQ1", priority=1, required_supply_idx=0),
        DemandEvent(time=0.0, node_idx=3, weight=2.0, unique_id="REQ2", priority=2, required_supply_idx=1),
    ]
    return solver, drone_states, demands, dist_matrix


def test_assignment_solver_extracts_multi_stop_route(monkeypatch):
    solver, drone_states, demands, dist_matrix = _build_solver_fixture()

    class FakeSolver:
        def __init__(self):
            self.options = {}

        def solve(self, model, tee=True):
            model.used[0].set_value(1)
            model.serve[0, 0].set_value(1)
            model.serve[0, 1].set_value(1)
            model.visit[0, 0].set_value(1)
            model.visit[0, 1].set_value(1)
            model.visit[0, 2].set_value(1)
            model.visit[0, 3].set_value(1)
            model.x[0, -1, 0].set_value(1)
            model.x[0, 0, 1].set_value(1)
            model.x[0, 1, 2].set_value(1)
            model.x[0, 2, 3].set_value(1)
            model.x[0, 3, -2].set_value(1)
            model.arrival[0, 0].set_value(0.01)
            model.arrival[0, 1].set_value(0.02)
            model.arrival[0, 2].set_value(0.03)
            model.arrival[0, 3].set_value(0.04)
            model.end_time[0].set_value(0.05)
            model.unassigned[0].set_value(0)
            model.unassigned[1].set_value(0)
            return types.SimpleNamespace(
                solver=types.SimpleNamespace(
                    termination_condition=TerminationCondition.optimal
                )
            )

    monkeypatch.setattr(
        "llm4fairrouting.routing.assignment_model.SolverFactory",
        lambda name: FakeSolver(),
    )

    assignments = solver.solve_assignment(drone_states, demands, current_time=0.0)

    assert len(assignments) == 1
    route = assignments[0]
    assert route["served_demand_ids"] == ["REQ1", "REQ2"]
    assert [stop["type"] for stop in route["route_stops"]] == [
        "pickup",
        "pickup",
        "delivery",
        "delivery",
        "station",
    ]
    assert route["path_node_ids"] == ["L1", "S1", "S2", "D1", "D2", "L1"]
    assert route["delivery_times_h"]["REQ1"] == pytest.approx(0.03)
    assert route["delivery_times_h"]["REQ2"] == pytest.approx(0.04)
    assert (4, 0) in dist_matrix.queries
    assert (0, 1) in dist_matrix.queries
    assert (2, 3) in dist_matrix.queries


def test_simulator_executes_multi_demand_route_in_single_sortie(monkeypatch):
    supply_points = [
        Point(id="S1", lon=0.0, lat=0.0, alt=0.0, x=10.0, y=0.0, type="supply"),
        Point(id="S2", lon=0.0, lat=0.0, alt=0.0, x=20.0, y=0.0, type="supply"),
    ]
    demand_points = [
        Point(id="D1", lon=0.0, lat=0.0, alt=0.0, x=30.0, y=0.0, type="demand"),
        Point(id="D2", lon=0.0, lat=0.0, alt=0.0, x=40.0, y=0.0, type="demand"),
    ]
    station_points = [
        Point(id="L1", lon=0.0, lat=0.0, alt=0.0, x=0.0, y=0.0, type="station"),
    ]
    dist_matrix = np.array(
        [
            [0.0, 10.0, 20.0, 30.0, 10.0],
            [10.0, 0.0, 10.0, 20.0, 20.0],
            [20.0, 10.0, 0.0, 10.0, 30.0],
            [30.0, 20.0, 10.0, 0.0, 40.0],
            [10.0, 20.0, 30.0, 40.0, 0.0],
        ]
    )
    drones = [
        Drone(id="U11", station_id=0, station_name="L1", max_payload=10.0, max_range=500.0, speed=10.0),
    ]
    demands = [
        DemandEvent(time=0.0, node_idx=2, weight=2.0, unique_id="REQ1", priority=1, required_supply_idx=0),
        DemandEvent(time=0.0, node_idx=3, weight=3.0, unique_id="REQ2", priority=2, required_supply_idx=1),
    ]
    simulator = FinalDroneSimulator(
        supply_points=supply_points,
        demand_points=demand_points,
        station_points=station_points,
        drones_static=drones,
        dist_matrix=dist_matrix,
        demand_events=demands,
        noise_cost_matrix=None,
        noise_weight=0.0,
        time_step=0.0005,
        solve_interval=0.01,
        time_limit=1,
    )

    def fake_solve_assignment(idle_drones, pending, current_time):
        drone = idle_drones[0]
        pending_by_id = {d.unique_id: d for d in pending}
        return [
            {
                "drone": drone,
                "drone_idx": 0,
                "served_demands": [
                    {
                        "demand": pending_by_id["REQ1"],
                        "demand_idx": 0,
                        "supply_idx": 0,
                        "supply_node": 0,
                        "delivery_time_h": current_time + 0.01,
                    },
                    {
                        "demand": pending_by_id["REQ2"],
                        "demand_idx": 1,
                        "supply_idx": 1,
                        "supply_node": 1,
                        "delivery_time_h": current_time + 0.02,
                    },
                ],
                "served_demand_ids": ["REQ1", "REQ2"],
                "route_stops": [
                    {"type": "pickup", "node": 0, "demand_id": "REQ1", "demand_node": 2, "supply_node": 0, "weight": 2.0, "priority": 1},
                    {"type": "pickup", "node": 1, "demand_id": "REQ2", "demand_node": 3, "supply_node": 1, "weight": 3.0, "priority": 2},
                    {"type": "delivery", "node": 2, "demand_id": "REQ1", "demand_node": 2, "supply_node": 0, "weight": 2.0, "priority": 1},
                    {"type": "delivery", "node": 3, "demand_id": "REQ2", "demand_node": 3, "supply_node": 1, "weight": 3.0, "priority": 2},
                    {"type": "station", "node": 4, "demand_id": None, "demand_node": None, "supply_node": None, "weight": 0.0, "priority": None},
                ],
                "path_node_indices": [4, 0, 1, 2, 3, 4],
                "path_node_ids": ["L1", "S1", "S2", "D1", "D2", "L1"],
                "path_str": "L1 -> S1 -> S2 -> D1 -> D2 -> L1",
            }
        ]

    monkeypatch.setattr(simulator.cplex_solver, "solve_assignment", fake_solve_assignment)

    simulator.run_until_complete(max_time=0.2)

    assert all(demand.assigned_drone == "U11" for demand in demands)
    assert all(demand.served_time is not None for demand in demands)
    assert simulator.drone_states[0].status == DroneStatus.IDLE
    assert simulator.drone_states[0].current_load == pytest.approx(0.0)
    assert simulator.drone_states[0].remaining_payload == pytest.approx(10.0)
    path_details = simulator.get_drone_path_details()
    assert path_details[0]["path_node_ids"] == ["L1", "S1", "S2", "D1", "D2", "L1"]
