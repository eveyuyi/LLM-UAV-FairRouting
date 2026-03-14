import types

import pytest

pytest.importorskip("pyomo.environ")

from llm4fairrouting.routing.assignment_model import CplexSolver, TerminationCondition
from llm4fairrouting.routing.domain import DemandEvent, Drone, DroneState, DroneStatus, Point


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
        raise KeyError((i, j))


def _build_solver_fixture():
    all_points = [
        Point(id="S1", lon=0.0, lat=0.0, alt=0.0, type="supply"),
        Point(id="S2", lon=0.0, lat=0.0, alt=0.0, type="supply"),
        Point(id="D1", lon=0.0, lat=0.0, alt=0.0, type="demand"),
        Point(id="D2", lon=0.0, lat=0.0, alt=0.0, type="demand"),
        Point(id="L1", lon=0.0, lat=0.0, alt=0.0, type="station"),
    ]
    dist_values = {
        (4, 0): 10.0,
        (0, 2): 20.0,
        (2, 4): 30.0,
        (4, 1): 11.0,
        (1, 3): 21.0,
        (3, 4): 31.0,
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


def test_assignment_solver_uses_only_required_supply_arcs(monkeypatch):
    solver, drone_states, demands, dist_matrix = _build_solver_fixture()

    class FakeSolver:
        def __init__(self):
            self.options = {}

        def solve(self, model, tee=True):
            model.assign[0, 0].set_value(1)
            model.assign[0, 1].set_value(0)
            model.assign[1, 0].set_value(0)
            model.assign[1, 1].set_value(1)
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

    assert {(item["demand_idx"], item["supply_idx"]) for item in assignments} == {
        (0, 0),
        (1, 1),
    }
    assert {(item["demand_idx"], item["supply_node"]) for item in assignments} == {
        (0, 0),
        (1, 1),
    }

    assert set(dist_matrix.queries) == {
        (4, 0),
        (0, 2),
        (2, 4),
        (4, 1),
        (1, 3),
        (3, 4),
    }
    assert (0, 3) not in dist_matrix.queries
    assert (1, 2) not in dist_matrix.queries
