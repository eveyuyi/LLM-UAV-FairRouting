from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from llm4fairrouting.routing.domain import Point
from llm4fairrouting.workflow import static_heuristic_solver as static_module


class _FakeElementwiseProblem:
    def __init__(self, n_var, n_obj, xl, xu):
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = xl
        self.xu = xu


class _FakeNSGA3:
    def __init__(self, pop_size, ref_dirs):
        self.pop_size = pop_size
        self.ref_dirs = ref_dirs


def _install_fake_pymoo(monkeypatch, candidate_vectors):
    pymoo_module = types.ModuleType("pymoo")
    algorithms_module = types.ModuleType("pymoo.algorithms")
    moo_module = types.ModuleType("pymoo.algorithms.moo")
    nsga3_module = types.ModuleType("pymoo.algorithms.moo.nsga3")
    core_module = types.ModuleType("pymoo.core")
    problem_module = types.ModuleType("pymoo.core.problem")
    optimize_module = types.ModuleType("pymoo.optimize")
    util_module = types.ModuleType("pymoo.util")
    ref_dirs_module = types.ModuleType("pymoo.util.ref_dirs")

    def fake_minimize(problem, algorithm, termination, seed, verbose):
        for vector in candidate_vectors:
            out = {}
            problem._evaluate(vector, out)
        return {"seed": seed, "termination": termination, "pop_size": algorithm.pop_size}

    nsga3_module.NSGA3 = _FakeNSGA3
    problem_module.ElementwiseProblem = _FakeElementwiseProblem
    optimize_module.minimize = fake_minimize

    def fake_ref_dirs(_name, n_obj, n_partitions=0):
        return [[1.0 if idx == 0 else 0.0 for idx in range(int(n_obj))]]

    ref_dirs_module.get_reference_directions = fake_ref_dirs

    monkeypatch.setitem(sys.modules, "pymoo", pymoo_module)
    monkeypatch.setitem(sys.modules, "pymoo.algorithms", algorithms_module)
    monkeypatch.setitem(sys.modules, "pymoo.algorithms.moo", moo_module)
    monkeypatch.setitem(sys.modules, "pymoo.algorithms.moo.nsga3", nsga3_module)
    monkeypatch.setitem(sys.modules, "pymoo.core", core_module)
    monkeypatch.setitem(sys.modules, "pymoo.core.problem", problem_module)
    monkeypatch.setitem(sys.modules, "pymoo.optimize", optimize_module)
    monkeypatch.setitem(sys.modules, "pymoo.util", util_module)
    monkeypatch.setitem(sys.modules, "pymoo.util.ref_dirs", ref_dirs_module)


def test_create_fixed_total_drones_round_robin():
    stations = [
        Point(id="L1", lon=0.0, lat=0.0, alt=50.0, type="station"),
        Point(id="L2", lon=1.0, lat=1.0, alt=50.0, type="station"),
    ]
    drones = static_module._create_fixed_total_drones(
        stations,
        total_drones=3,
        max_payload=60.0,
        max_range=1000.0,
        speed=60.0,
    )

    assert [drone.id for drone in drones] == ["U11", "U21", "U12"]
    assert [drone.station_id for drone in drones] == [0, 1, 0]


def test_run_nsga3_static_heuristic_search_saves_frontier_results(monkeypatch, tmp_path):
    candidate_vectors = [
        [0.1, 0.1, 0.1, 0.1],
        [0.7, 0.8, 0.7, 0.8],
        [0.4, 0.3, 0.5, 0.4],
    ]
    _install_fake_pymoo(monkeypatch, candidate_vectors)

    run_summaries = [
        {
            "final_total_distance_m": 100.0,
            "average_delivery_time_h": 1.0,
            "final_total_noise_impact": 10.0,
            "service_rate": 0.9,
            "service_rate_loss": 0.1,
            "n_used_drones": 1,
        },
        {
            "final_total_distance_m": 140.0,
            "average_delivery_time_h": 1.5,
            "final_total_noise_impact": 12.0,
            "service_rate": 0.8,
            "service_rate_loss": 0.2,
            "n_used_drones": 2,
        },
        {
            "final_total_distance_m": 90.0,
            "average_delivery_time_h": 1.3,
            "final_total_noise_impact": 11.0,
            "service_rate": 0.85,
            "service_rate_loss": 0.15,
            "n_used_drones": 1,
        },
    ]
    counter = {"value": 0}

    def fake_static_solver(**kwargs):
        idx = counter["value"]
        counter["value"] += 1
        return {
            "time_window": kwargs["window"]["time_window"],
            "weight_config": kwargs["weight_config"],
            "feasible_demands": kwargs["window"]["demands"],
            "n_demands_total": len(kwargs["window"]["demands"]),
            "n_demands_filtered": 0,
            "solution": {
                "solve_status": "heuristic_completed",
                "solve_time_s": 0.02,
                "objective_value": 1.0,
                "run_summary": run_summaries[idx],
                "per_demand_results": [],
                "per_drone_details": [],
                "drone_path_details": [],
                "analytics_artifacts": {},
            },
            "n_supply": 1,
        }

    monkeypatch.setattr(static_module, "solve_single_window_static_heuristic", fake_static_solver)

    payload = static_module.run_nsga3_static_heuristic_search(
        window={"time_window": "2024-03-15T00:00-00:05", "demands": [{"demand_id": "REQ001"}]},
        weight_config={
            "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
            "demand_configs": [{"demand_id": "REQ001", "priority": 1}],
            "supplementary_constraints": [],
        },
        stations_path="stations.csv",
        building_path="buildings.csv",
        max_solver_stations=1,
        max_payload=60.0,
        max_range=1000.0,
        noise_weight=0.5,
        drone_speed=60.0,
        output_dir=str(tmp_path / "search"),
        total_drones=3,
        drone_activation_cost=1000.0,
        pop_size=3,
        n_generations=2,
        seed=7,
        problem_id="static_case",
    )

    assert payload["problem_meta"]["problem_id"] == "static_case"
    assert payload["solver_meta"]["solver_name"] == "nsga3_static_heuristic"
    assert len(payload["candidates"]) == 3
    assert len(payload["frontier"]) == 2
    assert Path(payload["candidates_json"]).exists()
    assert Path(payload["frontier_json"]).exists()
    assert Path(payload["summary_json"]).exists()
    assert all(Path(item["frontier_result_path"]).exists() for item in payload["frontier"])

    saved = json.loads(Path(payload["frontier"][0]["frontier_result_path"]).read_text(encoding="utf-8"))
    assert saved[0]["time_window"] == "2024-03-15T00:00-00:05"
