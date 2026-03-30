from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from uuid import uuid4

from llm4fairrouting.multiobjective import nsga3_heuristic as nsga3_heuristic_module
from llm4fairrouting.multiobjective import nsga3_search as nsga3_module
from llm4fairrouting.workflow import solver_adapter as solver_adapter_module


def _make_case_dir(prefix: str) -> Path:
    path = Path.cwd() / ".test_artifacts" / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def test_run_nsga3_pareto_search_saves_frontier_results(monkeypatch):
    candidate_vectors = [
        [0.1, 0.1, 0.1, 0.1],
        [0.8, 0.8, 0.8, 0.8],
        [0.4, 0.2, 0.7, 0.3],
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
            "final_total_distance_m": 150.0,
            "average_delivery_time_h": 1.5,
            "final_total_noise_impact": 15.0,
            "service_rate": 0.8,
            "service_rate_loss": 0.2,
            "n_used_drones": 2,
        },
        {
            "final_total_distance_m": 90.0,
            "average_delivery_time_h": 1.3,
            "final_total_noise_impact": 12.0,
            "service_rate": 0.85,
            "service_rate_loss": 0.15,
            "n_used_drones": 1,
        },
    ]
    counter = {"value": 0}

    def fake_solve_windows_dynamically(**kwargs):
        idx = counter["value"]
        counter["value"] += 1
        return [
            {
                "time_window": kwargs["windows"][0]["time_window"],
                "weight_config": kwargs["weight_configs"][kwargs["windows"][0]["time_window"]],
                "feasible_demands": kwargs["windows"][0]["demands"],
                "n_demands_total": len(kwargs["windows"][0]["demands"]),
                "n_demands_filtered": 0,
                "solution": {
                    "solve_mode": "dynamic_periodic",
                    "solve_status": "completed",
                    "solve_time_s": 0.1,
                    "drone_speed_ms": kwargs["drone_speed"],
                    "snapshot_time_h": 0.0,
                    "snapshot_time_window_end": kwargs["windows"][0]["time_window"],
                    "busy_drones": [],
                    "total_distance": run_summaries[idx]["final_total_distance_m"],
                    "total_noise_impact": run_summaries[idx]["final_total_noise_impact"],
                    "objective_value": 1.0,
                    "run_summary": run_summaries[idx],
                    "analytics_artifacts": {},
                    "demand_event_results": {},
                    "drone_path_details": [],
                },
                "n_supply": 1,
            }
        ]

    monkeypatch.setattr(nsga3_module, "solve_windows_dynamically", fake_solve_windows_dynamically)

    windows = [{"time_window": "2024-03-15T00:00-00:05", "demands": [{"demand_id": "REQ001"}]}]
    weight_configs = {
        "2024-03-15T00:00-00:05": {
            "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
            "demand_configs": [{"demand_id": "REQ001", "priority": 1}],
            "supplementary_constraints": [],
        }
    }

    output_dir = _make_case_dir("nsga3_search")
    payload = nsga3_module.run_nsga3_pareto_search(
            windows=windows,
            weight_configs=weight_configs,
            stations_path="stations.csv",
            building_path="buildings.csv",
            max_solver_stations=1,
            time_limit=5,
            max_drones_per_station=2,
            max_payload=60.0,
            max_range=1000.0,
            noise_weight=0.5,
            drone_speed=60.0,
            output_dir=str(output_dir),
            n_generations=2,
            pop_size=3,
            seed=7,
            problem_id="adapter_case",
        )

    assert payload["problem_meta"]["problem_id"] == "adapter_case"
    assert payload["solver_meta"]["solver_name"] == "nsga3"
    assert len(payload["candidates"]) == 3
    assert len(payload["frontier"]) == 2
    assert Path(payload["candidates_json"]).exists()
    assert Path(payload["frontier_json"]).exists()
    assert Path(payload["summary_json"]).exists()
    chart_types = {item["chart_type"] for item in payload["charts"]}
    assert "pareto_parallel_coordinates" in chart_types

    frontier_paths = [Path(item["frontier_result_path"]) for item in payload["frontier"]]
    assert all(path.exists() for path in frontier_paths)
    saved_frontier = json.loads(frontier_paths[0].read_text(encoding="utf-8"))
    assert saved_frontier[0]["time_window"] == "2024-03-15T00:00-00:05"
    assert all("_serialized_workflow_results" not in item for item in payload["candidates"])


def test_solver_adapter_cli_routes_to_nsga3(monkeypatch):
    captured = {}

    def fake_nsga3(**kwargs):
        captured.update(kwargs)
        return {"frontier": [{"solution_id": "nsga3_case"}]}

    monkeypatch.setattr(solver_adapter_module, "_load_weight_configs", lambda source: {"W": {"global_weights": {}}})
    monkeypatch.setattr(solver_adapter_module.json, "load", lambda handle: [{"time_window": "W", "demands": []}])
    monkeypatch.setattr(solver_adapter_module, "open", open, raising=False)
    monkeypatch.setitem(sys.modules, "llm4fairrouting.multiobjective.nsga3_search", types.SimpleNamespace(run_nsga3_pareto_search=fake_nsga3))

    argv = sys.argv[:]
    tmpdir = _make_case_dir("solver_adapter_cli")
    demands = tmpdir / "demands.json"
    weights = tmpdir / "weights.json"
    output = tmpdir / "solver_results.json"
    demands.write_text("[]", encoding="utf-8")
    weights.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        solver_adapter_module.Path,
        "mkdir",
        lambda self, parents=False, exist_ok=False: None,
        raising=False,
    )
    sys.argv = [
            "solver_adapter.py",
            "--demands", str(demands),
            "--weights", str(weights),
            "--output", str(output),
            "--solver-backend", "nsga3",
            "--nsga3-pop-size", "6",
            "--nsga3-n-generations", "3",
            "--nsga3-seed", "99",
        ]
    try:
        solver_adapter_module.main()
    finally:
        sys.argv = argv

    assert captured["pop_size"] == 6
    assert captured["n_generations"] == 3
    assert captured["seed"] == 99
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["frontier"][0]["solution_id"] == "nsga3_case"


def test_run_nsga3_heuristic_search_saves_frontier_results(monkeypatch):
    candidate_vectors = [
        [0.2, 0.2, 0.2, 0.2],
        [0.6, 0.3, 0.8, 0.5],
        [0.9, 0.9, 0.9, 0.9],
    ]
    _install_fake_pymoo(monkeypatch, candidate_vectors)

    run_summaries = [
        {
            "final_total_distance_m": 70.0,
            "average_delivery_time_h": 0.8,
            "final_total_noise_impact": 8.0,
            "service_rate": 0.95,
            "service_rate_loss": 0.05,
            "n_used_drones": 1,
        },
        {
            "final_total_distance_m": 85.0,
            "average_delivery_time_h": 0.7,
            "final_total_noise_impact": 7.5,
            "service_rate": 0.9,
            "service_rate_loss": 0.1,
            "n_used_drones": 2,
        },
        {
            "final_total_distance_m": 95.0,
            "average_delivery_time_h": 1.0,
            "final_total_noise_impact": 9.0,
            "service_rate": 0.85,
            "service_rate_loss": 0.15,
            "n_used_drones": 1,
        },
    ]
    counter = {"value": 0}

    def fake_solve_windows_dynamically(**kwargs):
        idx = counter["value"]
        counter["value"] += 1
        return [
            {
                "time_window": kwargs["windows"][0]["time_window"],
                "weight_config": kwargs["weight_configs"][kwargs["windows"][0]["time_window"]],
                "feasible_demands": kwargs["windows"][0]["demands"],
                "n_demands_total": len(kwargs["windows"][0]["demands"]),
                "n_demands_filtered": 0,
                "solution": {
                    "solve_mode": "dynamic_periodic",
                    "solve_status": "completed",
                    "solve_time_s": 0.02,
                    "drone_speed_ms": kwargs["drone_speed"],
                    "snapshot_time_h": 0.0,
                    "snapshot_time_window_end": kwargs["windows"][0]["time_window"],
                    "busy_drones": [],
                    "total_distance": run_summaries[idx]["final_total_distance_m"],
                    "total_noise_impact": run_summaries[idx]["final_total_noise_impact"],
                    "objective_value": 1.0,
                    "run_summary": run_summaries[idx],
                    "analytics_artifacts": {},
                    "demand_event_results": {},
                    "drone_path_details": [],
                },
                "n_supply": 1,
            }
        ]

    monkeypatch.setattr(nsga3_heuristic_module, "solve_windows_dynamically", fake_solve_windows_dynamically)

    windows = [{"time_window": "2024-03-15T00:00-00:05", "demands": [{"demand_id": "REQ001"}]}]
    weight_configs = {
        "2024-03-15T00:00-00:05": {
            "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
            "demand_configs": [{"demand_id": "REQ001", "priority": 1}],
            "supplementary_constraints": [],
        }
    }

    output_dir = _make_case_dir("nsga3_heuristic_search")
    payload = nsga3_heuristic_module.run_nsga3_heuristic_search(
        windows=windows,
        weight_configs=weight_configs,
        stations_path="stations.csv",
        building_path="buildings.csv",
        max_solver_stations=1,
        time_limit=5,
        max_drones_per_station=2,
        max_payload=60.0,
        max_range=1000.0,
        noise_weight=0.5,
        drone_speed=60.0,
        output_dir=str(output_dir),
        n_generations=2,
        pop_size=3,
        seed=11,
        problem_id="heuristic_case",
    )

    assert payload["problem_meta"]["problem_id"] == "heuristic_case"
    assert payload["solver_meta"]["solver_name"] == "nsga3_heuristic"
    assert len(payload["candidates"]) == 3
    assert len(payload["frontier"]) >= 1
    assert Path(payload["candidates_json"]).exists()
    assert Path(payload["frontier_json"]).exists()
    assert Path(payload["summary_json"]).exists()
    chart_types = {item["chart_type"] for item in payload["charts"]}
    assert "pareto_parallel_coordinates" in chart_types
    assert all(Path(item["frontier_result_path"]).exists() for item in payload["frontier"])


def test_solver_adapter_cli_routes_to_nsga3_heuristic(monkeypatch):
    captured = {}

    def fake_nsga3_heuristic(**kwargs):
        captured.update(kwargs)
        return {"frontier": [{"solution_id": "heuristic_case"}]}

    monkeypatch.setattr(solver_adapter_module, "_load_weight_configs", lambda source: {"W": {"global_weights": {}}})
    monkeypatch.setattr(solver_adapter_module.json, "load", lambda handle: [{"time_window": "W", "demands": []}])
    monkeypatch.setattr(solver_adapter_module, "open", open, raising=False)
    monkeypatch.setitem(sys.modules, "llm4fairrouting.multiobjective.nsga3_heuristic", types.SimpleNamespace(run_nsga3_heuristic_search=fake_nsga3_heuristic))

    argv = sys.argv[:]
    tmpdir = _make_case_dir("solver_adapter_cli_heuristic")
    demands = tmpdir / "demands.json"
    weights = tmpdir / "weights.json"
    output = tmpdir / "solver_results.json"
    demands.write_text("[]", encoding="utf-8")
    weights.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        solver_adapter_module.Path,
        "mkdir",
        lambda self, parents=False, exist_ok=False: None,
        raising=False,
    )
    sys.argv = [
        "solver_adapter.py",
        "--demands", str(demands),
        "--weights", str(weights),
        "--output", str(output),
        "--solver-backend", "nsga3_heuristic",
        "--nsga3-pop-size", "5",
        "--nsga3-n-generations", "2",
        "--nsga3-seed", "7",
    ]
    try:
        solver_adapter_module.main()
    finally:
        sys.argv = argv

    assert captured["pop_size"] == 5
    assert captured["n_generations"] == 2
    assert captured["seed"] == 7
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["frontier"][0]["solution_id"] == "heuristic_case"
