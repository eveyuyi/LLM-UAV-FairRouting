from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from uuid import uuid4


def _load_script_module():
    script_path = Path.cwd() / "scripts" / "run_20_window_comparison.py"
    spec = importlib.util.spec_from_file_location("run_20_window_comparison", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_case_dir(prefix: str) -> Path:
    path = Path.cwd() / ".test_artifacts" / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_collect_and_write_five_objective_csv():
    module = _load_script_module()
    base = _make_case_dir("five_objective_csv")
    cplex_run = base / "cplex_run"
    nsga3_run = base / "nsga3_run"
    heuristic_run = base / "heuristic_run"
    (cplex_run / "solver_analytics" / "pareto").mkdir(parents=True, exist_ok=True)
    (nsga3_run / "solver_analytics" / "nsga3").mkdir(parents=True, exist_ok=True)
    (heuristic_run / "solver_analytics" / "nsga3_heuristic").mkdir(parents=True, exist_ok=True)

    with open(cplex_run / "workflow_results.json", "w", encoding="utf-8") as handle:
        json.dump([
            {
                "run_summary": {
                    "final_total_distance_m": 120.0,
                    "average_delivery_time_h": 1.2,
                    "final_total_noise_impact": 12.0,
                    "service_rate": 0.8,
                    "service_rate_loss": 0.2,
                    "n_used_drones": 2,
                }
            }
        ], handle)
    with open(cplex_run / "solver_analytics" / "pareto" / "pareto_frontier.json", "w", encoding="utf-8") as handle:
        json.dump({
            "frontier": [
                {
                    "profile_id": "balanced",
                    "label": "Balanced",
                    "final_total_distance_m": 118.0,
                    "average_delivery_time_h": 1.1,
                    "final_total_noise_impact": 11.0,
                    "service_rate": 0.82,
                    "service_rate_loss": 0.18,
                    "n_used_drones": 2,
                }
            ]
        }, handle)

    with open(nsga3_run / "nsga3_results.json", "w", encoding="utf-8") as handle:
        json.dump({
            "search_meta": {"search_runtime_s": 10.0, "avg_candidate_runtime_s": 2.5},
            "frontier": [
                {
                    "solution_id": "nsga3_a",
                    "label": "NSGA3 A",
                    "frontier_result_path": "a.json",
                    "final_total_distance_m": 100.0,
                    "average_delivery_time_h": 1.0,
                    "final_total_noise_impact": 10.0,
                    "service_rate_loss": 0.1,
                    "n_used_drones": 1,
                    "run_summary": {"service_rate": 0.9},
                },
                {
                    "solution_id": "nsga3_b",
                    "label": "NSGA3 B",
                    "frontier_result_path": "b.json",
                    "final_total_distance_m": 130.0,
                    "average_delivery_time_h": 0.8,
                    "final_total_noise_impact": 9.0,
                    "service_rate_loss": 0.12,
                    "n_used_drones": 2,
                    "run_summary": {"service_rate": 0.88},
                },
            ],
        }, handle)

    with open(heuristic_run / "nsga3_heuristic_results.json", "w", encoding="utf-8") as handle:
        json.dump({
            "search_meta": {"search_runtime_s": 4.0, "avg_candidate_runtime_s": 1.0},
            "frontier": [
                {
                    "solution_id": "heur_a",
                    "label": "Heur A",
                    "frontier_result_path": "h.json",
                    "final_total_distance_m": 110.0,
                    "average_delivery_time_h": 0.95,
                    "final_total_noise_impact": 9.5,
                    "service_rate_loss": 0.11,
                    "n_used_drones": 1,
                    "run_summary": {"service_rate": 0.89},
                }
            ],
        }, handle)

    rows = module._collect_five_objective_rows(
        cplex_run_dir=cplex_run,
        nsga3_run_dir=nsga3_run,
        nsga3_heuristic_run_dir=heuristic_run,
    )
    assert any(row["backend"] == "cplex" and row["solution_scope"] == "main_solution" for row in rows)
    assert any(row["backend"] == "nsga3" and row["is_representative"] for row in rows)
    assert any(row["backend"] == "nsga3_heuristic" for row in rows)

    csv_path = base / "five_objectives_raw.csv"
    module._write_five_objective_csv(rows, csv_path)
    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        data = list(csv.DictReader(handle))
    assert len(data) == len(rows)
    assert {row["backend"] for row in data} == {"cplex", "nsga3", "nsga3_heuristic"}
