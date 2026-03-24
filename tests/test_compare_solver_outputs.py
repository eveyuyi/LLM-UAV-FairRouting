from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from evals.compare_solver_outputs import compare_solver_outputs



def _make_case_dir(prefix: str) -> Path:
    path = Path.cwd() / ".test_artifacts" / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path



def _write_frontier_solution(solution_dir: Path, n_nodes_visited: int) -> None:
    solution_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "time_window": "2024-03-15T00:00-00:05",
            "drone_path_details": [
                {
                    "drone_id": "U12",
                    "n_nodes_visited": n_nodes_visited,
                    "path_node_ids": ["L1", "S1", "D1", "L1"],
                }
            ],
        }
    ]
    with open(solution_dir / "workflow_results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)



def test_compare_solver_outputs_summarizes_three_solver_metrics():
    base = _make_case_dir("compare_solver_outputs")
    cplex_run = base / "cplex_run"
    nsga3_run = base / "nsga3_run"
    heuristic_run = base / "nsga3_heuristic_run"
    (cplex_run / "solver_analytics" / "pareto").mkdir(parents=True, exist_ok=True)
    (nsga3_run / "solver_analytics" / "nsga3").mkdir(parents=True, exist_ok=True)
    (heuristic_run / "solver_analytics" / "nsga3_heuristic").mkdir(parents=True, exist_ok=True)

    cplex_workflow = [
        {
            "time_window": "2024-03-15T00:00-00:05",
            "run_summary": {
                "final_total_distance_m": 120.0,
                "average_delivery_time_h": 1.2,
                "final_total_noise_impact": 12.0,
                "service_rate": 0.8,
                "service_rate_loss": 0.2,
                "n_used_drones": 2,
            },
            "drone_path_details": [
                {"drone_id": "U11", "n_nodes_visited": 4, "path_node_ids": ["L1", "S1", "D1", "L1"]}
            ],
        }
    ]
    with open(cplex_run / "workflow_results.json", "w", encoding="utf-8") as handle:
        json.dump(cplex_workflow, handle)
    with open(cplex_run / "solver_analytics" / "pareto" / "pareto_analysis.json", "w", encoding="utf-8") as handle:
        json.dump({"frontier_size": 2}, handle)
    with open(cplex_run / "solver_analytics" / "pareto" / "pareto_frontier.json", "w", encoding="utf-8") as handle:
        json.dump({"frontier": []}, handle)

    nsga3_solution_dir = nsga3_run / "solver_analytics" / "nsga3" / "frontier_solutions" / "sol1"
    _write_frontier_solution(nsga3_solution_dir, n_nodes_visited=5)
    nsga3_payload = {
        "search_meta": {"n_candidates_evaluated": 3},
        "pareto_analysis": {"frontier_size": 1},
        "frontier": [
            {
                "solution_id": "sol1",
                "final_total_distance_m": 100.0,
                "average_delivery_time_h": 1.0,
                "final_total_noise_impact": 10.0,
                "service_rate_loss": 0.1,
                "n_used_drones": 1,
                "run_summary": {"service_rate": 0.9, "service_rate_loss": 0.1, "n_used_drones": 1},
                "frontier_result_path": str(nsga3_solution_dir / "workflow_results.json"),
            }
        ],
    }
    with open(nsga3_run / "nsga3_results.json", "w", encoding="utf-8") as handle:
        json.dump(nsga3_payload, handle)

    heuristic_solution_dir = heuristic_run / "solver_analytics" / "nsga3_heuristic" / "frontier_solutions" / "heur1"
    _write_frontier_solution(heuristic_solution_dir, n_nodes_visited=6)
    heuristic_payload = {
        "search_meta": {"n_candidates_evaluated": 4},
        "pareto_analysis": {"frontier_size": 1},
        "frontier": [
            {
                "solution_id": "heur1",
                "final_total_distance_m": 110.0,
                "average_delivery_time_h": 0.95,
                "final_total_noise_impact": 9.5,
                "service_rate_loss": 0.12,
                "n_used_drones": 1,
                "run_summary": {"service_rate": 0.88, "service_rate_loss": 0.12, "n_used_drones": 1},
                "frontier_result_path": str(heuristic_solution_dir / "workflow_results.json"),
            }
        ],
    }
    with open(heuristic_run / "nsga3_heuristic_results.json", "w", encoding="utf-8") as handle:
        json.dump(heuristic_payload, handle)

    payload = compare_solver_outputs(
        cplex_run_dir=cplex_run,
        nsga3_run_dir=nsga3_run,
        nsga3_heuristic_run_dir=heuristic_run,
    )

    assert payload["cplex"]["run_summary"]["service_rate"] == 0.8
    assert payload["nsga3"]["frontier_size"] == 1
    assert payload["nsga3_heuristic"]["frontier_size"] == 1
    assert payload["nsga3"]["frontier_path_summary"]["all_solution_files_present"] is True
    assert payload["nsga3_heuristic"]["frontier_path_summary"]["all_solution_files_present"] is True
    assert payload["cross_solver_comparison"]["nsga3_solutions_dominating_cplex_main"] == ["sol1"]
    assert payload["cross_solver_comparison"]["nsga3_heuristic_solutions_dominating_cplex_main"] == ["heur1"]
    assert payload["cross_solver_comparison"]["best_backend_by_metric"]["service_rate"]["backend"] == "nsga3"



def test_compare_solver_outputs_remains_compatible_with_two_solver_inputs():
    base = _make_case_dir("compare_solver_outputs_dual")
    cplex_run = base / "cplex_run"
    nsga3_run = base / "nsga3_run"
    (cplex_run / "solver_analytics" / "pareto").mkdir(parents=True, exist_ok=True)
    (nsga3_run / "solver_analytics" / "nsga3").mkdir(parents=True, exist_ok=True)
    with open(cplex_run / "workflow_results.json", "w", encoding="utf-8") as handle:
        json.dump([
            {
                "run_summary": {
                    "final_total_distance_m": 50.0,
                    "average_delivery_time_h": 0.5,
                    "final_total_noise_impact": 5.0,
                    "service_rate": 0.7,
                    "service_rate_loss": 0.3,
                    "n_used_drones": 1,
                },
                "drone_path_details": [],
            }
        ], handle)
    with open(nsga3_run / "nsga3_results.json", "w", encoding="utf-8") as handle:
        json.dump({"frontier": [], "search_meta": {}}, handle)

    payload = compare_solver_outputs(cplex_run_dir=cplex_run, nsga3_run_dir=nsga3_run)

    assert "nsga3" in payload
    assert "nsga3_heuristic" not in payload
    assert payload["cross_solver_comparison"]["backend_comparisons"]["nsga3"]["frontier_size"] == 0
