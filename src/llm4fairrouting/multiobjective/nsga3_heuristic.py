"""NSGA-III outer-loop search backed by a greedy heuristic assignment solver."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from llm4fairrouting.multiobjective.nsga3_search import (
    _SearchBounds,
    _candidate_objectives,
    _clone_weight_configs_with_weights,
    _decode_candidate,
    _evaluation_cache_key,
    _first_run_summary,
    _problem_metadata,
    _public_candidate_view,
)
from llm4fairrouting.routing.analytics import (
    analyze_pareto_candidates,
    compute_pareto_frontier,
    export_visualizations,
    sanitize_label,
    write_json,
)
from llm4fairrouting.routing.heuristic_assignment import HeuristicAssignmentSolver
from llm4fairrouting.workflow.solver_adapter import (
    serialize_workflow_results,
    solve_windows_dynamically,
)


def _solver_metadata() -> Dict[str, object]:
    return {
        "solver_family": "evolutionary",
        "solver_name": "nsga3_heuristic",
        "inner_solver": "greedy_heuristic",
    }


def run_nsga3_heuristic_search(
    *,
    windows: List[Dict],
    weight_configs: Dict[str, Dict],
    stations_path: Optional[str],
    building_path: Optional[str],
    max_solver_stations: Optional[int],
    time_limit: int,
    max_drones_per_station: int,
    max_payload: float,
    max_range: float,
    noise_weight: float,
    drone_speed: float,
    output_dir: str,
    drone_activation_cost: float = 10000.0,
    pop_size: int = 20,
    n_generations: int = 10,
    seed: int = 42,
    save_all_candidate_results: bool = False,
    enable_conflict_refiner: bool = False,
    problem_id: Optional[str] = None,
) -> Dict[str, object]:
    try:
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
        from pymoo.util.ref_dirs import get_reference_directions
    except ImportError as exc:
        raise RuntimeError(
            "NSGA-III heuristic backend requires the optional 'pymoo' dependency."
        ) from exc

    bounds = _SearchBounds(
        activation_cost_min=min(100.0, float(drone_activation_cost)),
        activation_cost_max=max(5000.0, float(drone_activation_cost)),
    )
    search_output_dir = Path(output_dir)
    search_output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = search_output_dir / "candidates"
    frontier_dir = search_output_dir / "frontier_solutions"
    frontier_dir.mkdir(parents=True, exist_ok=True)
    if save_all_candidate_results:
        candidates_dir.mkdir(parents=True, exist_ok=True)

    resolved_problem_id = problem_id or f"nsga3_heuristic_{len(windows)}windows"
    problem_meta = _problem_metadata(
        problem_id=resolved_problem_id,
        windows=windows,
        stations_path=stations_path,
        building_path=building_path,
        time_limit=time_limit,
        max_solver_stations=max_solver_stations,
        max_drones_per_station=max_drones_per_station,
        max_payload=max_payload,
        max_range=max_range,
        noise_weight=noise_weight,
        drone_speed=drone_speed,
    )
    solver_meta = _solver_metadata()
    cache: Dict[tuple[float, ...], Dict[str, object]] = {}
    evaluated_candidates: List[Dict[str, object]] = []

    def evaluate_candidate(x: Sequence[float]) -> Dict[str, object]:
        cache_key = _evaluation_cache_key(x)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        decoded = _decode_candidate(x, bounds)
        candidate_index = len(evaluated_candidates)
        solution_id = f"nsga3_heuristic_candidate_{candidate_index:04d}"
        candidate_dir = candidates_dir / sanitize_label(solution_id)
        results = solve_windows_dynamically(
            windows=windows,
            weight_configs=_clone_weight_configs_with_weights(weight_configs, decoded["weights"]),
            stations_path=stations_path,
            building_path=building_path,
            max_solver_stations=max_solver_stations,
            time_limit=time_limit,
            max_drones_per_station=max_drones_per_station,
            max_payload=max_payload,
            max_range=max_range,
            noise_weight=noise_weight,
            drone_activation_cost=float(decoded["drone_activation_cost"]),
            drone_speed=drone_speed,
            analytics_output_dir=str(candidate_dir / "analytics") if save_all_candidate_results else None,
            enable_conflict_refiner=enable_conflict_refiner,
            assignment_solver_factory=HeuristicAssignmentSolver,
        )
        run_summary = _first_run_summary(results)
        objectives = _candidate_objectives(run_summary)
        serialized_results = serialize_workflow_results(results)
        candidate: Dict[str, object] = {
            "solution_id": solution_id,
            "profile_id": solution_id,
            "label": solution_id,
            "decision_vector": [float(item) for item in x],
            "weights": decoded["weights"],
            "drone_activation_cost": float(decoded["drone_activation_cost"]),
            "objectives": objectives,
            **objectives,
            "run_summary": run_summary,
            "problem_meta": problem_meta,
            "solver_meta": solver_meta,
            "_serialized_workflow_results": serialized_results,
        }
        if save_all_candidate_results:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            result_path = candidate_dir / "workflow_results.json"
            with open(result_path, "w", encoding="utf-8") as handle:
                json.dump(serialized_results, handle, ensure_ascii=False, indent=2)
            candidate["result_path"] = str(result_path)

        cache[cache_key] = candidate
        evaluated_candidates.append(candidate)
        return candidate

    class _WorkflowNSGA3HeuristicProblem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(n_var=4, n_obj=5, xl=0.0, xu=1.0)

        def _evaluate(self, x, out, *args, **kwargs):
            candidate = evaluate_candidate(x)
            objectives = candidate["objectives"]
            out["F"] = [
                objectives["final_total_distance_m"],
                objectives["average_delivery_time_h"],
                objectives["final_total_noise_impact"],
                objectives["service_rate_loss"],
                objectives["n_used_drones"],
            ]

    problem = _WorkflowNSGA3HeuristicProblem()
    ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=3)
    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    search_started_at = time.perf_counter()
    minimize(
        problem,
        algorithm,
        termination=("n_gen", n_generations),
        seed=seed,
        verbose=False,
    )

    search_runtime_s = round(time.perf_counter() - search_started_at, 6)
    avg_candidate_runtime_s = round(search_runtime_s / len(evaluated_candidates), 6) if evaluated_candidates else None
    print(
        f"[NSGA-III Heuristic] Completed {len(evaluated_candidates)} candidates in {search_runtime_s:.3f}s "
        f"(avg {avg_candidate_runtime_s:.3f}s/candidate) with frontier size pending evaluation"
    )

    frontier = compute_pareto_frontier(evaluated_candidates)
    frontier_ids = {item["solution_id"] for item in frontier}
    for candidate in evaluated_candidates:
        candidate["is_nondominated"] = candidate["solution_id"] in frontier_ids
        if not candidate["is_nondominated"]:
            continue
        solution_dir = frontier_dir / sanitize_label(str(candidate["solution_id"]))
        solution_dir.mkdir(parents=True, exist_ok=True)
        result_path = solution_dir / "workflow_results.json"
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(candidate["_serialized_workflow_results"], handle, ensure_ascii=False, indent=2)
        candidate["frontier_result_path"] = str(result_path)

    public_candidates = [_public_candidate_view(candidate) for candidate in evaluated_candidates]
    public_frontier = [
        _public_candidate_view(candidate)
        for candidate in evaluated_candidates
        if candidate["is_nondominated"]
    ]
    payload: Dict[str, object] = {
        "problem_meta": problem_meta,
        "solver_meta": solver_meta,
        "search_meta": {
            "pop_size": pop_size,
            "n_generations": n_generations,
            "seed": seed,
            "n_candidates_evaluated": len(evaluated_candidates),
            "search_runtime_s": search_runtime_s,
            "avg_candidate_runtime_s": avg_candidate_runtime_s,
        },
        "candidates": public_candidates,
        "frontier": public_frontier,
    }
    payload["candidates_json"] = write_json(
        payload["candidates"],
        search_output_dir / "nsga3_heuristic_candidates.json",
    )
    payload["frontier_json"] = write_json(
        payload["frontier"],
        search_output_dir / "nsga3_heuristic_pareto_frontier.json",
    )
    pareto_analysis = analyze_pareto_candidates(public_candidates)
    payload["pareto_analysis"] = pareto_analysis
    payload["summary_json"] = write_json(
        {
            "problem_meta": problem_meta,
            "solver_meta": solver_meta,
            "search_meta": payload["search_meta"],
            "frontier_size": len(public_frontier),
        },
        search_output_dir / "nsga3_heuristic_summary.json",
    )
    payload["pareto_analysis_json"] = write_json(
        pareto_analysis,
        search_output_dir / "pareto_analysis.json",
    )
    print(
        f"[NSGA-III Heuristic] Frontier size {len(public_frontier)} saved under {search_output_dir} "
        f"after {search_runtime_s:.3f}s"
    )
    payload["charts"] = export_visualizations(
        analytics={"solver_calls": [], "gantt_tasks": []},
        output_dir=search_output_dir / "charts",
        pareto_candidates=public_candidates,
    )
    return payload
