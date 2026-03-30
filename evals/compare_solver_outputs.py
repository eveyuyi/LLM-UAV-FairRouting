"""Compare workflow outputs across CPLEX, NSGA-III, and NSGA-III heuristic backends."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from llm4fairrouting.routing.analytics import PARETO_METRICS


DEFAULT_METRICS = PARETO_METRICS


SEARCH_BACKENDS = {
    "nsga3": {
        "results_filename": "nsga3_results.json",
        "analytics_subdir": "solver_analytics/nsga3",
    },
    "nsga3_heuristic": {
        "results_filename": "nsga3_heuristic_results.json",
        "analytics_subdir": "solver_analytics/nsga3_heuristic",
    },
}


def _load_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_json_if_exists(path: str | Path | None) -> object | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return _load_json(candidate)


def _resolve_run_path(run_dir: str | Path, relative: str) -> Path:
    return Path(run_dir) / relative


def _first_run_summary(workflow_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    for item in workflow_results:
        summary = item.get("run_summary") or {}
        if summary:
            return dict(summary)
    return {}


def _workflow_path_summary(workflow_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    drone_paths = []
    for item in workflow_results:
        drone_paths.extend(item.get("drone_path_details", []))
    moving = [path for path in drone_paths if int(path.get("n_nodes_visited", 0)) > 1]
    return {
        "n_path_entries": len(drone_paths),
        "n_moving_paths": len(moving),
        "max_nodes_visited": max((int(path.get("n_nodes_visited", 0)) for path in drone_paths), default=0),
        "avg_nodes_visited": (
            sum(int(path.get("n_nodes_visited", 0)) for path in drone_paths) / len(drone_paths)
            if drone_paths else 0.0
        ),
        "has_any_moving_drone": bool(moving),
    }


def _frontier_solution_path_summary(frontier: Sequence[Dict[str, object]]) -> Dict[str, object]:
    checked = []
    for candidate in frontier:
        result_path = candidate.get("frontier_result_path")
        if not result_path or not Path(result_path).exists():
            checked.append({"solution_id": candidate.get("solution_id"), "exists": False})
            continue
        payload = _load_json(result_path)
        path_summary = _workflow_path_summary(payload if isinstance(payload, list) else [])
        checked.append(
            {
                "solution_id": candidate.get("solution_id"),
                "exists": True,
                **path_summary,
            }
        )
    return {
        "n_frontier_solutions": len(frontier),
        "n_existing_solution_files": sum(1 for item in checked if item.get("exists")),
        "solutions": checked,
        "all_solution_files_present": all(item.get("exists") for item in checked) if checked else True,
    }


def _dominates(left: Dict[str, object], right: Dict[str, object], metrics: Sequence[str]) -> bool:
    left_vals = []
    right_vals = []
    for metric in metrics:
        left_val = left.get(metric)
        right_val = right.get(metric)
        if left_val is None or right_val is None:
            return False
        left_vals.append(float(left_val))
        right_vals.append(float(right_val))
    return all(l <= r for l, r in zip(left_vals, right_vals)) and any(l < r for l, r in zip(left_vals, right_vals))


def _frontier_extremes(frontier: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    extremes: Dict[str, Dict[str, object]] = {}
    for metric in DEFAULT_METRICS:
        valid = [item for item in frontier if item.get(metric) is not None]
        if not valid:
            continue
        best = min(valid, key=lambda item: float(item.get(metric, math.inf)))
        extremes[metric] = {
            "solution_id": best.get("solution_id"),
            "value": float(best.get(metric, math.inf)),
            "frontier_result_path": best.get("frontier_result_path"),
        }
    service_valid = [item for item in frontier if (item.get("run_summary") or {}).get("service_rate") is not None]
    if service_valid:
        best_service = max(
            service_valid,
            key=lambda item: float((item.get("run_summary") or {}).get("service_rate", -math.inf)),
        )
        extremes["service_rate"] = {
            "solution_id": best_service.get("solution_id"),
            "value": float((best_service.get("run_summary") or {}).get("service_rate", 0.0)),
            "frontier_result_path": best_service.get("frontier_result_path"),
        }
    return extremes


def _summarize_search_backend(
    *,
    backend_name: str,
    run_dir: str | Path,
) -> Dict[str, object]:
    run_dir = Path(run_dir)
    config = SEARCH_BACKENDS[backend_name]
    payload = _load_json(_resolve_run_path(run_dir, config["results_filename"]))
    if not isinstance(payload, dict):
        raise ValueError(f"{backend_name} results payload must be a dict")
    frontier = list(payload.get("frontier", []))
    analytics_dir = _resolve_run_path(run_dir, config["analytics_subdir"])
    return {
        "run_dir": str(run_dir),
        "search_meta": payload.get("search_meta", {}),
        "frontier_size": len(frontier),
        "pareto_analysis": payload.get("pareto_analysis") or _load_json_if_exists(analytics_dir / "pareto_analysis.json"),
        "frontier_extremes": _frontier_extremes(frontier),
        "frontier_path_summary": _frontier_solution_path_summary(frontier),
        "frontier": frontier,
        "solver_meta": payload.get("solver_meta", {}),
    }


def _best_backend_by_metric(
    *,
    cplex_run_summary: Dict[str, object],
    search_summaries: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    rankings: Dict[str, Dict[str, object]] = {}
    for metric in DEFAULT_METRICS:
        candidates: List[tuple[str, float]] = []
        cplex_value = cplex_run_summary.get(metric)
        if cplex_value is not None:
            candidates.append(("cplex_main", float(cplex_value)))
        for backend_name, summary in search_summaries.items():
            extreme = (summary.get("frontier_extremes") or {}).get(metric) or {}
            value = extreme.get("value")
            if value is not None:
                candidates.append((backend_name, float(value)))
        if candidates:
            winner = min(candidates, key=lambda item: item[1])
            rankings[metric] = {"backend": winner[0], "value": winner[1]}

    service_candidates: List[tuple[str, float]] = []
    cplex_service = cplex_run_summary.get("service_rate")
    if cplex_service is not None:
        service_candidates.append(("cplex_main", float(cplex_service)))
    for backend_name, summary in search_summaries.items():
        extreme = (summary.get("frontier_extremes") or {}).get("service_rate") or {}
        value = extreme.get("value")
        if value is not None:
            service_candidates.append((backend_name, float(value)))
    if service_candidates:
        winner = max(service_candidates, key=lambda item: item[1])
        rankings["service_rate"] = {"backend": winner[0], "value": winner[1]}
    return rankings


def compare_solver_outputs(
    *,
    cplex_run_dir: str | Path,
    nsga3_run_dir: str | Path | None = None,
    nsga3_heuristic_run_dir: str | Path | None = None,
) -> Dict[str, object]:
    cplex_run_dir = Path(cplex_run_dir)
    cplex_workflow_results = _load_json(_resolve_run_path(cplex_run_dir, "workflow_results.json"))
    if not isinstance(cplex_workflow_results, list):
        raise ValueError("CPLEX workflow_results.json must be a list")

    cplex_run_summary = _first_run_summary(cplex_workflow_results)
    cplex_path_summary = _workflow_path_summary(cplex_workflow_results)
    cplex_pareto = _load_json_if_exists(_resolve_run_path(cplex_run_dir, "solver_analytics/pareto/pareto_frontier.json"))
    cplex_pareto_analysis = _load_json_if_exists(_resolve_run_path(cplex_run_dir, "solver_analytics/pareto/pareto_analysis.json"))

    payload: Dict[str, object] = {
        "cplex": {
            "run_dir": str(cplex_run_dir),
            "run_summary": cplex_run_summary,
            "path_summary": cplex_path_summary,
            "pareto_frontier": cplex_pareto,
            "pareto_analysis": cplex_pareto_analysis,
        }
    }

    search_summaries: Dict[str, Dict[str, object]] = {}
    if nsga3_run_dir is not None:
        search_summaries["nsga3"] = _summarize_search_backend(backend_name="nsga3", run_dir=nsga3_run_dir)
        payload["nsga3"] = {k: v for k, v in search_summaries["nsga3"].items() if k != "frontier"}
    if nsga3_heuristic_run_dir is not None:
        search_summaries["nsga3_heuristic"] = _summarize_search_backend(
            backend_name="nsga3_heuristic",
            run_dir=nsga3_heuristic_run_dir,
        )
        payload["nsga3_heuristic"] = {k: v for k, v in search_summaries["nsga3_heuristic"].items() if k != "frontier"}

    cplex_main_candidate = {metric: cplex_run_summary.get(metric) for metric in DEFAULT_METRICS}
    backend_comparisons: Dict[str, Dict[str, object]] = {}
    for backend_name, summary in search_summaries.items():
        frontier = list(summary.get("frontier", []))
        dominating = [
            item.get("solution_id")
            for item in frontier
            if _dominates(item, cplex_main_candidate, DEFAULT_METRICS)
        ]
        best_service = (summary.get("frontier_extremes") or {}).get("service_rate") or {}
        backend_comparisons[backend_name] = {
            "frontier_solution_ids": [item.get("solution_id") for item in frontier],
            "solutions_dominating_cplex_main": dominating,
            "best_service_rate": best_service.get("value"),
            "frontier_size": len(frontier),
        }

    payload["cross_solver_comparison"] = {
        "cplex_main_metrics": cplex_main_candidate,
        "cplex_service_rate": cplex_run_summary.get("service_rate"),
        "backend_comparisons": backend_comparisons,
        "best_backend_by_metric": _best_backend_by_metric(
            cplex_run_summary=cplex_run_summary,
            search_summaries=search_summaries,
        ),
    }

    if "nsga3" in search_summaries:
        payload["cross_solver_comparison"]["nsga3_frontier_solution_ids"] = backend_comparisons["nsga3"]["frontier_solution_ids"]
        payload["cross_solver_comparison"]["nsga3_solutions_dominating_cplex_main"] = backend_comparisons["nsga3"]["solutions_dominating_cplex_main"]
        payload["cross_solver_comparison"]["best_nsga3_service_rate"] = backend_comparisons["nsga3"]["best_service_rate"]
    if "nsga3_heuristic" in search_summaries:
        payload["cross_solver_comparison"]["nsga3_heuristic_frontier_solution_ids"] = backend_comparisons["nsga3_heuristic"]["frontier_solution_ids"]
        payload["cross_solver_comparison"]["nsga3_heuristic_solutions_dominating_cplex_main"] = backend_comparisons["nsga3_heuristic"]["solutions_dominating_cplex_main"]
        payload["cross_solver_comparison"]["best_nsga3_heuristic_service_rate"] = backend_comparisons["nsga3_heuristic"]["best_service_rate"]

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPLEX, NSGA-III, and NSGA-III heuristic solver outputs.")
    parser.add_argument("--cplex-run-dir", required=True, help="Run directory produced by run_workflow with --solver-backend cplex")
    parser.add_argument("--nsga3-run-dir", help="Run directory produced by run_workflow with --solver-backend nsga3")
    parser.add_argument("--nsga3-heuristic-run-dir", help="Run directory produced by run_workflow with --solver-backend nsga3_heuristic")
    parser.add_argument("--output", default="evals/results/solver_comparison.json", help="Output JSON path")
    args = parser.parse_args()

    payload = compare_solver_outputs(
        cplex_run_dir=args.cplex_run_dir,
        nsga3_run_dir=args.nsga3_run_dir,
        nsga3_heuristic_run_dir=args.nsga3_heuristic_run_dir,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Solver comparison saved to {output_path}")


if __name__ == "__main__":
    main()
