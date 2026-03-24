"""Run a compact comparison across CPLEX, NSGA-III, NSGA-III heuristic, and priority evaluation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4fairrouting.routing.rrt_visualization import (
    export_rrt_3d_from_workflow_results,
    select_representative_frontier_solution,
)

DEFAULT_SLOTS = [0]


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) if not existing else str(SRC) + os.pathsep + existing
    return env


def _run(command: List[str], *, env: dict[str, str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _latest_run_dir(base_dir: Path) -> Path:
    candidates = [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("run_")]
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found under {base_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _time_slot_args(time_slots: List[int]) -> List[str]:
    return [str(slot) for slot in time_slots]


def _load_json(path: Path) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_search_runtime(results_path: Path) -> dict[str, float] | None:
    if not results_path.exists():
        return None
    with open(results_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    search_meta = payload.get("search_meta") or {}
    runtime_s = search_meta.get("search_runtime_s")
    avg_runtime_s = search_meta.get("avg_candidate_runtime_s")
    if runtime_s is None:
        return None
    result = {"search_runtime_s": float(runtime_s)}
    if avg_runtime_s is not None:
        result["avg_candidate_runtime_s"] = float(avg_runtime_s)
    return result


def _require_optional_dependency(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    raise SystemExit(
        f"Missing optional dependency '{module_name}'. "
        f"Install it first with: {install_hint}"
    )


def _export_search_representative_rrt_chart(
    *,
    results_json: Path,
    output_path: Path,
    topdown_output_path: Path,
    title_prefix: str,
) -> dict[str, object] | None:
    payload = _load_json(results_json)
    if not isinstance(payload, dict):
        return None
    frontier = list(payload.get("frontier", []))
    representative = select_representative_frontier_solution(frontier)
    if representative is None:
        return None
    workflow_results_path = representative.get("frontier_result_path")
    if not workflow_results_path:
        return None
    title = f"{title_prefix} Representative RRT 3D: {representative.get('solution_id', 'unknown')}"
    export_payload = export_rrt_3d_from_workflow_results(
        workflow_results_path,
        output_path,
        title=title,
        topdown_output_path=topdown_output_path,
    )
    if export_payload is None:
        return None
    export_payload["solution_id"] = representative.get("solution_id")
    return export_payload


def _first_run_summary_from_workflow(workflow_results_path: Path) -> dict[str, object]:
    payload = _load_json(workflow_results_path)
    if not isinstance(payload, list):
        return {}
    for item in payload:
        summary = item.get("run_summary") or {}
        if summary:
            return dict(summary)
    return {}


def _collect_five_objective_rows(
    *,
    cplex_run_dir: Path,
    nsga3_run_dir: Path,
    nsga3_heuristic_run_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    cplex_summary = _first_run_summary_from_workflow(cplex_run_dir / "workflow_results.json")
    if cplex_summary:
        rows.append({
            "backend": "cplex",
            "solution_scope": "main_solution",
            "solution_id": "cplex_main",
            "label": "CPLEX Main",
            "is_representative": True,
            "source_json": str(cplex_run_dir / "workflow_results.json"),
            "frontier_result_path": "",
            "search_runtime_s": "",
            "avg_candidate_runtime_s": "",
            "final_total_distance_m": cplex_summary.get("final_total_distance_m"),
            "average_delivery_time_h": cplex_summary.get("average_delivery_time_h"),
            "final_total_noise_impact": cplex_summary.get("final_total_noise_impact"),
            "service_rate": cplex_summary.get("service_rate"),
            "service_rate_loss": cplex_summary.get("service_rate_loss"),
            "n_used_drones": cplex_summary.get("n_used_drones"),
        })

    cplex_pareto_path = cplex_run_dir / "solver_analytics" / "pareto" / "pareto_frontier.json"
    if cplex_pareto_path.exists():
        cplex_pareto = _load_json(cplex_pareto_path)
        if isinstance(cplex_pareto, dict):
            for item in list(cplex_pareto.get("frontier", [])):
                rows.append({
                    "backend": "cplex",
                    "solution_scope": "pareto_profile",
                    "solution_id": item.get("profile_id") or item.get("label") or "cplex_profile",
                    "label": item.get("label") or item.get("profile_id") or "CPLEX Pareto",
                    "is_representative": False,
                    "source_json": str(cplex_pareto_path),
                    "frontier_result_path": item.get("analytics_dir", ""),
                    "search_runtime_s": "",
                    "avg_candidate_runtime_s": "",
                    "final_total_distance_m": item.get("final_total_distance_m"),
                    "average_delivery_time_h": item.get("average_delivery_time_h"),
                    "final_total_noise_impact": item.get("final_total_noise_impact"),
                    "service_rate": item.get("service_rate"),
                    "service_rate_loss": item.get("service_rate_loss"),
                    "n_used_drones": item.get("n_used_drones"),
                })

    for backend_name, run_dir, results_name in (
        ("nsga3", nsga3_run_dir, "nsga3_results.json"),
        ("nsga3_heuristic", nsga3_heuristic_run_dir, "nsga3_heuristic_results.json"),
    ):
        results_path = run_dir / results_name
        payload = _load_json(results_path)
        if not isinstance(payload, dict):
            continue
        frontier = list(payload.get("frontier", []))
        representative = select_representative_frontier_solution(frontier)
        representative_id = representative.get("solution_id") if representative else None
        search_meta = payload.get("search_meta") or {}
        for item in frontier:
            run_summary = item.get("run_summary") or {}
            rows.append({
                "backend": backend_name,
                "solution_scope": "frontier_solution",
                "solution_id": item.get("solution_id") or item.get("profile_id") or backend_name,
                "label": item.get("label") or item.get("solution_id") or backend_name,
                "is_representative": item.get("solution_id") == representative_id,
                "source_json": str(results_path),
                "frontier_result_path": item.get("frontier_result_path", ""),
                "search_runtime_s": search_meta.get("search_runtime_s", ""),
                "avg_candidate_runtime_s": search_meta.get("avg_candidate_runtime_s", ""),
                "final_total_distance_m": item.get("final_total_distance_m"),
                "average_delivery_time_h": item.get("average_delivery_time_h"),
                "final_total_noise_impact": item.get("final_total_noise_impact"),
                "service_rate": run_summary.get("service_rate"),
                "service_rate_loss": item.get("service_rate_loss", run_summary.get("service_rate_loss")),
                "n_used_drones": item.get("n_used_drones", run_summary.get("n_used_drones")),
            })
    return rows


def _write_five_objective_csv(rows: list[dict[str, object]], output_path: Path) -> str:
    fieldnames = [
        "backend",
        "solution_scope",
        "solution_id",
        "label",
        "is_representative",
        "source_json",
        "frontier_result_path",
        "search_runtime_s",
        "avg_candidate_runtime_s",
        "final_total_distance_m",
        "average_delivery_time_h",
        "final_total_noise_impact",
        "service_rate",
        "service_rate_loss",
        "n_used_drones",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-window comparison for CPLEX, NSGA-III, and NSGA-III heuristic.")
    parser.add_argument("--output-root", default="data/compare_runs", help="Output root for comparison runs")
    parser.add_argument("--dialogues", default=str(ROOT / "data" / "seed" / "daily_demand_dialogues.jsonl"))
    parser.add_argument("--stations", default=str(ROOT / "data" / "seed" / "drone_station_locations.csv"))
    parser.add_argument("--building-data", default=str(ROOT / "data" / "seed" / "building_information.csv"))
    parser.add_argument("--ground-truth", default=str(ROOT / "data" / "seed" / "daily_demand_events.csv"))
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-slots", type=int, nargs="+", default=DEFAULT_SLOTS)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--time-limit", type=int, default=180)
    parser.add_argument("--max-solver-stations", type=int, default=1)
    parser.add_argument("--max-drones-per-station", type=int, default=3)
    parser.add_argument("--max-payload", type=float, default=60.0)
    parser.add_argument("--max-range", type=float, default=200000.0)
    parser.add_argument("--noise-weight", type=float, default=0.5)
    parser.add_argument("--drone-activation-cost", type=float, default=1000.0)
    parser.add_argument("--drone-speed", type=float, default=60.0)
    parser.add_argument("--nsga3-pop-size", type=int, default=4)
    parser.add_argument("--nsga3-n-generations", type=int, default=2)
    parser.add_argument("--nsga3-seed", type=int, default=42)
    args = parser.parse_args()

    _require_optional_dependency(
        "pymoo",
        r".\.venv\Scripts\python.exe -m pip install pymoo",
    )

    compare_dir = Path(args.output_root) / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cplex_root = compare_dir / "cplex"
    nsga3_root = compare_dir / "nsga3"
    nsga3_heuristic_root = compare_dir / "nsga3_heuristic"
    compare_dir.mkdir(parents=True, exist_ok=True)
    env = _python_env()

    common_args = [
        "--dialogues", args.dialogues,
        "--stations", args.stations,
        "--building-data", args.building_data,
        "--window", str(args.window),
        "--time-limit", str(args.time_limit),
        "--max-solver-stations", str(args.max_solver_stations),
        "--max-drones-per-station", str(args.max_drones_per_station),
        "--max-payload", str(args.max_payload),
        "--max-range", str(args.max_range),
        "--noise-weight", str(args.noise_weight),
        "--drone-activation-cost", str(args.drone_activation_cost),
        "--drone-speed", str(args.drone_speed),
        "--time-slots", *_time_slot_args(args.time_slots),
    ]
    if args.offline:
        common_args.append("--offline")

    _run(
        [
            sys.executable,
            "-m",
            "llm4fairrouting.workflow.run_workflow",
            "--output-dir", str(cplex_root),
            "--solver-backend", "cplex",
            "--pareto-scan",
            *common_args,
        ],
        env=env,
    )
    cplex_run_dir = _latest_run_dir(cplex_root)

    _run(
        [
            sys.executable,
            "-m",
            "llm4fairrouting.workflow.run_workflow",
            "--output-dir", str(nsga3_root),
            "--solver-backend", "nsga3",
            "--nsga3-pop-size", str(args.nsga3_pop_size),
            "--nsga3-n-generations", str(args.nsga3_n_generations),
            "--nsga3-seed", str(args.nsga3_seed),
            *common_args,
        ],
        env=env,
    )
    nsga3_run_dir = _latest_run_dir(nsga3_root)
    nsga3_runtime = _load_search_runtime(nsga3_run_dir / "nsga3_results.json")
    if nsga3_runtime is not None:
        print(
            f"[NSGA-III Runtime] total={nsga3_runtime['search_runtime_s']:.3f}s"
            + (
                f", avg={nsga3_runtime['avg_candidate_runtime_s']:.3f}s/candidate"
                if 'avg_candidate_runtime_s' in nsga3_runtime else ""
            )
        )

    _run(
        [
            sys.executable,
            "-m",
            "llm4fairrouting.workflow.run_workflow",
            "--output-dir", str(nsga3_heuristic_root),
            "--solver-backend", "nsga3_heuristic",
            "--nsga3-pop-size", str(args.nsga3_pop_size),
            "--nsga3-n-generations", str(args.nsga3_n_generations),
            "--nsga3-seed", str(args.nsga3_seed),
            *common_args,
        ],
        env=env,
    )
    nsga3_heuristic_run_dir = _latest_run_dir(nsga3_heuristic_root)
    nsga3_heuristic_runtime = _load_search_runtime(nsga3_heuristic_run_dir / "nsga3_heuristic_results.json")
    if nsga3_heuristic_runtime is not None:
        print(
            f"[NSGA-III Heuristic Runtime] total={nsga3_heuristic_runtime['search_runtime_s']:.3f}s"
            + (
                f", avg={nsga3_heuristic_runtime['avg_candidate_runtime_s']:.3f}s/candidate"
                if 'avg_candidate_runtime_s' in nsga3_heuristic_runtime else ""
            )
        )

    _run(
        [
            sys.executable,
            str(ROOT / "evals" / "eval_priority_alignment.py"),
            "--weights", str(cplex_run_dir / "weight_configs"),
            "--demands", str(cplex_run_dir / "extracted_demands.json"),
            "--dialogues", args.dialogues,
            "--ground-truth", args.ground_truth,
            "--output", str(compare_dir / "priority_alignment.json"),
        ],
        env=env,
    )

    _run(
        [
            sys.executable,
            str(ROOT / "evals" / "compare_solver_outputs.py"),
            "--cplex-run-dir", str(cplex_run_dir),
            "--nsga3-run-dir", str(nsga3_run_dir),
            "--nsga3-heuristic-run-dir", str(nsga3_heuristic_run_dir),
            "--output", str(compare_dir / "solver_comparison.json"),
        ],
        env=env,
    )

    rrt_chart_dir = compare_dir / "rrt_3d_charts"
    cplex_rrt_chart = export_rrt_3d_from_workflow_results(
        cplex_run_dir / "workflow_results.json",
        rrt_chart_dir / "cplex_main_rrt_3d.png",
        title="CPLEX Main Solution RRT 3D",
        topdown_output_path=rrt_chart_dir / "cplex_main_rrt_topdown.png",
    )
    nsga3_rrt_chart = _export_search_representative_rrt_chart(
        results_json=nsga3_run_dir / "nsga3_results.json",
        output_path=rrt_chart_dir / "nsga3_representative_rrt_3d.png",
        topdown_output_path=rrt_chart_dir / "nsga3_representative_rrt_topdown.png",
        title_prefix="NSGA-III",
    )
    nsga3_heuristic_rrt_chart = _export_search_representative_rrt_chart(
        results_json=nsga3_heuristic_run_dir / "nsga3_heuristic_results.json",
        output_path=rrt_chart_dir / "nsga3_heuristic_representative_rrt_3d.png",
        topdown_output_path=rrt_chart_dir / "nsga3_heuristic_representative_rrt_topdown.png",
        title_prefix="NSGA-III Heuristic",
    )
    for label, payload in (("CPLEX", cplex_rrt_chart), ("NSGA-III", nsga3_rrt_chart), ("NSGA-III Heuristic", nsga3_heuristic_rrt_chart)):
        if payload is not None:
            print(f"[{label} RRT 3D] {payload['chart_path']}")
            if payload.get("topdown_chart_path"):
                print(f"[{label} RRT Top-Down] {payload['topdown_chart_path']}")

    five_objective_rows = _collect_five_objective_rows(
        cplex_run_dir=cplex_run_dir,
        nsga3_run_dir=nsga3_run_dir,
        nsga3_heuristic_run_dir=nsga3_heuristic_run_dir,
    )
    five_objective_csv = _write_five_objective_csv(
        five_objective_rows,
        compare_dir / "five_objectives_raw.csv",
    )
    print(f"[Five Objectives CSV] {five_objective_csv}")

    manifest = {
        "compare_dir": str(compare_dir),
        "cplex_run_dir": str(cplex_run_dir),
        "nsga3_run_dir": str(nsga3_run_dir),
        "nsga3_heuristic_run_dir": str(nsga3_heuristic_run_dir),
        "priority_alignment_json": str(compare_dir / "priority_alignment.json"),
        "solver_comparison_json": str(compare_dir / "solver_comparison.json"),
        "time_slots": args.time_slots,
        "time_limit_s": args.time_limit,
        "nsga3_pop_size": args.nsga3_pop_size,
        "nsga3_n_generations": args.nsga3_n_generations,
        "drone_activation_cost": args.drone_activation_cost,
        "nsga3_runtime": nsga3_runtime,
        "nsga3_heuristic_runtime": nsga3_heuristic_runtime,
        "rrt_3d_visualizations": {
            "cplex_main": cplex_rrt_chart,
            "nsga3_representative": nsga3_rrt_chart,
            "nsga3_heuristic_representative": nsga3_heuristic_rrt_chart,
        },
        "five_objectives_csv": five_objective_csv,
        "five_objective_rows": len(five_objective_rows),
    }
    with open(compare_dir / "comparison_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Comparison manifest saved to {compare_dir / 'comparison_manifest.json'}")


if __name__ == "__main__":
    main()
