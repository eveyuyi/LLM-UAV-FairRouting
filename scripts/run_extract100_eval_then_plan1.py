"""Run the two-stage experiment: extract/evaluate 100 windows, then plan 1 window."""

from __future__ import annotations

import argparse
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


DEFAULT_EXTRACT_COUNT = 100
DEFAULT_PLAN_SLOT = 0


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


def _slot_range(count: int, *, start: int = 0) -> List[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    return list(range(start, start + count))


def _require_optional_dependency(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    raise SystemExit(
        f"Missing optional dependency '{module_name}'. Install it first with: {install_hint}"
    )


def _build_workflow_command(
    *,
    python_exe: str,
    output_dir: Path,
    dialogue_path: str,
    stations_path: str,
    building_path: str,
    time_slots: List[int],
    window_minutes: int,
    time_limit: int,
    max_solver_stations: int,
    max_drones_per_station: int,
    max_payload: float,
    max_range: float,
    noise_weight: float,
    drone_activation_cost: float,
    drone_speed: float,
    solver_backend: str,
    offline: bool,
    skip_solver: bool,
    nsga3_pop_size: int,
    nsga3_n_generations: int,
    nsga3_seed: int,
) -> List[str]:
    command = [
        python_exe,
        "-m",
        "llm4fairrouting.workflow.run_workflow",
        "--output-dir",
        str(output_dir),
        "--dialogues",
        dialogue_path,
        "--stations",
        stations_path,
        "--building-data",
        building_path,
        "--window",
        str(window_minutes),
        "--time-limit",
        str(time_limit),
        "--max-solver-stations",
        str(max_solver_stations),
        "--max-drones-per-station",
        str(max_drones_per_station),
        "--max-payload",
        str(max_payload),
        "--max-range",
        str(max_range),
        "--noise-weight",
        str(noise_weight),
        "--drone-activation-cost",
        str(drone_activation_cost),
        "--drone-speed",
        str(drone_speed),
        "--time-slots",
        *[str(slot) for slot in time_slots],
    ]
    if offline:
        command.append("--offline")
    if skip_solver:
        command.append("--skip-solver")
    else:
        command.extend([
            "--solver-backend",
            solver_backend,
        ])
        if solver_backend in {"nsga3", "nsga3_heuristic"}:
            command.extend([
                "--nsga3-pop-size",
                str(nsga3_pop_size),
                "--nsga3-n-generations",
                str(nsga3_n_generations),
                "--nsga3-seed",
                str(nsga3_seed),
            ])
    return command


def _build_priority_alignment_command(
    *,
    python_exe: str,
    weights_path: Path,
    demands_path: Path,
    dialogues_path: str,
    ground_truth_csv: str,
    urgent_threshold: int,
    output_path: Path,
) -> List[str]:
    return [
        python_exe,
        str(ROOT / "evals" / "eval_priority_alignment.py"),
        "--weights",
        str(weights_path),
        "--demands",
        str(demands_path),
        "--dialogues",
        dialogues_path,
        "--ground-truth",
        ground_truth_csv,
        "--urgent-threshold",
        str(urgent_threshold),
        "--output",
        str(output_path),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract/evaluate 100 windows, then run a single-window planning experiment."
    )
    parser.add_argument("--output-root", default="data/extract_eval_plan_runs", help="Output root for this two-stage experiment")
    parser.add_argument("--dialogues", default=str(ROOT / "data" / "seed" / "daily_demand_dialogues.jsonl"))
    parser.add_argument("--stations", default=str(ROOT / "data" / "seed" / "drone_station_locations.csv"))
    parser.add_argument("--building-data", default=str(ROOT / "data" / "seed" / "building_information.csv"))
    parser.add_argument("--ground-truth", default=str(ROOT / "data" / "seed" / "daily_demand_events_manifest.jsonl"))
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=False, help="Run extraction/ranking without calling an LLM")
    parser.add_argument("--extract-window-count", type=int, default=DEFAULT_EXTRACT_COUNT, help="Number of 5-minute windows to extract/evaluate")
    parser.add_argument("--extract-window-start", type=int, default=0, help="Starting slot index for extraction/evaluation")
    parser.add_argument("--plan-slot", type=int, default=DEFAULT_PLAN_SLOT, help="Single 5-minute slot to use for planning")
    parser.add_argument("--plan-backend", choices=("cplex", "nsga3", "nsga3_heuristic"), default="cplex")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--time-limit", type=int, default=180)
    parser.add_argument("--max-solver-stations", type=int, default=1)
    parser.add_argument("--max-drones-per-station", type=int, default=3)
    parser.add_argument("--max-payload", type=float, default=60.0)
    parser.add_argument("--max-range", type=float, default=200000.0)
    parser.add_argument("--noise-weight", type=float, default=0.5)
    parser.add_argument("--drone-activation-cost", type=float, default=1000.0)
    parser.add_argument("--drone-speed", type=float, default=60.0)
    parser.add_argument("--urgent-threshold", type=int, default=2)
    parser.add_argument("--nsga3-pop-size", type=int, default=4)
    parser.add_argument("--nsga3-n-generations", type=int, default=2)
    parser.add_argument("--nsga3-seed", type=int, default=42)
    args = parser.parse_args()

    if args.plan_backend in {"nsga3", "nsga3_heuristic"}:
        _require_optional_dependency(
            "pymoo",
            r".\.venv\Scripts\python.exe -m pip install pymoo",
        )

    session_dir = Path(args.output_root) / f"extract_eval_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    extract_output_root = session_dir / "extract_eval"
    plan_output_root = session_dir / "plan"
    priority_eval_path = session_dir / "priority_alignment.json"
    manifest_path = session_dir / "experiment_manifest.json"

    session_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    env = _python_env()

    extract_slots = _slot_range(args.extract_window_count, start=args.extract_window_start)
    extract_command = _build_workflow_command(
        python_exe=python_exe,
        output_dir=extract_output_root,
        dialogue_path=args.dialogues,
        stations_path=args.stations,
        building_path=args.building_data,
        time_slots=extract_slots,
        window_minutes=args.window,
        time_limit=args.time_limit,
        max_solver_stations=args.max_solver_stations,
        max_drones_per_station=args.max_drones_per_station,
        max_payload=args.max_payload,
        max_range=args.max_range,
        noise_weight=args.noise_weight,
        drone_activation_cost=args.drone_activation_cost,
        drone_speed=args.drone_speed,
        solver_backend=args.plan_backend,
        offline=args.offline,
        skip_solver=True,
        nsga3_pop_size=args.nsga3_pop_size,
        nsga3_n_generations=args.nsga3_n_generations,
        nsga3_seed=args.nsga3_seed,
    )
    _run(extract_command, env=env)
    extract_run_dir = _latest_run_dir(extract_output_root)

    priority_eval_command = _build_priority_alignment_command(
        python_exe=python_exe,
        weights_path=extract_run_dir / "weight_configs",
        demands_path=extract_run_dir / "extracted_demands.json",
        dialogues_path=args.dialogues,
        ground_truth_csv=args.ground_truth,
        urgent_threshold=args.urgent_threshold,
        output_path=priority_eval_path,
    )
    _run(priority_eval_command, env=env)

    plan_command = _build_workflow_command(
        python_exe=python_exe,
        output_dir=plan_output_root,
        dialogue_path=args.dialogues,
        stations_path=args.stations,
        building_path=args.building_data,
        time_slots=[args.plan_slot],
        window_minutes=args.window,
        time_limit=args.time_limit,
        max_solver_stations=args.max_solver_stations,
        max_drones_per_station=args.max_drones_per_station,
        max_payload=args.max_payload,
        max_range=args.max_range,
        noise_weight=args.noise_weight,
        drone_activation_cost=args.drone_activation_cost,
        drone_speed=args.drone_speed,
        solver_backend=args.plan_backend,
        offline=args.offline,
        skip_solver=False,
        nsga3_pop_size=args.nsga3_pop_size,
        nsga3_n_generations=args.nsga3_n_generations,
        nsga3_seed=args.nsga3_seed,
    )
    _run(plan_command, env=env)
    plan_run_dir = _latest_run_dir(plan_output_root)

    manifest = {
        "session_dir": str(session_dir),
        "python_executable": python_exe,
        "offline": bool(args.offline),
        "extract": {
            "window_start": args.extract_window_start,
            "window_count": args.extract_window_count,
            "time_slots": extract_slots,
            "run_dir": str(extract_run_dir),
            "extracted_demands": str(extract_run_dir / "extracted_demands.json"),
            "weight_configs": str(extract_run_dir / "weight_configs"),
        },
        "evaluation": {
            "priority_alignment_json": str(priority_eval_path),
            "ground_truth_csv": args.ground_truth,
            "urgent_threshold": args.urgent_threshold,
        },
        "planning": {
            "time_slot": args.plan_slot,
            "backend": args.plan_backend,
            "run_dir": str(plan_run_dir),
            "workflow_results": str(plan_run_dir / "workflow_results.json"),
            "solver_analytics_dir": str(plan_run_dir / "solver_analytics"),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("\nExperiment finished.")
    print(f"  Session directory         : {session_dir}")
    print(f"  Extracted demands         : {extract_run_dir / 'extracted_demands.json'}")
    print(f"  Priority alignment result : {priority_eval_path}")
    print(f"  Planning workflow results : {plan_run_dir / 'workflow_results.json'}")
    print(f"  Experiment manifest       : {manifest_path}")


if __name__ == "__main__":
    main()
