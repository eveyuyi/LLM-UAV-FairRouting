"""Workflow runner for Module 2 -> Module 3 -> solver using a fixed dialogue dataset."""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from typing import Dict, Optional

from llm4fairrouting.config.runtime_env import (
    env_bool,
    env_float,
    env_int,
    env_int_list,
    env_text,
    prepare_env_file,
)
from llm4fairrouting.llm.demand_extraction import (
    extract_all_demands,
    extract_demands_offline,
)
from llm4fairrouting.workflow.solver_adapter import (
    run_multiobjective_pareto_scan,
    serialize_workflow_results,
    solve_windows_dynamically,
)
from llm4fairrouting.multiobjective.nsga3_heuristic import run_nsga3_heuristic_search
from llm4fairrouting.multiobjective.nsga3_search import run_nsga3_pareto_search
from llm4fairrouting.routing.rrt_visualization import select_representative_frontier_solution
from llm4fairrouting.llm.priority_inference import (
    adjust_weights,
    adjust_weights_offline,
)
from llm4fairrouting.llm.client_utils import create_openai_client
from llm4fairrouting.data.seed_paths import (
    BUILDING_DATA_FILENAME,
    BUILDING_DATA_PATH,
    DEMAND_DIALOGUES_FILENAME,
    DEMAND_DIALOGUES_PATH,
    STATION_DATA_FILENAME,
    STATION_DATA_PATH,
)


class _TeeStdout:
    """Simple tee for sys.stdout that also writes to a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
        for stream in self._streams:
            stream.flush()
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _build_run_dir(base_dir: Path, model: str, noise_weight: float) -> Path:
    """Create a timestamped run directory under *base_dir*.

    Format: ``<base_dir>/run_<YYYYMMDD_HHMMSS>_<model>_noise<w>/``
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = model.replace("/", "-")
    run_name = f"run_{ts}_{tag}_noise{noise_weight}"
    return base_dir / run_name


def _dialogue_time_slot(dialogue: Dict) -> Optional[int]:
    """Return the canonical 5-minute ``time_slot`` for a dialogue when available."""
    metadata = dialogue.get("metadata", {})
    raw_slot = metadata.get("time_slot")
    if raw_slot not in (None, ""):
        try:
            return int(raw_slot)
        except (TypeError, ValueError):
            pass

    timestamp = str(dialogue.get("timestamp", "")).strip()
    if not timestamp:
        return None

    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None

    minutes_from_midnight = parsed.hour * 60 + parsed.minute
    return minutes_from_midnight // 5


def _filter_dialogues_by_time_slots(dialogues: list[Dict], time_slots: Optional[list[int]]) -> list[Dict]:
    """Keep only dialogues whose canonical 5-minute slot is listed in ``time_slots``."""
    if time_slots is None:
        return list(dialogues)

    allowed_slots = {int(slot) for slot in time_slots}
    return [
        dialogue
        for dialogue in dialogues
        if _dialogue_time_slot(dialogue) in allowed_slots
    ]


_WINDOW_LABEL_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T(?P<start_hour>\d{2}):(?P<start_min>\d{2})-\d{2}:\d{2}$"
)


def _timestamp_time_slot(timestamp: object) -> Optional[int]:
    raw = str(timestamp or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return (parsed.hour * 60 + parsed.minute) // 5


def _window_time_slots(window: Dict) -> set[int]:
    slots: set[int] = set()
    label = str(window.get("time_window", "")).strip().split("::", 1)[0]
    match = _WINDOW_LABEL_RE.match(label)
    if match:
        start_hour = int(match.group("start_hour"))
        start_min = int(match.group("start_min"))
        slots.add((start_hour * 60 + start_min) // 5)

    for demand in window.get("demands", []) or []:
        slot = _timestamp_time_slot(demand.get("request_timestamp"))
        if slot is not None:
            slots.add(slot)

    return slots


def _filter_extracted_windows_by_time_slots(
    windows: list[Dict],
    time_slots: Optional[list[int]],
) -> list[Dict]:
    if time_slots is None:
        return list(windows)

    allowed_slots = {int(slot) for slot in time_slots}
    return [
        window
        for window in windows
        if _window_time_slots(window) & allowed_slots
    ]


def _load_precomputed_window_results(path: str) -> list[Dict]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        with open(source, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with open(source, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Precomputed extracted demands must be a JSON/JSONL list: {path}")
    return payload


def _extract_drone_path_details(all_solutions: list[Dict]) -> list[Dict]:
    """Return the first available structured drone-path payload from workflow results."""
    for solution_entry in all_solutions:
        solution = solution_entry.get("solution") or {}
        details = solution.get("drone_path_details")
        if details:
            return list(details)
        details = solution_entry.get("drone_path_details")
        if details:
            return list(details)
    return []


def _resolve_search_result_path(path_text: str) -> Optional[Path]:
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    if candidate.exists():
        return candidate.resolve()
    project_candidate = (PROJECT_ROOT / candidate).resolve()
    return project_candidate if project_candidate.exists() else None


def _load_representative_search_workflow_results(search_payload: Optional[Dict]) -> list[Dict]:
    frontier = list((search_payload or {}).get("frontier") or [])
    representative = select_representative_frontier_solution(frontier)
    if not representative:
        return []

    result_path_text = str(representative.get("frontier_result_path") or "").strip()
    if not result_path_text:
        return []

    result_path = _resolve_search_result_path(result_path_text)
    if result_path is None:
        return []

    with open(result_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


# ============================================================================
# Pipeline Runner
# ============================================================================

def run_workflow(
    output_dir: str,
    dialogue_path: Optional[str] = None,
    extracted_demands_path: Optional[str] = None,
    stations_path: Optional[str] = None,
    offline: bool = False,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    window_minutes: int = 5,
    time_limit: int = 10,
    max_drones_per_station: int = 3,
    max_payload: float = 60.0,
    max_range: float = 200000.0,
    max_solver_stations: Optional[int] = 1,
    skip_solver: bool = False,
    noise_weight: float = 0.5,
    drone_activation_cost: float = 1000.0,
    building_path: Optional[str] = None,
    drone_speed: float = 60.0,
    time_slots: Optional[list[int]] = None,
    pareto_scan: bool = False,
    enable_conflict_refiner: bool = False,
    solver_backend: str = "cplex",
    nsga3_pop_size: int = 20,
    nsga3_n_generations: int = 10,
    nsga3_seed: int = 42,
    nsga3_save_all_candidate_results: bool = False,
):
    """Run the workflow from a canonical dialogue dataset through ranking and solving.

    Backward compatibility note:
    - When ``extracted_demands_path`` is ``None`` (the default), the workflow behaves the
      same as before and executes Module 2 on the provided dialogues.
    - When ``extracted_demands_path`` is set, Module 2 is intentionally skipped and the
      provided precomputed window demands are used as the fixed input to Module 3 and the
      solver. This is mainly for evaluation/debugging, especially pure-LLM3 comparisons.
    """
    base_dir = Path(output_dir)
    run_dir = _build_run_dir(base_dir, model, noise_weight)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_stations_path = str(stations_path or STATION_DATA_PATH)
    resolved_dialogue_path = str(dialogue_path or DEMAND_DIALOGUES_PATH)

    # 将 stdout 同步写入终端和 log 文件
    log_path = run_dir / "workflow.log"
    original_stdout = sys.stdout
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _TeeStdout(original_stdout, log_file)

    try:
        # 记录命令行（如果是通过 CLI 调用）
        try:
            cmdline = " ".join(sys.argv)
            print(f"[COMMAND] {cmdline}")
        except Exception:
            pass

        weight_configs_dir = run_dir / "weight_configs"
        weight_configs_dir.mkdir(exist_ok=True)

        # Save run metadata for reproducibility
        run_meta = {
            "created_at": datetime.now().isoformat(),
            "model": model,
            "offline": offline,
            "noise_weight": noise_weight,
            "drone_activation_cost": drone_activation_cost,
            "temperature": temperature,
            "window_minutes": window_minutes,
            "time_limit": time_limit,
            "max_drones_per_station": max_drones_per_station,
            "max_payload": max_payload,
            "max_range": max_range,
            "max_solver_stations": max_solver_stations,
            "building_path": building_path or str(BUILDING_DATA_PATH),
            "drone_speed": drone_speed,
            "stations_path": resolved_stations_path,
            "dialogue_path": resolved_dialogue_path,
            "extracted_demands_path": extracted_demands_path,
            "time_slots": time_slots,
            "skip_solver": skip_solver,
            "pareto_scan": pareto_scan,
            "enable_conflict_refiner": enable_conflict_refiner,
            "solver_backend": solver_backend,
            "nsga3_pop_size": nsga3_pop_size,
            "nsga3_n_generations": nsga3_n_generations,
            "nsga3_seed": nsga3_seed,
            "nsga3_save_all_candidate_results": nsga3_save_all_candidate_results,
        }
        with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(run_meta, f, ensure_ascii=False, indent=2)
        print(f"Run directory: {run_dir}")

        client = None

        if not offline:
            client = create_openai_client(api_base, api_key)

        # ----------------------------------------------------------------
        # Step 2: Module 2 — 按窗口提取需求
        # ----------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Step 2: Context Extraction (Module 2)")
        print("=" * 60)

        if extracted_demands_path:
            print("=" * 60)
            print("Input: Precomputed Extracted Demands")
            print("=" * 60)
            window_results = _load_precomputed_window_results(extracted_demands_path)
            print(f"  Loaded {len(window_results)} demand windows from {extracted_demands_path}")
            if time_slots is not None:
                filtered_windows = _filter_extracted_windows_by_time_slots(window_results, time_slots)
                print(
                    f"  Applied time-slot filter {time_slots}: "
                    f"kept {len(filtered_windows)} / {len(window_results)} windows"
                )
                window_results = filtered_windows
        else:
            print("=" * 60)
            print("Input: Daily Demand Dialogues")
            print("=" * 60)

            with open(resolved_dialogue_path, "r", encoding="utf-8") as f:
                dialogues = [json.loads(l.strip()) for l in f if l.strip()]
            print(f"  Loaded {len(dialogues)} dialogues from {resolved_dialogue_path}")
            if time_slots is not None:
                filtered_dialogues = _filter_dialogues_by_time_slots(dialogues, time_slots)
                print(
                    f"  Applied time-slot filter {time_slots}: "
                    f"kept {len(filtered_dialogues)} / {len(dialogues)} dialogues"
                )
                dialogues = filtered_dialogues

            if offline:
                window_results = extract_demands_offline(dialogues, window_minutes)
            else:
                window_results = extract_all_demands(
                    dialogues,
                    client,
                    model,
                    window_minutes,
                    temperature,
                )

        demands_path = run_dir / "extracted_demands.json"
        with open(demands_path, "w", encoding="utf-8") as f:
            json.dump(window_results, f, ensure_ascii=False, indent=2)
        print(f"  Saved extracted demands to {demands_path}")

        total_demands = sum(len(w.get("demands", [])) for w in window_results)
        print(f"  Extracted {total_demands} demands across {len(window_results)} windows")

        # ----------------------------------------------------------------
        # Step 3: Module 3 — 逐窗口调整权重 + 求解
        # ----------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Step 3: Priority Ranking + Solve (Module 3)")
        print("=" * 60)

        all_solutions = []
        windows_to_solve = []
        weight_configs_by_window: Dict[str, Dict] = {}

        for w_idx, window in enumerate(window_results):
            tw = window.get("time_window", f"window_{w_idx}")
            demands = window.get("demands", [])

            if not demands:
                print(f"\n  Window {tw}: no demands, skipping")
                continue

            print(f"\n  ---- Window {tw}: {len(demands)} demands ----")

            # 3a: 权重调整
            if offline:
                weight_config = adjust_weights_offline(demands)
            else:
                weight_config = adjust_weights(demands, client, model)

            weight_config["time_window"] = tw

            wc_path = weight_configs_dir / f"weight_config_window{w_idx}.json"
            with open(wc_path, "w", encoding="utf-8") as f:
                json.dump(weight_config, f, ensure_ascii=False, indent=2)
            weight_configs_by_window[tw] = weight_config

            if skip_solver:
                print("  Solver skipped (--skip-solver)")
                all_solutions.append(
                    {
                        "time_window": tw,
                        "weight_config": weight_config,
                        "feasible_demands": demands,
                        "n_demands_total": len(demands),
                        "n_demands_filtered": 0,
                        "solution": None,
                        "n_supply": 0,
                    }
                )
                continue

            windows_to_solve.append(
                {
                    "time_window": tw,
                    "demands": demands,
                }
            )

        search_payload = None
        search_results_filename = None
        if not skip_solver and windows_to_solve:
            if solver_backend == "nsga3":
                search_payload = run_nsga3_pareto_search(
                    windows=windows_to_solve,
                    weight_configs=weight_configs_by_window,
                    stations_path=resolved_stations_path,
                    building_path=building_path or str(BUILDING_DATA_PATH),
                    max_solver_stations=max_solver_stations,
                    time_limit=time_limit,
                    max_drones_per_station=max_drones_per_station,
                    max_payload=max_payload,
                    max_range=max_range,
                    noise_weight=noise_weight,
                    drone_activation_cost=drone_activation_cost,
                    drone_speed=drone_speed,
                    output_dir=str(run_dir / "solver_analytics" / "nsga3"),
                    pop_size=nsga3_pop_size,
                    n_generations=nsga3_n_generations,
                    seed=nsga3_seed,
                    save_all_candidate_results=nsga3_save_all_candidate_results,
                    enable_conflict_refiner=enable_conflict_refiner,
                    problem_id=run_dir.name,
                )
                search_results_filename = "nsga3_results.json"
            elif solver_backend == "nsga3_heuristic":
                search_payload = run_nsga3_heuristic_search(
                    windows=windows_to_solve,
                    weight_configs=weight_configs_by_window,
                    stations_path=resolved_stations_path,
                    building_path=building_path or str(BUILDING_DATA_PATH),
                    max_solver_stations=max_solver_stations,
                    time_limit=time_limit,
                    max_drones_per_station=max_drones_per_station,
                    max_payload=max_payload,
                    max_range=max_range,
                    noise_weight=noise_weight,
                    drone_activation_cost=drone_activation_cost,
                    drone_speed=drone_speed,
                    output_dir=str(run_dir / "solver_analytics" / "nsga3_heuristic"),
                    pop_size=nsga3_pop_size,
                    n_generations=nsga3_n_generations,
                    seed=nsga3_seed,
                    save_all_candidate_results=nsga3_save_all_candidate_results,
                    enable_conflict_refiner=enable_conflict_refiner,
                    problem_id=run_dir.name,
                )
                search_results_filename = "nsga3_heuristic_results.json"
            else:
                all_solutions.extend(
                    solve_windows_dynamically(
                        windows=windows_to_solve,
                        weight_configs=weight_configs_by_window,
                        stations_path=resolved_stations_path,
                        building_path=building_path or str(BUILDING_DATA_PATH),
                        max_solver_stations=max_solver_stations,
                        time_limit=time_limit,
                        max_drones_per_station=max_drones_per_station,
                        max_payload=max_payload,
                        max_range=max_range,
                        noise_weight=noise_weight,
                        drone_activation_cost=drone_activation_cost,
                        drone_speed=drone_speed,
                        analytics_output_dir=str(run_dir / "solver_analytics"),
                        enable_conflict_refiner=enable_conflict_refiner,
                    )
                )
                if pareto_scan:
                    run_multiobjective_pareto_scan(
                        windows=windows_to_solve,
                        weight_configs=weight_configs_by_window,
                        stations_path=resolved_stations_path,
                        building_path=building_path or str(BUILDING_DATA_PATH),
                        max_solver_stations=max_solver_stations,
                        time_limit=time_limit,
                        max_drones_per_station=max_drones_per_station,
                        max_payload=max_payload,
                        max_range=max_range,
                        noise_weight=noise_weight,
                        drone_activation_cost=drone_activation_cost,
                        drone_speed=drone_speed,
                        analytics_output_dir=str(run_dir / "solver_analytics" / "pareto"),
                        enable_conflict_refiner=enable_conflict_refiner,
                    )

        # ----------------------------------------------------------------
        # 保存汇总结果（含完整 eval 字段）
        # ----------------------------------------------------------------
        workflow_results_payload = serialize_workflow_results(all_solutions)
        if not workflow_results_payload and search_payload is not None:
            workflow_results_payload = _load_representative_search_workflow_results(search_payload)
            if not workflow_results_payload:
                print("  Warning: no representative frontier workflow result could be loaded")

        summary_path = run_dir / "workflow_results.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                workflow_results_payload,
                f,
                ensure_ascii=False,
                indent=2,
            )
        if search_payload is not None and search_results_filename is not None:
            search_summary_path = run_dir / search_results_filename
            with open(search_summary_path, "w", encoding="utf-8") as f:
                json.dump(search_payload, f, ensure_ascii=False, indent=2)
        drone_paths = _extract_drone_path_details(workflow_results_payload)
        drone_paths_path = run_dir / "drone_path_results.json"
        legacy_drone_paths_path = run_dir / "drone_paths.json"
        if drone_paths:
            with open(drone_paths_path, "w", encoding="utf-8") as f:
                json.dump(drone_paths, f, ensure_ascii=False, indent=2)
            with open(legacy_drone_paths_path, "w", encoding="utf-8") as f:
                json.dump(drone_paths, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print("Workflow finished. Outputs:")
        print(f"  Run directory   : {run_dir}")
        print(f"  Weight configs  : {weight_configs_dir}")
        print(f"  Workflow results: {summary_path}")
        if search_payload is not None and search_results_filename is not None:
            print(f"  Search summary  : {run_dir / search_results_filename}")
            search_meta = search_payload.get("search_meta") or {}
            runtime_s = search_meta.get("search_runtime_s")
            avg_runtime_s = search_meta.get("avg_candidate_runtime_s")
            if runtime_s is not None:
                if avg_runtime_s is not None:
                    print(f"  Search runtime  : {float(runtime_s):.3f}s total, {float(avg_runtime_s):.3f}s/candidate")
                else:
                    print(f"  Search runtime  : {float(runtime_s):.3f}s total")
        if drone_paths:
            print(f"  Drone paths     : {drone_paths_path}")
        analytics_dir = run_dir / "solver_analytics"
        if analytics_dir.exists():
            print(f"  Solver analytics: {analytics_dir}")
        print(f"  Log file        : {log_path}")
        print(f"{'=' * 60}")

        return all_solutions
    finally:
        # 恢复 stdout 并关闭日志文件
        sys.stdout = original_stdout
        log_file.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    active_env_file = prepare_env_file(PROJECT_ROOT)
    parser = argparse.ArgumentParser(
        description="llm4fairrouting Workflow Runner (Module 2 → 3 → solver)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Offline extraction + ranking (skip the solver)
  python -m llm4fairrouting.workflow.run_workflow --offline --skip-solver

  # Online extraction + ranking + solving from the canonical dialogue dataset
  OPENAI_API_KEY=YOUR_KEY python -m llm4fairrouting.workflow.run_workflow
""",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(active_env_file) if active_env_file else None,
        help="Environment file path; defaults to the project .env when present",
    )

    parser.add_argument(
        "--dialogues",
        type=str,
        default=env_text("LLM4FAIRROUTING_DIALOGUES", str(DEMAND_DIALOGUES_PATH)),
        help=f"Dialogue dataset path (default {DEMAND_DIALOGUES_FILENAME})",
    )
    parser.add_argument(
        "--extracted-demands",
        type=str,
        default=env_text("LLM4FAIRROUTING_EXTRACTED_DEMANDS"),
        help="Optional precomputed extracted_demands.json to skip Module 2 and evaluate Module 3 on fixed inputs",
    )
    parser.add_argument(
        "--stations", type=str,
        default=env_text("LLM4FAIRROUTING_STATIONS", str(STATION_DATA_PATH)),
        help=f"{STATION_DATA_FILENAME} station data path",
    )

    parser.add_argument(
        "--output-dir", type=str,
        default=env_text("LLM4FAIRROUTING_OUTPUT_DIR", str(PROJECT_ROOT / "results")),
        help="Output root directory; each run creates a timestamped subdirectory",
    )
    parser.add_argument(
        "--building-data",
        type=str,
        default=env_text("LLM4FAIRROUTING_BUILDING_DATA", str(BUILDING_DATA_PATH)),
        help=f"{BUILDING_DATA_FILENAME} path used for realistic distance/noise modeling",
    )
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=env_bool("LLM4FAIRROUTING_OFFLINE", False),
        help="Run without calling an LLM",
    )
    parser.add_argument(
        "--skip-solver",
        action=argparse.BooleanOptionalAction,
        default=env_bool("LLM4FAIRROUTING_SKIP_SOLVER", False),
        help="Skip the solver stage",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=env_text("OPENAI_BASE_URL"),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=env_text("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=env_text("LLM4FAIRROUTING_MODEL", "gpt-4o-mini"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=env_float("LLM4FAIRROUTING_TEMPERATURE", 0.0),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=env_int("LLM4FAIRROUTING_WINDOW", 5),
        help="Time-window size in minutes",
    )
    parser.add_argument(
        "--time-slots",
        type=int,
        nargs="+",
        default=env_int_list("LLM4FAIRROUTING_TIME_SLOTS"),
        help="Only process dialogues whose metadata.time_slot falls in these 5-minute slots",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=env_int("LLM4FAIRROUTING_TIME_LIMIT", 10),
        help="Solver time limit in seconds",
    )
    parser.add_argument(
        "--max-solver-stations",
        type=int,
        default=env_int("LLM4FAIRROUTING_MAX_SOLVER_STATIONS", 1),
        help="Maximum number of real stations to include in solving; 0 means all stations",
    )
    parser.add_argument(
        "--max-drones-per-station",
        type=int,
        default=env_int("LLM4FAIRROUTING_MAX_DRONES_PER_STATION", 3),
        help="Maximum number of drones available at each station",
    )
    parser.add_argument(
        "--max-payload",
        type=float,
        default=env_float("LLM4FAIRROUTING_MAX_PAYLOAD", 60.0),
        help="Maximum payload in kilograms",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=env_float("LLM4FAIRROUTING_MAX_RANGE", 200000.0),
        help="Maximum drone range in meters",
    )
    parser.add_argument(
        "--drone-speed",
        type=float,
        default=env_float("LLM4FAIRROUTING_DRONE_SPEED", 60.0),
        help="Drone speed in meters per second for dynamic simulation",
    )
    parser.add_argument("--noise-weight", type=float, default=env_float("LLM4FAIRROUTING_NOISE_WEIGHT", 0.5),
                        help="Noise-cost weight in the objective")
    parser.add_argument(
        "--drone-activation-cost",
        type=float,
        default=env_float("LLM4FAIRROUTING_DRONE_ACTIVATION_COST", 1000.0),
        help="Activation cost per used drone in the objective",
    )
    parser.add_argument(
        "--pareto-scan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a weighted-sum Pareto scan and export the Pareto frontier artifacts",
    )
    parser.add_argument(
        "--enable-conflict-refiner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request conflict diagnostics when a solve is infeasible",
    )
    parser.add_argument(
        "--solver-backend",
        type=str,
        choices=("cplex", "nsga3", "nsga3_heuristic"),
        default=env_text("LLM4FAIRROUTING_SOLVER_BACKEND", "cplex"),
        help="Solver backend: exact CPLEX routing, NSGA-III over CPLEX, or NSGA-III over the greedy heuristic backend",
    )
    parser.add_argument(
        "--nsga3-pop-size",
        type=int,
        default=env_int("LLM4FAIRROUTING_NSGA3_POP_SIZE", 20),
        help="Population size for NSGA-III when --solver-backend nsga3 is selected",
    )
    parser.add_argument(
        "--nsga3-n-generations",
        type=int,
        default=env_int("LLM4FAIRROUTING_NSGA3_N_GENERATIONS", 10),
        help="Number of generations for NSGA-III when --solver-backend nsga3 is selected",
    )
    parser.add_argument(
        "--nsga3-seed",
        type=int,
        default=env_int("LLM4FAIRROUTING_NSGA3_SEED", 42),
        help="Random seed for NSGA-III",
    )
    parser.add_argument(
        "--nsga3-save-all-candidate-results",
        action=argparse.BooleanOptionalAction,
        default=env_bool("LLM4FAIRROUTING_NSGA3_SAVE_ALL_RESULTS", False),
        help="Persist per-candidate workflow results for NSGA-III",
    )
    args = parser.parse_args()

    run_workflow(
        output_dir=args.output_dir,
        dialogue_path=args.dialogues,
        extracted_demands_path=args.extracted_demands,
        stations_path=args.stations,
        offline=args.offline,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        window_minutes=args.window,
        time_limit=args.time_limit,
        max_drones_per_station=args.max_drones_per_station,
        max_payload=args.max_payload,
        max_range=args.max_range,
        max_solver_stations=args.max_solver_stations,
        skip_solver=args.skip_solver,
        noise_weight=args.noise_weight,
        drone_activation_cost=args.drone_activation_cost,
        building_path=args.building_data,
        drone_speed=args.drone_speed,
        time_slots=args.time_slots,
        pareto_scan=args.pareto_scan,
        enable_conflict_refiner=args.enable_conflict_refiner,
        solver_backend=args.solver_backend,
        nsga3_pop_size=args.nsga3_pop_size,
        nsga3_n_generations=args.nsga3_n_generations,
        nsga3_seed=args.nsga3_seed,
        nsga3_save_all_candidate_results=args.nsga3_save_all_candidate_results,
    )


if __name__ == "__main__":
    main()
