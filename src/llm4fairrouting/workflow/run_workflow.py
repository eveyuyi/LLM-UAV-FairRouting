"""Workflow runner for Module 2 -> Module 3 -> solver using a fixed dialogue dataset."""

import argparse
import json
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
    serialize_workflow_results,
    solve_windows_dynamically,
)
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


# ============================================================================
# Pipeline Runner
# ============================================================================

def run_workflow(
    output_dir: str,
    dialogue_path: Optional[str] = None,
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
    building_path: Optional[str] = None,
    drone_speed: float = 60.0,
    time_slots: Optional[list[int]] = None,
):
    """Run the workflow from a canonical dialogue dataset through ranking and solving."""
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
            "time_slots": time_slots,
            "skip_solver": skip_solver,
        }
        with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(run_meta, f, ensure_ascii=False, indent=2)
        print(f"Run directory: {run_dir}")

        client = None

        if not offline:
            client = create_openai_client(api_base, api_key)

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

        # ----------------------------------------------------------------
        # Step 2: Module 2 — 按窗口提取需求
        # ----------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Step 2: Context Extraction (Module 2)")
        print("=" * 60)

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

        if not skip_solver and windows_to_solve:
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
                    drone_speed=drone_speed,
                )
            )

        # ----------------------------------------------------------------
        # 保存汇总结果（含完整 eval 字段）
        # ----------------------------------------------------------------
        summary_path = run_dir / "workflow_results.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                serialize_workflow_results(all_solutions),
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n{'=' * 60}")
        print("Workflow finished. Outputs:")
        print(f"  Run directory   : {run_dir}")
        print(f"  Weight configs  : {weight_configs_dir}")
        print(f"  Workflow results: {summary_path}")
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
        "--drone-speed",
        type=float,
        default=env_float("LLM4FAIRROUTING_DRONE_SPEED", 60.0),
        help="Drone speed in meters per second for dynamic simulation",
    )
    parser.add_argument("--noise-weight", type=float, default=env_float("LLM4FAIRROUTING_NOISE_WEIGHT", 0.5),
                        help="Noise-cost weight in the objective")
    args = parser.parse_args()

    run_workflow(
        output_dir=args.output_dir,
        dialogue_path=args.dialogues,
        stations_path=args.stations,
        offline=args.offline,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        window_minutes=args.window,
        time_limit=args.time_limit,
        max_solver_stations=args.max_solver_stations,
        skip_solver=args.skip_solver,
        noise_weight=args.noise_weight,
        building_path=args.building_data,
        drone_speed=args.drone_speed,
        time_slots=args.time_slots,
    )


if __name__ == "__main__":
    main()
