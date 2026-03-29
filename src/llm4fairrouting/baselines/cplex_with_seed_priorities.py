"""Baseline solver that reads the shared seed demand events directly."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llm4fairrouting.data.event_data import (
    event_record_to_solver_demand,
    ground_truth_priority_from_record,
    load_event_records,
)
from llm4fairrouting.data.seed_paths import (
    BUILDING_DATA_PATH,
    DEMAND_EVENTS_MANIFEST_PATH,
    DEMAND_EVENTS_PATH,
    STATION_DATA_PATH,
)
from llm4fairrouting.workflow.solver_adapter import (
    run_multiobjective_pareto_scan,
    serialize_workflow_results,
    solve_windows_dynamically,
)


def _build_run_dir(base_dir: Path, noise_weight: float) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"baseline_seed_{ts}_noise{noise_weight}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _window_label_for_timestamp(timestamp: datetime, window_minutes: int) -> str:
    abs_start = (timestamp.hour * 60 + timestamp.minute) // window_minutes * window_minutes
    abs_end = abs_start + window_minutes
    h_start, m_start = divmod(abs_start, 60)
    h_end, m_end = divmod(abs_end, 60)
    return (
        f"{timestamp.date().isoformat()}T{h_start:02d}:{m_start:02d}"
        f"-{h_end:02d}:{m_end:02d}"
    )


def _normalize_priority(priority: object) -> int:
    try:
        value = int(priority)
    except (TypeError, ValueError):
        value = 4
    return min(max(value, 1), 4)


def build_seed_priority_inputs(
    *,
    csv_path: str,
    base_date: str,
    window_minutes: int,
    n_events: Optional[int] = None,
    time_slots: Optional[List[int]] = None,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    events = load_event_records(csv_path)
    if time_slots is not None:
        allowed = {int(slot) for slot in time_slots}
        events = [event for event in events if int(event.get("time_slot", -1)) in allowed]
    if n_events is not None and len(events) > n_events:
        events = sorted(
            events[:n_events],
            key=lambda item: (int(item.get("time_slot", 0)), str(item.get("event_id", ""))),
        )
    windows_map: Dict[str, List[Dict]] = {}
    priority_map: Dict[str, List[Tuple[str, int]]] = {}

    for event in events:
        timestamp = datetime.strptime(base_date, "%Y-%m-%d") + timedelta(
            minutes=int(event["time_slot"]) * 5
        )
        label = _window_label_for_timestamp(timestamp, window_minutes)
        demand_id = str(event.get("event_id", event.get("unique_id", ""))).strip() or f"REQ_{event['time_slot']}"
        priority = _normalize_priority(ground_truth_priority_from_record(event))
        demand = event_record_to_solver_demand(event, base_date=base_date)
        demand["demand_id"] = demand_id

        windows_map.setdefault(label, []).append(demand)
        priority_map.setdefault(label, []).append((demand_id, priority))

    windows = []
    weight_configs: Dict[str, Dict] = {}
    for label in sorted(windows_map.keys()):
        demands = windows_map[label]
        configs = []
        for rank, (demand_id, priority) in enumerate(
            sorted(priority_map[label], key=lambda item: (item[1], item[0])),
            start=1,
        ):
            configs.append({
                "demand_id": demand_id,
                "priority": priority,
                "window_rank": rank,
                "reasoning": "Seed priority from event manifest",
            })
        windows.append({"time_window": label, "demands": demands})
        weight_configs[label] = {
            "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
            "demand_configs": configs,
            "supplementary_constraints": [],
        }

    return windows, weight_configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct baseline that solves the shared seed event manifest priorities.",
    )
    parser.add_argument(
        "--csv",
        default=str(DEMAND_EVENTS_MANIFEST_PATH),
        help="Path to the rich event manifest (CSV fallback still supported)",
    )
    parser.add_argument("--output-dir", default="results", help="Output root directory")
    parser.add_argument("--stations", default=str(STATION_DATA_PATH), help="Path to station metadata")
    parser.add_argument("--building-data", default=str(BUILDING_DATA_PATH), help="Path to building data")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--base-date", default="2024-03-15")
    parser.add_argument("--n-events", type=int, default=None)
    parser.add_argument("--time-slots", type=int, nargs="+", default=None)
    parser.add_argument("--time-limit", type=int, default=10)
    parser.add_argument("--max-solver-stations", type=int, default=1)
    parser.add_argument("--max-drones-per-station", type=int, default=3)
    parser.add_argument("--max-payload", type=float, default=60.0)
    parser.add_argument("--max-range", type=float, default=200000.0)
    parser.add_argument("--noise-weight", type=float, default=0.5)
    parser.add_argument("--drone-speed", type=float, default=60.0)
    parser.add_argument("--pareto-scan", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-conflict-refiner", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    windows, weight_configs = build_seed_priority_inputs(
        csv_path=args.csv,
        base_date=args.base_date,
        window_minutes=args.window,
        n_events=args.n_events,
        time_slots=args.time_slots,
    )

    run_dir = _build_run_dir(Path(args.output_dir), args.noise_weight)
    results = solve_windows_dynamically(
        windows=windows,
        weight_configs=weight_configs,
        stations_path=args.stations,
        building_path=args.building_data,
        max_solver_stations=args.max_solver_stations,
        time_limit=args.time_limit,
        max_drones_per_station=args.max_drones_per_station,
        max_payload=args.max_payload,
        max_range=args.max_range,
        noise_weight=args.noise_weight,
        drone_speed=args.drone_speed,
        analytics_output_dir=str(run_dir / "solver_analytics"),
        enable_conflict_refiner=args.enable_conflict_refiner,
    )
    if args.pareto_scan:
        run_multiobjective_pareto_scan(
            windows=windows,
            weight_configs=weight_configs,
            stations_path=args.stations,
            building_path=args.building_data,
            max_solver_stations=args.max_solver_stations,
            time_limit=args.time_limit,
            max_drones_per_station=args.max_drones_per_station,
            max_payload=args.max_payload,
            max_range=args.max_range,
            noise_weight=args.noise_weight,
            drone_speed=args.drone_speed,
            analytics_output_dir=str(run_dir / "solver_analytics" / "pareto"),
            enable_conflict_refiner=args.enable_conflict_refiner,
        )

    with open(run_dir / "workflow_results.json", "w", encoding="utf-8") as f:
        json.dump(serialize_workflow_results(results), f, ensure_ascii=False, indent=2)
    with open(run_dir / "weight_configs.json", "w", encoding="utf-8") as f:
        json.dump(weight_configs, f, ensure_ascii=False, indent=2)

    print(f"Direct baseline solved {len(windows)} windows")
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
