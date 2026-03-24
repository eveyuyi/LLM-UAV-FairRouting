"""Baseline solver that reads the shared seed demand events directly."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llm4fairrouting.data.seed_paths import (
    BUILDING_DATA_PATH,
    DEMAND_EVENTS_PATH,
    STATION_DATA_PATH,
)
from llm4fairrouting.llm.dialogue_generation import load_demand_events
from llm4fairrouting.workflow.solver_adapter import (
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


def build_uniform_priority_inputs(
        *,
        csv_path: str,
        base_date: str,
        window_minutes: int,
        n_events: Optional[int] = None,
        time_slots: Optional[List[int]] = None,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    生成所有需求优先级均为1的输入数据。
    内部调用 build_seed_priority_inputs 获得原始 windows 和 weight_configs，
    然后修改 weight_configs，将 priority 统一设为1，并按 demand_id 排序重新分配 window_rank。
    """
    # 1. 调用原函数获取基础数据（需求列表、窗口划分、原始权重配置）
    windows, orig_weight_configs = _build_seed_priority_inputs(
        csv_path=csv_path,
        base_date=base_date,
        window_minutes=window_minutes,
        n_events=n_events,
        time_slots=time_slots
    )

    # 2. 构建统一优先级的权重配置
    uniform_weight_configs = {}
    for label, window_data in zip(orig_weight_configs.keys(), windows):
        demands = window_data["demands"]
        # 提取所有 demand_id，统一优先级为1
        items = [(demand["demand_id"], 1) for demand in demands]
        # 按 demand_id 排序（确保确定性顺序）
        items_sorted = sorted(items, key=lambda x: x[0])
        demand_configs = []
        for rank, (demand_id, pri) in enumerate(items_sorted, start=1):
            demand_configs.append({
                "demand_id": demand_id,
                "priority": pri,
                "window_rank": rank,
                "reasoning": "Uniform priority (all 1) for baseline comparison",
            })
        # 复制原配置中的全局权重和补充约束（深拷贝必要部分，此处均为简单类型，直接 .copy() 即可）
        uniform_weight_configs[label] = {
            "global_weights": orig_weight_configs[label]["global_weights"].copy(),
            "demand_configs": demand_configs,
            "supplementary_constraints": orig_weight_configs[label]["supplementary_constraints"].copy(),
        }

    return windows, uniform_weight_configs


def _build_seed_priority_inputs(
        *,
        csv_path: str,
        base_date: str,
        window_minutes: int,
        n_events: Optional[int] = None,
        time_slots: Optional[List[int]] = None,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    events = load_demand_events(csv_path, n_events=n_events, time_slots=time_slots)
    # 只取前219行
    events = events[:219]

    windows_map: Dict[str, List[Dict]] = {}
    priority_map: Dict[str, List[Tuple[str, int]]] = {}

    for event in events:
        timestamp = datetime.strptime(base_date, "%Y-%m-%d") + timedelta(
            minutes=int(event["time_slot"]) * 5
        )
        label = _window_label_for_timestamp(timestamp, window_minutes)
        demand_id = str(event.get("event_id", event.get("unique_id", ""))).strip() or f"REQ_{event['time_slot']}"
        priority = _normalize_priority(event.get("priority", 4))
        demand = {
            "demand_id": demand_id,
            "request_timestamp": timestamp.isoformat(timespec="seconds"),
            "origin": {
                "fid": str(event.get("supply_fid", "")),
                "coords": [float(event["supply_lon"]), float(event["supply_lat"])],
                "type": "supply_station",
            },
            "destination": {
                "fid": str(event.get("demand_node_id", event.get("demand_fid", ""))),
                "coords": [float(event["demand_lon"]), float(event["demand_lat"])],
                "type": "residential_area",
            },
            "cargo": {
                "type": str(event.get("material_type", "medicine")),
                "weight_kg": float(event.get("quantity_kg", event.get("material_weight", 2.0))),
            },
            "priority_evaluation_signals": {},
        }

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
                "reasoning": "Seed priority from daily_demand_events.csv",
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
        description="Direct baseline that solves the shared seed demand events with CSV priorities.",
    )
    parser.add_argument("--csv", default=str(DEMAND_EVENTS_PATH), help="Path to daily_demand_events.csv")
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
    args = parser.parse_args()

    windows, weight_configs = build_uniform_priority_inputs(
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
    )

    with open(run_dir / "workflow_results.json", "w", encoding="utf-8") as f:
        json.dump(serialize_workflow_results(results), f, ensure_ascii=False, indent=2)
    with open(run_dir / "weight_configs.json", "w", encoding="utf-8") as f:
        json.dump(weight_configs, f, ensure_ascii=False, indent=2)

    print(f"Direct baseline solved {len(windows)} windows")
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    import sys

    sys.argv = [
        'cplex_with_no_priorities.py',
        "--output-dir", "data/result_without_priority",
    ]
    main()
