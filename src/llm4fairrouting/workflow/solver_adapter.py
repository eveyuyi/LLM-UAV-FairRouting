"""
Module 3b: Solver Adapter

负责将 Module 2 的结构化需求和 Module 3a 的权重配置送入求解器。
该模块既可被 run_workflow 复用，也可作为独立脚本运行。

统一使用共享 routing 核心作为求解后端。

求解模型特性（对齐 baseline）：
- 目标函数: ``(distance + noise_weight * noise) * (1/priority)``
- 允许部分需求不被服务（unassigned 变量 + 大罚项 1e9）
- 每架无人机单次求解最多接 1 个需求（single_task_per_drone）
- 需求必须从指定供给点取货（supply_demand_matching）
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time as _time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Ensure CPLEX binary is on PATH (previously handled by drone_cplex_real_data import)
_CPLEX_BIN = "/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx"
if os.path.isdir(_CPLEX_BIN) and _CPLEX_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CPLEX_BIN + ":" + os.environ.get("PATH", "")

from llm4fairrouting.data.seed_paths import (
    BUILDING_DATA_FILENAME,
    BUILDING_DATA_PATH,
    STATION_DATA_FILENAME,
    STATION_DATA_PATH,
)
from llm4fairrouting.data.stations import load_station_data
from llm4fairrouting.routing.domain import Point


SolverPoint = Point


def demands_to_solver_inputs(
    demands: List[Dict],
) -> tuple[list[SolverPoint], list[SolverPoint], list[float], list[int]]:
    """从结构化需求构建 supply_points, demand_points, demand_weights, demand_required_supply。

    Returns
    -------
    supply_points : list[SolverPoint]
    demand_points : list[SolverPoint]
    demand_weights : list[float]
    demand_required_supply : list[int]
        Index into supply_points for each demand's required pickup origin.
    """
    supply_set: Dict[str, SolverPoint] = {}
    supply_order: Dict[str, int] = {}
    demand_points: List[SolverPoint] = []
    demand_weights: List[float] = []
    demand_required_supply: List[int] = []

    for demand in demands:
        origin = demand.get("origin", {})
        origin_fid = str(origin.get("fid", "")).strip()
        origin_coords = origin.get("coords", [0.0, 0.0])
        if origin_fid and origin_fid not in supply_set:
            supply_order[origin_fid] = len(supply_set)
            supply_set[origin_fid] = SolverPoint(
                id=f"S_{origin_fid}",
                lon=float(origin_coords[0]),
                lat=float(origin_coords[1]),
                alt=50.0,
                type="supply",
            )

        dest = demand.get("destination", {})
        dest_fid = str(dest.get("fid", len(demand_points) + 1))
        dest_coords = dest.get("coords", [0.0, 0.0])
        demand_points.append(SolverPoint(
            id=f"D_{dest_fid}",
            lon=float(dest_coords[0]),
            lat=float(dest_coords[1]),
            alt=50.0,
            type="demand",
        ))

        cargo = demand.get("cargo", {})
        demand_weights.append(float(cargo.get("weight_kg", 2.0)))

        if origin_fid and origin_fid in supply_order:
            demand_required_supply.append(supply_order[origin_fid])
        else:
            demand_required_supply.append(0)

    return list(supply_set.values()), demand_points, demand_weights, demand_required_supply


def create_mock_stations(n: int = 3) -> list[SolverPoint]:
    """创建 mock 站点，在没有真实站点文件时兜底。"""
    stations = [
        SolverPoint(id="L1", lon=113.85, lat=22.68, alt=50.0, type="station"),
        SolverPoint(id="L2", lon=113.92, lat=22.68, alt=50.0, type="station"),
        SolverPoint(id="L3", lon=113.82, lat=22.67, alt=50.0, type="station"),
    ]
    return stations[:n]

def _select_station_dicts(
    stations: List[Dict],
    demands: List[Dict],
    max_stations: Optional[int],
) -> List[Dict]:
    """Preserve source order and take the first N stations, matching the baseline demo."""
    if not stations:
        return []
    if max_stations is None or max_stations <= 0 or len(stations) <= max_stations:
        return list(stations)
    return list(stations[:max_stations])


def _load_station_dicts(station_path: str) -> List[Dict]:
    df = load_station_data(station_path)

    stations: List[Dict] = []
    for _, row in df.iterrows():
        lat_val = row["latitude"]
        lon_val = row["longitude"]
        try:
            lat = float(lat_val)
            lon = float(lon_val)
        except (TypeError, ValueError):
            continue
        if math.isnan(lat) or math.isnan(lon):
            continue

        station_idx = len(stations) + 1
        name_val = row["station_name"] if "station_name" in row.index else f"L{station_idx}"
        name = str(name_val).strip() or f"L{station_idx}"
        stations.append({
            "station_id": f"L{station_idx}",
            "name": name,
            "lon": lon,
            "lat": lat,
        })

    return stations


def load_solver_station_points(
    stations_path: Optional[str],
    demands: List[Dict],
    max_stations: Optional[int] = 1,
) -> List[SolverPoint]:
    """加载求解器站点。

    默认使用项目 seed 真实站点，并与 baseline demo 一样按源数据顺序截取。
    """
    resolved_path = str(stations_path or STATION_DATA_PATH)

    try:
        station_dicts = _load_station_dicts(resolved_path)
    except Exception as exc:
        count = max_stations if max_stations and max_stations > 0 else 3
        print(f"  真实站点文件加载失败 ({exc})，回退到 mock stations")
        return create_mock_stations(count)

    selected = _select_station_dicts(station_dicts, demands, max_stations)
    if len(selected) < len(station_dicts):
        print(
            f"  站点文件共 {len(station_dicts)} 个站点，"
            f"按 baseline demo 口径取前 {len(selected)} 个参与求解"
        )
    else:
        print(f"  复用真实站点 {len(selected)} 个参与求解")

    return [
        SolverPoint(
            id=str(station["station_id"]),
            lon=float(station["lon"]),
            lat=float(station["lat"]),
            alt=50.0,
            type="station",
        )
        for station in selected
    ]


DEFAULT_DRONE_SPEED_MS = 60.0
DEFAULT_SIMULATION_PADDING_H = 24.0


def _parse_window_bounds(time_window: str) -> Tuple[datetime, datetime]:
    """Parse labels like ``2024-03-15T00:00-00:05`` into datetimes."""
    date_part, span = time_window.split("T", 1)
    start_str, end_str = span.split("-", 1)
    start_dt = datetime.fromisoformat(f"{date_part}T{start_str}")

    end_hour, end_minute = [int(token) for token in end_str.split(":")]
    if end_hour == 24:
        end_dt = datetime.fromisoformat(f"{date_part}T00:00") + timedelta(days=1)
    else:
        end_dt = datetime.fromisoformat(f"{date_part}T{end_str}")
        if end_dt < start_dt:
            end_dt += timedelta(days=1)
    return start_dt, end_dt


def _resolve_request_datetime(demand: Dict, window_start: datetime) -> datetime:
    ts = demand.get("request_timestamp")
    if ts:
        try:
            return datetime.fromisoformat(str(ts))
        except ValueError:
            pass
    return window_start


def _point_key(fid: object, coords: List[float], fallback: str) -> str:
    fid_text = str(fid).strip() if fid is not None else ""
    if fid_text:
        return f"fid:{fid_text}"
    if len(coords) == 2:
        lon = round(float(coords[0]), 7)
        lat = round(float(coords[1]), 7)
        return f"coords:{lon}:{lat}"
    return fallback


def _build_window_payloads(
    windows: List[Dict],
    weight_configs: Dict[str, Dict],
    max_payload: float,
) -> List[Dict]:
    payloads: List[Dict] = []

    for w_idx, window in enumerate(windows):
        time_window = window.get("time_window", f"window_{w_idx}")
        demands = window.get("demands", [])
        if not demands:
            continue

        weight_config = weight_configs.get(time_window)
        if weight_config is None:
            print(f"窗口 {time_window} 缺少权重配置，跳过")
            continue

        window_weight_config = copy.deepcopy(weight_config)
        feasible_demands = [
            copy.deepcopy(demand)
            for demand in demands
            if demand.get("cargo", {}).get("weight_kg", 0.0) <= max_payload
        ]
        skipped = len(demands) - len(feasible_demands)
        if skipped:
            print(f"  窗口 {time_window}: 跳过 {skipped} 条超载需求 (>{max_payload}kg)")

        feasible_ids = {demand["demand_id"] for demand in feasible_demands}
        window_weight_config["demand_configs"] = [
            config for config in window_weight_config.get("demand_configs", [])
            if config["demand_id"] in feasible_ids
        ]

        start_dt, end_dt = _parse_window_bounds(time_window)
        payloads.append({
            "time_window": time_window,
            "window_start_dt": start_dt,
            "window_end_dt": end_dt,
            "weight_config": window_weight_config,
            "feasible_demands": feasible_demands,
            "n_demands_total": len(demands),
            "n_demands_filtered": skipped,
        })

    payloads.sort(key=lambda payload: payload["window_start_dt"])
    return payloads


def solve_windows_dynamically(
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
    drone_speed: float = DEFAULT_DRONE_SPEED_MS,
) -> List[Dict]:
    """Use the shared routing core to solve the full window sequence."""
    window_payloads = _build_window_payloads(
        windows=windows,
        weight_configs=weight_configs,
        max_payload=max_payload,
    )
    if not window_payloads:
        return []

    all_feasible_demands = [
        demand
        for payload in window_payloads
        for demand in payload["feasible_demands"]
    ]
    if not all_feasible_demands:
        return [
            {
                "time_window": payload["time_window"],
                "weight_config": payload["weight_config"],
                "feasible_demands": [],
                "n_demands_total": payload["n_demands_total"],
                "n_demands_filtered": payload["n_demands_filtered"],
                "solution": None,
                "n_supply": 0,
            }
            for payload in window_payloads
        ]

    station_points = load_solver_station_points(
        stations_path=stations_path,
        demands=all_feasible_demands,
        max_stations=max_solver_stations,
    )
    if not station_points:
        return [
            {
                "time_window": payload["time_window"],
                "weight_config": payload["weight_config"],
                "feasible_demands": payload["feasible_demands"],
                "n_demands_total": payload["n_demands_total"],
                "n_demands_filtered": payload["n_demands_filtered"],
                "solution": None,
                "n_supply": 0,
            }
            for payload in window_payloads
        ]

    try:
        import numpy as np
        from scipy.spatial import cKDTree

        from llm4fairrouting.data.building_information import load_building_partitions
        from llm4fairrouting.routing.domain import (
            DemandEvent as _DemandEvent,
            Point as _RoutingPoint,
            create_drones,
        )
        from llm4fairrouting.routing.path_costs import (
            FLIGHT_HEIGHT as _FLIGHT_HEIGHT,
            build_realistic_distance_and_noise_matrices,
            create_obstacles_from_buildings,
        )
        from llm4fairrouting.routing.simulator import (
            FinalDroneSimulator as _FinalDroneSimulator,
        )
    except Exception as exc:
        raise RuntimeError(f"动态求解依赖加载失败: {exc}") from exc

    building_source = str(building_path or BUILDING_DATA_PATH)
    if not Path(building_source).exists():
        raise FileNotFoundError(
            f"动态求解需要建筑数据文件，未找到: {building_source}"
        )

    supply_points: List = []
    demand_points: List = []
    supply_key_to_idx: Dict[str, int] = {}
    demand_key_to_idx: Dict[str, int] = {}
    event_records: List[Dict] = []

    for payload in window_payloads:
        demand_cfg_map = {
            config["demand_id"]: config
            for config in payload["weight_config"].get("demand_configs", [])
        }
        for idx, demand in enumerate(payload["feasible_demands"]):
            origin = demand.get("origin", {})
            origin_coords = origin.get("coords", [])
            origin_key = _point_key(
                origin.get("fid"),
                origin_coords,
                fallback=f"{payload['time_window']}:origin:{idx}",
            )
            if origin_key not in supply_key_to_idx:
                supply_key_to_idx[origin_key] = len(supply_points)
                supply_points.append(
                    _RoutingPoint(
                        id=f"S_{origin.get('fid', len(supply_points) + 1)}",
                        lon=float(origin_coords[0]),
                        lat=float(origin_coords[1]),
                        alt=_FLIGHT_HEIGHT,
                        type="supply",
                    )
                )

            dest = demand.get("destination", {})
            dest_coords = dest.get("coords", [])
            dest_key = _point_key(
                dest.get("fid"),
                dest_coords,
                fallback=f"{payload['time_window']}:destination:{idx}",
            )
            if dest_key not in demand_key_to_idx:
                demand_key_to_idx[dest_key] = len(demand_points)
                demand_points.append(
                    _RoutingPoint(
                        id=f"D_{dest.get('fid', len(demand_points) + 1)}",
                        lon=float(dest_coords[0]),
                        lat=float(dest_coords[1]),
                        alt=_FLIGHT_HEIGHT,
                        type="demand",
                    )
                )

            original_demand_id = demand.get("demand_id", f"D{idx + 1}")
            event_id = f"{payload['time_window']}::{original_demand_id}::{idx}"
            demand["solver_event_id"] = event_id

            cfg = demand_cfg_map.get(original_demand_id, {})
            event_records.append({
                "event_id": event_id,
                "original_demand_id": original_demand_id,
                "time_window": payload["time_window"],
                "request_dt": _resolve_request_datetime(
                    demand, payload["window_start_dt"]
                ),
                "weight": float(demand.get("cargo", {}).get("weight_kg", 0.0)),
                "priority": int(cfg.get("priority", 3)),
                "required_supply_idx": supply_key_to_idx[origin_key],
                "demand_point_idx": demand_key_to_idx[dest_key],
                "demand_point_id": demand_points[demand_key_to_idx[dest_key]].id,
            })

    if not supply_points or not demand_points:
        return [
            {
                "time_window": payload["time_window"],
                "weight_config": payload["weight_config"],
                "feasible_demands": payload["feasible_demands"],
                "n_demands_total": payload["n_demands_total"],
                "n_demands_filtered": payload["n_demands_filtered"],
                "solution": None,
                "n_supply": len(supply_points),
            }
            for payload in window_payloads
        ]

    station_cplex_points = [
        _RoutingPoint(
            id=station.id,
            lon=station.lon,
            lat=station.lat,
            alt=_FLIGHT_HEIGHT,
            type="station",
        )
        for station in station_points
    ]

    all_points = supply_points + demand_points + station_cplex_points
    ref_lat = float(np.mean([point.lat for point in all_points]))
    ref_lon = float(np.mean([point.lon for point in all_points]))
    for point in all_points:
        point.to_enu(ref_lat, ref_lon, 0.0)

    _, residences, all_buildings = load_building_partitions(building_source)
    selected_coords = {(point.lon, point.lat) for point in all_points}
    obstacles_raw = create_obstacles_from_buildings(
        all_buildings,
        selected_coords,
        min_obstacle_height=30.0,
        obstacle_radius=20.0,
    )
    residential_positions = []
    for _, row in residences.iterrows():
        residence = _RoutingPoint(
            id="",
            lon=float(row["longitude"]),
            lat=float(row["latitude"]),
            alt=float(row["ground_elevation_m"]),
            type="residential",
        )
        residence.to_enu(ref_lat, ref_lon, 0.0)
        residential_positions.append([residence.x, residence.y, residence.z])

    residential_positions_np = np.array(residential_positions, dtype=float)
    residential_tree = cKDTree(residential_positions_np)
    dist_matrix, noise_cost_matrix = build_realistic_distance_and_noise_matrices(
        task_points=all_points,
        obstacles_raw=obstacles_raw,
        residential_positions=residential_positions_np,
        residential_tree=residential_tree,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        flight_height=_FLIGHT_HEIGHT,
    )

    base_dt = min(record["request_dt"] for record in event_records)
    n_supply = len(supply_points)
    demand_events: List = []
    for record in event_records:
        event_time_h = max(
            0.0,
            (record["request_dt"] - base_dt).total_seconds() / 3600.0,
        )
        demand_events.append(
            _DemandEvent(
                time=event_time_h,
                node_idx=n_supply + record["demand_point_idx"],
                weight=record["weight"],
                unique_id=record["event_id"],
                priority=record["priority"],
                required_supply_idx=record["required_supply_idx"],
                demand_point_id=record["demand_point_id"],
            )
        )
        record["event_time_h"] = event_time_h

    drones = create_drones(
        station_cplex_points,
        drones_per_station=max_drones_per_station,
        max_payload=max_payload,
        max_range=max_range,
        speed=drone_speed,
    )
    simulator = _FinalDroneSimulator(
        supply_points=supply_points,
        demand_points=demand_points,
        station_points=station_cplex_points,
        drones_static=drones,
        dist_matrix=dist_matrix,
        demand_events=demand_events,
        noise_cost_matrix=noise_cost_matrix,
        noise_weight=noise_weight,
        time_step=0.001,
        solve_interval=0.05,
        time_limit=time_limit,
    )

    window_snapshots: Dict[str, Dict] = {}
    for payload in window_payloads:
        end_time_h = max(
            0.0,
            (payload["window_end_dt"] - base_dt).total_seconds() / 3600.0,
        )
        payload["window_end_h"] = end_time_h

        step_started_at = _time.time()
        simulator.advance_to(end_time_h)
        snapshot = simulator.snapshot_state()
        snapshot["solve_time_s"] = round(_time.time() - step_started_at, 3)
        window_snapshots[payload["time_window"]] = snapshot

    last_event_time_h = max(record["event_time_h"] for record in event_records)
    simulator.run_until_complete(last_event_time_h + DEFAULT_SIMULATION_PADDING_H)

    demand_event_lookup = {event.unique_id: event for event in demand_events}
    simulation_completed = simulator.is_complete()
    results: List[Dict] = []

    for payload in window_payloads:
        snapshot = window_snapshots[payload["time_window"]]
        demand_event_results: Dict[str, Dict] = {}
        for demand in payload["feasible_demands"]:
            event_id = demand["solver_event_id"]
            event = demand_event_lookup[event_id]
            served_time_s = (
                round(event.served_time * 3600.0, 1)
                if event.served_time is not None else None
            )
            demand_event_results[event_id] = {
                "event_id": event_id,
                "assigned_drone": event.assigned_drone,
                "assigned_time_h": event.assigned_time,
                "served_time_h": event.served_time,
                "served_time_s": served_time_s,
                "required_supply_idx": event.required_supply_idx,
                "supply_node": event.supply_node,
                "demand_point_id": event.demand_point_id,
                "is_assigned_by_snapshot": event_id in snapshot["assigned_ids"],
                "is_served_by_snapshot": event_id in snapshot["served_ids"],
                "is_served_eventually": event.served_time is not None,
            }

        results.append({
            "time_window": payload["time_window"],
            "weight_config": payload["weight_config"],
            "feasible_demands": payload["feasible_demands"],
            "n_demands_total": payload["n_demands_total"],
            "n_demands_filtered": payload["n_demands_filtered"],
            "solution": {
                "solve_mode": "dynamic_periodic",
                "solve_status": "completed" if simulation_completed else "max_time_limit",
                "solve_time_s": snapshot["solve_time_s"],
                "drone_speed_ms": drone_speed,
                "snapshot_time_h": snapshot["current_time_h"],
                "snapshot_time_window_end": payload["window_end_dt"].isoformat(),
                "snapshot_served_ids": snapshot["served_ids"],
                "snapshot_assigned_ids": snapshot["assigned_ids"],
                "snapshot_pending_ids": snapshot["pending_ids"],
                "busy_drones": snapshot["busy_drones"],
                "total_distance": snapshot["total_distance_m"],
                "total_noise_impact": snapshot["total_noise_impact"],
                "objective_value": None,
                "demand_event_results": demand_event_results,
            },
            "n_supply": n_supply,
        })

    return results


def _build_simplified_noise_matrix(dist_matrix):
    """Build a simplified noise cost matrix when detailed building data is unavailable.

    Approximates the number of affected residential buildings per flight segment
    as proportional to path length (~1 building per 200 m of urban flight).
    This serves as a lightweight stand-in for the full RRT+building noise
    workflow in ``llm4fairrouting.routing.path_costs.build_realistic_distance_and_noise_matrices``.
    """
    import numpy as np

    n = dist_matrix.shape[0]
    noise = np.zeros((n, n), dtype=int)
    scale = 200.0
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            if d > 0:
                noise[i, j] = max(1, int(d / scale))
                noise[j, i] = noise[i, j]
    return noise


def _build_solution_from_assignments(
    assignments: List[Dict],
    demand_events: list,
    dist_matrix,
    cplex_points: list,
    station_indices: List[int],
    n_supply: int,
    cruise_speed: float,
    solve_time: float,
) -> Dict:
    """Convert CplexSolver assignment list to the solution dict expected by
    ``serialize_workflow_results``.

    Each assignment maps one drone to one demand (single-task-per-drone model).
    Path per drone: station -> supply -> demand -> station.
    """
    if not assignments:
        return {
            "drones_used": 0,
            "total_distance": 0.0,
            "objective_value": 0.0,
            "solve_time_s": round(solve_time, 3),
            "solve_status": "feasible",
            "assignments": [],
            "paths": [],
        }

    sol_assignments: List[Dict] = []
    sol_paths: List[list] = []
    total_distance = 0.0
    used_drones: set = set()

    for assign in assignments:
        ds = assign["drone"]
        demand = assign["demand"]
        supply_node = assign["supply_node"]
        demand_node = demand.node_idx
        station_node = station_indices[ds.station_id]

        used_drones.add(ds.drone_id)

        path = [station_node, supply_node, demand_node, station_node]
        path_labels = [cplex_points[node].id for node in path]

        delivery_times: Dict[int, float] = {}
        cumulative = 0.0
        for k in range(len(path) - 1):
            seg = float(dist_matrix[path[k], path[k + 1]])
            cumulative += seg / cruise_speed
            delivery_times[path[k + 1]] = round(cumulative, 1)

        drone_dist = sum(
            float(dist_matrix[path[k], path[k + 1]])
            for k in range(len(path) - 1)
        )
        total_distance += drone_dist

        sol_assignments.append({
            "drone_id": ds.drone_id,
            "station_id": ds.station_id,
            "station_name": cplex_points[station_node].id,
            "supply_idx": assign["supply_idx"],
            "demand_indices": [demand_node],
            "path_labels": path_labels,
            "path_str": " -> ".join(path_labels),
            "demand_delivery_times_s": delivery_times,
            "total_mission_time_s": round(cumulative, 1),
        })
        sol_paths.append(path)

    return {
        "drones_used": len(used_drones),
        "total_distance": total_distance,
        "objective_value": total_distance,
        "solve_time_s": round(solve_time, 3),
        "solve_status": "optimal",
        "assignments": sol_assignments,
        "paths": sol_paths,
    }


def solve_window_demands(
    *,
    time_window: str,
    demands: List[Dict],
    weight_config: Dict,
    stations_path: Optional[str],
    max_solver_stations: Optional[int],
    time_limit: int,
    max_drones_per_station: int,
    max_payload: float,
    max_range: float,
    noise_weight: float,
    drone_speed: float = DEFAULT_DRONE_SPEED_MS,
    building_path: Optional[str] = None,
) -> Dict:
    """单窗口兼容入口，内部复用动态周期重求解核心。"""
    results = solve_windows_dynamically(
        windows=[{"time_window": time_window, "demands": demands}],
        weight_configs={time_window: weight_config},
        stations_path=stations_path,
        building_path=building_path,
        max_solver_stations=max_solver_stations,
        time_limit=time_limit,
        max_drones_per_station=max_drones_per_station,
        max_payload=max_payload,
        max_range=max_range,
        noise_weight=noise_weight,
        drone_speed=drone_speed,
    )
    if results:
        return results[0]

    return {
        "time_window": time_window,
        "weight_config": copy.deepcopy(weight_config),
        "feasible_demands": [],
        "n_demands_total": len(demands),
        "n_demands_filtered": len(demands),
        "solution": None,
        "n_supply": 0,
    }


def serialize_workflow_results(all_solutions: List[Dict]) -> List[Dict]:
    """将求解结果转换为 JSON 可直接落盘的摘要结构。"""
    serializable: List[Dict] = []

    for solution_entry in all_solutions:
        weight_config = solution_entry["weight_config"]
        solution = solution_entry["solution"]
        feasible_demands = solution_entry.get("feasible_demands", [])
        n_supply = solution_entry.get("n_supply", 0)

        entry: Dict = {
            "time_window": solution_entry["time_window"],
            "n_demands_extracted": solution_entry.get("n_demands_total", len(feasible_demands)),
            "n_demands_feasible": len(feasible_demands),
            "n_demands_filtered": solution_entry.get("n_demands_filtered", 0),
            "has_solution": solution is not None,
            "global_weights": weight_config.get("global_weights", {}),
            "n_supplementary_constraints": len(
                weight_config.get("supplementary_constraints", [])
            ),
        }

        if solution:
            if solution.get("solve_mode") == "dynamic_periodic":
                drone_speed = float(solution.get("drone_speed_ms", DEFAULT_DRONE_SPEED_MS))
                total_dist = float(solution.get("total_distance", 0.0))
                total_noise = float(solution.get("total_noise_impact", 0.0))
                entry.update({
                    "solve_status": solution.get("solve_status", "dynamic_periodic"),
                    "solve_time_s": solution.get("solve_time_s", 0.0),
                    "objective_value": solution.get("objective_value"),
                    "drones_used": len({
                        outcome.get("assigned_drone")
                        for outcome in solution.get("demand_event_results", {}).values()
                        if outcome.get("assigned_drone")
                    }),
                    "total_distance_m": round(total_dist, 2),
                    "total_noise_impact": round(total_noise, 2),
                    "total_estimated_time_s": round(total_dist / drone_speed, 1),
                    "cruise_speed_ms": drone_speed,
                    "snapshot_time_h": solution.get("snapshot_time_h"),
                    "snapshot_time_window_end": solution.get("snapshot_time_window_end"),
                    "busy_drones_at_window_end": solution.get("busy_drones", []),
                })

                config_map = {
                    config["demand_id"]: config
                    for config in weight_config.get("demand_configs", [])
                }
                demand_event_results = solution.get("demand_event_results", {})

                per_demand = []
                for demand in feasible_demands:
                    demand_id = demand.get("demand_id", "")
                    solver_event_id = demand.get("solver_event_id", demand_id)
                    outcome = demand_event_results.get(solver_event_id, {})
                    config = config_map.get(demand_id, {})
                    cargo = demand.get("cargo", {})
                    origin = demand.get("origin", {})
                    dest = demand.get("destination", {})
                    vuln = demand.get("priority_evaluation_signals", {}).get(
                        "population_vulnerability", {}
                    )
                    delivery_time_s = outcome.get("served_time_s")
                    per_demand.append({
                        "demand_id": demand_id,
                        "solver_event_id": solver_event_id,
                        "source_dialogue_id": demand.get("source_dialogue_id"),
                        "request_timestamp": demand.get("request_timestamp"),
                        "demand_tier": demand.get("demand_tier", cargo.get("demand_tier", "")),
                        "cargo_type": cargo.get("type", ""),
                        "cargo_type_cn": cargo.get("type_cn", ""),
                        "weight_kg": cargo.get("weight_kg", 0.0),
                        "temperature_sensitive": cargo.get("temperature_sensitive", False),
                        "alpha": config.get("alpha", 1.0),
                        "beta": config.get("beta", 1.0),
                        "priority": config.get("priority", 3),
                        "llm_reasoning": config.get("reasoning", ""),
                        "elderly_involved": vuln.get("elderly_involved", False),
                        "vulnerable_community": vuln.get("vulnerable_community", False),
                        "time_sensitivity": demand.get("priority_evaluation_signals", {}).get(
                            "time_sensitivity", ""
                        ),
                        "requester_role": demand.get("priority_evaluation_signals", {}).get(
                            "requester_role", ""
                        ),
                        "nearby_facility": demand.get("priority_evaluation_signals", {}).get(
                            "nearby_critical_facility", ""
                        ),
                        "is_assigned_by_window_end": outcome.get("is_assigned_by_snapshot", False),
                        "is_served_by_window_end": outcome.get("is_served_by_snapshot", False),
                        "is_served": outcome.get("is_served_eventually", False),
                        "assigned_drone": outcome.get("assigned_drone"),
                        "assigned_time_h": outcome.get("assigned_time_h"),
                        "delivery_time_h": outcome.get("served_time_h"),
                        "delivery_time_s": delivery_time_s,
                        "delivery_time_min": (
                            round(delivery_time_s / 60, 2)
                            if delivery_time_s is not None else None
                        ),
                        "origin_fid": origin.get("fid"),
                        "origin_coords": origin.get("coords", []),
                        "dest_fid": dest.get("fid"),
                        "dest_coords": dest.get("coords", []),
                        "dest_type": dest.get("type", ""),
                    })

                n_served_final = sum(1 for demand in per_demand if demand["is_served"])
                n_served_snapshot = sum(
                    1 for demand in per_demand if demand["is_served_by_window_end"]
                )
                entry["n_demands_served"] = n_served_final
                entry["n_demands_served_by_window_end"] = n_served_snapshot
                entry["service_rate"] = (
                    round(n_served_final / len(feasible_demands), 3)
                    if feasible_demands else 0.0
                )
                entry["service_rate_by_window_end"] = (
                    round(n_served_snapshot / len(feasible_demands), 3)
                    if feasible_demands else 0.0
                )
                entry["per_demand_results"] = per_demand

                drone_groups: Dict[str, List[Dict]] = {}
                for demand_result in per_demand:
                    drone_id = demand_result.get("assigned_drone")
                    if not drone_id:
                        continue
                    drone_groups.setdefault(drone_id, []).append(demand_result)

                entry["per_drone_details"] = [
                    {
                        "drone_id": drone_id,
                        "n_demands_assigned_in_window": len(items),
                        "n_demands_served_in_window": sum(
                            1 for item in items if item["is_served"]
                        ),
                        "demand_ids_served": [
                            item["demand_id"] for item in items if item["is_served"]
                        ],
                        "demand_ids_assigned": [item["demand_id"] for item in items],
                        "latest_delivery_time_h": max(
                            (
                                item["delivery_time_h"]
                                for item in items
                                if item["delivery_time_h"] is not None
                            ),
                            default=None,
                        ),
                    }
                    for drone_id, items in sorted(drone_groups.items())
                ]
                serializable.append(entry)
                continue

            cruise_speed = 15.0
            total_dist = solution.get("total_distance", 0.0)

            entry.update({
                "solve_status": solution.get("solve_status", "unknown"),
                "solve_time_s": solution.get("solve_time_s", 0.0),
                "objective_value": solution.get("objective_value", total_dist),
                "drones_used": solution.get("drones_used", 0),
                "total_distance_m": round(total_dist, 2),
                "total_estimated_time_s": round(total_dist / cruise_speed, 1),
                "cruise_speed_ms": cruise_speed,
            })

            config_map = {
                config["demand_id"]: config
                for config in weight_config.get("demand_configs", [])
            }

            served_map: Dict[int, str] = {}
            delivery_time_map: Dict[int, float] = {}
            for assignment in solution.get("assignments", []):
                delivery_times = assignment.get("demand_delivery_times_s", {})
                for node_idx in assignment.get("demand_indices", []):
                    local_idx = node_idx - n_supply
                    served_map[local_idx] = assignment["drone_id"]
                    if node_idx in delivery_times:
                        delivery_time_map[local_idx] = delivery_times[node_idx]

            per_demand = []
            for idx, demand in enumerate(feasible_demands):
                demand_id = demand.get("demand_id", f"D{idx + 1}")
                config = config_map.get(demand_id, {})
                cargo = demand.get("cargo", {})
                origin = demand.get("origin", {})
                dest = demand.get("destination", {})
                vuln = demand.get("priority_evaluation_signals", {}).get(
                    "population_vulnerability", {}
                )
                per_demand.append({
                    "demand_id": demand_id,
                    "source_dialogue_id": demand.get("source_dialogue_id"),
                    "demand_tier": demand.get("demand_tier", cargo.get("demand_tier", "")),
                    "cargo_type": cargo.get("type", ""),
                    "cargo_type_cn": cargo.get("type_cn", ""),
                    "weight_kg": cargo.get("weight_kg", 0.0),
                    "temperature_sensitive": cargo.get("temperature_sensitive", False),
                    "alpha": config.get("alpha", 1.0),
                    "beta": config.get("beta", 1.0),
                    "priority": config.get("priority", 3),
                    "llm_reasoning": config.get("reasoning", ""),
                    "elderly_involved": vuln.get("elderly_involved", False),
                    "vulnerable_community": vuln.get("vulnerable_community", False),
                    "time_sensitivity": demand.get("priority_evaluation_signals", {}).get(
                        "time_sensitivity", ""
                    ),
                    "requester_role": demand.get("priority_evaluation_signals", {}).get(
                        "requester_role", ""
                    ),
                    "nearby_facility": demand.get("priority_evaluation_signals", {}).get(
                        "nearby_critical_facility", ""
                    ),
                    "is_served": idx in served_map,
                    "assigned_drone": served_map.get(idx),
                    "delivery_time_s": delivery_time_map.get(idx),
                    "delivery_time_min": (
                        round(delivery_time_map[idx] / 60, 2)
                        if idx in delivery_time_map else None
                    ),
                    "origin_fid": origin.get("fid"),
                    "origin_coords": origin.get("coords", []),
                    "dest_fid": dest.get("fid"),
                    "dest_coords": dest.get("coords", []),
                    "dest_type": dest.get("type", ""),
                })

            n_served = sum(1 for demand in per_demand if demand["is_served"])
            entry["n_demands_served"] = n_served
            entry["service_rate"] = (
                round(n_served / len(feasible_demands), 3)
                if feasible_demands else 0.0
            )
            entry["per_demand_results"] = per_demand

            per_drone = []
            paths = solution.get("paths", [])
            for idx_assignment, assignment in enumerate(solution.get("assignments", [])):
                local_idxs = [
                    node_idx - n_supply
                    for node_idx in assignment.get("demand_indices", [])
                ]
                total_weight = sum(
                    feasible_demands[local_idx].get("cargo", {}).get("weight_kg", 0.0)
                    for local_idx in local_idxs
                    if 0 <= local_idx < len(feasible_demands)
                )
                demand_ids_served = [
                    feasible_demands[local_idx].get("demand_id", f"D{local_idx + 1}")
                    for local_idx in local_idxs
                    if 0 <= local_idx < len(feasible_demands)
                ]
                per_drone.append({
                    "drone_id": assignment.get("drone_id"),
                    "station_name": assignment.get("station_name", ""),
                    "station_id": assignment.get("station_id", -1),
                    "supply_idx": assignment.get("supply_idx", -1),
                    "n_demands_served": len(demand_ids_served),
                    "total_weight_kg": round(total_weight, 3),
                    "demand_ids_served": demand_ids_served,
                    "path_str": assignment.get("path_str", ""),
                    "path_labels": assignment.get("path_labels", []),
                    "path_node_indices": paths[idx_assignment] if idx_assignment < len(paths) else [],
                    "total_mission_time_s": assignment.get("total_mission_time_s"),
                    "total_mission_time_min": (
                        round(assignment["total_mission_time_s"] / 60, 2)
                        if assignment.get("total_mission_time_s") else None
                    ),
                })

            entry["per_drone_details"] = per_drone

        serializable.append(entry)

    return serializable


def _load_weight_configs(source_path: str) -> Dict[str, Dict]:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"权重配置不存在: {source_path}")

    configs: Dict[str, Dict] = {}

    if source.is_dir():
        json_files = sorted(source.glob("*.json"))
        for path in json_files:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            time_window = config.get("time_window")
            if time_window:
                configs[time_window] = config
        return configs

    with open(source, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        payload = [payload]

    for config in payload:
        time_window = config.get("time_window")
        if time_window:
            configs[time_window] = config

    return configs


def main():
    parser = argparse.ArgumentParser(description="Module 3b: Solver Adapter")
    parser.add_argument(
        "--demands",
        type=str,
        default=str(PROJECT_ROOT / "data" / "drone" / "extracted_demands.json"),
        help="Module 2 输出的 extracted_demands.json",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Module 3a 输出的权重配置（JSON 文件或目录）",
    )
    parser.add_argument(
        "--stations",
        type=str,
        default=str(STATION_DATA_PATH),
        help=f"真实站点文件 {STATION_DATA_FILENAME}；默认使用项目 seed 数据",
    )
    parser.add_argument(
        "--building-data",
        type=str,
        default=str(BUILDING_DATA_PATH),
        help=f"真实建筑文件 {BUILDING_DATA_FILENAME}；用于构建真实距离/噪声矩阵",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "drone" / "solver_results.json"),
        help="求解结果输出路径",
    )
    parser.add_argument(
        "--time-limit", type=int, default=10, help="求解器时间限制（秒）"
    )
    parser.add_argument(
        "--max-drones-per-station", type=int, default=3, help="每个站点最大无人机数"
    )
    parser.add_argument(
        "--max-payload", type=float, default=60.0, help="最大载重（kg）"
    )
    parser.add_argument(
        "--max-range", type=float, default=200000.0, help="最大航程（m）"
    )
    parser.add_argument(
        "--drone-speed", type=float, default=DEFAULT_DRONE_SPEED_MS, help="无人机飞行速度（m/s）"
    )
    parser.add_argument(
        "--noise-weight",
        type=float,
        default=0.5,
        help="噪声成本权重（>0 启用噪声优先级）",
    )
    parser.add_argument(
        "--max-solver-stations",
        type=int,
        default=1,
        help="求解时最多使用多少个真实站点；默认 1 以对齐 baseline demo，0 表示使用全部",
    )
    args = parser.parse_args()

    with open(args.demands, "r", encoding="utf-8") as f:
        windows = json.load(f)

    weight_configs = _load_weight_configs(args.weights)
    all_solutions = solve_windows_dynamically(
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

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialize_workflow_results(all_solutions), f, ensure_ascii=False, indent=2)

    print(f"结果保存至 {out_path}")


if __name__ == "__main__":
    main()


serialize_pipeline_results = serialize_workflow_results
