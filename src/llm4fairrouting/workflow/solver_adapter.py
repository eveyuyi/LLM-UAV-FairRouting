"""
Module 3b: Solver Adapter

负责将 Module 2 的结构化需求和 Module 3a 的权重配置送入求解器。
该模块既可被 run_workflow 复用，也可作为独立脚本运行。

统一使用共享 routing 核心作为求解后端。

求解模型特性（对齐 baseline）：
- 目标函数: ``distance + noise_weight * noise + drone_activation_cost * used + priority-scaled unassigned penalty``
- 允许部分需求不被服务（unassigned 变量 + 大罚项 1e9）
- 支持 Pickup and Delivery Problem 多任务路径：单架无人机可在一次出行中串联多个取送货任务
- 需求必须从指定供给点取货，并通过路径级配对/顺序/容量/时间约束保证可行
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
from llm4fairrouting.routing.analytics import (
    analyze_pareto_candidates,
    build_default_pareto_profiles,
    compute_pareto_frontier,
    export_visualizations,
    resolve_objective_weights,
    sanitize_label,
    write_json,
)
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
        demand_weights.append(_safe_weight(cargo, default=2.0))

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
    """Parse labels like ``2024-03-15T00:00-00:05`` into datetimes.

    Window labels may carry scenario suffixes such as ``::technical`` or
    ``::direct``. Those suffixes are metadata only and should not affect the
    temporal bounds used by the dynamic solver.
    """
    normalized_window = str(time_window).strip().split("::", 1)[0]
    if "T" not in normalized_window:
        ref = datetime(2024, 3, 15, 0, 0)
        return ref, ref + timedelta(minutes=5)
    date_part, span = normalized_window.split("T", 1)
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


def _safe_weight(cargo: Dict, default: float = 0.0) -> float:
    """Return cargo weight as float, treating None/'unknown'/non-numeric as default."""
    v = cargo.get("weight_kg")
    try:
        return float(v) if v is not None else default
    except (ValueError, TypeError):
        return default


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
            if _safe_weight(demand.get("cargo", {})) <= max_payload
            and len(demand.get("origin", {}).get("coords", [])) >= 2
            and len(demand.get("destination", {}).get("coords", [])) >= 2
        ]
        skipped = len(demands) - len(feasible_demands)
        if skipped:
            print(f"  窗口 {time_window}: 跳过 {skipped} 条无效需求 (超载或坐标缺失)")

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


def _build_objective_weight_schedule(
    window_payloads: List[Dict],
    base_dt: datetime,
) -> List[Dict[str, object]]:
    schedule: List[Dict[str, object]] = []
    for payload in window_payloads:
        start_h = max(
            0.0,
            (payload["window_start_dt"] - base_dt).total_seconds() / 3600.0,
        )
        end_h = max(
            0.0,
            (payload["window_end_dt"] - base_dt).total_seconds() / 3600.0,
        )
        payload["window_start_h"] = start_h
        payload["window_end_h"] = end_h
        schedule.append(
            {
                "time_window": payload["time_window"],
                "start_time_h": start_h,
                "end_time_h": end_h,
                "global_weights": resolve_objective_weights(
                    payload["weight_config"].get("global_weights")
                ),
            }
        )
    return schedule


def _override_global_weights(
    weight_configs: Dict[str, Dict],
    global_weights: Dict[str, float],
) -> Dict[str, Dict]:
    overridden: Dict[str, Dict] = {}
    resolved = resolve_objective_weights(global_weights)
    for time_window, config in weight_configs.items():
        cloned = copy.deepcopy(config)
        cloned["global_weights"] = resolved
        overridden[time_window] = cloned
    return overridden


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
    drone_activation_cost: float = 1000.0,
    drone_speed: float = DEFAULT_DRONE_SPEED_MS,
    analytics_output_dir: Optional[str] = None,
    enable_conflict_refiner: bool = False,
    assignment_solver_factory=None,
    independent_windows: bool = False,
) -> List[Dict]:
    """Solve windows either as a continuous simulation or as independent per-window experiments.

    When ``independent_windows=True`` each window starts with UAVs reset to their
    stations and only the demands that belong to that window are injected.  Delivery
    latency is measured from the window-local t=0, giving realistic sub-hour figures
    that are comparable across windows without cross-window queuing artefacts.
    """
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
            build_lazy_distance_and_noise_matrices,
            create_obstacles_from_buildings,
            export_rrt_paths_for_edges,
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
            deadline_minutes = float(
                demand.get("time_constraint", {}).get("deadline_minutes", 0.0) or 0.0
            )
            event_records.append({
                "event_id": event_id,
                "original_demand_id": original_demand_id,
                "source_event_id": demand.get("source_event_id"),
                "time_window": payload["time_window"],
                "request_dt": _resolve_request_datetime(
                    demand, payload["window_start_dt"]
                ),
                "weight": _safe_weight(demand.get("cargo", {})),
                "priority": int(cfg.get("priority", 3)),
                "deadline_minutes": deadline_minutes,
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
    dist_matrix, noise_cost_matrix = build_lazy_distance_and_noise_matrices(
        task_points=all_points,
        obstacles_raw=obstacles_raw,
        residential_positions=residential_positions_np,
        residential_tree=residential_tree,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        flight_height=_FLIGHT_HEIGHT,
    )

    n_supply = len(supply_points)

    # ── Independent-window mode ────────────────────────────────────────────────
    # Each window gets a fresh simulator with UAVs reset to station; no demand
    # carries over across window boundaries.
    if independent_windows:
        tw_to_records: Dict[str, List[Dict]] = {}
        for record in event_records:
            tw_to_records.setdefault(record["time_window"], []).append(record)

        ind_results: List[Dict] = []
        for payload in window_payloads:
            tw = payload["time_window"]
            window_records = tw_to_records.get(tw, [])
            if not window_records:
                ind_results.append({
                    "time_window": tw,
                    "weight_config": payload["weight_config"],
                    "feasible_demands": payload["feasible_demands"],
                    "n_demands_total": payload["n_demands_total"],
                    "n_demands_filtered": payload["n_demands_filtered"],
                    "solution": None,
                    "n_supply": n_supply,
                })
                continue

            window_base_dt = payload["window_start_dt"]
            window_demand_events = []
            for record in window_records:
                event_time_h = max(
                    0.0,
                    (record["request_dt"] - window_base_dt).total_seconds() / 3600.0,
                )
                record["event_time_h"] = event_time_h
                window_demand_events.append(
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

            fresh_drones = create_drones(
                station_cplex_points,
                drones_per_station=max_drones_per_station,
                max_payload=max_payload,
                max_range=max_range,
                speed=drone_speed,
            )
            fresh_solver = None
            if assignment_solver_factory is not None:
                fresh_solver = assignment_solver_factory(
                    drones=fresh_drones,
                    supply_indices=list(range(n_supply)),
                    station_indices=list(range(
                        n_supply + len(demand_points),
                        n_supply + len(demand_points) + len(station_cplex_points),
                    )),
                    dist_matrix=dist_matrix,
                    all_points=all_points,
                    noise_cost_matrix=noise_cost_matrix,
                    noise_weight=noise_weight,
                    drone_activation_cost=drone_activation_cost,
                    time_limit=time_limit,
                    analytics_output_dir=None,
                    enable_conflict_refiner=enable_conflict_refiner,
                )

            window_sim = _FinalDroneSimulator(
                supply_points=supply_points,
                demand_points=demand_points,
                station_points=station_cplex_points,
                drones_static=fresh_drones,
                dist_matrix=dist_matrix,
                demand_events=window_demand_events,
                noise_cost_matrix=noise_cost_matrix,
                noise_weight=noise_weight,
                drone_activation_cost=drone_activation_cost,
                time_step=0.001,
                solve_interval=0.05,
                time_limit=time_limit,
                objective_weight_schedule=None,
                analytics_output_dir=None,
                enable_conflict_refiner=enable_conflict_refiner,
                assignment_solver=fresh_solver,
            )

            step_started_at = _time.time()
            # 2-hour cap per window; enough for any UAV round-trip
            window_sim.run_until_complete(2.0)
            solve_time_w = round(_time.time() - step_started_at, 3)

            window_drone_paths = window_sim.get_drone_path_details()
            window_analytics = window_sim.get_solver_analytics()
            window_run_summary = window_analytics.get("run_summary", {})

            rrt_edges = []
            for dp in window_drone_paths:
                ni = [int(i) for i in dp.get("path_node_indices", [])]
                rrt_edges.extend((ni[p], ni[p + 1]) for p in range(len(ni) - 1))
            window_run_summary["n_rrt_paths_used"] = len(
                export_rrt_paths_for_edges(dist_matrix, rrt_edges)
            )

            ev_by_id = {ev.unique_id: ev for ev in window_demand_events}
            rec_by_id = {rec["event_id"]: rec for rec in window_records}

            demand_event_results: Dict[str, Dict] = {}
            for demand in payload["feasible_demands"]:
                event_id = demand["solver_event_id"]
                ev = ev_by_id.get(event_id)
                rec = rec_by_id.get(event_id, {})
                if ev is None:
                    continue
                srv_s = round(ev.served_time * 3600.0, 1) if ev.served_time is not None else None
                lat_h = (
                    max(0.0, float(ev.served_time) - float(ev.time))
                    if ev.served_time is not None else None
                )
                lat_s = round(lat_h * 3600.0, 1) if lat_h is not None else None
                demand_event_results[event_id] = {
                    "event_id": event_id,
                    "source_event_id": rec.get("source_event_id") or demand.get("source_event_id"),
                    "assigned_drone": ev.assigned_drone,
                    "assigned_time_h": ev.assigned_time,
                    "request_time_h": ev.time,
                    "served_time_h": ev.served_time,
                    "served_time_s": srv_s,
                    "delivery_latency_h": lat_h,
                    "delivery_latency_s": lat_s,
                    "delivery_latency_min": round(lat_s / 60, 2) if lat_s is not None else None,
                    "deadline_minutes": rec.get("deadline_minutes"),
                    "required_supply_idx": ev.required_supply_idx,
                    "supply_node": ev.supply_node,
                    "demand_point_id": ev.demand_point_id,
                    "is_assigned_by_snapshot": ev.assigned_drone is not None,
                    "is_served_by_snapshot": ev.served_time is not None,
                    "is_served_eventually": ev.served_time is not None,
                }

            served_ids = [ev.unique_id for ev in window_demand_events if ev.served_time is not None]
            assigned_ids = [ev.unique_id for ev in window_demand_events if ev.assigned_drone is not None]
            pending_ids = [ev.unique_id for ev in window_demand_events if ev.served_time is None]
            ind_results.append({
                "time_window": tw,
                "weight_config": payload["weight_config"],
                "feasible_demands": payload["feasible_demands"],
                "n_demands_total": payload["n_demands_total"],
                "n_demands_filtered": payload["n_demands_filtered"],
                "solution": {
                    "solve_mode": "independent_window",
                    "solve_status": "completed" if window_sim.is_complete() else "max_time_limit",
                    "solve_time_s": solve_time_w,
                    "drone_speed_ms": drone_speed,
                    "snapshot_time_h": window_sim.current_time,
                    "snapshot_time_window_end": payload["window_end_dt"].isoformat(),
                    "snapshot_served_ids": served_ids,
                    "snapshot_assigned_ids": assigned_ids,
                    "snapshot_pending_ids": pending_ids,
                    "busy_drones": [],
                    "total_distance": window_sim.total_distance,
                    "total_noise_impact": window_sim.total_noise_impact,
                    "objective_value": None,
                    "demand_event_results": demand_event_results,
                    "drone_path_details": window_drone_paths,
                    "run_summary": window_run_summary,
                    "analytics_artifacts": {},
                },
                "n_supply": n_supply,
            })

        return ind_results
    # ── end independent-window mode ────────────────────────────────────────────

    base_dt = min(record["request_dt"] for record in event_records)
    objective_weight_schedule = _build_objective_weight_schedule(window_payloads, base_dt)
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
    assignment_solver = None
    if assignment_solver_factory is not None:
        assignment_solver = assignment_solver_factory(
            drones=drones,
            supply_indices=list(range(len(supply_points))),
            station_indices=list(range(
                len(supply_points) + len(demand_points),
                len(supply_points) + len(demand_points) + len(station_cplex_points),
            )),
            dist_matrix=dist_matrix,
            all_points=all_points,
            noise_cost_matrix=noise_cost_matrix,
            noise_weight=noise_weight,
            drone_activation_cost=drone_activation_cost,
            time_limit=time_limit,
            analytics_output_dir=analytics_output_dir,
            enable_conflict_refiner=enable_conflict_refiner,
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
        drone_activation_cost=drone_activation_cost,
        time_step=0.001,
        solve_interval=0.05,
        time_limit=time_limit,
        objective_weight_schedule=objective_weight_schedule,
        analytics_output_dir=analytics_output_dir,
        enable_conflict_refiner=enable_conflict_refiner,
        assignment_solver=assignment_solver,
    )

    window_snapshots: Dict[str, Dict] = {}
    for payload in window_payloads:
        step_started_at = _time.time()
        simulator.advance_to(payload["window_end_h"])
        snapshot = simulator.snapshot_state()
        snapshot["solve_time_s"] = round(_time.time() - step_started_at, 3)
        window_snapshots[payload["time_window"]] = snapshot

    last_event_time_h = max(record["event_time_h"] for record in event_records)
    simulator.run_until_complete(last_event_time_h + DEFAULT_SIMULATION_PADDING_H)
    simulator.print_summary()
    final_drone_paths = simulator.get_drone_path_details()
    solver_analytics = simulator.get_solver_analytics()
    run_summary = solver_analytics.get("run_summary", {})
    analytics_artifacts: Dict[str, object] = {}
    rrt_path_edges = []
    for drone_path in final_drone_paths:
        node_indices = [int(idx) for idx in drone_path.get("path_node_indices", [])]
        rrt_path_edges.extend(
            (node_indices[pos], node_indices[pos + 1])
            for pos in range(len(node_indices) - 1)
        )
    rrt_paths_payload = export_rrt_paths_for_edges(dist_matrix, rrt_path_edges)
    run_summary["n_rrt_paths_used"] = len(rrt_paths_payload)
    if analytics_output_dir:
        analytics_dir = Path(analytics_output_dir)
        analytics_dir.mkdir(parents=True, exist_ok=True)
        if rrt_paths_payload:
            rrt_paths_json = write_json(rrt_paths_payload, analytics_dir / "rrt_paths.json")
        else:
            rrt_paths_json = None
        analytics_json_path = write_json(
            solver_analytics,
            analytics_dir / "solver_analytics.json",
        )
        chart_artifacts = export_visualizations(
            analytics=solver_analytics,
            output_dir=analytics_dir / "charts",
        )
        analytics_artifacts = {
            "analytics_json": analytics_json_path,
            "charts": chart_artifacts,
            "rrt_paths_json": rrt_paths_json,
        }

    demand_event_lookup = {event.unique_id: event for event in demand_events}
    event_record_lookup = {record["event_id"]: record for record in event_records}
    simulation_completed = simulator.is_complete()
    results: List[Dict] = []

    for payload in window_payloads:
        snapshot = window_snapshots[payload["time_window"]]
        demand_event_results: Dict[str, Dict] = {}
        for demand in payload["feasible_demands"]:
            event_id = demand["solver_event_id"]
            event = demand_event_lookup[event_id]
            record = event_record_lookup.get(event_id, {})
            served_time_s = (
                round(event.served_time * 3600.0, 1)
                if event.served_time is not None else None
            )
            delivery_latency_h = (
                max(0.0, float(event.served_time) - float(event.time))
                if event.served_time is not None else None
            )
            delivery_latency_s = (
                round(delivery_latency_h * 3600.0, 1)
                if delivery_latency_h is not None else None
            )
            demand_event_results[event_id] = {
                "event_id": event_id,
                "source_event_id": record.get("source_event_id") or demand.get("source_event_id"),
                "assigned_drone": event.assigned_drone,
                "assigned_time_h": event.assigned_time,
                "request_time_h": event.time,
                "served_time_h": event.served_time,
                "served_time_s": served_time_s,
                "delivery_latency_h": delivery_latency_h,
                "delivery_latency_s": delivery_latency_s,
                "deadline_minutes": record.get("deadline_minutes"),
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
                "drone_path_details": final_drone_paths,
                "run_summary": run_summary,
                "analytics_artifacts": analytics_artifacts,
            },
            "n_supply": n_supply,
        })

    return results


def run_multiobjective_pareto_scan(
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
    drone_activation_cost: float = 1000.0,
    drone_speed: float = DEFAULT_DRONE_SPEED_MS,
    analytics_output_dir: Optional[str] = None,
    enable_conflict_refiner: bool = False,
    pareto_profiles: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    if pareto_profiles is None:
        base_weights = None
        if weight_configs:
            first_window = next(iter(weight_configs.values()))
            base_weights = first_window.get("global_weights")
        pareto_profiles = build_default_pareto_profiles(base_weights)

    output_dir = Path(analytics_output_dir) if analytics_output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[Dict[str, object]] = []
    for profile in pareto_profiles:
        profile_id = str(profile.get("profile_id", f"profile_{len(candidates)}"))
        label = str(profile.get("label", profile_id))
        weights = resolve_objective_weights(profile.get("weights"))
        profile_dir = output_dir / "profiles" / sanitize_label(profile_id) if output_dir is not None else None

        results = solve_windows_dynamically(
            windows=windows,
            weight_configs=_override_global_weights(weight_configs, weights),
            stations_path=stations_path,
            building_path=building_path,
            max_solver_stations=max_solver_stations,
            time_limit=time_limit,
            max_drones_per_station=max_drones_per_station,
            max_payload=max_payload,
            max_range=max_range,
            noise_weight=noise_weight,
            drone_activation_cost=drone_activation_cost,
            drone_speed=drone_speed,
            analytics_output_dir=str(profile_dir) if profile_dir is not None else None,
            enable_conflict_refiner=enable_conflict_refiner,
        )

        run_summary: Dict[str, object] = {}
        for result in results:
            solution = result.get("solution") or {}
            run_summary = solution.get("run_summary") or {}
            if run_summary:
                break

        candidate = {
            "profile_id": profile_id,
            "label": label,
            "weights": weights,
            **run_summary,
        }
        if profile_dir is not None:
            candidate["analytics_dir"] = str(profile_dir)
        candidates.append(candidate)

    frontier = compute_pareto_frontier(candidates)
    pareto_analysis = analyze_pareto_candidates(candidates)
    payload = {
        "candidates": candidates,
        "frontier": frontier,
        "pareto_analysis": pareto_analysis,
    }

    if output_dir is not None:
        payload["pareto_json"] = write_json(payload, output_dir / "pareto_frontier.json")
        payload["pareto_analysis_json"] = write_json(pareto_analysis, output_dir / "pareto_analysis.json")
        payload["charts"] = export_visualizations(
            analytics={"solver_calls": [], "gantt_tasks": []},
            output_dir=output_dir / "charts",
            pareto_candidates=candidates,
        )

    return payload


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
                    "run_summary": solution.get("run_summary", {}),
                    "analytics_artifacts": solution.get("analytics_artifacts", {}),
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
                    ) or {}
                    delivery_time_s = outcome.get("served_time_s")
                    delivery_latency_s = outcome.get("delivery_latency_s")
                    deadline_minutes = demand.get("time_constraint", {}).get("deadline_minutes")
                    per_demand.append({
                        "demand_id": demand_id,
                        "solver_event_id": solver_event_id,
                        "source_event_id": demand.get("source_event_id") or outcome.get("source_event_id"),
                        "source_dialogue_id": demand.get("source_dialogue_id"),
                        "request_timestamp": demand.get("request_timestamp"),
                        "demand_tier": demand.get("demand_tier", cargo.get("demand_tier", "")),
                        "cargo_type": cargo.get("type", ""),
                        "cargo_type_cn": cargo.get("type_cn", ""),
                        "weight_kg": cargo.get("weight_kg", 0.0),
                        "temperature_sensitive": cargo.get("temperature_sensitive", False),
                        "priority": config.get("priority", 3),
                        "window_rank": config.get("window_rank"),
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
                        "delivery_latency_h": outcome.get("delivery_latency_h"),
                        "delivery_latency_s": delivery_latency_s,
                        "delivery_latency_min": (
                            round(delivery_latency_s / 60, 2)
                            if delivery_latency_s is not None else None
                        ),
                        "deadline_minutes": deadline_minutes,
                        "is_deadline_met": (
                            delivery_latency_s is not None and deadline_minutes is not None
                            and float(delivery_latency_s) <= float(deadline_minutes) * 60.0
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
                entry["drone_path_details"] = list(solution.get("drone_path_details", []))

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
                    _safe_weight(feasible_demands[local_idx].get("cargo", {}))
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
            entry["drone_path_details"] = per_drone

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
        help="Module 2 output extracted_demands.json",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Module 3a output weight configs (JSON file or directory)",
    )
    parser.add_argument(
        "--stations",
        type=str,
        default=str(STATION_DATA_PATH),
        help=f"Station metadata file ({STATION_DATA_FILENAME})",
    )
    parser.add_argument(
        "--building-data",
        type=str,
        default=str(BUILDING_DATA_PATH),
        help=f"Building data file ({BUILDING_DATA_FILENAME}) used for realistic distance and noise modeling",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "drone" / "solver_results.json"),
        help="Output path for solver results",
    )
    parser.add_argument("--time-limit", type=int, default=10, help="Solver time limit in seconds")
    parser.add_argument("--max-drones-per-station", type=int, default=3, help="Maximum drones per station")
    parser.add_argument("--max-payload", type=float, default=60.0, help="Maximum payload in kg")
    parser.add_argument("--max-range", type=float, default=200000.0, help="Maximum range in meters")
    parser.add_argument(
        "--drone-speed",
        type=float,
        default=DEFAULT_DRONE_SPEED_MS,
        help="Drone speed in meters per second",
    )
    parser.add_argument(
        "--noise-weight",
        type=float,
        default=0.5,
        help="Noise cost weight used by the routing objective",
    )
    parser.add_argument(
        "--drone-activation-cost",
        type=float,
        default=1000.0,
        help="Activation cost per drone; lower values encourage parallel drone usage",
    )
    parser.add_argument(
        "--max-solver-stations",
        type=int,
        default=1,
        help="Maximum number of real stations used during solving; 0 means all stations",
    )
    parser.add_argument(
        "--pareto-scan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the existing weighted-profile Pareto scan",
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
        default="cplex",
        help="Solver backend: exact CPLEX routing, NSGA-III over CPLEX, or NSGA-III over the greedy heuristic backend",
    )
    parser.add_argument("--nsga3-pop-size", type=int, default=20, help="Population size for NSGA-III")
    parser.add_argument("--nsga3-n-generations", type=int, default=10, help="Number of generations for NSGA-III")
    parser.add_argument("--nsga3-seed", type=int, default=42, help="Random seed for NSGA-III")
    parser.add_argument(
        "--nsga3-save-all-candidate-results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist per-candidate workflow results in addition to the default frontier solution files",
    )
    args = parser.parse_args()

    with open(args.demands, "r", encoding="utf-8") as f:
        windows = json.load(f)

    weight_configs = _load_weight_configs(args.weights)
    analytics_dir = str(Path(args.output).with_suffix("")) + "_analytics"

    if args.solver_backend in {"nsga3", "nsga3_heuristic"}:
        if args.solver_backend == "nsga3":
            from llm4fairrouting.multiobjective.nsga3_search import run_nsga3_pareto_search as run_search
            search_output_dir = analytics_dir + "/nsga3"
        else:
            from llm4fairrouting.multiobjective.nsga3_heuristic import run_nsga3_heuristic_search as run_search
            search_output_dir = analytics_dir + "/nsga3_heuristic"

        payload = run_search(
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
            drone_activation_cost=args.drone_activation_cost,
            drone_speed=args.drone_speed,
            output_dir=search_output_dir,
            pop_size=args.nsga3_pop_size,
            n_generations=args.nsga3_n_generations,
            seed=args.nsga3_seed,
            save_all_candidate_results=args.nsga3_save_all_candidate_results,
            enable_conflict_refiner=args.enable_conflict_refiner,
            problem_id=Path(args.output).stem,
        )
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"{args.solver_backend} results saved to {out_path}")
        return

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
        drone_activation_cost=args.drone_activation_cost,
        drone_speed=args.drone_speed,
        analytics_output_dir=analytics_dir,
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
            drone_activation_cost=args.drone_activation_cost,
            drone_speed=args.drone_speed,
            analytics_output_dir=analytics_dir + "/pareto",
            enable_conflict_refiner=args.enable_conflict_refiner,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialize_workflow_results(all_solutions), f, ensure_ascii=False, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
