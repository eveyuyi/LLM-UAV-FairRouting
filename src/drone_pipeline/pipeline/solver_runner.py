"""
Module 3b: Solver Runner

负责将 Module 2 的结构化需求和 Module 3a 的权重配置送入求解器。
该模块既可被 run_pipeline 复用，也可作为独立脚本运行。

统一使用 ``cplex_with_priority_noise.CplexSolver`` 作为求解后端。

求解模型特性（对齐 cplex_with_priority_noise）：
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Ensure CPLEX binary is on PATH (previously handled by drone_cplex_real_data import)
_CPLEX_BIN = "/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx"
if os.path.isdir(_CPLEX_BIN) and _CPLEX_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CPLEX_BIN + ":" + os.environ.get("PATH", "")

from drone_pipeline.pipeline.dialogue_generator import load_stations


@dataclass
class SolverPoint:
    """轻量点结构，避免在非求解场景提前依赖 pyomo。"""

    id: str
    lon: float
    lat: float
    alt: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    type: str = ""

    def to_enu(self, ref_lat: float, ref_lon: float, ref_alt: float):
        lat_scale = 111000.0
        lon_scale = 111000.0 * math.cos(math.radians(ref_lat))
        self.x = (self.lon - ref_lon) * lon_scale
        self.y = (self.lat - ref_lat) * lat_scale
        self.z = self.alt - ref_alt


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


def _iter_active_coords(demands: List[Dict]) -> Iterable[tuple[float, float]]:
    for demand in demands:
        for key in ("origin", "destination"):
            coords = demand.get(key, {}).get("coords", [])
            if len(coords) == 2:
                yield float(coords[0]), float(coords[1])


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    import math

    radius = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _select_station_dicts(
    stations: List[Dict],
    demands: List[Dict],
    max_stations: Optional[int],
) -> List[Dict]:
    if not stations:
        return []
    if max_stations is None or max_stations <= 0 or len(stations) <= max_stations:
        return list(stations)

    active_coords = list(_iter_active_coords(demands))
    if not active_coords:
        return stations[:max_stations]

    def station_score(station: Dict) -> float:
        lon = float(station["lon"])
        lat = float(station["lat"])
        return min(
            _haversine_m(lon, lat, active_lon, active_lat)
            for active_lon, active_lat in active_coords
        )

    ranked = sorted(
        stations,
        key=lambda station: (station_score(station), station.get("station_id", "")),
    )
    return ranked[:max_stations]


def load_solver_station_points(
    stations_path: Optional[str],
    demands: List[Dict],
    max_stations: Optional[int] = 10,
) -> List[SolverPoint]:
    """加载求解器站点。

    优先复用 stations_path；若站点过多，则按距离当前窗口需求最近优先截取。
    """
    if not stations_path:
        count = max_stations if max_stations and max_stations > 0 else 3
        print("  未提供站点文件，回退到 mock stations")
        return create_mock_stations(count)

    try:
        station_dicts = load_stations(stations_path)
    except Exception as exc:
        count = max_stations if max_stations and max_stations > 0 else 3
        print(f"  站点文件加载失败 ({exc})，回退到 mock stations")
        return create_mock_stations(count)

    selected = _select_station_dicts(station_dicts, demands, max_stations)
    if len(selected) < len(station_dicts):
        print(
            f"  站点文件共 {len(station_dicts)} 个站点，"
            f"为当前窗口选取最近的 {len(selected)} 个参与求解"
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


def _build_simplified_noise_matrix(dist_matrix):
    """Build a simplified noise cost matrix when detailed building data is unavailable.

    Approximates the number of affected residential buildings per flight segment
    as proportional to path length (~1 building per 200 m of urban flight).
    This serves as a lightweight stand-in for the full RRT+building noise
    pipeline in ``cplex_with_priority_noise.build_realistic_distance_and_noise_matrices``.
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
    ``serialize_pipeline_results``.

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
) -> Dict:
    """求解单个时间窗口。

    统一使用 ``cplex_with_priority_noise.CplexSolver`` 作为求解后端，
    使 LLM pipeline 和 solver-only baseline 共享同一求解引擎。

    模型特性（对齐 cplex_with_priority_noise）：
    - 目标函数: ``(distance + noise_weight * noise) * (1/priority)``
    - 允许部分需求不被服务（unassigned + 大罚项）
    - 每架无人机单次求解最多接 1 个需求（single_task_per_drone）
    - 需求必须从指定供给点取货（supply_demand_matching）
    """
    window_weight_config = copy.deepcopy(weight_config)

    feasible_demands = [
        demand for demand in demands
        if demand.get("cargo", {}).get("weight_kg", 0.0) <= max_payload
    ]
    skipped = len(demands) - len(feasible_demands)
    if skipped:
        print(f"  跳过 {skipped} 条超载需求 (>{max_payload}kg)")

    if not feasible_demands:
        print("  无可行需求，跳过")
        return {
            "time_window": time_window,
            "weight_config": window_weight_config,
            "feasible_demands": [],
            "n_demands_total": len(demands),
            "n_demands_filtered": skipped,
            "solution": None,
            "n_supply": 0,
        }

    feasible_ids = {demand["demand_id"] for demand in feasible_demands}
    window_weight_config["demand_configs"] = [
        config for config in window_weight_config.get("demand_configs", [])
        if config["demand_id"] in feasible_ids
    ]

    supply_points, demand_points, demand_weights, demand_required_supply = (
        demands_to_solver_inputs(feasible_demands)
    )
    station_points = load_solver_station_points(
        stations_path=stations_path,
        demands=feasible_demands,
        max_stations=max_solver_stations,
    )

    if not supply_points or not demand_points or not station_points:
        print("  供给点、需求点或站点为空，跳过")
        return {
            "time_window": time_window,
            "weight_config": window_weight_config,
            "feasible_demands": feasible_demands,
            "n_demands_total": len(demands),
            "n_demands_filtered": skipped,
            "solution": None,
            "n_supply": len(supply_points),
        }

    # Defer heavy imports until after early-return checks
    import numpy as np

    n_supply = len(supply_points)
    n_demand = len(demand_points)
    n_station = len(station_points)

    print(
        f"  供给点 {n_supply}, 需求点 {n_demand}, "
        f"站点 {n_station}"
    )

    # ----- Import shared solver backend (cplex_with_priority_noise) -----
    try:
        from drone_pipeline.utils.cplex_with_priority_noise import (
            CplexSolver as _CplexSolver,
            Point as _CplexPoint,
            Drone as _CplexDrone,
            DroneState as _DroneState,
            DroneStatus as _DroneStatus,
            DemandEvent as _DemandEvent,
        )
    except Exception as exc:
        print(f"  求解依赖加载失败: {exc}")
        return {
            "time_window": time_window,
            "weight_config": window_weight_config,
            "feasible_demands": feasible_demands,
            "n_demands_total": len(demands),
            "n_demands_filtered": skipped,
            "solution": None,
            "n_supply": n_supply,
        }

    # ----- Convert SolverPoints → CplexPoints with ENU coordinates -----
    all_solver_pts = supply_points + demand_points + station_points
    ref_lat = float(np.mean([p.lat for p in all_solver_pts]))
    ref_lon = float(np.mean([p.lon for p in all_solver_pts]))

    cplex_points: List = []
    for sp in all_solver_pts:
        cp = _CplexPoint(id=sp.id, lon=sp.lon, lat=sp.lat, alt=sp.alt, type=sp.type)
        cp.to_enu(ref_lat, ref_lon, 0.0)
        cplex_points.append(cp)

    # ----- Euclidean distance matrix -----
    n_pts = len(cplex_points)
    dist_matrix = np.zeros((n_pts, n_pts))
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            dx = cplex_points[i].x - cplex_points[j].x
            dy = cplex_points[i].y - cplex_points[j].y
            dz = cplex_points[i].z - cplex_points[j].z
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # ----- Noise cost matrix (simplified when building data unavailable) -----
    noise_cost_matrix = None
    if noise_weight > 0:
        noise_cost_matrix = _build_simplified_noise_matrix(dist_matrix)
        print(f"  噪声矩阵: {noise_cost_matrix.shape}, 权重={noise_weight}")

    # ----- Index lists (same layout as cplex_with_priority_noise) -----
    supply_indices = list(range(n_supply))
    station_indices = list(range(n_supply + n_demand, n_supply + n_demand + n_station))

    # ----- Create drones (all idle at their stations) -----
    cruise_speed = 15.0
    cplex_drones: List = []
    for s_idx in range(n_station):
        for d_idx in range(max_drones_per_station):
            cplex_drones.append(_CplexDrone(
                id=f"U{s_idx + 1}{d_idx + 1}",
                station_id=s_idx,
                station_name=station_points[s_idx].id,
                max_payload=max_payload,
                max_range=max_range,
                speed=cruise_speed,
            ))

    drone_states: List = []
    for drone in cplex_drones:
        station_node = station_indices[drone.station_id]
        drone_states.append(_DroneState(
            drone_id=drone.id,
            station_id=drone.station_id,
            current_node=station_node,
            remaining_range=drone.max_range,
            remaining_payload=drone.max_payload,
            status=_DroneStatus.IDLE,
            position_x=cplex_points[station_node].x,
            position_y=cplex_points[station_node].y,
        ))

    # ----- Create DemandEvents with LLM priority + required_supply_idx -----
    demand_cfg_map = {
        dc["demand_id"]: dc
        for dc in window_weight_config.get("demand_configs", [])
    }
    demand_events: List = []
    for idx, demand in enumerate(feasible_demands):
        demand_id = demand.get("demand_id", f"D{idx + 1}")
        cfg = demand_cfg_map.get(demand_id, {})
        priority = cfg.get("priority", 3)
        demand_events.append(_DemandEvent(
            time=0.0,
            node_idx=n_supply + idx,
            weight=demand_weights[idx],
            unique_id=demand_id,
            priority=priority,
            required_supply_idx=demand_required_supply[idx],
            demand_point_id=demand_points[idx].id,
        ))

    # ----- Solve via shared CplexSolver backend -----
    solver = _CplexSolver(
        drones=cplex_drones,
        supply_indices=supply_indices,
        station_indices=station_indices,
        dist_matrix=dist_matrix,
        all_points=cplex_points,
        noise_cost_matrix=noise_cost_matrix,
        noise_weight=noise_weight,
        time_limit=time_limit,
    )

    t0 = _time.time()
    try:
        assignments = solver.solve_assignment(
            drone_states, demand_events, current_time=0.0,
        )
        solve_time = _time.time() - t0
        solution = _build_solution_from_assignments(
            assignments=assignments,
            demand_events=demand_events,
            dist_matrix=dist_matrix,
            cplex_points=cplex_points,
            station_indices=station_indices,
            n_supply=n_supply,
            cruise_speed=cruise_speed,
            solve_time=solve_time,
        )

        n_served = len(assignments)
        n_unserved = n_demand - n_served
        print(
            f"  求解完成: {n_served}/{n_demand} 需求已分配, "
            f"{n_unserved} 未分配, 耗时 {solve_time:.2f}s"
        )
    except Exception as exc:
        print(f"  求解失败: {exc}")
        solution = None

    return {
        "time_window": time_window,
        "weight_config": window_weight_config,
        "feasible_demands": feasible_demands,
        "n_demands_total": len(demands),
        "n_demands_filtered": skipped,
        "solution": solution,
        "n_supply": n_supply,
    }


def serialize_pipeline_results(all_solutions: List[Dict]) -> List[Dict]:
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
    parser = argparse.ArgumentParser(description="Module 3b: Solver Runner")
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
        default=None,
        help="真实站点文件 latest_location.xlsx；不传则回退到 mock stations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "drone" / "solver_results.json"),
        help="求解结果输出路径",
    )
    parser.add_argument(
        "--time-limit", type=int, default=300, help="求解器时间限制（秒）"
    )
    parser.add_argument(
        "--max-drones-per-station", type=int, default=6, help="每个站点最大无人机数"
    )
    parser.add_argument(
        "--max-payload", type=float, default=25.0, help="最大载重（kg）"
    )
    parser.add_argument(
        "--max-range", type=float, default=40000.0, help="最大航程（m）"
    )
    parser.add_argument(
        "--noise-weight",
        type=float,
        default=0.0,
        help="噪声成本权重（>0 启用噪声优先级）",
    )
    parser.add_argument(
        "--max-solver-stations",
        type=int,
        default=10,
        help="求解时最多使用多少个真实站点；0 表示使用全部",
    )
    args = parser.parse_args()

    with open(args.demands, "r", encoding="utf-8") as f:
        windows = json.load(f)

    weight_configs = _load_weight_configs(args.weights)
    all_solutions = []

    for window in windows:
        time_window = window.get("time_window", "")
        demands = window.get("demands", [])
        if not demands:
            continue

        weight_config = weight_configs.get(time_window)
        if weight_config is None:
            print(f"窗口 {time_window} 缺少权重配置，跳过")
            continue

        print(f"\n---- 窗口 {time_window}: {len(demands)} 条需求 ----")
        result = solve_window_demands(
            time_window=time_window,
            demands=demands,
            weight_config=weight_config,
            stations_path=args.stations,
            max_solver_stations=args.max_solver_stations,
            time_limit=args.time_limit,
            max_drones_per_station=args.max_drones_per_station,
            max_payload=args.max_payload,
            max_range=args.max_range,
            noise_weight=args.noise_weight,
        )
        all_solutions.append(result)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialize_pipeline_results(all_solutions), f, ensure_ascii=False, indent=2)

    print(f"结果保存至 {out_path}")


if __name__ == "__main__":
    main()
