"""Static single-window heuristic solver for independent planning experiments."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from llm4fairrouting.data.building_information import load_building_partitions
from llm4fairrouting.data.seed_paths import BUILDING_DATA_PATH
from llm4fairrouting.multiobjective.nsga3_search import (
    _SearchBounds,
    _candidate_objectives,
    _decode_candidate,
    _evaluation_cache_key,
)
from llm4fairrouting.routing.analytics import (
    analyze_pareto_candidates,
    compute_pareto_frontier,
    export_visualizations,
    sanitize_label,
    write_json,
)
from llm4fairrouting.routing.domain import (
    DemandEvent,
    Drone,
    DroneState,
    DroneStatus,
    Point,
)
from llm4fairrouting.routing.heuristic_assignment import HeuristicAssignmentSolver
from llm4fairrouting.routing.path_costs import (
    FLIGHT_HEIGHT,
    build_lazy_distance_and_noise_matrices,
    create_obstacles_from_buildings,
)
from llm4fairrouting.workflow.solver_adapter import _point_key, load_solver_station_points


def _create_fixed_total_drones(
    station_points: List[Point],
    *,
    total_drones: int,
    max_payload: float,
    max_range: float,
    speed: float,
) -> List[Drone]:
    if total_drones <= 0 or not station_points:
        return []

    drones: List[Drone] = []
    per_station_counts = [0 for _ in station_points]
    for drone_idx in range(total_drones):
        station_idx = drone_idx % len(station_points)
        per_station_counts[station_idx] += 1
        drones.append(
            Drone(
                id=f"U{station_idx + 1}{per_station_counts[station_idx]}",
                station_id=station_idx,
                station_name=station_points[station_idx].id,
                max_payload=max_payload,
                max_range=max_range,
                speed=speed,
            )
        )
    return drones


def _create_initial_drone_states(
    drones: List[Drone],
    *,
    n_supply: int,
    n_demand: int,
    n_station: int,
    all_points: List[Point],
) -> List[DroneState]:
    station_indices = list(range(n_supply + n_demand, n_supply + n_demand + n_station))
    states: List[DroneState] = []
    for drone in drones:
        station_node = station_indices[drone.station_id]
        states.append(
            DroneState(
                drone_id=drone.id,
                station_id=drone.station_id,
                current_node=station_node,
                remaining_range=drone.max_range,
                remaining_payload=drone.max_payload,
                current_load=0.0,
                status=DroneStatus.IDLE,
                executed_path=[station_node],
                position_x=all_points[station_node].x,
                position_y=all_points[station_node].y,
                task_queue=[],
            )
        )
    return states


def _build_static_window_problem(
    *,
    window: Dict,
    weight_config: Dict,
    stations_path: Optional[str],
    building_path: Optional[str],
    max_solver_stations: Optional[int],
    max_payload: float,
) -> Dict[str, object]:
    time_window = str(window.get("time_window", "window_0"))
    demands = list(window.get("demands", []))

    feasible_demands = [
        copy.deepcopy(demand)
        for demand in demands
        if float(demand.get("cargo", {}).get("weight_kg", 0.0) or 0.0) <= max_payload
    ]
    skipped = len(demands) - len(feasible_demands)

    supply_points: List[Point] = []
    demand_points: List[Point] = []
    supply_key_to_idx: Dict[str, int] = {}
    demand_key_to_idx: Dict[str, int] = {}
    event_records: List[Dict[str, object]] = []

    demand_cfg_map = {
        str(config.get("demand_id", "")): config
        for config in weight_config.get("demand_configs", [])
    }

    for idx, demand in enumerate(feasible_demands):
        origin = demand.get("origin", {})
        origin_coords = list(origin.get("coords", []))
        origin_key = _point_key(
            origin.get("fid"),
            origin_coords,
            fallback=f"{time_window}:origin:{idx}",
        )
        if origin_key not in supply_key_to_idx:
            supply_key_to_idx[origin_key] = len(supply_points)
            supply_points.append(
                Point(
                    id=f"S_{origin.get('fid', len(supply_points) + 1)}",
                    lon=float(origin_coords[0]),
                    lat=float(origin_coords[1]),
                    alt=FLIGHT_HEIGHT,
                    type="supply",
                )
            )

        dest = demand.get("destination", {})
        dest_coords = list(dest.get("coords", []))
        dest_key = _point_key(
            dest.get("fid"),
            dest_coords,
            fallback=f"{time_window}:destination:{idx}",
        )
        if dest_key not in demand_key_to_idx:
            demand_key_to_idx[dest_key] = len(demand_points)
            demand_points.append(
                Point(
                    id=f"D_{dest.get('fid', len(demand_points) + 1)}",
                    lon=float(dest_coords[0]),
                    lat=float(dest_coords[1]),
                    alt=FLIGHT_HEIGHT,
                    type="demand",
                )
            )

        original_demand_id = str(demand.get("demand_id", f"D{idx + 1}"))
        event_id = f"{time_window}::{original_demand_id}::{idx}"
        demand["solver_event_id"] = event_id

        cfg = demand_cfg_map.get(original_demand_id, {})
        deadline_minutes = float(
            demand.get("time_constraint", {}).get("deadline_minutes", 0.0) or 0.0
        )
        event_records.append(
            {
                "event_id": event_id,
                "original_demand_id": original_demand_id,
                "source_event_id": demand.get("source_event_id"),
                "time_window": time_window,
                "weight": float(demand.get("cargo", {}).get("weight_kg", 0.0)),
                "priority": int(cfg.get("priority", 3)),
                "deadline_minutes": deadline_minutes,
                "required_supply_idx": supply_key_to_idx[origin_key],
                "demand_point_idx": demand_key_to_idx[dest_key],
                "demand_point_id": demand_points[demand_key_to_idx[dest_key]].id,
            }
        )

    station_points = load_solver_station_points(
        stations_path=stations_path,
        demands=feasible_demands,
        max_stations=max_solver_stations,
    )

    building_source = str(building_path or BUILDING_DATA_PATH)
    if not Path(building_source).exists():
        raise FileNotFoundError(f"Building data file not found: {building_source}")

    station_solver_points = [
        Point(
            id=station.id,
            lon=station.lon,
            lat=station.lat,
            alt=FLIGHT_HEIGHT,
            type="station",
        )
        for station in station_points
    ]

    all_points = supply_points + demand_points + station_solver_points
    if all_points:
        ref_lat = float(np.mean([point.lat for point in all_points]))
        ref_lon = float(np.mean([point.lon for point in all_points]))
        for point in all_points:
            point.to_enu(ref_lat, ref_lon, 0.0)
    else:
        ref_lat = 0.0
        ref_lon = 0.0

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
        residence = Point(
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
        flight_height=FLIGHT_HEIGHT,
    )

    demand_events: List[DemandEvent] = []
    n_supply = len(supply_points)
    for record in event_records:
        demand_events.append(
            DemandEvent(
                time=0.0,
                node_idx=n_supply + int(record["demand_point_idx"]),
                weight=float(record["weight"]),
                unique_id=str(record["event_id"]),
                priority=int(record["priority"]),
                required_supply_idx=int(record["required_supply_idx"]),
                demand_point_id=str(record["demand_point_id"]),
            )
        )

    return {
        "time_window": time_window,
        "weight_config": weight_config,
        "feasible_demands": feasible_demands,
        "n_demands_total": len(demands),
        "n_demands_filtered": skipped,
        "supply_points": supply_points,
        "demand_points": demand_points,
        "station_points": station_solver_points,
        "all_points": all_points,
        "dist_matrix": dist_matrix,
        "noise_cost_matrix": noise_cost_matrix,
        "demand_events": demand_events,
        "event_records": event_records,
    }


def _build_static_run_summary(
    *,
    demand_events: List[DemandEvent],
    route_plans: List[Dict[str, object]],
    objective_breakdown: Dict[str, float],
) -> Dict[str, object]:
    served_delivery_times = []
    served_ids = set()
    used_drone_ids = sorted(
        str(route_plan.get("drone").drone_id)
        for route_plan in route_plans
        if route_plan.get("drone") is not None
    )
    for route_plan in route_plans:
        served_ids.update(str(item) for item in route_plan.get("served_demand_ids", []))
        served_delivery_times.extend(
            max(0.0, float(delivery_time_h))
            for delivery_time_h in (route_plan.get("delivery_times_h") or {}).values()
        )

    service_rate = round(len(served_ids) / len(demand_events), 6) if demand_events else 0.0
    return {
        "total_demands": len(demand_events),
        "served_demands": len(served_ids),
        "service_rate": service_rate,
        "service_rate_loss": round(1.0 - service_rate, 6),
        "final_total_distance_m": float(objective_breakdown.get("distance_m", 0.0)),
        "final_total_noise_impact": float(objective_breakdown.get("noise_impact", 0.0)),
        "average_delivery_time_h": round(sum(served_delivery_times) / len(served_delivery_times), 6)
        if served_delivery_times else None,
        "max_delivery_time_h": round(max(served_delivery_times), 6) if served_delivery_times else None,
        "n_used_drones": len(used_drone_ids),
        "used_drone_ids": used_drone_ids,
        "n_solver_calls": 1,
        "n_conflict_reports": 0,
    }


def solve_single_window_static_heuristic(
    *,
    window: Dict,
    weight_config: Dict,
    stations_path: Optional[str],
    building_path: Optional[str],
    max_solver_stations: Optional[int],
    max_payload: float,
    max_range: float,
    noise_weight: float,
    drone_activation_cost: float,
    drone_speed: float,
    total_drones: int = 3,
) -> Dict[str, object]:
    problem = _build_static_window_problem(
        window=window,
        weight_config=weight_config,
        stations_path=stations_path,
        building_path=building_path,
        max_solver_stations=max_solver_stations,
        max_payload=max_payload,
    )

    all_points = list(problem["all_points"])
    n_supply = len(problem["supply_points"])
    n_demand = len(problem["demand_points"])
    n_station = len(problem["station_points"])
    station_points = list(problem["station_points"])
    demand_events = list(problem["demand_events"])
    event_records = list(problem["event_records"])

    if not station_points:
        return {
            "time_window": problem["time_window"],
            "weight_config": weight_config,
            "feasible_demands": problem["feasible_demands"],
            "n_demands_total": problem["n_demands_total"],
            "n_demands_filtered": problem["n_demands_filtered"],
            "solution": None,
            "n_supply": n_supply,
        }

    drones = _create_fixed_total_drones(
        station_points,
        total_drones=total_drones,
        max_payload=max_payload,
        max_range=max_range,
        speed=drone_speed,
    )
    drone_states = _create_initial_drone_states(
        drones,
        n_supply=n_supply,
        n_demand=n_demand,
        n_station=n_station,
        all_points=all_points,
    )

    solver = HeuristicAssignmentSolver(
        drones=drones,
        supply_indices=list(range(n_supply)),
        station_indices=list(range(n_supply + n_demand, n_supply + n_demand + n_station)),
        dist_matrix=problem["dist_matrix"],
        all_points=all_points,
        noise_cost_matrix=problem["noise_cost_matrix"],
        noise_weight=noise_weight,
        drone_activation_cost=drone_activation_cost,
        time_limit=0,
        analytics_output_dir=None,
        enable_conflict_refiner=False,
    )
    route_plans = solver.solve_assignment(
        drone_states=drone_states,
        demands=demand_events,
        current_time=0.0,
        objective_weights=weight_config.get("global_weights"),
        solve_context={"time_window": problem["time_window"]},
    )
    solve_details = dict(solver.last_solve_details or {})
    objective_breakdown = dict(solve_details.get("objective_breakdown") or {})
    run_summary = _build_static_run_summary(
        demand_events=demand_events,
        route_plans=route_plans,
        objective_breakdown=objective_breakdown,
    )

    config_map = {
        str(config.get("demand_id", "")): config
        for config in weight_config.get("demand_configs", [])
    }
    event_record_lookup = {
        str(record["event_id"]): record
        for record in event_records
    }
    demand_event_lookup = {
        str(event.unique_id): event
        for event in demand_events
    }

    served_lookup: Dict[str, Dict[str, object]] = {}
    drone_path_details: List[Dict[str, object]] = []
    for route_plan in route_plans:
        drone_state = route_plan["drone"]
        for demand_meta in route_plan.get("served_demands", []):
            demand = demand_meta["demand"]
            event_id = str(demand.unique_id)
            delivery_time_h = demand_meta.get("delivery_time_h")
            served_lookup[event_id] = {
                "assigned_drone": drone_state.drone_id,
                "delivery_time_h": delivery_time_h,
                "delivery_time_s": round(float(delivery_time_h) * 3600.0, 1)
                if delivery_time_h is not None else None,
                "delivery_latency_h": max(0.0, float(delivery_time_h))
                if delivery_time_h is not None else None,
                "delivery_latency_s": round(max(0.0, float(delivery_time_h)) * 3600.0, 1)
                if delivery_time_h is not None else None,
            }

        drone_path_details.append(
            {
                "drone_id": drone_state.drone_id,
                "station_id": drone_state.station_id,
                "station_name": station_points[drone_state.station_id].id
                if 0 <= drone_state.station_id < len(station_points) else "",
                "path_node_indices": list(route_plan.get("path_node_indices", [])),
                "path_node_ids": list(route_plan.get("path_node_ids", [])),
                "path_str": route_plan.get("path_str", ""),
                "served_demand_ids": list(route_plan.get("served_demand_ids", [])),
                "total_distance_m": float(route_plan.get("total_distance_m", 0.0)),
                "total_mission_time_h": float(route_plan.get("total_mission_time_h", 0.0)),
                "total_mission_time_s": float(route_plan.get("total_mission_time_s", 0.0)),
            }
        )

    per_demand_results: List[Dict[str, object]] = []
    for demand in problem["feasible_demands"]:
        event_id = str(demand.get("solver_event_id", demand.get("demand_id", "")))
        record = event_record_lookup.get(event_id, {})
        event = demand_event_lookup.get(event_id)
        served = served_lookup.get(event_id, {})
        demand_id = str(demand.get("demand_id", ""))
        cargo = demand.get("cargo", {})
        origin = demand.get("origin", {})
        dest = demand.get("destination", {})
        config = config_map.get(demand_id, {})
        deadline_minutes = demand.get("time_constraint", {}).get("deadline_minutes")
        delivery_latency_s = served.get("delivery_latency_s")
        per_demand_results.append(
            {
                "demand_id": demand_id,
                "solver_event_id": event_id,
                "source_event_id": demand.get("source_event_id") or record.get("source_event_id"),
                "source_dialogue_id": demand.get("source_dialogue_id"),
                "request_timestamp": demand.get("request_timestamp"),
                "priority": config.get("priority", record.get("priority", 3)),
                "window_rank": config.get("window_rank"),
                "llm_reasoning": config.get("reasoning", ""),
                "demand_tier": demand.get("demand_tier", cargo.get("demand_tier", "")),
                "cargo_type": cargo.get("type", ""),
                "weight_kg": cargo.get("weight_kg", 0.0),
                "origin_fid": origin.get("fid"),
                "dest_fid": dest.get("fid"),
                "deadline_minutes": deadline_minutes,
                "is_served": bool(served),
                "assigned_drone": served.get("assigned_drone"),
                "delivery_time_h": served.get("delivery_time_h"),
                "delivery_time_s": served.get("delivery_time_s"),
                "delivery_time_min": round(float(served["delivery_time_s"]) / 60.0, 2)
                if served.get("delivery_time_s") is not None else None,
                "delivery_latency_h": served.get("delivery_latency_h"),
                "delivery_latency_s": delivery_latency_s,
                "delivery_latency_min": round(float(delivery_latency_s) / 60.0, 2)
                if delivery_latency_s is not None else None,
                "is_deadline_met": (
                    delivery_latency_s is not None
                    and deadline_minutes not in (None, "")
                    and float(delivery_latency_s) <= float(deadline_minutes) * 60.0
                ),
                "required_supply_idx": int(event.required_supply_idx) if event is not None and event.required_supply_idx is not None else None,
                "demand_point_id": event.demand_point_id if event is not None else "",
            }
        )

    solution = {
        "solve_mode": "static_heuristic_single_window",
        "solve_status": solve_details.get("termination_condition", "heuristic_completed"),
        "solve_time_s": solve_details.get("solve_time_s", 0.0),
        "objective_value": solve_details.get("objective_value"),
        "objective_breakdown": objective_breakdown,
        "assignment_backend": "heuristic",
        "run_summary": run_summary,
        "per_demand_results": per_demand_results,
        "per_drone_details": drone_path_details,
        "drone_path_details": drone_path_details,
        "analytics_artifacts": {},
    }
    return {
        "time_window": problem["time_window"],
        "weight_config": weight_config,
        "feasible_demands": problem["feasible_demands"],
        "n_demands_total": problem["n_demands_total"],
        "n_demands_filtered": problem["n_demands_filtered"],
        "solution": solution,
        "n_supply": n_supply,
    }


def serialize_static_heuristic_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    serialized: List[Dict[str, object]] = []
    for entry in results:
        solution = entry.get("solution") or {}
        weight_config = entry.get("weight_config") or {}
        serialized.append(
            {
                "time_window": entry.get("time_window"),
                "n_demands_extracted": entry.get("n_demands_total", len(entry.get("feasible_demands", []))),
                "n_demands_feasible": len(entry.get("feasible_demands", [])),
                "n_demands_filtered": entry.get("n_demands_filtered", 0),
                "global_weights": weight_config.get("global_weights", {}),
                "n_supplementary_constraints": len(weight_config.get("supplementary_constraints", [])),
                "solve_status": solution.get("solve_status"),
                "solve_time_s": solution.get("solve_time_s"),
                "objective_value": solution.get("objective_value"),
                "run_summary": solution.get("run_summary", {}),
                "per_demand_results": list(solution.get("per_demand_results", [])),
                "per_drone_details": list(solution.get("per_drone_details", [])),
                "drone_path_details": list(solution.get("drone_path_details", [])),
                "analytics_artifacts": solution.get("analytics_artifacts", {}),
            }
        )
    return serialized


def _problem_metadata(
    *,
    problem_id: str,
    window: Dict,
    stations_path: Optional[str],
    building_path: Optional[str],
    max_solver_stations: Optional[int],
    total_drones: int,
    max_payload: float,
    max_range: float,
    noise_weight: float,
    drone_speed: float,
) -> Dict[str, object]:
    return {
        "problem_type": "static_single_window_routing",
        "problem_id": problem_id,
        "time_window": window.get("time_window"),
        "n_demands": len(window.get("demands", [])),
        "stations_path": stations_path,
        "building_path": building_path,
        "max_solver_stations": max_solver_stations,
        "total_drones": total_drones,
        "max_payload_kg": max_payload,
        "max_range_m": max_range,
        "noise_weight": noise_weight,
        "drone_speed_ms": drone_speed,
    }


def run_nsga3_static_heuristic_search(
    *,
    window: Dict,
    weight_config: Dict,
    stations_path: Optional[str],
    building_path: Optional[str],
    max_solver_stations: Optional[int],
    max_payload: float,
    max_range: float,
    noise_weight: float,
    drone_speed: float,
    output_dir: str,
    total_drones: int = 3,
    drone_activation_cost: float = 1000.0,
    pop_size: int = 20,
    n_generations: int = 10,
    seed: int = 42,
    save_all_candidate_results: bool = False,
    problem_id: Optional[str] = None,
) -> Dict[str, object]:
    try:
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
        from pymoo.util.ref_dirs import get_reference_directions
    except ImportError as exc:
        raise RuntimeError(
            "Static NSGA-III heuristic backend requires the optional 'pymoo' dependency."
        ) from exc

    bounds = _SearchBounds(
        activation_cost_min=min(100.0, float(drone_activation_cost)),
        activation_cost_max=max(5000.0, float(drone_activation_cost)),
    )
    search_output_dir = Path(output_dir)
    search_output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = search_output_dir / "candidates"
    frontier_dir = search_output_dir / "frontier_solutions"
    frontier_dir.mkdir(parents=True, exist_ok=True)
    if save_all_candidate_results:
        candidates_dir.mkdir(parents=True, exist_ok=True)

    resolved_problem_id = problem_id or f"nsga3_static_heuristic_{sanitize_label(window.get('time_window'))}"
    problem_meta = _problem_metadata(
        problem_id=resolved_problem_id,
        window=window,
        stations_path=stations_path,
        building_path=building_path,
        max_solver_stations=max_solver_stations,
        total_drones=total_drones,
        max_payload=max_payload,
        max_range=max_range,
        noise_weight=noise_weight,
        drone_speed=drone_speed,
    )
    solver_meta = {
        "solver_family": "evolutionary",
        "solver_name": "nsga3_static_heuristic",
        "inner_solver": "greedy_heuristic_static",
    }
    cache: Dict[tuple[float, ...], Dict[str, object]] = {}
    evaluated_candidates: List[Dict[str, object]] = []

    def evaluate_candidate(x) -> Dict[str, object]:
        cache_key = _evaluation_cache_key(x)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        decoded = _decode_candidate(x, bounds)
        candidate_index = len(evaluated_candidates)
        solution_id = f"nsga3_static_heuristic_candidate_{candidate_index:04d}"
        candidate_dir = candidates_dir / sanitize_label(solution_id)
        candidate_weight_config = json.loads(json.dumps(weight_config))
        candidate_weight_config["global_weights"] = decoded["weights"]

        results = [
            solve_single_window_static_heuristic(
                window=window,
                weight_config=candidate_weight_config,
                stations_path=stations_path,
                building_path=building_path,
                max_solver_stations=max_solver_stations,
                max_payload=max_payload,
                max_range=max_range,
                noise_weight=noise_weight,
                drone_activation_cost=float(decoded["drone_activation_cost"]),
                drone_speed=drone_speed,
                total_drones=total_drones,
            )
        ]
        run_summary = dict((results[0].get("solution") or {}).get("run_summary") or {})
        objectives = _candidate_objectives(run_summary)
        serialized_results = serialize_static_heuristic_results(results)
        candidate: Dict[str, object] = {
            "solution_id": solution_id,
            "profile_id": solution_id,
            "label": solution_id,
            "decision_vector": [float(item) for item in x],
            "weights": decoded["weights"],
            "drone_activation_cost": float(decoded["drone_activation_cost"]),
            "objectives": objectives,
            **objectives,
            "run_summary": run_summary,
            "problem_meta": problem_meta,
            "solver_meta": solver_meta,
            "_serialized_results": serialized_results,
        }
        if save_all_candidate_results:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            result_path = candidate_dir / "workflow_results.json"
            with open(result_path, "w", encoding="utf-8") as handle:
                json.dump(serialized_results, handle, ensure_ascii=False, indent=2)
            candidate["result_path"] = str(result_path)

        cache[cache_key] = candidate
        evaluated_candidates.append(candidate)
        return candidate

    class _StaticNSGA3Problem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(n_var=4, n_obj=5, xl=0.0, xu=1.0)

        def _evaluate(self, x, out, *args, **kwargs):
            candidate = evaluate_candidate(x)
            objectives = candidate["objectives"]
            out["F"] = [
                objectives["final_total_distance_m"],
                objectives["average_delivery_time_h"],
                objectives["final_total_noise_impact"],
                objectives["service_rate_loss"],
                objectives["n_used_drones"],
            ]

    problem = _StaticNSGA3Problem()
    ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=3)
    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    search_started_at = time.perf_counter()
    minimize(
        problem,
        algorithm,
        termination=("n_gen", n_generations),
        seed=seed,
        verbose=False,
    )

    search_runtime_s = round(time.perf_counter() - search_started_at, 6)
    avg_candidate_runtime_s = round(search_runtime_s / len(evaluated_candidates), 6) if evaluated_candidates else None
    print(
        f"[Static NSGA-III Heuristic] Window {window.get('time_window')} "
        f"completed {len(evaluated_candidates)} candidates in {search_runtime_s:.3f}s "
        f"(avg {avg_candidate_runtime_s:.3f}s/candidate)"
    )

    frontier = compute_pareto_frontier(evaluated_candidates)
    frontier_ids = {item["solution_id"] for item in frontier}
    for candidate in evaluated_candidates:
        candidate["is_nondominated"] = candidate["solution_id"] in frontier_ids
        if not candidate["is_nondominated"]:
            continue
        solution_dir = frontier_dir / sanitize_label(str(candidate["solution_id"]))
        solution_dir.mkdir(parents=True, exist_ok=True)
        result_path = solution_dir / "workflow_results.json"
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(candidate["_serialized_results"], handle, ensure_ascii=False, indent=2)
        candidate["frontier_result_path"] = str(result_path)

    public_candidates = [
        {key: value for key, value in candidate.items() if not str(key).startswith("_")}
        for candidate in evaluated_candidates
    ]
    public_frontier = [
        {key: value for key, value in candidate.items() if not str(key).startswith("_")}
        for candidate in evaluated_candidates
        if candidate["is_nondominated"]
    ]
    payload: Dict[str, object] = {
        "problem_meta": problem_meta,
        "solver_meta": solver_meta,
        "search_meta": {
            "pop_size": pop_size,
            "n_generations": n_generations,
            "seed": seed,
            "n_candidates_evaluated": len(evaluated_candidates),
            "search_runtime_s": search_runtime_s,
            "avg_candidate_runtime_s": avg_candidate_runtime_s,
        },
        "candidates": public_candidates,
        "frontier": public_frontier,
    }
    payload["candidates_json"] = write_json(
        payload["candidates"],
        search_output_dir / "nsga3_static_heuristic_candidates.json",
    )
    payload["frontier_json"] = write_json(
        payload["frontier"],
        search_output_dir / "nsga3_static_heuristic_pareto_frontier.json",
    )
    pareto_analysis = analyze_pareto_candidates(public_candidates)
    payload["pareto_analysis"] = pareto_analysis
    payload["summary_json"] = write_json(
        {
            "problem_meta": problem_meta,
            "solver_meta": solver_meta,
            "search_meta": payload["search_meta"],
            "frontier_size": len(public_frontier),
        },
        search_output_dir / "nsga3_static_heuristic_summary.json",
    )
    payload["pareto_analysis_json"] = write_json(
        pareto_analysis,
        search_output_dir / "pareto_analysis.json",
    )
    payload["charts"] = export_visualizations(
        analytics={"solver_calls": [], "gantt_tasks": []},
        output_dir=search_output_dir / "charts",
        pareto_candidates=public_candidates,
    )
    return payload
