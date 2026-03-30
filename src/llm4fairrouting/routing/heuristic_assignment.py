"""Greedy assignment heuristic used by the standalone NSGA-III backend."""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from llm4fairrouting.routing.analytics import resolve_objective_weights
from llm4fairrouting.routing.domain import (
    DemandEvent,
    Drone,
    DroneState,
    Point,
    priority_service_score,
)


class HeuristicAssignmentSolver:
    """Fast greedy route builder with the same public interface as ``CplexSolver``."""

    def __init__(
        self,
        drones: List[Drone],
        supply_indices: List[int],
        station_indices: List[int],
        dist_matrix: np.ndarray,
        all_points: List[Point],
        noise_cost_matrix: np.ndarray = None,
        noise_weight: float = 0.0,
        drone_activation_cost: float = 10000.0,
        time_limit: int = 0,
        analytics_output_dir: Optional[str] = None,
        enable_conflict_refiner: bool = False,
    ):
        self.drones = drones
        self.supply_indices = supply_indices
        self.station_indices = station_indices
        self.dist_matrix = dist_matrix
        self.all_points = all_points
        self.noise_cost_matrix = noise_cost_matrix
        self.noise_weight = noise_weight
        self.drone_activation_cost = drone_activation_cost
        self.time_limit = time_limit
        self.analytics_output_dir = Path(analytics_output_dir) if analytics_output_dir else None
        self.enable_conflict_refiner = enable_conflict_refiner
        self.last_solve_details: Dict[str, object] = {}
        self.PENALTY_UNASSIGNED = 1e9

    @staticmethod
    def _matrix_value(matrix, i: int, j: int) -> float:
        return float(matrix[int(i), int(j)])

    def _distance(self, i: int, j: int) -> float:
        return self._matrix_value(self.dist_matrix, i, j)

    def _noise(self, i: int, j: int) -> float:
        if self.noise_cost_matrix is None:
            return 0.0
        return self._matrix_value(self.noise_cost_matrix, i, j)

    def _demand_supply_node(self, demand: DemandEvent) -> Optional[int]:
        try:
            supply_idx = int(demand.required_supply_idx)
        except (TypeError, ValueError):
            return None
        if not 0 <= supply_idx < len(self.supply_indices):
            return None
        return int(self.supply_indices[supply_idx])

    def _bundle_metrics(
        self,
        *,
        drone_state: DroneState,
        pickup_order: Sequence[DemandEvent],
        delivery_order: Sequence[DemandEvent],
        current_time: float,
    ) -> Dict[str, object]:
        drone_spec = next(drone for drone in self.drones if drone.id == drone_state.drone_id)
        speed_ms = max(float(drone_spec.speed), 1e-6)
        current_node = int(drone_state.current_node)
        home_station = int(self.station_indices[drone_state.station_id])
        current_time_h = float(current_time)
        total_distance_m = 0.0
        total_noise = 0.0
        total_delivery_time_s = 0.0
        path_node_indices = [current_node]
        path_node_ids = [self.all_points[current_node].id]
        route_stops: List[Dict[str, object]] = []
        delivery_times_h: Dict[str, float] = {}
        served_demand_ids: List[str] = []
        served_demands: List[Dict[str, object]] = []

        for demand in pickup_order:
            supply_node = self._demand_supply_node(demand)
            if supply_node is None:
                return {"feasible": False, "reason": "invalid_supply"}
            dist = self._distance(current_node, supply_node)
            noise = self._noise(current_node, supply_node)
            total_distance_m += dist
            total_noise += noise
            current_time_h += dist / speed_ms / 3600.0
            current_node = supply_node
            route_stops.append(
                {
                    "type": "pickup",
                    "node": supply_node,
                    "demand_id": demand.unique_id,
                    "demand_node": int(demand.node_idx),
                    "supply_node": supply_node,
                    "weight": float(demand.weight),
                    "priority": int(demand.priority),
                    "demand_point_id": demand.demand_point_id,
                }
            )
            path_node_indices.append(supply_node)
            path_node_ids.append(self.all_points[supply_node].id)

        for demand in delivery_order:
            demand_node = int(demand.node_idx)
            supply_node = self._demand_supply_node(demand)
            if supply_node is None:
                return {"feasible": False, "reason": "invalid_supply"}
            dist = self._distance(current_node, demand_node)
            noise = self._noise(current_node, demand_node)
            total_distance_m += dist
            total_noise += noise
            current_time_h += dist / speed_ms / 3600.0
            current_node = demand_node
            delivery_times_h[demand.unique_id] = current_time_h
            total_delivery_time_s += max(0.0, current_time_h - current_time) * 3600.0
            served_demand_ids.append(demand.unique_id)
            served_demands.append(
                {
                    "demand": demand,
                    "demand_idx": None,
                    "supply_idx": int(demand.required_supply_idx) if demand.required_supply_idx is not None else None,
                    "supply_node": supply_node,
                    "delivery_time_h": current_time_h,
                }
            )
            route_stops.append(
                {
                    "type": "delivery",
                    "node": demand_node,
                    "demand_id": demand.unique_id,
                    "demand_node": demand_node,
                    "supply_node": supply_node,
                    "weight": float(demand.weight),
                    "priority": int(demand.priority),
                    "demand_point_id": demand.demand_point_id,
                }
            )
            path_node_indices.append(demand_node)
            path_node_ids.append(self.all_points[demand_node].id)

        dist = self._distance(current_node, home_station)
        noise = self._noise(current_node, home_station)
        total_distance_m += dist
        total_noise += noise
        current_time_h += dist / speed_ms / 3600.0
        route_stops.append(
            {
                "type": "station",
                "node": home_station,
                "demand_id": None,
                "demand_node": None,
                "supply_node": None,
                "weight": 0.0,
                "priority": None,
                "demand_point_id": None,
            }
        )
        path_node_indices.append(home_station)
        path_node_ids.append(self.all_points[home_station].id)

        return {
            "feasible": True,
            "route_stops": route_stops,
            "path_node_indices": path_node_indices,
            "path_node_ids": path_node_ids,
            "path_str": " -> ".join(path_node_ids),
            "distance_m": total_distance_m,
            "noise_impact": total_noise,
            "delivery_time_s": total_delivery_time_s,
            "delivery_times_h": delivery_times_h,
            "served_demand_ids": served_demand_ids,
            "served_demands": served_demands,
            "mission_time_h": max(0.0, current_time_h - current_time),
            "mission_time_s": round(max(0.0, current_time_h - current_time) * 3600.0, 1),
            "bundle_weight": sum(float(d.weight) for d in pickup_order),
        }

    def _bundle_objective(
        self,
        *,
        drone_state: DroneState,
        pickup_order: Sequence[DemandEvent],
        delivery_order: Sequence[DemandEvent],
        current_time: float,
        active_weights: Dict[str, float],
        include_activation_cost: bool,
    ) -> Tuple[float, Dict[str, object]]:
        metrics = self._bundle_metrics(
            drone_state=drone_state,
            pickup_order=pickup_order,
            delivery_order=delivery_order,
            current_time=current_time,
        )
        if not metrics.get("feasible"):
            return float("inf"), metrics
        objective = (
            active_weights["w_distance"] * float(metrics["distance_m"])
            + active_weights["w_time"] * float(metrics["delivery_time_s"])
            + self.noise_weight * active_weights["w_risk"] * float(metrics["noise_impact"])
        )
        if include_activation_cost and pickup_order:
            objective += self.drone_activation_cost
        metrics["objective_value"] = objective
        return objective, metrics

    def _improve_order(
        self,
        order: Sequence[DemandEvent],
        evaluator: Callable[[Sequence[DemandEvent]], float],
    ) -> Tuple[List[DemandEvent], int]:
        order = list(order)
        if len(order) < 2:
            return order, 0

        current_cost = evaluator(order)
        moves_applied = 0
        improved = True
        while improved:
            improved = False
            best_order = list(order)
            best_cost = current_cost

            for i in range(len(order)):
                for j in range(i + 1, len(order)):
                    candidate = list(order)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    score = evaluator(candidate)
                    if score + 1e-9 < best_cost:
                        best_order = candidate
                        best_cost = score

            for i in range(len(order)):
                for j in range(len(order)):
                    if i == j:
                        continue
                    candidate = list(order)
                    item = candidate.pop(i)
                    candidate.insert(j, item)
                    score = evaluator(candidate)
                    if score + 1e-9 < best_cost:
                        best_order = candidate
                        best_cost = score

            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    candidate = list(order)
                    candidate[i : j + 1] = reversed(candidate[i : j + 1])
                    score = evaluator(candidate)
                    if score + 1e-9 < best_cost:
                        best_order = candidate
                        best_cost = score

            if best_cost + 1e-9 < current_cost:
                order = best_order
                current_cost = best_cost
                moves_applied += 1
                improved = True

        return list(order), moves_applied

    def _select_bundle_for_drone(
        self,
        *,
        drone_state: DroneState,
        available_demands: List[DemandEvent],
        current_time: float,
        active_weights: Dict[str, float],
        max_priority_level: int,
    ) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
        remaining_capacity = float(drone_state.remaining_payload)
        remaining_range = float(drone_state.remaining_range)
        pickup_order: List[DemandEvent] = []
        delivery_order: List[DemandEvent] = []
        selected_ids: List[str] = []
        selection_attempts = 0
        local_search_moves = 0
        current_objective = 0.0

        while True:
            best_choice: Optional[Dict[str, object]] = None
            for demand in available_demands:
                if demand.unique_id in selected_ids:
                    continue
                if float(demand.weight) > remaining_capacity + 1e-9:
                    continue
                if self._demand_supply_node(demand) is None:
                    continue
                selection_attempts += 1
                for pickup_pos in range(len(pickup_order) + 1):
                    candidate_pickups = list(pickup_order)
                    candidate_pickups.insert(pickup_pos, demand)
                    for delivery_pos in range(len(delivery_order) + 1):
                        candidate_deliveries = list(delivery_order)
                        candidate_deliveries.insert(delivery_pos, demand)
                        objective, metrics = self._bundle_objective(
                            drone_state=drone_state,
                            pickup_order=candidate_pickups,
                            delivery_order=candidate_deliveries,
                            current_time=current_time,
                            active_weights=active_weights,
                            include_activation_cost=True,
                        )
                        if not metrics.get("feasible"):
                            continue
                        if float(metrics["distance_m"]) > remaining_range + 1e-9:
                            continue
                        if float(metrics["bundle_weight"]) > float(drone_state.remaining_payload) + 1e-9:
                            continue
                        delta_objective = objective - current_objective
                        wait_h = max(0.0, float(current_time) - float(demand.time))
                        priority_score = max(1, priority_service_score(int(demand.priority), max_priority_level))
                        normalized_delta = delta_objective / priority_score
                        selection_key = (
                            int(demand.priority),
                            normalized_delta - 300.0 * wait_h,
                            -wait_h,
                            float(metrics["distance_m"]),
                            str(demand.unique_id),
                        )
                        if best_choice is None or selection_key < best_choice["selection_key"]:
                            best_choice = {
                                "demand": demand,
                                "pickup_order": candidate_pickups,
                                "delivery_order": candidate_deliveries,
                                "metrics": metrics,
                                "objective": objective,
                                "selection_key": selection_key,
                            }

            if best_choice is None:
                break

            pickup_order = best_choice["pickup_order"]
            delivery_order = best_choice["delivery_order"]
            current_objective = float(best_choice["objective"])
            chosen_demand = best_choice["demand"]
            selected_ids.append(chosen_demand.unique_id)
            remaining_capacity -= float(chosen_demand.weight)

            pickup_order, pickup_moves = self._improve_order(
                pickup_order,
                lambda order: self._bundle_objective(
                    drone_state=drone_state,
                    pickup_order=order,
                    delivery_order=delivery_order,
                    current_time=current_time,
                    active_weights=active_weights,
                    include_activation_cost=True,
                )[0],
            )
            delivery_order, delivery_moves = self._improve_order(
                delivery_order,
                lambda order: self._bundle_objective(
                    drone_state=drone_state,
                    pickup_order=pickup_order,
                    delivery_order=order,
                    current_time=current_time,
                    active_weights=active_weights,
                    include_activation_cost=True,
                )[0],
            )
            local_search_moves += pickup_moves + delivery_moves
            current_objective, improved_metrics = self._bundle_objective(
                drone_state=drone_state,
                pickup_order=pickup_order,
                delivery_order=delivery_order,
                current_time=current_time,
                active_weights=active_weights,
                include_activation_cost=True,
            )
            if not improved_metrics.get("feasible") or float(improved_metrics["distance_m"]) > remaining_range + 1e-9:
                pickup_order = best_choice["pickup_order"]
                delivery_order = best_choice["delivery_order"]
                current_objective = float(best_choice["objective"])

        if not pickup_order:
            return None, {
                "selection_attempts": selection_attempts,
                "local_search_moves": local_search_moves,
                "selected_demands": 0,
            }

        final_objective, final_metrics = self._bundle_objective(
            drone_state=drone_state,
            pickup_order=pickup_order,
            delivery_order=delivery_order,
            current_time=current_time,
            active_weights=active_weights,
            include_activation_cost=True,
        )
        if not final_metrics.get("feasible"):
            return None, {
                "selection_attempts": selection_attempts,
                "local_search_moves": local_search_moves,
                "selected_demands": 0,
            }

        route_plan = {
            "drone": drone_state,
            "drone_idx": None,
            "route_stops": list(final_metrics["route_stops"]),
            "route_virtual_nodes": [],
            "served_demands": list(final_metrics["served_demands"]),
            "served_demand_ids": list(final_metrics["served_demand_ids"]),
            "delivery_times_h": dict(final_metrics["delivery_times_h"]),
            "path_node_indices": list(final_metrics["path_node_indices"]),
            "path_node_ids": list(final_metrics["path_node_ids"]),
            "path_str": str(final_metrics["path_str"]),
            "total_distance_m": float(final_metrics["distance_m"]),
            "total_mission_time_h": float(final_metrics["mission_time_h"]),
            "total_mission_time_s": float(final_metrics["mission_time_s"]),
        }
        route_metrics = {
            "distance_m": float(final_metrics["distance_m"]),
            "noise_impact": float(final_metrics["noise_impact"]),
            "delivery_time_s": float(final_metrics["delivery_time_s"]),
            "activation_cost": float(self.drone_activation_cost),
            "objective_value": float(final_objective),
            "selection_attempts": selection_attempts,
            "local_search_moves": local_search_moves,
            "selected_demands": len(selected_ids),
        }
        return route_plan, route_metrics

    def solve_assignment(
        self,
        drone_states: List[DroneState],
        demands: List[DemandEvent],
        current_time: float,
        objective_weights: Optional[Dict[str, float]] = None,
        solve_context: Optional[Dict[str, object]] = None,
    ) -> List[Dict]:
        solve_started_at = _time.time()
        solve_context = dict(solve_context or {})
        active_weights = resolve_objective_weights(objective_weights)

        if not demands or not drone_states:
            self.last_solve_details = {
                "time_window": solve_context.get("time_window"),
                "current_time_h": round(current_time, 6),
                "objective_weights": active_weights,
                "solve_time_s": 0.0,
                "termination_condition": "heuristic_skipped",
                "model_size": {"drones": len(drone_states), "demands": len(demands)},
                "objective_value": 0.0,
                "objective_breakdown": {
                    "distance_m": 0.0,
                    "delivery_time_s": 0.0,
                    "noise_impact": 0.0,
                    "activation_cost": 0.0,
                    "unassigned_penalty": 0.0,
                    "weighted_distance_cost": 0.0,
                    "weighted_time_cost": 0.0,
                    "weighted_risk_cost": 0.0,
                },
                "convergence_trace": [],
                "conflict_refiner": {"status": "not_requested"},
                "solver_log_path": None,
                "assignment_backend": "heuristic",
            }
            return []

        available_demands = sorted(
            list(demands),
            key=lambda demand: (int(demand.priority), float(demand.time), str(demand.unique_id)),
        )
        max_priority_level = max((max(1, int(d.priority)) for d in available_demands), default=1)
        route_plans: List[Dict[str, object]] = []
        objective_breakdown = {
            "distance_m": 0.0,
            "delivery_time_s": 0.0,
            "noise_impact": 0.0,
            "activation_cost": 0.0,
            "unassigned_penalty": 0.0,
        }
        total_selection_attempts = 0
        total_local_search_moves = 0
        served_ids: set[str] = set()

        for drone_state in drone_states:
            route_plan, metrics = self._select_bundle_for_drone(
                drone_state=drone_state,
                available_demands=available_demands,
                current_time=current_time,
                active_weights=active_weights,
                max_priority_level=max_priority_level,
            )
            total_selection_attempts += int(metrics.get("selection_attempts", 0))
            total_local_search_moves += int(metrics.get("local_search_moves", 0))
            if route_plan is None:
                continue
            route_plans.append(route_plan)
            served_ids.update(route_plan["served_demand_ids"])
            objective_breakdown["distance_m"] += float(metrics["distance_m"])
            objective_breakdown["delivery_time_s"] += float(metrics["delivery_time_s"])
            objective_breakdown["noise_impact"] += float(metrics["noise_impact"])
            objective_breakdown["activation_cost"] += float(metrics["activation_cost"])
            # Prevent later drones from being assigned the same demand bundle again.
            available_demands = [
                demand for demand in available_demands
                if demand.unique_id not in served_ids
            ]
            if not available_demands:
                break

        remaining_demands = [demand for demand in available_demands if demand.unique_id not in served_ids]
        for demand in remaining_demands:
            objective_breakdown["unassigned_penalty"] += self.PENALTY_UNASSIGNED * priority_service_score(
                int(demand.priority),
                max_priority_level,
            )

        objective_breakdown["weighted_distance_cost"] = (
            active_weights["w_distance"] * objective_breakdown["distance_m"]
        )
        objective_breakdown["weighted_time_cost"] = (
            active_weights["w_time"] * objective_breakdown["delivery_time_s"]
        )
        objective_breakdown["weighted_risk_cost"] = (
            self.noise_weight * active_weights["w_risk"] * objective_breakdown["noise_impact"]
        )
        objective_value = (
            objective_breakdown["weighted_distance_cost"]
            + objective_breakdown["weighted_time_cost"]
            + objective_breakdown["weighted_risk_cost"]
            + objective_breakdown["activation_cost"]
            + objective_breakdown["unassigned_penalty"]
        )

        self.last_solve_details = {
            "time_window": solve_context.get("time_window"),
            "current_time_h": round(current_time, 6),
            "objective_weights": active_weights,
            "solve_time_s": round(_time.time() - solve_started_at, 6),
            "termination_condition": "heuristic_completed",
            "model_size": {
                "drones": len(drone_states),
                "demands": len(demands),
                "heuristic_selection_attempts": total_selection_attempts,
                "local_search_moves_applied": total_local_search_moves,
            },
            "objective_value": objective_value,
            "objective_breakdown": objective_breakdown,
            "convergence_trace": [],
            "conflict_refiner": {"status": "not_requested"},
            "solver_log_path": None,
            "assignment_backend": "heuristic",
        }
        return route_plans
