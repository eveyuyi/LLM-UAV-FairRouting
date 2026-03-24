"""CPLEX PDP assignment model shared by the baseline and workflow.

The current workflow fixes a required pickup station for each demand before the
solver runs. We still keep that preprocessing step, but the optimization model
itself is now a pickup-and-delivery routing problem:

- binary arc variables choose a full route for each drone
- visit / serve binaries bind pickup-delivery pairs to the same drone
- MTZ order variables eliminate subtours and enforce pickup-before-delivery
- time variables propagate arrival times along the chosen route
- load variables propagate onboard cargo and enforce drone capacity

This lets a drone execute multiple pickup-and-delivery tasks in a single sortie
before returning to its home station.
"""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    Set,
    SolverFactory,
    TerminationCondition,
    Var,
    minimize,
    value,
)

from llm4fairrouting.routing.analytics import (
    parse_cplex_incumbent_trace,
    resolve_objective_weights,
    sanitize_label,
)
from llm4fairrouting.routing.domain import (
    DemandEvent,
    Drone,
    DroneState,
    Point,
    priority_service_score,
)


class CplexSolver:
    def __init__(
        self,
        drones: List[Drone],
        supply_indices: List[int],
        station_indices: List[int],
        dist_matrix: np.ndarray,
        all_points: List[Point],
        noise_cost_matrix: np.ndarray = None,
        noise_weight: float = 0.0,
        drone_activation_cost: float = 1000.0,
        time_limit: int = 10,
        analytics_output_dir: Optional[str] = None,
        enable_conflict_refiner: bool = False,
    ):
        self.drones = drones
        self.supply_indices = supply_indices
        self.station_indices = station_indices
        self.dist_matrix = dist_matrix
        self.all_points = all_points
        self.time_limit = time_limit
        self.PENALTY_UNASSIGNED = 1e9
        self.noise_cost_matrix = noise_cost_matrix
        self.noise_weight = noise_weight
        self.drone_activation_cost = drone_activation_cost
        self.analytics_output_dir = Path(analytics_output_dir) if analytics_output_dir else None
        self.enable_conflict_refiner = enable_conflict_refiner
        self.last_solve_details: Dict[str, object] = {}
        self.default_objective_weights = resolve_objective_weights(None)

    def _count_model_size(self, model: ConcreteModel) -> Dict[str, int]:
        variables = list(model.component_data_objects(Var, active=True))
        constraints = list(model.component_data_objects(Constraint, active=True))
        return {
            "variables": len(variables),
            "binary_variables": sum(1 for var in variables if var.is_binary()),
            "constraints": len(constraints),
        }

    def _build_conflict_summary(
        self,
        *,
        solve_name: str,
        termination: object,
    ) -> Dict[str, object]:
        if not self.enable_conflict_refiner:
            return {"status": "not_requested"}

        summary = {
            "status": "requested",
            "requested": True,
            "termination_condition": str(termination),
            "message": (
                "The current Pyomo shell integration can export diagnostics, but true "
                "automated CPLEX Conflict Refiner runs need a direct CPLEX API or CLI bridge."
            ),
        }
        if self.analytics_output_dir is not None:
            summary["artifact_dir"] = str(self.analytics_output_dir / "conflicts" / solve_name)
        return summary

    @staticmethod
    def _has_usable_incumbent(results: object, model: ConcreteModel) -> bool:
        solution_container = getattr(results, "solution", None)
        if solution_container is not None:
            try:
                if len(solution_container) > 0:
                    return True
            except TypeError:
                pass

        for var in model.component_data_objects(Var, active=True):
            if value(var, exception=False) is not None:
                return True
        return False

    def solve_assignment(
        self,
        drone_states: List[DroneState],
        demands: List[DemandEvent],
        current_time: float,
        objective_weights: Optional[Dict[str, float]] = None,
        solve_context: Optional[Dict[str, object]] = None,
    ) -> List[Dict]:
        if not demands or not drone_states:
            self.last_solve_details = {
                "time_window": (solve_context or {}).get("time_window"),
                "objective_weights": resolve_objective_weights(objective_weights),
                "solve_time_s": 0.0,
                "termination_condition": "skipped",
                "model_size": {},
                "convergence_trace": [],
                "conflict_refiner": {"status": "not_run"},
            }
            return []

        active_weights = resolve_objective_weights(objective_weights)
        solve_context = dict(solve_context or {})
        solve_name = sanitize_label(
            f"{solve_context.get('time_window', 'windowless')}_t{current_time:.3f}"
        )

        print(f"\n  CPLEX求解: {len(drone_states)}架无人机, {len(demands)}个需求")
        model = ConcreteModel()
        model.name = f"assignment_t{current_time:.3f}"

        start_node = -1
        end_node = -2
        n_demands = len(demands)
        pickup_nodes = {d: d for d in range(n_demands)}
        delivery_nodes = {d: n_demands + d for d in range(n_demands)}
        service_nodes = list(range(2 * n_demands))
        route_node_count = max(1, len(service_nodes))

        model.DRONES = Set(initialize=range(len(drone_states)))
        model.DEMANDS = Set(initialize=range(len(demands)))
        model.SERVICE_NODES = Set(initialize=service_nodes, ordered=True)
        drone_specs = {drone.id: drone for drone in self.drones}

        drone_initial_load = {
            u: max(0.0, float(getattr(drone_states[u], "current_load", 0.0)))
            for u in model.DRONES
        }
        model.drone_initial_load = Param(model.DRONES, initialize=drone_initial_load)
        drone_payload = {
            u: max(0.0, float(drone_states[u].remaining_payload))
            for u in model.DRONES
        }
        model.drone_available_payload = Param(model.DRONES, initialize=drone_payload)
        drone_capacity = {
            u: drone_initial_load[u] + drone_payload[u]
            for u in model.DRONES
        }
        model.drone_capacity = Param(model.DRONES, initialize=drone_capacity)
        drone_range = {u: drone_states[u].remaining_range for u in model.DRONES}
        model.drone_range = Param(model.DRONES, initialize=drone_range)
        drone_pos = {u: drone_states[u].current_node for u in model.DRONES}
        model.drone_pos = Param(model.DRONES, initialize=drone_pos)
        drone_station = {
            u: self.station_indices[drone_states[u].station_id]
            for u in model.DRONES
        }
        model.drone_station = Param(model.DRONES, initialize=drone_station)

        demand_weight = {d: float(demands[d].weight) for d in model.DEMANDS}
        model.demand_weight = Param(model.DEMANDS, initialize=demand_weight)

        demand_pos = {d: demands[d].node_idx for d in model.DEMANDS}
        model.demand_pos = Param(model.DEMANDS, initialize=demand_pos)

        required_supply_idx: Dict[int, int] = {}
        required_supply_node: Dict[int, int] = {}
        valid_required_supply: Dict[int, bool] = {}
        for d in model.DEMANDS:
            supply_idx_raw = demands[d].required_supply_idx
            try:
                supply_idx = int(supply_idx_raw)
            except (TypeError, ValueError):
                supply_idx = -1

            is_valid = 0 <= supply_idx < len(self.supply_indices)
            required_supply_idx[d] = supply_idx
            required_supply_node[d] = self.supply_indices[supply_idx] if is_valid else -1
            valid_required_supply[d] = is_valid

        node_demand: Dict[int, int] = {}
        node_kind: Dict[int, str] = {}
        node_physical: Dict[int, int] = {}
        node_load_delta: Dict[int, float] = {}
        for d in model.DEMANDS:
            pickup_node = pickup_nodes[d]
            delivery_node = delivery_nodes[d]
            node_demand[pickup_node] = d
            node_demand[delivery_node] = d
            node_kind[pickup_node] = "pickup"
            node_kind[delivery_node] = "delivery"
            node_physical[pickup_node] = required_supply_node[d]
            node_physical[delivery_node] = demand_pos[d]
            node_load_delta[pickup_node] = demand_weight[d]
            node_load_delta[delivery_node] = -demand_weight[d]

        priority_level = {d: max(1, int(demands[d].priority)) for d in model.DEMANDS}
        max_priority_level = max(priority_level.values(), default=1)
        unassigned_penalty = {
            d: self.PENALTY_UNASSIGNED * priority_service_score(
                priority_level[d], max_priority_level
            )
            for d in model.DEMANDS
        }

        arc_list: List[Tuple[int, int]] = []
        route_from_nodes = [start_node] + service_nodes
        route_to_nodes = service_nodes + [end_node]
        for i in route_from_nodes:
            if i == end_node:
                continue
            for j in route_to_nodes:
                if j == start_node or i == j:
                    continue
                if i == start_node and j == end_node:
                    continue
                arc_list.append((i, j))

        model.ARCS = Set(dimen=2, initialize=arc_list, ordered=True)

        outgoing_arcs: Dict[int, List[Tuple[int, int]]] = {}
        incoming_arcs: Dict[int, List[Tuple[int, int]]] = {}
        for i, j in arc_list:
            outgoing_arcs.setdefault(i, []).append((i, j))
            incoming_arcs.setdefault(j, []).append((i, j))

        def _physical_node_for_endpoint(u: int, node: int) -> int:
            if node == start_node:
                return int(drone_pos[u])
            if node == end_node:
                return int(drone_station[u])
            return int(node_physical[node])

        arc_distance: Dict[Tuple[int, int, int], float] = {}
        arc_noise: Dict[Tuple[int, int, int], float] = {}
        arc_time_h: Dict[Tuple[int, int, int], float] = {}
        arc_cost: Dict[Tuple[int, int, int], float] = {}
        for u in model.DRONES:
            speed = max(float(drone_specs[drone_states[u].drone_id].speed), 1e-6)
            for i, j in model.ARCS:
                from_node = _physical_node_for_endpoint(u, i)
                to_node = _physical_node_for_endpoint(u, j)
                if from_node < 0 or to_node < 0:
                    arc_distance[(u, i, j)] = 0.0
                    arc_noise[(u, i, j)] = 0.0
                    arc_time_h[(u, i, j)] = 0.0
                    arc_cost[(u, i, j)] = 0.0
                    continue

                dist = float(self.dist_matrix[from_node, to_node])
                noise = (
                    float(self.noise_cost_matrix[from_node, to_node])
                    if self.noise_cost_matrix is not None else 0.0
                )
                arc_distance[(u, i, j)] = dist
                arc_noise[(u, i, j)] = noise
                arc_time_h[(u, i, j)] = dist / speed / 3600.0
                arc_cost[(u, i, j)] = dist + self.noise_weight * noise

        max_demand_weight = max(demand_weight.values(), default=0.0)
        big_time_h = {
            u: (
                float(drone_range[u])
                / max(float(drone_specs[drone_states[u].drone_id].speed), 1e-6)
                / 3600.0
            ) + 1.0
            for u in model.DRONES
        }
        big_load = {
            u: float(drone_capacity[u]) + max_demand_weight + 1.0
            for u in model.DRONES
        }

        model.used = Var(model.DRONES, within=Binary, initialize=0)
        model.serve = Var(model.DRONES, model.DEMANDS, within=Binary, initialize=0)
        model.visit = Var(model.DRONES, model.SERVICE_NODES, within=Binary, initialize=0)
        model.x = Var(model.DRONES, model.ARCS, within=Binary, initialize=0)
        model.order = Var(
            model.DRONES,
            model.SERVICE_NODES,
            within=NonNegativeReals,
            bounds=(0, route_node_count),
            initialize=0,
        )
        model.arrival = Var(
            model.DRONES,
            model.SERVICE_NODES,
            within=NonNegativeReals,
            initialize=0,
        )
        model.end_time = Var(model.DRONES, within=NonNegativeReals, initialize=0)
        model.cargo_load = Var(
            model.DRONES,
            model.SERVICE_NODES,
            within=NonNegativeReals,
            initialize=0,
        )
        model.unassigned = Var(model.DEMANDS, within=Binary, initialize=0)
        effective_risk_weight = self.noise_weight * active_weights["w_risk"]

        def obj_rule(m):
            total_distance_cost = sum(
                arc_distance[(u, i, j)] * m.x[u, i, j]
                for u in m.DRONES
                for i, j in m.ARCS
            )
            total_time_cost = sum(
                3600.0 * m.arrival[u, delivery_nodes[d]]
                for u in m.DRONES
                for d in m.DEMANDS
            )
            total_risk_cost = sum(
                arc_noise[(u, i, j)] * m.x[u, i, j]
                for u in m.DRONES
                for i, j in m.ARCS
            )
            activation_cost = sum(
                self.drone_activation_cost * m.used[u]
                for u in m.DRONES
            )
            penalty = sum(unassigned_penalty[d] * m.unassigned[d] for d in m.DEMANDS)
            return (
                active_weights["w_distance"] * total_distance_cost
                + active_weights["w_time"] * total_time_cost
                + effective_risk_weight * total_risk_cost
                + activation_cost
                + penalty
            )

        model.obj = Objective(rule=obj_rule, sense=minimize)

        def demand_coverage(m, d):
            return sum(m.serve[u, d] for u in m.DRONES) + m.unassigned[d] == 1

        model.con_demand_coverage = Constraint(model.DEMANDS, rule=demand_coverage)

        def invalid_required_supply(m, d):
            if valid_required_supply[d]:
                return Constraint.Skip
            return m.unassigned[d] == 1

        model.con_invalid_required_supply = Constraint(
            model.DEMANDS,
            rule=invalid_required_supply,
        )

        def serve_requires_route(m, u, d):
            return m.serve[u, d] <= m.used[u]

        model.con_serve_requires_route = Constraint(
            model.DRONES,
            model.DEMANDS,
            rule=serve_requires_route,
        )

        def pickup_visit_match(m, u, d):
            return m.visit[u, pickup_nodes[d]] == m.serve[u, d]

        def delivery_visit_match(m, u, d):
            return m.visit[u, delivery_nodes[d]] == m.serve[u, d]

        model.con_pickup_visit_match = Constraint(
            model.DRONES,
            model.DEMANDS,
            rule=pickup_visit_match,
        )
        model.con_delivery_visit_match = Constraint(
            model.DRONES,
            model.DEMANDS,
            rule=delivery_visit_match,
        )

        def overweight_request_forbidden(m, u, d):
            if demand_weight[d] > drone_capacity[u] + 1e-9:
                return m.serve[u, d] == 0
            return Constraint.Skip

        model.con_overweight_request_forbidden = Constraint(
            model.DRONES,
            model.DEMANDS,
            rule=overweight_request_forbidden,
        )

        def route_start(m, u):
            return sum(m.x[u, start_node, j] for _, j in outgoing_arcs.get(start_node, [])) == m.used[u]

        def route_end(m, u):
            return sum(m.x[u, i, end_node] for i, _ in incoming_arcs.get(end_node, [])) == m.used[u]

        model.con_route_start = Constraint(model.DRONES, rule=route_start)
        model.con_route_end = Constraint(model.DRONES, rule=route_end)

        def movement_implies_used(m, u):
            return sum(m.x[u, i, j] for i, j in m.ARCS) >= m.used[u]

        def used_implies_movement(m, u):
            return sum(m.x[u, i, j] for i, j in m.ARCS) <= len(arc_list) * m.used[u]

        model.con_movement_implies_used = Constraint(
            model.DRONES,
            rule=movement_implies_used,
        )
        model.con_used_implies_movement = Constraint(
            model.DRONES,
            rule=used_implies_movement,
        )

        def inbound_flow(m, u, n):
            return sum(m.x[u, i, n] for i, _ in incoming_arcs.get(n, [])) == m.visit[u, n]

        def outbound_flow(m, u, n):
            return sum(m.x[u, n, j] for _, j in outgoing_arcs.get(n, [])) == m.visit[u, n]

        model.con_inbound_flow = Constraint(
            model.DRONES,
            model.SERVICE_NODES,
            rule=inbound_flow,
        )
        model.con_outbound_flow = Constraint(
            model.DRONES,
            model.SERVICE_NODES,
            rule=outbound_flow,
        )

        def range_constraint(m, u):
            total_dist = sum(
                arc_distance[(u, i, j)] * m.x[u, i, j]
                for i, j in m.ARCS
            )
            return total_dist <= m.drone_range[u]

        model.con_range = Constraint(model.DRONES, rule=range_constraint)

        def order_lower_bound(m, u, n):
            return m.order[u, n] >= m.visit[u, n]

        def order_upper_bound(m, u, n):
            return m.order[u, n] <= route_node_count * m.visit[u, n]

        model.con_order_lower_bound = Constraint(
            model.DRONES,
            model.SERVICE_NODES,
            rule=order_lower_bound,
        )
        model.con_order_upper_bound = Constraint(
            model.DRONES,
            model.SERVICE_NODES,
            rule=order_upper_bound,
        )

        def mtz_subtour_elimination(m, u, i, j):
            if i == j or i == start_node or j == end_node:
                return Constraint.Skip
            return m.order[u, j] >= m.order[u, i] + 1 - route_node_count * (1 - m.x[u, i, j])

        model.con_mtz = Constraint(
            model.DRONES,
            model.ARCS,
            rule=mtz_subtour_elimination,
        )

        def pickup_before_delivery(m, u, d):
            return (
                m.order[u, delivery_nodes[d]]
                >= m.order[u, pickup_nodes[d]] + 1 - route_node_count * (1 - m.serve[u, d])
            )

        model.con_pickup_before_delivery = Constraint(
            model.DRONES,
            model.DEMANDS,
            rule=pickup_before_delivery,
        )

        def arrival_upper_bound(m, u, n):
            return m.arrival[u, n] <= big_time_h[u] * m.visit[u, n]

        model.con_arrival_upper_bound = Constraint(
            model.DRONES,
            model.SERVICE_NODES,
            rule=arrival_upper_bound,
        )

        def start_time_lower(m, u, n):
            return (
                m.arrival[u, n]
                >= arc_time_h[(u, start_node, n)] - big_time_h[u] * (1 - m.x[u, start_node, n])
            )

        def start_time_upper(m, u, n):
            return (
                m.arrival[u, n]
                <= arc_time_h[(u, start_node, n)] + big_time_h[u] * (1 - m.x[u, start_node, n])
            )

        model.con_start_time_lower = Constraint(
            model.DRONES,
            model.SERVICE_NODES,
            rule=start_time_lower,
        )
        model.con_start_time_upper = Constraint(
            model.DRONES,
            model.SERVICE_NODES,
            rule=start_time_upper,
        )

        def propagation_time_lower(m, u, i, j):
            if i == start_node or j == end_node:
                return Constraint.Skip
            return (
                m.arrival[u, j]
                >= m.arrival[u, i] + arc_time_h[(u, i, j)] - big_time_h[u] * (1 - m.x[u, i, j])
            )

        def propagation_time_upper(m, u, i, j):
            if i == start_node or j == end_node:
                return Constraint.Skip
            return (
                m.arrival[u, j]
                <= m.arrival[u, i] + arc_time_h[(u, i, j)] + big_time_h[u] * (1 - m.x[u, i, j])
            )

        model.con_propagation_time_lower = Constraint(
            model.DRONES,
            model.ARCS,
            rule=propagation_time_lower,
        )
        model.con_propagation_time_upper = Constraint(
            model.DRONES,
            model.ARCS,
            rule=propagation_time_upper,
        )

        def route_end_time_upper(m, u):
            return m.end_time[u] <= big_time_h[u] * m.used[u]

        model.con_route_end_time_upper = Constraint(
            model.DRONES,
            rule=route_end_time_upper,
        )

        def route_end_time_lower(m, u, i):
            if i == start_node:
                return Constraint.Skip
            return (
                m.end_time[u]
                >= m.arrival[u, i] + arc_time_h[(u, i, end_node)] - big_time_h[u] * (1 - m.x[u, i, end_node])
            )

        def route_end_time_exact_upper(m, u, i):
            if i == start_node:
                return Constraint.Skip
            return (
                m.end_time[u]
                <= m.arrival[u, i] + arc_time_h[(u, i, end_node)] + big_time_h[u] * (1 - m.x[u, i, end_node])
            )

        model.con_route_end_time_lower = Constraint(
            model.DRONES,
            model.SERVICE_NODES,
            rule=route_end_time_lower,
        )
        model.con_route_end_time_exact_upper = Constraint(
            model.DRONES,
            model.SERVICE_NODES,
            rule=route_end_time_exact_upper,
        )

        def load_upper_bound(m, u, n):
            return m.cargo_load[u, n] <= m.drone_capacity[u] * m.visit[u, n]

        model.con_load_upper_bound = Constraint(
            model.DRONES,
            model.SERVICE_NODES,
            rule=load_upper_bound,
        )

        def start_load_lower(m, u, n):
            target_delta = node_load_delta[n]
            return (
                m.cargo_load[u, n]
                >= m.drone_initial_load[u] + target_delta - big_load[u] * (1 - m.x[u, start_node, n])
            )

        def start_load_upper(m, u, n):
            target_delta = node_load_delta[n]
            return (
                m.cargo_load[u, n]
                <= m.drone_initial_load[u] + target_delta + big_load[u] * (1 - m.x[u, start_node, n])
            )

        model.con_start_load_lower = Constraint(
            model.DRONES,
            model.SERVICE_NODES,
            rule=start_load_lower,
        )
        model.con_start_load_upper = Constraint(
            model.DRONES,
            model.SERVICE_NODES,
            rule=start_load_upper,
        )

        def propagation_load_lower(m, u, i, j):
            if i == start_node or j == end_node:
                return Constraint.Skip
            target_delta = node_load_delta[j]
            return (
                m.cargo_load[u, j]
                >= m.cargo_load[u, i] + target_delta - big_load[u] * (1 - m.x[u, i, j])
            )

        def propagation_load_upper(m, u, i, j):
            if i == start_node or j == end_node:
                return Constraint.Skip
            target_delta = node_load_delta[j]
            return (
                m.cargo_load[u, j]
                <= m.cargo_load[u, i] + target_delta + big_load[u] * (1 - m.x[u, i, j])
            )

        model.con_propagation_load_lower = Constraint(
            model.DRONES,
            model.ARCS,
            rule=propagation_load_lower,
        )
        model.con_propagation_load_upper = Constraint(
            model.DRONES,
            model.ARCS,
            rule=propagation_load_upper,
        )

        model_size = self._count_model_size(model)
        solver = SolverFactory("cplex")
        solver.options["timelimit"] = self.time_limit
        if self.enable_conflict_refiner:
            solver.options["conflictalg"] = 3

        log_path = None
        if self.analytics_output_dir is not None:
            log_dir = self.analytics_output_dir / "cplex_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{solve_name}.log"

        solve_kwargs = {"tee": True}
        if log_path is not None:
            solve_kwargs["logfile"] = str(log_path)

        solve_started_at = _time.time()
        try:
            try:
                results = solver.solve(model, **solve_kwargs)
            except TypeError:
                solve_kwargs.pop("logfile", None)
                results = solver.solve(model, **solve_kwargs)

            solve_time_s = round(_time.time() - solve_started_at, 6)
            termination = results.solver.termination_condition
            convergence_trace = parse_cplex_incumbent_trace(log_path)
            conflict_summary = self._build_conflict_summary(
                solve_name=solve_name,
                termination=termination,
            )
            incumbent_available = (
                termination == TerminationCondition.maxTimeLimit
                and self._has_usable_incumbent(results, model)
            )

            if termination in [
                TerminationCondition.optimal,
                TerminationCondition.feasible,
            ] or incumbent_available:
                if incumbent_available:
                    print("    CPLEX 达到 time limit，但已找到可行 incumbent，提取当前近似解")
                objective_breakdown = {
                    "distance_m": sum(
                        arc_distance[(u, i, j)] * float(value(model.x[u, i, j]))
                        for u in model.DRONES
                        for i, j in model.ARCS
                    ),
                    "delivery_time_s": sum(
                        3600.0 * float(value(model.arrival[u, delivery_nodes[d]]))
                        for u in model.DRONES
                        for d in model.DEMANDS
                    ),
                    "noise_impact": sum(
                        arc_noise[(u, i, j)] * float(value(model.x[u, i, j]))
                        for u in model.DRONES
                        for i, j in model.ARCS
                    ),
                    "activation_cost": sum(
                        self.drone_activation_cost * float(value(model.used[u]))
                        for u in model.DRONES
                    ),
                    "unassigned_penalty": sum(
                        unassigned_penalty[d] * float(value(model.unassigned[d]))
                        for d in model.DEMANDS
                    ),
                }
                objective_breakdown["weighted_distance_cost"] = (
                    active_weights["w_distance"] * objective_breakdown["distance_m"]
                )
                objective_breakdown["weighted_time_cost"] = (
                    active_weights["w_time"] * objective_breakdown["delivery_time_s"]
                )
                objective_breakdown["weighted_risk_cost"] = (
                    effective_risk_weight * objective_breakdown["noise_impact"]
                )
                objective_value = float(value(model.obj))

                route_plans: List[Dict] = []
                total_served = 0
                for u in model.DRONES:
                    successor: Dict[int, int] = {}
                    for i, j in model.ARCS:
                        if value(model.x[u, i, j]) > 0.5:
                            successor[i] = j

                    if start_node not in successor:
                        continue

                    current_virtual = start_node
                    route_virtual_nodes: List[int] = []
                    visited_virtual: set[int] = set()
                    while current_virtual in successor:
                        next_virtual = successor[current_virtual]
                        if next_virtual == end_node:
                            break
                        if next_virtual in visited_virtual:
                            break
                        route_virtual_nodes.append(next_virtual)
                        visited_virtual.add(next_virtual)
                        current_virtual = next_virtual

                    if not route_virtual_nodes:
                        continue

                    route_stops: List[Dict] = []
                    path_node_indices = [int(drone_pos[u])]
                    path_node_ids = [self.all_points[int(drone_pos[u])].id]
                    delivery_times_h: Dict[str, float] = {}
                    served_demands: List[Dict] = []
                    served_demand_ids: List[str] = []

                    for virtual_node in route_virtual_nodes:
                        demand_idx = node_demand[virtual_node]
                        demand = demands[demand_idx]
                        physical_node = node_physical[virtual_node]
                        stop_type = node_kind[virtual_node]
                        stop = {
                            "type": stop_type,
                            "node": physical_node,
                            "demand_id": demand.unique_id,
                            "demand_node": demand.node_idx,
                            "supply_node": required_supply_node[demand_idx],
                            "weight": float(demand.weight),
                            "priority": int(demand.priority),
                            "demand_point_id": demand.demand_point_id,
                        }
                        route_stops.append(stop)
                        path_node_indices.append(int(physical_node))
                        path_node_ids.append(self.all_points[int(physical_node)].id)

                        if stop_type == "delivery":
                            delivery_time_h = current_time + float(value(model.arrival[u, virtual_node]))
                            delivery_times_h[demand.unique_id] = delivery_time_h
                            served_demands.append({
                                "demand": demand,
                                "demand_idx": demand_idx,
                                "supply_idx": required_supply_idx[demand_idx],
                                "supply_node": required_supply_node[demand_idx],
                                "delivery_time_h": delivery_time_h,
                            })
                            served_demand_ids.append(demand.unique_id)

                    station_node = int(drone_station[u])
                    route_stops.append({
                        "type": "station",
                        "node": station_node,
                        "demand_id": None,
                        "demand_node": None,
                        "supply_node": None,
                        "weight": 0.0,
                        "priority": None,
                        "demand_point_id": None,
                    })
                    path_node_indices.append(station_node)
                    path_node_ids.append(self.all_points[station_node].id)

                    total_route_distance = sum(
                        arc_distance[(u, i, j)] * float(value(model.x[u, i, j]))
                        for i, j in model.ARCS
                    )
                    route_plans.append({
                        "drone": drone_states[u],
                        "drone_idx": u,
                        "route_stops": route_stops,
                        "route_virtual_nodes": route_virtual_nodes,
                        "served_demands": served_demands,
                        "served_demand_ids": served_demand_ids,
                        "delivery_times_h": delivery_times_h,
                        "path_node_indices": path_node_indices,
                        "path_node_ids": path_node_ids,
                        "path_str": " -> ".join(path_node_ids),
                        "total_distance_m": total_route_distance,
                        "total_mission_time_h": float(value(model.end_time[u])),
                        "total_mission_time_s": round(float(value(model.end_time[u])) * 3600.0, 1),
                    })
                    total_served += len(served_demands)

                print(
                    f"    CPLEX规划了 {len(route_plans)} 条无人机路径, 覆盖 {total_served} 个需求"
                )
                self.last_solve_details = {
                    "time_window": solve_context.get("time_window"),
                    "current_time_h": round(current_time, 6),
                    "objective_weights": active_weights,
                    "solve_time_s": solve_time_s,
                    "termination_condition": str(termination),
                    "model_size": {
                        **model_size,
                        "drones": len(drone_states),
                        "demands": len(demands),
                        "service_nodes": len(service_nodes),
                        "arcs": len(arc_list),
                    },
                    "objective_value": objective_value,
                    "objective_breakdown": objective_breakdown,
                    "convergence_trace": convergence_trace,
                    "conflict_refiner": conflict_summary,
                    "solver_log_path": str(log_path) if log_path is not None else None,
                    "accepted_with_incumbent": incumbent_available,
                }
                return route_plans
            print(f"    求解失败: {termination}")
            self.last_solve_details = {
                "time_window": solve_context.get("time_window"),
                "current_time_h": round(current_time, 6),
                "objective_weights": active_weights,
                "solve_time_s": solve_time_s,
                "termination_condition": str(termination),
                "model_size": {
                    **model_size,
                    "drones": len(drone_states),
                    "demands": len(demands),
                    "service_nodes": len(service_nodes),
                    "arcs": len(arc_list),
                },
                "objective_value": None,
                "objective_breakdown": None,
                "convergence_trace": convergence_trace,
                "conflict_refiner": conflict_summary,
                "solver_log_path": str(log_path) if log_path is not None else None,
            }
            return []
        except Exception as e:
            print(f"    CPLEX异常: {e}")
            self.last_solve_details = {
                "time_window": solve_context.get("time_window"),
                "current_time_h": round(current_time, 6),
                "objective_weights": active_weights,
                "solve_time_s": round(_time.time() - solve_started_at, 6),
                "termination_condition": "error",
                "model_size": {
                    **model_size,
                    "drones": len(drone_states),
                    "demands": len(demands),
                    "service_nodes": len(service_nodes),
                    "arcs": len(arc_list),
                },
                "objective_value": None,
                "objective_breakdown": None,
                "convergence_trace": parse_cplex_incumbent_trace(log_path),
                "conflict_refiner": self._build_conflict_summary(
                    solve_name=solve_name,
                    termination="error",
                ),
                "solver_log_path": str(log_path) if log_path is not None else None,
                "error": str(e),
            }
            return []
