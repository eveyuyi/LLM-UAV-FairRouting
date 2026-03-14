"""CPLEX assignment model shared by the baseline and workflow.

The workflow fixes a single supply node for each demand before the solver runs.
That lets us optimize the assignment model from a full ``drone x supply x demand``
tensor down to a direct ``drone x demand`` decision space without changing the
underlying routing semantics.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Objective,
    Param,
    Set,
    SolverFactory,
    TerminationCondition,
    Var,
    minimize,
    value,
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
        time_limit: int = 10,
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

    def solve_assignment(
        self,
        drone_states: List[DroneState],
        demands: List[DemandEvent],
        current_time: float,
    ) -> List[Dict]:
        if not demands or not drone_states:
            return []

        print(f"\n  CPLEX求解: {len(drone_states)}架无人机, {len(demands)}个需求")
        model = ConcreteModel()
        model.name = f"assignment_t{current_time:.3f}"

        model.DRONES = Set(initialize=range(len(drone_states)))
        model.DEMANDS = Set(initialize=range(len(demands)))

        drone_payload = {u: drone_states[u].remaining_payload for u in model.DRONES}
        model.drone_payload = Param(model.DRONES, initialize=drone_payload)
        drone_range = {u: drone_states[u].remaining_range for u in model.DRONES}
        model.drone_range = Param(model.DRONES, initialize=drone_range)
        drone_pos = {u: drone_states[u].current_node for u in model.DRONES}
        model.drone_pos = Param(model.DRONES, initialize=drone_pos)
        drone_station = {
            u: self.station_indices[drone_states[u].station_id]
            for u in model.DRONES
        }
        model.drone_station = Param(model.DRONES, initialize=drone_station)

        demand_weight = {d: demands[d].weight for d in model.DEMANDS}
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

        priority_level = {d: max(1, int(demands[d].priority)) for d in model.DEMANDS}
        max_priority_level = max(priority_level.values(), default=1)
        unassigned_penalty = {
            d: self.PENALTY_UNASSIGNED * priority_service_score(
                priority_level[d], max_priority_level
            )
            for d in model.DEMANDS
        }

        trip_distance = {}
        trip_noise = {}
        for u in model.DRONES:
            for d in model.DEMANDS:
                if not valid_required_supply[d]:
                    trip_distance[(u, d)] = 0.0
                    trip_noise[(u, d)] = 0.0
                    continue

                supply_node = required_supply_node[d]
                trip_distance[(u, d)] = (
                    self.dist_matrix[drone_pos[u], supply_node]
                    + self.dist_matrix[supply_node, demand_pos[d]]
                    + self.dist_matrix[demand_pos[d], drone_station[u]]
                )
                if self.noise_cost_matrix is not None:
                    trip_noise[(u, d)] = (
                        self.noise_cost_matrix[drone_pos[u], supply_node]
                        + self.noise_cost_matrix[supply_node, demand_pos[d]]
                        + self.noise_cost_matrix[demand_pos[d], drone_station[u]]
                    )
                else:
                    trip_noise[(u, d)] = 0.0

        trip_cost = {
            (u, d): trip_distance[(u, d)] + self.noise_weight * trip_noise[(u, d)]
            for u in model.DRONES
            for d in model.DEMANDS
        }

        model.assign = Var(model.DRONES, model.DEMANDS, within=Binary)
        model.unassigned = Var(model.DEMANDS, within=Binary)

        def obj_rule(m):
            total_assignment_cost = sum(
                trip_cost[(u, d)] * m.assign[u, d]
                for u in m.DRONES
                for d in m.DEMANDS
            )
            penalty = sum(unassigned_penalty[d] * m.unassigned[d] for d in m.DEMANDS)
            return total_assignment_cost + penalty

        model.obj = Objective(rule=obj_rule, sense=minimize)

        def demand_coverage(m, d):
            return sum(m.assign[u, d] for u in m.DRONES) + m.unassigned[d] == 1

        model.con_demand_coverage = Constraint(model.DEMANDS, rule=demand_coverage)

        def payload_constraint(m, u):
            return sum(m.demand_weight[d] * m.assign[u, d] for d in m.DEMANDS) <= m.drone_payload[u]

        model.con_payload = Constraint(model.DRONES, rule=payload_constraint)

        def range_constraint(m, u):
            total_dist = sum(
                trip_distance[(u, d)] * m.assign[u, d]
                for d in m.DEMANDS
            )
            return total_dist <= m.drone_range[u]

        model.con_range = Constraint(model.DRONES, rule=range_constraint)

        def required_supply_validity(m, u, d):
            if not valid_required_supply[d]:
                return m.assign[u, d] == 0
            return Constraint.Skip

        model.con_required_supply_validity = Constraint(
            model.DRONES,
            model.DEMANDS,
            rule=required_supply_validity,
        )

        def single_task_per_drone(m, u):
            return sum(m.assign[u, d] for d in m.DEMANDS) <= 1

        model.con_single_task = Constraint(model.DRONES, rule=single_task_per_drone)

        solver = SolverFactory("cplex")
        solver.options["timelimit"] = self.time_limit
        try:
            results = solver.solve(model, tee=True)
            if results.solver.termination_condition in [
                TerminationCondition.optimal,
                TerminationCondition.feasible,
            ]:
                assignments = []
                for u in model.DRONES:
                    for d in model.DEMANDS:
                        if value(model.assign[u, d]) > 0.5:
                            supply_idx = required_supply_idx[d]
                            assignments.append({
                                "drone": drone_states[u],
                                "drone_idx": u,
                                "demand": demands[d],
                                "demand_idx": d,
                                "supply_idx": supply_idx,
                                "supply_node": self.supply_indices[supply_idx],
                            })
                print(f"    CPLEX分配了 {len(assignments)} 个需求")
                return assignments
            print(f"    求解失败: {results.solver.termination_condition}")
            return []
        except Exception as e:
            print(f"    CPLEX异常: {e}")
            return []
