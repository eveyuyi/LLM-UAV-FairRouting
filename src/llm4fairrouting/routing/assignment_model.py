"""CPLEX assignment model shared by the baseline and workflow."""

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
        model.SUPPLY = Set(initialize=range(len(self.supply_indices)))

        drone_payload = {u: drone_states[u].remaining_payload for u in model.DRONES}
        model.drone_payload = Param(model.DRONES, initialize=drone_payload)
        drone_range = {u: drone_states[u].remaining_range for u in model.DRONES}
        model.drone_range = Param(model.DRONES, initialize=drone_range)
        drone_pos = {u: drone_states[u].current_node for u in model.DRONES}
        model.drone_pos = Param(model.DRONES, initialize=drone_pos)
        drone_station = {u: self.station_indices[drone_states[u].station_id] for u in model.DRONES}
        model.drone_station = Param(model.DRONES, initialize=drone_station)

        demand_weight = {d: demands[d].weight for d in model.DEMANDS}
        model.demand_weight = Param(model.DEMANDS, initialize=demand_weight)
        demand_pos = {d: demands[d].node_idx for d in model.DEMANDS}
        model.demand_pos = Param(model.DEMANDS, initialize=demand_pos)
        demand_required_supply = {d: demands[d].required_supply_idx for d in model.DEMANDS}
        model.demand_required_supply = Param(model.DEMANDS, initialize=demand_required_supply)

        priority_level = {d: max(1, int(demands[d].priority)) for d in model.DEMANDS}
        max_priority_level = max(priority_level.values(), default=1)
        unassigned_penalty = {
            d: self.PENALTY_UNASSIGNED * priority_service_score(
                priority_level[d], max_priority_level
            )
            for d in model.DEMANDS
        }

        dist_drone_supply = {
            (u, s): self.dist_matrix[drone_pos[u], self.supply_indices[s]]
            for u in model.DRONES for s in model.SUPPLY
        }
        dist_supply_demand = {
            (s, d): self.dist_matrix[self.supply_indices[s], demand_pos[d]]
            for s in model.SUPPLY for d in model.DEMANDS
        }
        dist_demand_station = {
            (d, u): self.dist_matrix[demand_pos[d], drone_station[u]]
            for u in model.DRONES for d in model.DEMANDS
        }

        if self.noise_cost_matrix is not None:
            noise_drone_supply = {
                (u, s): self.noise_cost_matrix[drone_pos[u], self.supply_indices[s]]
                for u in model.DRONES for s in model.SUPPLY
            }
            noise_supply_demand = {
                (s, d): self.noise_cost_matrix[self.supply_indices[s], demand_pos[d]]
                for s in model.SUPPLY for d in model.DEMANDS
            }
            noise_demand_station = {
                (d, u): self.noise_cost_matrix[demand_pos[d], drone_station[u]]
                for u in model.DRONES for d in model.DEMANDS
            }
        else:
            noise_drone_supply = {(u, s): 0 for u in model.DRONES for s in model.SUPPLY}
            noise_supply_demand = {(s, d): 0 for s in model.SUPPLY for d in model.DEMANDS}
            noise_demand_station = {(d, u): 0 for u in model.DRONES for d in model.DEMANDS}

        model.x = Var(model.DRONES, model.SUPPLY, model.DEMANDS, within=Binary)
        model.y = Var(model.DRONES, model.DEMANDS, within=Binary)
        model.z = Var(model.DRONES, model.SUPPLY, within=Binary)
        model.used = Var(model.DRONES, within=Binary)
        model.unassigned = Var(model.DEMANDS, within=Binary)

        def obj_rule(m):
            total_assignment_cost = 0
            for u in m.DRONES:
                for s in m.SUPPLY:
                    for d in m.DEMANDS:
                        dist = (
                            dist_drone_supply[(u, s)]
                            + dist_supply_demand[(s, d)]
                            + dist_demand_station[(d, u)]
                        )
                        noise = (
                            noise_drone_supply[(u, s)]
                            + noise_supply_demand[(s, d)]
                            + noise_demand_station[(d, u)]
                        )
                        total_cost = dist + self.noise_weight * noise
                        total_assignment_cost += total_cost * m.x[u, s, d]
            penalty = sum(unassigned_penalty[d] * m.unassigned[d] for d in m.DEMANDS)
            return total_assignment_cost + penalty

        model.obj = Objective(rule=obj_rule, sense=minimize)

        def demand_coverage(m, d):
            return sum(m.y[u, d] for u in m.DRONES) + m.unassigned[d] == 1

        model.con_demand_coverage = Constraint(model.DEMANDS, rule=demand_coverage)

        def payload_constraint(m, u):
            return sum(m.demand_weight[d] * m.y[u, d] for d in m.DEMANDS) <= m.drone_payload[u]

        model.con_payload = Constraint(model.DRONES, rule=payload_constraint)

        def range_constraint(m, u):
            total_dist = 0
            for s in m.SUPPLY:
                for d in m.DEMANDS:
                    total_dist += (
                        dist_drone_supply[(u, s)]
                        + dist_supply_demand[(s, d)]
                        + dist_demand_station[(d, u)]
                    ) * m.x[u, s, d]
            return total_dist <= m.drone_range[u]

        model.con_range = Constraint(model.DRONES, rule=range_constraint)

        def usage_link(m, u, d):
            return m.y[u, d] <= m.used[u]

        model.con_usage_link = Constraint(model.DRONES, model.DEMANDS, rule=usage_link)

        def single_supply(m, u):
            return sum(m.z[u, s] for s in m.SUPPLY) <= m.used[u]

        model.con_single_supply = Constraint(model.DRONES, rule=single_supply)

        def need_supply(m, u):
            return sum(m.y[u, d] for d in m.DEMANDS) <= sum(m.z[u, s] for s in m.SUPPLY) * len(m.DEMANDS)

        model.con_need_supply = Constraint(model.DRONES, rule=need_supply)

        def x_to_y(m, u, s, d):
            return m.x[u, s, d] <= m.y[u, d]

        model.con_x_to_y = Constraint(model.DRONES, model.SUPPLY, model.DEMANDS, rule=x_to_y)

        def x_to_z(m, u, s, d):
            return m.x[u, s, d] <= m.z[u, s]

        model.con_x_to_z = Constraint(model.DRONES, model.SUPPLY, model.DEMANDS, rule=x_to_z)

        def y_to_x(m, u, d):
            return m.y[u, d] <= sum(m.x[u, s, d] for s in m.SUPPLY)

        model.con_y_to_x = Constraint(model.DRONES, model.DEMANDS, rule=y_to_x)

        def supply_demand_matching(m, u, s, d):
            if s != m.demand_required_supply[d]:
                return m.x[u, s, d] == 0
            return Constraint.Skip

        model.con_supply_demand_matching = Constraint(
            model.DRONES, model.SUPPLY, model.DEMANDS, rule=supply_demand_matching
        )

        def single_task_per_drone(m, u):
            return sum(m.y[u, d] for d in m.DEMANDS) <= 1

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
                    if value(model.used[u]) > 0.5:
                        for d in model.DEMANDS:
                            if value(model.y[u, d]) > 0.5:
                                for s in model.SUPPLY:
                                    if value(model.z[u, s]) > 0.5:
                                        assignments.append({
                                            "drone": drone_states[u],
                                            "drone_idx": u,
                                            "demand": demands[d],
                                            "demand_idx": d,
                                            "supply_idx": s,
                                            "supply_node": self.supply_indices[s],
                                        })
                                        break
                print(f"    CPLEX分配了 {len(assignments)} 个需求")
                return assignments
            print(f"    求解失败: {results.solver.termination_condition}")
            return []
        except Exception as e:
            print(f"    CPLEX异常: {e}")
            return []
