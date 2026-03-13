"""Dynamic routing simulator shared across workflow and baselines."""

from __future__ import annotations

import heapq
import math
from typing import Dict, List

import numpy as np

from llm4fairrouting.routing.assignment_model import CplexSolver
from llm4fairrouting.routing.domain import DemandEvent, Drone, DroneState, DroneStatus, Point
from llm4fairrouting.routing.serialization import serialize_simulator_snapshot


class FinalDroneSimulator:
    def __init__(
        self,
        supply_points: List[Point],
        demand_points: List[Point],
        station_points: List[Point],
        drones_static: List[Drone],
        dist_matrix: np.ndarray,
        demand_events: List[DemandEvent],
        noise_cost_matrix: np.ndarray = None,
        noise_weight: float = 0.0,
        time_step: float = 0.001,
        solve_interval: float = 0.05,
        time_limit: int = 10,
    ):
        self.supply_points = supply_points
        self.demand_points = demand_points
        self.station_points = station_points
        self.drones_static = drones_static
        self.dist_matrix = dist_matrix
        self.all_demand_events = demand_events
        self.time_step = time_step
        self.solve_interval = solve_interval

        self.all_points = supply_points + demand_points + station_points
        self.n_supply = len(supply_points)
        self.n_demand = len(demand_points)
        self.n_station = len(station_points)

        self.supply_indices = list(range(self.n_supply))
        self.demand_indices = list(range(self.n_supply, self.n_supply + self.n_demand))
        self.station_indices = list(range(
            self.n_supply + self.n_demand,
            self.n_supply + self.n_demand + self.n_station,
        ))

        self.drone_states = []
        for d in drones_static:
            station_node = self.station_indices[d.station_id]
            ds = DroneState(
                drone_id=d.id,
                station_id=d.station_id,
                current_node=station_node,
                remaining_range=d.max_range,
                remaining_payload=d.max_payload,
                status=DroneStatus.IDLE,
                position_x=self.all_points[station_node].x,
                position_y=self.all_points[station_node].y,
                executed_path=[station_node],
                task_queue=[],
            )
            self.drone_states.append(ds)

        print(f"\n初始化 {len(self.drone_states)} 架无人机")

        self.unserved_demands = []
        self.unserved_demands_dict = {}
        self.completed_demands = []

        self.event_queue = []
        for ev in demand_events:
            heapq.heappush(self.event_queue, (ev.time, ev))

        self.current_time = 0.0
        self.last_solve_time = -1.0
        self.total_distance = 0.0
        self.total_noise_impact = 0.0

        self.cplex_solver = CplexSolver(
            drones=drones_static,
            supply_indices=self.supply_indices,
            station_indices=self.station_indices,
            dist_matrix=dist_matrix,
            all_points=self.all_points,
            noise_cost_matrix=noise_cost_matrix,
            noise_weight=noise_weight,
            time_limit=time_limit,
        )

    def run(self, end_time: float):
        print(f"\n开始模拟，结束时间 {end_time} 小时")
        print("=" * 50)

        self.advance_to(end_time)
        self._print_summary()

    def advance_to(self, end_time: float):
        if end_time < self.current_time:
            raise ValueError(
                f"end_time={end_time:.3f}h 早于当前模拟时间 {self.current_time:.3f}h"
            )

        while self.current_time + self.time_step < end_time:
            self._process_new_demands()
            self._update_positions()
            self._check_arrivals()

            if self.current_time - self.last_solve_time >= self.solve_interval:
                self._solve_assignment()
                self.last_solve_time = self.current_time

            self.current_time += self.time_step

        if self.current_time < end_time:
            self.current_time = end_time
            self._process_new_demands()
            self._update_positions()
            self._check_arrivals()

            if self.current_time - self.last_solve_time >= self.solve_interval:
                self._solve_assignment()
                self.last_solve_time = self.current_time

    def is_complete(self) -> bool:
        has_pending_events = bool(self.event_queue)
        has_unfinished_demands = any(d is not None for d in self.unserved_demands)
        has_active_drones = any(
            ds.status != DroneStatus.IDLE or ds.task_queue for ds in self.drone_states
        )
        return not has_pending_events and not has_unfinished_demands and not has_active_drones

    def run_until_complete(self, max_time: float):
        while self.current_time < max_time and not self.is_complete():
            next_time = min(self.current_time + self.solve_interval, max_time)
            self.advance_to(next_time)

        if not self.is_complete():
            print(f"警告: 模拟在 {self.current_time:.3f}h 达到上限，仍有未完成任务")

    def snapshot_state(self) -> Dict[str, object]:
        return serialize_simulator_snapshot(
            current_time=self.current_time,
            all_demand_events=self.all_demand_events,
            unserved_demands=self.unserved_demands,
            drone_states=self.drone_states,
            total_distance=self.total_distance,
            total_noise_impact=self.total_noise_impact,
        )

    def _update_positions(self):
        for ds in self.drone_states:
            if ds.status != DroneStatus.IDLE and ds.target_node is not None:
                target = self.all_points[ds.target_node]
                dx = target.x - ds.position_x
                dy = target.y - ds.position_y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > 0:
                    speed_per_step = self.drones_static[0].speed * self.time_step * 3600
                    if dist <= speed_per_step:
                        ds.position_x = target.x
                        ds.position_y = target.y
                    else:
                        ds.position_x += dx * speed_per_step / dist
                        ds.position_y += dy * speed_per_step / dist

    def _check_arrivals(self):
        for ds in self.drone_states:
            if ds.status != DroneStatus.IDLE and ds.target_node is not None:
                target = self.all_points[ds.target_node]
                if abs(ds.position_x - target.x) < 1 and abs(ds.position_y - target.y) < 1:
                    self._handle_arrival(ds)

    def _handle_arrival(self, ds: DroneState):
        arrived_node = ds.target_node
        ds.position_x = self.all_points[arrived_node].x
        ds.position_y = self.all_points[arrived_node].y
        ds.current_node = arrived_node
        ds.executed_path.append(arrived_node)

        node_name = self.all_points[arrived_node].id

        if ds.status == DroneStatus.TO_SUPPLY:
            print(f"[{self.current_time:.3f}] 无人机 {ds.drone_id} 到达供给点 {node_name}")

            if ds.task_queue:
                current_task = ds.task_queue[0]
                if current_task["type"] == "delivery":
                    demand_node = current_task["demand_node"]
                    dist = self.dist_matrix[arrived_node, demand_node]
                    ds.remaining_range -= dist
                    self.total_distance += dist

                    if self.cplex_solver.noise_cost_matrix is not None:
                        noise = self.cplex_solver.noise_cost_matrix[arrived_node, demand_node]
                        self.total_noise_impact += noise

                    ds.status = DroneStatus.TO_DEMAND
                    ds.target_node = demand_node
                    ds.assigned_demand_id = current_task["demand_id"]
                    ds.assigned_demand_node = demand_node
                    ds.assigned_demand_weight = current_task["weight"]

                    demand_name = self.all_points[demand_node].id
                    print(f"    前往需求点 {demand_name} 送货")
                else:
                    print("    错误: 任务队列中的任务类型错误")
                    self._return_to_station(ds)
            else:
                print("    错误: 到达供给点但任务队列为空")
                self._return_to_station(ds)

        elif ds.status == DroneStatus.TO_DEMAND:
            print(f"[{self.current_time:.3f}] 无人机 {ds.drone_id} 到达需求点 {node_name}")

            if ds.task_queue and ds.task_queue[0]["type"] == "delivery":
                current_task = ds.task_queue.pop(0)
                demand_id = current_task["demand_id"]

                demand = self.unserved_demands_dict.get(demand_id)
                if demand and demand.node_idx == arrived_node:
                    ds.remaining_payload -= demand.weight
                    demand.served_time = self.current_time
                    self.completed_demands.append(demand)

                    for i, d in enumerate(self.unserved_demands):
                        if d is not None and d.unique_id == demand_id:
                            self.unserved_demands[i] = None
                            break

                    if demand_id in self.unserved_demands_dict:
                        del self.unserved_demands_dict[demand_id]

                    print(
                        f"    送达需求 {demand.unique_id} ({demand.demand_point_id}), 剩余载重 {ds.remaining_payload:.1f}kg"
                    )

                    if ds.task_queue:
                        print(f"    还有 {len(ds.task_queue)} 个任务等待执行")
                        next_task = ds.task_queue[0]
                        if next_task["type"] == "delivery":
                            supply_node = next_task["supply_node"]
                            dist = self.dist_matrix[ds.current_node, supply_node]
                            ds.remaining_range -= dist
                            self.total_distance += dist

                            if self.cplex_solver.noise_cost_matrix is not None:
                                noise = self.cplex_solver.noise_cost_matrix[ds.current_node, supply_node]
                                self.total_noise_impact += noise

                            ds.status = DroneStatus.TO_SUPPLY
                            ds.target_node = supply_node
                            ds.assigned_supply_node = supply_node

                            supply_name = self.all_points[supply_node].id
                            demand_name = self.all_points[next_task["demand_node"]].id
                            print(f"    开始执行下一个任务: 前往 {supply_name} 取货，送 {demand_name}")
                    else:
                        self._return_to_station(ds)
                else:
                    print(f"    错误: 找不到匹配的需求 {demand_id}")
                    self._return_to_station(ds)
            else:
                print("    错误: 到达需求点但任务队列为空")
                self._return_to_station(ds)

        elif ds.status == DroneStatus.RETURNING:
            print(f"[{self.current_time:.3f}] 无人机 {ds.drone_id} 到达站点")

            if ds.task_queue:
                print(f"    还有 {len(ds.task_queue)} 个任务，准备执行下一个任务")
                next_task = ds.task_queue[0]
                if next_task["type"] == "delivery":
                    supply_node = next_task["supply_node"]
                    dist = self.dist_matrix[ds.current_node, supply_node]
                    ds.remaining_range -= dist
                    self.total_distance += dist

                    if self.cplex_solver.noise_cost_matrix is not None:
                        noise = self.cplex_solver.noise_cost_matrix[ds.current_node, supply_node]
                        self.total_noise_impact += noise

                    ds.status = DroneStatus.TO_SUPPLY
                    ds.target_node = supply_node
                    ds.assigned_supply_node = supply_node

                    supply_name = self.all_points[supply_node].id
                    demand_name = self.all_points[next_task["demand_node"]].id
                    print(f"    前往供给点 {supply_name} 取货 (为 {demand_name} 送货)")
                else:
                    print("    错误: 任务类型错误")
                    ds.status = DroneStatus.IDLE
                    ds.target_node = None
            else:
                ds.status = DroneStatus.CHARGING
                ds.target_node = None
                ds.status = DroneStatus.IDLE
                ds.remaining_range = self.drones_static[0].max_range
                ds.remaining_payload = self.drones_static[0].max_payload
                ds.assigned_demand_id = None
                ds.assigned_demand_node = None
                ds.assigned_demand_weight = None
                ds.assigned_supply_node = None
                print("    充电完成，变为空闲")

    def _return_to_station(self, ds: DroneState):
        station_node = self.station_indices[ds.station_id]
        dist = self.dist_matrix[ds.current_node, station_node]
        ds.remaining_range -= dist
        self.total_distance += dist

        if self.cplex_solver.noise_cost_matrix is not None:
            noise = self.cplex_solver.noise_cost_matrix[ds.current_node, station_node]
            self.total_noise_impact += noise

        ds.status = DroneStatus.RETURNING
        ds.target_node = station_node

        station_name = self.all_points[station_node].id
        print(f"    返回站点 {station_name}, 距离 {dist:.0f}m")

    def _process_new_demands(self):
        while self.event_queue and self.event_queue[0][0] <= self.current_time:
            _, ev = heapq.heappop(self.event_queue)
            self.unserved_demands.append(ev)
            self.unserved_demands_dict[ev.unique_id] = ev
            point = self.all_points[ev.node_idx]
            print(f"[{self.current_time:.3f}] 新需求 {ev.unique_id} ({point.id}, {ev.weight}kg, 优先级{ev.priority})")

    def _solve_assignment(self):
        idle_drones = [ds for ds in self.drone_states if ds.status == DroneStatus.IDLE]
        pending = [d for d in self.unserved_demands if d is not None and d.assigned_drone is None]

        if not idle_drones or not pending:
            return

        print(f"\n[{self.current_time:.3f}] 调用CPLEX求解器")
        print(f"  空闲无人机: {len(idle_drones)}, 待处理需求: {len(pending)}")

        assignments = self.cplex_solver.solve_assignment(idle_drones, pending, self.current_time)

        for assign in assignments:
            drone = assign["drone"]
            demand = assign["demand"]
            supply_node = assign["supply_node"]

            demand.assigned_drone = drone.drone_id
            demand.assigned_time = self.current_time
            demand.supply_node = supply_node

            task = {
                "type": "delivery",
                "demand_id": demand.unique_id,
                "demand_node": demand.node_idx,
                "supply_node": supply_node,
                "weight": demand.weight,
                "priority": demand.priority,
            }
            drone.task_queue.append(task)

            if drone.status == DroneStatus.IDLE and len(drone.task_queue) == 1:
                dist = self.dist_matrix[drone.current_node, supply_node]
                drone.remaining_range -= dist
                self.total_distance += dist

                if self.cplex_solver.noise_cost_matrix is not None:
                    noise = self.cplex_solver.noise_cost_matrix[drone.current_node, supply_node]
                    self.total_noise_impact += noise

                drone.status = DroneStatus.TO_SUPPLY
                drone.target_node = supply_node
                drone.assigned_supply_node = supply_node

                supply_name = self.all_points[supply_node].id
                demand_name = self.all_points[demand.node_idx].id
                print(
                    f"  → 无人机 {drone.drone_id} 开始执行任务 {len(drone.task_queue)}: "
                    f"去 {supply_name} 取货，送 {demand_name} ({demand.unique_id}, 优先级{demand.priority})"
                )
            else:
                supply_name = self.all_points[supply_node].id
                demand_name = self.all_points[demand.node_idx].id
                print(
                    f"  → 无人机 {drone.drone_id} 新增任务到队列 (队列长度 {len(drone.task_queue)}): "
                    f"{supply_name}→{demand_name} ({demand.unique_id}, 优先级{demand.priority})"
                )

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("模拟结果汇总")
        print("=" * 70)

        completed = len(self.completed_demands)
        total = len(self.all_demand_events)
        print(f"总需求: {total}")
        print(f"完成: {completed}")
        if total > 0:
            print(f"完成率: {completed / total * 100:.1f}%")
        print(f"总飞行距离: {self.total_distance / 1000:.2f} km")
        print(f"总噪声影响: {self.total_noise_impact:.2f} (受影响建筑数)")

        print("\n需求明细:")
        print("-" * 80)
        demand_by_point = {}
        for ev in self.all_demand_events:
            point_id = ev.demand_point_id
            if point_id not in demand_by_point:
                demand_by_point[point_id] = []
            demand_by_point[point_id].append(ev)

        for point_id in sorted(demand_by_point.keys()):
            print(f"\n{point_id}:")
            for ev in demand_by_point[point_id]:
                status = "✓" if ev.served_time else "✗"
                drone = f" (由{ev.assigned_drone})" if ev.assigned_drone else ""
                time_info = f" 送达:{ev.served_time:.3f}" if ev.served_time else ""
                supply_name = f"S{ev.required_supply_idx + 1}"
                print(
                    f"  {ev.unique_id}: {status} {ev.weight}kg 优先级{ev.priority} "
                    f"(需从{supply_name}取货){drone}{time_info}"
                )

        print("\n无人机路径:")
        print("-" * 80)
        for ds in self.drone_states:
            if len(ds.executed_path) > 1:
                path_str = " -> ".join([self.all_points[n].id for n in ds.executed_path])
                print(f"  {ds.drone_id}: {path_str}")
                print(f"    总移动节点数: {len(ds.executed_path)}")
                if ds.task_queue:
                    print(f"    剩余任务队列: {len(ds.task_queue)} 个")
            else:
                print(f"  {ds.drone_id}: 没有移动")
