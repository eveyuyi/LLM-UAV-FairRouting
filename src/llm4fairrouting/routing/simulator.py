"""Dynamic routing simulator shared across workflow and baselines."""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional

import numpy as np

from llm4fairrouting.routing.analytics import resolve_objective_weights
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
        drone_activation_cost: float = 10000.0,
        time_step: float = 0.001,
        solve_interval: float = 0.05,
        time_limit: int = 10,
        objective_weight_schedule: Optional[List[Dict[str, object]]] = None,
        analytics_output_dir: Optional[str] = None,
        enable_conflict_refiner: bool = False,
    ):
        self.supply_points = supply_points
        self.demand_points = demand_points
        self.station_points = station_points
        self.drones_static = drones_static
        self.drone_specs = {drone.id: drone for drone in drones_static}
        self.dist_matrix = dist_matrix
        self.all_demand_events = demand_events
        self.time_step = time_step
        self.solve_interval = solve_interval
        self.objective_weight_schedule = list(objective_weight_schedule or [])
        self.analytics_output_dir = analytics_output_dir
        self.enable_conflict_refiner = enable_conflict_refiner

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
                current_load=0.0,
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
        self.solver_calls: List[Dict[str, object]] = []
        self.gantt_tasks: List[Dict[str, object]] = []
        self.active_legs: Dict[str, Dict[str, object]] = {}

        self.cplex_solver = CplexSolver(
            drones=drones_static,
            supply_indices=self.supply_indices,
            station_indices=self.station_indices,
            dist_matrix=dist_matrix,
            all_points=self.all_points,
            noise_cost_matrix=noise_cost_matrix,
            noise_weight=noise_weight,
            drone_activation_cost=drone_activation_cost,
            time_limit=time_limit,
            analytics_output_dir=analytics_output_dir,
            enable_conflict_refiner=enable_conflict_refiner,
        )

    def run(self, end_time: float):
        print(f"\n开始模拟，结束时间 {end_time} 小时")
        print("=" * 50)

        self.advance_to(end_time)
        self.print_summary()

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

    def get_drone_path_details(self) -> List[Dict[str, object]]:
        """Return the full executed path for each drone in visit order."""
        details: List[Dict[str, object]] = []
        for ds in self.drone_states:
            path_node_indices = [int(node_idx) for node_idx in ds.executed_path]
            path_node_ids = [self.all_points[node_idx].id for node_idx in path_node_indices]
            details.append(
                {
                    "drone_id": ds.drone_id,
                    "station_id": ds.station_id,
                    "station_name": self.all_points[self.station_indices[ds.station_id]].id,
                    "current_node_index": int(ds.current_node),
                    "current_node_id": self.all_points[ds.current_node].id,
                    "final_status": ds.status.value,
                    "remaining_range_m": round(float(ds.remaining_range), 3),
                    "remaining_payload_kg": round(float(ds.remaining_payload), 3),
                    "current_load_kg": round(float(ds.current_load), 3),
                    "pending_task_count": len(ds.task_queue),
                    "path_node_indices": path_node_indices,
                    "path_node_ids": path_node_ids,
                    "path_str": " -> ".join(path_node_ids),
                    "n_nodes_visited": len(path_node_ids),
                    "n_legs_completed": max(0, len(path_node_ids) - 1),
                }
            )
        return details

    def print_summary(self):
        self._print_summary()

    def snapshot_state(self) -> Dict[str, object]:
        return serialize_simulator_snapshot(
            current_time=self.current_time,
            all_demand_events=self.all_demand_events,
            unserved_demands=self.unserved_demands,
            drone_states=self.drone_states,
            total_distance=self.total_distance,
            total_noise_impact=self.total_noise_impact,
        )

    def get_solver_analytics(self) -> Dict[str, object]:
        served_events = [
            event
            for event in self.all_demand_events
            if event.served_time is not None
        ]
        delivery_latencies = [
            max(0.0, float(event.served_time) - float(event.time))
            for event in served_events
        ]
        run_summary = {
            "total_demands": len(self.all_demand_events),
            "served_demands": len(served_events),
            "service_rate": round(
                len(served_events) / len(self.all_demand_events),
                6,
            ) if self.all_demand_events else 0.0,
            "final_total_distance_m": float(self.total_distance),
            "final_total_noise_impact": float(self.total_noise_impact),
            "average_delivery_time_h": round(
                sum(delivery_latencies) / len(delivery_latencies),
                6,
            ) if delivery_latencies else None,
            "max_delivery_time_h": round(max(delivery_latencies), 6) if delivery_latencies else None,
            "n_solver_calls": len(self.solver_calls),
            "n_conflict_reports": sum(
                1
                for call in self.solver_calls
                if call.get("conflict_refiner", {}).get("status")
                not in {None, "not_requested", "not_run"}
            ),
        }
        return {
            "run_summary": run_summary,
            "solver_calls": list(self.solver_calls),
            "gantt_tasks": list(self.gantt_tasks),
            "objective_weight_schedule": list(self.objective_weight_schedule),
        }

    def _active_window_context(self) -> Dict[str, object]:
        if not self.objective_weight_schedule:
            return {
                "time_window": None,
                "weights": resolve_objective_weights(None),
            }

        for entry in self.objective_weight_schedule:
            start_time_h = float(entry.get("start_time_h", 0.0))
            raw_end = entry.get("end_time_h")
            end_time_h = None if raw_end is None else float(raw_end)
            if self.current_time + 1e-9 < start_time_h:
                continue
            if end_time_h is None or self.current_time <= end_time_h + 1e-9:
                return {
                    "time_window": entry.get("time_window"),
                    "window_start_h": start_time_h,
                    "window_end_h": end_time_h,
                    "weights": resolve_objective_weights(entry.get("global_weights")),
                }

        last_entry = self.objective_weight_schedule[-1]
        return {
            "time_window": last_entry.get("time_window"),
            "window_start_h": float(last_entry.get("start_time_h", 0.0)),
            "window_end_h": last_entry.get("end_time_h"),
            "weights": resolve_objective_weights(last_entry.get("global_weights")),
        }

    def _finish_active_leg(self, ds: DroneState, arrived_node: int):
        leg = self.active_legs.pop(ds.drone_id, None)
        if leg is None:
            return
        leg["end_time_h"] = round(self.current_time, 6)
        leg["duration_h"] = round(
            max(0.0, float(leg["end_time_h"]) - float(leg["start_time_h"])),
            6,
        )
        leg["arrived_node_index"] = int(arrived_node)
        leg["arrived_node_id"] = self.all_points[arrived_node].id

    def _update_positions(self):
        for ds in self.drone_states:
            if ds.status != DroneStatus.IDLE and ds.target_node is not None:
                target = self.all_points[ds.target_node]
                dx = target.x - ds.position_x
                dy = target.y - ds.position_y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > 0:
                    drone_speed = self.drone_specs[ds.drone_id].speed
                    speed_per_step = drone_speed * self.time_step * 3600
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
        self._finish_active_leg(ds, arrived_node)

        node_name = self.all_points[arrived_node].id
        current_task = ds.task_queue.pop(0) if ds.task_queue else None
        if current_task is None:
            print(f"[{self.current_time:.3f}] 无人机 {ds.drone_id} 到达 {node_name}，但没有待执行停靠点")
            self._return_to_station(ds)
            return

        task_type = current_task["type"]
        drone_spec = self.drone_specs[ds.drone_id]

        if task_type == "pickup":
            print(f"[{self.current_time:.3f}] 无人机 {ds.drone_id} 到达供给点 {node_name}")
            ds.current_load += float(current_task.get("weight", 0.0))
            ds.remaining_payload = max(0.0, drone_spec.max_payload - ds.current_load)
            print(
                f"    完成取货 {current_task.get('demand_id')}，当前载货 {ds.current_load:.1f}kg，"
                f"剩余可用载重 {ds.remaining_payload:.1f}kg"
            )

        elif task_type == "delivery":
            print(f"[{self.current_time:.3f}] 无人机 {ds.drone_id} 到达需求点 {node_name}")
            demand_id = current_task["demand_id"]
            demand = self.unserved_demands_dict.get(demand_id)
            if demand and demand.node_idx == arrived_node:
                ds.current_load = max(0.0, ds.current_load - float(demand.weight))
                ds.remaining_payload = min(
                    drone_spec.max_payload,
                    drone_spec.max_payload - ds.current_load,
                )
                demand.served_time = self.current_time
                self.completed_demands.append(demand)

                for i, d in enumerate(self.unserved_demands):
                    if d is not None and d.unique_id == demand_id:
                        self.unserved_demands[i] = None
                        break

                if demand_id in self.unserved_demands_dict:
                    del self.unserved_demands_dict[demand_id]

                print(
                    f"    送达需求 {demand.unique_id} ({demand.demand_point_id})，"
                    f"当前载货 {ds.current_load:.1f}kg，剩余可用载重 {ds.remaining_payload:.1f}kg"
                )
            else:
                print(f"    错误: 找不到匹配的需求 {demand_id}")
                self._return_to_station(ds)
                return

        elif task_type == "station":
            print(f"[{self.current_time:.3f}] 无人机 {ds.drone_id} 到达站点")
            self._finish_route_at_station(ds)
            return

        else:
            print(f"    错误: 未知停靠点类型 {task_type}")
            self._return_to_station(ds)
            return

        if ds.task_queue:
            print(f"    还有 {len(ds.task_queue)} 个停靠点等待执行")
            self._dispatch_next_stop(ds)
        elif ds.current_node != self.station_indices[ds.station_id]:
            self._return_to_station(ds)
        else:
            self._finish_route_at_station(ds)

    def _apply_leg_cost(self, from_node: int, to_node: int):
        dist = self.dist_matrix[from_node, to_node]
        self.total_distance += dist

        if self.cplex_solver.noise_cost_matrix is not None:
            noise = self.cplex_solver.noise_cost_matrix[from_node, to_node]
            self.total_noise_impact += noise

        return dist

    def _dispatch_next_stop(self, ds: DroneState):
        if not ds.task_queue:
            if ds.current_node == self.station_indices[ds.station_id]:
                self._finish_route_at_station(ds)
            else:
                self._return_to_station(ds)
            return

        next_task = ds.task_queue[0]
        target_node = next_task["node"]
        dist = self._apply_leg_cost(ds.current_node, target_node)
        ds.remaining_range -= dist
        ds.target_node = target_node
        travel_time_h = dist / max(self.drone_specs[ds.drone_id].speed, 1e-6) / 3600.0
        task_type = str(next_task["type"])
        gantt_task_type = {
            "pickup": "pickup_leg",
            "delivery": "delivery_leg",
            "station": "return_leg",
        }.get(task_type, "delivery_leg")
        leg_record = {
            "drone_id": ds.drone_id,
            "task_type": gantt_task_type,
            "demand_id": next_task.get("demand_id"),
            "start_time_h": round(self.current_time, 6),
            "planned_end_time_h": round(self.current_time + travel_time_h, 6),
            "from_node_index": int(ds.current_node),
            "from_node_id": self.all_points[ds.current_node].id,
            "to_node_index": int(target_node),
            "to_node_id": self.all_points[target_node].id,
            "time_window": self._active_window_context().get("time_window"),
        }
        self.active_legs[ds.drone_id] = leg_record
        self.gantt_tasks.append(leg_record)

        if next_task["type"] == "pickup":
            ds.status = DroneStatus.TO_SUPPLY
            ds.assigned_supply_node = target_node
            ds.assigned_demand_id = next_task.get("demand_id")
            ds.assigned_demand_node = next_task.get("demand_node")
            ds.assigned_demand_weight = next_task.get("weight")
            demand_name = self.all_points[next_task["demand_node"]].id
            print(
                f"    前往供给点 {self.all_points[target_node].id} 取货，"
                f"对应需求 {demand_name} ({next_task.get('demand_id')})"
            )
        elif next_task["type"] == "delivery":
            ds.status = DroneStatus.TO_DEMAND
            ds.assigned_supply_node = next_task.get("supply_node")
            ds.assigned_demand_id = next_task.get("demand_id")
            ds.assigned_demand_node = target_node
            ds.assigned_demand_weight = next_task.get("weight")
            print(
                f"    前往需求点 {self.all_points[target_node].id} 送货 "
                f"({next_task.get('demand_id')})"
            )
        else:
            ds.status = DroneStatus.RETURNING
            ds.assigned_supply_node = None
            ds.assigned_demand_id = None
            ds.assigned_demand_node = None
            ds.assigned_demand_weight = None
            print(f"    返回站点 {self.all_points[target_node].id}，距离 {dist:.0f}m")

    def _finish_route_at_station(self, ds: DroneState):
        drone_spec = self.drone_specs[ds.drone_id]
        ds.status = DroneStatus.CHARGING
        ds.target_node = None
        ds.status = DroneStatus.IDLE
        ds.remaining_range = drone_spec.max_range
        ds.current_load = 0.0
        ds.remaining_payload = drone_spec.max_payload
        ds.assigned_demand_id = None
        ds.assigned_demand_node = None
        ds.assigned_demand_weight = None
        ds.assigned_supply_node = None
        print("    充电完成，变为空闲")

    def _return_to_station(self, ds: DroneState):
        station_node = self.station_indices[ds.station_id]
        if ds.task_queue and ds.task_queue[0].get("type") == "station":
            self._dispatch_next_stop(ds)
            return

        ds.task_queue.insert(0, {
            "type": "station",
            "node": station_node,
            "demand_id": None,
            "demand_node": None,
            "supply_node": None,
            "weight": 0.0,
            "priority": None,
            "demand_point_id": None,
        })
        self._dispatch_next_stop(ds)

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

        solve_context = self._active_window_context()
        assignments = self.cplex_solver.solve_assignment(
            idle_drones,
            pending,
            self.current_time,
            objective_weights=solve_context.get("weights"),
            solve_context=solve_context,
        )

        solve_details = dict(self.cplex_solver.last_solve_details or {})
        solve_details.setdefault("pending_demands", len(pending))
        solve_details.setdefault("idle_drones", len(idle_drones))
        solve_details.setdefault("time_window", solve_context.get("time_window"))

        assigned_demand_ids: List[str] = []
        for assign in assignments:
            drone = assign["drone"]
            served_demands = assign.get("served_demands", [])
            route_stops = list(assign.get("route_stops", []))
            if not served_demands or not route_stops:
                continue

            for served in served_demands:
                demand = served["demand"]
                demand.assigned_drone = drone.drone_id
                demand.assigned_time = self.current_time
                demand.supply_node = served.get("supply_node")
                assigned_demand_ids.append(demand.unique_id)

            drone.task_queue = route_stops

            print(
                f"  → 无人机 {drone.drone_id} 执行多任务路径: "
                f"{assign.get('path_str', ' -> '.join(assign.get('path_node_ids', [])))}"
            )
            print(
                f"    本次路径覆盖 {len(served_demands)} 个需求: "
                f"{', '.join(assign.get('served_demand_ids', []))}"
            )
            self._dispatch_next_stop(drone)

        solve_details["assigned_route_count"] = len(assignments)
        solve_details["assigned_demand_ids"] = assigned_demand_ids
        self.solver_calls.append(solve_details)

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
        for detail in self.get_drone_path_details():
            if detail["n_nodes_visited"] > 1:
                print(f"  {detail['drone_id']}: {detail['path_str']}")
                print(f"    总移动节点数: {detail['n_nodes_visited']}")
                print(f"    完成航段数: {detail['n_legs_completed']}")
                if detail["pending_task_count"]:
                    print(f"    剩余任务队列: {detail['pending_task_count']} 个")
            else:
                print(f"  {detail['drone_id']}: 没有移动")
