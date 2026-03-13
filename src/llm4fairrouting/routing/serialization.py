"""Serialization helpers for routing simulation state."""

from __future__ import annotations

from typing import Iterable

from llm4fairrouting.routing.domain import DemandEvent, DroneState, DroneStatus


def serialize_simulator_snapshot(
    *,
    current_time: float,
    all_demand_events: Iterable[DemandEvent],
    unserved_demands: Iterable[DemandEvent | None],
    drone_states: Iterable[DroneState],
    total_distance: float,
    total_noise_impact: float,
) -> dict[str, object]:
    served_ids = [
        ev.unique_id
        for ev in all_demand_events
        if ev.served_time is not None and ev.served_time <= current_time
    ]
    assigned_ids = [
        ev.unique_id
        for ev in all_demand_events
        if ev.assigned_time is not None and ev.assigned_time <= current_time
    ]
    pending_ids = [
        ev.unique_id
        for ev in unserved_demands
        if ev is not None
    ]
    busy_drones = [
        ds.drone_id
        for ds in drone_states
        if ds.status != DroneStatus.IDLE or ds.task_queue
    ]
    return {
        "current_time_h": round(current_time, 6),
        "served_ids": served_ids,
        "assigned_ids": assigned_ids,
        "pending_ids": pending_ids,
        "busy_drones": busy_drones,
        "total_distance_m": float(total_distance),
        "total_noise_impact": float(total_noise_impact),
    }
