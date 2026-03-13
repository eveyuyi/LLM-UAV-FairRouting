"""Routing domain objects shared across workflow and baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


@dataclass
class Point:
    id: str
    lon: float
    lat: float
    alt: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    type: str = ""

    def to_enu(self, ref_lat, ref_lon, ref_alt):
        lat_scale = 111000.0
        lon_scale = 111000.0 * math.cos(math.radians(ref_lat))
        self.x = (self.lon - ref_lon) * lon_scale
        self.y = (self.lat - ref_lat) * lat_scale
        self.z = self.alt - ref_alt


@dataclass
class Drone:
    id: str
    station_id: int
    station_name: str
    max_payload: float
    max_range: float
    speed: float


class DroneStatus(Enum):
    IDLE = "idle"
    TO_SUPPLY = "to_supply"
    TO_DEMAND = "to_demand"
    RETURNING = "returning"
    CHARGING = "charging"


@dataclass
class DroneState:
    drone_id: str
    station_id: int
    current_node: int
    remaining_range: float
    remaining_payload: float
    status: DroneStatus = DroneStatus.IDLE
    target_node: Optional[int] = None
    arrival_time: float = 0.0
    executed_path: List[int] = field(default_factory=list)
    position_x: float = 0.0
    position_y: float = 0.0
    assigned_demand_id: Optional[str] = None
    assigned_demand_node: Optional[int] = None
    assigned_demand_weight: Optional[float] = None
    assigned_supply_node: Optional[int] = None
    task_queue: List[Dict] = field(default_factory=list)


@dataclass
class DemandEvent:
    time: float
    node_idx: int
    weight: float
    unique_id: str
    priority: int
    assigned_drone: Optional[str] = None
    assigned_time: Optional[float] = None
    served_time: Optional[float] = None
    supply_node: Optional[int] = None
    required_supply_idx: Optional[int] = None
    demand_point_id: str = ""

    def __lt__(self, other):
        if not isinstance(other, DemandEvent):
            return NotImplemented
        return self.priority < other.priority


def priority_service_score(priority: int, max_priority: int) -> int:
    """Map ``priority=1`` (highest) to the largest service score."""
    normalized = max(1, int(priority))
    bounded = min(normalized, max_priority)
    return max_priority + 1 - bounded


def create_drones(
    station_points: List[Point],
    drones_per_station: int = 3,
    max_payload: float = 50.0,
    max_range: float = 150000.0,
    speed: float = 60.0,
) -> List[Drone]:
    drones = []
    for s_idx, station in enumerate(station_points):
        for d_idx in range(drones_per_station):
            drone = Drone(
                id=f"U{s_idx + 1}{d_idx + 1}",
                station_id=s_idx,
                station_name=station.id,
                max_payload=max_payload,
                max_range=max_range,
                speed=speed,
            )
            drones.append(drone)

    return drones
