"""Baseline/demo script built on the shared routing core."""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from llm4fairrouting.data.building_information import load_building_partitions
from llm4fairrouting.data.demand_event_generation import (
    generate_demand_events as generate_shared_demand_events,
)
from llm4fairrouting.data.seed_paths import BUILDING_DATA_PATH, STATION_DATA_PATH
from llm4fairrouting.data.stations import load_station_data
from llm4fairrouting.routing.domain import DemandEvent, Point, create_drones
from llm4fairrouting.routing.path_costs import (
    FLIGHT_HEIGHT,
    NOISE_THRESHOLD,
    build_realistic_distance_and_noise_matrices,
    create_obstacles_from_buildings,
)


def create_points(
    hospitals: pd.DataFrame,
    residences: pd.DataFrame,
    stations_df: pd.DataFrame,
    num_supply: int = 2,
    num_demand: int = 4,
    num_stations: int = 1,
    flight_height: float = FLIGHT_HEIGHT,
) -> Tuple[List[Point], List[Point], List[Point]]:
    """Create the static demo points from the seed datasets."""
    supply_points = []
    selected_hospitals = hospitals.head(min(num_supply, len(hospitals)))
    for idx, (_, row) in enumerate(selected_hospitals.iterrows()):
        supply_points.append(Point(
            id=f"S{idx + 1}",
            lon=float(row["longitude"]),
            lat=float(row["latitude"]),
            alt=flight_height,
            type="supply",
        ))

    demand_points = []
    selected_residences = residences.head(min(num_demand, len(residences)))
    for idx, (_, row) in enumerate(selected_residences.iterrows()):
        demand_points.append(Point(
            id=f"D{idx + 1}",
            lon=float(row["longitude"]),
            lat=float(row["latitude"]),
            alt=flight_height,
            type="demand",
        ))

    station_points = []
    selected_stations = stations_df.head(min(num_stations, len(stations_df)))
    for idx, (_, row) in enumerate(selected_stations.iterrows()):
        station_points.append(Point(
            id=f"L{idx + 1}",
            lon=float(row["longitude"]),
            lat=float(row["latitude"]),
            alt=flight_height,
            type="station",
        ))

    return supply_points, demand_points, station_points


def generate_demand_events(
    demand_points: List[Point],
    supply_points: List[Point],
    num_events: int = 8,
    sim_duration: float = 2.0,
) -> List[DemandEvent]:
    """Generate the same randomized demo demand stream used by the original baseline."""
    return generate_shared_demand_events(
        demand_points=demand_points,
        supply_points=supply_points,
        num_events=num_events,
        sim_duration=sim_duration,
        verbose=True,
    )


def main():
    from llm4fairrouting.routing.simulator import FinalDroneSimulator

    print("=" * 70)
    print("无人机动态配送 - 基于RRT真实路径和噪声矩阵")
    print("=" * 70)

    random.seed(42)
    np.random.seed(42)

    try:
        print(f"读取建筑数据文件: {BUILDING_DATA_PATH}")
        hospitals, residences, all_buildings = load_building_partitions(str(BUILDING_DATA_PATH))
        print(f"医疗卫生用地数量: {len(hospitals)}")
        print(f"居住用地数量: {len(residences)}")
        print(f"建筑物总数: {len(all_buildings)}")

        print(f"读取站点数据文件: {STATION_DATA_PATH}")
        stations_df = load_station_data(str(STATION_DATA_PATH))
        print(f"站点数量: {len(stations_df)}")

        supply_points, demand_points, station_points = create_points(
            hospitals,
            residences,
            stations_df,
            num_supply=2,
            num_demand=4,
            num_stations=1,
            flight_height=FLIGHT_HEIGHT,
        )
        print(
            f"\n实际创建: {len(supply_points)}个供给点, "
            f"{len(demand_points)}个需求点, {len(station_points)}个站点"
        )
    except FileNotFoundError:
        print("使用测试数据...")
        supply_points = [
            Point(id="S1", lon=116.30, lat=39.90, alt=FLIGHT_HEIGHT, type="supply"),
            Point(id="S2", lon=116.31, lat=39.90, alt=FLIGHT_HEIGHT, type="supply"),
        ]
        demand_points = [
            Point(id="D1", lon=116.32, lat=39.90, alt=FLIGHT_HEIGHT, type="demand"),
            Point(id="D2", lon=116.30, lat=39.91, alt=FLIGHT_HEIGHT, type="demand"),
            Point(id="D3", lon=116.31, lat=39.91, alt=FLIGHT_HEIGHT, type="demand"),
            Point(id="D4", lon=116.32, lat=39.91, alt=FLIGHT_HEIGHT, type="demand"),
        ]
        station_points = [
            Point(id="L1", lon=116.30, lat=39.90, alt=FLIGHT_HEIGHT, type="station")
        ]
        residences = pd.DataFrame()
        all_buildings = None

    all_points = supply_points + demand_points + station_points
    ref_lat = np.mean([p.lat for p in all_points])
    ref_lon = np.mean([p.lon for p in all_points])
    for point in all_points:
        point.to_enu(ref_lat, ref_lon, 0)

    selected_coords = {(point.lon, point.lat) for point in all_points}
    obstacles_raw = []
    if all_buildings is not None:
        obstacles_raw = create_obstacles_from_buildings(
            all_buildings,
            selected_coords,
            min_obstacle_height=30.0,
            obstacle_radius=20.0,
        )
    else:
        for _ in range(5):
            obstacles_raw.append({
                "lon": 116.30 + random.uniform(-0.01, 0.03),
                "lat": 39.90 + random.uniform(-0.01, 0.03),
                "alt": 70.0 + random.uniform(-10, 10),
                "radius": 20.0,
            })

    residential_positions = []
    if not residences.empty:
        for _, row in residences.iterrows():
            residence = Point(
                id="",
                lon=float(row["longitude"]),
                lat=float(row["latitude"]),
                alt=float(row["ground_elevation_m"]),
            )
            residence.to_enu(ref_lat, ref_lon, 0)
            residential_positions.append([residence.x, residence.y, residence.z])
    else:
        for point in demand_points:
            residence = Point(id="", lon=point.lon, lat=point.lat, alt=0.0)
            residence.to_enu(ref_lat, ref_lon, 0)
            residential_positions.append([residence.x, residence.y, residence.z])

    residential_positions = np.array(residential_positions)
    residential_tree = cKDTree(residential_positions)
    print(f"居住建筑数量: {len(residential_positions)}")

    dist_matrix, noise_cost_matrix = build_realistic_distance_and_noise_matrices(
        task_points=all_points,
        obstacles_raw=obstacles_raw,
        residential_positions=residential_positions,
        residential_tree=residential_tree,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        flight_height=FLIGHT_HEIGHT,
        obstacle_radius=20.0,
        rrt_step_size=30.0,
        rrt_max_iter=30000,
        rrt_goal_bias=0.15,
        grid_cell_size=100.0,
        noise_threshold=NOISE_THRESHOLD,
        search_radius=400.0,
    )

    drones_static = create_drones(
        station_points,
        drones_per_station=3,
        max_payload=60.0,
        max_range=200000.0,
        speed=60.0,
    )
    demand_events = generate_demand_events(
        demand_points,
        supply_points,
        num_events=8,
        sim_duration=2.0,
    )

    sim = FinalDroneSimulator(
        supply_points=supply_points,
        demand_points=demand_points,
        station_points=station_points,
        drones_static=drones_static,
        dist_matrix=dist_matrix,
        demand_events=demand_events,
        noise_cost_matrix=noise_cost_matrix,
        noise_weight=0.5,
        time_step=0.001,
    )
    sim.run(end_time=4.0)


if __name__ == "__main__":
    main()
