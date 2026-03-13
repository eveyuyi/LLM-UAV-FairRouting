"""Path planning and path-based cost calculations."""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from llm4fairrouting.routing.domain import Point

NOISE_SOURCE_LEVEL = 75.0
NOISE_THRESHOLD = 45.0
ATMOSPHERIC_ABSORPTION = 0.001
FLIGHT_HEIGHT = 60.0
DEBUG_NOISE = False


class NoiseCalculator:
    """噪音计算器（优化版）"""

    @staticmethod
    def spherical_spreading_loss(distance):
        if distance <= 0:
            return 0
        return 18 * math.log10(distance) if distance > 1 else 0

    @staticmethod
    def atmospheric_absorption_loss(distance):
        return ATMOSPHERIC_ABSORPTION * distance

    @staticmethod
    def ground_effect_loss(distance, height, ground_type="urban"):
        absorption = {"water": 0.05, "grass": 0.15, "urban": 0.3, "other": 0.2}.get(ground_type, 0.2)
        height_factor = max(0, 1 - height / 150)
        return absorption * distance * height_factor

    @staticmethod
    def height_noise_reduction(height):
        if height >= 120:
            return 8
        if height >= 90:
            return 6
        if height >= 60:
            return 4
        if height >= 30:
            return 2
        return 0

    @staticmethod
    def calculate_noise_level(source_level, distance_3d, flight_height, ground_type="urban"):
        if distance_3d <= 0:
            return source_level

        spreading_loss = NoiseCalculator.spherical_spreading_loss(distance_3d)
        atmospheric_loss = NoiseCalculator.atmospheric_absorption_loss(distance_3d)
        ground_loss = NoiseCalculator.ground_effect_loss(distance_3d, flight_height, ground_type)
        height_reduction = NoiseCalculator.height_noise_reduction(flight_height)

        noise_level = source_level - spreading_loss - atmospheric_loss - ground_loss - height_reduction
        return max(0, noise_level)


class FastRRTPlanner:
    """
    三维RRT路径规划器，使用KD树和空间网格加速，障碍物为球体。
    """

    def __init__(
        self,
        obstacles: List[Tuple[np.ndarray, float]],
        bounds: Tuple[float, float, float, float, float, float],
        step_size: float = 30.0,
        max_iter: int = 50000,
        goal_bias: float = 0.1,
        grid_cell_size: float = 100.0,
        infeasible_penalty_factor: float = 2.0,
    ):
        self.obstacles = obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_bias = goal_bias
        self.infeasible_penalty_factor = infeasible_penalty_factor

        self.grid = {}
        self.cell_size = grid_cell_size
        for center, radius in obstacles:
            ix = int(center[0] // grid_cell_size)
            iy = int(center[1] // grid_cell_size)
            iz = int(center[2] // grid_cell_size)
            key = (ix, iy, iz)
            if key not in self.grid:
                self.grid[key] = []
            self.grid[key].append((center, radius))

    def _get_nearby_obstacles(self, a: np.ndarray, b: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        min_x = min(a[0], b[0]) - self.step_size
        max_x = max(a[0], b[0]) + self.step_size
        min_y = min(a[1], b[1]) - self.step_size
        max_y = max(a[1], b[1]) + self.step_size
        min_z = min(a[2], b[2]) - self.step_size
        max_z = max(a[2], b[2]) + self.step_size

        start_ix = int(min_x // self.cell_size)
        end_ix = int(max_x // self.cell_size)
        start_iy = int(min_y // self.cell_size)
        end_iy = int(max_y // self.cell_size)
        start_iz = int(min_z // self.cell_size)
        end_iz = int(max_z // self.cell_size)

        nearby = []
        for ix in range(start_ix, end_ix + 1):
            for iy in range(start_iy, end_iy + 1):
                for iz in range(start_iz, end_iz + 1):
                    key = (ix, iy, iz)
                    if key in self.grid:
                        nearby.extend(self.grid[key])
        return nearby

    def _collision_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        nearby_obs = self._get_nearby_obstacles(a, b)
        for center, radius in nearby_obs:
            ab = b - a
            t = np.dot(center - a, ab) / np.dot(ab, ab) if np.dot(ab, ab) != 0 else 0.5
            t = max(0.0, min(1.0, t))
            closest = a + t * ab
            dist = np.linalg.norm(closest - center)
            if dist < radius:
                return False
        return True

    def _random_sample(self) -> np.ndarray:
        x = random.uniform(self.bounds[0], self.bounds[1])
        y = random.uniform(self.bounds[2], self.bounds[3])
        z = random.uniform(self.bounds[4], self.bounds[5])
        return np.array([x, y, z])

    def plan(self, start: np.ndarray, goal: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        if self._collision_free(start, goal):
            return np.linalg.norm(goal - start), [start, goal]

        nodes = [start]
        parents = [-1]
        costs = [0.0]
        tree = cKDTree([start])

        for _ in range(self.max_iter):
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = self._random_sample()

            dist, idx = tree.query(sample, k=1)
            nearest = nodes[idx]

            direction = sample - nearest
            dist_to_sample = np.linalg.norm(direction)
            if dist_to_sample < 1e-6:
                continue
            new_pos = nearest + (direction / dist_to_sample) * min(self.step_size, dist_to_sample)

            if not self._collision_free(nearest, new_pos):
                continue

            new_idx = len(nodes)
            nodes.append(new_pos)
            parents.append(idx)
            costs.append(costs[idx] + np.linalg.norm(new_pos - nearest))
            tree = cKDTree(nodes)

            if np.linalg.norm(new_pos - goal) <= self.step_size:
                if self._collision_free(new_pos, goal):
                    path = [goal]
                    cur_idx = new_idx
                    while cur_idx != -1:
                        path.append(nodes[cur_idx])
                        cur_idx = parents[cur_idx]
                    path.reverse()
                    total_cost = costs[new_idx] + np.linalg.norm(new_pos - goal)
                    return total_cost, path

        straight = np.linalg.norm(goal - start)
        print(f"  RRT警告: 未找到路径，使用直线距离×{self.infeasible_penalty_factor} 替代")
        return straight * self.infeasible_penalty_factor, [start, goal]


def compute_path_noise_impact(
    path: List[np.ndarray],
    residential_positions: np.ndarray,
    residential_tree: cKDTree,
    source_level: float,
    threshold: float,
    flight_height: float,
    ground_type: str = "urban",
    search_radius: float = 400.0,
) -> int:
    affected = set()
    total_checks = 0

    for point in path:
        indices = residential_tree.query_ball_point(point, search_radius)
        total_checks += len(indices)

        for idx in indices:
            if idx in affected:
                continue

            bx, by, bz = residential_positions[idx]
            d = np.linalg.norm(point - np.array([bx, by, bz]))
            noise = NoiseCalculator.calculate_noise_level(
                source_level, d, flight_height, ground_type
            )

            if noise > threshold:
                affected.add(idx)

    if DEBUG_NOISE and len(path) > 0:
        print(f"    路径点{len(path)}个, 检查建筑{total_checks}次, 发现{len(affected)}个受影响建筑")

    return len(affected)


def create_obstacles_from_buildings(
    building_df: pd.DataFrame,
    selected_coords: set,
    min_obstacle_height: float = 30.0,
    obstacle_radius: float = 20.0,
) -> List[dict]:
    obstacles = []
    for _, row in building_df.iterrows():
        lon, lat = float(row["longitude"]), float(row["latitude"])
        if (lon, lat) in selected_coords:
            continue
        height = float(row["building_height_m"])
        if height <= min_obstacle_height:
            continue
        dem = float(row["ground_elevation_m"])
        obstacles.append({
            "lon": lon,
            "lat": lat,
            "alt": dem + height / 2.0,
            "radius": obstacle_radius,
        })
    return obstacles


def build_realistic_distance_and_noise_matrices(
    task_points: List[Point],
    obstacles_raw: List[dict],
    residential_positions: np.ndarray,
    residential_tree: cKDTree,
    ref_lat: float,
    ref_lon: float,
    flight_height: float,
    obstacle_radius: float = 20.0,
    rrt_step_size: float = 30.0,
    rrt_max_iter: int = 30000,
    rrt_goal_bias: float = 0.15,
    grid_cell_size: float = 100.0,
    noise_threshold: float = NOISE_THRESHOLD,
    search_radius: float = 400.0,
) -> Tuple[np.ndarray, np.ndarray]:
    print("\n" + "=" * 60)
    print("开始构建真实距离矩阵和噪声矩阵")
    print("=" * 60)

    n = len(task_points)
    point_ids = [p.id for p in task_points]
    print(f"任务点: {point_ids}")

    obstacles = []
    for obs in obstacles_raw:
        p = Point(id="", lon=obs["lon"], lat=obs["lat"], alt=obs["alt"])
        p.to_enu(ref_lat, ref_lon, ref_alt=0.0)
        center = np.array([p.x, p.y, p.z])
        obstacles.append((center, obs["radius"]))

    print(f"障碍物数量: {len(obstacles)}")

    all_x = [p.x for p in task_points] + [c[0] for c, _ in obstacles]
    all_y = [p.y for p in task_points] + [c[1] for c, _ in obstacles]
    all_z = [p.z for p in task_points] + [c[2] for c, _ in obstacles]
    margin = 200.0
    bounds = (
        min(all_x) - margin, max(all_x) + margin,
        min(all_y) - margin, max(all_y) + margin,
        min(all_z) - margin, max(all_z) + margin,
    )
    print(
        f"空间边界: X[{bounds[0]:.0f}, {bounds[1]:.0f}], "
        f"Y[{bounds[2]:.0f}, {bounds[3]:.0f}], "
        f"Z[{bounds[4]:.0f}, {bounds[5]:.0f}]"
    )

    planner = FastRRTPlanner(
        obstacles=obstacles,
        bounds=bounds,
        step_size=rrt_step_size,
        max_iter=rrt_max_iter,
        goal_bias=rrt_goal_bias,
        grid_cell_size=grid_cell_size,
    )

    positions = [np.array([p.x, p.y, p.z]) for p in task_points]
    dist_matrix = np.zeros((n, n))
    noise_matrix = np.zeros((n, n), dtype=int)

    total_pairs = n * (n - 1) // 2
    current_pair = 0

    for i in range(n):
        for j in range(i + 1, n):
            current_pair += 1
            print(f"\n[{current_pair}/{total_pairs}] 计算路径 {task_points[i].id} -> {task_points[j].id} ...")

            length, path = planner.plan(positions[i], positions[j])
            dist_matrix[i, j] = length
            dist_matrix[j, i] = length
            print(f"  路径长度: {length:.1f}米, 路径点数: {len(path)}")

            num_affected = compute_path_noise_impact(
                path=path,
                residential_positions=residential_positions,
                residential_tree=residential_tree,
                source_level=NOISE_SOURCE_LEVEL,
                threshold=noise_threshold,
                flight_height=flight_height,
                ground_type="urban",
                search_radius=search_radius,
            )

            noise_matrix[i, j] = num_affected
            noise_matrix[j, i] = num_affected
            print(f"  完成: 影响建筑 {num_affected} 个")

    print("\n矩阵构建完成")
    print(f"距离矩阵范围: [{np.min(dist_matrix[dist_matrix > 0]):.0f}, {np.max(dist_matrix):.0f}]米")
    print(f"噪声矩阵范围: [{np.min(noise_matrix)}, {np.max(noise_matrix)}]个建筑")
    print(f"非零噪声路径数: {np.sum(noise_matrix > 0) // 2}条")

    return dist_matrix, noise_matrix


def calculate_distance_matrix(points: List[Point]) -> np.ndarray:
    n = len(points)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i].x - points[j].x
            dy = points[i].y - points[j].y
            dz = points[i].z - points[j].z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix
