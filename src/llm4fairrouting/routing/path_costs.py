"""Path planning and path-based cost calculations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

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

    def _random_sample(self, rng: random.Random) -> np.ndarray:
        x = rng.uniform(self.bounds[0], self.bounds[1])
        y = rng.uniform(self.bounds[2], self.bounds[3])
        z = rng.uniform(self.bounds[4], self.bounds[5])
        return np.array([x, y, z])

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        rng: random.Random | None = None,
    ) -> Tuple[float, List[np.ndarray]]:
        if rng is None:
            rng = random.Random()

        if self._collision_free(start, goal):
            return np.linalg.norm(goal - start), [start, goal]

        nodes = [start]
        parents = [-1]
        costs = [0.0]
        tree = cKDTree([start])

        for _ in range(self.max_iter):
            if rng.random() < self.goal_bias:
                sample = goal
            else:
                sample = self._random_sample(rng)

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


def _stable_pair_seed(a: Point, b: Point) -> int:
    pair_tokens = sorted([
        f"{a.id}|{a.lon:.8f}|{a.lat:.8f}|{a.alt:.3f}",
        f"{b.id}|{b.lon:.8f}|{b.lat:.8f}|{b.alt:.3f}",
    ])
    digest = hashlib.sha256("::".join(pair_tokens).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


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


@dataclass
class _PathPlanningContext:
    task_points: List[Point]
    positions: List[np.ndarray]
    planner: FastRRTPlanner
    residential_positions: np.ndarray
    residential_tree: cKDTree
    noise_threshold: float
    flight_height: float
    search_radius: float


def _prepare_path_planning_context(
    task_points: List[Point],
    obstacles_raw: List[dict],
    residential_positions: np.ndarray,
    residential_tree: cKDTree,
    ref_lat: float,
    ref_lon: float,
    flight_height: float,
    rrt_step_size: float,
    rrt_max_iter: int,
    rrt_goal_bias: float,
    grid_cell_size: float,
    noise_threshold: float,
    search_radius: float,
) -> _PathPlanningContext:
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

    return _PathPlanningContext(
        task_points=task_points,
        positions=positions,
        planner=planner,
        residential_positions=residential_positions,
        residential_tree=residential_tree,
        noise_threshold=noise_threshold,
        flight_height=flight_height,
        search_radius=search_radius,
    )


def _compute_pair_metrics(context: _PathPlanningContext, i: int, j: int) -> Tuple[float, int, List[np.ndarray]]:
    pair_rng = random.Random(
        _stable_pair_seed(context.task_points[i], context.task_points[j])
    )
    length, path = context.planner.plan(
        context.positions[i],
        context.positions[j],
        rng=pair_rng,
    )
    num_affected = compute_path_noise_impact(
        path=path,
        residential_positions=context.residential_positions,
        residential_tree=context.residential_tree,
        source_level=NOISE_SOURCE_LEVEL,
        threshold=context.noise_threshold,
        flight_height=context.flight_height,
        ground_type="urban",
        search_radius=context.search_radius,
    )
    return float(length), int(num_affected), path


class LazySymmetricMatrix:
    """Symmetric matrix view that computes pair costs the first time they are requested."""

    def __init__(self, cache: "LazyPathCostCache", metric: str):
        self._cache = cache
        self._metric = metric

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._cache.size, self._cache.size)

    @property
    def computed_pairs(self) -> int:
        return self._cache.computed_pairs

    def __getitem__(self, key: Tuple[int, int]) -> float | int:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("LazySymmetricMatrix expects a tuple index like matrix[i, j]")
        i, j = (int(key[0]), int(key[1]))
        distance, noise = self._cache.get_pair_cost(i, j)
        if self._metric == "distance":
            return distance
        if self._metric == "noise":
            return noise
        raise ValueError(f"Unsupported metric: {self._metric}")


class LazyPathCostCache:
    """Cache distance/noise pairs so the dynamic solver only plans paths it actually uses."""

    def __init__(self, context: _PathPlanningContext):
        self._context = context
        self._pair_costs: Dict[Tuple[int, int], Tuple[float, int]] = {}
        self._pair_paths: Dict[Tuple[int, int], Dict[str, object]] = {}
        self._compute_count = 0

    @property
    def size(self) -> int:
        return len(self._context.task_points)

    @property
    def computed_pairs(self) -> int:
        return len(self._pair_costs)

    def get_pair_cost(self, i: int, j: int) -> Tuple[float, int]:
        if i == j:
            return 0.0, 0

        key = (i, j) if i < j else (j, i)
        if key not in self._pair_costs:
            self._pair_costs[key] = self._compute_pair(*key)
        return self._pair_costs[key]

    def _compute_pair(self, i: int, j: int) -> Tuple[float, int]:
        self._compute_count += 1
        start = self._context.task_points[i]
        goal = self._context.task_points[j]
        print(f"\n[Path Cache {self._compute_count}] computing {start.id} -> {goal.id} ...")

        length, num_affected, path = _compute_pair_metrics(self._context, i, j)
        self._pair_paths[(i, j)] = {
            "from_index": i,
            "to_index": j,
            "from_id": self._context.task_points[i].id,
            "to_id": self._context.task_points[j].id,
            "path_xyz": [[round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)] for p in path],
            "n_waypoints": len(path),
            "path_length_m": round(float(length), 3),
            "noise_impact": int(num_affected),
        }
        print(f"  path length: {length:.1f} m | waypoints: {len(path)}")
        print(f"  affected buildings: {num_affected}")
        return length, num_affected

    def get_pair_path(self, i: int, j: int) -> Optional[Dict[str, object]]:
        if i == j:
            return None

        key = (i, j) if i < j else (j, i)
        if key not in self._pair_costs:
            self._pair_costs[key] = self._compute_pair(*key)
        payload = self._pair_paths.get(key)
        if payload is None:
            return None
        return dict(payload)


def export_rrt_paths_for_edges(
    distance_matrix: object,
    edges: Sequence[Tuple[int, int]],
    output_path: str | Path | None = None,
) -> List[Dict[str, object]]:
    if not isinstance(distance_matrix, LazySymmetricMatrix):
        return []

    exported: List[Dict[str, object]] = []
    seen: set[Tuple[int, int]] = set()
    for start_idx, end_idx in edges:
        key = (int(start_idx), int(end_idx))
        norm = key if key[0] < key[1] else (key[1], key[0])
        if norm in seen or norm[0] == norm[1]:
            continue
        seen.add(norm)
        payload = distance_matrix._cache.get_pair_path(*norm)
        if payload is None:
            continue
        exported.append(payload)

    exported.sort(key=lambda item: (str(item.get("from_id", "")), str(item.get("to_id", ""))))
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            import json
            json.dump(exported, handle, ensure_ascii=False, indent=2)
    return exported



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
    context = _prepare_path_planning_context(
        task_points=task_points,
        obstacles_raw=obstacles_raw,
        residential_positions=residential_positions,
        residential_tree=residential_tree,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        flight_height=flight_height,
        rrt_step_size=rrt_step_size,
        rrt_max_iter=rrt_max_iter,
        rrt_goal_bias=rrt_goal_bias,
        grid_cell_size=grid_cell_size,
        noise_threshold=noise_threshold,
        search_radius=search_radius,
    )
    dist_matrix = np.zeros((n, n))
    noise_matrix = np.zeros((n, n), dtype=int)

    total_pairs = n * (n - 1) // 2
    current_pair = 0

    for i in range(n):
        for j in range(i + 1, n):
            current_pair += 1
            print(f"\n[{current_pair}/{total_pairs}] 计算路径 {task_points[i].id} -> {task_points[j].id} ...")

            length, num_affected, path = _compute_pair_metrics(context, i, j)
            dist_matrix[i, j] = length
            dist_matrix[j, i] = length
            print(f"  path length: {length:.1f} m | waypoints: {len(path)}")
            noise_matrix[i, j] = num_affected
            noise_matrix[j, i] = num_affected
            print(f"  完成: 影响建筑 {num_affected} 个")

    print("\n矩阵构建完成")
    print(f"距离矩阵范围: [{np.min(dist_matrix[dist_matrix > 0]):.0f}, {np.max(dist_matrix):.0f}]米")
    print(f"噪声矩阵范围: [{np.min(noise_matrix)}, {np.max(noise_matrix)}]个建筑")
    print(f"非零噪声路径数: {np.sum(noise_matrix > 0) // 2}条")

    return dist_matrix, noise_matrix


def build_lazy_distance_and_noise_matrices(
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
) -> Tuple[LazySymmetricMatrix, LazySymmetricMatrix]:
    del obstacle_radius

    print("\n" + "=" * 60)
    print("初始化按需路径距离/噪声缓存")
    print("=" * 60)
    print(f"任务点数量: {len(task_points)}")

    context = _prepare_path_planning_context(
        task_points=task_points,
        obstacles_raw=obstacles_raw,
        residential_positions=residential_positions,
        residential_tree=residential_tree,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        flight_height=flight_height,
        rrt_step_size=rrt_step_size,
        rrt_max_iter=rrt_max_iter,
        rrt_goal_bias=rrt_goal_bias,
        grid_cell_size=grid_cell_size,
        noise_threshold=noise_threshold,
        search_radius=search_radius,
    )
    print("路径将在首次访问对应节点对时计算，并跨窗口复用缓存")

    cache = LazyPathCostCache(context)
    return LazySymmetricMatrix(cache, "distance"), LazySymmetricMatrix(cache, "noise")
