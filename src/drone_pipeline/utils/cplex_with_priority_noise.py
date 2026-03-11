import heapq
import math
import os
import random
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    SolverFactory, Binary, minimize, value, TerminationCondition
)

from drone_pipeline.seed_data import BUILDING_DATA_PATH, STATION_DATA_PATH
from drone_pipeline.utils.building_data import (
    HEALTHCARE_LAND_USE,
    RESIDENTIAL_LAND_USE,
    load_building_data,
)
from drone_pipeline.utils.station_data import load_station_data

# 尝试导入诊断工具（可选）
try:
    from pyomo.contrib.iis import write_iis, log_infeasible_constraints
    from pyomo.util.infeasible import log_infeasible_bounds

    HAS_DIAGNOSTICS = True
except ImportError:
    HAS_DIAGNOSTICS = False
    print("警告: pyomo.contrib 未安装，无法进行不可行性诊断")

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = [
    'SimHei',  # Windows 黑体
    'Microsoft YaHei',  # Windows 微软雅黑
    'STHeiti',  # Mac 华文黑体
    'Heiti SC',  # Mac 黑体-简
    'PingFang SC',  # Mac 苹方
    'Noto Sans CJK SC',  # Linux 思源黑体
    'WenQuanYi Zen Hei',  # Linux 文泉驿正黑
    'Arial Unicode MS',  # Mac Arial Unicode
    'DejaVu Sans'  # Linux DejaVu
]
plt.rcParams['axes.unicode_minus'] = False


# ========== 基础数据结构 ==========
@dataclass
class Point:
    id: str
    lon: float
    lat: float
    alt: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    type: str = ''

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
    TO_SUPPLY = "to_supply"  # 前往供给点取货
    TO_DEMAND = "to_demand"  # 前往需求点送货
    RETURNING = "returning"  # 返回站点
    CHARGING = "charging"  # 充电


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
    # 新增：任务队列，支持串行执行多个任务
    task_queue: List[Dict] = field(default_factory=list)


@dataclass
class DemandEvent:
    time: float
    node_idx: int
    weight: float
    unique_id: str
    priority: int
    assigned_drone: Optional[str] = None
    served_time: Optional[float] = None
    supply_node: Optional[int] = None
    # 新增：对应的供给点索引
    required_supply_idx: Optional[int] = None
    # 新增：需求点编号（用于区分同一个位置的不同需求）
    demand_point_id: str = ""

    def __lt__(self, other):
        """定义 DemandEvent 的比较方法，用于 heapq"""
        if not isinstance(other, DemandEvent):
            return NotImplemented
        return self.priority > other.priority


# ========== 噪声计算参数 ==========
NOISE_SOURCE_LEVEL = 75.0      # 无人机在10米处的噪音水平（dB）
NOISE_THRESHOLD = 45.0         # 噪声阈值
ATMOSPHERIC_ABSORPTION = 0.001 # 大气吸收系数
FLIGHT_HEIGHT = 60.0           # 无人机飞行高度（米）
DEBUG_NOISE = False

class NoiseCalculator:
    """噪音计算器（优化版）"""

    @staticmethod
    def spherical_spreading_loss(distance):
        """球面扩散损失"""
        if distance <= 0:
            return 0
        return 18 * math.log10(distance) if distance > 1 else 0

    @staticmethod
    def atmospheric_absorption_loss(distance):
        """大气吸收损失"""
        return ATMOSPHERIC_ABSORPTION * distance

    @staticmethod
    def ground_effect_loss(distance, height, ground_type='urban'):
        """地面效应损失"""
        absorption = {'water': 0.05, 'grass': 0.15, 'urban': 0.3, 'other': 0.2}.get(ground_type, 0.2)
        height_factor = max(0, 1 - height / 150)
        return absorption * distance * height_factor

    @staticmethod
    def height_noise_reduction(height):
        """高度衰减"""
        if height >= 120:
            return 8
        elif height >= 90:
            return 6
        elif height >= 60:
            return 4
        elif height >= 30:
            return 2
        else:
            return 0

    @staticmethod
    def calculate_noise_level(source_level, distance_3d, flight_height, ground_type='urban'):
        """计算噪声级"""
        if distance_3d <= 0:
            return source_level

        spreading_loss = NoiseCalculator.spherical_spreading_loss(distance_3d)
        atmospheric_loss = NoiseCalculator.atmospheric_absorption_loss(distance_3d)
        ground_loss = NoiseCalculator.ground_effect_loss(distance_3d, flight_height, ground_type)
        height_reduction = NoiseCalculator.height_noise_reduction(flight_height)

        noise_level = source_level - spreading_loss - atmospheric_loss - ground_loss - height_reduction
        return max(0, noise_level)


# ========== RRT路径规划器 ==========
class FastRRTPlanner:
    """
    三维RRT路径规划器，使用KD树和空间网格加速，障碍物为球体。
    """

    def __init__(
            self,
            obstacles: List[Tuple[np.ndarray, float]],  # (center, radius)
            bounds: Tuple[float, float, float, float, float, float],
            step_size: float = 30.0,
            max_iter: int = 50000,
            goal_bias: float = 0.1,
            grid_cell_size: float = 100.0,
            infeasible_penalty_factor: float = 2.0
    ):
        self.obstacles = obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_bias = goal_bias
        self.infeasible_penalty_factor = infeasible_penalty_factor

        # 构建空间网格
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
        """获取线段ab可能经过的网格中的障碍物"""
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
        """检查线段ab是否与任何障碍物碰撞"""
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
        """在边界内均匀随机采样"""
        x = random.uniform(self.bounds[0], self.bounds[1])
        y = random.uniform(self.bounds[2], self.bounds[3])
        z = random.uniform(self.bounds[4], self.bounds[5])
        return np.array([x, y, z])

    def plan(self, start: np.ndarray, goal: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """
        使用RRT搜索从start到goal的路径，返回 (路径长度, 路径点列表)。
        若找不到，返回直线距离 * infeasible_penalty_factor 和直线路径。
        """
        # 快速检查直线是否可行
        if self._collision_free(start, goal):
            return np.linalg.norm(goal - start), [start, goal]

        # 节点存储
        nodes = [start]  # 坐标列表
        parents = [-1]   # 父节点索引
        costs = [0.0]    # 从起点到当前节点的累计代价

        # 初始化KD树
        tree = cKDTree([start])

        for _ in range(self.max_iter):
            # 采样
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = self._random_sample()

            # 最近邻搜索
            dist, idx = tree.query(sample, k=1)
            nearest = nodes[idx]

            # 扩展方向
            direction = sample - nearest
            dist_to_sample = np.linalg.norm(direction)
            if dist_to_sample < 1e-6:
                continue
            new_pos = nearest + (direction / dist_to_sample) * min(self.step_size, dist_to_sample)

            # 碰撞检测
            if not self._collision_free(nearest, new_pos):
                continue

            # 添加新节点
            new_idx = len(nodes)
            nodes.append(new_pos)
            parents.append(idx)
            costs.append(costs[idx] + np.linalg.norm(new_pos - nearest))
            # 更新KD树（重建，简单但节点数不多时可行）
            tree = cKDTree(nodes)

            # 尝试连接目标
            if np.linalg.norm(new_pos - goal) <= self.step_size:
                if self._collision_free(new_pos, goal):
                    # 回溯路径
                    path = []
                    # 先加goal
                    path.append(goal)
                    # 回溯到new_pos
                    cur_idx = new_idx
                    while cur_idx != -1:
                        path.append(nodes[cur_idx])
                        cur_idx = parents[cur_idx]
                    path.reverse()  # 从start到goal
                    total_cost = costs[new_idx] + np.linalg.norm(new_pos - goal)
                    return total_cost, path

        # 未找到路径，返回惩罚值
        straight = np.linalg.norm(goal - start)
        print(f"  RRT警告: 未找到路径，使用直线距离×{self.infeasible_penalty_factor} 替代")
        return straight * self.infeasible_penalty_factor, [start, goal]


# ========== 噪声影响计算 ==========
def compute_path_noise_impact(
        path: List[np.ndarray],
        residential_positions: np.ndarray,
        residential_tree: cKDTree,
        source_level: float,
        threshold: float,
        flight_height: float,
        ground_type: str = 'urban',
        search_radius: float = 400.0
) -> int:
    """
    计算路径影响的居住建筑数量（优化版）
    返回受影响建筑数量
    """
    affected = set()
    total_checks = 0

    for point in path:
        # 查询半径内建筑索引
        indices = residential_tree.query_ball_point(point, search_radius)
        total_checks += len(indices)

        for idx in indices:
            if idx in affected:
                continue

            bx, by, bz = residential_positions[idx]
            # 三维欧氏距离
            d = np.linalg.norm(point - np.array([bx, by, bz]))

            # 计算噪声级
            noise = NoiseCalculator.calculate_noise_level(
                source_level, d, flight_height, ground_type
            )

            if noise > threshold:
                affected.add(idx)

    if DEBUG_NOISE and len(path) > 0:
        print(f"    路径点{len(path)}个, 检查建筑{total_checks}次, 发现{len(affected)}个受影响建筑")

    return len(affected)


# ========== 数据读取 ==========
def read_building_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取建筑数据文件并规范化为英文列名。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"建筑数据文件不存在: {file_path}")
    print(f"读取建筑数据文件: {file_path}")
    df = load_building_data(file_path)

    hospitals = df[df["land_use_type"] == HEALTHCARE_LAND_USE].copy()
    residences = df[df["land_use_type"] == RESIDENTIAL_LAND_USE].copy()

    print(f"医疗卫生用地数量: {len(hospitals)}")
    print(f"居住用地数量: {len(residences)}")
    print(f"建筑物总数: {len(df)}")

    return hospitals, residences, df


def read_station_data(file_path: str) -> pd.DataFrame:
    """读取站点数据文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"站点数据文件不存在: {file_path}")
    print(f"读取站点数据文件: {file_path}")
    df = load_station_data(file_path)
    print(f"站点数量: {len(df)}")
    return df


def create_points(hospitals: pd.DataFrame,
                  residences: pd.DataFrame,
                  stations_df: pd.DataFrame,
                  num_supply: int = 2,
                  num_demand: int = 4,
                  num_stations: int = 1,
                  flight_height: float = FLIGHT_HEIGHT) -> Tuple[List[Point], List[Point], List[Point]]:
    """创建静态点，海拔设为飞行高度"""
    supply_points = []
    n_supply = min(num_supply, len(hospitals))
    selected_hospitals = hospitals.head(n_supply)

    for idx, (_, row) in enumerate(selected_hospitals.iterrows()):
        point = Point(
            id=f"S{idx + 1}",
            lon=float(row["longitude"]),
            lat=float(row["latitude"]),
            alt=flight_height,
            type='supply'
        )
        supply_points.append(point)

    demand_points = []
    n_demand = min(num_demand, len(residences))
    selected_residences = residences.head(n_demand)

    for idx, (_, row) in enumerate(selected_residences.iterrows()):
        point = Point(
            id=f"D{idx + 1}",
            lon=float(row["longitude"]),
            lat=float(row["latitude"]),
            alt=flight_height,
            type='demand'
        )
        demand_points.append(point)

    station_points = []
    n_stations = min(num_stations, len(stations_df))
    selected_stations = stations_df.head(n_stations)

    for idx, (_, row) in enumerate(selected_stations.iterrows()):
        point = Point(
            id=f"L{idx + 1}",
            lon=float(row["longitude"]),
            lat=float(row["latitude"]),
            alt=flight_height,
            type='station'
        )
        station_points.append(point)

    return supply_points, demand_points, station_points


def create_obstacles_from_buildings(
        building_df: pd.DataFrame,
        selected_coords: set,
        min_obstacle_height: float = 30.0,
        obstacle_radius: float = 20.0
) -> List[dict]:
    """
    从建筑数据中生成球体障碍物列表。
    selected_coords: 已选为任务点的坐标集合 (lon, lat)
    """
    obstacles = []
    for _, row in building_df.iterrows():
        lon, lat = float(row["longitude"]), float(row["latitude"])
        if (lon, lat) in selected_coords:
            continue
        height = float(row["building_height_m"])
        if height <= min_obstacle_height:
            continue
        dem = float(row["ground_elevation_m"])
        # 球心位于建筑物几何中心
        center_alt = dem + height / 2.0
        # 临时创建点用于ENU转换（将在后续进行）
        obstacles.append({
            'lon': lon,
            'lat': lat,
            'alt': center_alt,
            'radius': obstacle_radius
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
        search_radius: float = 400.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建基于RRT的真实距离矩阵和噪声矩阵。
    返回：
        dist_matrix: 浮点数矩阵（米）
        noise_matrix: 整数矩阵（受影响建筑数量）
    """
    print("\n" + "=" * 60)
    print("开始构建真实距离矩阵和噪声矩阵")
    print("=" * 60)

    n = len(task_points)
    point_ids = [p.id for p in task_points]
    print(f"任务点: {point_ids}")

    # 转换障碍物中心到ENU
    obstacles = []
    for obs in obstacles_raw:
        p = Point(id="", lon=obs['lon'], lat=obs['lat'], alt=obs['alt'])
        p.to_enu(ref_lat, ref_lon, ref_alt=0.0)
        center = np.array([p.x, p.y, p.z])
        obstacles.append((center, obs['radius']))

    print(f"障碍物数量: {len(obstacles)}")

    # 计算空间边界
    all_x = [p.x for p in task_points] + [c[0] for c, _ in obstacles]
    all_y = [p.y for p in task_points] + [c[1] for c, _ in obstacles]
    all_z = [p.z for p in task_points] + [c[2] for c, _ in obstacles]
    margin = 200.0
    bounds = (
        min(all_x) - margin, max(all_x) + margin,
        min(all_y) - margin, max(all_y) + margin,
        min(all_z) - margin, max(all_z) + margin
    )
    print(f"空间边界: X[{bounds[0]:.0f}, {bounds[1]:.0f}], "
          f"Y[{bounds[2]:.0f}, {bounds[3]:.0f}], "
          f"Z[{bounds[4]:.0f}, {bounds[5]:.0f}]")

    # 创建RRT规划器
    planner = FastRRTPlanner(
        obstacles=obstacles,
        bounds=bounds,
        step_size=rrt_step_size,
        max_iter=rrt_max_iter,
        goal_bias=rrt_goal_bias,
        grid_cell_size=grid_cell_size
    )

    # 任务点位置数组
    positions = [np.array([p.x, p.y, p.z]) for p in task_points]

    # 初始化矩阵
    dist_matrix = np.zeros((n, n))
    noise_matrix = np.zeros((n, n), dtype=int)

    total_pairs = n * (n - 1) // 2
    current_pair = 0

    for i in range(n):
        for j in range(i + 1, n):
            current_pair += 1
            print(f"\n[{current_pair}/{total_pairs}] 计算路径 {task_points[i].id} -> {task_points[j].id} ...")

            # 规划路径
            length, path = planner.plan(positions[i], positions[j])
            dist_matrix[i, j] = length
            dist_matrix[j, i] = length
            print(f"  路径长度: {length:.1f}米, 路径点数: {len(path)}")

            # 计算噪声影响
            num_affected = compute_path_noise_impact(
                path=path,
                residential_positions=residential_positions,
                residential_tree=residential_tree,
                source_level=NOISE_SOURCE_LEVEL,
                threshold=noise_threshold,
                flight_height=flight_height,
                ground_type='urban',
                search_radius=search_radius
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
    """计算欧氏距离矩阵（备用，此处不再使用）"""
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


def create_drones(station_points: List[Point],
                  drones_per_station: int = 3,
                  max_payload: float = 50.0,
                  max_range: float = 150000.0,
                  speed: float = 60.0) -> List[Drone]:
    """创建无人机"""
    drones = []
    for s_idx, station in enumerate(station_points):
        for d_idx in range(drones_per_station):
            drone = Drone(
                id=f"U{s_idx + 1}{d_idx + 1}",
                station_id=s_idx,
                station_name=station.id,
                max_payload=max_payload,
                max_range=max_range,
                speed=speed
            )
            drones.append(drone)

    return drones


# ========== 改进的需求生成器 ==========
def generate_demand_events(
        demand_points: List[Point],
        supply_points: List[Point],
        num_events: int = 8,
        sim_duration: float = 2.0
) -> List[DemandEvent]:
    """
    生成需求事件，支持同一个需求点生成多个需求。
    需求在时间上分布，模拟真实场景。
    """
    events = []
    n_supply = len(supply_points)
    n_demand = len(demand_points)

    # 生成不重复的优先级列表（1到num_events）
    priorities = list(range(1, num_events + 1))
    random.shuffle(priorities)

    # 记录每个需求点已经生成了多少个需求
    demand_point_counter = {f"D{i + 1}": 0 for i in range(n_demand)}

    print(f"\n开始生成 {num_events} 个需求...")
    print(f"供给点数量: {n_supply}, 需求点数量: {n_demand}")

    for i in range(num_events):
        # 随机选择需求点（允许重复选择，模拟同一个地点多个需求）
        d_idx = random.randint(0, n_demand - 1)
        node_idx = d_idx + n_supply
        demand_point_id = demand_points[d_idx].id

        # 更新计数器
        demand_point_counter[demand_point_id] += 1
        occurrence = demand_point_counter[demand_point_id]

        # 随机选择对应的供给点（不同需求可能从不同供给点取货）
        required_supply_idx = random.randint(0, n_supply - 1)

        # 生成需求时间（分布在0.1到1.5之间）
        t = round(0.1 + random.random() * 1.4, 3)

        # 生成随机重量（5kg到30kg之间）
        weight = round(5.0 + random.random() * 25.0, 1)

        # 优先级
        priority = priorities[i]

        # 创建唯一ID，包含需求点编号和出现次数
        unique_id = f"DEM_{demand_point_id}_{occurrence:02d}"

        events.append(DemandEvent(
            time=t,
            node_idx=node_idx,
            weight=weight,
            unique_id=unique_id,
            priority=priority,
            required_supply_idx=required_supply_idx,
            demand_point_id=demand_point_id
        ))

    # 按时间排序
    events.sort(key=lambda x: x.time)

    # 打印生成的详细信息
    print(f"\n生成 {num_events} 个需求，优先级分别为: {[ev.priority for ev in events]}")
    print("\n需求详情（按时间顺序）:")
    print("-" * 80)
    for i, ev in enumerate(events):
        print(f"  {ev.unique_id}: {ev.demand_point_id} (节点{ev.node_idx}) | "
              f"时间={ev.time:.3f}h | 重量={ev.weight}kg | "
              f"优先级={ev.priority} | 必须从 S{ev.required_supply_idx + 1} 取货")

    # 统计每个需求点的需求数量
    print("\n需求点分布统计:")
    for demand_id, count in demand_point_counter.items():
        if count > 0:
            print(f"  {demand_id}: {count} 个需求")

    return events


# ========== CPLEX求解器（加入噪声成本）==========
class CplexSolver:
    def __init__(self, drones: List[Drone], supply_indices: List[int], station_indices: List[int],
                 dist_matrix: np.ndarray, all_points: List[Point],
                 noise_cost_matrix: np.ndarray = None, noise_weight: float = 0.0,
                 time_limit: int = 10):
        self.drones = drones
        self.supply_indices = supply_indices
        self.station_indices = station_indices
        self.dist_matrix = dist_matrix
        self.all_points = all_points
        self.time_limit = time_limit
        self.PENALTY_UNASSIGNED = 1e9
        self.noise_cost_matrix = noise_cost_matrix
        self.noise_weight = noise_weight

    def solve_assignment(self, drone_states: List[DroneState], demands: List[DemandEvent],
                         current_time: float) -> List[Dict]:
        if not demands or not drone_states:
            return []

        print(f"\n  CPLEX求解: {len(drone_states)}架无人机, {len(demands)}个需求")
        model = ConcreteModel()
        model.name = f"assignment_t{current_time:.3f}"

        # 集合
        model.DRONES = Set(initialize=range(len(drone_states)))
        model.DEMANDS = Set(initialize=range(len(demands)))
        model.SUPPLY = Set(initialize=range(len(self.supply_indices)))

        # 参数
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

        # 优先级权重
        priority_weight = {d: 1.0 / demands[d].priority for d in model.DEMANDS}

        # 预计算距离和噪声成本
        # 距离
        dist_drone_supply = {(u, s): self.dist_matrix[drone_pos[u], self.supply_indices[s]]
                             for u in model.DRONES for s in model.SUPPLY}
        dist_supply_demand = {(s, d): self.dist_matrix[self.supply_indices[s], demand_pos[d]]
                              for s in model.SUPPLY for d in model.DEMANDS}
        dist_demand_station = {(d, u): self.dist_matrix[demand_pos[d], drone_station[u]]
                               for u in model.DRONES for d in model.DEMANDS}

        # 噪声成本
        if self.noise_cost_matrix is not None:
            noise_drone_supply = {(u, s): self.noise_cost_matrix[drone_pos[u], self.supply_indices[s]]
                                  for u in model.DRONES for s in model.SUPPLY}
            noise_supply_demand = {(s, d): self.noise_cost_matrix[self.supply_indices[s], demand_pos[d]]
                                   for s in model.SUPPLY for d in model.DEMANDS}
            noise_demand_station = {(d, u): self.noise_cost_matrix[demand_pos[d], drone_station[u]]
                                    for u in model.DRONES for d in model.DEMANDS}
        else:
            noise_drone_supply = {(u, s): 0 for u in model.DRONES for s in model.SUPPLY}
            noise_supply_demand = {(s, d): 0 for s in model.SUPPLY for d in model.DEMANDS}
            noise_demand_station = {(d, u): 0 for u in model.DRONES for d in model.DEMANDS}

        # 变量
        model.x = Var(model.DRONES, model.SUPPLY, model.DEMANDS, within=Binary)
        model.y = Var(model.DRONES, model.DEMANDS, within=Binary)
        model.z = Var(model.DRONES, model.SUPPLY, within=Binary)
        model.used = Var(model.DRONES, within=Binary)
        model.unassigned = Var(model.DEMANDS, within=Binary)

        # 目标函数：加权距离 + 加权噪声
        def obj_rule(m):
            total_weighted_cost = 0
            for u in m.DRONES:
                for s in m.SUPPLY:
                    for d in m.DEMANDS:
                        # 总距离
                        dist = (dist_drone_supply[(u, s)] +
                                dist_supply_demand[(s, d)] +
                                dist_demand_station[(d, u)])
                        # 总噪声影响
                        noise = (noise_drone_supply[(u, s)] +
                                 noise_supply_demand[(s, d)] +
                                 noise_demand_station[(d, u)])
                        # 加权成本：距离 + 噪声权重 * 噪声
                        total_cost = dist + self.noise_weight * noise
                        total_weighted_cost += total_cost * priority_weight[d] * m.x[u, s, d]
            penalty = self.PENALTY_UNASSIGNED * sum(m.unassigned[d] for d in m.DEMANDS)
            return total_weighted_cost + penalty

        model.obj = Objective(rule=obj_rule, sense=minimize)

        # 基础约束（与之前相同）
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
                    total_dist += (dist_drone_supply[(u, s)] +
                                   dist_supply_demand[(s, d)] +
                                   dist_demand_station[(d, u)]) * m.x[u, s, d]
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

        # 需求点必须从对应的供给点取货
        def supply_demand_matching(m, u, s, d):
            if s != m.demand_required_supply[d]:
                return m.x[u, s, d] == 0
            else:
                return Constraint.Skip

        model.con_supply_demand_matching = Constraint(
            model.DRONES, model.SUPPLY, model.DEMANDS, rule=supply_demand_matching
        )

        # 无人机一次只能执行一个任务（串行）
        def single_task_per_drone(m, u):
            return sum(m.y[u, d] for d in m.DEMANDS) <= 1

        model.con_single_task = Constraint(model.DRONES, rule=single_task_per_drone)

        # 求解
        solver = SolverFactory('cplex')
        solver.options['timelimit'] = self.time_limit
        try:
            results = solver.solve(model, tee=True)
            if results.solver.termination_condition in [TerminationCondition.optimal,
                                                        TerminationCondition.feasible]:
                assignments = []
                for u in model.DRONES:
                    if value(model.used[u]) > 0.5:
                        for d in model.DEMANDS:
                            if value(model.y[u, d]) > 0.5:
                                for s in model.SUPPLY:
                                    if value(model.z[u, s]) > 0.5:
                                        assignments.append({
                                            'drone': drone_states[u],
                                            'drone_idx': u,
                                            'demand': demands[d],
                                            'demand_idx': d,
                                            'supply_idx': s,
                                            'supply_node': self.supply_indices[s]
                                        })
                                        break
                print(f"    CPLEX分配了 {len(assignments)} 个需求")
                return assignments
            else:
                print(f"    求解失败: {results.solver.termination_condition}")
                return []
        except Exception as e:
            print(f"    CPLEX异常: {e}")
            return []


# ========== 模拟器（加入噪声成本）==========
class FinalDroneSimulator:
    def __init__(self,
                 supply_points: List[Point],
                 demand_points: List[Point],
                 station_points: List[Point],
                 drones_static: List[Drone],
                 dist_matrix: np.ndarray,
                 demand_events: List[DemandEvent],
                 noise_cost_matrix: np.ndarray = None,
                 noise_weight: float = 0.0,
                 time_step: float = 0.001):

        self.supply_points = supply_points
        self.demand_points = demand_points
        self.station_points = station_points
        self.drones_static = drones_static
        self.dist_matrix = dist_matrix
        self.all_demand_events = demand_events
        self.time_step = time_step

        # 创建所有点列表和索引
        self.all_points = supply_points + demand_points + station_points
        self.n_supply = len(supply_points)
        self.n_demand = len(demand_points)
        self.n_station = len(station_points)

        self.supply_indices = list(range(self.n_supply))
        self.demand_indices = list(range(self.n_supply, self.n_supply + self.n_demand))
        self.station_indices = list(range(self.n_supply + self.n_demand,
                                          self.n_supply + self.n_demand + self.n_station))

        # 初始化无人机状态
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
                task_queue=[]
            )
            self.drone_states.append(ds)

        print(f"\n初始化 {len(self.drone_states)} 架无人机")

        # 需求管理
        self.unserved_demands = []
        self.unserved_demands_dict = {}
        self.completed_demands = []

        # 事件队列
        self.event_queue = []
        for ev in demand_events:
            heapq.heappush(self.event_queue, (ev.time, ev))

        self.current_time = 0.0
        self.total_distance = 0.0
        self.total_noise_impact = 0.0  # 新增：累计噪声影响

        # 创建CPLEX求解器（传入噪声矩阵和权重）
        self.cplex_solver = CplexSolver(
            drones=drones_static,
            supply_indices=self.supply_indices,
            station_indices=self.station_indices,
            dist_matrix=dist_matrix,
            all_points=self.all_points,
            noise_cost_matrix=noise_cost_matrix,
            noise_weight=noise_weight,
            time_limit=10
        )

    def run(self, end_time: float):
        print(f"\n开始模拟，结束时间 {end_time} 小时")
        print("=" * 50)

        last_solve_time = -1
        solve_interval = 0.05  # 3分钟

        while self.current_time < end_time:
            # 处理新需求
            self._process_new_demands()

            # 更新无人机位置
            self._update_positions()

            # 检查到达
            self._check_arrivals()

            # 定期求解（只分配给空闲无人机）
            if self.current_time - last_solve_time >= solve_interval:
                self._solve_assignment()
                last_solve_time = self.current_time

            self.current_time += self.time_step

        self._print_summary()

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
                if current_task['type'] == 'delivery':
                    demand_node = current_task['demand_node']
                    dist = self.dist_matrix[arrived_node, demand_node]
                    ds.remaining_range -= dist
                    self.total_distance += dist

                    # 累计噪声影响
                    if self.cplex_solver.noise_cost_matrix is not None:
                        noise = self.cplex_solver.noise_cost_matrix[arrived_node, demand_node]
                        self.total_noise_impact += noise

                    ds.status = DroneStatus.TO_DEMAND
                    ds.target_node = demand_node
                    ds.assigned_demand_id = current_task['demand_id']
                    ds.assigned_demand_node = demand_node
                    ds.assigned_demand_weight = current_task['weight']

                    demand_name = self.all_points[demand_node].id
                    print(f"    前往需求点 {demand_name} 送货")
                else:
                    print(f"    错误: 任务队列中的任务类型错误")
                    self._return_to_station(ds)
            else:
                print(f"    错误: 到达供给点但任务队列为空")
                self._return_to_station(ds)

        elif ds.status == DroneStatus.TO_DEMAND:
            print(f"[{self.current_time:.3f}] 无人机 {ds.drone_id} 到达需求点 {node_name}")

            if ds.task_queue and ds.task_queue[0]['type'] == 'delivery':
                current_task = ds.task_queue.pop(0)
                demand_id = current_task['demand_id']

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
                        f"    送达需求 {demand.unique_id} ({demand.demand_point_id}), 剩余载重 {ds.remaining_payload:.1f}kg")

                    if ds.task_queue:
                        print(f"    还有 {len(ds.task_queue)} 个任务等待执行")
                        # 直接执行下一个任务
                        next_task = ds.task_queue[0]
                        if next_task['type'] == 'delivery':
                            # 下一个任务需要先去供给点取货
                            supply_node = next_task['supply_node']
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
                            demand_name = self.all_points[next_task['demand_node']].id
                            print(f"    开始执行下一个任务: 前往 {supply_name} 取货，送 {demand_name}")
                    else:
                        self._return_to_station(ds)
                else:
                    print(f"    错误: 找不到匹配的需求 {demand_id}")
                    self._return_to_station(ds)
            else:
                print(f"    错误: 到达需求点但任务队列为空")
                self._return_to_station(ds)

        elif ds.status == DroneStatus.RETURNING:
            print(f"[{self.current_time:.3f}] 无人机 {ds.drone_id} 到达站点")

            if ds.task_queue:
                print(f"    还有 {len(ds.task_queue)} 个任务，准备执行下一个任务")
                next_task = ds.task_queue[0]
                if next_task['type'] == 'delivery':
                    supply_node = next_task['supply_node']
                    dist = self.dist_matrix[ds.current_node, supply_node]
                    ds.remaining_range -= dist
                    self.total_distance += dist

                    # 累计噪声影响
                    if self.cplex_solver.noise_cost_matrix is not None:
                        noise = self.cplex_solver.noise_cost_matrix[ds.current_node, supply_node]
                        self.total_noise_impact += noise

                    ds.status = DroneStatus.TO_SUPPLY
                    ds.target_node = supply_node
                    ds.assigned_supply_node = supply_node

                    supply_name = self.all_points[supply_node].id
                    demand_name = self.all_points[next_task['demand_node']].id
                    print(f"    前往供给点 {supply_name} 取货 (为 {demand_name} 送货)")
                else:
                    print(f"    错误: 任务类型错误")
                    ds.status = DroneStatus.IDLE
                    ds.target_node = None
            else:
                # 没有任务了，充电
                ds.status = DroneStatus.CHARGING
                ds.target_node = None
                ds.status = DroneStatus.IDLE
                ds.remaining_range = self.drones_static[0].max_range
                ds.remaining_payload = self.drones_static[0].max_payload
                ds.assigned_demand_id = None
                ds.assigned_demand_node = None
                ds.assigned_demand_weight = None
                ds.assigned_supply_node = None
                print(f"    充电完成，变为空闲")

    def _return_to_station(self, ds: DroneState):
        station_node = self.station_indices[ds.station_id]
        dist = self.dist_matrix[ds.current_node, station_node]
        ds.remaining_range -= dist
        self.total_distance += dist

        # 累计噪声影响
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
        # 过滤掉已经分配和None的需求
        pending = [d for d in self.unserved_demands if d is not None and d.assigned_drone is None]

        if not idle_drones or not pending:
            return

        print(f"\n[{self.current_time:.3f}] 调用CPLEX求解器")
        print(f"  空闲无人机: {len(idle_drones)}, 待处理需求: {len(pending)}")

        assignments = self.cplex_solver.solve_assignment(idle_drones, pending, self.current_time)

        for assign in assignments:
            drone = assign['drone']
            demand = assign['demand']
            supply_node = assign['supply_node']

            demand.assigned_drone = drone.drone_id
            demand.supply_node = supply_node

            task = {
                'type': 'delivery',
                'demand_id': demand.unique_id,
                'demand_node': demand.node_idx,
                'supply_node': supply_node,
                'weight': demand.weight,
                'priority': demand.priority
            }
            drone.task_queue.append(task)

            if drone.status == DroneStatus.IDLE and len(drone.task_queue) == 1:
                first_task = drone.task_queue[0]
                dist = self.dist_matrix[drone.current_node, supply_node]
                drone.remaining_range -= dist
                self.total_distance += dist

                # 累计噪声影响
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
        # 按需求点分组显示
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


# ========== 主程序 ==========
def main():
    print("=" * 70)
    print("无人机动态配送 - 基于RRT真实路径和噪声矩阵")
    print("=" * 70)

    # 设置随机种子，保证结果可重现
    random.seed(42)
    np.random.seed(42)

    # 读取数据
    try:
        hospitals, residences, all_buildings = read_building_data(str(BUILDING_DATA_PATH))
        stations_df = read_station_data(str(STATION_DATA_PATH))

        # 配置：2个供给点，4个需求点，1个站点
        supply_points, demand_points, station_points = create_points(
            hospitals, residences, stations_df,
            num_supply=2, num_demand=4, num_stations=1,
            flight_height=FLIGHT_HEIGHT
        )
        print(f"\n实际创建: {len(supply_points)}个供给点, {len(demand_points)}个需求点, {len(station_points)}个站点")

    except FileNotFoundError:
        print("使用测试数据...")
        # 创建测试数据：2个供给点，4个需求点，1个站点
        supply_points = [
            Point(id="S1", lon=116.30, lat=39.90, alt=FLIGHT_HEIGHT, type='supply'),
            Point(id="S2", lon=116.31, lat=39.90, alt=FLIGHT_HEIGHT, type='supply')
        ]
        demand_points = [
            Point(id="D1", lon=116.32, lat=39.90, alt=FLIGHT_HEIGHT, type='demand'),
            Point(id="D2", lon=116.30, lat=39.91, alt=FLIGHT_HEIGHT, type='demand'),
            Point(id="D3", lon=116.31, lat=39.91, alt=FLIGHT_HEIGHT, type='demand'),
            Point(id="D4", lon=116.32, lat=39.91, alt=FLIGHT_HEIGHT, type='demand')
        ]
        station_points = [
            Point(id="L1", lon=116.30, lat=39.90, alt=FLIGHT_HEIGHT, type='station')
        ]
        # 创建简单的建筑物数据用于障碍物（测试用）
        # 由于没有真实建筑数据，这里生成一些随机障碍物
        # 在实际使用中，应确保文件存在

    # 坐标转换
    all_points = supply_points + demand_points + station_points
    ref_lat = np.mean([p.lat for p in all_points])
    ref_lon = np.mean([p.lon for p in all_points])
    for p in all_points:
        p.to_enu(ref_lat, ref_lon, 0)

    # ========== 新增：构建障碍物和居住建筑数据 ==========
    # 收集已选任务点坐标
    selected_coords = set()
    for p in all_points:
        selected_coords.add((p.lon, p.lat))

    # 从建筑数据生成障碍物（如果数据存在）
    obstacles_raw = []
    if 'all_buildings' in locals():
        obstacles_raw = create_obstacles_from_buildings(
            all_buildings,
            selected_coords,
            min_obstacle_height=30.0,
            obstacle_radius=20.0,
        )
    else:
        # 测试数据：生成一些随机障碍物
        for _ in range(5):
            lon = 116.30 + random.uniform(-0.01, 0.03)
            lat = 39.90 + random.uniform(-0.01, 0.03)
            alt = 70.0 + random.uniform(-10, 10)
            obstacles_raw.append({
                'lon': lon,
                'lat': lat,
                'alt': alt,
                'radius': 20.0
            })

    # 构建居住建筑点集（用于噪声评估）
    residential_positions = []
    if 'residences' in locals() and not residences.empty:
        for _, row in residences.iterrows():
            p = Point(
                id="",
                lon=float(row["longitude"]),
                lat=float(row["latitude"]),
                alt=float(row["ground_elevation_m"]),
            )
            p.to_enu(ref_lat, ref_lon, 0)
            residential_positions.append([p.x, p.y, p.z])
    else:
        # 测试：使用需求点作为居住建筑（模拟）
        for p in demand_points:
            # 居住建筑在地面（alt 为 0）
            p_ground = Point(id="", lon=p.lon, lat=p.lat, alt=0.0)
            p_ground.to_enu(ref_lat, ref_lon, 0)
            residential_positions.append([p_ground.x, p_ground.y, p_ground.z])

    residential_positions = np.array(residential_positions)
    residential_tree = cKDTree(residential_positions)
    print(f"居住建筑数量: {len(residential_positions)}")

    # ========== 构建真实距离矩阵和噪声矩阵 ==========
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
        search_radius=400.0
    )

    # 创建无人机：3架无人机（从同一个站点）
    drones_static = create_drones(
        station_points,
        drones_per_station=3,
        max_payload=60.0,
        max_range=200000.0,
        speed=60.0
    )

    # 生成需求：8个需求，分布在4个需求点上
    demand_events = generate_demand_events(
        demand_points,
        supply_points,
        num_events=8,
        sim_duration=2.0
    )

    noise_weight = 0.5  # 噪声权重

    # 创建模拟器
    sim = FinalDroneSimulator(
        supply_points=supply_points,
        demand_points=demand_points,
        station_points=station_points,
        drones_static=drones_static,
        dist_matrix=dist_matrix,
        demand_events=demand_events,
        noise_cost_matrix=noise_cost_matrix,
        noise_weight=noise_weight,
        time_step=0.001
    )

    # 运行模拟
    sim.run(end_time=4.0)  # 模拟4小时


if __name__ == "__main__":
    main()
