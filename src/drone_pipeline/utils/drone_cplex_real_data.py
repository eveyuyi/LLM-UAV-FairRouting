import math
import os
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    SolverFactory, Binary, NonNegativeReals,
    minimize, value, TerminationCondition
)

import warnings
warnings.filterwarnings('ignore')

# 确保 CPLEX 二进制可被 Pyomo 找到
_CPLEX_BIN = "/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx"
if os.path.isdir(_CPLEX_BIN) and _CPLEX_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CPLEX_BIN + ":" + os.environ.get("PATH", "")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 1. 数据结构
# ============================================================================

@dataclass
class Point:
    """地理坐标点"""
    id: str
    lon: float      # 经度
    lat: float      # 纬度
    alt: float      # 高度
    x: float = 0.0  # ENU东向坐标
    y: float = 0.0  # ENU北向坐标
    z: float = 0.0  # ENU高程
    type: str = ''  # 类型
    
    def to_enu(self, ref_lat: float, ref_lon: float, ref_alt: float):
        """经纬度转ENU坐标"""
        lat_scale = 111000.0  # 1度 ≈ 111km
        lon_scale = 111000.0 * math.cos(math.radians(ref_lat))
        
        self.x = (self.lon - ref_lon) * lon_scale
        self.y = (self.lat - ref_lat) * lat_scale
        self.z = self.alt - ref_alt


@dataclass
class Drone:
    """无人机"""
    id: str
    station_id: int    # 所属站点索引
    station_name: str  # 站点名称
    max_payload: float = 10.0   # 最大载重(kg)
    max_range: float = 20000.0  # 最大航程(m)


# ============================================================================
# 2. 数据读取模块
# ============================================================================

def read_building_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取建筑数据，返回医院和居住区数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"建筑数据文件不存在: {file_path}")
    
    print(f"正在读取建筑数据: {file_path}")
    df = pd.read_excel(file_path)
    print(f"总记录数: {len(df)}")
    
    # 检查必需列
    required_cols = ['type', '经度', '纬度']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"建筑数据缺少必需列: {col}")
    
    # 提取医院和居住区
    hospitals = df[df['type'] == '医疗卫生用地'].copy()
    residences = df[df['type'] == '居住用地'].copy()
    
    print(f"医疗卫生用地（供给点候选）: {len(hospitals)}个")
    print(f"居住用地（需求点候选）: {len(residences)}个")
    
    return hospitals, residences


def read_station_data(file_path: str) -> pd.DataFrame:
    """读取无人机站点数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"站点数据文件不存在: {file_path}")
    
    print(f"\n正在读取站点数据: {file_path}")
    df = pd.read_excel(file_path)
    print(f"站点总数: {len(df)}")
    
    # 检查必需列
    required_cols = ['经度', '纬度', '站点名']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"站点数据缺少必需列: {col}")
    
    return df


def create_points_from_data(hospitals: pd.DataFrame, 
                           residences: pd.DataFrame,
                           stations_df: pd.DataFrame,
                           num_supply: int = 3,
                           num_demand: int = 6,
                           num_stations: int = 3) -> Tuple[List[Point], List[Point], List[Point], List[float]]:
    """从真实数据创建供给点、需求点和站点"""
    
    print("\n生成供给点、需求点和站点...")
    
    # ===== 1. 创建供给点 =====
    supply_points = []
    n_supply = min(num_supply, len(hospitals))
    selected_hospitals = hospitals.head(n_supply)
    
    for idx, (_, row) in enumerate(selected_hospitals.iterrows()):
        building_height = float(row.get('Height', 0)) if 'Height' in row else 0
        dem_height = float(row.get('DEM高度', 50)) if 'DEM高度' in row else 50
        total_height = dem_height + building_height
        
        point = Point(
            id=f"S{idx+1}",
            lon=float(row['经度']),
            lat=float(row['纬度']),
            alt=total_height,
            type='supply'
        )
        supply_points.append(point)
        print(f"  供给点 {point.id}: 经度={point.lon:.6f}, 纬度={point.lat:.6f}, 高度={point.alt:.1f}m")
    
    # ===== 2. 创建需求点 =====
    demand_points = []
    n_demand = min(num_demand, len(residences))
    selected_residences = residences.head(n_demand)
    
    # 生成随机重量
    np.random.seed(42)
    demand_weights = []
    
    for idx, (_, row) in enumerate(selected_residences.iterrows()):
        building_height = float(row.get('Height', 0)) if 'Height' in row else 0
        dem_height = float(row.get('DEM高度', 50)) if 'DEM高度' in row else 50
        total_height = dem_height + building_height
        
        point = Point(
            id=f"D{idx+1}",
            lon=float(row['经度']),
            lat=float(row['纬度']),
            alt=total_height,
            type='demand'
        )
        demand_points.append(point)
        
        weight = round(np.random.uniform(1.0, 8.0), 1)
        demand_weights.append(weight)
        
        print(f"  需求点 {point.id}: 经度={point.lon:.6f}, 纬度={point.lat:.6f}, "
              f"高度={point.alt:.1f}m, 重量={weight}kg")
    
    # ===== 3. 创建站点 =====
    station_points = []
    n_stations = min(num_stations, len(stations_df))
    selected_stations = stations_df.head(n_stations)
    
    for idx, (_, row) in enumerate(selected_stations.iterrows()):
        point = Point(
            id=f"L{idx+1}",
            lon=float(row['经度']),
            lat=float(row['纬度']),
            alt=50.0,
            type='station'
        )
        station_points.append(point)
        print(f"  站点 {point.id}: {row['站点名']}, 经度={point.lon:.6f}, 纬度={point.lat:.6f}")
    
    return supply_points, demand_points, station_points, demand_weights


def create_drones(station_points: List[Point], 
                 drones_per_station: int = 6,
                 max_payload: float = 10.0,
                 max_range: float = 20000.0) -> List[Drone]:
    """创建无人机"""
    drones = []
    for s_idx, station in enumerate(station_points):
        for d_idx in range(drones_per_station):
            drone = Drone(
                id=f"U{s_idx+1}{d_idx+1}",
                station_id=s_idx,
                station_name=station.id,
                max_payload=max_payload,
                max_range=max_range
            )
            drones.append(drone)
    
    print(f"\n生成无人机: {len(drones)}架")
    print(f"  每站点 {drones_per_station} 架")
    print(f"  最大载重: {max_payload}kg")
    print(f"  最大航程: {max_range/1000}km")
    
    return drones


# ============================================================================
# 3. 距离矩阵计算
# ============================================================================

def calculate_distance_matrix(supply_points: List[Point],
                             demand_points: List[Point],
                             station_points: List[Point]) -> Dict:
    """计算基于真实坐标的距离矩阵"""
    
    print("\n计算距离矩阵...")
    
    # 合并所有点
    all_points = supply_points + demand_points + station_points
    n_points = len(all_points)
    
    # 计算参考点
    ref_lat = np.mean([p.lat for p in all_points])
    ref_lon = np.mean([p.lon for p in all_points])
    ref_alt = np.mean([p.alt for p in all_points])
    
    print(f"参考点: 纬度={ref_lat:.6f}, 经度={ref_lon:.6f}, 高度={ref_alt:.1f}m")
    
    # 转换为ENU坐标
    for p in all_points:
        p.to_enu(ref_lat, ref_lon, ref_alt)
    
    # 计算距离矩阵
    dist_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                dx = all_points[i].x - all_points[j].x
                dy = all_points[i].y - all_points[j].y
                dz = all_points[i].z - all_points[j].z
                dist_matrix[i, j] = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # 索引映射
    n_supply = len(supply_points)
    n_demand = len(demand_points)
    n_station = len(station_points)
    
    indices = {
        'supply': list(range(n_supply)),
        'demand': list(range(n_supply, n_supply + n_demand)),
        'station': list(range(n_supply + n_demand, n_supply + n_demand + n_station))
    }
    
    print(f"距离矩阵维度: {n_points}×{n_points}")
    print(f"  供给点: {n_supply}个 (索引 {indices['supply']})")
    print(f"  需求点: {n_demand}个 (索引 {indices['demand']})")
    print(f"  站点: {n_station}个 (索引 {indices['station']})")
    
    return {
        'matrix': dist_matrix,
        'points': all_points,
        'indices': indices,
        'n_supply': n_supply,
        'n_demand': n_demand,
        'n_station': n_station,
        'n_nodes': n_points,
        'ref_point': (ref_lat, ref_lon, ref_alt)
    }


# ============================================================================
# 4. CPLEX求解器（修复状态判断）
# ============================================================================

class CplexDroneSolver:
    """基于真实数据的CPLEX无人机路径求解器 - 修复状态判断"""
    
    def __init__(self, 
                 supply_points: List[Point],
                 demand_points: List[Point],
                 station_points: List[Point],
                 demand_weights: List[float],
                 drones: List[Drone],
                 dist_info: Dict):
        
        self.supply_points = supply_points
        self.demand_points = demand_points
        self.station_points = station_points
        self.demand_weights = demand_weights
        self.drones = drones
        self.dist = dist_info['matrix']
        self.indices = dist_info['indices']
        
        # 基本参数
        self.n_drones = len(drones)
        self.n_supply = dist_info['n_supply']
        self.n_demand = dist_info['n_demand']
        self.n_station = dist_info['n_station']
        self.n_nodes = dist_info['n_nodes']
        
        self.model = None
        self.results = None
        self.solution = None
        self.solve_time = 0
        self.solve_status = None
        self.cruise_speed: float = 15.0  # default, overridden by ParameterizedDroneSolver
        
        print(f"\nCPLEX求解器初始化:")
        print(f"  - 无人机数量: {self.n_drones}")
        print(f"  - 供给点数量: {self.n_supply}")
        print(f"  - 需求点数量: {self.n_demand}")
        print(f"  - 站点数量: {self.n_station}")
        print(f"  - 总节点数: {self.n_nodes}")
    
    def build_model(self):
        """构建MILP模型"""
        model = ConcreteModel()
        
        # ===== 集合 =====
        model.U = Set(initialize=range(self.n_drones))
        model.S = Set(initialize=self.indices['supply'])
        model.D = Set(initialize=self.indices['demand'])
        model.L = Set(initialize=self.indices['station'])
        model.N = Set(initialize=range(self.n_nodes))
        
        # ===== 参数 =====
        # 距离矩阵
        def dist_init(m, i, j):
            return self.dist[i, j]
        model.dist = Param(model.N, model.N, initialize=dist_init)
        
        # 需求重量
        def weight_init(m, d):
            d_idx = list(model.D).index(d)
            return self.demand_weights[d_idx]
        model.weight = Param(model.D, initialize=weight_init)
        
        # 无人机参数
        model.capacity = Param(initialize=self.drones[0].max_payload)
        model.max_range = Param(initialize=self.drones[0].max_range)
        
        # ===== 变量 =====
        # 路径变量
        model.x = Var(model.U, model.N, model.N, within=Binary)
        
        # 分配变量
        model.y = Var(model.U, model.D, within=Binary)
        model.z = Var(model.U, model.S, within=Binary)
        
        # 无人机是否使用
        model.used = Var(model.U, within=Binary, bounds=(0, 1))
        
        # ===== 目标：最小化总距离 =====
        def obj_rule(m):
            return sum(m.dist[i, j] * m.x[u, i, j] 
                      for u in m.U for i in m.N for j in m.N)
        model.obj = Objective(rule=obj_rule, sense=minimize)
        
        # ===== 约束条件 =====
        
        # C1: 每个需求点必须被服务一次
        def demand_c1(m, d):
            return sum(m.y[u, d] for u in m.U) == 1
        model.con_demand = Constraint(model.D, rule=demand_c1)
        
        # C2: 载重约束
        def capacity_c2(m, u):
            return sum(m.weight[d] * m.y[u, d] for d in m.D) <= m.capacity * m.used[u]
        model.con_capacity = Constraint(model.U, rule=capacity_c2)
        
        # C3: 使用关联
        def usage_c3(m, u, d):
            return m.y[u, d] <= m.used[u]
        model.con_usage = Constraint(model.U, model.D, rule=usage_c3)
        
        # C4: 每架无人机最多选一个供给点
        def supply_c4(m, u):
            return sum(m.z[u, s] for s in m.S) <= m.used[u]
        model.con_supply = Constraint(model.U, rule=supply_c4)
        
        # C5: 服务需求点必须选供给点
        def need_supply_c5(m, u):
            return sum(m.y[u, d] for d in m.D) <= sum(m.z[u, s] for s in m.S) * self.n_demand
        model.con_need_supply = Constraint(model.U, rule=need_supply_c5)
        
        # C6: 从站点出发
        def start_c6(m, u):
            station_id = self.drones[u].station_id
            station_node = self.indices['station'][station_id]
            return sum(m.x[u, station_node, j] for j in m.N if j != station_node) == m.used[u]
        model.con_start = Constraint(model.U, rule=start_c6)
        
        # C7: 返回站点
        def return_c7(m, u):
            station_id = self.drones[u].station_id
            station_node = self.indices['station'][station_id]
            return sum(m.x[u, j, station_node] for j in m.N if j != station_node) == m.used[u]
        model.con_return = Constraint(model.U, rule=return_c7)
        
        # C8: 需求点流入 = 是否服务
        def demand_flow_c8(m, u, d):
            return sum(m.x[u, j, d] for j in m.N) == m.y[u, d]
        model.con_demand_flow = Constraint(model.U, model.D, rule=demand_flow_c8)
        
        # C9: 供给点流入 = 是否选择
        def supply_flow_c9(m, u, s):
            return sum(m.x[u, j, s] for j in m.N) == m.z[u, s]
        model.con_supply_flow = Constraint(model.U, model.S, rule=supply_flow_c9)
        
        # C10: 流量守恒（需求点）
        def flow_balance_c10(m, u, d):
            inflow = sum(m.x[u, j, d] for j in m.N)
            outflow = sum(m.x[u, d, j] for j in m.N)
            return inflow == outflow
        model.con_flow = Constraint(model.U, model.D, rule=flow_balance_c10)
        
        # C11: 续航约束
        def range_c11(m, u):
            total = sum(m.dist[i, j] * m.x[u, i, j] for i in m.N for j in m.N)
            return total <= m.max_range * m.used[u]
        model.con_range = Constraint(model.U, rule=range_c11)
        
        # C12: 禁止自环
        def no_self_c12(m, u, i):
            return m.x[u, i, i] == 0
        model.con_no_self = Constraint(model.U, model.N, rule=no_self_c12)
        
        self.model = model
        return model
    

    def solve(self, time_limit=300):
        """使用CPLEX求解 - 修复状态判断"""
        if not self.model:
            self.build_model()

        print("\n" + "=" * 60)
        print("调用CPLEX求解器...")
        print("=" * 60)

        start = time.time()

        try:
            # 检查CPLEX是否可用
            if not SolverFactory('cplex').available():
                raise Exception("CPLEX求解器不可用，请检查安装")

            solver = SolverFactory('cplex')

            # 正确设置CPLEX参数
            solver.options['timelimit'] = time_limit
            solver.options['mip_tolerances_mipgap'] = 0.1
            solver.options['threads'] = 4

            print(f"正在求解中（时间限制: {time_limit}秒）...")
            print("CPLEX参数:")
            print(f"  - timelimit: {time_limit}")
            print(f"  - mip_tolerances_mipgap: 0.1")
            print(f"  - threads: 4")

            # 使用tee=False避免交互式命令行问题
            self.results = solver.solve(self.model, tee=False, load_solutions=True)

            self.solve_time = time.time() - start

            # 获取求解状态
            if self.results.solver.status:
                self.solve_status = self.results.solver.termination_condition
                print(f"\n求解状态: {self.solve_status}")
                print(f"求解时间: {self.solve_time:.2f}秒")
            else:
                self.solve_status = "unknown"
                print(f"\n求解状态: {self.solve_status}")
                print(f"求解时间: {self.solve_time:.2f}秒")

            # 定义成功终止条件列表（移除不存在的minTimeLimit）
            success_conditions = [
                TerminationCondition.optimal,
                TerminationCondition.feasible,
                TerminationCondition.locallyOptimal,
                TerminationCondition.maxTimeLimit,
                # TerminationCondition.acceptable
            ]

            # 检查是否求解成功
            is_success = self.solve_status in success_conditions

            if is_success:
                print("✓ 找到可行解，正在提取结果...")

                # 获取目标函数值
                try:
                    obj_value = value(self.model.obj)
                    print(f"目标函数值: {obj_value:.2f}m ({obj_value / 1000:.2f}km)")
                except Exception as e:
                    obj_value = 0
                    print(f"警告: 无法获取目标函数值 - {e}")

                # 提取解
                self.solution = self._extract_solution_safe(obj_value)
                self.solution['solve_time_s'] = round(self.solve_time, 3)
                self.solution['solve_status'] = str(self.solve_status)

                if self.solution['drones_used'] > 0:
                    print(f"✓ 成功提取解: {self.solution['drones_used']}架无人机被使用")
                else:
                    print("⚠ 警告: 没有无人机被使用，可能无可行解")

                return self.solution
            else:
                print(f"✗ 求解失败: {self.solve_status}")
                print("可能的原因:")
                print("  1. 模型无可行解（约束过强）")
                print("  2. 求解时间不足")
                print("  3. CPLEX许可证问题")
                return None

        except Exception as e:
            print(f"\n✗ CPLEX求解失败: {e}")
            print("\n请检查:")
            print("1. CPLEX是否已正确安装")
            print("2. CPLEX许可证是否有效")
            print("3. 模型是否可行（尝试放宽约束）")
            raise
    def _extract_solution_safe(self, obj_value):
        """安全提取求解结果"""
        solution = {
            'drones_used': 0,
            'total_distance': obj_value,
            'objective_value': obj_value,
            'solve_time_s': 0.0,
            'solve_status': 'unknown',
            'assignments': [],
            'paths': []
        }
        
        try:
            # 安全获取used变量值
            used_values = {}
            for u in range(self.n_drones):
                try:
                    val = value(self.model.used[u])
                    used_values[u] = val if val is not None and not math.isnan(val) else 0
                except:
                    used_values[u] = 0
            
            # 安全获取y变量值
            y_values = {}
            for u in range(self.n_drones):
                for d in self.model.D:
                    try:
                        val = value(self.model.y[u, d])
                        y_values[(u, d)] = val if val is not None and not math.isnan(val) else 0
                    except:
                        y_values[(u, d)] = 0
            
            # 安全获取z变量值
            z_values = {}
            for u in range(self.n_drones):
                for s in self.model.S:
                    try:
                        val = value(self.model.z[u, s])
                        z_values[(u, s)] = val if val is not None and not math.isnan(val) else 0
                    except:
                        z_values[(u, s)] = 0
            
            # 提取解
            for u in range(self.n_drones):
                if used_values.get(u, 0) > 0.5:
                    solution['drones_used'] += 1
                    
                    # 获取服务的需求点
                    served_demands = []
                    for d in self.model.D:
                        if y_values.get((u, d), 0) > 0.5:
                            served_demands.append(int(d))
                    
                    # 获取选择的供给点
                    selected_supply = -1
                    for s in self.model.S:
                        if z_values.get((u, s), 0) > 0.5:
                            selected_supply = int(s)
                            break
                    
                    if served_demands:
                        # 重建路径 + 计算时间（先于 assignment 入列，方便一并存入）
                        path = self._reconstruct_path_safe(u, selected_supply, served_demands)
                        path_labels = [self._node_label(n) for n in path]
                        delivery_times = self._compute_path_delivery_times(path)
                        total_mission_time_s = delivery_times[path[-1]] if path else 0.0

                        solution['assignments'].append({
                            'drone_id': self.drones[u].id,
                            'station_id': self.drones[u].station_id,
                            'station_name': self.drones[u].station_name,
                            'supply_idx': selected_supply,
                            'demand_indices': served_demands,
                            'path_labels': path_labels,
                            'path_str': ' -> '.join(path_labels),
                            'demand_delivery_times_s': delivery_times,
                            'total_mission_time_s': round(total_mission_time_s, 1),
                        })

                        solution['paths'].append(path)
            
        except Exception as e:
            print(f"提取解时出错: {e}")
        
        return solution
    
    def _node_label(self, node_idx: int) -> str:
        """将节点全局索引映射为可读标签（如 L1, S_ST001, D_2200）。"""
        if node_idx < self.n_supply:
            return self.supply_points[node_idx].id
        elif node_idx < self.n_supply + self.n_demand:
            return self.demand_points[node_idx - self.n_supply].id
        elif node_idx < self.n_nodes:
            return self.station_points[node_idx - self.n_supply - self.n_demand].id
        return f"N{node_idx}"

    def _compute_path_delivery_times(self, path: List[int]) -> Dict[int, float]:
        """沿路径累计飞行时间（秒）。返回 {node_idx: cumulative_time_s}。"""
        speed = getattr(self, "cruise_speed", 15.0)
        times: Dict[int, float] = {}
        cumulative = 0.0
        for i in range(len(path) - 1):
            dist = float(self.dist[path[i], path[i + 1]])
            cumulative += dist / speed
            times[path[i + 1]] = round(cumulative, 1)
        return times

    def _reconstruct_path_safe(self, u, supply_idx, demand_indices):
        """安全重建无人机路径"""
        try:
            station_id = self.drones[u].station_id
            station_node = self.indices['station'][station_id]
            
            path = [station_node]
            if supply_idx != -1:
                supply_node = self.indices['supply'][supply_idx]
                path.append(supply_node)
            
            path.extend(demand_indices)
            path.append(station_node)
            
            return path
        except:
            return []
    
    def print_solution(self):
        """打印求解结果（含路径顺序、送达时间）"""
        if not self.solution:
            print("\n无解或求解失败")
            return

        print("\n" + "=" * 60)
        print("无人机配送路径规划结果")
        print("=" * 60)

        print(f"\n【求解状态】")
        print(f"  求解状态: {self.solve_status}")
        print(f"  求解时间: {self.solve_time:.2f}秒")

        total_dist = self.solution['total_distance']
        print(f"\n【全局统计】")
        print(f"  总飞行距离: {total_dist:.2f}m ({total_dist / 1000:.2f}km)")
        print(f"  使用无人机: {self.solution['drones_used']}/{self.n_drones}")

        if self.solution['drones_used'] == 0:
            print("\n⚠ 警告: 没有找到可行解！")
            print("可能的原因:")
            print("1. 无人机载重不足以服务任何需求点")
            print("2. 无人机航程不足以完成任何任务")
            print("3. 约束条件过强")
            return

        # ---- 需求完成情况汇总 ----
        served_demand_nodes = {
            d for assign in self.solution['assignments']
            for d in assign['demand_indices']
        }
        n_served = len(served_demand_nodes)
        n_total = self.n_demand
        rate = n_served / n_total * 100 if n_total else 0.0

        print(f"\n【需求完成情况】")
        print(f"  总需求: {n_total}")
        print(f"  已完成: {n_served}")
        print(f"  完成率: {rate:.1f}%")

        # ---- 需求明细（含送达时间）----
        # Build lookup: demand_node_idx → (drone_id, delivery_time_s)
        demand_info: Dict[int, tuple] = {}
        for assign in self.solution['assignments']:
            ddt = assign.get('demand_delivery_times_s', {})
            for d_node in assign['demand_indices']:
                demand_info[d_node] = (assign['drone_id'], ddt.get(d_node))

        print(f"\n【需求明细】")
        for d_node in sorted(self.indices['demand']):
            local_idx = d_node - self.n_supply
            d_label = self.demand_points[local_idx].id
            weight = self.demand_weights[local_idx] if local_idx < len(self.demand_weights) else 0.0
            if d_node in demand_info:
                drone_id, t_s = demand_info[d_node]
                t_str = f"{t_s:.0f}s ({t_s / 60:.1f}min)" if t_s is not None else "N/A"
                print(f"  {d_label}: ✓ {weight:.1f}kg (由{drone_id}) 送达: {t_str}")
            else:
                print(f"  {d_label}: ✗ {weight:.1f}kg (未服务)")

        # ---- 无人机路径 ----
        print(f"\n【无人机路径】")
        for assign in self.solution['assignments']:
            path_str = assign.get('path_str', '(无路径)')
            mission_t = assign.get('total_mission_time_s', 0.0)
            total_w = sum(
                self.demand_weights[d - self.n_supply]
                for d in assign['demand_indices']
                if 0 <= d - self.n_supply < len(self.demand_weights)
            )
            print(f"  {assign['drone_id']}: {path_str}")
            print(f"    └ 载重: {total_w:.1f}kg | 任务总时: {mission_t:.0f}s ({mission_t / 60:.1f}min)")


# ============================================================================
# 5. 结果保存和可视化
# ============================================================================

def save_and_visualize(supply_points: List[Point],
                      demand_points: List[Point],
                      station_points: List[Point],
                      dist_info: Dict,
                      solver: CplexDroneSolver,
                      output_dir: str = "cplex_real_results"):
    """保存结果并可视化"""
    
    if not solver.solution or solver.solution['drones_used'] == 0:
        print("\n没有可行解，跳过结果保存和可视化")
        return
    
    # 创建输出目录
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    print(f"\n保存结果到: {output_dir}")
    
    # ===== 1. 保存距离矩阵 =====
    df_dist = pd.DataFrame(
        dist_info['matrix'],
        index=[p.id for p in dist_info['points']],
        columns=[p.id for p in dist_info['points']]
    )
    df_dist.to_csv(output_dir / "distance_matrix.csv")
    print(f"  ✓ 距离矩阵已保存")
    
    # ===== 2. 保存求解结果 =====
    with open(output_dir / "solution.txt", 'w', encoding='utf-8') as f:
        f.write("无人机配送路径规划结果（基于真实数据）\n")
        f.write("="*70 + "\n\n")
        
        f.write("【求解信息】\n")
        f.write(f"  求解状态: {solver.solve_status}\n")
        f.write(f"  求解时间: {solver.solve_time:.2f}秒\n\n")
        
        f.write("【输入数据】\n")
        f.write(f"  供给点数量: {len(supply_points)}\n")
        f.write(f"  需求点数量: {len(demand_points)}\n")
        f.write(f"  站点数量: {len(station_points)}\n")
        f.write(f"  无人机数量: {solver.n_drones}\n\n")
        
        f.write("【需求点详情】\n")
        for i, dp in enumerate(demand_points):
            f.write(f"  {dp.id}: 经度={dp.lon:.6f}, 纬度={dp.lat:.6f}, "
                   f"高度={dp.alt:.1f}m, 重量={solver.demand_weights[i]}kg\n")
        f.write("\n")
        
        f.write("【求解结果】\n")
        f.write(f"  总飞行距离: {solver.solution['total_distance']:.2f}m\n")
        f.write(f"  使用无人机: {solver.solution['drones_used']}/{solver.n_drones}\n\n")
        
        f.write("【无人机任务分配】\n")
        for assign in solver.solution['assignments']:
            f.write(f"\n  {assign['drone_id']}:\n")
            f.write(f"    站点: {assign['station_name']}\n")
            
            if assign['supply_idx'] != -1:
                supply_id = supply_points[assign['supply_idx']].id
                f.write(f"    供给点: {supply_id}\n")
            
            demand_ids = [demand_points[d - solver.n_supply].id 
                         for d in assign['demand_indices']]
            f.write(f"    需求点顺序: {' → '.join(demand_ids)}\n")
            
            # 完整路径
            path_nodes = [assign['station_name']]
            if assign['supply_idx'] != -1:
                path_nodes.append(supply_points[assign['supply_idx']].id)
            path_nodes.extend(demand_ids)
            path_nodes.append(assign['station_name'])
            f.write(f"    完整路径: {' → '.join(path_nodes)}\n")
    
    print(f"  ✓ 求解结果已保存")
    
    # ===== 3. 可视化 =====
    try:
        plt.figure(figsize=(14, 10))
        
        points = dist_info['points']
        
        # 绘制所有点
        colors = {'supply': 'green', 'demand': 'red', 'station': 'blue'}
        labels = {'supply': '供给点', 'demand': '需求点', 'station': '站点'}
        
        legend_added = set()
        for p in points:
            if p.type not in legend_added:
                plt.scatter(p.x, p.y, c=colors[p.type], s=100, alpha=0.7, label=labels[p.type])
                legend_added.add(p.type)
            else:
                plt.scatter(p.x, p.y, c=colors[p.type], s=100, alpha=0.7)
            plt.annotate(p.id, (p.x, p.y), fontsize=9, 
                        xytext=(5, 5), textcoords='offset points')
        
        # 绘制路径
        colors_cycle = ['orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        for i, path in enumerate(solver.solution['paths']):
            if path:
                color = colors_cycle[i % len(colors_cycle)]
                path_x = [points[idx].x for idx in path]
                path_y = [points[idx].y for idx in path]
                
                drone_id = solver.solution['assignments'][i]['drone_id']
                plt.plot(path_x, path_y, color=color, linewidth=2, alpha=0.7,
                        label=f'{drone_id}')
        
        plt.xlabel("东向坐标 (米)")
        plt.ylabel("北向坐标 (米)")
        plt.title(f"无人机配送路径规划（{len(demand_points)}个需求点，{solver.solution['drones_used']}架无人机）")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        # 保存图片
        plt.savefig(output_dir / "delivery_paths.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "delivery_paths.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 路径图已保存 (PNG/PDF)")
        
    except Exception as e:
        print(f"  ✗ 可视化失败: {e}")
    
    print(f"\n所有结果已保存至: {output_dir}")
    return output_dir


# ============================================================================
# 6. 主程序
# ============================================================================

def main():
    """主程序 - 基于真实数据（修复状态判断）"""
    
    print("="*70)
    print("无人机配送路径规划系统 - CPLEX精确求解（基于真实数据）")
    print("="*70)
    
    try:
        # ===== 1. 读取真实数据 =====
        print("\n【步骤1】读取真实数据文件")
        hospitals, residences = read_building_data("target_area.xlsx")
        stations_df = read_station_data("latest_location.xlsx")
        
        # ===== 2. 生成规划点 =====
        print("\n【步骤2】生成供给点、需求点和站点")
        supply_points, demand_points, station_points, demand_weights = create_points_from_data(
            hospitals=hospitals,
            residences=residences,
            stations_df=stations_df,
            num_supply=3,
            num_demand=6,
            num_stations=3
        )
        
        # ===== 3. 创建无人机 =====
        print("\n【步骤3】创建无人机")
        drones = create_drones(
            station_points=station_points,
            drones_per_station=6,
            max_payload=10.0,
            max_range=20000.0
        )
        
        # ===== 4. 计算距离矩阵 =====
        print("\n【步骤4】基于真实坐标计算距离矩阵")
        dist_info = calculate_distance_matrix(
            supply_points=supply_points,
            demand_points=demand_points,
            station_points=station_points
        )
        
        # ===== 5. CPLEX求解 =====
        print("\n【步骤5】CPLEX优化求解")
        solver = CplexDroneSolver(
            supply_points=supply_points,
            demand_points=demand_points,
            station_points=station_points,
            demand_weights=demand_weights,
            drones=drones,
            dist_info=dist_info
        )
        
        # 构建模型并求解
        solver.build_model()
        solution = solver.solve(time_limit=300)
        
        # 打印结果
        solver.print_solution()
        
        # ===== 6. 保存结果 =====
        if solution and solution['drones_used'] > 0:
            print("\n【步骤6】保存结果和可视化")
            save_and_visualize(
                supply_points=supply_points,
                demand_points=demand_points,
                station_points=station_points,
                dist_info=dist_info,
                solver=solver
            )
        else:
            print("\n【步骤6】无可行解，跳过结果保存")
        
        print("\n" + "="*70)
        print("程序执行完成！")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n错误: 找不到数据文件 - {e}")
        print("\n请确保以下文件在当前目录：")
        print("  1. target_area.xlsx  - 建筑数据（包含医院和居住区）")
        print("  2. latest_location.xlsx - 无人机站点数据")
        
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()