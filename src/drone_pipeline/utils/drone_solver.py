"""
Module 3b: ParameterizedDroneSolver — 在 CplexDroneSolver 基础上扩展：

1. 多目标函数  U = w_d·D + (alpha_d · w_t)·T_d + (beta_d · w_r)·R_d
2. 每需求点 alpha/beta/priority
3. 补充约束插槽（噪音规避、限速覆盖等）
4. 时间估算  T = D / speed
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    SolverFactory, Binary, NonNegativeReals,
    minimize, value, TerminationCondition,
)

from drone_pipeline.utils.drone_cplex_real_data import (
    CplexDroneSolver, Point, Drone,
    calculate_distance_matrix,
)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class DemandConfig:
    """单个需求点的 LLM 分配参数。"""
    demand_id: str
    alpha: float = 1.0    # 时间权重缩放
    beta: float = 1.0     # 风险权重缩放
    priority: int = 3     # 1=最高, 5=最低
    reasoning: str = ""


@dataclass
class SupplementaryConstraint:
    """补充约束（由 LLM 识别）。"""
    type: str                          # noise_avoidance | speed_override | no_fly_zone
    description: str = ""
    affected_zone: Optional[Dict] = None   # {"center": [lng, lat], "radius_m": ...}
    affected_demand: Optional[str] = None
    time_window: Optional[List[str]] = None


@dataclass
class GlobalWeights:
    """全局目标函数权重。"""
    w_distance: float = 1.0
    w_time: float = 1.0
    w_risk: float = 1.0


# ============================================================================
# 参数化求解器
# ============================================================================

class ParameterizedDroneSolver(CplexDroneSolver):
    """扩展 CplexDroneSolver，支持 LLM 驱动的多目标优化。"""

    DEFAULT_SPEED_MS = 15.0  # 默认巡航速度 m/s

    def __init__(
        self,
        supply_points: List[Point],
        demand_points: List[Point],
        station_points: List[Point],
        demand_weights: List[float],
        drones: List[Drone],
        dist_info: Dict,
        *,
        demand_configs: Optional[List[DemandConfig]] = None,
        global_weights: Optional[GlobalWeights] = None,
        supplementary_constraints: Optional[List[SupplementaryConstraint]] = None,
        cruise_speed: float = 15.0,
    ):
        super().__init__(
            supply_points, demand_points, station_points,
            demand_weights, drones, dist_info,
        )

        # 默认值：如果没传 demand_configs，所有需求使用 alpha=1, beta=1, priority=3
        if demand_configs is None:
            demand_configs = [
                DemandConfig(demand_id=f"D{i+1}", alpha=1.0, beta=1.0, priority=3)
                for i in range(self.n_demand)
            ]
        self.demand_configs = demand_configs

        self.global_weights = global_weights or GlobalWeights()
        self.supplementary_constraints = supplementary_constraints or []
        self.cruise_speed = cruise_speed

        # 构建需求索引 → config 映射
        self._demand_cfg_map: Dict[int, DemandConfig] = {}
        demand_indices = self.indices["demand"]
        for idx, node_idx in enumerate(demand_indices):
            if idx < len(self.demand_configs):
                self._demand_cfg_map[node_idx] = self.demand_configs[idx]

        print(f"  参数化扩展:")
        print(f"    - 全局权重: w_d={self.global_weights.w_distance}, "
              f"w_t={self.global_weights.w_time}, w_r={self.global_weights.w_risk}")
        print(f"    - 巡航速度: {self.cruise_speed} m/s")
        print(f"    - 补充约束: {len(self.supplementary_constraints)} 条")

    # ------------------------------------------------------------------
    # 风险评估
    # ------------------------------------------------------------------

    def _compute_risk_matrix(self) -> np.ndarray:
        """计算每条边的风险分数。

        当前使用简化模型：高人口密度区域 + 补充约束区域 → 更高风险。
        """
        n = self.n_nodes
        risk = np.ones((n, n))

        for sc in self.supplementary_constraints:
            if sc.type in ("noise_avoidance", "no_fly_zone") and sc.affected_zone:
                center = sc.affected_zone.get("center", [0, 0])
                radius = sc.affected_zone.get("radius_m", 300)
                all_points = (
                    self.supply_points + self.demand_points + self.station_points
                )
                for i, p in enumerate(all_points):
                    dist_to_center = self._haversine_m(
                        p.lon, p.lat, center[0], center[1]
                    )
                    if dist_to_center < radius:
                        for j in range(n):
                            risk[i, j] += 5.0
                            risk[j, i] += 5.0

        return risk

    @staticmethod
    def _haversine_m(lon1, lat1, lon2, lat2) -> float:
        R = 6371000
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # ------------------------------------------------------------------
    # 构建多目标模型
    # ------------------------------------------------------------------

    def build_model(self):
        """构建带有多目标和补充约束的 MILP 模型。"""
        # 先调用父类构建基础模型
        super().build_model()
        model = self.model

        gw = self.global_weights
        risk_matrix = self._compute_risk_matrix()
        speed = self.cruise_speed

        # ---- 删除父类的纯距离目标 ----
        model.del_component(model.obj)

        # ---- 新参数: 风险矩阵 ----
        def risk_init(m, i, j):
            return float(risk_matrix[i, j])
        model.risk = Param(model.N, model.N, initialize=risk_init)

        # ---- alpha/beta per demand ----
        def alpha_init(m, d):
            cfg = self._demand_cfg_map.get(d)
            return cfg.alpha if cfg else 1.0
        model.alpha = Param(model.D, initialize=alpha_init)

        def beta_init(m, d):
            cfg = self._demand_cfg_map.get(d)
            return cfg.beta if cfg else 1.0
        model.beta = Param(model.D, initialize=beta_init)

        def priority_init(m, d):
            cfg = self._demand_cfg_map.get(d)
            return float(cfg.priority if cfg else 3)
        model.priority_w = Param(model.D, initialize=priority_init)

        # ---- 多目标函数 ----
        # U = Σ_u Σ_{i,j} x[u,i,j] · (
        #       w_d · dist[i,j]
        #     + w_t · (dist[i,j]/speed) · Σ_d (alpha_d · y[u,d] / |D|)
        #     + w_r · risk[i,j] · Σ_d (beta_d · y[u,d] / |D|)
        # )
        # 简化: 对每条使用的边，用该无人机所服务需求的平均 alpha/beta 来加权。
        # 由于 alpha/beta 与 y 耦合是非线性的，这里采用近似线性化：
        # 按 priority 加权的目标。

        n_demand = self.n_demand

        def multi_obj_rule(m):
            dist_term = gw.w_distance * sum(
                m.dist[i, j] * m.x[u, i, j]
                for u in m.U for i in m.N for j in m.N
            )

            time_term = gw.w_time * sum(
                (m.dist[i, j] / speed)
                * m.x[u, i, j]
                * (1.0 / max(n_demand, 1))
                * sum(m.alpha[d] * m.y[u, d] for d in m.D)
                for u in m.U for i in m.N for j in m.N
            )

            risk_term = gw.w_risk * sum(
                m.risk[i, j]
                * m.x[u, i, j]
                * (1.0 / max(n_demand, 1))
                * sum(m.beta[d] * m.y[u, d] for d in m.D)
                for u in m.U for i in m.N for j in m.N
            )

            return dist_term + time_term + risk_term

        model.obj = Objective(rule=multi_obj_rule, sense=minimize)

        # ---- 补充约束: 禁飞区 ----
        self._add_supplementary_constraints(model)

        self.model = model
        return model

    def _add_supplementary_constraints(self, model):
        """注入 LLM 识别的补充约束。"""
        all_points = self.supply_points + self.demand_points + self.station_points
        forbidden_edges = set()

        for sc in self.supplementary_constraints:
            if sc.type in ("no_fly_zone", "noise_avoidance") and sc.affected_zone:
                center = sc.affected_zone.get("center", [0, 0])
                radius = sc.affected_zone.get("radius_m", 300)

                for i, p in enumerate(all_points):
                    d = self._haversine_m(p.lon, p.lat, center[0], center[1])
                    if d < radius:
                        for j in range(len(all_points)):
                            if i != j:
                                forbidden_edges.add((i, j))

        if forbidden_edges:
            edges_list = sorted(forbidden_edges)
            edge_set = Set(initialize=range(len(edges_list)))
            model.no_fly_edges = edge_set

            edge_map = {idx: (i, j) for idx, (i, j) in enumerate(edges_list)}

            def no_fly_rule(m, u, e):
                i, j = edge_map[e]
                return m.x[u, i, j] == 0

            model.con_no_fly = Constraint(model.U, model.no_fly_edges, rule=no_fly_rule)
            print(f"    添加禁飞边约束: {len(edges_list)} 条")

    # ------------------------------------------------------------------
    # 求解后的额外统计
    # ------------------------------------------------------------------

    def print_solution(self):
        """打印求解结果 + 参数化信息。"""
        super().print_solution()

        if not self.solution or self.solution["drones_used"] == 0:
            return

        print(f"\n【参数化扩展信息】")
        print(f"  全局权重: w_d={self.global_weights.w_distance}, "
              f"w_t={self.global_weights.w_time}, w_r={self.global_weights.w_risk}")
        print(f"  补充约束: {len(self.supplementary_constraints)} 条")
        print(f"\n  需求配置:")
        for cfg in self.demand_configs:
            print(f"    {cfg.demand_id}: alpha={cfg.alpha}, beta={cfg.beta}, "
                  f"priority={cfg.priority} | {cfg.reasoning}")


# ============================================================================
# 工厂函数: 从 Module 2 + Module 3a 的 JSON 输出构建求解器
# ============================================================================

def build_solver_from_pipeline(
    demands: List[Dict],
    weight_config: Dict,
    supply_points: List[Point],
    demand_points: List[Point],
    station_points: List[Point],
    demand_weights: List[float],
    drones: List[Drone],
    dist_info: Dict,
    cruise_speed: float = 15.0,
) -> ParameterizedDroneSolver:
    """从 pipeline 的 JSON 数据构建 ParameterizedDroneSolver。"""

    gw_data = weight_config.get("global_weights", {})
    global_weights = GlobalWeights(
        w_distance=gw_data.get("w_distance", 1.0),
        w_time=gw_data.get("w_time", 1.0),
        w_risk=gw_data.get("w_risk", 1.0),
    )

    demand_configs = []
    for dc in weight_config.get("demand_configs", []):
        demand_configs.append(DemandConfig(
            demand_id=dc["demand_id"],
            alpha=dc.get("alpha", 1.0),
            beta=dc.get("beta", 1.0),
            priority=dc.get("priority", 3),
            reasoning=dc.get("reasoning", ""),
        ))

    supp_constraints = []
    for sc in weight_config.get("supplementary_constraints", []):
        supp_constraints.append(SupplementaryConstraint(
            type=sc.get("type", ""),
            description=sc.get("description", ""),
            affected_zone=sc.get("affected_zone"),
            affected_demand=sc.get("affected_demand"),
            time_window=sc.get("time_window"),
        ))

    return ParameterizedDroneSolver(
        supply_points=supply_points,
        demand_points=demand_points,
        station_points=station_points,
        demand_weights=demand_weights,
        drones=drones,
        dist_info=dist_info,
        demand_configs=demand_configs,
        global_weights=global_weights,
        supplementary_constraints=supp_constraints,
        cruise_speed=cruise_speed,
    )
