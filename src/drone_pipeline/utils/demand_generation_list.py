import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 设置随机种子以保证可重复性
random.seed(42)
np.random.seed(42)

# ========== 配置参数 ==========
INPUT_FILE = "target_area.xlsx"          # 输入建筑数据文件
OUTPUT_FILE = "daily_demands.csv"        # 输出需求文件

# 时间窗设置
TIME_WINDOW_MINUTES = 5                   # 每个时间窗长度（分钟）
NUM_WINDOWS = 24 * 60 // TIME_WINDOW_MINUTES  # 一天的时间窗数量（288个）

# 每个时间窗的需求数量范围
DEMANDS_PER_WINDOW_MIN = 4
DEMANDS_PER_WINDOW_MAX = 10

# 需求类型比例（医用 vs 普通）
MEDICAL_RATIO = 0.2        # 医用需求占20%
COMMERCIAL_RATIO = 0.8      # 普通需求占80%（剩余比例）

# 优先级分布（1为最高，4为最低）
PRIORITIES = [1, 2, 3, 4]
PRIORITY_PROBS = [0.25, 0.25, 0.25, 0.25]  # 均匀分布

# 重量范围（kg）
WEIGHT_MIN = 5.0
WEIGHT_MAX = 30.0

# 供给点选取数量（每种类型选取前N个作为可用供给点）
NUM_SUPPLY_MEDICAL = 5      # 医用供给点数量
NUM_SUPPLY_COMMERCIAL = 5   # 商业供给点数量

# 目标函数权重（全局参数，可在模拟时使用）
DISTANCE_WEIGHT = 1.0
NOISE_WEIGHT = 0.5

# ========== 读取建筑数据 ==========
def load_buildings(file_path):
    """读取建筑数据，返回医疗、商业、居住用地DataFrame"""
    df = pd.read_excel(file_path)
    print(f"读取建筑数据，共 {len(df)} 条记录")
    print("用地类型分布：")
    print(df['type'].value_counts())

    # 分离各类用地
    medical = df[df['type'] == '医疗卫生用地'].copy()
    residential = df[df['type'] == '居住用地'].copy()

    # 处理商业用地：优先从 '商业用地' 类型中获取，若不存在则从其他类型中选取（此处假设从医疗中拆分一部分作为演示）
    commercial = df[df['type'] == '商业用地'].copy()
    if len(commercial) == 0:
        print("警告：未找到'商业用地'类型，将从医疗卫生用地中随机分配一部分作为商业供给点。")
        # 从医疗用地中随机抽取一半作为商业（仅用于演示，实际应使用真实数据）
        if len(medical) >= 2:
            medical_indices = medical.index.tolist()
            random.shuffle(medical_indices)
            split_point = len(medical_indices) // 2
            commercial_indices = medical_indices[:split_point]
            commercial = medical.loc[commercial_indices].copy()
            commercial['type'] = '商业用地'  # 修改类型
            medical = medical.drop(commercial_indices)  # 剩余仍为医疗
        else:
            # 若医疗不足，则创建虚拟商业点（用第一个医疗点）
            commercial = medical.head(1).copy()
            commercial['type'] = '商业用地'

    print(f"医疗用地数量: {len(medical)}")
    print(f"商业用地数量: {len(commercial)}")
    print(f"居住用地数量: {len(residential)}")

    return medical, commercial, residential

# ========== 构建供给点列表 ==========
def build_supply_points(medical_df, commercial_df, num_medical, num_commercial):
    """从医疗和商业用地中选取指定数量的供给点，返回列表，每个元素为字典包含 fid, lon, lat, type"""
    supply_points = []

    # 医疗供给点
    medical_selected = medical_df.head(num_medical)
    for idx, row in medical_selected.iterrows():
        supply_points.append({
            'fid': f"MED_{idx}",          # 用索引作为fid，可自定义
            'lon': float(row['经度']),
            'lat': float(row['纬度']),
            'type': '医疗'
        })

    # 商业供给点
    commercial_selected = commercial_df.head(num_commercial)
    for idx, row in commercial_selected.iterrows():
        supply_points.append({
            'fid': f"COM_{idx}",
            'lon': float(row['经度']),
            'lat': float(row['纬度']),
            'type': '商业'
        })

    print(f"共选取 {len(supply_points)} 个供给点：医疗 {len(medical_selected)}，商业 {len(commercial_selected)}")
    return supply_points

# ========== 构建需求点列表 ==========
def build_demand_points(residential_df):
    """从居住用地中构建需求点列表，每个元素为字典包含 fid, lon, lat"""
    demand_points = []
    for idx, row in residential_df.iterrows():
        demand_points.append({
            'fid': f"DEM_{idx}",
            'lon': float(row['经度']),
            'lat': float(row['纬度'])
        })
    print(f"共 {len(demand_points)} 个潜在需求点")
    return demand_points

# ========== 生成一天的需求 ==========
def generate_daily_demands(supply_points, demand_points):
    """生成一天288个时间窗的需求，返回需求列表"""
    demands = []
    window_duration = TIME_WINDOW_MINUTES / 60.0  # 小时

    for window_idx in range(NUM_WINDOWS):
        # 当前时间窗的开始时间（小时）
        time_hour = window_idx * window_duration

        # 随机生成该窗内的需求数量
        num_demands = random.randint(DEMANDS_PER_WINDOW_MIN, DEMANDS_PER_WINDOW_MAX)

        for d in range(num_demands):
            # 随机选择需求点
            demand = random.choice(demand_points)

            # 随机决定需求类型（医用/普通）
            if random.random() < MEDICAL_RATIO:
                demand_type = '医疗'
                # 从医疗供给点中随机选一个
                suitable_supplies = [s for s in supply_points if s['type'] == '医疗']
            else:
                demand_type = '普通'
                suitable_supplies = [s for s in supply_points if s['type'] == '商业']

            # 若没有合适的供给点，则跳过（理论上应有）
            if not suitable_supplies:
                continue

            supply = random.choice(suitable_supplies)

            # 随机生成优先级
            priority = random.choices(PRIORITIES, weights=PRIORITY_PROBS)[0]

            # 随机生成重量
            weight = round(random.uniform(WEIGHT_MIN, WEIGHT_MAX), 1)

            # 生成唯一ID：时间窗序号+该窗内序号
            unique_id = f"DEM_{window_idx:03d}_{d:02d}"

            demands.append({
                'time': round(time_hour, 4),          # 小时，保留4位小数
                'demand_fid': demand['fid'],
                'demand_lon': demand['lon'],
                'demand_lat': demand['lat'],
                'priority': priority,
                'supply_fid': supply['fid'],
                'supply_lon': supply['lon'],
                'supply_lat': supply['lat'],
                'supply_type': supply['type'],
                'weight': weight,
                'unique_id': unique_id
            })

    print(f"生成需求总数: {len(demands)}")
    return demands

# ========== 主程序 ==========
def main():
    # 1. 读取建筑数据
    medical_df, commercial_df, residential_df = load_buildings(INPUT_FILE)

    # 2. 构建供给点
    supply_points = build_supply_points(medical_df, commercial_df, NUM_SUPPLY_MEDICAL, NUM_SUPPLY_COMMERCIAL)

    # 3. 构建需求点
    demand_points = build_demand_points(residential_df)

    # 4. 生成一天的需求
    demands = generate_daily_demands(supply_points, demand_points)

    # 5. 转换为DataFrame并保存CSV
    df_demands = pd.DataFrame(demands)

    # 添加全局权重信息（可作为注释或单独列）
    df_demands.attrs['distance_weight'] = DISTANCE_WEIGHT
    df_demands.attrs['noise_weight'] = NOISE_WEIGHT

    # 保存到CSV
    df_demands.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"需求数据已保存至 {OUTPUT_FILE}")

    # 显示前几行示例
    print("\n前10条需求示例：")
    print(df_demands.head(10).to_string())

    # 输出统计信息
    print(f"\n需求类型统计：")
    supply_type_counts = df_demands['supply_type'].value_counts()
    for t, cnt in supply_type_counts.items():
        print(f"  {t}: {cnt} ({cnt/len(df_demands)*100:.1f}%)")

    print(f"\n优先级分布：")
    priority_counts = df_demands['priority'].value_counts().sort_index()
    for p, cnt in priority_counts.items():
        print(f"  优先级{p}: {cnt} ({cnt/len(df_demands)*100:.1f}%)")

    print(f"\n全局权重：距离权重 = {DISTANCE_WEIGHT}, 噪声权重 = {NOISE_WEIGHT}")

if __name__ == "__main__":
    main()