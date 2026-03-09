"""
端到端 pipeline runner — 串联 Module 1 → Module 2 → Module 3。

支持两种运行模式:
  --offline   不调用 LLM，使用规则数据跑通全流程
  (默认)      调用 LLM API 进行对话生成 + context extraction + weight adjustment

Module 1 数据来源（优先级由高到低）:
  1. --csv + --stations  从 demand_events_5min.csv 生成对话（推荐）
  2. --dialogues         直接加载已有 JSONL 对话文件（兼容旧流程）
"""

import argparse
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from typing import Dict, List, Optional

from drone_pipeline.pipeline.context_extractor import (
    extract_all_demands,
    extract_demands_offline,
    group_by_time_window,
)
from drone_pipeline.pipeline.weight_adjuster import adjust_weights, adjust_weights_offline


# ============================================================================
# 从 Module 2 的需求构建求解器输入
# ============================================================================

def _demands_to_solver_inputs(demands: List[Dict]) -> tuple:
    """从结构化需求构建 supply_points, demand_points, demand_weights。

    注意：station_points 需要外部提供（来自站点数据文件）。
    """
    from drone_pipeline.utils.drone_cplex_real_data import Point

    supply_set = {}
    demand_points = []
    demand_weights = []

    for d in demands:
        origin = d.get("origin", {})
        ofid = origin.get("fid")
        if ofid and ofid not in supply_set:
            coords = origin.get("coords", [0, 0])
            supply_set[ofid] = Point(
                id=f"S_{ofid}",
                lon=coords[0],
                lat=coords[1],
                alt=50.0,
                type="supply",
            )

        dest = d.get("destination", {})
        dfid = dest.get("fid")
        coords = dest.get("coords", [0, 0])
        demand_points.append(Point(
            id=f"D_{dfid}",
            lon=coords[0],
            lat=coords[1],
            alt=50.0,
            type="demand",
        ))

        cargo = d.get("cargo", {})
        demand_weights.append(cargo.get("weight_kg", 2.0))

    supply_points = list(supply_set.values())
    return supply_points, demand_points, demand_weights


def _create_mock_stations(n: int = 3) -> list:
    """创建 mock 站点（当没有站点数据文件时使用）。"""
    from drone_pipeline.utils.drone_cplex_real_data import Point

    stations = [
        Point(id="L1", lon=113.85, lat=22.68, alt=50.0, type="station"),
        Point(id="L2", lon=113.92, lat=22.68, alt=50.0, type="station"),
        Point(id="L3", lon=113.82, lat=22.67, alt=50.0, type="station"),
    ]
    return stations[:n]


# ============================================================================
# Pipeline Runner
# ============================================================================

def run_pipeline(
    output_dir: str,
    # Module 1 数据来源
    csv_path: Optional[str] = None,
    stations_path: Optional[str] = None,
    dialogue_path: Optional[str] = None,   # 兼容旧接口
    # 运行模式
    offline: bool = False,
    # LLM
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    # Module 1 参数
    n_events: Optional[int] = None,
    time_slots: Optional[List[int]] = None,
    base_date: str = "2024-03-15",
    dialogue_batch_size: int = 5,
    # Module 2 参数
    window_minutes: int = 5,
    # Solver 参数
    time_limit: int = 300,
    max_drones_per_station: int = 6,
    max_payload: float = 25.0,
    max_range: float = 40000.0,
    skip_solver: bool = False,
):
    """端到端运行 pipeline（Module 1 → Module 2 → Module 3）。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = None

    if not offline:
        from openai import OpenAI
        base = api_base or os.getenv("LLMOPT_API_BASE_URL", "http://35.220.164.252:3888/v1/")
        key = api_key or os.getenv("LLMOPT_API_KEY")
        if not key:
            raise ValueError("需要 API key: 设置 LLMOPT_API_KEY 或 --api-key")
        client = OpenAI(base_url=base, api_key=key)

    # ----------------------------------------------------------------
    # Step 1: Module 1 — 生成对话
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Step 1: 对话生成 (Module 1)")
    print("=" * 60)

    if csv_path and stations_path:
        # 新路径：从 demand_events_5min.csv 生成对话
        from drone_pipeline.pipeline.dialogue_generator import generate_dialogues
        dialogues = generate_dialogues(
            csv_path=csv_path,
            xlsx_path=stations_path,
            offline=offline,
            client=client,
            model=model,
            base_date=base_date,
            n_events=n_events,
            time_slots=time_slots,
            temperature=temperature,
            batch_size=dialogue_batch_size,
        )
        # 保存生成的对话（方便复用）
        dlg_out = output_dir / "generated_dialogues.jsonl"
        from drone_pipeline.pipeline.dialogue_generator import save_dialogues
        save_dialogues(dialogues, str(dlg_out))
    elif dialogue_path:
        # 兼容旧路径：直接加载 JSONL 对话文件
        with open(dialogue_path, "r", encoding="utf-8") as f:
            dialogues = [json.loads(l.strip()) for l in f if l.strip()]
        print(f"  加载 {len(dialogues)} 条对话（来自文件: {dialogue_path}）")
    else:
        raise ValueError(
            "必须提供 csv_path + stations_path（从 CSV 生成对话）"
            " 或 dialogue_path（直接加载 JSONL 文件）"
        )

    print(f"  共 {len(dialogues)} 条对话")

    # ----------------------------------------------------------------
    # Step 2: Module 2 — 按窗口提取需求
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Context Extraction (Module 2)")
    print("=" * 60)

    if offline:
        window_results = extract_demands_offline(dialogues, window_minutes)
    else:
        window_results = extract_all_demands(
            dialogues, client, model, window_minutes, temperature,
        )

    demands_path = output_dir / "extracted_demands.json"
    with open(demands_path, "w", encoding="utf-8") as f:
        json.dump(window_results, f, ensure_ascii=False, indent=2)
    print(f"  需求保存至 {demands_path}")

    total_demands = sum(len(w.get("demands", [])) for w in window_results)
    print(f"  共提取 {total_demands} 条需求，分布于 {len(window_results)} 个窗口")

    # ----------------------------------------------------------------
    # Step 3: Module 3 — 逐窗口调整权重 + 求解
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Weight Adjustment + Solve (Module 3)")
    print("=" * 60)

    all_solutions = []

    for w_idx, window in enumerate(window_results):
        tw = window.get("time_window", f"window_{w_idx}")
        demands = window.get("demands", [])

        if not demands:
            print(f"\n  窗口 {tw}: 无需求，跳过")
            continue

        print(f"\n  ---- 窗口 {tw}: {len(demands)} 条需求 ----")

        # 3a: 权重调整
        if offline:
            weight_config = adjust_weights_offline(demands)
        else:
            weight_config = adjust_weights(demands, client, model)

        weight_config["time_window"] = tw

        # 保存权重配置
        wc_path = output_dir / f"weight_config_{w_idx}.json"
        with open(wc_path, "w", encoding="utf-8") as f:
            json.dump(weight_config, f, ensure_ascii=False, indent=2)

        if skip_solver:
            print(f"  跳过求解 (--skip-solver)")
            all_solutions.append({
                "time_window": tw,
                "weight_config": weight_config,
                "feasible_demands": demands,
                "n_demands_total": len(demands),
                "n_demands_filtered": 0,
                "solution": None,
                "n_supply": 0,
            })
            continue

        # 3b: 构建求解器输入 — 过滤超载需求
        feasible_demands = [
            d for d in demands
            if d.get("cargo", {}).get("weight_kg", 0) <= max_payload
        ]
        skipped = len(demands) - len(feasible_demands)
        if skipped:
            print(f"  跳过 {skipped} 条超载需求 (>{max_payload}kg)")
        if not feasible_demands:
            print(f"  无可行需求，跳过")
            all_solutions.append({
                "time_window": tw,
                "weight_config": weight_config,
                "feasible_demands": [],
                "n_demands_total": len(demands),
                "n_demands_filtered": skipped,
                "solution": None,
                "n_supply": 0,
            })
            continue

        # 同步更新 weight_config 中的 demand_configs
        feasible_ids = {d["demand_id"] for d in feasible_demands}
        weight_config["demand_configs"] = [
            dc for dc in weight_config.get("demand_configs", [])
            if dc["demand_id"] in feasible_ids
        ]

        supply_pts, demand_pts, d_weights = _demands_to_solver_inputs(feasible_demands)
        station_pts = _create_mock_stations()

        if not supply_pts or not demand_pts:
            print(f"  供给点或需求点为空，跳过")
            continue

        print(f"  供给点 {len(supply_pts)}, 需求点 {len(demand_pts)}, "
              f"站点 {len(station_pts)}")

        from drone_pipeline.utils.drone_cplex_real_data import create_drones, calculate_distance_matrix
        from drone_pipeline.utils.drone_solver import build_solver_from_pipeline

        drones = create_drones(
            station_pts,
            drones_per_station=max_drones_per_station,
            max_payload=max_payload,
            max_range=max_range,
        )

        dist_info = calculate_distance_matrix(supply_pts, demand_pts, station_pts)

        solver = build_solver_from_pipeline(
            demands=demands,
            weight_config=weight_config,
            supply_points=supply_pts,
            demand_points=demand_pts,
            station_points=station_pts,
            demand_weights=d_weights,
            drones=drones,
            dist_info=dist_info,
        )

        solver.build_model()

        try:
            solution = solver.solve(time_limit=time_limit)
            solver.print_solution()
        except Exception as e:
            print(f"  求解失败: {e}")
            solution = None

        all_solutions.append({
            "time_window": tw,
            "weight_config": weight_config,
            "feasible_demands": feasible_demands,
            "n_demands_total": len(demands),
            "n_demands_filtered": skipped,
            "solution": solution,
            "n_supply": len(supply_pts),
        })

    # ----------------------------------------------------------------
    # 保存汇总结果（含完整 eval 字段）
    # ----------------------------------------------------------------
    summary_path = output_dir / "pipeline_results.json"
    serializable = []
    for s in all_solutions:
        wc = s["weight_config"]
        sol = s["solution"]
        feasible_demands = s.get("feasible_demands", [])
        n_supply = s.get("n_supply", 0)

        entry: Dict = {
            "time_window": s["time_window"],
            "n_demands_extracted": s.get("n_demands_total", len(feasible_demands)),
            "n_demands_feasible": len(feasible_demands),
            "n_demands_filtered": s.get("n_demands_filtered", 0),
            "has_solution": sol is not None,
            "global_weights": wc.get("global_weights", {}),
            "n_supplementary_constraints": len(wc.get("supplementary_constraints", [])),
        }

        if sol:
            cruise_speed = 15.0  # m/s default
            total_dist = sol.get("total_distance", 0.0)

            entry.update({
                "solve_status": sol.get("solve_status", "unknown"),
                "solve_time_s": sol.get("solve_time_s", 0.0),
                "objective_value": sol.get("objective_value", total_dist),
                "drones_used": sol.get("drones_used", 0),
                "total_distance_m": round(total_dist, 2),
                "total_estimated_time_s": round(total_dist / cruise_speed, 1),
                "cruise_speed_ms": cruise_speed,
            })

            # Build demand_id → weight config map
            dc_map = {
                dc["demand_id"]: dc
                for dc in wc.get("demand_configs", [])
            }

            # Map served demands: local_demand_idx → (drone_id, delivery_time_s)
            served_map: Dict[int, str] = {}
            delivery_time_map: Dict[int, float] = {}
            for assign in sol.get("assignments", []):
                ddt = assign.get("demand_delivery_times_s", {})
                for node_idx in assign.get("demand_indices", []):
                    local_idx = node_idx - n_supply
                    served_map[local_idx] = assign["drone_id"]
                    if node_idx in ddt:
                        delivery_time_map[local_idx] = ddt[node_idx]

            # Per-demand results
            per_demand = []
            for i, fd in enumerate(feasible_demands):
                did = fd.get("demand_id", f"D{i+1}")
                dc = dc_map.get(did, {})
                cargo = fd.get("cargo", {})
                origin = fd.get("origin", {})
                dest = fd.get("destination", {})
                vuln = fd.get("priority_evaluation_signals", {}).get(
                    "population_vulnerability", {}
                )
                per_demand.append({
                    "demand_id": did,
                    "source_dialogue_id": fd.get("source_dialogue_id"),
                    "demand_tier": fd.get("demand_tier", cargo.get("demand_tier", "")),
                    "cargo_type": cargo.get("type", ""),
                    "cargo_type_cn": cargo.get("type_cn", ""),
                    "weight_kg": cargo.get("weight_kg", 0.0),
                    "temperature_sensitive": cargo.get("temperature_sensitive", False),
                    "alpha": dc.get("alpha", 1.0),
                    "beta": dc.get("beta", 1.0),
                    "priority": dc.get("priority", 3),
                    "llm_reasoning": dc.get("reasoning", ""),
                    "elderly_involved": vuln.get("elderly_involved", False),
                    "vulnerable_community": vuln.get("vulnerable_community", False),
                    "time_sensitivity": fd.get("priority_evaluation_signals", {}).get(
                        "time_sensitivity", ""
                    ),
                    "requester_role": fd.get("priority_evaluation_signals", {}).get(
                        "requester_role", ""
                    ),
                    "nearby_facility": fd.get("priority_evaluation_signals", {}).get(
                        "nearby_critical_facility", ""
                    ),
                    "is_served": i in served_map,
                    "assigned_drone": served_map.get(i),
                    "delivery_time_s": delivery_time_map.get(i),
                    "delivery_time_min": round(delivery_time_map[i] / 60, 2) if i in delivery_time_map else None,
                    "origin_fid": origin.get("fid"),
                    "origin_coords": origin.get("coords", []),
                    "dest_fid": dest.get("fid"),
                    "dest_coords": dest.get("coords", []),
                    "dest_type": dest.get("type", ""),
                })

            n_served = sum(1 for d in per_demand if d["is_served"])
            entry["n_demands_served"] = n_served
            entry["service_rate"] = round(n_served / len(feasible_demands), 3) if feasible_demands else 0.0
            entry["per_demand_results"] = per_demand

            # Per-drone details
            per_drone = []
            paths = sol.get("paths", [])
            for i_a, assign in enumerate(sol.get("assignments", [])):
                local_idxs = [
                    node_idx - n_supply
                    for node_idx in assign.get("demand_indices", [])
                ]
                total_weight = sum(
                    feasible_demands[i].get("cargo", {}).get("weight_kg", 0.0)
                    for i in local_idxs
                    if 0 <= i < len(feasible_demands)
                )
                demand_ids_served = [
                    feasible_demands[i].get("demand_id", f"D{i+1}")
                    for i in local_idxs
                    if 0 <= i < len(feasible_demands)
                ]
                per_drone.append({
                    "drone_id": assign.get("drone_id"),
                    "station_name": assign.get("station_name", ""),
                    "station_id": assign.get("station_id", -1),
                    "supply_idx": assign.get("supply_idx", -1),
                    "n_demands_served": len(demand_ids_served),
                    "total_weight_kg": round(total_weight, 3),
                    "demand_ids_served": demand_ids_served,
                    "path_str": assign.get("path_str", ""),
                    "path_labels": assign.get("path_labels", []),
                    "path_node_indices": paths[i_a] if i_a < len(paths) else [],
                    "total_mission_time_s": assign.get("total_mission_time_s"),
                    "total_mission_time_min": round(assign["total_mission_time_s"] / 60, 2) if assign.get("total_mission_time_s") else None,
                })

            entry["per_drone_details"] = per_drone

        serializable.append(entry)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Pipeline 完成！结果保存至 {output_dir}")
    print(f"{'=' * 60}")

    return all_solutions


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Drone Delivery Pipeline Runner (Module 1 → 2 → 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 离线全流程（从 CSV 生成对话，跳过求解器）
  python pipeline/run_pipeline.py --offline --skip-solver

  # 指定 CSV + 站点文件，仅处理前 20 条事件
  python pipeline/run_pipeline.py --offline --skip-solver --n-events 20

  # 在线 LLM 模式
  python pipeline/run_pipeline.py --api-key YOUR_KEY --n-events 10
""",
    )

    # Module 1 数据来源（二选一）
    src_group = parser.add_argument_group("Module 1 数据来源")
    src_group.add_argument(
        "--csv", type=str,
        default=str(PROJECT_ROOT / "data" / "seed" / "demand_events_5min.csv"),
        help="demand_events_5min.csv 路径（与 --stations 一起使用）",
    )
    src_group.add_argument(
        "--stations", type=str,
        default=str(PROJECT_ROOT / "data" / "seed" / "latest_location.xlsx"),
        help="latest_location.xlsx 站点数据路径",
    )
    src_group.add_argument(
        "--dialogues", type=str, default=None,
        help="[兼容] 直接加载已有 JSONL 对话文件（不指定时使用 --csv + --stations）",
    )

    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "drone" / "pipeline_output"),
        help="输出目录",
    )
    parser.add_argument("--offline", action="store_true", help="离线模式，不调用 LLM")
    parser.add_argument("--skip-solver", action="store_true", help="跳过 CPLEX 求解")
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--window", type=int, default=5, help="时间窗口（分钟），默认 5 以匹配 demand_events_5min.csv 粒度")
    parser.add_argument("--time-limit", type=int, default=300, help="求解器时间限制（秒）")
    parser.add_argument("--n-events", type=int, default=None, help="最多处理的事件数")
    parser.add_argument("--time-slots", type=int, nargs="+", default=None,
                        help="仅处理指定 time_slot（空格分隔）")
    parser.add_argument("--base-date", type=str, default="2024-03-15", help="基准日期")
    parser.add_argument("--dialogue-batch-size", type=int, default=5,
                        help="LLM 对话生成批大小")
    args = parser.parse_args()

    # 确定 Module 1 数据来源
    csv_path = None
    stations_path = None
    if args.dialogues:
        # 显式指定了旧格式文件
        dialogue_path = args.dialogues
    else:
        # 默认从 CSV 生成
        csv_path = args.csv
        stations_path = args.stations
        dialogue_path = None

    run_pipeline(
        output_dir=args.output_dir,
        csv_path=csv_path,
        stations_path=stations_path,
        dialogue_path=dialogue_path,
        offline=args.offline,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        n_events=args.n_events,
        time_slots=args.time_slots,
        base_date=args.base_date,
        dialogue_batch_size=args.dialogue_batch_size,
        window_minutes=args.window,
        time_limit=args.time_limit,
        skip_solver=args.skip_solver,
    )


if __name__ == "__main__":
    main()
