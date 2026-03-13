"""
端到端 workflow runner — 串联 Module 1 → Module 2 → Module 3。

支持两种运行模式:
  --offline   不调用 LLM，使用规则数据跑通全流程
  (默认)      调用 LLM API 进行对话生成 + context extraction + weight adjustment

Module 1 数据来源（优先级由高到低）:
  1. --csv + --stations  从 daily_demand_events.csv 生成对话（推荐）
  2. --dialogues         直接加载已有 JSONL 对话文件（兼容旧流程）
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from typing import Dict, List, Optional

from llm4fairrouting.llm.demand_extraction import (
    extract_all_demands,
    extract_demands_offline,
)
from llm4fairrouting.workflow.solver_adapter import (
    serialize_workflow_results,
    solve_windows_dynamically,
)
from llm4fairrouting.llm.priority_inference import (
    adjust_weights,
    adjust_weights_offline,
)
from llm4fairrouting.data.seed_paths import (
    BUILDING_DATA_FILENAME,
    BUILDING_DATA_PATH,
    DEMAND_EVENTS_FILENAME,
    DEMAND_EVENTS_PATH,
    STATION_DATA_FILENAME,
    STATION_DATA_PATH,
)


class _TeeStdout:
    """Simple tee for sys.stdout that also writes to a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
        for stream in self._streams:
            stream.flush()
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _build_run_dir(base_dir: Path, model: str, noise_weight: float) -> Path:
    """Create a timestamped run directory under *base_dir*.

    Format: ``<base_dir>/run_<YYYYMMDD_HHMMSS>_<model>_noise<w>/``
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = model.replace("/", "-")
    run_name = f"run_{ts}_{tag}_noise{noise_weight}"
    return base_dir / run_name


# ============================================================================
# Pipeline Runner
# ============================================================================

def run_workflow(
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
    # Solver 参数（默认值与 baseline demo 对齐）
    time_limit: int = 10,
    max_drones_per_station: int = 3,
    max_payload: float = 60.0,
    max_range: float = 200000.0,
    max_solver_stations: Optional[int] = 1,
    skip_solver: bool = False,
    # Noise 参数
    noise_weight: float = 0.5,
    building_path: Optional[str] = None,
    drone_speed: float = 60.0,
):
    """端到端运行 workflow（Module 1 → Module 2 → Module 3）。"""
    base_dir = Path(output_dir)
    run_dir = _build_run_dir(base_dir, model, noise_weight)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_stations_path = str(stations_path or STATION_DATA_PATH)

    # 将 stdout 同步写入终端和 log 文件
    log_path = run_dir / "workflow.log"
    original_stdout = sys.stdout
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _TeeStdout(original_stdout, log_file)

    try:
        # 记录命令行（如果是通过 CLI 调用）
        try:
            cmdline = " ".join(sys.argv)
            print(f"[COMMAND] {cmdline}")
        except Exception:
            pass

        weight_configs_dir = run_dir / "weight_configs"
        weight_configs_dir.mkdir(exist_ok=True)

        # Save run metadata for reproducibility
        run_meta = {
            "created_at": datetime.now().isoformat(),
            "model": model,
            "offline": offline,
            "noise_weight": noise_weight,
            "temperature": temperature,
            "window_minutes": window_minutes,
            "time_limit": time_limit,
            "max_drones_per_station": max_drones_per_station,
            "max_payload": max_payload,
            "max_range": max_range,
            "max_solver_stations": max_solver_stations,
            "building_path": building_path or str(BUILDING_DATA_PATH),
            "drone_speed": drone_speed,
            "csv_path": csv_path,
            "stations_path": resolved_stations_path,
            "dialogue_path": dialogue_path,
            "time_slots": time_slots,
            "base_date": base_date,
            "skip_solver": skip_solver,
        }
        with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(run_meta, f, ensure_ascii=False, indent=2)
        print(f"Run directory: {run_dir}")

        client = None

        if not offline:
            from openai import OpenAI

            base = api_base or os.getenv(
                "LLMOPT_API_BASE_URL", "http://35.220.164.252:3888/v1/"
            )
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

        if csv_path:
            # 从 CSV 生成对话（stations_path 可选）
            from llm4fairrouting.llm.dialogue_generation import generate_dialogues

            dialogues = generate_dialogues(
                csv_path=csv_path,
                xlsx_path=resolved_stations_path,
                offline=offline,
                client=client,
                model=model,
                base_date=base_date,
                n_events=n_events,
                time_slots=time_slots,
                temperature=temperature,
                batch_size=dialogue_batch_size,
            )
            dlg_out = run_dir / "generated_dialogues.jsonl"
            from llm4fairrouting.llm.dialogue_generation import save_dialogues

            save_dialogues(dialogues, str(dlg_out))
        elif dialogue_path:
            # 兼容旧路径：直接加载 JSONL 对话文件
            with open(dialogue_path, "r", encoding="utf-8") as f:
                dialogues = [json.loads(l.strip()) for l in f if l.strip()]
            print(f"  加载 {len(dialogues)} 条对话（来自文件: {dialogue_path}）")
        else:
            raise ValueError(
                "必须提供 csv_path（从 CSV 生成对话）"
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
                dialogues,
                client,
                model,
                window_minutes,
                temperature,
            )

        demands_path = run_dir / "extracted_demands.json"
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
        windows_to_solve = []
        weight_configs_by_window: Dict[str, Dict] = {}

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

            wc_path = weight_configs_dir / f"weight_config_window{w_idx}.json"
            with open(wc_path, "w", encoding="utf-8") as f:
                json.dump(weight_config, f, ensure_ascii=False, indent=2)
            weight_configs_by_window[tw] = weight_config

            if skip_solver:
                print(f"  跳过求解 (--skip-solver)")
                all_solutions.append(
                    {
                        "time_window": tw,
                        "weight_config": weight_config,
                        "feasible_demands": demands,
                        "n_demands_total": len(demands),
                        "n_demands_filtered": 0,
                        "solution": None,
                        "n_supply": 0,
                    }
                )
                continue

            windows_to_solve.append(
                {
                    "time_window": tw,
                    "demands": demands,
                }
            )

        if not skip_solver and windows_to_solve:
            all_solutions.extend(
                solve_windows_dynamically(
                    windows=windows_to_solve,
                    weight_configs=weight_configs_by_window,
                    stations_path=resolved_stations_path,
                    building_path=building_path or str(BUILDING_DATA_PATH),
                    max_solver_stations=max_solver_stations,
                    time_limit=time_limit,
                    max_drones_per_station=max_drones_per_station,
                    max_payload=max_payload,
                    max_range=max_range,
                    noise_weight=noise_weight,
                    drone_speed=drone_speed,
                )
            )

        # ----------------------------------------------------------------
        # 保存汇总结果（含完整 eval 字段）
        # ----------------------------------------------------------------
        summary_path = run_dir / "workflow_results.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                serialize_workflow_results(all_solutions),
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n{'=' * 60}")
        print("Workflow 完成！结果保存至:")
        print(f"  Run directory   : {run_dir}")
        print(f"  Weight configs  : {weight_configs_dir}")
        print(f"  Workflow results: {summary_path}")
        print(f"  Log file        : {log_path}")
        print(f"{'=' * 60}")

        return all_solutions
    finally:
        # 恢复 stdout 并关闭日志文件
        sys.stdout = original_stdout
        log_file.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="llm4fairrouting Workflow Runner (Module 1 → 2 → 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 离线全流程（从 CSV 生成对话，跳过求解器）
  python -m llm4fairrouting.workflow.run_workflow --offline --skip-solver

  # 指定 CSV + 站点文件，仅处理前 20 条事件
  python -m llm4fairrouting.workflow.run_workflow --offline --skip-solver --n-events 20

  # 在线 LLM 模式
  python -m llm4fairrouting.workflow.run_workflow --api-key YOUR_KEY --n-events 10
""",
    )

    # Module 1 数据来源（二选一）
    src_group = parser.add_argument_group("Module 1 数据来源")
    src_group.add_argument(
        "--csv", type=str,
        default=str(DEMAND_EVENTS_PATH),
        help=f"需求事件 CSV 路径（默认 {DEMAND_EVENTS_FILENAME}）",
    )
    src_group.add_argument(
        "--stations", type=str,
        default=str(STATION_DATA_PATH),
        help=f"{STATION_DATA_FILENAME} 站点数据路径（默认使用项目 seed 数据）",
    )
    src_group.add_argument(
        "--dialogues", type=str, default=None,
        help="[兼容] 直接加载已有 JSONL 对话文件（不指定时使用 --csv + --stations）",
    )

    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "results"),
        help="输出根目录（每次运行自动创建带时间戳的子目录）",
    )
    parser.add_argument(
        "--building-data",
        type=str,
        default=str(BUILDING_DATA_PATH),
        help=f"{BUILDING_DATA_FILENAME} 建筑物数据路径（用于真实距离/噪声矩阵）",
    )
    parser.add_argument("--offline", action="store_true", help="离线模式，不调用 LLM")
    parser.add_argument("--skip-solver", action="store_true", help="跳过 CPLEX 求解")
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--window", type=int, default=5, help="时间窗口（分钟），默认 5 以匹配需求事件的 5 分钟粒度")
    parser.add_argument("--time-limit", type=int, default=10, help="求解器时间限制（秒）")
    parser.add_argument("--n-events", type=int, default=None, help="最多处理的事件数")
    parser.add_argument("--time-slots", type=int, nargs="+", default=None,
                        help="仅处理指定 time_slot（空格分隔）")
    parser.add_argument("--base-date", type=str, default="2024-03-15", help="基准日期")
    parser.add_argument("--dialogue-batch-size", type=int, default=5,
                        help="LLM 对话生成批大小")
    parser.add_argument(
        "--max-solver-stations",
        type=int,
        default=1,
        help="求解时最多使用多少个真实站点；默认 1 以对齐 baseline demo，0 表示使用全部",
    )
    parser.add_argument(
        "--drone-speed",
        type=float,
        default=60.0,
        help="动态模拟中的无人机飞行速度（m/s）",
    )
    parser.add_argument("--noise-weight", type=float, default=0.5,
                        help="噪声成本权重（>0 启用噪声优先级，按 1/priority 加权）")
    args = parser.parse_args()

    # 确定 Module 1 数据来源
    csv_path = None
    stations_path = None
    if args.dialogues:
        # 显式指定了旧格式文件
        dialogue_path = args.dialogues
    else:
        # 默认从 CSV 生成（stations 可选）
        csv_path = args.csv
        stations_path = args.stations  # may be None
        dialogue_path = None

    run_workflow(
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
        max_solver_stations=args.max_solver_stations,
        skip_solver=args.skip_solver,
        noise_weight=args.noise_weight,
        building_path=args.building_data,
        drone_speed=args.drone_speed,
    )


if __name__ == "__main__":
    main()
