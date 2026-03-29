"""Run a static independent-window experiment using only NSGA-III heuristic planning."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evals.eval_priority_alignment import evaluate_priority_alignment
from llm4fairrouting.data.event_data import load_ground_truth_event_index
from llm4fairrouting.routing.rrt_visualization import select_representative_frontier_solution
from llm4fairrouting.workflow.static_heuristic_solver import run_nsga3_static_heuristic_search


def _python_env() -> Dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) if not existing else str(SRC) + os.pathsep + existing
    return env


def _run(command: List[str], *, env: Dict[str, str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _latest_run_dir(base_dir: Path) -> Path:
    candidates = [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("run_")]
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found under {base_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: object) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(target)


def _safe_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _dialogue_time_slot(dialogue: Dict[str, object]) -> Optional[int]:
    metadata = dialogue.get("metadata", {}) if isinstance(dialogue, dict) else {}
    raw_slot = metadata.get("time_slot")
    if raw_slot not in (None, ""):
        try:
            return int(raw_slot)
        except (TypeError, ValueError):
            pass

    timestamp = str(dialogue.get("timestamp", "")).strip() if isinstance(dialogue, dict) else ""
    if len(timestamp) < 16:
        return None
    try:
        hour = int(timestamp[11:13])
        minute = int(timestamp[14:16])
    except ValueError:
        return None
    return (hour * 60 + minute) // 5


def _time_window_to_slot(time_window: str) -> Optional[int]:
    text = str(time_window).strip()
    if "T" not in text:
        return None
    try:
        time_part = text.split("T", 1)[1]
        start_part = time_part.split("-", 1)[0]
        hour_str, minute_str = start_part.split(":")[:2]
        return (int(hour_str) * 60 + int(minute_str)) // 5
    except (IndexError, ValueError):
        return None


def _available_time_slots(dialogues_path: str | Path) -> List[int]:
    slots = set()
    with open(dialogues_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            slot = _dialogue_time_slot(json.loads(line))
            if slot is not None:
                slots.add(slot)
    return sorted(slots)


def _sample_time_slots(dialogues_path: str | Path, *, sample_size: int, seed: int) -> List[int]:
    available = _available_time_slots(dialogues_path)
    if len(available) < sample_size:
        raise ValueError(f"Requested {sample_size} slots but only found {len(available)}")
    rng = random.Random(seed)
    return sorted(rng.sample(available, sample_size))


def _load_dialogues(dialogues_path: str | Path) -> List[Dict[str, object]]:
    dialogues: List[Dict[str, object]] = []
    with open(dialogues_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))
    return dialogues


def _load_ground_truth_events(path: str | Path) -> Dict[str, Dict[str, object]]:
    return load_ground_truth_event_index(path)


def _evaluate_demand_extraction(
    *,
    extracted_demands_path: str | Path,
    dialogues_path: str | Path,
    ground_truth_csv: str | Path,
    selected_slots: Sequence[int],
) -> Dict[str, object]:
    selected = {int(slot) for slot in selected_slots}
    dialogues = _load_dialogues(dialogues_path)
    ground_truth = _load_ground_truth_events(ground_truth_csv)

    dialogue_to_event: Dict[str, str] = {}
    selected_event_ids = set()
    for dialogue in dialogues:
        slot = _dialogue_time_slot(dialogue)
        if slot not in selected:
            continue
        dialogue_id = str(dialogue.get("dialogue_id", "")).strip()
        event_id = str((dialogue.get("metadata") or {}).get("event_id", "")).strip()
        if dialogue_id and event_id and event_id in ground_truth:
            dialogue_to_event[dialogue_id] = event_id
            selected_event_ids.add(event_id)

    payload = _load_json(extracted_demands_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload: {extracted_demands_path}")

    tp = 0
    fp = 0
    matched = set()
    exact_hits = 0
    for window in payload:
        for demand in window.get("demands", []):
            dialogue_id = str(demand.get("source_dialogue_id", "")).strip()
            event_id = str(demand.get("source_event_id", "")).strip() or dialogue_to_event.get(dialogue_id, "")
            if not event_id or event_id not in selected_event_ids or event_id not in ground_truth:
                fp += 1
                continue
            if event_id in matched:
                fp += 1
                continue
            matched.add(event_id)
            tp += 1
            truth = ground_truth[event_id]
            exact_hits += int(
                str((demand.get("origin") or {}).get("fid", "")).strip() == str(truth.get("supply_fid", "")).strip()
                and str((demand.get("destination") or {}).get("fid", "")).strip() == str(truth.get("demand_fid", "")).strip()
                and abs(float((demand.get("cargo") or {}).get("weight_kg", 0.0) or 0.0) - float(truth.get("material_weight", 0.0) or 0.0)) <= 1e-6
            )
    fn = len(selected_event_ids - matched)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = 2.0 * precision * recall / (precision + recall) if precision and recall else (0.0 if precision == 0.0 or recall == 0.0 else None)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": (exact_hits / tp) if tp else None,
    }


def _load_weight_configs_by_window(weights_dir: str | Path) -> Dict[str, Dict[str, object]]:
    mapping: Dict[str, Dict[str, object]] = {}
    for path in sorted(Path(weights_dir).glob("*.json")):
        payload = _load_json(path)
        if isinstance(payload, dict) and payload.get("time_window"):
            mapping[str(payload["time_window"])] = payload
    return mapping


def _build_workflow_command(
    *,
    python_exe: str,
    output_dir: Path,
    dialogue_path: str,
    stations_path: str,
    building_path: str,
    time_slots: Sequence[int],
    window_minutes: int,
    time_limit: int,
    max_solver_stations: int,
    max_drones_per_station: int,
    max_payload: float,
    max_range: float,
    noise_weight: float,
    drone_activation_cost: float,
    drone_speed: float,
    offline: bool,
) -> List[str]:
    command = [
        python_exe,
        "-m",
        "llm4fairrouting.workflow.run_workflow",
        "--output-dir", str(output_dir),
        "--dialogues", dialogue_path,
        "--stations", stations_path,
        "--building-data", building_path,
        "--window", str(window_minutes),
        "--time-limit", str(time_limit),
        "--max-solver-stations", str(max_solver_stations),
        "--max-drones-per-station", str(max_drones_per_station),
        "--max-payload", str(max_payload),
        "--max-range", str(max_range),
        "--noise-weight", str(noise_weight),
        "--drone-activation-cost", str(drone_activation_cost),
        "--drone-speed", str(drone_speed),
        "--time-slots", *[str(slot) for slot in time_slots],
        "--skip-solver",
    ]
    if offline:
        command.append("--offline")
    return command


def _numeric_summary(rows: Sequence[Dict[str, object]], key: str) -> Dict[str, Optional[float]]:
    values = [float(value) for value in (row.get(key) for row in rows) if _safe_float(value) is not None]
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _write_csv(path: str | Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return str(target)


def _plot_evaluation_metrics(extraction_metrics: Dict[str, object], priority_metrics: Dict[str, object], output_path: str | Path) -> Optional[str]:
    plt = _safe_import_matplotlib()
    if plt is None:
        return None
    items = [
        ("Extraction Precision", extraction_metrics.get("precision")),
        ("Extraction Recall", extraction_metrics.get("recall")),
        ("Extraction F1", extraction_metrics.get("f1")),
        ("Extraction Exact", extraction_metrics.get("exact_match_rate")),
        ("Priority Accuracy", priority_metrics.get("accuracy")),
        ("Priority Macro-F1", priority_metrics.get("macro_f1")),
        ("Priority Weighted-F1", priority_metrics.get("weighted_f1")),
        ("Priority Top-K Hit", (priority_metrics.get("top_k_hit_rate") or {}).get("hit_rate")),
    ]
    labels = [label for label, value in items if _safe_float(value) is not None]
    values = [float(value) for _, value in items if _safe_float(value) is not None]
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    pos = list(range(len(values)))
    bars = ax.bar(pos, values, color=["#4c78a8"] * 4 + ["#f58518"] * max(0, len(values) - 4))
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Extraction and Priority Evaluation")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return str(target)


def _plot_planning_dashboard(rows: Sequence[Dict[str, object]], output_path: str | Path) -> Optional[str]:
    plt = _safe_import_matplotlib()
    if plt is None or not rows:
        return None
    sorted_rows = sorted(rows, key=lambda item: int(item.get("slot", 0)))
    labels = [str(row["slot"]) for row in sorted_rows]
    metrics = [
        ("Frontier Size", "frontier_size", "#4c78a8"),
        ("Search Runtime (s)", "search_runtime_s", "#f58518"),
        ("Representative Service Rate", "representative_service_rate", "#54a24b"),
        ("Representative Distance (m)", "representative_distance_m", "#e45756"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    for ax, (title, key, color) in zip(list(axes.flatten()), metrics):
        vals = [0.0 if _safe_float(row.get(key)) is None else float(row.get(key)) for row in sorted_rows]
        ax.bar(labels, vals, color=color)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return str(target)


def _plot_frontier_scatter(rows: Sequence[Dict[str, object]], output_path: str | Path) -> Optional[str]:
    plt = _safe_import_matplotlib()
    if plt is None:
        return None
    valid_rows = [row for row in rows if _safe_float(row.get("final_total_distance_m")) is not None and _safe_float(row.get("service_rate")) is not None]
    if not valid_rows:
        return None
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    scatter = ax.scatter(
        [float(row["final_total_distance_m"]) for row in valid_rows],
        [float(row["service_rate"]) for row in valid_rows],
        c=[int(row["slot"]) for row in valid_rows],
        cmap="viridis",
        s=46,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_xlabel("Total Distance (m)")
    ax.set_ylabel("Service Rate")
    ax.set_title("Static NSGA-III Heuristic Frontier Solutions")
    ax.grid(alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Time Slot")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 20 independent static NSGA-III heuristic experiments after dialogue extraction.")
    parser.add_argument("--output-root", default="data/static_independent_runs")
    parser.add_argument("--dialogues", default=str(ROOT / "data" / "seed" / "daily_demand_dialogues.jsonl"))
    parser.add_argument("--stations", default=str(ROOT / "data" / "seed" / "drone_station_locations.csv"))
    parser.add_argument("--building-data", default=str(ROOT / "data" / "seed" / "building_information.csv"))
    parser.add_argument("--ground-truth", default=str(ROOT / "data" / "seed" / "daily_demand_events_manifest.jsonl"))
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--time-limit", type=int, default=180)
    parser.add_argument("--max-solver-stations", type=int, default=1)
    parser.add_argument("--max-drones-per-station", type=int, default=3)
    parser.add_argument("--static-total-drones", type=int, default=3)
    parser.add_argument("--max-payload", type=float, default=60.0)
    parser.add_argument("--max-range", type=float, default=200000.0)
    parser.add_argument("--noise-weight", type=float, default=0.5)
    parser.add_argument("--drone-activation-cost", type=float, default=1000.0)
    parser.add_argument("--drone-speed", type=float, default=60.0)
    parser.add_argument("--urgent-threshold", type=int, default=2)
    parser.add_argument("--nsga3-pop-size", type=int, default=6)
    parser.add_argument("--nsga3-n-generations", type=int, default=3)
    parser.add_argument("--nsga3-seed", type=int, default=42)
    args = parser.parse_args()

    session_dir = Path(args.output_root) / f"static_independent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    extract_root = session_dir / "extract_eval"
    planning_root = session_dir / "planning"
    charts_dir = session_dir / "charts"
    session_dir.mkdir(parents=True, exist_ok=True)

    selected_slots = _sample_time_slots(args.dialogues, sample_size=args.sample_size, seed=args.sample_seed)
    selected_slot_set = set(selected_slots)
    _write_json(session_dir / "sampled_slots.json", {"selected_slots": selected_slots, "sample_seed": args.sample_seed})
    print(f"[Sampled Slots] {selected_slots}")

    env = _python_env()
    extract_command = _build_workflow_command(
        python_exe=sys.executable,
        output_dir=extract_root,
        dialogue_path=args.dialogues,
        stations_path=args.stations,
        building_path=args.building_data,
        time_slots=selected_slots,
        window_minutes=args.window,
        time_limit=args.time_limit,
        max_solver_stations=args.max_solver_stations,
        max_drones_per_station=args.max_drones_per_station,
        max_payload=args.max_payload,
        max_range=args.max_range,
        noise_weight=args.noise_weight,
        drone_activation_cost=args.drone_activation_cost,
        drone_speed=args.drone_speed,
        offline=args.offline,
    )
    _run(extract_command, env=env)
    extract_run_dir = _latest_run_dir(extract_root)

    extraction_metrics = _evaluate_demand_extraction(
        extracted_demands_path=extract_run_dir / "extracted_demands.json",
        dialogues_path=args.dialogues,
        ground_truth_csv=args.ground_truth,
        selected_slots=selected_slots,
    )
    extraction_metrics_path = session_dir / "demand_extraction_metrics.json"
    _write_json(extraction_metrics_path, extraction_metrics)

    priority_metrics = evaluate_priority_alignment(
        weights_path=str(extract_run_dir / "weight_configs"),
        demands_path=str(extract_run_dir / "extracted_demands.json"),
        dialogues_path=str(args.dialogues),
        ground_truth_csv=str(args.ground_truth),
        urgent_threshold=args.urgent_threshold,
    )
    priority_metrics_path = session_dir / "priority_alignment.json"
    _write_json(priority_metrics_path, priority_metrics)

    extracted_windows = _load_json(extract_run_dir / "extracted_demands.json")
    if not isinstance(extracted_windows, list):
        raise ValueError("Expected extracted_demands.json to contain a list")
    weight_configs = _load_weight_configs_by_window(extract_run_dir / "weight_configs")
    planning_rows: List[Dict[str, object]] = []
    frontier_rows: List[Dict[str, object]] = []
    planning_cases: List[Dict[str, object]] = []

    for window in extracted_windows:
        time_window = str(window.get("time_window", ""))
        if not time_window:
            continue
        slot = _time_window_to_slot(time_window)
        if slot not in selected_slot_set:
            continue
        weight_config = weight_configs.get(time_window)
        if weight_config is None:
            continue

        case_dir = planning_root / f"slot_{slot:03d}"
        payload = run_nsga3_static_heuristic_search(
            window=window,
            weight_config=weight_config,
            stations_path=args.stations,
            building_path=args.building_data,
            max_solver_stations=args.max_solver_stations,
            max_payload=args.max_payload,
            max_range=args.max_range,
            noise_weight=args.noise_weight,
            drone_speed=args.drone_speed,
            output_dir=str(case_dir / "solver_analytics" / "nsga3_static_heuristic"),
            total_drones=args.static_total_drones,
            drone_activation_cost=args.drone_activation_cost,
            pop_size=args.nsga3_pop_size,
            n_generations=args.nsga3_n_generations,
            seed=args.nsga3_seed,
            problem_id=f"slot_{slot:03d}",
        )
        frontier = list(payload.get("frontier", []))
        representative = select_representative_frontier_solution(frontier)
        search_meta = payload.get("search_meta") or {}
        best_service = max(
            (float((item.get("run_summary") or {}).get("service_rate", 0.0)) for item in frontier),
            default=None,
        )
        row = {
            "slot": slot,
            "time_window": time_window,
            "frontier_size": len(frontier),
            "n_candidates_evaluated": search_meta.get("n_candidates_evaluated"),
            "search_runtime_s": _safe_float(search_meta.get("search_runtime_s")),
            "avg_candidate_runtime_s": _safe_float(search_meta.get("avg_candidate_runtime_s")),
            "representative_solution_id": (representative or {}).get("solution_id"),
            "representative_distance_m": _safe_float((representative or {}).get("final_total_distance_m")),
            "representative_delivery_time_h": _safe_float((representative or {}).get("average_delivery_time_h")),
            "representative_noise_impact": _safe_float((representative or {}).get("final_total_noise_impact")),
            "representative_service_rate": _safe_float(((representative or {}).get("run_summary") or {}).get("service_rate")),
            "representative_n_used_drones": _safe_float((representative or {}).get("n_used_drones")),
            "best_service_rate": best_service,
            "min_distance_m": min((_safe_float(item.get("final_total_distance_m")) for item in frontier if _safe_float(item.get("final_total_distance_m")) is not None), default=None),
            "min_delivery_time_h": min((_safe_float(item.get("average_delivery_time_h")) for item in frontier if _safe_float(item.get("average_delivery_time_h")) is not None), default=None),
            "min_noise_impact": min((_safe_float(item.get("final_total_noise_impact")) for item in frontier if _safe_float(item.get("final_total_noise_impact")) is not None), default=None),
            "results_json": payload.get("summary_json"),
            "representative_frontier_result_path": (representative or {}).get("frontier_result_path"),
        }
        planning_rows.append(row)
        planning_cases.append(
            {
                "slot": slot,
                "time_window": time_window,
                "case_dir": str(case_dir),
                "search_summary_json": payload.get("summary_json"),
                "frontier_json": payload.get("frontier_json"),
            }
        )
        for item in frontier:
            frontier_rows.append(
                {
                    "slot": slot,
                    "time_window": time_window,
                    "solution_id": item.get("solution_id"),
                    "label": item.get("label"),
                    "final_total_distance_m": item.get("final_total_distance_m"),
                    "average_delivery_time_h": item.get("average_delivery_time_h"),
                    "final_total_noise_impact": item.get("final_total_noise_impact"),
                    "service_rate": (item.get("run_summary") or {}).get("service_rate"),
                    "service_rate_loss": item.get("service_rate_loss"),
                    "n_used_drones": item.get("n_used_drones"),
                    "frontier_result_path": item.get("frontier_result_path"),
                }
            )
        print(
            f"[Static Planning] slot={slot} frontier={len(frontier)} "
            f"runtime={float(search_meta.get('search_runtime_s', 0.0)):.3f}s"
        )

    planning_summary = {
        "selected_slots": selected_slots,
        "aggregate": {
            "frontier_size": _numeric_summary(planning_rows, "frontier_size"),
            "search_runtime_s": _numeric_summary(planning_rows, "search_runtime_s"),
            "representative_service_rate": _numeric_summary(planning_rows, "representative_service_rate"),
            "representative_distance_m": _numeric_summary(planning_rows, "representative_distance_m"),
            "best_service_rate": _numeric_summary(planning_rows, "best_service_rate"),
        },
        "per_window": planning_rows,
    }
    planning_summary_path = session_dir / "planning_summary.json"
    frontier_rows_path = session_dir / "planning_frontier_rows.json"
    _write_json(planning_summary_path, planning_summary)
    _write_json(frontier_rows_path, frontier_rows)

    planning_csv_path = session_dir / "planning_summary.csv"
    frontier_csv_path = session_dir / "planning_frontier_rows.csv"
    _write_csv(planning_csv_path, planning_rows, fieldnames=list(planning_rows[0].keys()) if planning_rows else ["slot"])
    _write_csv(frontier_csv_path, frontier_rows, fieldnames=list(frontier_rows[0].keys()) if frontier_rows else ["slot"])

    evaluation_chart = _plot_evaluation_metrics(extraction_metrics, priority_metrics, charts_dir / "evaluation_metrics.png")
    planning_chart = _plot_planning_dashboard(planning_rows, charts_dir / "planning_window_dashboard.png")
    frontier_chart = _plot_frontier_scatter(frontier_rows, charts_dir / "planning_frontier_scatter.png")

    manifest = {
        "session_dir": str(session_dir),
        "selected_slots": selected_slots,
        "extract_run_dir": str(extract_run_dir),
        "demand_extraction_metrics_json": str(extraction_metrics_path),
        "priority_alignment_json": str(priority_metrics_path),
        "planning_summary_json": str(planning_summary_path),
        "planning_summary_csv": str(planning_csv_path),
        "planning_cases": planning_cases,
        "charts": {
            "evaluation_metrics": evaluation_chart,
            "planning_dashboard": planning_chart,
            "planning_frontier_scatter": frontier_chart,
        },
    }
    manifest_path = session_dir / "experiment_manifest.json"
    _write_json(manifest_path, manifest)

    print("\nStatic independent experiment finished.")
    print(f"  Session directory          : {session_dir}")
    print(f"  Demand extraction metrics  : {extraction_metrics_path}")
    print(f"  Priority alignment metrics : {priority_metrics_path}")
    print(f"  Planning summary JSON      : {planning_summary_path}")
    print(f"  Planning summary CSV       : {planning_csv_path}")
    print(f"  Manifest                   : {manifest_path}")


if __name__ == "__main__":
    main()
