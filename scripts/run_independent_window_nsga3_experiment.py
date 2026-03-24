"""Run a random independent-window experiment with Module 2/3 evaluation and NSGA-III planning."""

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
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evals.eval_priority_alignment import evaluate_priority_alignment
from llm4fairrouting.routing.rrt_visualization import select_representative_frontier_solution


DEFAULT_SAMPLE_SIZE = 20
DEFAULT_NSGA3_POP_SIZE = 4
DEFAULT_NSGA3_N_GENERATIONS = 2
DEFAULT_NSGA3_SEED = 42


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


def _available_time_slots(
    dialogues_path: str | Path,
    *,
    slot_min: Optional[int] = None,
    slot_max: Optional[int] = None,
) -> List[int]:
    slots = set()
    with open(dialogues_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            slot = _dialogue_time_slot(json.loads(line))
            if slot is None:
                continue
            if slot_min is not None and slot < slot_min:
                continue
            if slot_max is not None and slot > slot_max:
                continue
            slots.add(slot)
    return sorted(slots)


def _sample_time_slots(
    dialogues_path: str | Path,
    *,
    sample_size: int,
    seed: int,
    slot_min: Optional[int] = None,
    slot_max: Optional[int] = None,
) -> List[int]:
    available = _available_time_slots(dialogues_path, slot_min=slot_min, slot_max=slot_max)
    if len(available) < sample_size:
        raise ValueError(
            f"Requested {sample_size} slots but only found {len(available)} available slots"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(available, sample_size))


def _load_dialogues(dialogues_path: str | Path) -> List[Dict[str, object]]:
    dialogues: List[Dict[str, object]] = []
    with open(dialogues_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            dialogues.append(json.loads(line))
    return dialogues


def _load_ground_truth_events(path: str | Path) -> Dict[str, Dict[str, object]]:
    events: Dict[str, Dict[str, object]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event_id = str(row.get("event_id") or row.get("unique_id") or "").strip()
            if not event_id:
                continue
            events[event_id] = {
                "event_id": event_id,
                "priority": int(row.get("priority", 4)),
                "supply_fid": str(row.get("supply_fid", "")).strip(),
                "demand_fid": str(row.get("demand_fid", "")).strip(),
                "material_weight": _safe_float(row.get("material_weight")),
            }
    return events


def _evaluate_demand_extraction(
    *,
    extracted_demands_path: str | Path,
    dialogues_path: str | Path,
    ground_truth_csv: str | Path,
    selected_slots: Sequence[int],
    weight_tolerance_kg: float = 1e-6,
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

    extracted_payload = _load_json(extracted_demands_path)
    if not isinstance(extracted_payload, list):
        raise ValueError(f"Expected a list in extracted demands JSON: {extracted_demands_path}")

    matched_event_ids = set()
    tp = 0
    fp = 0
    duplicates = 0
    unresolved_predictions = 0
    exact_match_hits = 0
    origin_hits = 0
    destination_hits = 0
    weight_hits = 0
    per_prediction: List[Dict[str, object]] = []

    for window in extracted_payload:
        for demand in window.get("demands", []):
            dialogue_id = str(demand.get("source_dialogue_id", "")).strip()
            event_id = str(demand.get("source_event_id", "")).strip() or dialogue_to_event.get(dialogue_id, "")
            if not event_id:
                unresolved_predictions += 1
                fp += 1
                per_prediction.append(
                    {
                        "event_id": None,
                        "dialogue_id": dialogue_id,
                        "match": "fp_unresolved",
                    }
                )
                continue

            if event_id not in selected_event_ids or event_id not in ground_truth:
                fp += 1
                per_prediction.append(
                    {
                        "event_id": event_id,
                        "dialogue_id": dialogue_id,
                        "match": "fp_out_of_scope",
                    }
                )
                continue

            if event_id in matched_event_ids:
                duplicates += 1
                fp += 1
                per_prediction.append(
                    {
                        "event_id": event_id,
                        "dialogue_id": dialogue_id,
                        "match": "fp_duplicate",
                    }
                )
                continue

            tp += 1
            matched_event_ids.add(event_id)
            truth = ground_truth[event_id]
            pred_origin = str((demand.get("origin") or {}).get("fid", "")).strip()
            pred_destination = str((demand.get("destination") or {}).get("fid", "")).strip()
            pred_weight = _safe_float((demand.get("cargo") or {}).get("weight_kg"))
            true_weight = _safe_float(truth.get("material_weight"))

            origin_match = pred_origin == str(truth.get("supply_fid", "")).strip()
            destination_match = pred_destination == str(truth.get("demand_fid", "")).strip()
            weight_match = (
                pred_weight is not None
                and true_weight is not None
                and abs(pred_weight - true_weight) <= weight_tolerance_kg
            )

            origin_hits += int(origin_match)
            destination_hits += int(destination_match)
            weight_hits += int(weight_match)
            exact_match = origin_match and destination_match and weight_match
            exact_match_hits += int(exact_match)

            per_prediction.append(
                {
                    "event_id": event_id,
                    "dialogue_id": dialogue_id,
                    "match": "tp",
                    "origin_match": origin_match,
                    "destination_match": destination_match,
                    "weight_match": weight_match,
                    "exact_match": exact_match,
                }
            )

    fn = len(selected_event_ids - matched_event_ids)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision not in (None, 0.0) and recall not in (None, 0.0)
        else (0.0 if precision == 0.0 or recall == 0.0 else None)
    )
    matched_count = max(tp, 1)
    return {
        "n_selected_slots": len(selected),
        "n_true_events": len(selected_event_ids),
        "n_predicted_demands": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "duplicates": duplicates,
        "unresolved_predictions": unresolved_predictions,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": exact_match_hits / matched_count if tp else None,
        "origin_fid_accuracy": origin_hits / matched_count if tp else None,
        "destination_fid_accuracy": destination_hits / matched_count if tp else None,
        "weight_accuracy": weight_hits / matched_count if tp else None,
        "selected_slots": list(sorted(selected)),
        "per_prediction": per_prediction,
    }


def _time_window_to_slot(time_window: str) -> int:
    span = str(time_window).split("T", 1)[1]
    start_text = span.split("-", 1)[0]
    hour_text, minute_text = start_text.split(":")
    return (int(hour_text) * 60 + int(minute_text)) // 5


def _load_weight_configs_by_window(weights_dir: str | Path) -> Dict[str, Dict[str, object]]:
    weights_path = Path(weights_dir)
    mapping: Dict[str, Dict[str, object]] = {}
    for path in sorted(weights_path.glob("*.json")):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        time_window = payload.get("time_window")
        if time_window:
            mapping[str(time_window)] = payload
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
    skip_solver: bool,
) -> List[str]:
    command = [
        python_exe,
        "-m",
        "llm4fairrouting.workflow.run_workflow",
        "--output-dir",
        str(output_dir),
        "--dialogues",
        dialogue_path,
        "--stations",
        stations_path,
        "--building-data",
        building_path,
        "--window",
        str(window_minutes),
        "--time-limit",
        str(time_limit),
        "--max-solver-stations",
        str(max_solver_stations),
        "--max-drones-per-station",
        str(max_drones_per_station),
        "--max-payload",
        str(max_payload),
        "--max-range",
        str(max_range),
        "--noise-weight",
        str(noise_weight),
        "--drone-activation-cost",
        str(drone_activation_cost),
        "--drone-speed",
        str(drone_speed),
        "--time-slots",
        *[str(slot) for slot in time_slots],
    ]
    if offline:
        command.append("--offline")
    if skip_solver:
        command.append("--skip-solver")
    return command


def _build_solver_command(
    *,
    python_exe: str,
    demands_path: Path,
    weights_path: Path,
    stations_path: str,
    building_path: str,
    output_path: Path,
    time_limit: int,
    max_solver_stations: int,
    max_drones_per_station: int,
    max_payload: float,
    max_range: float,
    noise_weight: float,
    drone_activation_cost: float,
    drone_speed: float,
    solver_backend: str,
    nsga3_pop_size: int,
    nsga3_n_generations: int,
    nsga3_seed: int,
) -> List[str]:
    return [
        python_exe,
        "-m",
        "llm4fairrouting.workflow.solver_adapter",
        "--demands",
        str(demands_path),
        "--weights",
        str(weights_path),
        "--stations",
        stations_path,
        "--building-data",
        building_path,
        "--output",
        str(output_path),
        "--time-limit",
        str(time_limit),
        "--max-solver-stations",
        str(max_solver_stations),
        "--max-drones-per-station",
        str(max_drones_per_station),
        "--max-payload",
        str(max_payload),
        "--max-range",
        str(max_range),
        "--noise-weight",
        str(noise_weight),
        "--drone-activation-cost",
        str(drone_activation_cost),
        "--drone-speed",
        str(drone_speed),
        "--solver-backend",
        solver_backend,
        "--nsga3-pop-size",
        str(nsga3_pop_size),
        "--nsga3-n-generations",
        str(nsga3_n_generations),
        "--nsga3-seed",
        str(nsga3_seed),
    ]


def _solution_service_rate(item: Dict[str, object]) -> Optional[float]:
    run_summary = item.get("run_summary") or {}
    value = _safe_float(run_summary.get("service_rate"))
    if value is not None:
        return value
    loss = _safe_float(item.get("service_rate_loss"))
    return None if loss is None else max(0.0, 1.0 - loss)


def _summarize_nsga3_frontier(
    *,
    results_path: str | Path,
    slot: int,
    time_window: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    payload = _load_json(results_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in NSGA-III results: {results_path}")

    frontier = list(payload.get("frontier", []))
    representative = select_representative_frontier_solution(frontier)
    search_meta = payload.get("search_meta") or {}
    frontier_rows: List[Dict[str, object]] = []

    best_service_rate = None
    min_distance = None
    min_latency = None
    min_noise = None
    min_used_drones = None

    for item in frontier:
        row = {
            "slot": slot,
            "time_window": time_window,
            "solution_id": item.get("solution_id"),
            "label": item.get("label"),
            "final_total_distance_m": _safe_float(item.get("final_total_distance_m")),
            "average_delivery_time_h": _safe_float(item.get("average_delivery_time_h")),
            "final_total_noise_impact": _safe_float(item.get("final_total_noise_impact")),
            "service_rate": _solution_service_rate(item),
            "service_rate_loss": _safe_float(item.get("service_rate_loss")),
            "n_used_drones": _safe_float(item.get("n_used_drones")),
            "frontier_result_path": item.get("frontier_result_path"),
            "results_json": str(results_path),
        }
        frontier_rows.append(row)

        service_rate = row["service_rate"]
        if service_rate is not None:
            best_service_rate = service_rate if best_service_rate is None else max(best_service_rate, service_rate)
        if row["final_total_distance_m"] is not None:
            min_distance = row["final_total_distance_m"] if min_distance is None else min(min_distance, row["final_total_distance_m"])
        if row["average_delivery_time_h"] is not None:
            min_latency = row["average_delivery_time_h"] if min_latency is None else min(min_latency, row["average_delivery_time_h"])
        if row["final_total_noise_impact"] is not None:
            min_noise = row["final_total_noise_impact"] if min_noise is None else min(min_noise, row["final_total_noise_impact"])
        if row["n_used_drones"] is not None:
            min_used_drones = row["n_used_drones"] if min_used_drones is None else min(min_used_drones, row["n_used_drones"])

    representative_service_rate = _solution_service_rate(representative or {})
    summary = {
        "slot": slot,
        "time_window": time_window,
        "frontier_size": len(frontier),
        "n_candidates_evaluated": search_meta.get("n_candidates_evaluated"),
        "search_runtime_s": _safe_float(search_meta.get("search_runtime_s")),
        "avg_candidate_runtime_s": _safe_float(search_meta.get("avg_candidate_runtime_s")),
        "results_json": str(results_path),
        "representative_solution_id": (representative or {}).get("solution_id"),
        "representative_frontier_result_path": (representative or {}).get("frontier_result_path"),
        "representative_distance_m": _safe_float((representative or {}).get("final_total_distance_m")),
        "representative_delivery_time_h": _safe_float((representative or {}).get("average_delivery_time_h")),
        "representative_noise_impact": _safe_float((representative or {}).get("final_total_noise_impact")),
        "representative_service_rate": representative_service_rate,
        "representative_n_used_drones": _safe_float((representative or {}).get("n_used_drones")),
        "best_service_rate": best_service_rate,
        "min_distance_m": min_distance,
        "min_delivery_time_h": min_latency,
        "min_noise_impact": min_noise,
        "min_used_drones": min_used_drones,
    }
    return summary, frontier_rows


def _numeric_summary(rows: Sequence[Dict[str, object]], key: str) -> Dict[str, Optional[float]]:
    values = [float(value) for value in (row.get(key) for row in rows) if _safe_float(value) is not None]
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _aggregate_planning_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "n_windows": len(rows),
        "frontier_size": _numeric_summary(rows, "frontier_size"),
        "search_runtime_s": _numeric_summary(rows, "search_runtime_s"),
        "representative_service_rate": _numeric_summary(rows, "representative_service_rate"),
        "representative_distance_m": _numeric_summary(rows, "representative_distance_m"),
        "best_service_rate": _numeric_summary(rows, "best_service_rate"),
        "min_distance_m": _numeric_summary(rows, "min_distance_m"),
        "min_delivery_time_h": _numeric_summary(rows, "min_delivery_time_h"),
        "min_noise_impact": _numeric_summary(rows, "min_noise_impact"),
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


def _plot_evaluation_metrics(
    *,
    extraction_metrics: Dict[str, object],
    priority_metrics: Dict[str, object],
    output_path: str | Path,
) -> Optional[str]:
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
    positions = list(range(len(values)))
    bars = ax.bar(positions, values, color=["#4c78a8"] * 4 + ["#f58518"] * max(0, len(values) - 4))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Demand Extraction and Priority Evaluation")
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
    axes_list = list(axes.flatten())
    for ax, (title, key, color) in zip(axes_list, metrics):
        values = [row.get(key) for row in sorted_rows]
        numeric = [0.0 if _safe_float(value) is None else float(value) for value in values]
        ax.bar(labels, numeric, color=color, alpha=0.9)
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

    valid_rows = [
        row for row in rows
        if _safe_float(row.get("final_total_distance_m")) is not None
        and _safe_float(row.get("service_rate")) is not None
    ]
    if not valid_rows:
        return None

    xs = [float(row["final_total_distance_m"]) for row in valid_rows]
    ys = [float(row["service_rate"]) for row in valid_rows]
    colors = [int(row["slot"]) for row in valid_rows]

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    scatter = ax.scatter(xs, ys, c=colors, cmap="viridis", s=46, alpha=0.85, edgecolors="black", linewidths=0.3)
    ax.set_xlabel("Total Distance (m)")
    ax.set_ylabel("Service Rate")
    ax.set_title("Aggregated NSGA-III Frontier Solutions Across Independent Windows")
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
    parser = argparse.ArgumentParser(
        description="Randomly sample windows, evaluate extraction/priority, and solve each window independently with NSGA-III."
    )
    parser.add_argument("--output-root", default="data/independent_window_experiments", help="Output root for experiment sessions")
    parser.add_argument("--dialogues", default=str(ROOT / "data" / "seed" / "daily_demand_dialogues.jsonl"))
    parser.add_argument("--stations", default=str(ROOT / "data" / "seed" / "drone_station_locations.csv"))
    parser.add_argument("--building-data", default=str(ROOT / "data" / "seed" / "building_information.csv"))
    parser.add_argument("--ground-truth", default=str(ROOT / "data" / "seed" / "daily_demand_events.csv"))
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True, help="Run extraction/ranking without calling an LLM")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of random 5-minute slots to sample")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed used for time-slot sampling")
    parser.add_argument("--slot-min", type=int, default=None, help="Optional minimum slot index to sample from")
    parser.add_argument("--slot-max", type=int, default=None, help="Optional maximum slot index to sample from")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--time-limit", type=int, default=180)
    parser.add_argument("--max-solver-stations", type=int, default=1)
    parser.add_argument("--max-drones-per-station", type=int, default=3)
    parser.add_argument("--max-payload", type=float, default=60.0)
    parser.add_argument("--max-range", type=float, default=200000.0)
    parser.add_argument("--noise-weight", type=float, default=0.5)
    parser.add_argument("--drone-activation-cost", type=float, default=1000.0)
    parser.add_argument("--drone-speed", type=float, default=60.0)
    parser.add_argument("--urgent-threshold", type=int, default=2)
    parser.add_argument("--solver-backend", choices=("nsga3", "nsga3_heuristic"), default="nsga3")
    parser.add_argument("--nsga3-pop-size", type=int, default=DEFAULT_NSGA3_POP_SIZE)
    parser.add_argument("--nsga3-n-generations", type=int, default=DEFAULT_NSGA3_N_GENERATIONS)
    parser.add_argument("--nsga3-seed", type=int, default=DEFAULT_NSGA3_SEED)
    args = parser.parse_args()

    session_dir = Path(args.output_root) / f"independent_window_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    extraction_root = session_dir / "extract_eval"
    planning_root = session_dir / "planning"
    charts_dir = session_dir / "charts"
    manifest_path = session_dir / "experiment_manifest.json"
    session_dir.mkdir(parents=True, exist_ok=True)

    selected_slots = _sample_time_slots(
        args.dialogues,
        sample_size=args.sample_size,
        seed=args.sample_seed,
        slot_min=args.slot_min,
        slot_max=args.slot_max,
    )
    _write_json(
        session_dir / "sampled_slots.json",
        {
            "sample_size": args.sample_size,
            "sample_seed": args.sample_seed,
            "selected_slots": selected_slots,
            "slot_min": args.slot_min,
            "slot_max": args.slot_max,
        },
    )
    print(f"[Sampled Slots] {selected_slots}")

    python_exe = sys.executable
    env = _python_env()

    extract_command = _build_workflow_command(
        python_exe=python_exe,
        output_dir=extraction_root,
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
        skip_solver=True,
    )
    _run(extract_command, env=env)
    extract_run_dir = _latest_run_dir(extraction_root)

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

    extracted_windows_payload = _load_json(extract_run_dir / "extracted_demands.json")
    if not isinstance(extracted_windows_payload, list):
        raise ValueError("Expected extracted_demands.json to contain a list of windows")
    weights_by_window = _load_weight_configs_by_window(extract_run_dir / "weight_configs")

    planning_rows: List[Dict[str, object]] = []
    frontier_rows: List[Dict[str, object]] = []
    planning_cases: List[Dict[str, object]] = []

    for window in extracted_windows_payload:
        time_window = str(window.get("time_window", ""))
        if not time_window:
            continue
        slot = _time_window_to_slot(time_window)
        if slot not in selected_slots:
            continue
        weight_config = weights_by_window.get(time_window)
        if weight_config is None:
            raise KeyError(f"Missing weight config for window {time_window}")

        case_dir = planning_root / f"slot_{slot:03d}"
        demands_path = case_dir / "demands.json"
        weights_path = case_dir / "weights.json"
        results_path = case_dir / f"{args.solver_backend}_results.json"
        _write_json(demands_path, [window])
        _write_json(weights_path, weight_config)

        solve_command = _build_solver_command(
            python_exe=python_exe,
            demands_path=demands_path,
            weights_path=weights_path,
            stations_path=args.stations,
            building_path=args.building_data,
            output_path=results_path,
            time_limit=args.time_limit,
            max_solver_stations=args.max_solver_stations,
            max_drones_per_station=args.max_drones_per_station,
            max_payload=args.max_payload,
            max_range=args.max_range,
            noise_weight=args.noise_weight,
            drone_activation_cost=args.drone_activation_cost,
            drone_speed=args.drone_speed,
            solver_backend=args.solver_backend,
            nsga3_pop_size=args.nsga3_pop_size,
            nsga3_n_generations=args.nsga3_n_generations,
            nsga3_seed=args.nsga3_seed,
        )
        _run(solve_command, env=env)

        planning_summary, case_frontier_rows = _summarize_nsga3_frontier(
            results_path=results_path,
            slot=slot,
            time_window=time_window,
        )
        planning_rows.append(planning_summary)
        frontier_rows.extend(case_frontier_rows)
        planning_cases.append(
            {
                "slot": slot,
                "time_window": time_window,
                "case_dir": str(case_dir),
                "demands_json": str(demands_path),
                "weights_json": str(weights_path),
                "results_json": str(results_path),
                "representative_frontier_result_path": planning_summary.get("representative_frontier_result_path"),
            }
        )

    planning_summary_payload = {
        "solver_backend": args.solver_backend,
        "selected_slots": selected_slots,
        "aggregate": _aggregate_planning_rows(planning_rows),
        "per_window": planning_rows,
    }
    planning_summary_path = session_dir / "planning_summary.json"
    frontier_summary_path = session_dir / "planning_frontier_rows.json"
    _write_json(planning_summary_path, planning_summary_payload)
    _write_json(frontier_summary_path, frontier_rows)

    planning_csv_path = session_dir / "planning_summary.csv"
    frontier_csv_path = session_dir / "planning_frontier_rows.csv"
    _write_csv(
        planning_csv_path,
        planning_rows,
        fieldnames=[
            "slot",
            "time_window",
            "frontier_size",
            "n_candidates_evaluated",
            "search_runtime_s",
            "avg_candidate_runtime_s",
            "representative_solution_id",
            "representative_distance_m",
            "representative_delivery_time_h",
            "representative_noise_impact",
            "representative_service_rate",
            "representative_n_used_drones",
            "best_service_rate",
            "min_distance_m",
            "min_delivery_time_h",
            "min_noise_impact",
            "min_used_drones",
            "results_json",
            "representative_frontier_result_path",
        ],
    )
    _write_csv(
        frontier_csv_path,
        frontier_rows,
        fieldnames=[
            "slot",
            "time_window",
            "solution_id",
            "label",
            "final_total_distance_m",
            "average_delivery_time_h",
            "final_total_noise_impact",
            "service_rate",
            "service_rate_loss",
            "n_used_drones",
            "frontier_result_path",
            "results_json",
        ],
    )

    evaluation_chart = _plot_evaluation_metrics(
        extraction_metrics=extraction_metrics,
        priority_metrics=priority_metrics,
        output_path=charts_dir / "evaluation_metrics.png",
    )
    planning_chart = _plot_planning_dashboard(
        planning_rows,
        charts_dir / "planning_window_dashboard.png",
    )
    frontier_scatter_chart = _plot_frontier_scatter(
        frontier_rows,
        charts_dir / "planning_frontier_scatter.png",
    )

    manifest = {
        "session_dir": str(session_dir),
        "python_executable": python_exe,
        "offline": bool(args.offline),
        "selected_slots": selected_slots,
        "extraction_eval": {
            "run_dir": str(extract_run_dir),
            "extracted_demands": str(extract_run_dir / "extracted_demands.json"),
            "weight_configs": str(extract_run_dir / "weight_configs"),
            "demand_extraction_metrics_json": str(extraction_metrics_path),
            "priority_alignment_json": str(priority_metrics_path),
        },
        "planning": {
            "solver_backend": args.solver_backend,
            "cases": planning_cases,
            "summary_json": str(planning_summary_path),
            "summary_csv": str(planning_csv_path),
            "frontier_rows_json": str(frontier_summary_path),
            "frontier_rows_csv": str(frontier_csv_path),
        },
        "charts": {
            "evaluation_metrics": evaluation_chart,
            "planning_dashboard": planning_chart,
            "planning_frontier_scatter": frontier_scatter_chart,
        },
    }
    _write_json(manifest_path, manifest)

    print("\nIndependent-window experiment finished.")
    print(f"  Session directory           : {session_dir}")
    print(f"  Sampled slots               : {selected_slots}")
    print(f"  Demand extraction metrics   : {extraction_metrics_path}")
    print(f"  Priority alignment metrics  : {priority_metrics_path}")
    print(f"  Planning summary JSON       : {planning_summary_path}")
    print(f"  Planning summary CSV        : {planning_csv_path}")
    if evaluation_chart:
        print(f"  Evaluation chart            : {evaluation_chart}")
    if planning_chart:
        print(f"  Planning dashboard          : {planning_chart}")
    if frontier_scatter_chart:
        print(f"  Frontier scatter chart      : {frontier_scatter_chart}")
    print(f"  Experiment manifest         : {manifest_path}")


if __name__ == "__main__":
    main()
