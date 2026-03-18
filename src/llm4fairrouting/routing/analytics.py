"""Analytics helpers for CPLEX traces, Pareto scans, and visualization outputs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_OBJECTIVE_WEIGHTS = {
    "w_distance": 1.0,
    "w_time": 1.0,
    "w_risk": 1.0,
}

PARETO_METRICS = (
    "final_total_distance_m",
    "average_delivery_time_h",
    "final_total_noise_impact",
)


def sanitize_label(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "artifact"
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text.strip("._-") or "artifact"


def ensure_directory(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_objective_weights(weights: Dict[str, float] | None) -> Dict[str, float]:
    merged = dict(DEFAULT_OBJECTIVE_WEIGHTS)
    if not weights:
        return merged

    for key in DEFAULT_OBJECTIVE_WEIGHTS:
        raw = weights.get(key, merged[key])
        try:
            merged[key] = max(0.0, float(raw))
        except (TypeError, ValueError):
            continue
    return merged


def build_default_pareto_profiles(
    base_weights: Dict[str, float] | None = None,
) -> List[Dict[str, object]]:
    base = resolve_objective_weights(base_weights)
    return [
        {
            "profile_id": "balanced",
            "label": "Balanced",
            "weights": base,
        },
        {
            "profile_id": "distance_focus",
            "label": "Distance Focus",
            "weights": {
                "w_distance": base["w_distance"] * 3.0,
                "w_time": base["w_time"],
                "w_risk": base["w_risk"],
            },
        },
        {
            "profile_id": "time_focus",
            "label": "Time Focus",
            "weights": {
                "w_distance": base["w_distance"],
                "w_time": base["w_time"] * 3.0,
                "w_risk": base["w_risk"],
            },
        },
        {
            "profile_id": "risk_focus",
            "label": "Risk Focus",
            "weights": {
                "w_distance": base["w_distance"],
                "w_time": base["w_time"],
                "w_risk": base["w_risk"] * 3.0,
            },
        },
        {
            "profile_id": "time_risk_balance",
            "label": "Time-Risk",
            "weights": {
                "w_distance": base["w_distance"],
                "w_time": base["w_time"] * 2.0,
                "w_risk": base["w_risk"] * 2.0,
            },
        },
    ]


def parse_cplex_incumbent_trace(log_path: Path | str | None) -> List[Dict[str, float]]:
    if log_path is None:
        return []

    path = Path(log_path)
    if not path.exists():
        return []

    found_pattern = re.compile(
        r"Found incumbent of value\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
        r".*?after\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+sec",
        re.IGNORECASE,
    )
    time_pattern = re.compile(r"Time\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)
    node_pattern = re.compile(
        r"^\*\s+\S+\s+\S+\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
    )

    trace: List[Dict[str, float]] = []
    last_time = 0.0
    seen: set[tuple[float, float]] = set()

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            match = found_pattern.search(line)
            if match:
                objective = float(match.group(1))
                elapsed = float(match.group(2))
                key = (round(elapsed, 6), round(objective, 6))
                if key not in seen:
                    seen.add(key)
                    last_time = elapsed
                    trace.append(
                        {
                            "elapsed_time_s": round(elapsed, 6),
                            "objective_value": objective,
                        }
                    )
                continue

            time_match = time_pattern.search(line)
            if time_match:
                last_time = float(time_match.group(1))

            node_match = node_pattern.match(line)
            if node_match:
                objective = float(node_match.group(1))
                key = (round(last_time, 6), round(objective, 6))
                if key not in seen:
                    seen.add(key)
                    trace.append(
                        {
                            "elapsed_time_s": round(last_time, 6),
                            "objective_value": objective,
                        }
                    )

    trace.sort(key=lambda item: (item["elapsed_time_s"], item["objective_value"]))
    return trace


def _dominates(
    left: Dict[str, object],
    right: Dict[str, object],
    metrics: Sequence[str],
) -> bool:
    left_values = []
    right_values = []
    for metric in metrics:
        left_val = left.get(metric)
        right_val = right.get(metric)
        if left_val is None or right_val is None:
            return False
        left_values.append(float(left_val))
        right_values.append(float(right_val))

    return all(lv <= rv for lv, rv in zip(left_values, right_values)) and any(
        lv < rv for lv, rv in zip(left_values, right_values)
    )


def compute_pareto_frontier(
    candidates: Iterable[Dict[str, object]],
    metrics: Sequence[str] = PARETO_METRICS,
) -> List[Dict[str, object]]:
    pool = [dict(candidate) for candidate in candidates]
    frontier: List[Dict[str, object]] = []

    for candidate in pool:
        if any(_dominates(other, candidate, metrics) for other in pool if other is not candidate):
            candidate["is_nondominated"] = False
            continue
        candidate["is_nondominated"] = True
        frontier.append(candidate)

    frontier.sort(
        key=lambda item: tuple(
            float(item.get(metric, math.inf))
            for metric in metrics
        )
    )
    return frontier


def write_json(payload: Dict[str, object] | List[object], output_path: Path | str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def _safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def generate_convergence_chart(
    convergence_trace: Sequence[Dict[str, object]],
    output_path: Path | str,
    title: str = "CPLEX Convergence Curve",
) -> str | None:
    if not convergence_trace:
        return None

    plt = _safe_import_matplotlib()
    if plt is None:
        return None

    x_vals = [float(point["elapsed_time_s"]) for point in convergence_trace]
    y_vals = [float(point["objective_value"]) for point in convergence_trace]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.step(x_vals, y_vals, where="post", linewidth=2.0, color="#1f77b4")
    ax.scatter(x_vals, y_vals, color="#d62728", s=22)
    ax.set_title(title)
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Objective Value")
    ax.grid(alpha=0.3, linestyle="--")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def generate_scale_time_chart(
    solver_calls: Sequence[Dict[str, object]],
    output_path: Path | str,
) -> str | None:
    points = [
        call
        for call in solver_calls
        if call.get("solve_time_s") is not None and call.get("model_size", {}).get("variables") is not None
    ]
    if not points:
        return None

    plt = _safe_import_matplotlib()
    if plt is None:
        return None

    x_vals = [float(point["model_size"]["variables"]) for point in points]
    y_vals = [float(point["solve_time_s"]) for point in points]
    sizes = [
        max(36.0, float(point["model_size"].get("constraints", 0.0)) * 0.2)
        for point in points
    ]
    colors = [float(point.get("pending_demands", 0.0)) for point in points]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    scatter = ax.scatter(
        x_vals,
        y_vals,
        s=sizes,
        c=colors,
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.set_title("Solve Time vs Problem Size")
    ax.set_xlabel("Model Variables")
    ax.set_ylabel("Solve Time (s)")
    ax.grid(alpha=0.3, linestyle="--")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Pending Demands")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def generate_gantt_chart(
    gantt_tasks: Sequence[Dict[str, object]],
    output_path: Path | str,
) -> str | None:
    tasks = [
        task
        for task in gantt_tasks
        if task.get("start_time_h") is not None and task.get("end_time_h") is not None
    ]
    if not tasks:
        return None

    plt = _safe_import_matplotlib()
    if plt is None:
        return None

    drones = sorted({str(task.get("drone_id", "")) for task in tasks})
    if not drones:
        return None

    y_index = {drone_id: idx for idx, drone_id in enumerate(drones)}
    colors = {
        "pickup_leg": "#2ca02c",
        "delivery_leg": "#ff7f0e",
        "return_leg": "#9467bd",
    }

    fig_height = max(4.0, 0.5 * len(drones) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    for task in tasks:
        drone_id = str(task.get("drone_id", ""))
        start = float(task["start_time_h"])
        end = float(task["end_time_h"])
        duration = max(end - start, 1e-6)
        task_type = str(task.get("task_type", "delivery_leg"))
        label = str(task.get("demand_id") or task.get("to_node_id") or task_type)
        ax.barh(
            y_index[drone_id],
            duration,
            left=start,
            height=0.55,
            color=colors.get(task_type, "#1f77b4"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.88,
        )
        ax.text(start + duration / 2.0, y_index[drone_id], label, ha="center", va="center", fontsize=7)

    ax.set_title("Drone Schedule Gantt")
    ax.set_xlabel("Simulation Time (h)")
    ax.set_ylabel("Drone")
    ax.set_yticks([y_index[drone_id] for drone_id in drones], drones)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def generate_pareto_chart(
    candidates: Sequence[Dict[str, object]],
    output_path: Path | str,
) -> str | None:
    points = [
        candidate
        for candidate in candidates
        if candidate.get("final_total_distance_m") is not None
        and candidate.get("final_total_noise_impact") is not None
        and candidate.get("average_delivery_time_h") is not None
    ]
    if not points:
        return None

    plt = _safe_import_matplotlib()
    if plt is None:
        return None

    frontier = compute_pareto_frontier(points)
    frontier_ids = {str(point.get("profile_id")) for point in frontier}

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    scatter = ax.scatter(
        [float(point["final_total_distance_m"]) for point in points],
        [float(point["final_total_noise_impact"]) for point in points],
        c=[float(point["average_delivery_time_h"]) for point in points],
        cmap="plasma",
        s=[
            120.0 if str(point.get("profile_id")) in frontier_ids else 60.0
            for point in points
        ],
        edgecolors=[
            "black" if str(point.get("profile_id")) in frontier_ids else "white"
            for point in points
        ],
        linewidths=0.8,
        alpha=0.9,
    )

    for point in points:
        ax.annotate(
            str(point.get("label") or point.get("profile_id") or ""),
            (
                float(point["final_total_distance_m"]),
                float(point["final_total_noise_impact"]),
            ),
            fontsize=7,
            xytext=(5, 4),
            textcoords="offset points",
        )

    ax.set_title("Pareto Frontier")
    ax.set_xlabel("Final Total Distance (m)")
    ax.set_ylabel("Final Total Noise Impact")
    ax.grid(alpha=0.3, linestyle="--")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Average Delivery Time (h)")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def export_visualizations(
    *,
    analytics: Dict[str, object],
    output_dir: Path | str | None,
    pareto_candidates: Sequence[Dict[str, object]] | None = None,
) -> List[Dict[str, str]]:
    directory = ensure_directory(output_dir)
    if directory is None:
        return []

    artifacts: List[Dict[str, str]] = []

    convergence_points: List[Dict[str, object]] = []
    for call in analytics.get("solver_calls", []):
        convergence_points.extend(call.get("convergence_trace", []))
    if convergence_points:
        convergence_points.sort(
            key=lambda item: (
                float(item.get("elapsed_time_s", 0.0)),
                float(item.get("objective_value", 0.0)),
            )
        )
        saved = generate_convergence_chart(
            convergence_points,
            directory / "convergence_curve.png",
        )
        if saved:
            artifacts.append({"chart_type": "convergence_curve", "path": saved})

    saved = generate_scale_time_chart(
        analytics.get("solver_calls", []),
        directory / "solve_time_vs_problem_size.png",
    )
    if saved:
        artifacts.append({"chart_type": "solve_time_scale", "path": saved})

    saved = generate_gantt_chart(
        analytics.get("gantt_tasks", []),
        directory / "drone_schedule_gantt.png",
    )
    if saved:
        artifacts.append({"chart_type": "gantt", "path": saved})

    if pareto_candidates:
        saved = generate_pareto_chart(
            pareto_candidates,
            directory / "pareto_frontier.png",
        )
        if saved:
            artifacts.append({"chart_type": "pareto_frontier", "path": saved})

    return artifacts
