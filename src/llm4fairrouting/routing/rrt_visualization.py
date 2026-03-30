"""Helpers for exporting 3D and top-down RRT path visualizations from saved workflow artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from llm4fairrouting.routing.analytics import PARETO_METRICS


NODE_STYLES = {
    "station": {"label": "Station", "color": "#1f77b4", "marker": "s"},
    "supply": {"label": "Supply", "color": "#2ca02c", "marker": "o"},
    "demand": {"label": "Demand", "color": "#ff7f0e", "marker": "^"},
    "other": {"label": "Other", "color": "#7f7f7f", "marker": "x"},
}


def _safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        return plt
    except Exception:
        return None


def _load_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(base_path: Path, candidate: str | Path | None) -> Optional[Path]:
    if candidate in (None, ""):
        return None
    raw = Path(candidate)
    attempts = []
    if raw.is_absolute():
        attempts.append(raw)
    else:
        attempts.extend([
            Path.cwd() / raw,
            base_path.parent / raw,
            raw,
        ])
    for attempt in attempts:
        if attempt.exists():
            return attempt.resolve()
    return None


def _infer_node_type(node_id: object) -> str:
    text = str(node_id or "").strip().upper()
    if text.startswith("L"):
        return "station"
    if text.startswith("S"):
        return "supply"
    if text.startswith("D"):
        return "demand"
    return "other"


def _collect_nodes(rrt_paths: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    nodes: Dict[str, Dict[str, object]] = {}
    for path in rrt_paths:
        waypoints = path.get("path_xyz") or []
        if len(waypoints) < 2:
            continue
        start_id = str(path.get("from_id") or f"start_{len(nodes)}")
        end_id = str(path.get("to_id") or f"end_{len(nodes)}")
        nodes.setdefault(
            start_id,
            {
                "node_id": start_id,
                "node_type": _infer_node_type(start_id),
                "coords": [float(v) for v in waypoints[0]],
            },
        )
        nodes.setdefault(
            end_id,
            {
                "node_id": end_id,
                "node_type": _infer_node_type(end_id),
                "coords": [float(v) for v in waypoints[-1]],
            },
        )
    return list(nodes.values())


def resolve_rrt_paths_json_from_workflow_results(workflow_results_path: str | Path) -> Optional[Path]:
    workflow_path = Path(workflow_results_path)
    payload = _load_json(workflow_path)
    if isinstance(payload, list):
        for item in payload:
            artifacts = item.get("analytics_artifacts") or {}
            resolved = _resolve_path(workflow_path, artifacts.get("rrt_paths_json"))
            if resolved is not None:
                return resolved

    fallback = workflow_path.parent / "solver_analytics" / "rrt_paths.json"
    if fallback.exists():
        return fallback.resolve()
    return None


def select_representative_frontier_solution(
    frontier: Sequence[Dict[str, object]],
    metrics: Sequence[str] = PARETO_METRICS,
) -> Optional[Dict[str, object]]:
    candidates = [
        dict(item)
        for item in frontier
        if all(item.get(metric) is not None for metric in metrics)
    ]
    if not candidates:
        return dict(frontier[0]) if frontier else None

    spans: Dict[str, tuple[float, float]] = {}
    for metric in metrics:
        values = [float(item[metric]) for item in candidates]
        min_val = min(values)
        max_val = max(values)
        if max_val <= min_val:
            max_val = min_val + 1.0
        spans[metric] = (min_val, max_val)

    def score(item: Dict[str, object]) -> tuple[float, str]:
        total = 0.0
        for metric in metrics:
            min_val, max_val = spans[metric]
            total += (float(item[metric]) - min_val) / (max_val - min_val)
        return total, str(item.get("solution_id") or item.get("profile_id") or "")

    return min(candidates, key=score)


def _plot_node_markers_3d(ax, nodes: Sequence[Dict[str, object]]) -> None:
    legend_done = set()
    for node in nodes:
        node_type = str(node.get("node_type") or "other")
        style = NODE_STYLES.get(node_type, NODE_STYLES["other"])
        x, y, z = [float(v) for v in node.get("coords", [0.0, 0.0, 0.0])]
        label = style["label"] if node_type not in legend_done else None
        ax.scatter(x, y, z, color=style["color"], marker=style["marker"], s=48, label=label)
        ax.text(x, y, z, str(node.get("node_id", "")), fontsize=7)
        legend_done.add(node_type)


def _plot_node_markers_2d(ax, nodes: Sequence[Dict[str, object]]) -> None:
    legend_done = set()
    for node in nodes:
        node_type = str(node.get("node_type") or "other")
        style = NODE_STYLES.get(node_type, NODE_STYLES["other"])
        x, y = [float(v) for v in node.get("coords", [0.0, 0.0])[:2]]
        label = style["label"] if node_type not in legend_done else None
        ax.scatter(x, y, color=style["color"], marker=style["marker"], s=54, label=label, edgecolors="black", linewidths=0.4)
        ax.text(x, y, str(node.get("node_id", "")), fontsize=7)
        legend_done.add(node_type)


def generate_rrt_3d_chart(
    rrt_paths: Sequence[Dict[str, object]],
    output_path: str | Path,
    *,
    title: str = "RRT 3D Paths",
) -> Optional[str]:
    paths = [item for item in rrt_paths if item.get("path_xyz")]
    if not paths:
        return None

    plt = _safe_import_matplotlib()
    if plt is None:
        return None

    nodes = _collect_nodes(paths)
    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")

    for idx, path in enumerate(paths):
        waypoints = path.get("path_xyz") or []
        if len(waypoints) < 2:
            continue
        xs = [float(point[0]) for point in waypoints]
        ys = [float(point[1]) for point in waypoints]
        zs = [float(point[2]) for point in waypoints]
        color = cmap(idx % 20)
        label = f"{path.get('from_id', '?')}->{path.get('to_id', '?')}"
        ax.plot(xs, ys, zs, color=color, linewidth=2.0, alpha=0.95, label=label if idx < 8 else None)

    _plot_node_markers_3d(ax, nodes)
    ax.set_title(title)
    ax.set_xlabel("Projected X")
    ax.set_ylabel("Projected Y")
    ax.set_zlabel("Altitude")
    ax.grid(True, alpha=0.25)
    if len(paths) <= 8 or nodes:
        ax.legend(loc="best", fontsize=7)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def generate_rrt_topdown_chart(
    rrt_paths: Sequence[Dict[str, object]],
    output_path: str | Path,
    *,
    title: str = "RRT Top-Down View",
) -> Optional[str]:
    paths = [item for item in rrt_paths if item.get("path_xyz")]
    if not paths:
        return None

    plt = _safe_import_matplotlib()
    if plt is None:
        return None

    nodes = _collect_nodes(paths)
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    cmap = plt.get_cmap("tab20")

    for idx, path in enumerate(paths):
        waypoints = path.get("path_xyz") or []
        if len(waypoints) < 2:
            continue
        xs = [float(point[0]) for point in waypoints]
        ys = [float(point[1]) for point in waypoints]
        color = cmap(idx % 20)
        label = f"{path.get('from_id', '?')}->{path.get('to_id', '?')}"
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.95, label=label if idx < 8 else None)

    _plot_node_markers_2d(ax, nodes)
    ax.set_title(title)
    ax.set_xlabel("Projected X")
    ax.set_ylabel("Projected Y")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_aspect("equal", adjustable="datalim")
    if len(paths) <= 8 or nodes:
        ax.legend(loc="best", fontsize=7)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def export_rrt_3d_from_workflow_results(
    workflow_results_path: str | Path,
    output_path: str | Path,
    *,
    title: str,
    topdown_output_path: str | Path | None = None,
) -> Optional[Dict[str, object]]:
    rrt_json = resolve_rrt_paths_json_from_workflow_results(workflow_results_path)
    if rrt_json is None:
        return None

    payload = _load_json(rrt_json)
    if not isinstance(payload, list):
        return None

    chart_path = generate_rrt_3d_chart(payload, output_path, title=title)
    if chart_path is None:
        return None

    export_payload = {
        "workflow_results_path": str(Path(workflow_results_path)),
        "rrt_paths_json": str(rrt_json),
        "chart_path": chart_path,
        "n_rrt_paths": len(payload),
    }
    if topdown_output_path is not None:
        topdown_chart_path = generate_rrt_topdown_chart(
            payload,
            topdown_output_path,
            title=f"{title} (Top-Down)",
        )
        if topdown_chart_path is not None:
            export_payload["topdown_chart_path"] = topdown_chart_path
    return export_payload
