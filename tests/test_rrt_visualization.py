from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from llm4fairrouting.routing.rrt_visualization import (
    export_rrt_3d_from_workflow_results,
    resolve_rrt_paths_json_from_workflow_results,
    select_representative_frontier_solution,
)


def _make_case_dir(prefix: str) -> Path:
    path = Path.cwd() / ".test_artifacts" / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_resolve_rrt_paths_and_export_3d_chart_from_workflow_results():
    base = _make_case_dir("rrt_visualization")
    analytics_dir = base / "solver_analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    rrt_json = analytics_dir / "rrt_paths.json"
    with open(rrt_json, "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "from_id": "L1",
                    "to_id": "S1",
                    "path_xyz": [[0.0, 0.0, 50.0], [20.0, 5.0, 50.0], [35.0, 8.0, 52.0]],
                    "n_waypoints": 3,
                    "path_length_m": 36.2,
                    "noise_impact": 2,
                },
                {
                    "from_id": "S1",
                    "to_id": "D1",
                    "path_xyz": [[35.0, 8.0, 52.0], [48.0, 16.0, 52.0], [60.0, 20.0, 50.0]],
                    "n_waypoints": 3,
                    "path_length_m": 28.4,
                    "noise_impact": 1,
                },
            ],
            handle,
            ensure_ascii=False,
            indent=2,
        )

    workflow_results = [
        {
            "time_window": "2024-03-15T00:00-00:05",
            "analytics_artifacts": {"rrt_paths_json": str(rrt_json)},
            "run_summary": {},
            "drone_path_details": [],
        }
    ]
    workflow_path = base / "workflow_results.json"
    with open(workflow_path, "w", encoding="utf-8") as handle:
        json.dump(workflow_results, handle, ensure_ascii=False, indent=2)

    resolved = resolve_rrt_paths_json_from_workflow_results(workflow_path)
    assert resolved == rrt_json.resolve()

    payload = export_rrt_3d_from_workflow_results(
        workflow_path,
        base / "charts" / "rrt_3d.png",
        title="Test RRT 3D",
        topdown_output_path=base / "charts" / "rrt_topdown.png",
    )
    assert payload is not None
    assert Path(payload["chart_path"]).exists()
    assert Path(payload["topdown_chart_path"]).exists()
    assert payload["n_rrt_paths"] == 2


def test_select_representative_frontier_solution_prefers_balanced_candidate():
    representative = select_representative_frontier_solution(
        [
            {
                "solution_id": "distance_extreme",
                "final_total_distance_m": 10.0,
                "average_delivery_time_h": 5.0,
                "final_total_noise_impact": 5.0,
                "service_rate_loss": 0.5,
                "n_used_drones": 3,
            },
            {
                "solution_id": "balanced",
                "final_total_distance_m": 20.0,
                "average_delivery_time_h": 2.0,
                "final_total_noise_impact": 2.0,
                "service_rate_loss": 0.1,
                "n_used_drones": 1,
            },
            {
                "solution_id": "noise_extreme",
                "final_total_distance_m": 30.0,
                "average_delivery_time_h": 1.0,
                "final_total_noise_impact": 1.0,
                "service_rate_loss": 0.2,
                "n_used_drones": 2,
            },
        ]
    )

    assert representative is not None
    assert representative["solution_id"] == "balanced"
