from pathlib import Path

from llm4fairrouting.routing.analytics import (
    analyze_pareto_candidates,
    compute_pareto_frontier,
    export_visualizations,
    parse_cplex_incumbent_trace,
)


def test_parse_cplex_incumbent_trace_reads_multiple_formats():
    artifacts_dir = Path.cwd() / ".test_artifacts" / "solver_analytics"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifacts_dir / "cplex.log"
    log_path.write_text(
        "\n".join(
            [
                "Found incumbent of value 1450.5 after 0.12 sec.",
                "Time = 0.40 sec.  Dettime = 1.20 ticks",
                "*  12+  3  1200.0  0.0  0.00%",
            ]
        ),
        encoding="utf-8",
    )

    trace = parse_cplex_incumbent_trace(log_path)

    assert trace[0]["elapsed_time_s"] == 0.12
    assert trace[0]["objective_value"] == 1450.5
    assert trace[-1]["elapsed_time_s"] == 0.4
    assert trace[-1]["objective_value"] == 1200.0


def test_compute_pareto_frontier_marks_only_nondominated_points():
    frontier = compute_pareto_frontier(
        [
            {
                "profile_id": "balanced",
                "final_total_distance_m": 100.0,
                "average_delivery_time_h": 2.0,
                "final_total_noise_impact": 5.0,
                "service_rate_loss": 0.2,
                "n_used_drones": 2,
            },
            {
                "profile_id": "distance_focus",
                "final_total_distance_m": 90.0,
                "average_delivery_time_h": 2.5,
                "final_total_noise_impact": 4.0,
                "service_rate_loss": 0.25,
                "n_used_drones": 2,
            },
            {
                "profile_id": "dominated",
                "final_total_distance_m": 120.0,
                "average_delivery_time_h": 2.8,
                "final_total_noise_impact": 6.0,
                "service_rate_loss": 0.3,
                "n_used_drones": 3,
            },
        ]
    )

    assert {point["profile_id"] for point in frontier} == {"balanced", "distance_focus"}


def test_analyze_pareto_candidates_reports_extremes():
    analysis = analyze_pareto_candidates(
        [
            {
                "solution_id": "A",
                "final_total_distance_m": 100.0,
                "average_delivery_time_h": 2.0,
                "final_total_noise_impact": 5.0,
                "service_rate_loss": 0.2,
                "n_used_drones": 2,
            },
            {
                "solution_id": "B",
                "final_total_distance_m": 90.0,
                "average_delivery_time_h": 2.5,
                "final_total_noise_impact": 4.0,
                "service_rate_loss": 0.25,
                "n_used_drones": 2,
            },
            {
                "solution_id": "C",
                "final_total_distance_m": 120.0,
                "average_delivery_time_h": 2.8,
                "final_total_noise_impact": 6.0,
                "service_rate_loss": 0.3,
                "n_used_drones": 3,
            },
        ]
    )

    assert analysis["frontier_size"] == 2
    assert analysis["extreme_solutions"]["final_total_distance_m"]["best_solution_id"] == "B"
    assert analysis["metric_ranges"]["final_total_noise_impact"]["spread"] == 2.0


def test_export_visualizations_generates_parallel_coordinates_chart():
    output_dir = Path.cwd() / ".test_artifacts" / "solver_analytics_charts"
    artifacts = export_visualizations(
        analytics={"solver_calls": [], "gantt_tasks": []},
        output_dir=output_dir,
        pareto_candidates=[
            {
                "solution_id": "A",
                "final_total_distance_m": 100.0,
                "average_delivery_time_h": 2.0,
                "final_total_noise_impact": 5.0,
                "service_rate_loss": 0.2,
                "n_used_drones": 2,
            },
            {
                "solution_id": "B",
                "final_total_distance_m": 90.0,
                "average_delivery_time_h": 2.5,
                "final_total_noise_impact": 4.0,
                "service_rate_loss": 0.25,
                "n_used_drones": 2,
            },
            {
                "solution_id": "C",
                "final_total_distance_m": 120.0,
                "average_delivery_time_h": 2.8,
                "final_total_noise_impact": 6.0,
                "service_rate_loss": 0.3,
                "n_used_drones": 3,
            },
        ],
    )

    chart_paths = {item["chart_type"]: Path(item["path"]) for item in artifacts}
    assert "pareto_parallel_coordinates" in chart_paths
    assert chart_paths["pareto_parallel_coordinates"].exists()
