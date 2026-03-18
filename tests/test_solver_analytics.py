from llm4fairrouting.routing.analytics import (
    compute_pareto_frontier,
    parse_cplex_incumbent_trace,
)


def test_parse_cplex_incumbent_trace_reads_multiple_formats(tmp_path):
    log_path = tmp_path / "cplex.log"
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
            },
            {
                "profile_id": "distance_focus",
                "final_total_distance_m": 90.0,
                "average_delivery_time_h": 2.5,
                "final_total_noise_impact": 4.0,
            },
            {
                "profile_id": "dominated",
                "final_total_distance_m": 120.0,
                "average_delivery_time_h": 2.8,
                "final_total_noise_impact": 6.0,
            },
        ]
    )

    assert {point["profile_id"] for point in frontier} == {"balanced", "distance_focus"}
