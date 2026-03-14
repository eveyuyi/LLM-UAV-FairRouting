import numpy as np
from scipy.spatial import cKDTree

from llm4fairrouting.routing.domain import Point
from llm4fairrouting.routing.path_costs import (
    FLIGHT_HEIGHT,
    FastRRTPlanner,
    build_lazy_distance_and_noise_matrices,
)


def _build_points():
    points = [
        Point(id="S1", lon=113.8800, lat=22.8000, alt=FLIGHT_HEIGHT, type="supply"),
        Point(id="D1", lon=113.8810, lat=22.8010, alt=FLIGHT_HEIGHT, type="demand"),
        Point(id="L1", lon=113.8820, lat=22.8020, alt=FLIGHT_HEIGHT, type="station"),
    ]
    ref_lat = np.mean([point.lat for point in points])
    ref_lon = np.mean([point.lon for point in points])
    for point in points:
        point.to_enu(ref_lat, ref_lon, 0.0)
    return points, ref_lat, ref_lon


def test_lazy_path_matrices_compute_only_on_first_access(monkeypatch):
    points, ref_lat, ref_lon = _build_points()
    residential_positions = np.array([[0.0, 0.0, 0.0]])
    residential_tree = cKDTree(residential_positions)
    plan_calls = []

    def fake_plan(self, start, goal, rng=None):
        plan_calls.append((tuple(start.tolist()), tuple(goal.tolist())))
        return 123.0, [start, goal]

    monkeypatch.setattr(FastRRTPlanner, "plan", fake_plan)
    monkeypatch.setattr(
        "llm4fairrouting.routing.path_costs.compute_path_noise_impact",
        lambda **kwargs: 7,
    )

    dist_matrix, noise_matrix = build_lazy_distance_and_noise_matrices(
        task_points=points,
        obstacles_raw=[],
        residential_positions=residential_positions,
        residential_tree=residential_tree,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        flight_height=FLIGHT_HEIGHT,
    )

    assert dist_matrix.computed_pairs == 0
    assert noise_matrix.computed_pairs == 0

    assert dist_matrix[0, 1] == 123.0
    assert dist_matrix.computed_pairs == 1
    assert len(plan_calls) == 1

    assert noise_matrix[1, 0] == 7
    assert noise_matrix.computed_pairs == 1
    assert len(plan_calls) == 1

    assert dist_matrix[0, 0] == 0.0
    assert noise_matrix[0, 0] == 0
    assert dist_matrix.computed_pairs == 1


def test_eager_and_lazy_path_builders_match_for_same_points(monkeypatch):
    points, ref_lat, ref_lon = _build_points()
    residential_positions = np.array([[0.0, 0.0, 0.0]])
    residential_tree = cKDTree(residential_positions)
    obstacle_raw = [
        {
            "lon": 113.8805,
            "lat": 22.8005,
            "alt": FLIGHT_HEIGHT,
            "radius": 120.0,
        }
    ]

    from llm4fairrouting.routing.path_costs import (
        build_realistic_distance_and_noise_matrices,
    )

    eager_dist, eager_noise = build_realistic_distance_and_noise_matrices(
        task_points=points,
        obstacles_raw=obstacle_raw,
        residential_positions=residential_positions,
        residential_tree=residential_tree,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        flight_height=FLIGHT_HEIGHT,
    )

    lazy_dist, lazy_noise = build_lazy_distance_and_noise_matrices(
        task_points=points,
        obstacles_raw=obstacle_raw,
        residential_positions=residential_positions,
        residential_tree=residential_tree,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        flight_height=FLIGHT_HEIGHT,
    )

    lazy_dist_values = np.zeros_like(eager_dist)
    lazy_noise_values = np.zeros_like(eager_noise)
    for i in range(len(points)):
        for j in range(len(points)):
            lazy_dist_values[i, j] = lazy_dist[i, j]
            lazy_noise_values[i, j] = lazy_noise[i, j]

    np.testing.assert_allclose(eager_dist, lazy_dist_values)
    np.testing.assert_array_equal(eager_noise, lazy_noise_values)
