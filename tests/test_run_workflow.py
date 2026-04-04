from __future__ import annotations

import json

from llm4fairrouting.workflow import run_workflow as workflow_module


def _write_dialogues_jsonl(path, dialogues) -> None:
    path.write_text(
        "\n".join(json.dumps(dialogue, ensure_ascii=False) for dialogue in dialogues) + "\n",
        encoding="utf-8",
    )


def test_filter_dialogues_by_time_slots_falls_back_to_timestamp():
    dialogues = [
        {
            "dialogue_id": "D010",
            "timestamp": "2024-03-15T00:50:00",
            "conversation": "slot 10",
            "metadata": {},
        },
        {
            "dialogue_id": "D011",
            "timestamp": "2024-03-15T00:55:00",
            "conversation": "slot 11",
            "metadata": {},
        },
    ]

    filtered = workflow_module._filter_dialogues_by_time_slots(dialogues, [11])

    assert [dialogue["dialogue_id"] for dialogue in filtered] == ["D011"]


def test_run_workflow_filters_dialogues_before_module_2(monkeypatch, tmp_path):
    dialogue_path = tmp_path / "daily_demand_dialogues.jsonl"
    output_dir = tmp_path / "results"
    _write_dialogues_jsonl(
        dialogue_path,
        [
            {
                "dialogue_id": "D000",
                "timestamp": "2024-03-15T00:00:00",
                "conversation": "slot 0",
                "metadata": {"time_slot": 0},
            },
            {
                "dialogue_id": "D001",
                "timestamp": "2024-03-15T00:05:00",
                "conversation": "slot 1",
                "metadata": {"time_slot": 1},
            },
            {
                "dialogue_id": "D002",
                "timestamp": "2024-03-15T00:10:00",
                "conversation": "slot 2",
                "metadata": {"time_slot": 2},
            },
        ],
    )

    captured = {}

    def fake_extract(dialogues, window_minutes):
        captured["dialogue_ids"] = [dialogue["dialogue_id"] for dialogue in dialogues]
        captured["window_minutes"] = window_minutes
        return [{"time_window": "2024-03-15T00:05-00:10", "demands": []}]

    monkeypatch.setattr(
        workflow_module,
        "_build_run_dir",
        lambda base_dir, model, noise_weight: base_dir / "run_for_test",
    )
    monkeypatch.setattr(workflow_module, "extract_demands_offline", fake_extract)
    monkeypatch.setattr(workflow_module, "serialize_workflow_results", lambda results: results)

    workflow_module.run_workflow(
        output_dir=str(output_dir),
        dialogue_path=str(dialogue_path),
        stations_path=str(tmp_path / "stations.csv"),
        offline=True,
        skip_solver=True,
        time_slots=[1],
        window_minutes=5,
        building_path=str(tmp_path / "buildings.csv"),
    )

    assert captured["dialogue_ids"] == ["D001"]
    assert captured["window_minutes"] == 5

    run_meta = json.loads((output_dir / "run_for_test" / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["time_slots"] == [1]


def test_run_workflow_can_use_precomputed_extracted_demands(monkeypatch, tmp_path):
    extracted_demands_path = tmp_path / "fixed_demands.json"
    output_dir = tmp_path / "results"
    extracted_demands_path.write_text(
        json.dumps(
            [
                {
                    "time_window": "2024-03-15T00:00-00:05",
                    "demands": [{"demand_id": "REQ000"}],
                },
                {
                    "time_window": "2024-03-15T00:05-00:10",
                    "demands": [{"demand_id": "REQ001"}],
                },
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    captured = {}

    monkeypatch.setattr(
        workflow_module,
        "_build_run_dir",
        lambda base_dir, model, noise_weight: base_dir / "run_for_test",
    )
    monkeypatch.setattr(
        workflow_module,
        "extract_demands_offline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Module 2 should be skipped")),
    )

    def fake_adjust_weights_offline(demands):
        captured["demand_ids"] = [d["demand_id"] for d in demands]
        return {
            "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
            "demand_configs": [{"demand_id": demands[0]["demand_id"], "priority": 1, "reasoning": "test"}],
            "supplementary_constraints": [],
        }

    monkeypatch.setattr(workflow_module, "adjust_weights_offline", fake_adjust_weights_offline)
    monkeypatch.setattr(workflow_module, "serialize_workflow_results", lambda results: results)

    workflow_module.run_workflow(
        output_dir=str(output_dir),
        extracted_demands_path=str(extracted_demands_path),
        stations_path=str(tmp_path / "stations.csv"),
        offline=True,
        skip_solver=True,
        time_slots=[1],
        building_path=str(tmp_path / "buildings.csv"),
    )

    assert captured["demand_ids"] == ["REQ001"]
    saved_demands = json.loads((output_dir / "run_for_test" / "extracted_demands.json").read_text(encoding="utf-8"))
    assert len(saved_demands) == 1
    assert saved_demands[0]["time_window"] == "2024-03-15T00:05-00:10"


def test_extract_drone_path_details_returns_first_available_solution_payload():
    details = workflow_module._extract_drone_path_details(
        [
            {"solution": None},
            {
                "solution": {
                    "drone_path_details": [
                        {
                            "drone_id": "U11",
                            "path_str": "L1 -> S1 -> D1 -> L1",
                        }
                    ]
                }
            },
        ]
    )

    assert details == [{"drone_id": "U11", "path_str": "L1 -> S1 -> D1 -> L1"}]


def test_run_workflow_writes_drone_paths_json(monkeypatch, tmp_path):
    dialogue_path = tmp_path / "daily_demand_dialogues.jsonl"
    output_dir = tmp_path / "results"
    _write_dialogues_jsonl(
        dialogue_path,
        [
            {
                "dialogue_id": "D000",
                "timestamp": "2024-03-15T00:00:00",
                "conversation": "slot 0",
                "metadata": {"time_slot": 0},
            }
        ],
    )

    extracted_windows = [
        {
            "time_window": "2024-03-15T00:00-00:05",
            "demands": [{"demand_id": "REQ001"}],
        }
    ]
    weight_config = {
        "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
        "demand_configs": [{"demand_id": "REQ001", "priority": 1, "reasoning": "test"}],
        "supplementary_constraints": [],
    }
    drone_paths = [
        {
            "drone_id": "U11",
            "path_node_ids": ["L1", "S_COM_A", "D_DEM_1", "L1"],
            "path_str": "L1 -> S_COM_A -> D_DEM_1 -> L1",
        }
    ]

    monkeypatch.setattr(
        workflow_module,
        "_build_run_dir",
        lambda base_dir, model, noise_weight: base_dir / "run_for_test",
    )
    monkeypatch.setattr(workflow_module, "extract_demands_offline", lambda dialogues, window_minutes: extracted_windows)
    monkeypatch.setattr(workflow_module, "adjust_weights_offline", lambda demands: weight_config)
    monkeypatch.setattr(
        workflow_module,
        "solve_windows_dynamically",
        lambda **kwargs: [
            {
                "time_window": "2024-03-15T00:00-00:05",
                "weight_config": weight_config,
                "feasible_demands": [{"demand_id": "REQ001"}],
                "n_demands_total": 1,
                "n_demands_filtered": 0,
                "solution": {"drone_path_details": drone_paths},
                "n_supply": 1,
            }
        ],
    )
    monkeypatch.setattr(workflow_module, "serialize_workflow_results", lambda results: results)

    workflow_module.run_workflow(
        output_dir=str(output_dir),
        dialogue_path=str(dialogue_path),
        stations_path=str(tmp_path / "stations.csv"),
        offline=True,
        skip_solver=False,
        window_minutes=5,
        building_path=str(tmp_path / "buildings.csv"),
    )

    saved_paths = json.loads((output_dir / "run_for_test" / "drone_paths.json").read_text(encoding="utf-8"))
    assert saved_paths == drone_paths


def test_run_workflow_routes_to_nsga3_backend(monkeypatch, tmp_path):
    dialogue_path = tmp_path / "daily_demand_dialogues.jsonl"
    output_dir = tmp_path / "results"
    _write_dialogues_jsonl(
        dialogue_path,
        [
            {
                "dialogue_id": "D000",
                "timestamp": "2024-03-15T00:00:00",
                "conversation": "slot 0",
                "metadata": {"time_slot": 0},
            }
        ],
    )

    extracted_windows = [
        {
            "time_window": "2024-03-15T00:00-00:05",
            "demands": [{"demand_id": "REQ001"}],
        }
    ]
    weight_config = {
        "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
        "demand_configs": [{"demand_id": "REQ001", "priority": 1, "reasoning": "test"}],
        "supplementary_constraints": [],
    }
    captured = {}

    monkeypatch.setattr(
        workflow_module,
        "_build_run_dir",
        lambda base_dir, model, noise_weight: base_dir / "run_for_test",
    )
    monkeypatch.setattr(workflow_module, "extract_demands_offline", lambda dialogues, window_minutes: extracted_windows)
    monkeypatch.setattr(workflow_module, "adjust_weights_offline", lambda demands: weight_config)
    def fake_nsga3(**kwargs):
        captured["kwargs"] = kwargs
        return {"frontier": [{"solution_id": "nsga3_gen_candidate_0000"}]}

    monkeypatch.setattr(workflow_module, "run_nsga3_pareto_search", fake_nsga3)
    monkeypatch.setattr(workflow_module, "serialize_workflow_results", lambda results: results)

    workflow_module.run_workflow(
        output_dir=str(output_dir),
        dialogue_path=str(dialogue_path),
        stations_path=str(tmp_path / "stations.csv"),
        offline=True,
        skip_solver=False,
        solver_backend="nsga3",
        nsga3_pop_size=8,
        nsga3_n_generations=4,
        nsga3_seed=123,
        building_path=str(tmp_path / "buildings.csv"),
    )

    assert captured["kwargs"]["pop_size"] == 8
    assert captured["kwargs"]["n_generations"] == 4
    assert captured["kwargs"]["seed"] == 123
    nsga3_results = json.loads((output_dir / "run_for_test" / "nsga3_results.json").read_text(encoding="utf-8"))
    assert nsga3_results["frontier"][0]["solution_id"] == "nsga3_gen_candidate_0000"


def test_run_workflow_routes_to_nsga3_heuristic_backend(monkeypatch, tmp_path):
    dialogue_path = tmp_path / "daily_demand_dialogues.jsonl"
    output_dir = tmp_path / "results"
    _write_dialogues_jsonl(
        dialogue_path,
        [
            {
                "dialogue_id": "D000",
                "timestamp": "2024-03-15T00:00:00",
                "conversation": "slot 0",
                "metadata": {"time_slot": 0},
            }
        ],
    )

    extracted_windows = [
        {
            "time_window": "2024-03-15T00:00-00:05",
            "demands": [{"demand_id": "REQ001"}],
        }
    ]
    weight_config = {
        "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
        "demand_configs": [{"demand_id": "REQ001", "priority": 1, "reasoning": "test"}],
        "supplementary_constraints": [],
    }
    captured = {}

    monkeypatch.setattr(
        workflow_module,
        "_build_run_dir",
        lambda base_dir, model, noise_weight: base_dir / "run_for_test",
    )
    monkeypatch.setattr(workflow_module, "extract_demands_offline", lambda dialogues, window_minutes: extracted_windows)
    monkeypatch.setattr(workflow_module, "adjust_weights_offline", lambda demands: weight_config)

    def fake_nsga3_heuristic(**kwargs):
        captured["kwargs"] = kwargs
        return {"frontier": [{"solution_id": "nsga3_heuristic_candidate_0000"}]}

    monkeypatch.setattr(workflow_module, "run_nsga3_heuristic_search", fake_nsga3_heuristic)
    monkeypatch.setattr(workflow_module, "serialize_workflow_results", lambda results: results)

    workflow_module.run_workflow(
        output_dir=str(output_dir),
        dialogue_path=str(dialogue_path),
        stations_path=str(tmp_path / "stations.csv"),
        offline=True,
        skip_solver=False,
        solver_backend="nsga3_heuristic",
        nsga3_pop_size=5,
        nsga3_n_generations=2,
        nsga3_seed=55,
        building_path=str(tmp_path / "buildings.csv"),
    )

    assert captured["kwargs"]["pop_size"] == 5
    assert captured["kwargs"]["n_generations"] == 2
    assert captured["kwargs"]["seed"] == 55
    saved = json.loads((output_dir / "run_for_test" / "nsga3_heuristic_results.json").read_text(encoding="utf-8"))
    assert saved["frontier"][0]["solution_id"] == "nsga3_heuristic_candidate_0000"


def test_run_workflow_writes_representative_frontier_solution_for_nsga3_heuristic(monkeypatch, tmp_path):
    dialogue_path = tmp_path / "daily_demand_dialogues.jsonl"
    output_dir = tmp_path / "results"
    representative_path = tmp_path / "frontier_solution.json"
    representative_payload = [
        {
            "time_window": "2024-03-15T00:00-00:05",
            "drone_path_details": [
                {
                    "drone_id": "U11",
                    "path_str": "L1 -> S1 -> D1 -> L1",
                }
            ],
        }
    ]
    representative_path.write_text(json.dumps(representative_payload), encoding="utf-8")
    _write_dialogues_jsonl(
        dialogue_path,
        [
            {
                "dialogue_id": "D000",
                "timestamp": "2024-03-15T00:00:00",
                "conversation": "slot 0",
                "metadata": {"time_slot": 0},
            }
        ],
    )

    extracted_windows = [
        {
            "time_window": "2024-03-15T00:00-00:05",
            "demands": [{"demand_id": "REQ001"}],
        }
    ]
    weight_config = {
        "global_weights": {"w_distance": 1.0, "w_time": 1.0, "w_risk": 1.0},
        "demand_configs": [{"demand_id": "REQ001", "priority": 1, "reasoning": "test"}],
        "supplementary_constraints": [],
    }

    monkeypatch.setattr(
        workflow_module,
        "_build_run_dir",
        lambda base_dir, model, noise_weight: base_dir / "run_for_test",
    )
    monkeypatch.setattr(workflow_module, "extract_demands_offline", lambda dialogues, window_minutes: extracted_windows)
    monkeypatch.setattr(workflow_module, "adjust_weights_offline", lambda demands: weight_config)
    monkeypatch.setattr(
        workflow_module,
        "run_nsga3_heuristic_search",
        lambda **kwargs: {
            "frontier": [
                {
                    "solution_id": "sol1",
                    "final_total_distance_m": 10.0,
                    "average_delivery_time_h": 1.0,
                    "final_total_noise_impact": 1.0,
                    "service_rate_loss": 0.0,
                    "n_used_drones": 1,
                    "frontier_result_path": str(representative_path),
                }
            ]
        },
    )
    monkeypatch.setattr(workflow_module, "serialize_workflow_results", lambda results: results)

    workflow_module.run_workflow(
        output_dir=str(output_dir),
        dialogue_path=str(dialogue_path),
        stations_path=str(tmp_path / "stations.csv"),
        offline=True,
        skip_solver=False,
        solver_backend="nsga3_heuristic",
        building_path=str(tmp_path / "buildings.csv"),
    )

    saved_workflow = json.loads((output_dir / "run_for_test" / "workflow_results.json").read_text(encoding="utf-8"))
    saved_paths = json.loads((output_dir / "run_for_test" / "drone_paths.json").read_text(encoding="utf-8"))

    assert saved_workflow == representative_payload
    assert saved_paths == representative_payload[0]["drone_path_details"]
