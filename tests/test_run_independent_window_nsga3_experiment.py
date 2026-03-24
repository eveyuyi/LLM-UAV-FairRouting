from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_script_module():
    script_path = Path.cwd() / "scripts" / "run_independent_window_nsga3_experiment.py"
    spec = importlib.util.spec_from_file_location("run_independent_window_nsga3_experiment", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sample_time_slots_is_deterministic(tmp_path):
    module = _load_script_module()
    base = tmp_path
    dialogues = base / "dialogues.jsonl"
    dialogues.write_text(
        "\n".join(
            json.dumps(
                {
                    "dialogue_id": f"D{slot:03d}",
                    "timestamp": f"2024-03-15T{slot // 12:02d}:{(slot % 12) * 5:02d}:00",
                    "metadata": {"time_slot": slot},
                },
                ensure_ascii=False,
            )
            for slot in range(8)
        ),
        encoding="utf-8",
    )

    selected = module._sample_time_slots(dialogues, sample_size=3, seed=7)
    assert selected == module._sample_time_slots(dialogues, sample_size=3, seed=7)
    assert len(selected) == 3
    assert selected == sorted(selected)


def test_evaluate_demand_extraction_reports_tp_fp_fn_and_exact_match(tmp_path):
    module = _load_script_module()
    base = tmp_path

    dialogues = base / "dialogues.jsonl"
    dialogues.write_text(
        "\n".join(
            [
                json.dumps({"dialogue_id": "D1", "timestamp": "2024-03-15T00:00:00", "metadata": {"time_slot": 0, "event_id": "E1"}}, ensure_ascii=False),
                json.dumps({"dialogue_id": "D2", "timestamp": "2024-03-15T00:05:00", "metadata": {"time_slot": 1, "event_id": "E2"}}, ensure_ascii=False),
                json.dumps({"dialogue_id": "D3", "timestamp": "2024-03-15T00:10:00", "metadata": {"time_slot": 2, "event_id": "E3"}}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    ground_truth = base / "ground_truth.csv"
    with open(ground_truth, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["unique_id", "priority", "supply_fid", "demand_fid", "material_weight"])
        writer.writeheader()
        writer.writerows(
            [
                {"unique_id": "E1", "priority": 1, "supply_fid": "SUP1", "demand_fid": "DEM1", "material_weight": 5.0},
                {"unique_id": "E2", "priority": 2, "supply_fid": "SUP2", "demand_fid": "DEM2", "material_weight": 8.0},
            ]
        )

    extracted = base / "extracted_demands.json"
    extracted.write_text(
        json.dumps(
            [
                {
                    "time_window": "2024-03-15T00:00-00:05",
                    "demands": [
                        {
                            "source_dialogue_id": "D1",
                            "source_event_id": "E1",
                            "origin": {"fid": "SUP1"},
                            "destination": {"fid": "DEM1"},
                            "cargo": {"weight_kg": 5.0},
                        },
                        {
                            "source_dialogue_id": "D1",
                            "source_event_id": "E1",
                            "origin": {"fid": "SUP1"},
                            "destination": {"fid": "DEM1"},
                            "cargo": {"weight_kg": 5.0},
                        },
                    ],
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = module._evaluate_demand_extraction(
        extracted_demands_path=extracted,
        dialogues_path=dialogues,
        ground_truth_csv=ground_truth,
        selected_slots=[0, 1],
    )

    assert payload["tp"] == 1
    assert payload["fp"] == 1
    assert payload["fn"] == 1
    assert payload["duplicates"] == 1
    assert payload["precision"] == 0.5
    assert payload["recall"] == 0.5
    assert payload["f1"] == 0.5
    assert payload["exact_match_rate"] == 1.0


def test_build_solver_command_and_summarize_nsga3_frontier(tmp_path):
    module = _load_script_module()
    base = tmp_path
    results_path = base / "nsga3_results.json"
    results_path.write_text(
        json.dumps(
            {
                "search_meta": {
                    "n_candidates_evaluated": 4,
                    "search_runtime_s": 12.0,
                    "avg_candidate_runtime_s": 3.0,
                },
                "frontier": [
                    {
                        "solution_id": "sol_a",
                        "label": "A",
                        "final_total_distance_m": 100.0,
                        "average_delivery_time_h": 1.0,
                        "final_total_noise_impact": 10.0,
                        "service_rate_loss": 0.1,
                        "n_used_drones": 1,
                        "frontier_result_path": "a.json",
                        "run_summary": {"service_rate": 0.9},
                    },
                    {
                        "solution_id": "sol_b",
                        "label": "B",
                        "final_total_distance_m": 130.0,
                        "average_delivery_time_h": 0.8,
                        "final_total_noise_impact": 8.0,
                        "service_rate_loss": 0.2,
                        "n_used_drones": 2,
                        "frontier_result_path": "b.json",
                        "run_summary": {"service_rate": 0.8},
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary, frontier_rows = module._summarize_nsga3_frontier(
        results_path=results_path,
        slot=7,
        time_window="2024-03-15T00:35-00:40",
    )

    assert summary["frontier_size"] == 2
    assert summary["n_candidates_evaluated"] == 4
    assert summary["search_runtime_s"] == 12.0
    assert summary["representative_solution_id"] == "sol_a"
    assert summary["best_service_rate"] == 0.9
    assert summary["min_distance_m"] == 100.0
    assert len(frontier_rows) == 2

    command = module._build_solver_command(
        python_exe="python",
        demands_path=Path("demands.json"),
        weights_path=Path("weights.json"),
        stations_path="stations.csv",
        building_path="buildings.csv",
        output_path=Path("out.json"),
        time_limit=180,
        max_solver_stations=1,
        max_drones_per_station=3,
        max_payload=60.0,
        max_range=200000.0,
        noise_weight=0.5,
        drone_activation_cost=1000.0,
        drone_speed=60.0,
        solver_backend="nsga3",
        nsga3_pop_size=4,
        nsga3_n_generations=2,
        nsga3_seed=42,
    )

    assert command[:3] == ["python", "-m", "llm4fairrouting.workflow.solver_adapter"]
    assert command[command.index("--solver-backend") + 1] == "nsga3"
    assert command[command.index("--nsga3-pop-size") + 1] == "4"
