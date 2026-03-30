from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path.cwd() / "scripts" / "run_extract100_eval_then_plan1.py"
    spec = importlib.util.spec_from_file_location("run_extract100_eval_then_plan1", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_slot_range_and_workflow_command_building():
    module = _load_script_module()

    assert module._slot_range(3, start=5) == [5, 6, 7]

    command = module._build_workflow_command(
        python_exe="python",
        output_dir=Path("out"),
        dialogue_path="dialogues.jsonl",
        stations_path="stations.csv",
        building_path="buildings.csv",
        time_slots=[0, 1, 2],
        window_minutes=5,
        time_limit=180,
        max_solver_stations=1,
        max_drones_per_station=3,
        max_payload=60.0,
        max_range=200000.0,
        noise_weight=0.5,
        drone_activation_cost=1000.0,
        drone_speed=60.0,
        solver_backend="nsga3_heuristic",
        offline=False,
        skip_solver=False,
        nsga3_pop_size=4,
        nsga3_n_generations=2,
        nsga3_seed=42,
    )

    assert command[:3] == ["python", "-m", "llm4fairrouting.workflow.run_workflow"]
    assert "--time-slots" in command
    slot_index = command.index("--time-slots")
    assert command[slot_index + 1:slot_index + 4] == ["0", "1", "2"]
    assert command[command.index("--solver-backend") + 1] == "nsga3_heuristic"
    assert "--nsga3-pop-size" in command
    assert "--skip-solver" not in command


def test_priority_alignment_command_building():
    module = _load_script_module()
    command = module._build_priority_alignment_command(
        python_exe="python",
        weights_path=Path("run") / "weight_configs",
        demands_path=Path("run") / "extracted_demands.json",
        dialogues_path="dialogues.jsonl",
        ground_truth_path="events.jsonl",
        urgent_threshold=2,
        output_path=Path("evals") / "priority_alignment.json",
    )

    assert command[0] == "python"
    assert command[1].endswith("eval_priority_alignment.py")
    assert command[command.index("--weights") + 1].endswith("weight_configs")
    assert command[command.index("--demands") + 1].endswith("extracted_demands.json")
    assert command[command.index("--output") + 1].endswith("priority_alignment.json")
