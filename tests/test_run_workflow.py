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
