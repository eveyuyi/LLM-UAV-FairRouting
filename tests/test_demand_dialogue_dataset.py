from llm4fairrouting.data import demand_dialogue_dataset


def test_build_daily_demand_dialogues_wires_generation_and_save(monkeypatch, tmp_path):
    calls = {}

    monkeypatch.setattr(demand_dialogue_dataset, "create_openai_client", lambda api_base, api_key: "client")
    monkeypatch.setattr(
        demand_dialogue_dataset,
        "load_stations",
        lambda path: [{"station_id": "ST001", "name": "Hub", "lon": 113.8, "lat": 22.7}],
    )
    monkeypatch.setattr(
        demand_dialogue_dataset,
        "load_demand_events",
        lambda events_path, n_events=None, time_slots=None: [{"event_id": "EV1", "time_slot": 0}],
    )
    monkeypatch.setattr(
        demand_dialogue_dataset,
        "generate_dialogues_online",
        lambda **kwargs: [{"dialogue_id": "D0001", "conversation": "stub"}],
    )

    def _save(dialogues, output_path):
        calls["dialogues"] = dialogues
        calls["output_path"] = output_path

    monkeypatch.setattr(demand_dialogue_dataset, "save_dialogues", _save)

    output_path = tmp_path / "daily_demand_dialogues.jsonl"
    dialogues = demand_dialogue_dataset.build_daily_demand_dialogues(
        events_path="input.jsonl",
        stations_path="stations.csv",
        output_path=str(output_path),
        api_key="test-key",
    )

    assert dialogues == [{"dialogue_id": "D0001", "conversation": "stub"}]
    assert calls["dialogues"] == dialogues
    assert calls["output_path"] == str(output_path)
