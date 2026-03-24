from __future__ import annotations

import csv
import json
from pathlib import Path
from uuid import uuid4

from evals.eval_priority_alignment import evaluate_priority_alignment


def _make_case_dir(prefix: str) -> Path:
    path = Path.cwd() / ".test_artifacts" / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_evaluate_priority_alignment_reports_ranking_and_urgent_metrics():
    base = _make_case_dir("priority_alignment_eval")
    weights_dir = base / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    with open(weights_dir / "w1.json", "w", encoding="utf-8") as handle:
        json.dump({
            "time_window": "W1",
            "demand_configs": [
                {"demand_id": "REQ001", "priority": 1, "window_rank": 1, "reasoning": "life support"},
                {"demand_id": "REQ002", "priority": 4, "window_rank": 2, "reasoning": "routine"},
                {"demand_id": "REQ003", "priority": 2, "window_rank": 3, "reasoning": "urgent"},
            ]
        }, handle, ensure_ascii=False, indent=2)

    demands_path = base / "demands.json"
    with open(demands_path, "w", encoding="utf-8") as handle:
        json.dump([
            {
                "time_window": "W1",
                "demands": [
                    {"demand_id": "REQ001", "source_dialogue_id": "D1", "source_event_id": "E1"},
                    {"demand_id": "REQ002", "source_dialogue_id": "D2", "source_event_id": "E2"},
                    {"demand_id": "REQ003", "source_dialogue_id": "D3", "source_event_id": "E3"},
                ],
            }
        ], handle, ensure_ascii=False, indent=2)

    dialogues_path = base / "dialogues.jsonl"
    dialogues_path.write_text("\n".join([
        json.dumps({"dialogue_id": "D1", "metadata": {"event_id": "E1"}}, ensure_ascii=False),
        json.dumps({"dialogue_id": "D2", "metadata": {"event_id": "E2"}}, ensure_ascii=False),
        json.dumps({"dialogue_id": "D3", "metadata": {"event_id": "E3"}}, ensure_ascii=False),
    ]), encoding="utf-8")

    ground_truth = base / "ground_truth.csv"
    with open(ground_truth, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "priority"])
        writer.writeheader()
        writer.writerows([
            {"event_id": "E1", "priority": 1},
            {"event_id": "E2", "priority": 4},
            {"event_id": "E3", "priority": 2},
        ])

    payload = evaluate_priority_alignment(
        weights_path=str(weights_dir),
        demands_path=str(demands_path),
        dialogues_path=str(dialogues_path),
        ground_truth_csv=str(ground_truth),
        urgent_threshold=2,
    )

    assert payload["accuracy"] == 1.0
    assert payload["spearman"] == 1.0
    assert payload["priority_1_metrics"]["recall"] == 1.0
    assert payload["urgent_metrics"]["recall"] == 1.0
    assert payload["top_k_hit_rate"]["hit_rate"] == 1.0
