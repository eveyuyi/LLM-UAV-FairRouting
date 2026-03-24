from __future__ import annotations

import csv
import json
from pathlib import Path
from uuid import uuid4

from evals.eval_priority_operational_impact import summarize_operational_impact


def _make_case_dir(prefix: str) -> Path:
    path = Path.cwd() / ".test_artifacts" / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_run(run_dir: Path, per_demand_results):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "workflow_results.json", "w", encoding="utf-8") as handle:
        json.dump([{"time_window": "W1", "per_demand_results": per_demand_results}], handle, ensure_ascii=False, indent=2)


def test_operational_impact_reports_urgent_efficiency_gains():
    base = _make_case_dir("priority_operational_impact")
    dialogues = base / "dialogues.jsonl"
    dialogues.write_text("\n".join([
        json.dumps({"dialogue_id": "D1", "metadata": {"event_id": "E1", "delivery_deadline_minutes": 20}}, ensure_ascii=False),
        json.dumps({"dialogue_id": "D2", "metadata": {"event_id": "E2", "delivery_deadline_minutes": 60}}, ensure_ascii=False),
    ]), encoding="utf-8")

    gt = base / "ground_truth.csv"
    with open(gt, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "priority"])
        writer.writeheader()
        writer.writerows([
            {"event_id": "E1", "priority": 1},
            {"event_id": "E2", "priority": 4},
        ])

    baseline_run = base / "baseline"
    llm_run = base / "llm"
    _write_run(baseline_run, [
        {"demand_id": "REQ1", "source_dialogue_id": "D1", "source_event_id": "E1", "is_served": True, "delivery_latency_min": 30.0, "deadline_minutes": 20},
        {"demand_id": "REQ2", "source_dialogue_id": "D2", "source_event_id": "E2", "is_served": True, "delivery_latency_min": 40.0, "deadline_minutes": 60},
    ])
    _write_run(llm_run, [
        {"demand_id": "REQ1", "source_dialogue_id": "D1", "source_event_id": "E1", "is_served": True, "delivery_latency_min": 10.0, "deadline_minutes": 20},
        {"demand_id": "REQ2", "source_dialogue_id": "D2", "source_event_id": "E2", "is_served": True, "delivery_latency_min": 45.0, "deadline_minutes": 60},
    ])

    payload = summarize_operational_impact(
        run_dirs={"baseline": baseline_run, "llm": llm_run},
        dialogues_path=str(dialogues),
        ground_truth_csv=str(gt),
        urgent_threshold=2,
        reference_method="baseline",
    )

    assert payload["methods"]["baseline"]["priority_1"]["on_time_rate"] == 0.0
    assert payload["methods"]["llm"]["priority_1"]["on_time_rate"] == 1.0
    assert payload["comparisons"]["llm_vs_baseline"]["priority_1_on_time_rate_gain"] == 1.0
    assert payload["comparisons"]["llm_vs_baseline"]["urgent_latency_improvement"] > 0
