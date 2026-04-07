from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_analyze_hard_eval_subtypes_for_existing_eval_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "model_a_seed5101"
    pre_payload = {
        "per_item": [
            {
                "time_window": "hard_window::surface_contradiction_1",
                "demand_id": "D1",
                "event_id": "E1",
                "true_priority": 1,
                "pred_priority": 2,
                "window_rank": 1,
            },
            {
                "time_window": "hard_window::near_tie_1",
                "demand_id": "D2",
                "event_id": "E2",
                "true_priority": 2,
                "pred_priority": 3,
                "window_rank": 1,
            },
            {
                "time_window": "2024-03-15T00:00-00:05::counterfactual_1",
                "demand_id": "D3",
                "event_id": "E3",
                "true_priority": 1,
                "pred_priority": 2,
                "window_rank": 1,
            },
        ]
    }
    post_payload = {
        "per_item": [
            {
                "time_window": "hard_window::surface_contradiction_1",
                "demand_id": "D1",
                "event_id": "E1",
                "true_priority": 1,
                "pred_priority": 1,
                "window_rank": 1,
            },
            {
                "time_window": "hard_window::near_tie_1",
                "demand_id": "D2",
                "event_id": "E2",
                "true_priority": 2,
                "pred_priority": 3,
                "window_rank": 1,
            },
            {
                "time_window": "2024-03-15T00:00-00:05::counterfactual_1",
                "demand_id": "D3",
                "event_id": "E3",
                "true_priority": 1,
                "pred_priority": 1,
                "window_rank": 1,
            },
        ]
    }
    summary_payload = {
        "headline": "hard eval",
        "truth_source": "fixed_demands",
    }

    _write_json(run_dir / "evals" / "pre_alignment.json", pre_payload)
    _write_json(run_dir / "evals" / "post_alignment.json", post_payload)
    _write_json(run_dir / "evals" / "summary.json", summary_payload)

    subprocess.run(
        [sys.executable, "scripts/analyze_hard_eval_subtypes.py", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
    )

    breakdown = json.loads((run_dir / "evals" / "hard_subtype_breakdown.json").read_text(encoding="utf-8"))
    assert breakdown["subsets"]["surface_contradiction"]["verdict"]["overall"] == "post_better"
    assert breakdown["subsets"]["near_tie"]["verdict"]["overall"] == "mixed"
    assert breakdown["subsets"]["counterfactual"]["delta_post_minus_pre"]["accuracy"] == 1.0

    leaderboard = (tmp_path / "hard_subtype_leaderboard.jsonl").read_text(encoding="utf-8").splitlines()
    assert any('"subset": "surface_contradiction"' in line for line in leaderboard)
