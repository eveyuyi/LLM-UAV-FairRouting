from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_split_hard_eval_windows_for_root(tmp_path: Path) -> None:
    seed_1 = tmp_path / "seed_5101"
    seed_2 = tmp_path / "seed_5102"
    _write_jsonl(
        seed_1 / "llm3_grpo_hard.jsonl",
        [
            {"time_window": "2024-03-15T00:00-00:05::counterfactual_1"},
            {"time_window": "hard_window::surface_contradiction_1"},
            {"time_window": "hard_window::near_tie_1"},
            {"time_window": "hard_window::mixed_priority"},
            {"time_window": "hard_window::mystery_case"},
        ],
    )
    _write_jsonl(
        seed_2 / "llm3_grpo_hard.jsonl",
        [
            {"time_window": "hard_window::surface_contradiction_2"},
            {"time_window": "hard_window::near_tie_2"},
        ],
    )

    subprocess.run(
        [sys.executable, "scripts/split_hard_eval_windows.py", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
    )

    manifest_1 = json.loads((seed_1 / "hard_eval_subsets" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_1["counts"]["counterfactual"] == 1
    assert manifest_1["counts"]["surface_contradiction"] == 1
    assert manifest_1["counts"]["near_tie"] == 1
    assert manifest_1["counts"]["mixed_priority"] == 1
    assert manifest_1["counts"]["other"] == 1

    counterfactual_rows = [
        json.loads(line)
        for line in (seed_1 / "hard_eval_subsets" / "counterfactual.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(counterfactual_rows) == 1
    assert "counterfactual" in counterfactual_rows[0]["time_window"]

    aggregate_manifest = json.loads((tmp_path / "hard_eval_subsets" / "aggregate_manifest.json").read_text(encoding="utf-8"))
    assert len(aggregate_manifest["datasets"]) == 2
