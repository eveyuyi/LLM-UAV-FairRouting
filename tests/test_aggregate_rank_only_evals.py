from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_summary(root: Path, name: str, delta_priority_1: float, delta_top_k: float) -> None:
    run_dir = root / name / "evals"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "headline": f"{name} summary",
        "truth_source": "fixed_demands",
        "sample": {"n_selected_windows": 10},
        "verdicts": {"primary": {"overall": "mixed"}},
        "alignment": {
            "pre": {
                "accuracy": 0.7,
                "macro_f1": 0.6,
                "weighted_f1": 0.7,
                "spearman": 0.8,
                "kendall_tau": 0.7,
                "top_k_hit_rate": 0.5,
                "priority_1_recall": 0.4,
                "priority_1_f1": 0.5,
                "urgent_recall": 0.8,
                "urgent_f1": 0.7,
                "n_aligned_demands": 20,
            },
            "post": {
                "accuracy": 0.7,
                "macro_f1": 0.6,
                "weighted_f1": 0.7,
                "spearman": 0.8,
                "kendall_tau": 0.7,
                "top_k_hit_rate": 0.5 + delta_top_k,
                "priority_1_recall": 0.4 + delta_priority_1,
                "priority_1_f1": 0.5,
                "urgent_recall": 0.8,
                "urgent_f1": 0.7,
                "n_aligned_demands": 20,
            },
            "delta_post_minus_pre": {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "spearman": 0.0,
                "kendall_tau": 0.0,
                "top_k_hit_rate": delta_top_k,
                "priority_1_recall": delta_priority_1,
                "priority_1_f1": 0.0,
                "urgent_recall": 0.0,
                "urgent_f1": 0.0,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_aggregate_rank_only_evals(tmp_path: Path) -> None:
    _write_summary(tmp_path, "run_a", delta_priority_1=0.2, delta_top_k=0.1)
    _write_summary(tmp_path, "run_b", delta_priority_1=0.1, delta_top_k=0.3)

    subprocess.run(
        [sys.executable, "scripts/aggregate_rank_only_evals.py", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "leaderboard.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["run_name"] == "run_a"
    assert rows[0]["leaderboard_rank"] == 1
