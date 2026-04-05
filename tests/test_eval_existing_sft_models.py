from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_seed_dirs(dataset_root: Path, seeds: list[int]) -> None:
    for seed in seeds:
        (dataset_root / f"seed_{seed}").mkdir(parents=True, exist_ok=True)


def test_eval_existing_sft_models_dry_run_checkpoint_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "eval"
    checkpoint_root = tmp_path / "checkpoints" / "legacy_trial"
    (checkpoint_root / "global_step_8").mkdir(parents=True, exist_ok=True)
    _make_seed_dirs(dataset_root, [4109, 4110, 4111, 4112])

    cmd = [
        sys.executable,
        "scripts/eval_existing_sft_models.py",
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(output_root),
        "--model",
        f"legacy={checkpoint_root}",
        "--val-seeds",
        "4109-4110",
        "--test-seeds",
        "4111-4112",
        "--dry-run",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    manifest = json.loads(
        (
            output_root
            / "sft"
            / "sft_imported_legacy_legacy_trial"
            / "trial_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "planned"
    assert manifest["params"]["source_type"] == "checkpoint_root"
    assert manifest["evaluation"]["enabled"] is True

    leaderboard_path = output_root / "leaderboard.jsonl"
    assert leaderboard_path.exists()
    rows = [line for line in leaderboard_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows == []
