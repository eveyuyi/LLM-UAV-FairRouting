from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_seed_dirs(dataset_root: Path, seeds: list[int]) -> None:
    for seed in seeds:
        (dataset_root / f"seed_{seed}").mkdir(parents=True, exist_ok=True)


def test_sweep_llm3_train_dry_run_sft(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "sweep"
    _make_seed_dirs(dataset_root, [4101, 4102, 4103, 4104])

    cmd = [
        sys.executable,
        "scripts/sweep_llm3_train.py",
        "--stage",
        "sft",
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(output_root),
        "--model-path",
        "/tmp/fake-model",
        "--train-seeds",
        "4101-4102",
        "--val-seeds",
        "4103",
        "--test-seeds",
        "4104",
        "--sft-source-presets",
        "pipeline",
        "--sft-lrs",
        "1e-4",
        "--sft-global-batches",
        "64",
        "--dry-run",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    manifest = json.loads(
        (output_root / "sft" / "sft_pipeline_lr1em4_bs64" / "trial_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "planned"
    assert manifest["params"]["source_preset"] == "pipeline"
    assert manifest["splits"]["train_seeds"] == [4101, 4102]


def test_sweep_llm3_train_dry_run_grpo(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "sweep"
    _make_seed_dirs(dataset_root, [4101, 4102, 4103, 4104])

    base_trial_dir = output_root / "sft" / "sft_pipeline_lr1em4_bs64"
    base_trial_dir.mkdir(parents=True, exist_ok=True)
    (base_trial_dir / "trial_manifest.json").write_text(
        json.dumps(
            {
                "trial_name": "sft_pipeline_lr1em4_bs64",
                "paths": {
                    "latest_checkpoint": str(base_trial_dir / "checkpoints" / "global_step_54"),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "scripts/sweep_llm3_train.py",
        "--stage",
        "grpo",
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(output_root),
        "--train-seeds",
        "4101-4102",
        "--val-seeds",
        "4103",
        "--test-seeds",
        "4104",
        "--grpo-base-sft-trial",
        "sft_pipeline_lr1em4_bs64",
        "--grpo-actor-lrs",
        "1e-6",
        "--grpo-rollout-ns",
        "2",
        "--grpo-mini-batches",
        "8",
        "--grpo-kl-coefs",
        "1e-3",
        "--dry-run",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    manifest = json.loads(
        (
            output_root
            / "grpo"
            / "sft_pipeline_lr1em4_bs64"
            / "grpo_lr1em6_roll2_mini8_kl1em3"
            / "trial_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "planned"
    assert manifest["base_sft_trial"] == "sft_pipeline_lr1em4_bs64"
    assert manifest["params"]["rollout_n"] == 2
