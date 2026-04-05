from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sweep_llm3_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sweep_llm3_train", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_aggregate_alignment_results_and_refresh_leaderboard(tmp_path: Path) -> None:
    module = _load_module()

    result_a = {
        "per_item": [
            {"time_window": "w1", "event_id": "e1", "true_priority": 1, "pred_priority": 1, "window_rank": 1},
            {"time_window": "w1", "event_id": "e2", "true_priority": 2, "pred_priority": 2, "window_rank": 2},
        ]
    }
    result_b = {
        "per_item": [
            {"time_window": "w2", "event_id": "e3", "true_priority": 1, "pred_priority": 2, "window_rank": 2},
            {"time_window": "w2", "event_id": "e4", "true_priority": 3, "pred_priority": 3, "window_rank": 3},
        ]
    }
    aggregated = module._aggregate_alignment_results([result_a, result_b], urgent_threshold=2)
    assert aggregated["n_aligned_demands"] == 4
    assert aggregated["accuracy"] == 0.75
    assert aggregated["priority_1_metrics"]["recall"] == 0.5

    output_root = tmp_path / "sweep"
    trial_a = output_root / "sft" / "trial_a"
    trial_b = output_root / "sft" / "trial_b"
    trial_a.mkdir(parents=True, exist_ok=True)
    trial_b.mkdir(parents=True, exist_ok=True)

    def _write_manifest(path: Path, trial_name: str, priority_1_recall: float, top_k_hit_rate: float) -> None:
        payload = {
            "stage": "sft",
            "status": "completed",
            "trial_name": trial_name,
            "params": {
                "source_preset": "pipeline",
                "sources": ["pipeline"],
                "lr": "1e-4",
                "global_batch_size": 128,
            },
            "paths": {
                "latest_checkpoint": str(path / "checkpoints" / "global_step_54"),
                "trial_dir": str(path),
                },
                "evaluation": {
                    "enabled": True,
                    "priority_mode": "llm-only",
                    "baseline_mode": "rule-only",
                    "baseline": {
                    "metrics": {
                        "priority_1_recall": 0.4,
                        "top_k_hit_rate": 0.5,
                        "urgent_f1": 0.6,
                        "macro_f1": 0.6,
                        "accuracy": 0.6,
                    }
                },
                "post": {
                    "metrics": {
                        "n_aligned_demands": 20,
                        "priority_1_recall": priority_1_recall,
                        "top_k_hit_rate": top_k_hit_rate,
                        "urgent_f1": 0.7,
                        "macro_f1": 0.65,
                        "accuracy": 0.7,
                        "weighted_f1": 0.7,
                        "spearman": 0.8,
                        "kendall_tau": 0.7,
                        "priority_1_f1": 0.6,
                        "urgent_recall": 0.8,
                    }
                },
                "delta_vs_baseline": {
                    "priority_1_recall": round(priority_1_recall - 0.4, 6),
                    "top_k_hit_rate": round(top_k_hit_rate - 0.5, 6),
                    "urgent_f1": 0.1,
                    "macro_f1": 0.05,
                    "accuracy": 0.1,
                    "weighted_f1": 0.1,
                    "spearman": 0.1,
                    "kendall_tau": 0.1,
                    "priority_1_f1": 0.1,
                    "urgent_recall": 0.1,
                },
            },
        }
        (path / "trial_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_manifest(trial_a, "trial_a", priority_1_recall=0.9, top_k_hit_rate=0.8)
    _write_manifest(trial_b, "trial_b", priority_1_recall=0.6, top_k_hit_rate=0.9)

    module._refresh_sft_leaderboard(output_root)
    rows = [
        json.loads(line)
        for line in (output_root / "leaderboard.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["trial_name"] == "trial_a"
    assert rows[0]["leaderboard_rank"] == 1


def test_refresh_grpo_leaderboard(tmp_path: Path) -> None:
    module = _load_module()

    output_root = tmp_path / "sweep"
    trial_a = output_root / "grpo" / "base_a" / "trial_a"
    trial_b = output_root / "grpo" / "base_a" / "trial_b"
    trial_a.mkdir(parents=True, exist_ok=True)
    trial_b.mkdir(parents=True, exist_ok=True)

    def _write_manifest(path: Path, trial_name: str, priority_1_recall: float, top_k_hit_rate: float) -> None:
        payload = {
            "stage": "grpo",
            "status": "completed",
            "trial_name": trial_name,
            "base_sft_trial": "base_a",
            "params": {
                "actor_lr": "1e-6",
                "rollout_n": 2,
                "ppo_mini_batch_size": 8,
                "kl_loss_coef": "1e-3",
            },
            "paths": {
                "latest_checkpoint": str(path / "checkpoints" / "global_step_20"),
                "trial_dir": str(path),
            },
            "splits": {"val_seeds": [4109, 4110]},
            "evaluation": {
                "enabled": True,
                "priority_mode": "llm-only",
                "baseline_mode": "base-sft",
                "baseline": {
                    "metrics": {
                        "priority_1_recall": 0.5,
                        "top_k_hit_rate": 0.5,
                        "urgent_f1": 0.6,
                        "macro_f1": 0.6,
                        "accuracy": 0.6,
                    }
                },
                "post": {
                    "metrics": {
                        "n_aligned_demands": 20,
                        "priority_1_recall": priority_1_recall,
                        "top_k_hit_rate": top_k_hit_rate,
                        "urgent_f1": 0.7,
                        "macro_f1": 0.65,
                        "accuracy": 0.7,
                        "weighted_f1": 0.7,
                        "spearman": 0.8,
                        "kendall_tau": 0.7,
                        "priority_1_f1": 0.6,
                        "urgent_recall": 0.8,
                    }
                },
                "delta_vs_baseline": {
                    "priority_1_recall": round(priority_1_recall - 0.5, 6),
                    "top_k_hit_rate": round(top_k_hit_rate - 0.5, 6),
                    "urgent_f1": 0.1,
                    "macro_f1": 0.05,
                    "accuracy": 0.1,
                    "weighted_f1": 0.1,
                    "spearman": 0.1,
                    "kendall_tau": 0.1,
                    "priority_1_f1": 0.1,
                    "urgent_recall": 0.1,
                },
            },
        }
        (path / "trial_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_manifest(trial_a, "trial_a", priority_1_recall=0.9, top_k_hit_rate=0.8)
    _write_manifest(trial_b, "trial_b", priority_1_recall=0.6, top_k_hit_rate=0.9)

    module._refresh_grpo_leaderboard(output_root)
    rows = [
        json.loads(line)
        for line in (output_root / "grpo_leaderboard.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["trial_name"] == "trial_a"
    assert rows[0]["leaderboard_rank"] == 1
