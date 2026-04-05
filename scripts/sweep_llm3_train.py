"""Run a medium-scale LLM3 SFT/GRPO hyperparameter sweep on seed-split datasets.

This runner is intentionally lightweight:
- it reuses scripts/training_sft.sh and scripts/training_grpo.sh
- it keeps train/val/test split at the seed level
- it defaults to a modest medium-scale sweep recipe

Recommended usage:
1. Run SFT sweep first and inspect the resulting trial manifests.
2. Pick one or more best SFT trials.
3. Run GRPO sweep against those selected SFT trials.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "train" / "llm3_medium_5min_v1"
DEFAULT_SWEEP_ROOT = REPO_ROOT / "data" / "sweeps" / "llm3_medium_sweep_v1"
SOURCE_PRESETS = {
    "pipeline": ["pipeline"],
    "clean": ["clean"],
    "clean_pipeline": ["clean", "pipeline"],
}


@dataclass(frozen=True)
class SFTTrial:
    trial_name: str
    source_preset: str
    sources: List[str]
    lr: str
    global_batch_size: int


@dataclass(frozen=True)
class GRPOTrial:
    trial_name: str
    actor_lr: str
    rollout_n: int
    ppo_mini_batch_size: int
    kl_loss_coef: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_value(value: object) -> str:
    text = str(value)
    return (
        text.replace("-", "m")
        .replace(".", "p")
        .replace("/", "_")
        .replace(" ", "_")
    )


def parse_seed_specs(specs: Sequence[str]) -> List[int]:
    seeds = set()
    for raw_spec in specs:
        for part in str(raw_spec).split(","):
            spec = part.strip()
            if not spec:
                continue
            if re.fullmatch(r"\d+-\d+", spec):
                start, end = (int(piece) for piece in spec.split("-", 1))
                if end < start:
                    raise ValueError(f"Invalid seed range: {spec}")
                seeds.update(range(start, end + 1))
            elif re.fullmatch(r"\d+", spec):
                seeds.add(int(spec))
            else:
                raise ValueError(f"Unsupported seed spec: {spec}")
    return sorted(seeds)


def resolve_seed_dirs(dataset_root: Path, seeds: Sequence[int]) -> List[Path]:
    dirs: List[Path] = []
    missing: List[str] = []
    for seed in seeds:
        seed_dir = dataset_root / f"seed_{seed}"
        if seed_dir.is_dir():
            dirs.append(seed_dir)
        else:
            missing.append(str(seed_dir))
    if missing:
        raise FileNotFoundError(f"Missing seed directories: {missing}")
    return dirs


def build_sft_trials(
    source_presets: Sequence[str],
    lrs: Sequence[str],
    global_batch_sizes: Sequence[int],
) -> List[SFTTrial]:
    trials: List[SFTTrial] = []
    for source_preset, lr, global_batch_size in itertools.product(source_presets, lrs, global_batch_sizes):
        if source_preset not in SOURCE_PRESETS:
            raise ValueError(f"Unknown source preset: {source_preset}")
        trial_name = f"sft_{source_preset}_lr{_safe_value(lr)}_bs{global_batch_size}"
        trials.append(
            SFTTrial(
                trial_name=trial_name,
                source_preset=source_preset,
                sources=list(SOURCE_PRESETS[source_preset]),
                lr=str(lr),
                global_batch_size=int(global_batch_size),
            )
        )
    return trials


def build_grpo_trials(
    actor_lrs: Sequence[str],
    rollout_ns: Sequence[int],
    ppo_mini_batches: Sequence[int],
    kl_loss_coefs: Sequence[str],
) -> List[GRPOTrial]:
    trials: List[GRPOTrial] = []
    for actor_lr, rollout_n, ppo_mini_batch_size, kl_loss_coef in itertools.product(
        actor_lrs,
        rollout_ns,
        ppo_mini_batches,
        kl_loss_coefs,
    ):
        trial_name = (
            f"grpo_lr{_safe_value(actor_lr)}"
            f"_roll{int(rollout_n)}"
            f"_mini{int(ppo_mini_batch_size)}"
            f"_kl{_safe_value(kl_loss_coef)}"
        )
        trials.append(
            GRPOTrial(
                trial_name=trial_name,
                actor_lr=str(actor_lr),
                rollout_n=int(rollout_n),
                ppo_mini_batch_size=int(ppo_mini_batch_size),
                kl_loss_coef=str(kl_loss_coef),
            )
        )
    return trials


def latest_global_step_dir(root: Path) -> Path:
    candidates = sorted(
        (path for path in root.glob("global_step_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No global_step_* directories found under {root}")
    return candidates[-1]


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(shlex.quote(part) for part in command)
    if dry_run:
        log_path.write_text(f"[dry-run] {rendered}\n", encoding="utf-8")
        return 0

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[command] {rendered}\n")
        handle.flush()
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        return int(process.returncode)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def env_with_overrides(base: Dict[str, str], overrides: Dict[str, object]) -> Dict[str, str]:
    env = dict(base)
    for key, value in overrides.items():
        env[key] = str(value)
    return env


def python_command(conda_env: str) -> List[str]:
    if conda_env:
        return ["conda", "run", "--no-capture-output", "-n", conda_env, "env", "PYTHONNOUSERSITE=1", "python"]
    return [sys.executable]


def export_sft_parquet(
    *,
    train_dirs: Sequence[Path],
    val_dirs: Sequence[Path],
    sources: Sequence[str],
    train_out: Path,
    val_out: Path,
    seed: int,
    conda_env: str,
    dry_run: bool,
) -> None:
    train_unused_val = train_out.with_suffix(".unused_val.parquet")
    val_unused_val = val_out.with_suffix(".unused_val.parquet")

    py = python_command(conda_env)
    train_cmd = [*py, "scripts/export_llm3_to_verl_sft.py"]
    for input_dir in train_dirs:
        train_cmd += ["--input-dir", str(input_dir)]
    train_cmd += ["--sources", *sources, "--train-out", str(train_out), "--val-out", str(train_unused_val), "--val-ratio", "0", "--seed", str(seed)]

    val_cmd = [*py, "scripts/export_llm3_to_verl_sft.py"]
    for input_dir in val_dirs:
        val_cmd += ["--input-dir", str(input_dir)]
    val_cmd += ["--sources", *sources, "--train-out", str(val_out), "--val-out", str(val_unused_val), "--val-ratio", "0", "--seed", str(seed)]

    export_env = env_with_overrides(os.environ, {"PYTHONPATH": "src"})
    train_rc = run_command(
        train_cmd,
        cwd=REPO_ROOT,
        env=export_env,
        log_path=train_out.parent / "export_train.log",
        dry_run=dry_run,
    )
    if train_rc != 0:
        raise RuntimeError(f"SFT train export failed with exit code {train_rc}")
    val_rc = run_command(
        val_cmd,
        cwd=REPO_ROOT,
        env=export_env,
        log_path=val_out.parent / "export_val.log",
        dry_run=dry_run,
    )
    if val_rc != 0:
        raise RuntimeError(f"SFT val export failed with exit code {val_rc}")
    if not dry_run:
        train_unused_val.unlink(missing_ok=True)
        val_unused_val.unlink(missing_ok=True)


def export_grpo_parquet(
    *,
    train_dirs: Sequence[Path],
    val_dirs: Sequence[Path],
    train_out: Path,
    val_out: Path,
    seed: int,
    conda_env: str,
    dry_run: bool,
) -> None:
    train_unused_val = train_out.with_suffix(".unused_val.parquet")
    val_unused_val = val_out.with_suffix(".unused_val.parquet")

    py = python_command(conda_env)
    train_cmd = [*py, "scripts/export_llm3_to_verl_grpo.py"]
    for input_dir in train_dirs:
        train_cmd += ["--input-dir", str(input_dir)]
    train_cmd += ["--train-out", str(train_out), "--val-out", str(train_unused_val), "--val-ratio", "0", "--seed", str(seed)]

    val_cmd = [*py, "scripts/export_llm3_to_verl_grpo.py"]
    for input_dir in val_dirs:
        val_cmd += ["--input-dir", str(input_dir)]
    val_cmd += ["--train-out", str(val_out), "--val-out", str(val_unused_val), "--val-ratio", "0", "--seed", str(seed)]

    export_env = env_with_overrides(os.environ, {"PYTHONPATH": "src"})
    train_rc = run_command(
        train_cmd,
        cwd=REPO_ROOT,
        env=export_env,
        log_path=train_out.parent / "export_train.log",
        dry_run=dry_run,
    )
    if train_rc != 0:
        raise RuntimeError(f"GRPO train export failed with exit code {train_rc}")
    val_rc = run_command(
        val_cmd,
        cwd=REPO_ROOT,
        env=export_env,
        log_path=val_out.parent / "export_val.log",
        dry_run=dry_run,
    )
    if val_rc != 0:
        raise RuntimeError(f"GRPO val export failed with exit code {val_rc}")
    if not dry_run:
        train_unused_val.unlink(missing_ok=True)
        val_unused_val.unlink(missing_ok=True)


def run_sft_trial(
    *,
    trial: SFTTrial,
    train_dirs: Sequence[Path],
    val_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    output_root: Path,
    conda_env: str,
    model_path: str,
    export_seed: int,
    skip_existing: bool,
    dry_run: bool,
) -> Dict:
    trial_dir = output_root / "sft" / trial.trial_name
    data_dir = trial_dir / "data"
    ckpt_dir = trial_dir / "checkpoints"
    hydra_dir = trial_dir / "hydra"
    manifest_path = trial_dir / "trial_manifest.json"
    train_parquet = data_dir / "train.parquet"
    val_parquet = data_dir / "val.parquet"

    if skip_existing and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            return payload

    export_sft_parquet(
        train_dirs=train_dirs,
        val_dirs=val_dirs,
        sources=trial.sources,
        train_out=train_parquet,
        val_out=val_parquet,
        seed=export_seed,
        conda_env=conda_env,
        dry_run=dry_run,
    )

    env = env_with_overrides(
        os.environ,
        {
            "CONDA_ENV": conda_env,
            "AUTO_EXPORT_SFT": 0,
            "SFT_TRAIN_FILE": train_parquet,
            "SFT_VAL_FILE": val_parquet,
            "MODEL_PATH": model_path,
            "CKPT_DIR": ckpt_dir,
            "HYDRA_ROOT": hydra_dir,
            "TRAINER_EXPERIMENT_NAME": trial.trial_name,
            "SFT_GLOBAL_BATCH_SIZE": trial.global_batch_size,
            "SFT_LR": trial.lr,
            "SFT_EXPORT_SOURCES_STR": " ".join(trial.sources),
            "FAIL_ON_EXISTING_CKPT_DIR": 0 if skip_existing else 1,
        },
    )
    rc = run_command(
        ["bash", "scripts/training_sft.sh"],
        cwd=REPO_ROOT,
        env=env,
        log_path=trial_dir / "logs" / "training.log",
        dry_run=dry_run,
    )
    status = "planned" if dry_run else ("completed" if rc == 0 else "failed")
    latest_ckpt = str(ckpt_dir / "global_step_<pending>") if dry_run else None
    if status == "completed":
        latest_ckpt = str(latest_global_step_dir(ckpt_dir))

    payload = {
        "stage": "sft",
        "status": status,
        "trial_name": trial.trial_name,
        "created_at": _utc_now(),
        "params": asdict(trial),
        "paths": {
            "trial_dir": str(trial_dir),
            "train_parquet": str(train_parquet),
            "val_parquet": str(val_parquet),
            "checkpoint_dir": str(ckpt_dir),
            "latest_checkpoint": latest_ckpt,
        },
        "splits": {
            "train_seeds": [int(path.name.split("_", 1)[1]) for path in train_dirs],
            "val_seeds": [int(path.name.split("_", 1)[1]) for path in val_dirs],
            "test_seeds": [int(path.name.split("_", 1)[1]) for path in test_dirs],
        },
        "return_code": rc,
    }
    write_json(manifest_path, payload)
    return payload


def run_grpo_trial(
    *,
    base_sft_manifest: Dict,
    trial: GRPOTrial,
    train_dirs: Sequence[Path],
    val_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    output_root: Path,
    conda_env: str,
    export_seed: int,
    skip_existing: bool,
    dry_run: bool,
) -> Dict:
    base_name = str(base_sft_manifest["trial_name"])
    base_ckpt = Path(str(base_sft_manifest["paths"]["latest_checkpoint"]))
    trial_dir = output_root / "grpo" / base_name / trial.trial_name
    data_dir = trial_dir / "data"
    ckpt_dir = trial_dir / "checkpoints"
    hydra_dir = trial_dir / "hydra"
    merged_model_dir = trial_dir / "merged_hf"
    manifest_path = trial_dir / "trial_manifest.json"
    train_parquet = data_dir / "train.parquet"
    val_parquet = data_dir / "val.parquet"

    if skip_existing and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            return payload

    export_grpo_parquet(
        train_dirs=train_dirs,
        val_dirs=val_dirs,
        train_out=train_parquet,
        val_out=val_parquet,
        seed=export_seed,
        conda_env=conda_env,
        dry_run=dry_run,
    )

    env = env_with_overrides(
        os.environ,
        {
            "CONDA_ENV": conda_env,
            "AUTO_EXPORT_GRPO": 0,
            "GRPO_TRAIN_FILE": train_parquet,
            "GRPO_VAL_FILE": val_parquet,
            "MODEL_PATH": merged_model_dir / base_ckpt.name,
            "SFT_CKPT_DIR": base_ckpt,
            "SFT_GLOBAL_STEP": base_ckpt.name.rsplit("_", 1)[-1],
            "CKPT_DIR": ckpt_dir,
            "HYDRA_ROOT": hydra_dir,
            "TRAINER_EXPERIMENT_NAME": f"{base_name}__{trial.trial_name}",
            "GRPO_TRAIN_BATCH_SIZE": max(trial.ppo_mini_batch_size * 2, 8),
            "GRPO_PPO_MINI_BATCH_SIZE": trial.ppo_mini_batch_size,
            "GRPO_ROLLOUT_N": trial.rollout_n,
            "GRPO_ACTOR_LR": trial.actor_lr,
            "GRPO_KL_LOSS_COEF": trial.kl_loss_coef,
            "FAIL_ON_EXISTING_CKPT_DIR": 0 if skip_existing else 1,
        },
    )
    rc = run_command(
        ["bash", "scripts/training_grpo.sh"],
        cwd=REPO_ROOT,
        env=env,
        log_path=trial_dir / "logs" / "training.log",
        dry_run=dry_run,
    )
    status = "planned" if dry_run else ("completed" if rc == 0 else "failed")
    latest_ckpt = str(ckpt_dir / "global_step_<pending>") if dry_run else None
    if status == "completed":
        latest_ckpt = str(latest_global_step_dir(ckpt_dir))

    payload = {
        "stage": "grpo",
        "status": status,
        "trial_name": trial.trial_name,
        "base_sft_trial": base_name,
        "created_at": _utc_now(),
        "params": asdict(trial),
        "paths": {
            "trial_dir": str(trial_dir),
            "train_parquet": str(train_parquet),
            "val_parquet": str(val_parquet),
            "checkpoint_dir": str(ckpt_dir),
            "latest_checkpoint": latest_ckpt,
            "model_path": str(merged_model_dir / base_ckpt.name),
            "base_sft_checkpoint": str(base_ckpt),
        },
        "splits": {
            "train_seeds": [int(path.name.split("_", 1)[1]) for path in train_dirs],
            "val_seeds": [int(path.name.split("_", 1)[1]) for path in val_dirs],
            "test_seeds": [int(path.name.split("_", 1)[1]) for path in test_dirs],
        },
        "return_code": rc,
    }
    write_json(manifest_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a medium-scale LLM3 SFT/GRPO sweep.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--stage", choices=("sft", "grpo", "both"), default="sft")
    parser.add_argument("--conda-env", default="verl")
    parser.add_argument("--model-path", default="", help="HF base model path for SFT.")
    parser.add_argument("--train-seeds", nargs="+", default=["4101-4108"])
    parser.add_argument("--val-seeds", nargs="+", default=["4109-4110"])
    parser.add_argument("--test-seeds", nargs="+", default=["4111-4112"])
    parser.add_argument("--sft-source-presets", nargs="+", default=["pipeline", "clean_pipeline"])
    parser.add_argument("--sft-lrs", nargs="+", default=["5e-5", "1e-4", "2e-4"])
    parser.add_argument("--sft-global-batches", nargs="+", type=int, default=[128])
    parser.add_argument("--grpo-actor-lrs", nargs="+", default=["5e-7", "1e-6"])
    parser.add_argument("--grpo-rollout-ns", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--grpo-mini-batches", nargs="+", type=int, default=[8])
    parser.add_argument("--grpo-kl-coefs", nargs="+", default=["1e-3"])
    parser.add_argument(
        "--grpo-base-sft-trial",
        action="append",
        default=[],
        help="Existing SFT trial name to use as the GRPO base. Repeat this flag for multiple trials.",
    )
    parser.add_argument("--export-seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.stage in {"sft", "both"} and not args.model_path:
        raise ValueError("--model-path is required for --stage sft and --stage both.")

    train_dirs = resolve_seed_dirs(dataset_root, parse_seed_specs(args.train_seeds))
    val_dirs = resolve_seed_dirs(dataset_root, parse_seed_specs(args.val_seeds))
    test_dirs = resolve_seed_dirs(dataset_root, parse_seed_specs(args.test_seeds))

    sweep_plan = {
        "created_at": _utc_now(),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "stage": args.stage,
        "conda_env": args.conda_env,
        "model_path": args.model_path,
        "train_seeds": [int(path.name.split("_", 1)[1]) for path in train_dirs],
        "val_seeds": [int(path.name.split("_", 1)[1]) for path in val_dirs],
        "test_seeds": [int(path.name.split("_", 1)[1]) for path in test_dirs],
        "dry_run": args.dry_run,
    }
    write_json(output_root / "sweep_plan.json", sweep_plan)

    leaderboard_path = output_root / "leaderboard.jsonl"

    sft_manifests: List[Dict] = []
    if args.stage in {"sft", "both"}:
        sft_trials = build_sft_trials(
            source_presets=args.sft_source_presets,
            lrs=args.sft_lrs,
            global_batch_sizes=args.sft_global_batches,
        )
        for trial in sft_trials:
            manifest = run_sft_trial(
                trial=trial,
                train_dirs=train_dirs,
                val_dirs=val_dirs,
                test_dirs=test_dirs,
                output_root=output_root,
                conda_env=args.conda_env,
                model_path=args.model_path,
                export_seed=args.export_seed,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
            )
            sft_manifests.append(manifest)
            append_jsonl(leaderboard_path, manifest)

    if args.stage in {"grpo", "both"}:
        base_trials = list(args.grpo_base_sft_trial)
        if args.stage == "both" and not base_trials:
            if len(sft_manifests) == 1:
                base_trials = [str(sft_manifests[0]["trial_name"])]
            else:
                raise ValueError(
                    "--stage both requires exactly one SFT trial or explicit --grpo-base-sft-trial values.",
                )
        if args.stage == "grpo" and not base_trials:
            raise ValueError("--stage grpo requires at least one --grpo-base-sft-trial value.")

        grpo_trials = build_grpo_trials(
            actor_lrs=args.grpo_actor_lrs,
            rollout_ns=args.grpo_rollout_ns,
            ppo_mini_batches=args.grpo_mini_batches,
            kl_loss_coefs=args.grpo_kl_coefs,
        )
        for base_name in base_trials:
            base_manifest_path = output_root / "sft" / base_name / "trial_manifest.json"
            if not base_manifest_path.is_file():
                raise FileNotFoundError(f"Missing base SFT manifest: {base_manifest_path}")
            base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
            for trial in grpo_trials:
                manifest = run_grpo_trial(
                    base_sft_manifest=base_manifest,
                    trial=trial,
                    train_dirs=train_dirs,
                    val_dirs=val_dirs,
                    test_dirs=test_dirs,
                    output_root=output_root,
                    conda_env=args.conda_env,
                    export_seed=args.export_seed,
                    skip_existing=args.skip_existing,
                    dry_run=args.dry_run,
                )
                append_jsonl(leaderboard_path, manifest)


if __name__ == "__main__":
    main()
