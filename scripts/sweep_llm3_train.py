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
import csv
import itertools
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

try:
    from scipy.stats import kendalltau, spearmanr
except Exception:  # pragma: no cover - optional fallback
    kendalltau = None
    spearmanr = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "train" / "llm3_medium_5min_v1"
DEFAULT_SWEEP_ROOT = REPO_ROOT / "data" / "sweeps" / "llm3_medium_sweep_v1"
DEFAULT_STATIONS_PATH = REPO_ROOT / "data" / "seed" / "drone_station_locations.csv"
DEFAULT_BUILDING_DATA_PATH = REPO_ROOT / "data" / "seed" / "building_information.csv"
SOURCE_PRESETS = {
    "pipeline": ["pipeline"],
    "clean": ["clean"],
    "clean_pipeline": ["clean", "pipeline"],
}
LEADERBOARD_SORT_METRICS = (
    "priority_1_recall",
    "top_k_hit_rate",
    "urgent_f1",
    "macro_f1",
    "accuracy",
)


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


@dataclass(frozen=True)
class EvalSettings:
    enabled: bool
    priority_mode: str
    baseline_mode: str
    api_key: str
    served_model_name_prefix: str
    host: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    dtype: str
    trust_remote_code: bool
    urgent_threshold: int
    port_base: int
    startup_timeout_s: int
    startup_poll_interval_s: float
    stations_path: str
    building_data_path: str


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


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _safe_rank_metric(metric_fn, y_true: List[int], y_pred: List[int]) -> Optional[float]:
    if metric_fn is None or len(y_true) < 2:
        return None
    if len(set(y_true)) < 2 and len(set(y_pred)) < 2:
        return 1.0
    value = metric_fn(y_true, y_pred)
    if isinstance(value, tuple):
        value = value[0]
    return _safe_float(value)


def _binary_metrics(y_true: List[int], y_pred: List[int], positive_fn) -> Dict[str, Optional[float]]:
    true_bin = [1 if positive_fn(value) else 0 for value in y_true]
    pred_bin = [1 if positive_fn(value) else 0 for value in y_pred]
    support = sum(true_bin)
    predicted_positive = sum(pred_bin)
    if support == 0:
        return {
            "support": 0,
            "predicted_positive": predicted_positive,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    precision, recall, f1_values, _ = precision_recall_fscore_support(
        true_bin,
        pred_bin,
        labels=[1],
        average=None,
        zero_division=0,
    )
    return {
        "support": int(support),
        "predicted_positive": int(predicted_positive),
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1_values[0]),
    }


def _top_k_hit_rate(items: List[Dict[str, object]], urgent_threshold: int) -> Dict[str, Optional[float]]:
    true_k = sum(1 for item in items if int(item["true_priority"]) <= urgent_threshold)
    if true_k == 0:
        return {"k": 0, "hit_rate": None, "hits": 0}

    true_sorted = sorted(
        items,
        key=lambda item: (
            int(item["true_priority"]),
            str(item.get("time_window", "")),
            str(item.get("event_id", "")),
        ),
    )
    pred_sorted = sorted(
        items,
        key=lambda item: (
            int(item["pred_priority"]),
            int(item.get("window_rank") or 10**9),
            str(item.get("time_window", "")),
            str(item.get("event_id", "")),
        ),
    )
    true_top = {str(item["event_id"]) for item in true_sorted[:true_k]}
    pred_top = {str(item["event_id"]) for item in pred_sorted[:true_k]}
    hits = len(true_top & pred_top)
    return {"k": true_k, "hit_rate": hits / true_k, "hits": hits}


def _aggregate_alignment_results(results: Sequence[Dict[str, object]], urgent_threshold: int) -> Dict[str, object]:
    per_item: List[Dict[str, object]] = []
    for result in results:
        per_item.extend(result.get("per_item", []) or [])

    y_true = [int(item["true_priority"]) for item in per_item]
    y_pred = [int(item["pred_priority"]) for item in per_item]
    labels = sorted(set(y_true) | set(y_pred))
    aggregated: Dict[str, object] = {
        "n_aligned_demands": len(per_item),
        "labels": labels,
        "urgent_threshold": urgent_threshold,
        "per_item": per_item,
    }
    if not per_item:
        aggregated.update(
            {
                "accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "confusion_matrix": [],
                "per_priority_metrics": {},
                "spearman": None,
                "kendall_tau": None,
                "priority_1_metrics": {},
                "urgent_metrics": {},
                "top_k_hit_rate": {"k": 0, "hit_rate": None, "hits": 0},
            }
        )
        return aggregated

    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    aggregated.update(
        {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted")),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "per_priority_metrics": {
                str(label): {
                    "precision": float(precision[idx]),
                    "recall": float(recall[idx]),
                    "f1": float(f1_values[idx]),
                    "support": int(support[idx]),
                }
                for idx, label in enumerate(labels)
            },
            "spearman": _safe_rank_metric(spearmanr, y_true, y_pred),
            "kendall_tau": _safe_rank_metric(kendalltau, y_true, y_pred),
            "priority_1_metrics": _binary_metrics(y_true, y_pred, lambda value: value == 1),
            "urgent_metrics": _binary_metrics(y_true, y_pred, lambda value: value <= urgent_threshold),
            "top_k_hit_rate": _top_k_hit_rate(per_item, urgent_threshold),
        }
    )
    return aggregated


def _alignment_snapshot(result: Dict[str, object]) -> Dict[str, Optional[float]]:
    return {
        "accuracy": _safe_float(result.get("accuracy")),
        "macro_f1": _safe_float(result.get("macro_f1")),
        "weighted_f1": _safe_float(result.get("weighted_f1")),
        "spearman": _safe_float(result.get("spearman")),
        "kendall_tau": _safe_float(result.get("kendall_tau")),
        "top_k_hit_rate": _safe_float((result.get("top_k_hit_rate") or {}).get("hit_rate")),
        "priority_1_recall": _safe_float((result.get("priority_1_metrics") or {}).get("recall")),
        "priority_1_f1": _safe_float((result.get("priority_1_metrics") or {}).get("f1")),
        "urgent_recall": _safe_float((result.get("urgent_metrics") or {}).get("recall")),
        "urgent_f1": _safe_float((result.get("urgent_metrics") or {}).get("f1")),
        "n_aligned_demands": _safe_float(result.get("n_aligned_demands")),
    }


def _delta_snapshot(post_metrics: Dict[str, Optional[float]], base_metrics: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    payload: Dict[str, Optional[float]] = {}
    for key in (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "spearman",
        "kendall_tau",
        "top_k_hit_rate",
        "priority_1_recall",
        "priority_1_f1",
        "urgent_recall",
        "urgent_f1",
    ):
        post_value = post_metrics.get(key)
        base_value = base_metrics.get(key)
        if post_value is None or base_value is None:
            payload[key] = None
        else:
            payload[key] = round(float(post_value) - float(base_value), 6)
    return payload


def _leaderboard_primary_score(metrics: Dict[str, Optional[float]]) -> float:
    score = 0.0
    score += 100.0 * float(metrics.get("priority_1_recall") or 0.0)
    score += 10.0 * float(metrics.get("top_k_hit_rate") or 0.0)
    score += 1.0 * float(metrics.get("urgent_f1") or 0.0)
    score += 0.1 * float(metrics.get("macro_f1") or 0.0)
    score += 0.01 * float(metrics.get("accuracy") or 0.0)
    return round(score, 6)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _wait_for_http_ready(url: str, timeout_s: int, interval_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= int(response.status) < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(interval_s)
    raise TimeoutError(f"Timed out waiting for server readiness: {url}")


def _build_trial_record_row(manifest: Dict[str, object]) -> Dict[str, object]:
    row = {
        "stage": manifest.get("stage"),
        "status": manifest.get("status"),
        "trial_name": manifest.get("trial_name"),
        "base_sft_trial": manifest.get("base_sft_trial"),
        "created_at": manifest.get("created_at"),
    }
    row.update(manifest.get("params", {}) or {})
    return row


def _refresh_sft_leaderboard(output_root: Path) -> None:
    manifests = sorted(output_root.glob("sft/*/trial_manifest.json"))
    rows: List[Dict[str, object]] = []
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        evaluation = payload.get("evaluation") or {}
        if payload.get("status") != "completed" or not evaluation.get("enabled"):
            continue
        post_metrics = (evaluation.get("post") or {}).get("metrics") or {}
        baseline_metrics = (evaluation.get("baseline") or {}).get("metrics") or {}
        delta_metrics = (evaluation.get("delta_vs_baseline") or {})
        row = {
            "trial_name": payload.get("trial_name"),
            "source_preset": (payload.get("params") or {}).get("source_preset"),
            "sources": ",".join((payload.get("params") or {}).get("sources") or []),
            "lr": (payload.get("params") or {}).get("lr"),
            "global_batch_size": (payload.get("params") or {}).get("global_batch_size"),
            "priority_mode": evaluation.get("priority_mode"),
            "baseline_mode": evaluation.get("baseline_mode"),
            "n_val_seeds": len((payload.get("splits") or {}).get("val_seeds") or []),
            "n_aligned_demands": post_metrics.get("n_aligned_demands"),
            "primary_score": _leaderboard_primary_score(post_metrics),
            "checkpoint": ((payload.get("paths") or {}).get("latest_checkpoint")),
            "trial_dir": ((payload.get("paths") or {}).get("trial_dir")),
        }
        for key in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "spearman",
            "kendall_tau",
            "top_k_hit_rate",
            "priority_1_recall",
            "priority_1_f1",
            "urgent_recall",
            "urgent_f1",
        ):
            row[f"post_{key}"] = post_metrics.get(key)
            row[f"baseline_{key}"] = baseline_metrics.get(key)
            row[f"delta_{key}"] = delta_metrics.get(key)
        rows.append(row)

    rows.sort(
        key=lambda item: tuple(
            float(item.get(f"post_{metric}") or 0.0) for metric in LEADERBOARD_SORT_METRICS
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["leaderboard_rank"] = rank

    jsonl_path = output_root / "leaderboard.jsonl"
    csv_path = output_root / "leaderboard.csv"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


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


def ensure_hf_checkpoint(
    *,
    sft_checkpoint_dir: Path,
    merged_model_dir: Path,
    conda_env: str,
    skip_model_load_check: bool = False,
    dry_run: bool,
) -> Path:
    if dry_run:
        return merged_model_dir
    if (merged_model_dir / "config.json").is_file():
        return merged_model_dir
    merged_model_dir.parent.mkdir(parents=True, exist_ok=True)
    if conda_env:
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            conda_env,
            "env",
            "PYTHONNOUSERSITE=1",
            "bash",
            "scripts/export_sft_ckpt_to_hf.sh",
            str(sft_checkpoint_dir),
            str(merged_model_dir),
        ]
    else:
        command = ["bash", "scripts/export_sft_ckpt_to_hf.sh", str(sft_checkpoint_dir), str(merged_model_dir)]
    rc = run_command(
        command,
        cwd=REPO_ROOT,
        env=env_with_overrides(
            os.environ,
            {
                "MERGE_SKIP_MODEL_LOAD_CHECK": 1 if skip_model_load_check else 0,
            },
        ),
        log_path=merged_model_dir.parent / "merge_to_hf.log",
        dry_run=dry_run,
    )
    if rc != 0:
        raise RuntimeError(f"Failed to export SFT checkpoint to HF format: {sft_checkpoint_dir}")
    return merged_model_dir


def launch_vllm_server(
    *,
    model_path: Path,
    conda_env: str,
    settings: EvalSettings,
    served_model_name: str,
    port: int,
    log_path: Path,
    dry_run: bool,
):
    if dry_run:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = env_with_overrides(
        os.environ,
        {
            "CONDA_ENV": conda_env,
            "MODEL_PATH": model_path,
            "SERVED_MODEL_NAME": served_model_name,
            "HOST": settings.host,
            "PORT": port,
            "TENSOR_PARALLEL_SIZE": settings.tensor_parallel_size,
            "GPU_MEMORY_UTILIZATION": settings.gpu_memory_utilization,
            "MAX_MODEL_LEN": settings.max_model_len,
            "DTYPE": settings.dtype,
            "TRUST_REMOTE_CODE": 1 if settings.trust_remote_code else 0,
        },
    )
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", "scripts/serve_vllm_model.sh"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_http_ready(
            f"http://127.0.0.1:{port}/v1/models",
            timeout_s=settings.startup_timeout_s,
            interval_s=settings.startup_poll_interval_s,
        )
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        handle.close()
        raise
    process._codex_log_handle = handle  # type: ignore[attr-defined]
    return process


def stop_process(process) -> None:
    if process is None:
        return
    handle = getattr(process, "_codex_log_handle", None)
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)
    if handle is not None:
        handle.close()


def _latest_run_dir(root: Path) -> Path:
    candidates = sorted(root.glob("run_*"))
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_rank_only_workflow(
    *,
    output_dir: Path,
    extracted_demands_path: Path,
    dialogues_path: Path,
    stations_path: Path,
    building_data_path: Path,
    priority_mode: str,
    api_base: Optional[str],
    api_key: str,
    model_name: str,
    conda_env: str,
    dry_run: bool,
) -> Path:
    env = env_with_overrides(os.environ, {"PYTHONPATH": "src", "OPENAI_API_KEY": api_key})
    if api_base:
        env["OPENAI_BASE_URL"] = api_base
    command = [
        *python_command(conda_env),
        "-m",
        "llm4fairrouting.workflow.run_workflow",
        "--output-dir",
        str(output_dir),
        "--dialogues",
        str(dialogues_path),
        "--stations",
        str(stations_path),
        "--building-data",
        str(building_data_path),
        "--extracted-demands",
        str(extracted_demands_path),
        "--priority-mode",
        priority_mode,
        "--model",
        model_name,
        "--skip-solver",
    ]
    rc = run_command(
        command,
        cwd=REPO_ROOT,
        env=env,
        log_path=output_dir / "workflow.log",
        dry_run=dry_run,
    )
    if rc != 0:
        raise RuntimeError(f"Rank-only workflow failed: {output_dir}")
    if dry_run:
        planned_dir = output_dir / "run_<pending>"
        planned_dir.mkdir(parents=True, exist_ok=True)
        return planned_dir
    return _latest_run_dir(output_dir)


def evaluate_alignment(
    *,
    run_dir: Path,
    demands_path: Path,
    dialogues_path: Path,
    ground_truth_path: Path,
    truth_demands_path: Path,
    urgent_threshold: int,
    conda_env: str,
    output_path: Path,
    dry_run: bool,
) -> Dict[str, object]:
    if dry_run:
        payload = {
            "n_aligned_demands": None,
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "spearman": None,
            "kendall_tau": None,
            "priority_1_metrics": {},
            "urgent_metrics": {},
            "top_k_hit_rate": {"k": 0, "hit_rate": None, "hits": 0},
            "per_item": [],
            "truth_source": "fixed_demands",
        }
        write_json(output_path, payload)
        return payload
    py = python_command(conda_env)
    command = [
        *py,
        "evals/eval_priority_alignment.py",
        "--weights",
        str(run_dir / "weight_configs"),
        "--demands",
        str(demands_path),
        "--dialogues",
        str(dialogues_path),
        "--ground-truth",
        str(ground_truth_path),
        "--truth-source",
        "fixed_demands",
        "--truth-demands",
        str(truth_demands_path),
        "--urgent-threshold",
        str(urgent_threshold),
        "--output",
        str(output_path),
    ]
    rc = run_command(
        command,
        cwd=REPO_ROOT,
        env=env_with_overrides(os.environ, {"PYTHONPATH": "src"}),
        log_path=output_path.parent / f"{output_path.stem}.log",
        dry_run=dry_run,
    )
    if rc != 0:
        raise RuntimeError(f"Priority alignment evaluation failed: {output_path}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def evaluate_sft_trial(
    *,
    trial_dir: Path,
    latest_checkpoint: Path,
    val_dirs: Sequence[Path],
    conda_env: str,
    eval_settings: EvalSettings,
    dry_run: bool,
) -> Dict[str, object]:
    if not eval_settings.enabled:
        return {"enabled": False}

    merged_model_dir = ensure_hf_checkpoint(
        sft_checkpoint_dir=latest_checkpoint,
        merged_model_dir=trial_dir / "merged_hf" / latest_checkpoint.name,
        conda_env=conda_env,
        dry_run=dry_run,
    )
    return evaluate_served_model(
        trial_dir=trial_dir,
        merged_model_dir=merged_model_dir,
        val_dirs=val_dirs,
        conda_env=conda_env,
        eval_settings=eval_settings,
        dry_run=dry_run,
    )


def evaluate_served_model(
    *,
    trial_dir: Path,
    merged_model_dir: Path,
    val_dirs: Sequence[Path],
    conda_env: str,
    eval_settings: EvalSettings,
    dry_run: bool,
) -> Dict[str, object]:
    if not eval_settings.enabled:
        return {"enabled": False}

    evaluation_root = trial_dir / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    sweep_root = trial_dir.parents[1]
    baseline_cache_root = sweep_root / "_eval_cache" / "rule_only"

    baseline_by_seed: Dict[str, Dict[str, object]] = {}
    post_by_seed: Dict[str, Dict[str, object]] = {}
    baseline_results: List[Dict[str, object]] = []
    post_results: List[Dict[str, object]] = []

    for seed_dir in val_dirs:
        seed_name = seed_dir.name
        dialogues_path = seed_dir / "dialogues.jsonl"
        ground_truth_path = seed_dir / "events_manifest.jsonl"
        fixed_demands_path = seed_dir / "llm3_sft_pipeline.jsonl"
        baseline_root = baseline_cache_root / seed_name
        baseline_alignment_path = baseline_root / "alignment.json"
        if baseline_alignment_path.is_file() and not dry_run:
            baseline_result = json.loads(baseline_alignment_path.read_text(encoding="utf-8"))
        else:
            baseline_run_dir = run_rank_only_workflow(
                output_dir=baseline_root,
                extracted_demands_path=fixed_demands_path,
                dialogues_path=dialogues_path,
                stations_path=Path(eval_settings.stations_path),
                building_data_path=Path(eval_settings.building_data_path),
                priority_mode=eval_settings.baseline_mode,
                api_base=None,
                api_key=eval_settings.api_key,
                model_name="rule-only-baseline",
                conda_env=conda_env,
                dry_run=dry_run,
            )
            baseline_result = evaluate_alignment(
                run_dir=baseline_run_dir,
                demands_path=fixed_demands_path,
                dialogues_path=dialogues_path,
                ground_truth_path=ground_truth_path,
                truth_demands_path=fixed_demands_path,
                urgent_threshold=eval_settings.urgent_threshold,
                conda_env=conda_env,
                output_path=baseline_alignment_path,
                dry_run=dry_run,
            )
        baseline_by_seed[seed_name] = baseline_result
        baseline_results.append(baseline_result)

    port = _find_free_port() if not dry_run else eval_settings.port_base
    served_model_name = f"{eval_settings.served_model_name_prefix}-{trial_dir.name}"
    server_process = launch_vllm_server(
        model_path=merged_model_dir,
        conda_env=conda_env,
        settings=eval_settings,
        served_model_name=served_model_name,
        port=port,
        log_path=evaluation_root / "server.log",
        dry_run=dry_run,
    )
    try:
        api_base = f"http://127.0.0.1:{port}/v1"
        for seed_dir in val_dirs:
            seed_name = seed_dir.name
            dialogues_path = seed_dir / "dialogues.jsonl"
            ground_truth_path = seed_dir / "events_manifest.jsonl"
            fixed_demands_path = seed_dir / "llm3_sft_pipeline.jsonl"
            post_root = evaluation_root / "post_rank_only" / seed_name
            post_run_dir = run_rank_only_workflow(
                output_dir=post_root,
                extracted_demands_path=fixed_demands_path,
                dialogues_path=dialogues_path,
                stations_path=Path(eval_settings.stations_path),
                building_data_path=Path(eval_settings.building_data_path),
                priority_mode=eval_settings.priority_mode,
                api_base=api_base,
                api_key=eval_settings.api_key,
                model_name=served_model_name,
                conda_env=conda_env,
                dry_run=dry_run,
            )
            post_result = evaluate_alignment(
                run_dir=post_run_dir,
                demands_path=fixed_demands_path,
                dialogues_path=dialogues_path,
                ground_truth_path=ground_truth_path,
                truth_demands_path=fixed_demands_path,
                urgent_threshold=eval_settings.urgent_threshold,
                conda_env=conda_env,
                output_path=post_root / "alignment.json",
                dry_run=dry_run,
            )
            post_by_seed[seed_name] = post_result
            post_results.append(post_result)
    finally:
        stop_process(server_process)

    baseline_aggregate = _aggregate_alignment_results(baseline_results, eval_settings.urgent_threshold)
    post_aggregate = _aggregate_alignment_results(post_results, eval_settings.urgent_threshold)
    baseline_metrics = _alignment_snapshot(baseline_aggregate)
    post_metrics = _alignment_snapshot(post_aggregate)
    delta_metrics = _delta_snapshot(post_metrics, baseline_metrics)

    summary = {
        "enabled": True,
        "priority_mode": eval_settings.priority_mode,
        "baseline_mode": eval_settings.baseline_mode,
        "merged_model_path": str(merged_model_dir),
        "baseline": {
            "metrics": baseline_metrics,
            "by_seed": {seed: _alignment_snapshot(result) for seed, result in baseline_by_seed.items()},
        },
        "post": {
            "metrics": post_metrics,
            "by_seed": {seed: _alignment_snapshot(result) for seed, result in post_by_seed.items()},
        },
        "delta_vs_baseline": delta_metrics,
        "primary_score": _leaderboard_primary_score(post_metrics),
    }
    write_json(evaluation_root / "summary.json", summary)
    return summary


def run_sft_trial(
    *,
    trial: SFTTrial,
    train_dirs: Sequence[Path],
    val_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    output_root: Path,
    conda_env: str,
    model_path: str,
    eval_settings: EvalSettings,
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
        if payload.get("status") == "completed" and (
            not eval_settings.enabled or (payload.get("evaluation") or {}).get("enabled")
        ):
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

    evaluation = {"enabled": False}
    if status == "completed":
        evaluation = evaluate_sft_trial(
            trial_dir=trial_dir,
            latest_checkpoint=Path(latest_ckpt),
            val_dirs=val_dirs,
            conda_env=conda_env,
            eval_settings=eval_settings,
            dry_run=dry_run,
        )

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
        "evaluation": evaluation,
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
    parser.add_argument("--skip-sft-eval", action="store_true", help="Skip automatic rank-only evaluation after each SFT trial.")
    parser.add_argument("--sft-eval-priority-mode", default="llm-only", choices=("llm-only", "hybrid"))
    parser.add_argument("--sft-eval-baseline-mode", default="rule-only", choices=("rule-only",))
    parser.add_argument("--sft-eval-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--sft-eval-served-model-name-prefix", default="llm3-sweep")
    parser.add_argument("--sft-eval-host", default="0.0.0.0")
    parser.add_argument("--sft-eval-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--sft-eval-gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--sft-eval-max-model-len", type=int, default=4096)
    parser.add_argument("--sft-eval-dtype", default="bfloat16")
    parser.add_argument("--sft-eval-no-trust-remote-code", action="store_true")
    parser.add_argument("--sft-eval-urgent-threshold", type=int, default=2)
    parser.add_argument("--sft-eval-port-base", type=int, default=18080)
    parser.add_argument("--sft-eval-startup-timeout-s", type=int, default=180)
    parser.add_argument("--sft-eval-startup-poll-interval-s", type=float, default=2.0)
    parser.add_argument("--stations-path", type=Path, default=DEFAULT_STATIONS_PATH)
    parser.add_argument("--building-data-path", type=Path, default=DEFAULT_BUILDING_DATA_PATH)
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
    eval_settings = EvalSettings(
        enabled=not args.skip_sft_eval,
        priority_mode=args.sft_eval_priority_mode,
        baseline_mode=args.sft_eval_baseline_mode,
        api_key=args.sft_eval_api_key,
        served_model_name_prefix=args.sft_eval_served_model_name_prefix,
        host=args.sft_eval_host,
        tensor_parallel_size=args.sft_eval_tensor_parallel_size,
        gpu_memory_utilization=args.sft_eval_gpu_memory_utilization,
        max_model_len=args.sft_eval_max_model_len,
        dtype=args.sft_eval_dtype,
        trust_remote_code=not args.sft_eval_no_trust_remote_code,
        urgent_threshold=args.sft_eval_urgent_threshold,
        port_base=args.sft_eval_port_base,
        startup_timeout_s=args.sft_eval_startup_timeout_s,
        startup_poll_interval_s=args.sft_eval_startup_poll_interval_s,
        stations_path=str(args.stations_path.resolve()),
        building_data_path=str(args.building_data_path.resolve()),
    )

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
        "sft_eval": asdict(eval_settings),
        "dry_run": args.dry_run,
    }
    write_json(output_root / "sweep_plan.json", sweep_plan)

    trial_records_path = output_root / "trial_records.jsonl"

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
                eval_settings=eval_settings,
                export_seed=args.export_seed,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
            )
            sft_manifests.append(manifest)
            append_jsonl(trial_records_path, _build_trial_record_row(manifest))
            _refresh_sft_leaderboard(output_root)

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
                append_jsonl(trial_records_path, _build_trial_record_row(manifest))


if __name__ == "__main__":
    main()
