#!/usr/bin/env python3
"""Plot training-effect figures from eval summary CSVs.

Input CSVs:
  - data/eval_runs/test_seeds_hard_eval/model_metrics_summary.csv
  - data/eval_runs/test_seeds_norm_eval/model_metrics_summary.csv

Outputs:
  - data/eval_runs/plots/training_effect_hard_eval.png
  - data/eval_runs/plots/training_effect_norm_eval.png
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
HARD_CSV = ROOT / "data/eval_runs/test_seeds_hard_eval/model_metrics_summary.csv"
NORM_CSV = ROOT / "data/eval_runs/test_seeds_norm_eval/model_metrics_summary.csv"
OUT_DIR = ROOT / "data/eval_runs/plots"

TRAINING_TAGS = [
    "expA_gs594",
    "expB_gs50",
    "expB_gs100",
    "expB_gs150",
    "expB_gs171",
    "expC_gs100",
    "expC_gs300",
    "expC_gs500",
    "expC_gs600",
    "expC_gs700",
    "expC_gs800",
    "expC_gs900",
    "expC_gs992",
]
TAG_LABELS = [
    "A594",
    "B50",
    "B100",
    "B150",
    "B171",
    "C100",
    "C300",
    "C500",
    "C600",
    "C700",
    "C800",
    "C900",
    "C992",
]
BASELINE_TAG = "qwen3_4b_2507"
MODES = ["llm-only", "hybrid"]

METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
    ("spearman", "Spearman"),
    ("kendall_tau", "Kendall Tau"),
    ("urgent_f1", "Urgent F1"),
    ("p1_f1", "Priority-1 F1"),
    ("topk_hit_rate", "Top-k Hit Rate"),
]


def _to_float(value: str) -> float:
    return float(value) if value not in ("", "NA", "None", "nan") else float("nan")


def load_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    data: Dict[Tuple[str, str], Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["mode"], row["tag"])
            data[key] = {metric: _to_float(row[metric]) for metric, _ in METRICS}
    return data


def plot_split(path: Path, split_name: str, out_path: Path) -> None:
    rows = load_rows(path)

    fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True)
    axes = axes.flatten()
    x = list(range(len(TRAINING_TAGS)))

    for idx, (metric_key, metric_title) in enumerate(METRICS):
        ax = axes[idx]
        for mode in MODES:
            y = [rows[(mode, tag)][metric_key] for tag in TRAINING_TAGS]
            baseline = rows[(mode, BASELINE_TAG)][metric_key]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                label=f"{mode} (trained)",
            )
            ax.axhline(
                baseline,
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
                label=f"{mode} baseline(qwen)",
            )

        ax.set_title(metric_title, fontsize=11)
        ax.grid(alpha=0.25)
        if idx % 2 == 0:
            ax.set_ylabel("Score")

    for ax in axes[-2:]:
        ax.set_xticks(x)
        ax.set_xticklabels(TAG_LABELS, rotation=45, ha="right")
        ax.set_xlabel("Training checkpoints")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(
        f"Training Effect on {split_name} (seed-averaged metrics)",
        fontsize=15,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_split(HARD_CSV, "hard_eval", OUT_DIR / "training_effect_hard_eval.png")
    plot_split(NORM_CSV, "norm_eval", OUT_DIR / "training_effect_norm_eval.png")
    print(OUT_DIR / "training_effect_hard_eval.png")
    print(OUT_DIR / "training_effect_norm_eval.png")


if __name__ == "__main__":
    main()
