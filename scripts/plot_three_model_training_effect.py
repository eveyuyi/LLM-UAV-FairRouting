#!/usr/bin/env python3
"""Plot focused comparison for qwen3_4b_2507 vs expA_gs594 vs expB_gs171.

Outputs:
  - data/eval_runs/plots/training_effect_3models_hard_eval_llm-only.png
  - data/eval_runs/plots/training_effect_3models_hard_eval_hybrid.png
  - data/eval_runs/plots/training_effect_3models_norm_eval_llm-only.png
  - data/eval_runs/plots/training_effect_3models_norm_eval_hybrid.png
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/eval_runs/plots"

CSV_BY_SPLIT = {
    "hard_eval": ROOT / "data/eval_runs/test_seeds_hard_eval/model_metrics_summary.csv",
    "norm_eval": ROOT / "data/eval_runs/test_seeds_norm_eval/model_metrics_summary.csv",
}

TAGS = ["qwen3_4b_2507", "expA_gs594", "expB_gs171"]
DISPLAY_NAMES = {
    "qwen3_4b_2507": "Qwen3-4B (base)",
    "expA_gs594": "ExpA gs594",
    "expB_gs171": "ExpB gs171",
}
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


def plot_split_mode(split_name: str, mode: str, csv_path: Path, out_path: Path) -> None:
    rows = load_rows(csv_path)
    x = np.arange(len(TAGS))

    fig, axes = plt.subplots(4, 2, figsize=(17, 15))
    axes = axes.flatten()

    for idx, (metric_key, metric_title) in enumerate(METRICS):
        ax = axes[idx]
        y = [rows[(mode, tag)][metric_key] for tag in TAGS]
        colors = ["#8c8c8c", "#4c78a8", "#f58518"]
        ax.plot(x, y, color="#4c4c4c", linewidth=1.8, alpha=0.85, zorder=2)
        for xi, yi, c in zip(x, y, colors):
            ax.scatter([xi], [yi], color=c, s=85, zorder=3)

        # highlight training gain over qwen baseline within the same mode
        base = y[0]
        for j in [1, 2]:
            gain = y[j] - base
            ax.text(
                x[j],
                y[j],
                f"{gain:+.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="darkgreen" if gain >= 0 else "darkred",
                fontweight="bold",
            )
        ax.axhline(base, linestyle="--", linewidth=1.2, alpha=0.8, color="#666666")

        # Use a tight, non-zero y-range to make training gains visually salient.
        y_min = min(y)
        y_max = max(y)
        if y_max > y_min:
            pad = max((y_max - y_min) * 0.25, 0.003)
        else:
            pad = max(abs(y_max) * 0.05, 0.003)
        ax.set_ylim(y_min - pad, y_max + pad)

        ax.set_title(metric_title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES[t] for t in TAGS], rotation=18, ha="right")
        if idx % 2 == 0:
            ax.set_ylabel("Score")

    fig.suptitle(
        f"Training Effect Focus ({mode}) - {split_name}\n"
        "Dashed line is Qwen baseline; labels are gain vs Qwen",
        fontsize=15,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, csv_path in CSV_BY_SPLIT.items():
        for mode in MODES:
            out_path = OUT_DIR / f"training_effect_3models_{split}_{mode}.png"
            plot_split_mode(split, mode, csv_path, out_path)
            print(out_path)


if __name__ == "__main__":
    main()
