#!/usr/bin/env python3
"""Generate reusable ggplot-style training curves for llm3 SFT/GRPO runs.

Outputs:
- SFT loss figure
- GRPO combined comparison (absolute step)
- GRPO per-run overviews
- GRPO shared-range comparison (absolute step, clipped to overlap)
- GRPO normalized-progress comparison (0-100%)

Each figure is exported to both PNG and SVG.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

Series = List[Tuple[int, float]]
RunScalars = Dict[str, Series]

METRIC_TITLES: Dict[str, str] = {
    "train/loss": "Training Loss",
    "val/loss": "Validation Loss",
    "critic/score/mean": "Critic Score (Mean)",
    "critic/rewards/mean": "Reward (Mean)",
    "actor/ppo_kl": "PPO KL Divergence",
    "actor/entropy": "Policy Entropy",
    "actor/pg_loss": "Policy Gradient Loss",
    "actor/kl_loss": "KL Regularization Loss",
    "val-core/llm3_priority_window/acc/mean@1": "Validation Accuracy (Top-1 Mean)",
    "val-aux/llm3_priority_window/reward/mean@1": "Validation Reward (Top-1 Mean)",
}

GRPO_METRICS: List[str] = [
    "critic/score/mean",
    "critic/rewards/mean",
    "actor/ppo_kl",
    "actor/entropy",
    "actor/pg_loss",
    "actor/kl_loss",
    "val-core/llm3_priority_window/acc/mean@1",
    "val-aux/llm3_priority_window/reward/mean@1",
]

GRPO_OVERVIEW_METRICS: List[str] = [
    "critic/score/mean",
    "critic/rewards/mean",
    "actor/ppo_kl",
    "val-core/llm3_priority_window/acc/mean@1",
]


def _configure_style() -> None:
    plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 15,
        }
    )


def _load_run_scalars(run_dir: Path) -> RunScalars:
    data: Dict[str, List[Tuple[int, float]]] = {}
    event_files = sorted(run_dir.glob("events.out.tfevents*"))
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(
            str(event_file), size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            points = ea.Scalars(tag)
            if not points:
                continue
            data.setdefault(tag, []).extend((int(p.step), float(p.value)) for p in points)

    # Sort and de-duplicate by step, keeping the last seen value.
    for tag, points in list(data.items()):
        points.sort(key=lambda x: x[0])
        dedup: Dict[int, float] = {}
        for step, value in points:
            dedup[step] = value
        data[tag] = sorted(dedup.items(), key=lambda x: x[0])
    return data


def _save_png_svg(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=180)
    fig.savefig(output_dir / f"{stem}.svg")


def _plot_sft_loss(output_dir: Path, sft_data: RunScalars) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for tag in ("train/loss", "val/loss"):
        points = sft_data.get(tag, [])
        if len(points) < 2:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, linewidth=2.2, label=METRIC_TITLES.get(tag, tag))
    ax.set_title("SFT Learning Curves")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    _save_png_svg(fig, output_dir, "expA_sft_loss_curves_ggplot")
    plt.close(fig)


def _plot_grpo_combined_absolute(
    output_dir: Path,
    run_b_data: RunScalars,
    run_c_data: RunScalars,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=False)
    axes = axes.flatten()

    for i, metric in enumerate(GRPO_METRICS):
        ax = axes[i]
        for run_name, run_data in (
            ("expB-grpo-hard", run_b_data),
            ("expC-grpo-mixed", run_c_data),
        ):
            series = run_data.get(metric, [])
            if len(series) < 2:
                continue
            xs = [p[0] for p in series]
            ys = [p[1] for p in series]
            ax.plot(xs, ys, linewidth=1.9, label=run_name)
        ax.set_title(METRIC_TITLES.get(metric, metric))
        ax.set_xlabel("Step")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    # No suptitle for large multi-subplot figures.
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_png_svg(fig, output_dir, "grpo_combined_subplots_expB_expC_ggplot")
    plt.close(fig)


def _plot_grpo_overview(output_dir: Path, run_name: str, run_data: RunScalars) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=False)
    axes = axes.flatten()
    for ax, metric in zip(axes, GRPO_OVERVIEW_METRICS):
        series = run_data.get(metric, [])
        if len(series) >= 2:
            xs = [p[0] for p in series]
            ys = [p[1] for p in series]
            ax.plot(xs, ys, linewidth=2.0)
        ax.set_title(METRIC_TITLES.get(metric, metric))
        ax.set_xlabel("Step")
    fig.tight_layout()
    _save_png_svg(fig, output_dir, f"{run_name.replace('-', '_')}_overview_ggplot")
    plt.close(fig)


def _clip_to_max_step(series: Series, max_step: int) -> Series:
    return [(step, value) for step, value in series if step <= max_step]


def _normalize_step_to_pct(series: Series) -> List[Tuple[float, float]]:
    if not series:
        return []
    max_step = max(step for step, _ in series)
    if max_step <= 0:
        return [(0.0, value) for _, value in series]
    return [((step / max_step) * 100.0, value) for step, value in series]


def _plot_grpo_combined_shared_range(
    output_dir: Path,
    run_b_data: RunScalars,
    run_c_data: RunScalars,
) -> None:
    max_step_b = max((p[0] for p in run_b_data.get("training/global_step", [])), default=None)
    max_step_c = max((p[0] for p in run_c_data.get("training/global_step", [])), default=None)

    if max_step_b is None or max_step_c is None:
        # Fallback: infer from first metric that exists in both runs.
        candidates = [m for m in GRPO_METRICS if run_b_data.get(m) and run_c_data.get(m)]
        if not candidates:
            return
        m = candidates[0]
        max_step_b = max(step for step, _ in run_b_data[m])
        max_step_c = max(step for step, _ in run_c_data[m])

    shared_max_step = min(max_step_b, max_step_c)

    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=False)
    axes = axes.flatten()

    for i, metric in enumerate(GRPO_METRICS):
        ax = axes[i]
        for run_name, run_data in (
            ("expB-grpo-hard", run_b_data),
            ("expC-grpo-mixed", run_c_data),
        ):
            clipped = _clip_to_max_step(run_data.get(metric, []), shared_max_step)
            if len(clipped) < 2:
                continue
            xs = [p[0] for p in clipped]
            ys = [p[1] for p in clipped]
            ax.plot(xs, ys, linewidth=1.9, label=run_name)
        ax.set_title(METRIC_TITLES.get(metric, metric))
        ax.set_xlabel("Step (Shared Range)")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_png_svg(fig, output_dir, "grpo_combined_subplots_expB_expC_shared_range_ggplot")
    plt.close(fig)


def _plot_grpo_combined_normalized(
    output_dir: Path,
    run_b_data: RunScalars,
    run_c_data: RunScalars,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=False)
    axes = axes.flatten()

    for i, metric in enumerate(GRPO_METRICS):
        ax = axes[i]
        for run_name, run_data in (
            ("expB-grpo-hard", run_b_data),
            ("expC-grpo-mixed", run_c_data),
        ):
            normalized = _normalize_step_to_pct(run_data.get(metric, []))
            if len(normalized) < 2:
                continue
            xs = [p[0] for p in normalized]
            ys = [p[1] for p in normalized]
            ax.plot(xs, ys, linewidth=1.9, label=run_name)
        ax.set_title(METRIC_TITLES.get(metric, metric))
        ax.set_xlabel("Progress (% of Run)")
        ax.set_xlim(0, 100)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_png_svg(fig, output_dir, "grpo_combined_subplots_expB_expC_normalized_ggplot")
    plt.close(fig)


def _write_readme(output_dir: Path) -> None:
    readme = output_dir / "README.md"
    readme.write_text(
        "# large_v2 训练过程图\n\n"
        "## ggplot 风格（PNG + SVG）\n"
        "- `expA_sft_loss_curves_ggplot`: SFT train/val loss\n"
        "- `grpo_combined_subplots_expB_expC_ggplot`: GRPO 对比（绝对 step，全程）\n"
        "- `grpo_combined_subplots_expB_expC_shared_range_ggplot`: GRPO 对比（共同 step 区间）\n"
        "- `grpo_combined_subplots_expB_expC_normalized_ggplot`: GRPO 对比（归一化进度 0-100%）\n"
        "- `expB_grpo_hard_overview_ggplot`: expB 关键指标总览\n"
        "- `expC_grpo_mixed_overview_ggplot`: expC 关键指标总览\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot llm3 SFT/GRPO curves in ggplot style.")
    parser.add_argument(
        "--sft-run",
        default="tensorboard_log/llm3-sft/expA-sft-baseline",
        help="Path to SFT TensorBoard run directory.",
    )
    parser.add_argument(
        "--grpo-run-b",
        default="tensorboard_log/llm3-grpo/expB-grpo-hard",
        help="Path to first GRPO run directory (expB).",
    )
    parser.add_argument(
        "--grpo-run-c",
        default="tensorboard_log/llm3-grpo/expC-grpo-mixed",
        help="Path to second GRPO run directory (expC).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots/large_v2",
        help="Directory for generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    sft_dir = (root / args.sft_run).resolve()
    grpo_b_dir = (root / args.grpo_run_b).resolve()
    grpo_c_dir = (root / args.grpo_run_c).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_style()

    sft_data = _load_run_scalars(sft_dir)
    grpo_b_data = _load_run_scalars(grpo_b_dir)
    grpo_c_data = _load_run_scalars(grpo_c_dir)

    _plot_sft_loss(output_dir, sft_data)
    _plot_grpo_combined_absolute(output_dir, grpo_b_data, grpo_c_data)
    _plot_grpo_combined_shared_range(output_dir, grpo_b_data, grpo_c_data)
    _plot_grpo_combined_normalized(output_dir, grpo_b_data, grpo_c_data)
    _plot_grpo_overview(output_dir, "expB-grpo-hard", grpo_b_data)
    _plot_grpo_overview(output_dir, "expC-grpo-mixed", grpo_c_data)
    _write_readme(output_dir)

    print(f"output_dir={output_dir}")
    for p in sorted(output_dir.glob("*_ggplot.png")):
        print(f"png={p.name}")
    for p in sorted(output_dir.glob("*_ggplot.svg")):
        print(f"svg={p.name}")


if __name__ == "__main__":
    main()
