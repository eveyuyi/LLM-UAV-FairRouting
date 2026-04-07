#!/usr/bin/env python3
"""Export training curves (PNG) from TensorBoard events or text logs."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def _find_event_files(input_root: Path) -> List[Path]:
    return sorted(input_root.glob("**/events.out.tfevents*"))


def _extract_scalars_from_events(event_files: Sequence[Path]) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    from tensorboard.backend.event_processing import event_accumulator

    per_run: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for event_file in event_files:
        run_dir = str(event_file.parent)
        ea = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if not tags:
            continue
        run_data: Dict[str, List[Tuple[int, float]]] = {}
        for tag in tags:
            points = [(int(p.step), float(p.value)) for p in ea.Scalars(tag)]
            if points:
                run_data[tag] = points
        if run_data:
            per_run[run_dir] = run_data
    return per_run


_STEP_RE = re.compile(r"\b(?:global_)?step\b\s*[:=]\s*(\d+)", re.IGNORECASE)
_KV_FLOAT_RE = re.compile(
    r"\b([A-Za-z0-9_.\-/]*(?:loss|reward|score|kl|entropy|acc|accuracy|ppl)[A-Za-z0-9_.\-/]*)\b\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)"
)


def _iter_log_files(input_root: Path) -> Iterable[Path]:
    for p in sorted(input_root.glob("**/*.log")):
        if p.is_file() and p.stat().st_size > 0:
            yield p


def _extract_scalars_from_logs(input_root: Path) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    per_run: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for log_file in _iter_log_files(input_root):
        run_dir = str(log_file.parent)
        series: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        line_idx = 0
        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_idx += 1
                step_match = _STEP_RE.search(line)
                step = int(step_match.group(1)) if step_match else line_idx
                for key, raw in _KV_FLOAT_RE.findall(line):
                    value = float(raw)
                    if math.isfinite(value):
                        series[key.lower()].append((step, value))
        compact = {k: v for k, v in series.items() if len(v) >= 2}
        if compact:
            per_run[run_dir] = compact
    return per_run


def _is_target_metric(tag: str, mode: str) -> bool:
    lower = tag.lower()
    if mode == "all":
        return True
    if mode == "sft":
        return "loss" in lower or "ppl" in lower or "acc" in lower
    if mode == "grpo":
        return "reward" in lower or "score" in lower or "kl" in lower or "entropy" in lower
    return ("loss" in lower) or ("reward" in lower) or ("score" in lower)


def _plot_series(run_name: str, metric: str, points: Sequence[Tuple[int, float]], out_file: Path) -> None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, linewidth=1.8)
    plt.xlabel("step")
    plt.ylabel(metric)
    plt.title(f"{run_name} | {metric}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=140)
    plt.close()


def _write_index(index_file: Path, entries: Sequence[Tuple[str, str, str]]) -> None:
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with index_file.open("w", encoding="utf-8") as f:
        f.write("# Exported training curves\n\n")
        if not entries:
            f.write("No curves were exported.\n")
            return
        f.write("| Run | Metric | PNG |\n")
        f.write("|---|---|---|\n")
        for run, metric, png in entries:
            f.write(f"| `{run}` | `{metric}` | `{png}` |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export training curves to PNG.")
    parser.add_argument("--input-root", default="data/hydra_outputs", help="Hydra output root directory.")
    parser.add_argument("--output-dir", default="data/plots", help="PNG output directory.")
    parser.add_argument(
        "--mode",
        choices=["auto", "sft", "grpo", "all"],
        default="auto",
        help="Metric filter mode. auto keeps common loss/reward-like metrics.",
    )
    parser.add_argument("--max-curves-per-run", type=int, default=30, help="Maximum curves exported per run.")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    event_files = _find_event_files(input_root)
    source = "events"
    if event_files:
        per_run = _extract_scalars_from_events(event_files)
    else:
        source = "logs"
        per_run = _extract_scalars_from_logs(input_root)

    entries: List[Tuple[str, str, str]] = []
    total = 0
    for run_dir, metrics in sorted(per_run.items()):
        run_name = str(Path(run_dir).relative_to(input_root))
        selected = [m for m in sorted(metrics) if _is_target_metric(m, args.mode)]
        if args.mode == "auto":
            # In auto mode, suppress overly noisy metrics if we found clear loss/reward curves.
            core = [m for m in selected if ("loss" in m or "reward" in m or "score" in m)]
            if core:
                selected = core
        selected = selected[: args.max_curves_per_run]
        for metric in selected:
            points = metrics[metric]
            if len(points) < 2:
                continue
            out_file = output_dir / f"{_safe_name(run_name)}__{_safe_name(metric)}.png"
            _plot_series(run_name=run_name, metric=metric, points=points, out_file=out_file)
            entries.append((run_name, metric, str(out_file)))
            total += 1

    _write_index(output_dir / "README.md", entries)
    print(f"source={source}")
    print(f"runs={len(per_run)}")
    print(f"curves_exported={total}")
    print(f"output_dir={output_dir}")
    print(f"index={output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
