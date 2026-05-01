"""S2: Computational efficiency analysis.

Reads workflow.log and workflow_results.json from one or more run directories,
extracts per-module timing, and outputs a summary table.

Usage:
    python scripts/analyze_s2_timing.py \\
        --run-dirs data/eval_runs/m0c_*/run_* data/eval_runs/m1_*/run_* \\
        --output data/evals/s2_timing_summary.json
"""
import argparse
import json
import re
import statistics
from pathlib import Path


_STEP_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?Step (\d+):"
)
_MODULE3A_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?\[Module 3a\].*?Produced (\d+) demand configs"
)


def _parse_log_timings(log_path: Path) -> dict:
    """Extract step-level timings from workflow.log."""
    from datetime import datetime

    lines = log_path.read_text(encoding="utf-8").splitlines()
    step_times: dict[str, datetime] = {}
    for line in lines:
        m = _STEP_RE.search(line)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            step_times[f"step_{m.group(2)}"] = ts

    timings = {}
    if "step_2" in step_times and "step_3" in step_times:
        timings["module2_s"] = (step_times["step_3"] - step_times["step_2"]).total_seconds()
    return timings


def _parse_solver_timings(results_path: Path) -> list[float]:
    """Extract per-window NSGA-III solve times from workflow_results.json."""
    data = json.loads(results_path.read_text(encoding="utf-8"))
    times = []
    windows = data if isinstance(data, list) else data.get("solutions", [])
    for w in windows:
        rt = w.get("solve_time_s")
        if rt is None:
            analytics = (w.get("solution") or {}).get("solver_analytics") or {}
            rt = analytics.get("search_runtime_s")
        if rt is not None:
            times.append(float(rt))
    return times


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_s": None, "std_s": None, "min_s": None, "max_s": None}
    return {
        "count": len(values),
        "mean_s": round(statistics.mean(values), 3),
        "std_s": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min_s": round(min(values), 3),
        "max_s": round(max(values), 3),
    }


def _percentile(vals: list[float], p: float) -> float:
    s = sorted(vals)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 3)


def main():
    parser = argparse.ArgumentParser(description="S2: Timing analysis across run dirs")
    parser.add_argument("--run-dirs", nargs="+", required=True,
                        help="Paths to run_* directories (glob-expanded by shell)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Optional labels for each run-dir (same order)")
    parser.add_argument("--output", default="evals/results/s2_timing_summary.json")
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.run_dirs if Path(p).is_dir()]
    if not run_dirs:
        print("[error] No valid run directories found.")
        return

    labels = args.labels or [d.parent.name for d in run_dirs]
    if len(labels) < len(run_dirs):
        labels += [run_dirs[i].parent.name for i in range(len(labels), len(run_dirs))]

    per_method: dict[str, dict] = {}
    solver_times_all = []

    for label, run_dir in zip(labels, run_dirs):
        results_path = run_dir / "workflow_results.json"
        solver_times: list[float] = []
        if results_path.exists():
            solver_times = _parse_solver_timings(results_path)
            solver_times_all.extend(solver_times)
        per_method[label] = _summarize(solver_times)
        if solver_times:
            per_method[label]["p90_s"] = _percentile(solver_times, 90)
            per_method[label]["median_s"] = _percentile(solver_times, 50)

    summary = {
        "n_run_dirs": len(run_dirs),
        "nsga3_per_window_all": _summarize(solver_times_all),
        "per_method": per_method,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Timing summary saved to {out_path}")

    print("\n=== S2 Solver Timing (NSGA-III per window) ===")
    header = f"{'Method':<14} {'N':>4}  {'Mean(s)':>8}  {'Median(s)':>9}  {'P90(s)':>7}  {'Max(s)':>7}"
    print(header)
    for label, m in per_method.items():
        if m["count"] == 0:
            print(f"{label:<14}    0  (no data)")
            continue
        print(f"{label:<14} {m['count']:>4}  {m['mean_s']:>8.3f}  "
              f"{m.get('median_s', 0):>9.3f}  {m.get('p90_s', 0):>7.3f}  {m['max_s']:>7.3f}")


if __name__ == "__main__":
    main()
