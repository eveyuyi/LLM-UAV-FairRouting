"""S2 — Computational Efficiency Analysis.

Computes per-method solver timing, path cache efficiency, and demand throughput.

Usage:
  python scripts/eval_efficiency.py --seed 4111
"""
from __future__ import annotations
import argparse, json, re, subprocess
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent

METHODS = [
    ("M0a",    "formal_m0a_seed{seed}"),
    ("M0b",    "formal_m0b_seed{seed}"),
    ("M0c",    "formal_m0c_seed{seed}"),
    ("M1_pre", "formal_m1_pre_p2_seed{seed}"),
    ("M1_ft",  "formal_m1_ft_p2_seed{seed}"),
]

def find_latest_run(d: Path) -> Optional[Path]:
    runs = sorted(d.glob("run_*/"), reverse=True)
    return runs[0] if runs else None

def _mean(vals): return round(sum(vals)/len(vals), 2) if vals else None
def _median(vals):
    if not vals: return None
    s = sorted(vals); n = len(s)
    return round((s[n//2-1]+s[n//2])/2 if n%2==0 else s[n//2], 2)
def _pct(vals, p):
    if not vals: return None
    s = sorted(vals); idx = (len(s)-1)*p/100
    lo, hi = int(idx), min(int(idx)+1, len(s)-1)
    return round(s[lo] + (s[hi]-s[lo])*(idx-lo), 2)

def solver_stats(workflow_results: List[dict]) -> dict:
    times = [float(w["solve_time_s"]) for w in workflow_results if w.get("solve_time_s") is not None]
    n_demands = [w.get("n_demands_feasible", 0) for w in workflow_results]
    drones = [w.get("drones_used", 0) for w in workflow_results if w.get("drones_used")]
    dist   = [w.get("total_distance_m", 0) or 0 for w in workflow_results]
    return {
        "n_windows":         len(workflow_results),
        "avg_solve_time_s":  _mean(times),
        "median_solve_time_s": _median(times),
        "p90_solve_time_s":  _pct(times, 90),
        "total_solve_min":   round(sum(times)/60, 1) if times else None,
        "avg_demands_per_window": _mean(n_demands),
        "avg_drones_used":   _mean(drones),
        "total_distance_km": round(sum(dist)/1000, 1),
    }

def cache_stats_from_log(log_path: Path) -> dict:
    hits = misses = 0
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        hits   = len(re.findall(r"\[Path Cache\] \(disk\)", text))
        misses = len(re.findall(r"\[Path Cache \d+\] computing", text))
    except Exception:
        pass
    total = hits + misses
    return {
        "cache_hits":   hits,
        "cache_misses": misses,
        "cache_total":  total,
        "hit_rate_pct": round(hits/total*100, 1) if total > 0 else None,
    }

def _fmt(v, suffix="") -> str:
    return f"{v}{suffix}" if v is not None else "—"

def print_table(results: Dict[str, dict]) -> None:
    methods = list(results.keys())
    header = " | ".join(f"**{m}**" for m in methods)
    sep    = " | ".join("---" for _ in methods)
    def row(label, fn):
        return f"| {label:<38} | " + " | ".join(fn(results[m]) for m in methods) + " |"

    print("\n## S2 — Computational Efficiency\n")
    print(f"| {'Metric':<38} | {header} |")
    print(f"| {'---':<38} | {sep} |")
    print(row("Windows",                    lambda x: _fmt(x.get("solver","").get("n_windows"))))
    print(row("Avg solve time (s/window)",  lambda x: _fmt(x.get("solver","").get("avg_solve_time_s"))))
    print(row("P90 solve time (s/window)",  lambda x: _fmt(x.get("solver","").get("p90_solve_time_s"))))
    print(row("Total solver time (min)",    lambda x: _fmt(x.get("solver","").get("total_solve_min"))))
    print(row("Avg demands / window",       lambda x: _fmt(x.get("solver","").get("avg_demands_per_window"))))
    print(row("Avg drones used / window",   lambda x: _fmt(x.get("solver","").get("avg_drones_used"))))
    print(row("Total distance (km)",        lambda x: _fmt(x.get("solver","").get("total_distance_km"))))
    print(row("Path cache hits",            lambda x: _fmt(x.get("cache","").get("cache_hits"))))
    print(row("Path cache misses",          lambda x: _fmt(x.get("cache","").get("cache_misses"))))
    print(row("**Cache hit rate (%)**",     lambda x: _fmt(x.get("cache","").get("hit_rate_pct"), "%")))
    print()
    print("> Note: LLM2 and LLM3 module latencies are not captured in offline mode.")
    print("> Re-run one seed online (without --offline / --extracted-demands) to measure them.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="4111")
    parser.add_argument("--output", default="evals/results/efficiency_seed{seed}.json")
    args = parser.parse_args()

    # overall path cache size
    cache_dir = ROOT / "data" / "path_cache"
    total_cached = sum(1 for _ in cache_dir.rglob("*.json"))
    print(f"Path cache total entries: {total_cached:,}")

    all_results: Dict[str, dict] = {}
    for label, dir_tpl in METHODS:
        eval_dir = ROOT / "data" / "eval_runs" / dir_tpl.format(seed=args.seed)
        run_dir  = find_latest_run(eval_dir)
        if run_dir is None:
            print(f"  [{label}] no run found")
            continue
        wr_path  = run_dir / "workflow_results.json"
        log_path = run_dir / "workflow.log"
        if not wr_path.exists():
            print(f"  [{label}] no workflow_results.json")
            continue
        wr = json.loads(wr_path.read_text(encoding="utf-8"))
        print(f"  [{label}] {run_dir.name}")
        all_results[label] = {
            "run_dir":   str(run_dir),
            "solver":    solver_stats(wr),
            "cache":     cache_stats_from_log(log_path) if log_path.exists() else {},
        }

    print_table(all_results)
    all_results["path_cache_total_entries"] = total_cached

    out = Path(args.output.format(seed=args.seed))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
