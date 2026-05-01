"""
Formal experiment evaluation: M0a / M0b / M0c / M1_pre / M1_ft
Computes operational and fairness metrics, prints a markdown table,
and saves detailed JSON to evals/results/.

Usage:
  PYTHONPATH=src python scripts/eval_formal_comparison.py \
      --seed 4111 --split norm_eval \
      --output evals/results/formal_comparison_seed4111.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── constants ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
EVAL_ROOT = ROOT / "evals" / "results"

METHODS: List[Tuple[str, str]] = [
    ("M0a",      "formal_m0a_seed{seed}"),
    ("M0b",      "formal_m0b_seed{seed}"),
    ("M0c",      "formal_m0c_seed{seed}"),
    ("M1_pre",   "formal_m1_pre_p2_seed{seed}"),
    ("M1_ft",    "formal_m1_ft_p2_seed{seed}"),
    ("M1_gemini","formal_m1_gemini_seed{seed}"),
]

PRIORITY_WEIGHTS = {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}


# ── loaders ────────────────────────────────────────────────────────────────
def load_events_manifest(path: Path) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            index[ev["event_id"]] = ev
    return index


def load_workflow_results(run_dir: Path) -> List[dict]:
    path = run_dir / "workflow_results.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), f"Expected list in {path}"
    return data


def find_latest_run(eval_dir: Path) -> Optional[Path]:
    runs = sorted(eval_dir.glob("run_*/"), reverse=True)
    return runs[0] if runs else None


# ── metric helpers ──────────────────────────────────────────────────────────
def _mean(vals: List[float]) -> Optional[float]:
    return round(sum(vals) / len(vals), 2) if vals else None


def _pctl(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 2)


def _gini(vals: List[float]) -> Optional[float]:
    """Gini coefficient of a list of non-negative values. 0=equal, 1=maximal."""
    if len(vals) < 2:
        return None
    s = sorted(vals)
    n = len(s)
    total = sum(s)
    if total == 0:
        return 0.0
    cumsum = sum((i + 1) * v for i, v in enumerate(s))
    return round((2 * cumsum / (n * total)) - (n + 1) / n, 4)


def _jain(rates: List[float]) -> Optional[float]:
    """Jain's fairness index on a list of rates."""
    if not rates:
        return None
    n = len(rates)
    num = sum(rates) ** 2
    den = n * sum(r ** 2 for r in rates)
    return round(num / den, 4) if den > 0 else None


def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den > 0 else None


# ── per-demand row building ─────────────────────────────────────────────────
def build_rows(results: List[dict], manifest: Dict[str, dict]) -> List[dict]:
    """Flatten workflow_results into one row per demand, joined with ground truth."""
    rows: List[dict] = []
    for window in results:
        for d in window.get("per_demand_results", []):
            event_id = d.get("source_event_id") or d.get("demand_id", "")
            gt = manifest.get(event_id, {})
            vuln = gt.get("population_vulnerability") or {}
            rows.append({
                "event_id": event_id,
                "time_window": window.get("time_window", ""),
                # ground-truth priority (not method-assigned)
                "true_priority": int(gt.get("latent_priority", d.get("priority", 4))),
                "demand_tier": d.get("demand_tier", gt.get("demand_tier", "")),
                "elderly_involved": bool(vuln.get("elderly_involved") or d.get("elderly_involved")),
                "vulnerable_community": bool(vuln.get("vulnerable_community") or d.get("vulnerable_community")),
                "deadline_minutes": d.get("deadline_minutes") or gt.get("deadline_minutes"),
                "is_served": bool(d.get("is_served")),
                "is_deadline_met": bool(d.get("is_deadline_met")),
                "delivery_latency_min": d.get("delivery_latency_min"),
                "assigned_drone": d.get("assigned_drone"),
            })
    return rows


# ── subset metrics ──────────────────────────────────────────────────────────
def subset_metrics(rows: List[dict]) -> dict:
    if not rows:
        return {"count": 0}
    served = [r for r in rows if r["is_served"]]
    on_time = [r for r in rows if r["is_deadline_met"]]
    latencies = [r["delivery_latency_min"] for r in served if r["delivery_latency_min"] is not None]
    return {
        "count": len(rows),
        "served": len(served),
        "service_rate": _rate(len(served), len(rows)),
        "on_time_rate": _rate(len(on_time), len(rows)),
        "avg_latency_min": _mean(latencies),
        "p90_latency_min": _pctl(latencies, 90),
    }


# ── full method summary ─────────────────────────────────────────────────────
def method_metrics(rows: List[dict]) -> dict:
    # by priority tier
    by_prio = {p: subset_metrics([r for r in rows if r["true_priority"] == p]) for p in range(1, 5)}
    urgent = subset_metrics([r for r in rows if r["true_priority"] <= 2])
    overall = subset_metrics(rows)

    # priority-weighted scores
    served_rows = [r for r in rows if r["is_served"]]
    latencies_all = [r["delivery_latency_min"] for r in served_rows if r["delivery_latency_min"] is not None]

    def pw_score(key: str) -> Optional[float]:
        tw, aw = 0.0, 0.0
        for r in rows:
            w = PRIORITY_WEIGHTS.get(r["true_priority"], 1.0)
            tw += w
            aw += w * float(bool(r.get(key)))
        return round(aw / tw, 4) if tw > 0 else None

    # fairness: latency by priority tier → Jain index
    tier_avg_latencies = [
        m["avg_latency_min"]
        for m in by_prio.values()
        if m.get("avg_latency_min") is not None
    ]
    jain_latency = _jain([1 / l for l in tier_avg_latencies if l and l > 0]) if tier_avg_latencies else None

    # latency gap p1 vs p4
    lat_p1 = by_prio[1].get("avg_latency_min")
    lat_p4 = by_prio[4].get("avg_latency_min")
    latency_gap = round(lat_p4 - lat_p1, 2) if lat_p1 and lat_p4 else None

    # vulnerable group
    elderly = subset_metrics([r for r in rows if r["elderly_involved"]])
    vuln_comm = subset_metrics([r for r in rows if r["vulnerable_community"]])

    # window-level distance total
    return {
        "overall": overall,
        "by_priority": by_prio,
        "urgent": urgent,
        "priority_weighted_service_score": pw_score("is_served"),
        "priority_weighted_on_time_score": pw_score("is_deadline_met"),
        "gini_latency": _gini(latencies_all),
        "jain_fairness_by_tier": jain_latency,
        "latency_gap_p1_vs_p4_min": latency_gap,
        "elderly_involved": elderly,
        "vulnerable_community": vuln_comm,
    }


# ── window-level aggregate (distance, noise) ───────────────────────────────
def window_aggregate(results: List[dict]) -> dict:
    total_dist = sum(w.get("total_distance_m", 0) or 0 for w in results)
    total_noise = sum(w.get("total_noise_impact", 0) or 0 for w in results)
    n_windows = len(results)
    solve_times = [w.get("solve_time_s") for w in results if w.get("solve_time_s") is not None]
    return {
        "n_windows": n_windows,
        "total_distance_km": round(total_dist / 1000, 2),
        "total_noise_impact": round(total_noise, 2),
        "avg_solve_time_s": _mean([float(t) for t in solve_times]),
    }


# ── markdown table ──────────────────────────────────────────────────────────
def _fmt(v, pct: bool = False, decimals: int = 1) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.{decimals}f}%"
    return f"{v:.{decimals}f}"


def print_markdown_table(summary: dict) -> None:
    methods = list(summary["methods"].keys())
    m = summary["methods"]

    def row(label: str, fn) -> str:
        vals = " | ".join(fn(m[name]) for name in methods)
        return f"| {label:<42} | {vals} |"

    header = " | ".join(f"**{n}**" for n in methods)
    sep = " | ".join("---" for _ in methods)

    print("\n## Formal Experiment: M0 vs M1 — seed=4111, 96 windows\n")
    print(f"| {'Metric':<42} | {header} |")
    print(f"| {'---':<42} | {sep} |")

    # overall
    print(row("**Overall service rate**",   lambda x: _fmt(x["overall"]["service_rate"], pct=True)))
    print(row("**Overall on-time rate**",   lambda x: _fmt(x["overall"]["on_time_rate"], pct=True)))
    print(row("**Avg latency (min)**",      lambda x: _fmt(x["overall"]["avg_latency_min"])))
    print(row("**P90 latency (min)**",      lambda x: _fmt(x["overall"]["p90_latency_min"])))

    # priority 1
    print(row("Priority-1 service rate",    lambda x: _fmt(x["by_priority"][1]["service_rate"], pct=True)))
    print(row("Priority-1 on-time rate",    lambda x: _fmt(x["by_priority"][1]["on_time_rate"], pct=True)))
    print(row("Priority-1 avg latency (min)", lambda x: _fmt(x["by_priority"][1]["avg_latency_min"])))

    # urgent (p1+p2)
    print(row("Urgent (p1+p2) service rate",  lambda x: _fmt(x["urgent"]["service_rate"], pct=True)))
    print(row("Urgent (p1+p2) on-time rate",  lambda x: _fmt(x["urgent"]["on_time_rate"], pct=True)))
    print(row("Urgent avg latency (min)",     lambda x: _fmt(x["urgent"]["avg_latency_min"])))

    # priority-weighted
    print(row("**Priority-weighted service**", lambda x: _fmt(x["priority_weighted_service_score"], pct=True)))
    print(row("**Priority-weighted on-time**", lambda x: _fmt(x["priority_weighted_on_time_score"], pct=True)))

    # fairness
    print(row("Latency gap p1 vs p4 (min)",  lambda x: _fmt(x["latency_gap_p1_vs_p4_min"])))
    print(row("Gini coeff (latency)",        lambda x: _fmt(x["gini_latency"], decimals=3)))
    print(row("Jain fairness index (tiers)", lambda x: _fmt(x["jain_fairness_by_tier"], decimals=3)))

    # vulnerable
    print(row("Elderly: service rate",           lambda x: _fmt(x["elderly_involved"]["service_rate"], pct=True)))
    print(row("Elderly: on-time rate",           lambda x: _fmt(x["elderly_involved"]["on_time_rate"], pct=True)))
    print(row("Elderly: avg latency (min)",      lambda x: _fmt(x["elderly_involved"]["avg_latency_min"])))
    print(row("Vulnerable comm: service rate",   lambda x: _fmt(x["vulnerable_community"]["service_rate"], pct=True)))
    print(row("Vulnerable comm: on-time rate",   lambda x: _fmt(x["vulnerable_community"]["on_time_rate"], pct=True)))
    print(row("Vulnerable comm: avg latency (min)", lambda x: _fmt(x["vulnerable_community"]["avg_latency_min"])))

    # efficiency
    print(row("Total distance (km)",         lambda x: _fmt(x["window_aggregate"]["total_distance_km"], decimals=0)))

    print()

    # priority-by-tier latency breakdown
    print("### Avg Delivery Latency by True Priority (min)\n")
    tier_header = " | ".join(f"**{n}**" for n in methods)
    print(f"| {'Priority':<10} | {tier_header} |")
    print(f"| {'---':<10} | {' | '.join('---' for _ in methods)} |")
    for p in range(1, 5):
        vals = " | ".join(_fmt(m[n]["by_priority"][p]["avg_latency_min"]) for n in methods)
        print(f"| P{p:<9} | {vals} |")
    print()


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate formal M0/M1 comparison.")
    parser.add_argument("--seed", default="4111")
    parser.add_argument("--split", default="norm_eval")
    parser.add_argument("--output", default="evals/results/formal_comparison_seed{seed}.json")
    args = parser.parse_args()
    seed = args.seed

    dataset_dir = DATA_ROOT / "test" / "test_seeds" / args.split / f"seed_{seed}"
    manifest_path = dataset_dir / "events_manifest.jsonl"
    eval_dir = DATA_ROOT / "eval_runs"

    print(f"Loading ground-truth manifest: {manifest_path}")
    manifest = load_events_manifest(manifest_path)
    print(f"  {len(manifest)} events loaded\n")

    all_methods: Dict[str, dict] = {}
    for label, dir_tpl in METHODS:
        dir_name = dir_tpl.format(seed=seed)
        run_dir = find_latest_run(eval_dir / dir_name)
        if run_dir is None:
            print(f"[SKIP] {label}: no run found in {eval_dir / dir_name}")
            continue
        print(f"[{label}] {run_dir.name}")
        results = load_workflow_results(run_dir)
        rows = build_rows(results, manifest)
        metrics = method_metrics(rows)
        metrics["window_aggregate"] = window_aggregate(results)
        metrics["run_dir"] = str(run_dir)
        metrics["n_demands"] = len(rows)
        all_methods[label] = metrics
        print(f"  demands={len(rows)}, windows={len(results)}, "
              f"service_rate={metrics['overall']['service_rate']}, "
              f"on_time_rate={metrics['overall']['on_time_rate']}")

    summary = {"seed": seed, "split": args.split, "methods": all_methods}

    print_markdown_table(summary)

    output_path = Path(args.output.format(seed=seed))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved → {output_path}")


if __name__ == "__main__":
    main()
