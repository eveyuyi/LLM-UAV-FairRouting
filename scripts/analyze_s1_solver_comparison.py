"""S1: Solver-level comparison across priority modes (§5.2).

Reads workflow_results.json from run directories for each method,
joins with ground-truth priorities from events_manifest.jsonl,
and outputs per-GT-priority-group service metrics plus aggregate
fairness scores.

Usage:
    python scripts/analyze_s1_solver_comparison.py \
        --methods \
            m0a:data/eval_runs/m0a_norm_eval_seed4111/run_* \
            m0b:data/eval_runs/m0b_norm_eval_seed4111/run_* \
            m0c:data/eval_runs/m0c_norm_eval_seed4111/run_* \
            m1:data/eval_runs/m1_norm_eval_seed4111/run_* \
        --ground-truth data/test/test_seeds/norm_eval/seed_*/events_manifest.jsonl \
        --output data/evals/s1_solver_comparison.json

Multiple seeds: pass shell-expanded globs for both --methods and --ground-truth.
"""
import argparse
import json
import statistics
from pathlib import Path

# Priority weight for "priority_weighted_service_gain":
# P=1 (most urgent) counts 4x, P=4 (lowest) counts 1x
_PRIORITY_WEIGHTS = {1: 4, 2: 3, 3: 2, 4: 1}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _build_gt_map(gt_paths: list[Path]) -> dict[str, int]:
    gt: dict[str, int] = {}
    for p in gt_paths:
        if not p.exists():
            print(f"[warn] ground-truth file not found: {p}")
            continue
        for item in _load_jsonl(p):
            for demand in item.get("demands", []) or [item]:
                did = demand.get("demand_id") or demand.get("event_id")
                pri = demand.get("priority") or demand.get("latent_priority")
                if did and pri:
                    gt[str(did)] = int(pri)
    return gt


def _load_workflow_results(results_path: Path) -> list[dict]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("solutions", [])


def _collect_records(run_dir: Path, gt_map: dict[str, int]) -> list[dict]:
    results_path = run_dir / "workflow_results.json"
    if not results_path.exists():
        return []
    records = []
    for entry in _load_workflow_results(results_path):
        for d in entry.get("per_demand_results", []):
            did = str(d.get("demand_id", ""))
            if did not in gt_map:
                continue
            records.append({
                "demand_id": did,
                "gt_priority": gt_map[did],
                "assigned_priority": d.get("priority"),
                "is_served": bool(d.get("is_served", False)),
                "delivery_latency_s": d.get("delivery_latency_s"),
            })
    return records


def _compute_metrics(records: list[dict]) -> dict:
    if not records:
        return {"n_total": 0}

    total = len(records)
    served = sum(1 for r in records if r["is_served"])

    # Per GT-priority group service rates
    by_pri: dict[int, list[bool]] = {}
    for r in records:
        by_pri.setdefault(r["gt_priority"], []).append(r["is_served"])

    per_group: dict[str, object] = {}
    sr_values: list[float] = []
    for p in sorted(by_pri):
        g = by_pri[p]
        sr = round(sum(g) / len(g), 4) if g else 0.0
        per_group[f"p{p}_service_rate"] = sr
        per_group[f"p{p}_n"] = len(g)
        sr_values.append(sr)

    # Priority-weighted service gain (higher GT priority → higher weight)
    weight_total = sum(_PRIORITY_WEIGHTS.get(r["gt_priority"], 1) for r in records)
    weight_served = sum(
        _PRIORITY_WEIGHTS.get(r["gt_priority"], 1) for r in records if r["is_served"]
    )
    pw_gain = round(weight_served / weight_total, 4) if weight_total else 0.0

    # Jain's fairness index over per-group service rates
    n_groups = len(sr_values)
    if n_groups >= 2 and sum(x ** 2 for x in sr_values) > 0:
        jain = round(
            sum(sr_values) ** 2 / (n_groups * sum(x ** 2 for x in sr_values)), 4
        )
    else:
        jain = 1.0

    # Average delivery latency for served demands (minutes)
    latencies_s = [
        r["delivery_latency_s"] for r in records
        if r["is_served"] and r["delivery_latency_s"] is not None
    ]
    avg_latency_min = (
        round(statistics.mean(latencies_s) / 60, 2) if latencies_s else None
    )

    # Priority alignment: fraction where assigned == gt (only meaningful for rule/llm modes)
    aligned = [r for r in records if r["assigned_priority"] is not None]
    if aligned:
        priority_alignment = round(
            sum(1 for r in aligned if r["assigned_priority"] == r["gt_priority"]) / len(aligned),
            4,
        )
    else:
        priority_alignment = None

    return {
        "n_total": total,
        "n_served": served,
        "service_rate": round(served / total, 4),
        "priority_weighted_service_gain": pw_gain,
        "jain_fairness_index": jain,
        "avg_delivery_latency_min": avg_latency_min,
        "priority_alignment": priority_alignment,
        **per_group,
    }


def _fmt(val: object, width: int = 10) -> str:
    if val is None:
        return f"{'N/A':>{width}}"
    if isinstance(val, float):
        return f"{val:>{width}.4f}"
    return f"{str(val):>{width}}"


def main():
    parser = argparse.ArgumentParser(description="S1: Solver comparison by GT priority group")
    parser.add_argument(
        "--methods", nargs="+", required=True,
        help="name:path pairs. Path is a single run dir or shell-expanded glob list. "
             "e.g. m0a:data/eval_runs/m0a_seed4111/run_0 m0a:data/eval_runs/m0a_seed4112/run_0",
    )
    parser.add_argument(
        "--ground-truth", nargs="+", required=True,
        help="Path(s) to events_manifest.jsonl (shell-expanded by caller)",
    )
    parser.add_argument("--output", default="data/evals/s1_solver_comparison.json")
    args = parser.parse_args()

    gt_map = _build_gt_map([Path(p) for p in args.ground_truth])
    print(f"Loaded {len(gt_map)} ground-truth demand priorities")
    if not gt_map:
        print("[error] No ground-truth priorities loaded. Check --ground-truth paths.")
        return

    # Group run dirs by method name (same name can appear multiple times for multi-seed)
    method_dirs: dict[str, list[Path]] = {}
    for spec in args.methods:
        if ":" not in spec:
            print(f"[warn] skipping invalid spec (expected name:path): {spec}")
            continue
        name, path_str = spec.split(":", 1)
        rd = Path(path_str)
        if not rd.is_dir():
            print(f"[warn] {name}: not a directory: {rd}")
            continue
        method_dirs.setdefault(name, []).append(rd)

    results: dict[str, dict] = {}
    for method_name, run_dirs in method_dirs.items():
        all_records: list[dict] = []
        for rd in sorted(run_dirs):
            recs = _collect_records(rd, gt_map)
            all_records.extend(recs)
            print(f"  [{method_name}] {rd}: {len(recs)} demand records")

        if not all_records:
            print(f"  [{method_name}] WARNING: no per_demand_results found")
        metrics = _compute_metrics(all_records)
        results[method_name] = {
            "run_dirs": [str(d) for d in sorted(run_dirs)],
            "metrics": metrics,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSolver comparison saved to {out_path}")

    # --- terminal table ---
    methods = list(results.keys())
    if not methods:
        return

    key_metrics = [
        "n_total",
        "n_served",
        "service_rate",
        "priority_weighted_service_gain",
        "jain_fairness_index",
        "avg_delivery_latency_min",
        "priority_alignment",
        "p1_service_rate",
        "p1_n",
        "p2_service_rate",
        "p2_n",
        "p3_service_rate",
        "p3_n",
        "p4_service_rate",
        "p4_n",
    ]
    col_w = 12
    header = f"{'Metric':<35}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(f"\n{'=== S1 Solver Comparison (§5.2) '+'='*max(0, len(header)-33)}")
    print(header)
    print("-" * len(header))
    for k in key_metrics:
        row = f"{k:<35}"
        for m in methods:
            val = results[m]["metrics"].get(k)
            row += _fmt(val, col_w)
        print(row)


if __name__ == "__main__":
    main()
