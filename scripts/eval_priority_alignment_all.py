"""Contribution 3 — LLM3 priority alignment evaluation.

Compares M0c (rule-based), M1_pre (qwen3-base), M1_ft (qwen3-finetuned)
priority assignments against extraction_observable_priority ground truth.

Usage:
  python scripts/eval_priority_alignment_all.py --seed 4111
  python scripts/eval_priority_alignment_all.py --seed 4111 --all-seeds
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy.stats import kendalltau, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

ROOT = Path(__file__).resolve().parent.parent

# ── method → (weight_configs dir template, extracted_demands path template) ──
# Priority: use formal runs (slots 0-47) first, fall back to full norm_eval runs
METHOD_SPECS: List[Tuple[str, str, str]] = [
    # (label, weight_configs dir template, extracted_demands template)
    ("M0c",    "data/eval_runs/formal_m0c_seed{seed}/run_*/weight_configs",
               "data/eval_runs/formal_m0c_seed{seed}/run_*/extracted_demands.json"),
    ("M1_pre", "data/eval_runs/formal_m1_pre_seed{seed}/run_*/weight_configs",
               "data/eval_runs/formal_m1_pre_seed{seed}/run_*/extracted_demands.json"),
    ("M1_ft",  "data/eval_runs/formal_m1_ft_seed{seed}/run_*/weight_configs",
               "data/eval_runs/formal_m1_ft_seed{seed}/run_*/extracted_demands.json"),
    ("M1_gemini", "data/eval_runs/formal_m1_gemini_seed{seed}/run_*/weight_configs",
               "data/eval_runs/formal_m1_gemini_seed{seed}/run_*/extracted_demands.json"),
]

# For multi-seed alignment, also try the full 576-window runs
MULTISEED_SPECS: List[Tuple[str, str, str]] = [
    ("M1_pre", "data/eval_runs/m1_pre_qwen3base_norm_eval_seed{seed}/run_*/weight_configs",
               "data/eval_runs/m1_pre_qwen3base_norm_eval_seed{seed}/run_*/extracted_demands.json"),
]


def _glob_first(pattern: str) -> Optional[Path]:
    results = sorted(ROOT.glob(pattern), reverse=True)
    return results[0] if results else None


def load_weight_configs(wc_dir: Path) -> Dict[str, Dict[str, int]]:
    """Returns {time_window: {demand_id: priority}}."""
    result: Dict[str, Dict[str, int]] = {}
    for path in sorted(wc_dir.glob("weight_config_window*.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        tw = cfg.get("time_window", "")
        result[tw] = {
            d["demand_id"]: int(d.get("priority", 4))
            for d in cfg.get("demand_configs", [])
            if d.get("demand_id")
        }
    return result


def load_ground_truth(ed_path: Path) -> Dict[str, Dict[str, int]]:
    """Returns {time_window: {demand_id: extraction_observable_priority}}."""
    data = json.loads(ed_path.read_text(encoding="utf-8"))
    result: Dict[str, Dict[str, int]] = {}
    for window in data:
        tw = window.get("time_window", "")
        result[tw] = {}
        for dem in window.get("demands", []):
            did = dem.get("demand_id", "")
            labels = dem.get("labels") or {}
            eop = labels.get("extraction_observable_priority") or dem.get("extraction_observable_priority")
            if did and eop is not None:
                result[tw][did] = int(eop)
    return result


def align(wc: Dict[str, Dict[str, int]], gt: Dict[str, Dict[str, int]]) -> Tuple[List[int], List[int], List[dict]]:
    y_true, y_pred, items = [], [], []
    for tw, preds in wc.items():
        truths = gt.get(tw, {})
        for did, pred in preds.items():
            if did in truths:
                y_true.append(truths[did])
                y_pred.append(pred)
                items.append({"time_window": tw, "demand_id": did,
                               "true": truths[did], "pred": pred})
    return y_true, y_pred, items


def pairwise_accuracy(items: List[dict]) -> Optional[float]:
    by_window: Dict[str, List[dict]] = {}
    for it in items:
        by_window.setdefault(it["time_window"], []).append(it)
    correct = total = 0
    for window_items in by_window.values():
        n = len(window_items)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = window_items[i], window_items[j]
                if a["true"] == b["true"]:
                    continue
                total += 1
                # correct if ordering matches
                if (a["true"] < b["true"]) == (a["pred"] < b["pred"]):
                    correct += 1
    return round(correct / total, 4) if total > 0 else None


def compute_metrics(y_true: List[int], y_pred: List[int], items: List[dict]) -> dict:
    if not y_true:
        return {"n": 0}
    n = len(y_true)
    result: dict = {"n": n}

    if HAS_SKLEARN:
        labels = sorted(set(y_true) | set(y_pred))
        result["accuracy"]    = round(accuracy_score(y_true, y_pred), 4)
        result["macro_f1"]    = round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4)
        result["weighted_f1"] = round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 4)
        prec, rec, f1v, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0)
        result["per_priority"] = {
            str(lab): {"precision": round(float(prec[i]), 4),
                       "recall":    round(float(rec[i]), 4),
                       "f1":        round(float(f1v[i]), 4),
                       "support":   int(sup[i])}
            for i, lab in enumerate(labels)
        }
        # urgent (p1+p2) binary
        yt_bin = [1 if v <= 2 else 0 for v in y_true]
        yp_bin = [1 if v <= 2 else 0 for v in y_pred]
        result["urgent_recall"]    = round(float(precision_recall_fscore_support(
            yt_bin, yp_bin, average="binary", zero_division=0)[1]), 4)
        result["urgent_precision"] = round(float(precision_recall_fscore_support(
            yt_bin, yp_bin, average="binary", zero_division=0)[0]), 4)
        result["urgent_f1"]        = round(float(f1_score(yt_bin, yp_bin, average="binary", zero_division=0)), 4)
        # p1 binary
        yt_p1 = [1 if v == 1 else 0 for v in y_true]
        yp_p1 = [1 if v == 1 else 0 for v in y_pred]
        result["p1_recall"]    = round(float(precision_recall_fscore_support(
            yt_p1, yp_p1, average="binary", zero_division=0)[1]), 4)
        result["p1_precision"] = round(float(precision_recall_fscore_support(
            yt_p1, yp_p1, average="binary", zero_division=0)[0]), 4)
        result["p1_f1"]        = round(float(f1_score(yt_p1, yp_p1, average="binary", zero_division=0)), 4)

    if HAS_SCIPY and len(set(y_true)) > 1:
        result["spearman_r"]   = round(float(spearmanr(y_true, y_pred).correlation), 4)
        result["kendall_tau"]  = round(float(kendalltau(y_true, y_pred).statistic), 4)

    result["pairwise_accuracy"] = pairwise_accuracy(items)
    return result


def _fmt(v, pct=False) -> str:
    if v is None: return "—"
    if pct: return f"{v*100:.1f}%"
    return f"{v:.4f}"


def print_table(results: Dict[str, dict]) -> None:
    methods = list(results.keys())
    header = " | ".join(f"**{m}**" for m in methods)
    sep    = " | ".join("---" for _ in methods)
    def row(label, fn):
        return f"| {label:<35} | " + " | ".join(fn(results[m]) for m in methods) + " |"

    print("\n## Contribution 3 — LLM3 Priority Alignment\n")
    print(f"| {'Metric':<35} | {header} |")
    print(f"| {'---':<35} | {sep} |")
    print(row("n aligned demands",           lambda x: str(x.get("n", "—"))))
    print(row("**Accuracy**",                lambda x: _fmt(x.get("accuracy"))))
    print(row("Macro-F1",                    lambda x: _fmt(x.get("macro_f1"))))
    print(row("Weighted-F1",                 lambda x: _fmt(x.get("weighted_f1"))))
    print(row("**P1 Recall**",               lambda x: _fmt(x.get("p1_recall"))))
    print(row("P1 Precision",                lambda x: _fmt(x.get("p1_precision"))))
    print(row("P1 F1",                       lambda x: _fmt(x.get("p1_f1"))))
    print(row("**Urgent (p1+p2) Recall**",   lambda x: _fmt(x.get("urgent_recall"))))
    print(row("Urgent Precision",            lambda x: _fmt(x.get("urgent_precision"))))
    print(row("Urgent F1",                   lambda x: _fmt(x.get("urgent_f1"))))
    print(row("**Spearman r**",              lambda x: _fmt(x.get("spearman_r"))))
    print(row("Kendall τ",                   lambda x: _fmt(x.get("kendall_tau"))))
    print(row("**Pairwise accuracy**",       lambda x: _fmt(x.get("pairwise_accuracy"))))
    print()


def run_seed(seed: str, specs: list) -> Dict[str, dict]:
    results = {}
    for label, wc_tpl, ed_tpl in specs:
        wc_dir = _glob_first(wc_tpl.format(seed=seed))
        ed_path = _glob_first(ed_tpl.format(seed=seed))
        if wc_dir is None or ed_path is None:
            print(f"  [{label}] seed={seed}: missing data (wc={wc_dir}, ed={ed_path})")
            continue
        wc  = load_weight_configs(wc_dir)
        gt  = load_ground_truth(ed_path)
        y_true, y_pred, items = align(wc, gt)
        print(f"  [{label}] seed={seed}: {len(y_true)} aligned pairs from {len(wc)} windows")
        results[label] = compute_metrics(y_true, y_pred, items)
        results[label]["seed"] = seed
        results[label]["wc_dir"] = str(wc_dir)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="4111")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Also compute multi-seed M1_pre alignment using full norm_eval runs")
    parser.add_argument("--output", default="evals/results/priority_alignment_seed{seed}.json")
    args = parser.parse_args()

    print(f"\n=== Formal run alignment (seed={args.seed}, slots 0-47) ===")
    formal_results = run_seed(args.seed, METHOD_SPECS)
    print_table(formal_results)

    all_output = {"formal": formal_results}

    if args.all_seeds:
        print("=== M1_pre multi-seed alignment (full 576-window runs) ===")
        multiseed: Dict[str, dict] = {}
        for seed in ["4111", "4112", "5111"]:
            res = run_seed(seed, MULTISEED_SPECS)
            for label, metrics in res.items():
                multiseed.setdefault(label, {})[seed] = metrics
        all_output["multiseed_m1_pre"] = multiseed

        # aggregate
        for label, by_seed in multiseed.items():
            accs = [v["accuracy"] for v in by_seed.values() if v.get("accuracy") is not None]
            taus = [v["kendall_tau"] for v in by_seed.values() if v.get("kendall_tau") is not None]
            p1rs = [v["p1_recall"] for v in by_seed.values() if v.get("p1_recall") is not None]
            print(f"\n{label} multi-seed summary:")
            if accs:
                print(f"  accuracy:     mean={sum(accs)/len(accs):.4f}  seeds={[round(a,4) for a in accs]}")
            if taus:
                print(f"  kendall_tau:  mean={sum(taus)/len(taus):.4f}  seeds={[round(t,4) for t in taus]}")
            if p1rs:
                print(f"  p1_recall:    mean={sum(p1rs)/len(p1rs):.4f}  seeds={[round(r,4) for r in p1rs]}")

    out = Path(args.output.format(seed=args.seed))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
