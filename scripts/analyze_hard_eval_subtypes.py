from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

try:
    from scipy.stats import kendalltau, spearmanr
except Exception:  # pragma: no cover - optional fallback
    kendalltau = None
    spearmanr = None


SUBSET_ORDER = (
    "counterfactual",
    "surface_contradiction",
    "near_tie",
    "mixed_priority",
    "other",
)

ALIGNMENT_METRICS = (
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
)


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_hard_window(time_window: str) -> str:
    label = str(time_window or "")
    if "surface_contradiction" in label:
        return "surface_contradiction"
    if "near_tie" in label:
        return "near_tie"
    if "counterfactual" in label:
        return "counterfactual"
    if "mixed_priority" in label:
        return "mixed_priority"
    return "other"


def _float_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_rank_metric(metric_fn, y_true: List[int], y_pred: List[int]) -> Optional[float]:
    if metric_fn is None or len(y_true) < 2:
        return None
    if len(set(y_true)) < 2 and len(set(y_pred)) < 2:
        return 1.0
    value = metric_fn(y_true, y_pred)
    if isinstance(value, tuple):
        value = value[0]
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


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
        "support": support,
        "predicted_positive": predicted_positive,
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
            str(item.get("demand_id", "")),
        ),
    )
    pred_sorted = sorted(
        items,
        key=lambda item: (
            int(item["pred_priority"]),
            int(item.get("window_rank") or 10**9),
            str(item.get("time_window", "")),
            str(item.get("event_id", "")),
            str(item.get("demand_id", "")),
        ),
    )
    true_top = {
        str(item.get("event_id") or item.get("demand_id"))
        for item in true_sorted[:true_k]
    }
    pred_top = {
        str(item.get("event_id") or item.get("demand_id"))
        for item in pred_sorted[:true_k]
    }
    hits = len(true_top & pred_top)
    return {"k": true_k, "hit_rate": hits / true_k, "hits": hits}


def _empty_metrics() -> Dict[str, object]:
    return {
        "n_aligned_demands": 0,
        "accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "spearman": None,
        "kendall_tau": None,
        "top_k_hit_rate": None,
        "priority_1_recall": None,
        "priority_1_f1": None,
        "urgent_recall": None,
        "urgent_f1": None,
    }


def _metrics_from_items(items: List[Dict[str, object]], urgent_threshold: int) -> Dict[str, object]:
    if not items:
        return _empty_metrics()

    y_true = [int(item["true_priority"]) for item in items]
    y_pred = [int(item["pred_priority"]) for item in items]
    labels = sorted(set(y_true) | set(y_pred))
    priority_1 = _binary_metrics(y_true, y_pred, lambda value: value == 1)
    urgent = _binary_metrics(y_true, y_pred, lambda value: value <= urgent_threshold)
    topk = _top_k_hit_rate(items, urgent_threshold)
    return {
        "n_aligned_demands": len(items),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted")),
        "spearman": _safe_rank_metric(spearmanr, y_true, y_pred),
        "kendall_tau": _safe_rank_metric(kendalltau, y_true, y_pred),
        "top_k_hit_rate": _float_or_none(topk.get("hit_rate")),
        "priority_1_recall": _float_or_none(priority_1.get("recall")),
        "priority_1_f1": _float_or_none(priority_1.get("f1")),
        "urgent_recall": _float_or_none(urgent.get("recall")),
        "urgent_f1": _float_or_none(urgent.get("f1")),
    }


def _delta(post_metrics: Dict[str, object], pre_metrics: Dict[str, object]) -> Dict[str, Optional[float]]:
    return {
        key: (
            None
            if _float_or_none(post_metrics.get(key)) is None or _float_or_none(pre_metrics.get(key)) is None
            else round(float(post_metrics[key]) - float(pre_metrics[key]), 6)
        )
        for key in ALIGNMENT_METRICS
    }


def _score_metrics(metrics: Dict[str, Optional[float]]) -> Dict[str, object]:
    better: List[str] = []
    worse: List[str] = []
    unchanged: List[str] = []
    available: Dict[str, float] = {}

    for metric_name, metric_value in metrics.items():
        if metric_value is None:
            continue
        available[metric_name] = metric_value
        if abs(metric_value) < 1e-9:
            unchanged.append(metric_name)
        elif metric_value > 0:
            better.append(metric_name)
        else:
            worse.append(metric_name)

    if not available:
        overall = "inconclusive"
    elif better and not worse:
        overall = "post_better"
    elif worse and not better:
        overall = "pre_better"
    elif len(better) >= len(worse) + 2:
        overall = "post_better"
    elif len(worse) >= len(better) + 2:
        overall = "pre_better"
    else:
        overall = "mixed"

    return {
        "overall": overall,
        "better_metrics_for_post": sorted(better),
        "worse_metrics_for_post": sorted(worse),
        "unchanged_metrics": sorted(unchanged),
        "available_metrics": available,
    }


def _group_items(payload: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
    grouped = {subset: [] for subset in SUBSET_ORDER}
    for item in payload.get("per_item", []) or []:
        if not isinstance(item, dict):
            continue
        subset = classify_hard_window(str(item.get("time_window", "")))
        grouped.setdefault(subset, []).append(item)
    return grouped


def analyze_run(run_dir: Path, *, urgent_threshold: int = 2) -> Dict[str, object]:
    pre_path = run_dir / "evals" / "pre_alignment.json"
    post_path = run_dir / "evals" / "post_alignment.json"
    summary_path = run_dir / "evals" / "summary.json"

    pre_payload = _load_json(pre_path)
    post_payload = _load_json(post_path)
    summary_payload = _load_json(summary_path) if summary_path.exists() else {}

    pre_grouped = _group_items(pre_payload)
    post_grouped = _group_items(post_payload)
    subsets: Dict[str, object] = {}

    for subset in SUBSET_ORDER:
        pre_items = pre_grouped.get(subset, [])
        post_items = post_grouped.get(subset, [])
        pre_metrics = _metrics_from_items(pre_items, urgent_threshold)
        post_metrics = _metrics_from_items(post_items, urgent_threshold)
        delta = _delta(post_metrics, pre_metrics)
        verdict = _score_metrics(delta)
        subsets[subset] = {
            "pre": pre_metrics,
            "post": post_metrics,
            "delta_post_minus_pre": delta,
            "verdict": verdict,
        }

    analysis = {
        "run_dir": str(run_dir),
        "headline": summary_payload.get("headline"),
        "truth_source": summary_payload.get("truth_source"),
        "subsets": subsets,
    }
    output_path = run_dir / "evals" / "hard_subtype_breakdown.json"
    output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    return analysis


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    if (root / "evals" / "pre_alignment.json").is_file() and (root / "evals" / "post_alignment.json").is_file():
        yield root
        return
    for summary_path in sorted(root.rglob("evals/summary.json")):
        run_dir = summary_path.parent.parent
        if (run_dir / "evals" / "pre_alignment.json").is_file() and (run_dir / "evals" / "post_alignment.json").is_file():
            yield run_dir


def _row_from_analysis(analysis: Dict[str, object], subset: str) -> Dict[str, object]:
    subset_payload = ((analysis.get("subsets") or {}).get(subset) or {})
    pre = subset_payload.get("pre") or {}
    post = subset_payload.get("post") or {}
    delta = subset_payload.get("delta_post_minus_pre") or {}
    verdict = (subset_payload.get("verdict") or {}).get("overall")
    row: Dict[str, object] = {
        "run_name": Path(str(analysis.get("run_dir", ""))).name,
        "run_dir": analysis.get("run_dir"),
        "subset": subset,
        "overall": verdict,
        "truth_source": analysis.get("truth_source"),
        "n_aligned_demands": post.get("n_aligned_demands"),
    }
    for key in ALIGNMENT_METRICS:
        row[f"pre_{key}"] = pre.get(key)
        row[f"post_{key}"] = post.get(key)
        row[f"delta_{key}"] = delta.get(key)
    return row


def build_rows(analyses: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for analysis in analyses:
        for subset in SUBSET_ORDER:
            rows.append(_row_from_analysis(analysis, subset))
    rows.sort(
        key=lambda item: (
            str(item.get("subset", "")),
            float(item.get("delta_priority_1_recall") or 0.0),
            float(item.get("delta_top_k_hit_rate") or 0.0),
            float(item.get("delta_urgent_f1") or 0.0),
            float(item.get("delta_macro_f1") or 0.0),
            float(item.get("delta_accuracy") or 0.0),
        ),
        reverse=True,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc analyze hard-eval results by subtype using existing pre/post alignment JSONs.",
    )
    parser.add_argument("root", type=Path, help="A single eval run dir, or a parent directory containing multiple eval runs.")
    parser.add_argument("--urgent-threshold", type=int, default=2)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    analyses = [analyze_run(run_dir, urgent_threshold=args.urgent_threshold) for run_dir in _iter_run_dirs(root)]
    if not analyses:
        raise SystemExit(f"No evaluation runs with pre_alignment.json and post_alignment.json found under {root}")

    rows = build_rows(analyses)
    output_jsonl = args.output_jsonl or (root / "hard_subtype_leaderboard.jsonl")
    output_csv = args.output_csv or (root / "hard_subtype_leaderboard.csv")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rows:
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps({
        "root": str(root),
        "n_runs": len(analyses),
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
