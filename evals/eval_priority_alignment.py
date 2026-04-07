"""Evaluate alignment between ground-truth priorities and Module 3 priority outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llm4fairrouting.data.event_data import load_ground_truth_event_index
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

try:
    from scipy.stats import kendalltau, spearmanr
except Exception:  # pragma: no cover - optional fallback
    kendalltau = None
    spearmanr = None


def _load_weight_configs(source_path: str) -> Dict[str, Dict]:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Weight config source not found: {source_path}")

    configs: Dict[str, Dict] = {}
    if source.is_dir():
        for path in sorted(source.glob("*.json")):
            with open(path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            time_window = config.get("time_window")
            if time_window:
                configs[time_window] = config
        return configs

    with open(source, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = [payload]
    for config in payload:
        time_window = config.get("time_window")
        if time_window:
            configs[time_window] = config
    return configs


def _load_extracted_demands(path: str) -> Dict[str, Dict[str, Dict]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        with open(source, "r", encoding="utf-8") as handle:
            windows = [json.loads(line) for line in handle if line.strip()]
    else:
        with open(source, "r", encoding="utf-8") as handle:
            windows = json.load(handle)
    by_window: Dict[str, Dict[str, Dict]] = {}
    for window in windows:
        time_window = str(window.get("time_window", ""))
        by_window[time_window] = {
            str(demand.get("demand_id", "")): demand
            for demand in window.get("demands", [])
            if demand.get("demand_id")
        }
    return by_window


def _load_dialogue_metadata(path: str) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    by_dialogue: Dict[str, Dict[str, object]] = {}
    by_event: Dict[str, Dict[str, object]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            dialogue = json.loads(line)
            dialogue_id = str(dialogue.get("dialogue_id", "")).strip()
            metadata = dialogue.get("metadata", {})
            event_id = str(metadata.get("event_id", "")).strip()
            record = {
                "dialogue_id": dialogue_id,
                "event_id": event_id,
                "delivery_deadline_minutes": metadata.get("delivery_deadline_minutes"),
                "time_slot": metadata.get("time_slot"),
                "timestamp": dialogue.get("timestamp"),
            }
            if dialogue_id:
                by_dialogue[dialogue_id] = record
            if event_id:
                by_event[event_id] = record
    return by_dialogue, by_event


def _load_ground_truth_priorities(path: str) -> Dict[str, int]:
    return {
        event_id: int(payload["priority"])
        for event_id, payload in load_ground_truth_event_index(path).items()
    }


def _observable_priority_from_demand(demand: Optional[Dict]) -> Optional[int]:
    if not demand:
        return None
    labels = demand.get("labels", {}) or {}
    value = labels.get("extraction_observable_priority")
    if value is None:
        value = demand.get("extraction_observable_priority")
    if value is None:
        return None
    try:
        return int(value)
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
    precision, recall, f1, _ = precision_recall_fscore_support(
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
        "f1": float(f1[0]),
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
        ),
    )
    pred_sorted = sorted(
        items,
        key=lambda item: (
            int(item["pred_priority"]),
            int(item.get("window_rank") or 10**9),
            str(item.get("time_window", "")),
            str(item.get("event_id", "")),
        ),
    )
    true_top = {str(item["event_id"]) for item in true_sorted[:true_k]}
    pred_top = {str(item["event_id"]) for item in pred_sorted[:true_k]}
    hits = len(true_top & pred_top)
    return {"k": true_k, "hit_rate": hits / true_k, "hits": hits}


def evaluate_priority_alignment(
    *,
    weights_path: str,
    demands_path: str,
    dialogues_path: str,
    ground_truth_path: str,
    urgent_threshold: int = 2,
    truth_source: str = "auto",
    truth_demands_path: Optional[str] = None,
) -> Dict[str, object]:
    weight_configs = _load_weight_configs(weights_path)
    extracted_demands = _load_extracted_demands(demands_path)
    truth_demands = _load_extracted_demands(truth_demands_path) if truth_demands_path else {}
    dialogue_lookup, _event_lookup = _load_dialogue_metadata(dialogues_path)
    ground_truth = _load_ground_truth_priorities(ground_truth_path)
    if truth_source not in {"auto", "run_extracted", "fixed_demands", "ground_truth_manifest"}:
        raise ValueError(f"Unsupported truth_source: {truth_source}")
    if truth_source == "fixed_demands" and not truth_demands_path:
        raise ValueError("truth_demands_path is required when truth_source='fixed_demands'")

    y_true: List[int] = []
    y_pred: List[int] = []
    per_item: List[Dict[str, object]] = []

    for time_window, config in weight_configs.items():
        demand_lookup = extracted_demands.get(time_window, {})
        for demand_config in config.get("demand_configs", []):
            demand_id = str(demand_config.get("demand_id", ""))
            demand = demand_lookup.get(demand_id)
            if demand is None:
                continue
            dialogue_id = str(demand.get("source_dialogue_id", "")).strip()
            source_event_id = str(demand.get("source_event_id", "")).strip()
            event_id = source_event_id or str((dialogue_lookup.get(dialogue_id) or {}).get("event_id", "")).strip()
            true_priority = None
            if truth_source == "run_extracted":
                true_priority = _observable_priority_from_demand(demand)
            elif truth_source == "fixed_demands":
                truth_demand = (truth_demands.get(time_window) or {}).get(demand_id)
                true_priority = _observable_priority_from_demand(truth_demand)
            elif truth_source == "auto":
                true_priority = _observable_priority_from_demand(demand)

            if true_priority is None:
                if not event_id or event_id not in ground_truth:
                    continue
                true_priority = int(ground_truth[event_id])
            if not event_id:
                continue
            true_priority = int(true_priority)
            pred_priority = int(demand_config.get("priority", 4))
            y_true.append(true_priority)
            y_pred.append(pred_priority)
            per_item.append(
                {
                    "time_window": time_window,
                    "demand_id": demand_id,
                    "dialogue_id": dialogue_id,
                    "event_id": event_id,
                    "true_priority": true_priority,
                    "pred_priority": pred_priority,
                    "window_rank": demand_config.get("window_rank"),
                    "reasoning": demand_config.get("reasoning", ""),
                }
            )

    labels = sorted(set(y_true) | set(y_pred))
    results: Dict[str, object] = {
        "n_aligned_demands": len(y_true),
        "labels": labels,
        "urgent_threshold": urgent_threshold,
        "truth_source": truth_source,
        "truth_demands_path": truth_demands_path,
        "per_item": per_item,
    }
    if not y_true:
        results.update(
            {
                "accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "confusion_matrix": [],
                "per_priority_metrics": {},
                "spearman": None,
                "kendall_tau": None,
                "priority_1_metrics": {},
                "urgent_metrics": {},
                "top_k_hit_rate": {"k": 0, "hit_rate": None, "hits": 0},
            }
        )
        return results

    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_priority_metrics = {
        str(label): {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1_values[idx]),
            "support": int(support[idx]),
        }
        for idx, label in enumerate(labels)
    }

    results.update(
        {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted")),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "per_priority_metrics": per_priority_metrics,
            "spearman": _safe_rank_metric(spearmanr, y_true, y_pred),
            "kendall_tau": _safe_rank_metric(kendalltau, y_true, y_pred),
            "priority_1_metrics": _binary_metrics(y_true, y_pred, lambda value: value == 1),
            "urgent_metrics": _binary_metrics(y_true, y_pred, lambda value: value <= urgent_threshold),
            "top_k_hit_rate": _top_k_hit_rate(per_item, urgent_threshold),
        }
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate alignment between ground-truth priorities and LLM/rule priorities.")
    parser.add_argument("--weights", required=True, help="Weight configs JSON file or directory")
    parser.add_argument("--demands", required=True, help="Extracted demands JSON file")
    parser.add_argument("--dialogues", required=True, help="Dialogue JSONL file used to build extracted demands")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth rich event manifest JSONL/JSON")
    parser.add_argument("--urgent-threshold", type=int, default=2, help="Priorities <= threshold are treated as urgent")
    parser.add_argument(
        "--truth-source",
        choices=("auto", "run_extracted", "fixed_demands", "ground_truth_manifest"),
        default="auto",
        help="Which source to use for true priorities",
    )
    parser.add_argument(
        "--truth-demands",
        help="Optional fixed demands JSON/JSONL used as the truth source when --truth-source=fixed_demands",
    )
    parser.add_argument("--output", default="evals/results/priority_alignment.json", help="Output JSON path")
    args = parser.parse_args()

    results = evaluate_priority_alignment(
        weights_path=args.weights,
        demands_path=args.demands,
        dialogues_path=args.dialogues,
        ground_truth_path=args.ground_truth,
        urgent_threshold=args.urgent_threshold,
        truth_source=args.truth_source,
        truth_demands_path=args.truth_demands,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"Priority alignment results saved to {output_path}")


if __name__ == "__main__":
    main()
