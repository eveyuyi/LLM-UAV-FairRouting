"""Evaluate downstream operational impact of different priority inference methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


PRIORITY_WEIGHTS = {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}


def _load_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_run_specs(specs: List[str]) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Run spec must use name=path format: {spec}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
        if not name or not path.exists():
            raise ValueError(f"Invalid run spec: {spec}")
        runs[name] = path
    return runs


def _load_ground_truth_priorities(path: str | Path) -> Dict[str, int]:
    import csv

    priorities: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event_id = str(row.get("event_id") or row.get("unique_id") or "").strip()
            if not event_id:
                continue
            priorities[event_id] = int(row.get("priority", 4))
    return priorities


def _load_dialogue_metadata(path: str | Path) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
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
                "deadline_minutes": metadata.get("delivery_deadline_minutes"),
                "timestamp": dialogue.get("timestamp"),
            }
            if dialogue_id:
                by_dialogue[dialogue_id] = record
            if event_id:
                by_event[event_id] = record
    return by_dialogue, by_event


def _load_per_demand_results(run_dir: Path) -> List[Dict[str, object]]:
    workflow_results_path = run_dir / "workflow_results.json"
    payload = _load_json(workflow_results_path)
    if not isinstance(payload, list):
        raise ValueError(f"workflow_results.json must be a list: {workflow_results_path}")

    rows: List[Dict[str, object]] = []
    for entry in payload:
        time_window = entry.get("time_window")
        for item in entry.get("per_demand_results", []):
            row = dict(item)
            row["time_window"] = time_window
            rows.append(row)
    return rows


def _mean(values: List[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _pctl(values: List[float], percentile: float) -> float | None:
    return round(float(np.percentile(values, percentile)), 6) if values else None


def _safe_relative_improvement(reference: float | None, current: float | None) -> float | None:
    if reference in (None, 0) or current is None:
        return None
    return round((float(reference) - float(current)) / float(reference), 6)


def _safe_difference(current: float | None, reference: float | None) -> float | None:
    if current is None or reference is None:
        return None
    return round(float(current) - float(reference), 6)


def _align_run_rows(
    *,
    run_rows: List[Dict[str, object]],
    dialogue_lookup: Dict[str, Dict[str, object]],
    ground_truth: Dict[str, int],
) -> List[Dict[str, object]]:
    aligned: List[Dict[str, object]] = []
    for row in run_rows:
        dialogue_id = str(row.get("source_dialogue_id", "")).strip()
        source_event_id = str(row.get("source_event_id", "")).strip()
        dialogue_meta = dialogue_lookup.get(dialogue_id, {})
        event_id = source_event_id or str(dialogue_meta.get("event_id", "")).strip()
        if not event_id or event_id not in ground_truth:
            continue
        deadline_minutes = row.get("deadline_minutes")
        if deadline_minutes is None:
            deadline_minutes = dialogue_meta.get("deadline_minutes")
        delivery_latency_min = row.get("delivery_latency_min")
        if delivery_latency_min is None and row.get("delivery_latency_s") is not None:
            delivery_latency_min = round(float(row["delivery_latency_s"]) / 60.0, 6)
        served = bool(row.get("is_served"))
        aligned.append(
            {
                **row,
                "event_id": event_id,
                "true_priority": int(ground_truth[event_id]),
                "deadline_minutes": float(deadline_minutes) if deadline_minutes not in (None, "") else None,
                "delivery_latency_min": float(delivery_latency_min) if delivery_latency_min not in (None, "") else None,
                "is_served": served,
                "is_deadline_met": bool(
                    served
                    and delivery_latency_min is not None
                    and deadline_minutes not in (None, "")
                    and float(delivery_latency_min) <= float(deadline_minutes)
                ),
            }
        )
    return aligned


def _summarize_subset(items: List[Dict[str, object]]) -> Dict[str, object]:
    served = [item for item in items if item.get("is_served")]
    latencies = [float(item["delivery_latency_min"]) for item in served if item.get("delivery_latency_min") is not None]
    on_time = [item for item in items if item.get("is_deadline_met")]
    return {
        "count": len(items),
        "served_count": len(served),
        "service_rate": round(len(served) / len(items), 6) if items else None,
        "avg_delivery_latency_min": _mean(latencies),
        "p90_delivery_latency_min": _pctl(latencies, 90.0),
        "on_time_rate": round(len(on_time) / len(items), 6) if items else None,
    }


def _priority_weighted_score(items: List[Dict[str, object]], value_key: str) -> float | None:
    if not items:
        return None
    total_weight = 0.0
    achieved = 0.0
    for item in items:
        weight = PRIORITY_WEIGHTS.get(int(item.get("true_priority", 4)), 1.0)
        total_weight += weight
        achieved += weight * float(bool(item.get(value_key)))
    if total_weight <= 0.0:
        return None
    return round(achieved / total_weight, 6)


def summarize_operational_impact(
    *,
    run_dirs: Dict[str, Path],
    dialogues_path: str,
    ground_truth_csv: str,
    urgent_threshold: int = 2,
    reference_method: str | None = None,
) -> Dict[str, object]:
    dialogue_lookup, _event_lookup = _load_dialogue_metadata(dialogues_path)
    ground_truth = _load_ground_truth_priorities(ground_truth_csv)

    methods: Dict[str, Dict[str, object]] = {}
    for method_name, run_dir in run_dirs.items():
        aligned = _align_run_rows(
            run_rows=_load_per_demand_results(run_dir),
            dialogue_lookup=dialogue_lookup,
            ground_truth=ground_truth,
        )
        urgent_items = [item for item in aligned if int(item["true_priority"]) <= urgent_threshold]
        p1_items = [item for item in aligned if int(item["true_priority"]) == 1]
        methods[method_name] = {
            "run_dir": str(run_dir),
            "n_aligned_demands": len(aligned),
            "urgent_threshold": urgent_threshold,
            "overall": _summarize_subset(aligned),
            "priority_1": _summarize_subset(p1_items),
            "urgent": _summarize_subset(urgent_items),
            "priority_weighted_service_score": _priority_weighted_score(aligned, "is_served"),
            "priority_weighted_on_time_score": _priority_weighted_score(aligned, "is_deadline_met"),
        }

    method_names = list(run_dirs.keys())
    reference = reference_method or (method_names[0] if method_names else None)
    comparisons: Dict[str, Dict[str, object]] = {}
    if reference and reference in methods:
        ref_metrics = methods[reference]
        for method_name, metrics in methods.items():
            if method_name == reference:
                continue
            comparisons[f"{method_name}_vs_{reference}"] = {
                "reference_method": reference,
                "method": method_name,
                "priority_1_latency_improvement": _safe_relative_improvement(
                    ref_metrics["priority_1"]["avg_delivery_latency_min"],
                    metrics["priority_1"]["avg_delivery_latency_min"],
                ),
                "urgent_latency_improvement": _safe_relative_improvement(
                    ref_metrics["urgent"]["avg_delivery_latency_min"],
                    metrics["urgent"]["avg_delivery_latency_min"],
                ),
                "priority_1_service_rate_gain": _safe_difference(
                    metrics["priority_1"]["service_rate"],
                    ref_metrics["priority_1"]["service_rate"],
                ),
                "urgent_service_rate_gain": _safe_difference(
                    metrics["urgent"]["service_rate"],
                    ref_metrics["urgent"]["service_rate"],
                ),
                "priority_1_on_time_rate_gain": _safe_difference(
                    metrics["priority_1"]["on_time_rate"],
                    ref_metrics["priority_1"]["on_time_rate"],
                ),
                "urgent_on_time_rate_gain": _safe_difference(
                    metrics["urgent"]["on_time_rate"],
                    ref_metrics["urgent"]["on_time_rate"],
                ),
                "priority_weighted_service_gain": _safe_difference(
                    metrics["priority_weighted_service_score"],
                    ref_metrics["priority_weighted_service_score"],
                ),
                "priority_weighted_on_time_gain": _safe_difference(
                    metrics["priority_weighted_on_time_score"],
                    ref_metrics["priority_weighted_on_time_score"],
                ),
            }

    return {
        "reference_method": reference,
        "urgent_threshold": urgent_threshold,
        "methods": methods,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate downstream operational impact of different priority methods.")
    parser.add_argument("--run", action="append", required=True, help="Method run spec in name=run_dir format. Repeat for multiple methods.")
    parser.add_argument("--dialogues", required=True, help="Dialogue JSONL file used to build extracted demands")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth daily_demand_events.csv")
    parser.add_argument("--urgent-threshold", type=int, default=2, help="Priorities <= threshold are treated as urgent")
    parser.add_argument("--reference-method", help="Reference method name for gain/improvement calculations")
    parser.add_argument("--output", default="evals/results/priority_operational_impact.json", help="Output JSON path")
    args = parser.parse_args()

    payload = summarize_operational_impact(
        run_dirs=_parse_run_specs(args.run),
        dialogues_path=args.dialogues,
        ground_truth_csv=args.ground_truth,
        urgent_threshold=args.urgent_threshold,
        reference_method=args.reference_method,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Priority operational impact saved to {output_path}")


if __name__ == "__main__":
    main()
