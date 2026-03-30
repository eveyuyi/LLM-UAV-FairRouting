"""Manifest-only helpers for rich event records and downstream solver inputs."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from llm4fairrouting.data.event_structuring import build_gold_structured_demand
from llm4fairrouting.data.priority_labels import attach_priority_labels


def load_event_records(path: str | Path) -> List[Dict[str, object]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        with open(source, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with open(source, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "event_manifest" in payload:
            return list(payload["event_manifest"])
        if isinstance(payload, list):
            return list(payload)
        raise ValueError(f"Unsupported JSON payload for event records: {path}")
    raise ValueError(f"Only JSONL/JSON rich event manifests are supported: {path}")


def event_record_to_solver_demand(record: Dict[str, object], *, base_date: str = "2024-03-15") -> Dict[str, object]:
    demand = deepcopy(build_gold_structured_demand(record))
    labels = demand.get("labels", {})
    attach_priority_labels(
        demand,
        latent_priority=labels.get("latent_priority", record.get("latent_priority")),
        dialogue_observable_priority=record.get("dialogue_observable_priority"),
    )
    if not demand.get("request_timestamp"):
        timestamp = datetime.strptime(base_date, "%Y-%m-%d") + timedelta(minutes=int(record.get("time_slot", 0)) * 5)
        demand["request_timestamp"] = timestamp.isoformat(timespec="seconds")
    demand.setdefault("source_event_id", str(record.get("event_id", "")))
    return demand


def ground_truth_priority_from_record(record: Dict[str, object]) -> int:
    labels = dict(record.get("labels", {}) or {})
    if not labels and any(
        key in record
        for key in (
            "extraction_observable_priority",
            "dialogue_observable_priority",
            "latent_priority",
            "solver_useful_priority",
        )
    ):
        labels = {
            "extraction_observable_priority": record.get("extraction_observable_priority"),
            "dialogue_observable_priority": record.get("dialogue_observable_priority"),
            "latent_priority": record.get("latent_priority"),
            "solver_useful_priority": record.get("solver_useful_priority"),
        }
    elif not labels:
        gold = build_gold_structured_demand(record)
        labels = dict(gold.get("labels", {}) or {})
    return int(
        labels.get("extraction_observable_priority")
        or labels.get("dialogue_observable_priority")
        or record.get("dialogue_observable_priority")
        or labels.get("latent_priority")
        or record.get("latent_priority")
        or 4
    )


def load_ground_truth_event_index(path: str | Path) -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    for record in load_event_records(path):
        event_id = str(record.get("event_id", "")).strip()
        if not event_id:
            continue
        gold = build_gold_structured_demand(record)
        cargo = gold.get("cargo", {})
        origin = gold.get("origin", {})
        destination = gold.get("destination", {})
        index[event_id] = {
            "event_id": event_id,
            "priority": ground_truth_priority_from_record(record),
            "supply_fid": str(origin.get("fid", "")).strip(),
            "demand_fid": str(destination.get("fid") or destination.get("node_id") or "").strip(),
            "material_weight": float(cargo.get("weight_kg", record.get("weight_kg", 0.0)) or 0.0),
            "record": record,
        }
    return index
