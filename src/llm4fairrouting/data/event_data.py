"""Shared helpers for rich event manifests and ground-truth event loading."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from llm4fairrouting.data.demand_event_generation import (
    PRIORITY_TO_DEADLINE,
    PRIORITY_TO_TIER,
    build_gold_structured_demand,
)
from llm4fairrouting.data.priority_labels import attach_priority_labels


def _coerce_time_slot(row: Dict[str, object]) -> int:
    raw_slot = row.get("time_slot")
    if raw_slot not in (None, ""):
        try:
            return int(raw_slot)
        except (TypeError, ValueError):
            pass
    try:
        return int(round(float(row.get("time", 0.0) or 0.0) * 12))
    except (TypeError, ValueError):
        return 0


def _normalize_csv_record(row: Dict[str, object]) -> Dict[str, object]:
    priority = int(row.get("latent_priority") or row.get("priority") or 4)
    record = {
        "event_id": str(row.get("event_id") or row.get("unique_id") or "").strip(),
        "unique_id": str(row.get("unique_id") or row.get("event_id") or "").strip(),
        "time_slot": _coerce_time_slot(row),
        "time_hour": float(row.get("time", 0.0) or 0.0),
        "origin": {
            "fid": str(row.get("supply_fid", "")).strip(),
            "coords": [
                float(row.get("supply_lon", 0.0) or 0.0),
                float(row.get("supply_lat", 0.0) or 0.0),
            ],
            "type": "supply_station",
            "station_name": str(row.get("supply_fid", "")).strip(),
            "supply_type": str(row.get("supply_type", "")).strip(),
        },
        "destination": {
            "fid": str(row.get("demand_fid") or row.get("demand_node_id") or "").strip(),
            "node_id": str(row.get("demand_node_id") or row.get("demand_fid") or "").strip(),
            "coords": [
                float(row.get("demand_lon", 0.0) or 0.0),
                float(row.get("demand_lat", 0.0) or 0.0),
            ],
            "type": "residential_area",
        },
        "cargo": {
            "type": str(row.get("material_type", "medicine")),
            "weight_kg": float(row.get("quantity_kg", row.get("material_weight", 1.0)) or 1.0),
            "demand_tier": PRIORITY_TO_TIER.get(priority, "consumer"),
        },
        "weight_kg": float(row.get("quantity_kg", row.get("material_weight", 1.0)) or 1.0),
        "deadline_minutes": int(row.get("deadline_minutes") or row.get("delivery_deadline_minutes") or PRIORITY_TO_DEADLINE.get(priority, 120)),
        "requester_role": str(row.get("requester_role") or "consumer"),
        "special_handling": [],
        "population_vulnerability": {
            "elderly_ratio": 0.0,
            "population": 0,
            "elderly_involved": False,
            "children_involved": False,
            "vulnerable_community": False,
        },
        "receiver_ready": False,
        "operational_readiness": "Standard handoff readiness",
        "latent_priority": priority,
        "priority": priority,
    }
    record["gold_structured_demand"] = build_gold_structured_demand(record)
    return record


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

    records: List[Dict[str, object]] = []
    with open(source, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event_id = str(row.get("event_id") or row.get("unique_id") or "").strip()
            if not event_id:
                continue
            records.append(_normalize_csv_record(row))
    return records


def event_record_to_solver_demand(record: Dict[str, object], *, base_date: str = "2024-03-15") -> Dict[str, object]:
    demand = deepcopy(record.get("gold_structured_demand") or build_gold_structured_demand(record))
    labels = demand.get("labels", {})
    attach_priority_labels(
        demand,
        latent_priority=labels.get("latent_priority", record.get("latent_priority")),
        dialogue_observable_priority=labels.get("dialogue_observable_priority"),
    )
    if not demand.get("request_timestamp"):
        timestamp = datetime.strptime(base_date, "%Y-%m-%d") + timedelta(minutes=int(record.get("time_slot", 0)) * 5)
        demand["request_timestamp"] = timestamp.isoformat(timespec="seconds")
    demand.setdefault("source_event_id", str(record.get("event_id", "")))
    return demand


def ground_truth_priority_from_record(record: Dict[str, object]) -> int:
    gold = record.get("gold_structured_demand", {}) or {}
    labels = gold.get("labels", {}) or {}
    return int(
        labels.get("extraction_observable_priority")
        or record.get("extraction_observable_priority")
        or record.get("dialogue_observable_priority")
        or labels.get("dialogue_observable_priority")
        or record.get("latent_priority")
        or labels.get("latent_priority")
        or record.get("priority")
        or 4
    )


def load_ground_truth_event_index(path: str | Path) -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    for record in load_event_records(path):
        event_id = str(record.get("event_id") or record.get("unique_id") or "").strip()
        if not event_id:
            continue
        gold = record.get("gold_structured_demand") or build_gold_structured_demand(record)
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
