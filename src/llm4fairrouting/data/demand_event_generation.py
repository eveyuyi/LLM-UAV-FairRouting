"""Seed demand-event generation helpers and CLI."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from llm4fairrouting.data.building_information import (
    HEALTHCARE_LAND_USE,
    RESIDENTIAL_LAND_USE,
    load_building_data,
)
from llm4fairrouting.data.priority_labels import derive_priority_labels
from llm4fairrouting.data.seed_paths import (
    BUILDING_DATA_PATH,
    DEMAND_EVENTS_MANIFEST_PATH,
    DEMAND_EVENTS_PATH,
)
from llm4fairrouting.routing.domain import DemandEvent, Point

COMMERCIAL_LAND_USE = "commercial_service_land"
MEDICAL_SUPPLY_TYPE = "medical"
COMMERCIAL_SUPPLY_TYPE = "commercial"
DIALOGUE_STYLE_VARIANTS = ("direct", "technical", "conversational", "dispatch_brief")
PRIORITY_TO_TIER = {
    1: "life_support",
    2: "critical",
    3: "regular",
    4: "consumer",
}
PRIORITY_TO_DEADLINE = {
    1: 15,
    2: 30,
    3: 60,
    4: 120,
}
MATERIAL_LABELS = {
    "aed": "AED defibrillator",
    "blood_product": "blood product",
    "cardiac_drug": "cardiac emergency drug",
    "thrombolytic": "thrombolytic agent",
    "ventilator": "ventilator",
    "icu_drug": "ICU medication",
    "vaccine": "vaccine",
    "medicine": "medication",
    "protective_suit": "protective suit",
    "mask": "face mask",
    "disinfectant": "disinfectant solution",
    "food": "food/meal",
    "otc_drug": "OTC medication",
    "daily_supply": "daily supplies",
}
TEMPERATURE_SENSITIVE_MATERIALS = {"blood_product", "thrombolytic", "vaccine"}
UNIT_BY_MATERIAL = {
    "aed": "unit",
    "blood_product": "pack",
    "cardiac_drug": "dose",
    "thrombolytic": "dose",
    "ventilator": "unit",
    "icu_drug": "kit",
    "vaccine": "dose",
    "medicine": "box",
    "protective_suit": "set",
    "mask": "pack",
    "disinfectant": "bottle",
    "food": "meal",
    "otc_drug": "box",
    "daily_supply": "package",
}
OUTPUT_COLUMNS = [
    "time",
    "demand_fid",
    "demand_lon",
    "demand_lat",
    "priority",
    "supply_fid",
    "supply_lon",
    "supply_lat",
    "supply_type",
    "material_weight",
    "unique_id",
]
DEFAULT_MEDICAL_PRIORITIES = (1, 2, 3)
DEFAULT_MEDICAL_PRIORITY_PROBS = (1 / 3, 1 / 3, 1 / 3)
DEFAULT_COMMERCIAL_PRIORITY = 4

PrioritySampler = Callable[[int, random.Random, int], int]
TimeSampler = Callable[[int, random.Random, float], float]
SupplyIndexSampler = Callable[[int, random.Random, Sequence[Point]], int]
UniqueIdFactory = Callable[[int, str, int], str]


def _factor_spec(name: str, value: object, description: str, keywords: Sequence[str]) -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "description": description,
        "keywords": list(dict.fromkeys(str(keyword).lower() for keyword in keywords if keyword)),
    }


def _material_candidates(supply_type: str, priority: int) -> tuple[str, ...]:
    if supply_type == MEDICAL_SUPPLY_TYPE:
        mapping = {
            1: ("aed", "blood_product", "cardiac_drug", "thrombolytic"),
            2: ("ventilator", "icu_drug"),
            3: ("vaccine", "medicine", "protective_suit"),
            4: ("medicine", "vaccine"),
        }
    else:
        mapping = {
            1: ("medicine", "protective_suit", "disinfectant"),
            2: ("medicine", "mask", "disinfectant"),
            3: ("food", "mask", "daily_supply"),
            4: ("food", "daily_supply", "otc_drug"),
        }
    return mapping.get(int(priority), mapping[4])


def _destination_type_for_priority(priority: int, material_type: str, rng: random.Random) -> str:
    if priority == 1:
        options = ["hospital", "public_space"]
    elif priority == 2:
        options = ["hospital", "clinic"]
    elif priority == 3:
        options = ["clinic", "community_health_center", "residential_area"]
    else:
        options = ["residential_area", "office", "community_locker"]
    if material_type == "vaccine":
        options = ["clinic", "community_health_center"]
    return rng.choice(options)


def _requester_role_for_priority(priority: int, rng: random.Random) -> str:
    options = {
        1: ("emergency_doctor", "paramedic", "triage_nurse"),
        2: ("icu_nurse", "clinical_pharmacist", "ward_coordinator"),
        3: ("community_health_worker", "clinic_manager", "pharmacy_staff"),
        4: ("consumer", "family_caregiver", "office_administrator"),
    }
    return rng.choice(options.get(int(priority), options[4]))


def _special_handling_for_material(material_type: str) -> list[str]:
    special_handling = []
    if material_type in TEMPERATURE_SENSITIVE_MATERIALS:
        special_handling.append("cold_chain")
    if material_type in {"aed", "ventilator"}:
        special_handling.append("shock_protection")
    return special_handling


def _population_vulnerability(
    material_type: str,
    priority: int,
    destination_type: str,
    rng: random.Random,
) -> dict[str, object]:
    children = material_type == "vaccine" or (priority <= 2 and rng.random() < 0.25)
    elderly = destination_type in {"clinic", "community_health_center"} or rng.random() < 0.30
    elderly_ratio = round(rng.uniform(0.15, 0.62), 2)
    population = rng.randint(600, 9000)
    return {
        "elderly_ratio": elderly_ratio,
        "population": population,
        "elderly_involved": elderly,
        "children_involved": children,
        "vulnerable_community": bool(children or elderly or elderly_ratio >= 0.45),
    }


def _receiver_ready(priority: int, rng: random.Random) -> bool:
    threshold = {
        1: 0.95,
        2: 0.80,
        3: 0.60,
        4: 0.45,
    }
    return rng.random() < threshold.get(int(priority), 0.45)


def _operational_readiness(receiver_ready: bool, priority: int) -> str:
    if receiver_ready:
        if priority <= 2:
            return "Landing zone cleared; team waiting for immediate handoff"
        return "Receiver confirmed and ready for handoff"
    return "Standard handoff readiness"


def _scenario_context(priority: int, material_type: str, destination_type: str) -> str:
    scenario_map = {
        (1, "aed"): "CPR is already in progress and the local AED cabinet is empty.",
        (1, "blood_product"): "An active transfusion case needs immediate blood product support.",
        (1, "cardiac_drug"): "The emergency cart is short on the cardiac rescue dose.",
        (1, "thrombolytic"): "The stroke treatment window is closing quickly.",
        (2, "ventilator"): "A critical patient needs a backup ventilator before the next procedure.",
        (2, "icu_drug"): "The ICU team needs a refill before the next administration round.",
        (3, "vaccine"): "The afternoon vaccination block starts soon and stock is running low.",
        (4, "otc_drug"): "A same-day home-care order needs symptom relief medication.",
    }
    if (priority, material_type) in scenario_map:
        return scenario_map[(priority, material_type)]
    if destination_type in {"hospital", "clinic", "community_health_center"}:
        return "The local care team has a time-bound replenishment need."
    return "The receiver requested a same-day drone delivery."


def _priority_factor_bundle(
    *,
    material_type: str,
    deadline_minutes: int,
    requester_role: str,
    special_handling: list[str],
    vulnerability: dict[str, object],
    receiver_ready: bool,
    scenario_context: str,
    operational_readiness: str,
    destination_type: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    must_mention = [
        _factor_spec(
            "scenario_context",
            scenario_context,
            scenario_context,
            [
                "cpr",
                "cardiac arrest",
                "transfusion",
                "stroke",
                "backup ventilator",
                "vaccination",
                "same-day",
            ],
        ),
        _factor_spec(
            "deadline_minutes",
            deadline_minutes,
            f"Delivery is needed within {deadline_minutes} minutes.",
            [
                f"{deadline_minutes} min",
                f"{deadline_minutes}-minute",
                f"within {deadline_minutes} minutes",
            ],
        ),
        _factor_spec(
            "requester_role",
            requester_role,
            f"The request comes from the {requester_role.replace('_', ' ')}.",
            [requester_role.replace("_", " "), requester_role.replace("_", " ").title()],
        ),
    ]
    optional = [
        _factor_spec(
            "destination_type",
            destination_type,
            f"The receiving point is a {destination_type.replace('_', ' ')}.",
            [destination_type.replace("_", " "), "receiving point"],
        ),
    ]
    if special_handling:
        must_mention.append(
            _factor_spec(
                "special_handling",
                list(special_handling),
                f"Special handling is required: {', '.join(special_handling)}.",
                list(special_handling) + ["cold-chain", "shock-proof", "insulated"],
            )
        )
    if vulnerability.get("children_involved") or vulnerability.get("elderly_involved"):
        optional.append(
            _factor_spec(
                "population_vulnerability",
                vulnerability,
                "The receiver serves a vulnerable population.",
                ["child", "children", "elderly", "senior", "vulnerable community"],
            )
        )
    if receiver_ready:
        must_mention.append(
            _factor_spec(
                "receiver_ready",
                True,
                operational_readiness,
                ["landing zone", "standing by", "ready for handoff", "team waiting"],
            )
        )
    priority_factors = {
        "scenario_context": scenario_context,
        "deadline_minutes": deadline_minutes,
        "requester_role": requester_role,
        "special_handling": list(special_handling),
        "population_vulnerability": dict(vulnerability),
        "receiver_ready": receiver_ready,
        "operational_readiness": operational_readiness,
    }
    return priority_factors, must_mention, optional


def build_gold_structured_demand(event: dict[str, object]) -> dict[str, object]:
    event_id = str(event.get("event_id", event.get("unique_id", "")))
    priority = int(event.get("latent_priority", event.get("priority", DEFAULT_COMMERCIAL_PRIORITY)))
    cargo = dict(event.get("cargo", {}))
    weight_kg = float(
        cargo.get("weight_kg")
        or event.get("weight_kg")
        or event.get("weight")
        or event.get("material_weight")
        or 1.0
    )
    deadline_minutes = int(event.get("deadline_minutes", event.get("deadline", PRIORITY_TO_DEADLINE.get(priority, 120))))
    demand_tier = str(cargo.get("demand_tier") or PRIORITY_TO_TIER.get(priority, "consumer"))
    requester_role = str(event.get("requester_role", "consumer"))
    special_handling = list(event.get("special_handling", []))
    vulnerability = dict(event.get("population_vulnerability", {}))
    receiver_ready = bool(event.get("receiver_ready", False))
    operational_readiness = str(event.get("operational_readiness", _operational_readiness(receiver_ready, priority)))
    origin = dict(event.get("origin", {}))
    destination = dict(event.get("destination", {}))
    material_type = str(cargo.get("type", event.get("material_type", "medicine")))
    scenario_context = str(
        event.get("priority_factors", {}).get("scenario_context")
        or _scenario_context(priority, material_type, str(destination.get("type", "residential_area")))
    )
    quantity = max(1, round(weight_kg / {
        "aed": 2.0,
        "blood_product": 0.25,
        "cardiac_drug": 0.05,
        "thrombolytic": 0.05,
        "ventilator": 8.0,
        "icu_drug": 0.1,
        "vaccine": 0.3,
        "medicine": 0.5,
        "protective_suit": 0.8,
        "mask": 0.05,
        "disinfectant": 1.0,
        "food": 0.5,
        "otc_drug": 0.1,
        "daily_supply": 0.3,
    }.get(material_type, 0.5)))
    structured = {
        "demand_id": event_id,
        "source_event_id": event_id,
        "request_timestamp": event.get("request_timestamp"),
        "origin": {
            "station_name": str(origin.get("station_name", origin.get("fid", ""))),
            "fid": str(origin.get("fid", "")),
            "coords": list(origin.get("coords", [0.0, 0.0])),
            "type": "supply_station",
        },
        "destination": {
            "node_id": str(destination.get("node_id", destination.get("fid", ""))),
            "fid": str(destination.get("fid", "")),
            "coords": list(destination.get("coords", [0.0, 0.0])),
            "type": str(destination.get("type", "residential_area")),
        },
        "cargo": {
            "type": material_type,
            "type_cn": str(cargo.get("type_cn", MATERIAL_LABELS.get(material_type, material_type))),
            "weight_kg": weight_kg,
            "quantity": quantity,
            "quantity_unit": UNIT_BY_MATERIAL.get(material_type, "unit"),
            "temperature_sensitive": bool(cargo.get("temperature_sensitive", material_type in TEMPERATURE_SENSITIVE_MATERIALS)),
            "demand_tier": demand_tier,
        },
        "demand_tier": demand_tier,
        "time_constraint": {
            "type": "hard" if deadline_minutes <= 30 or demand_tier in {"life_support", "critical"} else "soft",
            "description": f"Delivery target within {deadline_minutes} minutes",
            "deadline_minutes": deadline_minutes,
        },
        "requester_role": requester_role,
        "special_handling": special_handling,
        "population_vulnerability": vulnerability,
        "operational_readiness": operational_readiness,
        "receiver_ready": receiver_ready,
        "priority_evaluation_signals": {
            "patient_condition": scenario_context,
            "time_sensitivity": (
                "Immediate action required"
                if deadline_minutes <= 15
                else "Urgent same-window delivery required"
                if deadline_minutes <= 30
                else "Timely delivery needed within the service window"
                if deadline_minutes <= 90
                else "Flexible same-day delivery"
            ),
            "population_vulnerability": vulnerability,
            "medical_urgency_self_report": scenario_context,
            "requester_role": requester_role,
            "scenario_context": scenario_context,
            "nearby_critical_facility": str(destination.get("type", "")),
            "operational_readiness": operational_readiness,
            "special_handling": special_handling,
        },
        "context_signals": [
            scenario_context,
            f"Structured deadline: {deadline_minutes} minutes",
            f"Requester role: {requester_role}",
            operational_readiness,
        ],
    }
    labels = derive_priority_labels(structured, latent_priority=priority)
    structured["labels"] = labels
    structured["latent_priority"] = labels.get("latent_priority", priority)
    structured["extraction_observable_priority"] = labels["extraction_observable_priority"]
    structured["solver_useful_priority"] = labels["solver_useful_priority"]
    return structured


def _record_to_dataframe_row(record: dict[str, object]) -> dict[str, object]:
    origin = dict(record.get("origin", {}))
    destination = dict(record.get("destination", {}))
    cargo = dict(record.get("cargo", {}))
    return {
        "time": round(float(record.get("time_hour", 0.0)), 4),
        "demand_fid": destination.get("fid", ""),
        "demand_lon": float(destination.get("coords", [0.0, 0.0])[0]),
        "demand_lat": float(destination.get("coords", [0.0, 0.0])[1]),
        "priority": int(record.get("latent_priority", record.get("priority", DEFAULT_COMMERCIAL_PRIORITY))),
        "supply_fid": origin.get("fid", ""),
        "supply_lon": float(origin.get("coords", [0.0, 0.0])[0]),
        "supply_lat": float(origin.get("coords", [0.0, 0.0])[1]),
        "supply_type": origin.get("supply_type", ""),
        "material_weight": round(float(cargo.get("weight_kg", record.get("weight_kg", 0.0))), 1),
        "unique_id": record.get("event_id", record.get("unique_id", "")),
    }


def save_event_manifest(records: Sequence[dict[str, object]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _default_priority_sampler(i: int, rng: random.Random, num_events: int) -> int:
    if i == 0:
        _default_priority_sampler.cache = list(range(1, num_events + 1))
        rng.shuffle(_default_priority_sampler.cache)
    return _default_priority_sampler.cache[i]


_default_priority_sampler.cache = []


def _default_time_sampler(_: int, rng: random.Random, sim_duration: float) -> float:
    lower = 0.1
    upper = max(lower, sim_duration - 0.5)
    if upper == lower:
        return round(lower, 3)
    return round(rng.uniform(lower, upper), 3)


def _default_supply_index_sampler(_: int, rng: random.Random, supply_points: Sequence[Point]) -> int:
    return rng.randrange(len(supply_points))


def _default_unique_id_factory(_: int, demand_point_id: str, occurrence: int) -> str:
    return f"DEM_{demand_point_id}_{occurrence:02d}"


def generate_demand_events(
    demand_points: Sequence[Point],
    supply_points: Sequence[Point],
    num_events: int = 8,
    sim_duration: float = 2.0,
    rng: random.Random | None = None,
    priority_sampler: PrioritySampler | None = None,
    time_sampler: TimeSampler | None = None,
    supply_index_sampler: SupplyIndexSampler | None = None,
    unique_id_factory: UniqueIdFactory | None = None,
    verbose: bool = True,
) -> list[DemandEvent]:
    """Generate randomized demand events for a demand/supply point set.

    The default samplers preserve the original baseline behavior.
    """
    if not demand_points:
        raise ValueError("At least one demand point is required to generate demand events.")
    if not supply_points:
        raise ValueError("At least one supply point is required to generate demand events.")
    if num_events < 0:
        raise ValueError("num_events must be non-negative.")

    rng = rng or random
    priority_sampler = priority_sampler or _default_priority_sampler
    time_sampler = time_sampler or _default_time_sampler
    supply_index_sampler = supply_index_sampler or _default_supply_index_sampler
    unique_id_factory = unique_id_factory or _default_unique_id_factory

    events: list[DemandEvent] = []
    n_supply = len(supply_points)
    demand_point_counter = {point.id: 0 for point in demand_points}

    if verbose:
        print(f"\nGenerating {num_events} demand events...")
        print(f"Supply points: {n_supply}, demand points: {len(demand_points)}")

    for i in range(num_events):
        d_idx = rng.randrange(len(demand_points))
        demand_point = demand_points[d_idx]
        node_idx = d_idx + n_supply

        demand_point_counter[demand_point.id] += 1
        occurrence = demand_point_counter[demand_point.id]
        required_supply_idx = int(supply_index_sampler(i, rng, supply_points))
        if required_supply_idx < 0 or required_supply_idx >= n_supply:
            raise ValueError(f"supply_index_sampler returned out-of-range index {required_supply_idx}")

        t = round(float(time_sampler(i, rng, sim_duration)), 3)
        weight = round(5.0 + rng.random() * 25.0, 1)
        priority = int(priority_sampler(i, rng, num_events))
        unique_id = unique_id_factory(i, demand_point.id, occurrence)

        events.append(DemandEvent(
            time=t,
            node_idx=node_idx,
            weight=weight,
            unique_id=unique_id,
            priority=priority,
            required_supply_idx=required_supply_idx,
            demand_point_id=demand_point.id,
        ))

    events.sort(key=lambda item: item.time)

    if verbose:
        print(f"\nGenerated {num_events} demand events with priorities: {[ev.priority for ev in events]}")
        print("\nDemand details (sorted by time):")
        print("-" * 80)
        for ev in events:
            print(
                f"  {ev.unique_id}: {ev.demand_point_id} (node {ev.node_idx}) | "
                f"time={ev.time:.3f}h | weight={ev.weight}kg | "
                f"priority={ev.priority} | pickup from S{ev.required_supply_idx + 1}"
            )

        print("\nDemand-point distribution:")
        for demand_id, count in demand_point_counter.items():
            if count > 0:
                print(f"  {demand_id}: {count} events")

    return events


def _select_commercial_rows(
    medical_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
    rng: random.Random,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not commercial_df.empty:
        return medical_df, commercial_df

    if medical_df.empty:
        raise ValueError("No commercial or medical land-use rows are available for supply selection.")

    medical_indices = medical_df.index.tolist()
    rng.shuffle(medical_indices)
    split_point = max(1, len(medical_indices) // 2)
    commercial_indices = medical_indices[:split_point]
    commercial = medical_df.loc[commercial_indices].copy()
    commercial["land_use_type"] = COMMERCIAL_LAND_USE
    remaining_medical = medical_df.drop(commercial_indices)
    if remaining_medical.empty:
        remaining_medical = medical_df.head(1).copy()
    return remaining_medical, commercial


def _build_supply_points(
    medical_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
    num_supply_medical: int,
    num_supply_commercial: int,
) -> tuple[list[Point], dict[str, list[int]]]:
    supply_points: list[Point] = []
    group_indices = {
        MEDICAL_SUPPLY_TYPE: [],
        COMMERCIAL_SUPPLY_TYPE: [],
    }

    selected_medical = medical_df.head(min(num_supply_medical, len(medical_df)))
    for idx, row in selected_medical.iterrows():
        group_indices[MEDICAL_SUPPLY_TYPE].append(len(supply_points))
        supply_points.append(Point(
            id=f"MED_{idx}",
            lon=float(row["longitude"]),
            lat=float(row["latitude"]),
            alt=0.0,
            type=MEDICAL_SUPPLY_TYPE,
        ))

    selected_commercial = commercial_df.head(min(num_supply_commercial, len(commercial_df)))
    for idx, row in selected_commercial.iterrows():
        group_indices[COMMERCIAL_SUPPLY_TYPE].append(len(supply_points))
        supply_points.append(Point(
            id=f"COM_{idx}",
            lon=float(row["longitude"]),
            lat=float(row["latitude"]),
            alt=0.0,
            type=COMMERCIAL_SUPPLY_TYPE,
        ))

    if not supply_points:
        raise ValueError("No supply points could be created from the building dataset.")

    return supply_points, group_indices


def _build_demand_points(residential_df: pd.DataFrame) -> list[Point]:
    demand_points = [
        Point(
            id=f"DEM_{idx}",
            lon=float(row["longitude"]),
            lat=float(row["latitude"]),
            alt=0.0,
            type="demand",
        )
        for idx, row in residential_df.iterrows()
    ]
    if not demand_points:
        raise ValueError("No residential rows are available for demand generation.")
    return demand_points


def _events_to_dataframe(events: Sequence[DemandEvent], demand_points: Sequence[Point], supply_points: Sequence[Point]) -> pd.DataFrame:
    n_supply = len(supply_points)
    rows = []
    for event in events:
        demand_idx = event.node_idx - n_supply
        if demand_idx < 0 or demand_idx >= len(demand_points):
            raise ValueError(f"Demand event {event.unique_id} has invalid node_idx={event.node_idx}")

        demand_point = demand_points[demand_idx]
        supply_point = supply_points[event.required_supply_idx or 0]
        rows.append({
            "time": round(event.time, 4),
            "demand_fid": demand_point.id,
            "demand_lon": demand_point.lon,
            "demand_lat": demand_point.lat,
            "priority": int(event.priority),
            "supply_fid": supply_point.id,
            "supply_lon": supply_point.lon,
            "supply_lat": supply_point.lat,
            "supply_type": supply_point.type,
            "material_weight": round(event.weight, 1),
            "unique_id": event.unique_id,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return df[OUTPUT_COLUMNS]


def generate_daily_demand_records(
    building_file: str,
    seed: int = 42,
    time_window_minutes: int = 5,
    demands_per_window_min: int = 4,
    demands_per_window_max: int = 10,
    medical_ratio: float = 0.2,
    medical_priorities: Sequence[int] = DEFAULT_MEDICAL_PRIORITIES,
    medical_priority_probs: Sequence[float] = DEFAULT_MEDICAL_PRIORITY_PROBS,
    commercial_priority: int = DEFAULT_COMMERCIAL_PRIORITY,
    num_supply_medical: int = 5,
    num_supply_commercial: int = 5,
) -> list[dict[str, object]]:
    """Generate the rich event manifest used for dialogue/extraction/priority training."""
    if time_window_minutes <= 0:
        raise ValueError("time_window_minutes must be positive.")
    if 24 * 60 % time_window_minutes != 0:
        raise ValueError("time_window_minutes must divide 1440 evenly.")
    if demands_per_window_min < 0 or demands_per_window_max < demands_per_window_min:
        raise ValueError("Invalid per-window demand range.")
    if not 0.0 <= medical_ratio <= 1.0:
        raise ValueError("medical_ratio must be between 0 and 1.")
    if not medical_priorities:
        raise ValueError("medical_priorities must not be empty.")
    if len(medical_priorities) != len(medical_priority_probs):
        raise ValueError("medical_priorities and medical_priority_probs must have the same length.")

    rng = random.Random(seed)
    buildings_df = load_building_data(building_file)
    medical_df = buildings_df[buildings_df["land_use_type"] == HEALTHCARE_LAND_USE].copy()
    commercial_df = buildings_df[buildings_df["land_use_type"] == COMMERCIAL_LAND_USE].copy()
    residential_df = buildings_df[buildings_df["land_use_type"] == RESIDENTIAL_LAND_USE].copy()

    medical_df, commercial_df = _select_commercial_rows(medical_df, commercial_df, rng)
    supply_points, supply_groups = _build_supply_points(
        medical_df,
        commercial_df,
        num_supply_medical=num_supply_medical,
        num_supply_commercial=num_supply_commercial,
    )
    demand_points = _build_demand_points(residential_df)

    if medical_ratio > 0.0 and not supply_groups[MEDICAL_SUPPLY_TYPE]:
        raise ValueError("medical_ratio > 0 requires at least one medical supply point.")
    if medical_ratio < 1.0 and not supply_groups[COMMERCIAL_SUPPLY_TYPE]:
        raise ValueError("medical_ratio < 1 requires at least one commercial supply point.")

    records: list[dict[str, object]] = []
    window_duration_hours = time_window_minutes / 60.0
    num_windows = 24 * 60 // time_window_minutes
    n_supply = len(supply_points)

    for window_idx in range(num_windows):
        num_demands = rng.randint(demands_per_window_min, demands_per_window_max)
        if num_demands == 0:
            continue

        window_time = round(window_idx * window_duration_hours, 4)
        supply_labels = [
            MEDICAL_SUPPLY_TYPE if rng.random() < medical_ratio else COMMERCIAL_SUPPLY_TYPE
            for _ in range(num_demands)
        ]
        window_priorities = [
            int(rng.choices(medical_priorities, weights=medical_priority_probs, k=1)[0])
            if label == MEDICAL_SUPPLY_TYPE
            else int(commercial_priority)
            for label in supply_labels
        ]

        def priority_sampler(i: int, _rng: random.Random, _num_events: int) -> int:
            return int(window_priorities[i])

        def time_sampler(_: int, _rng: random.Random, _sim_duration: float) -> float:
            return window_time

        def supply_index_sampler(i: int, local_rng: random.Random, _supply_points: Sequence[Point]) -> int:
            return local_rng.choice(supply_groups[supply_labels[i]])

        def unique_id_factory(i: int, _demand_point_id: str, _occurrence: int) -> str:
            return f"DEM_{window_idx:03d}_{i:02d}"

        window_events = generate_demand_events(
            demand_points=demand_points,
            supply_points=supply_points,
            num_events=num_demands,
            sim_duration=24.0,
            rng=rng,
            priority_sampler=priority_sampler,
            time_sampler=time_sampler,
            supply_index_sampler=supply_index_sampler,
            unique_id_factory=unique_id_factory,
            verbose=False,
        )

        for event in window_events:
            demand_idx = event.node_idx - n_supply
            if demand_idx < 0 or demand_idx >= len(demand_points):
                raise ValueError(f"Demand event {event.unique_id} has invalid node_idx={event.node_idx}")
            demand_point = demand_points[demand_idx]
            supply_point = supply_points[event.required_supply_idx or 0]
            priority = int(event.priority)
            material_type = rng.choice(_material_candidates(supply_point.type, priority))
            tier = PRIORITY_TO_TIER.get(priority, "consumer")
            deadline_minutes = PRIORITY_TO_DEADLINE.get(priority, 120)
            destination_type = _destination_type_for_priority(priority, material_type, rng)
            requester_role = _requester_role_for_priority(priority, rng)
            special_handling = _special_handling_for_material(material_type)
            vulnerability = _population_vulnerability(material_type, priority, destination_type, rng)
            receiver_ready = _receiver_ready(priority, rng)
            operational_readiness = _operational_readiness(receiver_ready, priority)
            scenario_context = _scenario_context(priority, material_type, destination_type)
            priority_factors, must_mention_factors, optional_factors = _priority_factor_bundle(
                material_type=material_type,
                deadline_minutes=deadline_minutes,
                requester_role=requester_role,
                special_handling=special_handling,
                vulnerability=vulnerability,
                receiver_ready=receiver_ready,
                scenario_context=scenario_context,
                operational_readiness=operational_readiness,
                destination_type=destination_type,
            )
            record = {
                "schema_version": "priority_observability_v1",
                "event_id": event.unique_id,
                "unique_id": event.unique_id,
                "time_slot": int(round(event.time * 12)),
                "time_hour": round(float(event.time), 4),
                "request_timestamp": None,
                "origin": {
                    "fid": supply_point.id,
                    "coords": [supply_point.lon, supply_point.lat],
                    "type": "supply_station",
                    "station_name": supply_point.id,
                    "supply_type": supply_point.type,
                },
                "destination": {
                    "fid": demand_point.id,
                    "node_id": demand_point.id,
                    "coords": [demand_point.lon, demand_point.lat],
                    "type": destination_type,
                },
                "cargo": {
                    "type": material_type,
                    "type_cn": MATERIAL_LABELS.get(material_type, material_type),
                    "weight_kg": round(float(event.weight), 1),
                    "quantity": max(1, round(float(event.weight))),
                    "quantity_unit": UNIT_BY_MATERIAL.get(material_type, "unit"),
                    "temperature_sensitive": material_type in TEMPERATURE_SENSITIVE_MATERIALS,
                    "demand_tier": tier,
                },
                "weight": round(float(event.weight), 1),
                "weight_kg": round(float(event.weight), 1),
                "deadline": deadline_minutes,
                "deadline_minutes": deadline_minutes,
                "requester_role": requester_role,
                "special_handling": special_handling,
                "population_vulnerability": vulnerability,
                "receiver_ready": receiver_ready,
                "operational_readiness": operational_readiness,
                "latent_priority": priority,
                "priority": priority,
                "priority_factors": priority_factors,
                "must_mention_factors": must_mention_factors,
                "optional_factors": optional_factors,
                "dialogue_styles": list(DIALOGUE_STYLE_VARIANTS),
                "supply_fid": supply_point.id,
                "supply_lon": supply_point.lon,
                "supply_lat": supply_point.lat,
                "supply_type": supply_point.type,
                "demand_fid": demand_point.id,
                "demand_lon": demand_point.lon,
                "demand_lat": demand_point.lat,
                "material_type": material_type,
            }
            record["gold_structured_demand"] = build_gold_structured_demand(record)
            record["solver_useful_priority"] = int(
                record["gold_structured_demand"]["labels"]["solver_useful_priority"]
            )
            records.append(record)

    records.sort(key=lambda item: (float(item["time_hour"]), str(item["event_id"])))
    return records


def generate_daily_demand_dataframe(
    building_file: str,
    seed: int = 42,
    time_window_minutes: int = 5,
    demands_per_window_min: int = 4,
    demands_per_window_max: int = 10,
    medical_ratio: float = 0.2,
    medical_priorities: Sequence[int] = DEFAULT_MEDICAL_PRIORITIES,
    medical_priority_probs: Sequence[float] = DEFAULT_MEDICAL_PRIORITY_PROBS,
    commercial_priority: int = DEFAULT_COMMERCIAL_PRIORITY,
    num_supply_medical: int = 5,
    num_supply_commercial: int = 5,
) -> pd.DataFrame:
    """Generate a full-day demand-event dataframe compatible with daily_demand_events.csv."""
    records = generate_daily_demand_records(
        building_file=building_file,
        seed=seed,
        time_window_minutes=time_window_minutes,
        demands_per_window_min=demands_per_window_min,
        demands_per_window_max=demands_per_window_max,
        medical_ratio=medical_ratio,
        medical_priorities=medical_priorities,
        medical_priority_probs=medical_priority_probs,
        commercial_priority=commercial_priority,
        num_supply_medical=num_supply_medical,
        num_supply_commercial=num_supply_commercial,
    )
    df = pd.DataFrame([_record_to_dataframe_row(record) for record in records], columns=OUTPUT_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return df.sort_values(["time", "unique_id"]).reset_index(drop=True)


def generate_daily_demand_csv(
    building_file: str = str(BUILDING_DATA_PATH),
    output_file: str = str(DEMAND_EVENTS_PATH),
    **kwargs,
) -> pd.DataFrame:
    df = generate_daily_demand_dataframe(building_file=building_file, **kwargs)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def generate_daily_demand_dataset(
    building_file: str = str(BUILDING_DATA_PATH),
    output_file: str | None = None,
    manifest_file: str | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    records = generate_daily_demand_records(building_file=building_file, **kwargs)
    df = pd.DataFrame([_record_to_dataframe_row(record) for record in records], columns=OUTPUT_COLUMNS)
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    if manifest_file:
        save_event_manifest(records, manifest_file)
    return df, records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a full-day daily_demand_events.csv from building_information.csv."
    )
    parser.add_argument("--input", default=str(BUILDING_DATA_PATH), help="Path to building_information CSV/XLSX")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional legacy CSV projection path for solver compatibility",
    )
    parser.add_argument(
        "--manifest-output",
        default=str(DEMAND_EVENTS_MANIFEST_PATH),
        help="Primary JSONL manifest output with latent/dialogue/extraction priority annotations",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--window-minutes", type=int, default=5, help="Length of each time window in minutes")
    parser.add_argument("--demands-per-window-min", type=int, default=4, help="Minimum demands per window")
    parser.add_argument("--demands-per-window-max", type=int, default=10, help="Maximum demands per window")
    parser.add_argument("--medical-ratio", type=float, default=0.2, help="Fraction of demands routed to medical supplies")
    parser.add_argument("--num-supply-medical", type=int, default=5, help="Number of medical supply points to use")
    parser.add_argument("--num-supply-commercial", type=int, default=5, help="Number of commercial supply points to use")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df, records = generate_daily_demand_dataset(
        building_file=args.input,
        output_file=args.output,
        manifest_file=args.manifest_output,
        seed=args.seed,
        time_window_minutes=args.window_minutes,
        demands_per_window_min=args.demands_per_window_min,
        demands_per_window_max=args.demands_per_window_max,
        medical_ratio=args.medical_ratio,
        num_supply_medical=args.num_supply_medical,
        num_supply_commercial=args.num_supply_commercial,
    )

    if args.output:
        print(f"Legacy CSV saved to {args.output}")
    print(f"Generated {len(df)} demand events")
    if args.manifest_output:
        print(f"Rich event manifest saved to {args.manifest_output} ({len(records)} records)")
    if not df.empty:
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
