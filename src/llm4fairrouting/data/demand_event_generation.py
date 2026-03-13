"""Seed demand-event generation helpers and CLI."""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from llm4fairrouting.data.building_information import (
    HEALTHCARE_LAND_USE,
    RESIDENTIAL_LAND_USE,
    load_building_data,
)
from llm4fairrouting.data.seed_paths import BUILDING_DATA_PATH, DEMAND_EVENTS_PATH
from llm4fairrouting.routing.domain import DemandEvent, Point

COMMERCIAL_LAND_USE = "commercial_service_land"
MEDICAL_SUPPLY_TYPE = "medical"
COMMERCIAL_SUPPLY_TYPE = "commercial"
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

    all_events: list[DemandEvent] = []
    window_duration_hours = time_window_minutes / 60.0
    num_windows = 24 * 60 // time_window_minutes

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

        all_events.extend(generate_demand_events(
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
        ))

    df = _events_to_dataframe(all_events, demand_points, supply_points)
    df = df.sort_values(["time", "unique_id"]).reset_index(drop=True)
    return df


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a full-day daily_demand_events.csv from building_information.csv."
    )
    parser.add_argument("--input", default=str(BUILDING_DATA_PATH), help="Path to building_information CSV/XLSX")
    parser.add_argument("--output", default=str(DEMAND_EVENTS_PATH), help="Path to output daily_demand_events.csv")
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
    df = generate_daily_demand_csv(
        building_file=args.input,
        output_file=args.output,
        seed=args.seed,
        time_window_minutes=args.window_minutes,
        demands_per_window_min=args.demands_per_window_min,
        demands_per_window_max=args.demands_per_window_max,
        medical_ratio=args.medical_ratio,
        num_supply_medical=args.num_supply_medical,
        num_supply_commercial=args.num_supply_commercial,
    )

    print(f"Demand events saved to {args.output}")
    print(f"Generated {len(df)} demand events")
    if not df.empty:
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
