"""Build the canonical LLM-generated dialogue dataset aligned with seed demand events."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from llm4fairrouting.config.runtime_env import (
    env_float,
    env_int,
    env_int_list,
    env_optional_int,
    env_text,
    prepare_env_file,
)
from llm4fairrouting.data.seed_paths import (
    DEMAND_DIALOGUES_PATH,
    PRIMARY_EVENT_DATA_PATH,
    STATION_DATA_PATH,
)
from llm4fairrouting.llm.client_utils import create_openai_client
from llm4fairrouting.llm.dialogue_generation import (
    generate_dialogues_online,
    load_demand_events,
    load_stations,
    save_dialogues,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_daily_demand_dialogues(
    *,
    events_path: str = str(PRIMARY_EVENT_DATA_PATH),
    stations_path: str | None = str(STATION_DATA_PATH),
    output_path: str = str(DEMAND_DIALOGUES_PATH),
    api_base: str | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    base_date: str = "2024-03-15",
    n_events: int | None = None,
    time_slots: list[int] | None = None,
    temperature: float = 0.2,
    batch_size: int = 5,
    styles: list[str] | None = None,
    max_concurrency: int = 1,
) -> list[dict]:
    """Materialize one canonical dialogue dataset from the seed demand events."""
    started_at = time.perf_counter()
    print(f"[Module 1] Building canonical dialogue dataset from {events_path}", flush=True)
    client = create_openai_client(api_base, api_key)
    stations = load_stations(stations_path) if stations_path else []
    print(f"[Module 1] Loaded {len(stations)} stations", flush=True)
    events = load_demand_events(events_path, n_events=n_events, time_slots=time_slots)
    if not events:
        raise ValueError(f"No demand events could be loaded from {events_path}")
    print(
        f"[Module 1] Loaded {len(events)} seed demand events "
        f"(time_slots={time_slots}, n_events={n_events})",
        flush=True,
    )

    dialogues = generate_dialogues_online(
        demand_events=events,
        stations=stations,
        client=client,
        model=model,
        base_date=base_date,
        temperature=temperature,
        batch_size=batch_size,
        styles=styles,
        max_concurrency=max_concurrency,
    )
    save_dialogues(dialogues, output_path)
    print(
        f"[Module 1] Canonical dialogue dataset ready in "
        f"{time.perf_counter() - started_at:.1f}s",
        flush=True,
    )
    return dialogues


def main() -> None:
    active_env_file = prepare_env_file(PROJECT_ROOT)
    parser = argparse.ArgumentParser(
        description="Build data/seed/daily_demand_dialogues.jsonl from the rich event manifest with an LLM.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(active_env_file) if active_env_file else None,
        help="Environment file path; defaults to the project .env when present",
    )
    parser.add_argument(
        "--events",
        default=env_text("LLM4FAIRROUTING_EVENTS", str(PRIMARY_EVENT_DATA_PATH)),
        help="Path to the rich event manifest JSONL",
    )
    parser.add_argument(
        "--stations",
        default=env_text("LLM4FAIRROUTING_STATIONS", str(STATION_DATA_PATH)),
        help="Optional station metadata file",
    )
    parser.add_argument(
        "--output",
        default=env_text("LLM4FAIRROUTING_DIALOGUES", str(DEMAND_DIALOGUES_PATH)),
        help="Output JSONL path",
    )
    parser.add_argument("--api-base", default=env_text("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=env_text("OPENAI_API_KEY"))
    parser.add_argument("--model", default=env_text("LLM4FAIRROUTING_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-date", default=env_text("LLM4FAIRROUTING_BASE_DATE", "2024-03-15"))
    parser.add_argument("--n-events", type=int, default=env_optional_int("LLM4FAIRROUTING_N_EVENTS"))
    parser.add_argument("--time-slots", type=int, nargs="+", default=env_int_list("LLM4FAIRROUTING_TIME_SLOTS"))
    parser.add_argument("--temperature", type=float, default=env_float("LLM4FAIRROUTING_TEMPERATURE", 0.2))
    parser.add_argument("--batch-size", type=int, default=env_int("LLM4FAIRROUTING_BATCH_SIZE", 5))
    parser.add_argument("--max-concurrency", type=int, default=env_int("LLM4FAIRROUTING_MAX_CONCURRENCY", 1))
    parser.add_argument("--styles", nargs="+", default=None)
    args = parser.parse_args()

    dialogues = build_daily_demand_dialogues(
        events_path=args.events,
        stations_path=args.stations,
        output_path=args.output,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        base_date=args.base_date,
        n_events=args.n_events,
        time_slots=args.time_slots,
        temperature=args.temperature,
        batch_size=args.batch_size,
        styles=args.styles,
        max_concurrency=args.max_concurrency,
    )
    print(f"Built {len(dialogues)} canonical seed dialogues at {args.output}", flush=True)


if __name__ == "__main__":
    main()
