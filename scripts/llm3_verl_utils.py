"""Shared helpers for exporting LLM3 data into VERL parquet datasets."""

from __future__ import annotations

import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4fairrouting.llm.ranking_prompt_utils import render_priority_ranking_prompt

ANSWER_LEAK_KEYS = {
    "labels",
    "gold_extraction",
    "latent_priority",
    "dialogue_observable_priority",
    "extraction_observable_priority",
    "extraction_observable_score",
    "extraction_observable_reasoning",
    "solver_useful_priority",
    "solver_useful_score",
    "solver_useful_reasoning",
    "priority_policy_version",
    "priority_labels",
    "pairwise_preferences",
    "critical_topk_targets",
    "demand_tier",
    "urgency",
}

SYSTEM_PROMPT = (
    "You are a priority-ranking assistant for drone delivery. "
    "Return JSON only and do not use Markdown."
)


def load_jsonl(path: str | Path) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _sanitize_value(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, nested_value in value.items():
            if key in ANSWER_LEAK_KEYS:
                continue
            cleaned[key] = _sanitize_value(nested_value)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def sanitize_demands(demands: Sequence[Dict]) -> List[Dict]:
    return [_sanitize_value(deepcopy(demand)) for demand in demands]


def build_prompt_text(time_window: str, demands: Sequence[Dict]) -> str:
    return render_priority_ranking_prompt(sanitize_demands(demands), time_window=time_window)


def build_response_text(priority_labels: Sequence[Dict]) -> str:
    response_payload = {
        "priority_labels": [
            {
                "demand_id": str(label.get("demand_id", "")),
                "priority": int(label.get("priority", 4)),
                "window_rank": int(label.get("window_rank", 1)),
                "reasoning": str(label.get("reasoning", "")),
            }
            for label in priority_labels
        ]
    }
    return json.dumps(response_payload, ensure_ascii=False, indent=2)


def shuffle_split(records: Sequence[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0).")

    if len(items) <= 1 or val_ratio == 0.0:
        return items, []

    val_size = max(1, int(round(len(items) * val_ratio)))
    val_size = min(val_size, len(items) - 1)
    return items[val_size:], items[:val_size]


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_parquet(records: Iterable[Dict], output_path: str | Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to write parquet files. "
            "Install it in your training environment with `pip install pyarrow`."
        ) from exc

    rows = list(records)
    ensure_parent_dir(output_path)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)


def collect_input_dirs(input_dirs: Sequence[str]) -> List[Path]:
    resolved = [Path(path).expanduser().resolve() for path in input_dirs]
    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input directories: {missing}")
    return resolved


def summarize_records(records: Sequence[Dict]) -> Dict[str, int]:
    by_source: Dict[str, int] = {}
    for record in records:
        source = str(record.get("data_source") or record.get("source_dataset") or "unknown")
        by_source[source] = by_source.get(source, 0) + 1
    return by_source
