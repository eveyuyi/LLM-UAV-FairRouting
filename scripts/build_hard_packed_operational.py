"""Build a packed hard-eval dataset for operational Route A/B runs.

The source ``llm3_grpo_hard.jsonl`` contains contrastive ranking windows with
2-4 demands per window. This script packs adjacent hard windows into larger
operational windows while keeping demand IDs unique inside each packed window
and recomputing window-level priority targets.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Mapping

from llm4fairrouting.data.priority_labels import build_window_priority_targets


DEFAULT_INPUT = Path("data/test/test_seeds/norm_eval/seed_4111/llm3_grpo_hard.jsonl")
DEFAULT_OUTPUT = Path("data/test/test_seeds/norm_eval/seed_4111/llm3_grpo_hard_packed_operational.jsonl")


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _window_start_label(time_window: object, fallback: str) -> str:
    text = str(time_window or "").strip()
    if "T" in text:
        return text.split("::", 1)[0]
    return fallback


def _unique_demands(group: List[dict]) -> List[dict]:
    seen_ids: set[str] = set()
    demands: List[dict] = []
    for source_index, window in enumerate(group):
        source_label = str(window.get("time_window", f"source_{source_index}"))
        for demand_index, demand in enumerate(window.get("demands", [])):
            copied = deepcopy(demand)
            demand_id = str(copied.get("demand_id") or "").strip()
            if not demand_id:
                demand_id = f"packed_missing_id_{source_index}_{demand_index}"
                copied["demand_id"] = demand_id
            if demand_id in seen_ids:
                continue
            seen_ids.add(demand_id)
            copied.setdefault("hard_packed_source_windows", []).append(source_label)
            demands.append(copied)
    return demands


def _packed_window(group: List[dict], pack_index: int) -> dict:
    source_labels = [str(window.get("time_window", "")) for window in group]
    fallback_label = f"hard_packed_window_{pack_index:03d}"
    base_label = _window_start_label(source_labels[0] if source_labels else "", fallback_label)
    time_window = f"{base_label}::packed_operational_{pack_index:03d}"
    demands = _unique_demands(group)
    targets = build_window_priority_targets(demands, priority_field="extraction_observable_priority")
    return {
        "time_window": time_window,
        "dataset_source": "hard_packed_operational",
        "packed_from_windows": source_labels,
        "source_window_count": len(group),
        "demands": demands,
        "priority_labels": targets["demand_configs"],
        "pairwise_preferences": targets["pairwise_preferences"],
        "critical_topk_targets": targets["critical_topk_targets"],
    }


def build_packed_windows(source_windows: List[dict], *, pack_size: int) -> List[dict]:
    if pack_size < 1:
        raise ValueError("pack_size must be >= 1")
    groups = [source_windows[index: index + pack_size] for index in range(0, len(source_windows), pack_size)]
    return [_packed_window(group, pack_index) for pack_index, group in enumerate(groups) if group]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pack-size", type=int, default=2, help="Number of source hard windows per packed window")
    args = parser.parse_args()

    source_windows = _load_jsonl(args.input)
    packed_windows = build_packed_windows(source_windows, pack_size=args.pack_size)
    _write_jsonl(packed_windows, args.output)

    n_source_demands = sum(len(window.get("demands", [])) for window in source_windows)
    n_packed_demands = sum(len(window.get("demands", [])) for window in packed_windows)
    sizes = [len(window.get("demands", [])) for window in packed_windows]
    print(f"source_windows={len(source_windows)} source_demands={n_source_demands}")
    print(f"packed_windows={len(packed_windows)} packed_demands={n_packed_demands}")
    print(f"packed_sizes={sizes}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
