#!/usr/bin/env python
"""Sample a fixed overlap test set from llm_selection raw or compact exports."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a deterministic overlap test set from llm_selection JSONL."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Source llm_selection JSONL file.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Output JSONL path for the sampled test set.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of records to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--dedupe-record-id",
        action="store_true",
        help="Keep only the first occurrence of each record_id before sampling. Recommended for oversampled compact exports.",
    )
    return parser.parse_args()


def load_records(path: Path) -> tuple[List[Dict], int]:
    records: List[Dict] = []
    malformed_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                malformed_lines += 1
    return records, malformed_lines


def dedupe_by_record_id(records: List[Dict]) -> Tuple[List[Dict], int]:
    seen = set()
    deduped: List[Dict] = []
    duplicates = 0
    for record in records:
        record_id = str(record.get("record_id") or "").strip()
        key = record_id if record_id else json.dumps(record, ensure_ascii=False, sort_keys=True)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduped.append(record)
    return deduped, duplicates


def record_profile(record: Dict) -> Dict[str, str]:
    target = record.get("selection_target") or {}
    labels = target.get("training_labels") or {}
    return {
        "scene_type": str(labels.get("scene_type") or ""),
        "selection_mode": str(target.get("selection_mode") or ""),
        "decision_difficulty": str(labels.get("decision_difficulty") or ""),
    }


def build_summary(
    records: List[Dict],
    sample_size: int,
    seed: int,
    input_jsonl: Path,
    output_jsonl: Path,
    malformed_lines: int,
    dedupe_enabled: bool,
    duplicate_records_skipped: int,
) -> Dict:
    scene_counter = Counter()
    mode_counter = Counter()
    difficulty_counter = Counter()
    record_ids: List[str] = []

    for record in records:
        profile = record_profile(record)
        scene_counter[profile["scene_type"] or ""] += 1
        mode_counter[profile["selection_mode"] or ""] += 1
        difficulty_counter[profile["decision_difficulty"] or ""] += 1
        record_ids.append(str(record.get("record_id") or ""))

    return {
        "input_jsonl": str(input_jsonl.resolve()),
        "output_jsonl": str(output_jsonl.resolve()),
        "sample_size": sample_size,
        "seed": seed,
        "malformed_input_lines_skipped": malformed_lines,
        "dedupe_record_id": dedupe_enabled,
        "duplicate_records_skipped": duplicate_records_skipped,
        "scene_type_counts": dict(scene_counter),
        "selection_mode_counts": dict(mode_counter),
        "decision_difficulty_counts": dict(difficulty_counter),
        "record_ids": record_ids,
    }


def main() -> None:
    args = parse_args()
    input_jsonl = args.input_jsonl.expanduser().resolve()
    output_jsonl = args.output_jsonl.expanduser().resolve()

    records, malformed_lines = load_records(input_jsonl)
    duplicate_records_skipped = 0
    if args.dedupe_record_id:
        records, duplicate_records_skipped = dedupe_by_record_id(records)
    if args.sample_size < 1:
        raise SystemExit("--sample-size must be >= 1")
    if len(records) < args.sample_size:
        raise SystemExit(
            f"Requested sample_size={args.sample_size}, but source only has {len(records)} records."
        )

    rng = random.Random(args.seed)
    sampled_indices = sorted(rng.sample(range(len(records)), args.sample_size))
    sampled_records = [records[idx] for idx in sampled_indices]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in sampled_records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    summary = build_summary(
        sampled_records,
        sample_size=args.sample_size,
        seed=args.seed,
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        malformed_lines=malformed_lines,
        dedupe_enabled=bool(args.dedupe_record_id),
        duplicate_records_skipped=duplicate_records_skipped,
    )
    summary_path = output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"sampled_records={len(sampled_records)}")
    print(f"valid_input_records={len(records)}")
    print(f"malformed_input_lines_skipped={malformed_lines}")
    print(f"duplicate_records_skipped={duplicate_records_skipped}")
    print(f"output_jsonl={output_jsonl}")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
