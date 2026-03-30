"""Export LLM3 GRPO JSONL windows into VERL-compatible parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from llm3_verl_utils import (
    build_prompt_text,
    collect_input_dirs,
    load_jsonl,
    shuffle_split,
    summarize_records,
    write_parquet,
)

GRPO_FILENAME = "llm3_grpo_hard.jsonl"
DATA_SOURCE = "llm3_priority_window"


def build_grpo_rows(input_dirs: List[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for input_dir in input_dirs:
        file_path = input_dir / GRPO_FILENAME
        if not file_path.exists():
            continue

        for sample in load_jsonl(file_path):
            prompt_text = build_prompt_text(
                time_window=str(sample.get("time_window", "")),
                demands=sample.get("demands", []),
            )
            demand_ids = [
                str(demand.get("demand_id", ""))
                for demand in sample.get("demands", [])
            ]
            rows.append(
                {
                    "data_source": DATA_SOURCE,
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "ability": "priority_ranking",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": {
                            "priority_labels": sample.get("priority_labels", []),
                            "pairwise_preferences": sample.get("pairwise_preferences", []),
                            "critical_topk_targets": sample.get("critical_topk_targets", []),
                        },
                    },
                    "extra_info": {
                        "time_window": sample.get("time_window"),
                        "dataset_source": sample.get("dataset_source"),
                        "num_demands": len(sample.get("demands", [])),
                        "demand_ids": demand_ids,
                        "input_dir": str(input_dir),
                    },
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export llm3_grpo_hard JSONL files into VERL RL parquet files.",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Directory that contains llm3_grpo_hard.jsonl. Repeat this flag for multiple shards.",
    )
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dirs = collect_input_dirs(args.input_dir)
    rows = build_grpo_rows(input_dirs=input_dirs)
    train_rows, val_rows = shuffle_split(rows, val_ratio=args.val_ratio, seed=args.seed)

    summary = {
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "source_counts": summarize_records(rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    write_parquet(train_rows, args.train_out)
    write_parquet(val_rows, args.val_out)
    print(f"Wrote train parquet: {Path(args.train_out).resolve()}")
    print(f"Wrote val parquet:   {Path(args.val_out).resolve()}")


if __name__ == "__main__":
    main()
