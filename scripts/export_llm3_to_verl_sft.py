"""Export LLM3 SFT JSONL windows into VERL-compatible parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from llm3_verl_utils import (
    SYSTEM_PROMPT,
    build_prompt_text,
    build_response_text,
    collect_input_dirs,
    load_jsonl,
    shuffle_split,
    summarize_records,
    write_parquet,
)

SFT_FILE_MAP = {
    "clean": ("llm3_sft_clean.jsonl", "llm3_sft_clean"),
    "pipeline": ("llm3_sft_pipeline.jsonl", "llm3_sft_pipeline"),
}


def build_sft_rows(input_dirs: List[Path], sources: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for input_dir in input_dirs:
        for source_name in sources:
            filename, source_dataset = SFT_FILE_MAP[source_name]
            file_path = input_dir / filename
            if not file_path.exists():
                continue

            for sample in load_jsonl(file_path):
                prompt_text = build_prompt_text(
                    time_window=str(sample.get("time_window", "")),
                    demands=sample.get("demands", []),
                )
                response_text = build_response_text(sample.get("priority_labels", []))
                rows.append(
                    {
                        "data_source": source_dataset,
                        "ability": "priority_ranking",
                        "prompt": prompt_text,
                        "response": response_text,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt_text},
                            {"role": "assistant", "content": response_text},
                        ],
                        "extra_info": {
                            "time_window": sample.get("time_window"),
                            "dataset_source": sample.get("dataset_source"),
                            "num_demands": len(sample.get("demands", [])),
                            "input_dir": str(input_dir),
                        },
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export llm3_sft_clean/pipeline JSONL files into VERL SFT parquet files.",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Directory that contains llm3_sft_clean.jsonl and/or llm3_sft_pipeline.jsonl. Repeat this flag for multiple shards.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(SFT_FILE_MAP),
        default=["clean", "pipeline"],
        help="Which SFT sources to include.",
    )
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dirs = collect_input_dirs(args.input_dir)
    rows = build_sft_rows(input_dirs=input_dirs, sources=args.sources)
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
