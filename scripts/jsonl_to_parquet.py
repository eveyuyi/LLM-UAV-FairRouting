"""Convert JSONL records to parquet for VERL training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSONL file to parquet.")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL path.")
    parser.add_argument("--output-parquet", required=True, help="Output parquet path.")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at line {line_no}: {exc}") from exc
    return rows


def write_parquet(rows: List[Dict], out_path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required. Install with: pip install pyarrow"
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_parquet).expanduser().resolve()
    rows = load_jsonl(input_path)
    write_parquet(rows, output_path)
    print(f"rows={len(rows)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
