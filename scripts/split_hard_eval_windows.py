from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


SUBSET_ORDER = (
    "counterfactual",
    "surface_contradiction",
    "near_tie",
    "mixed_priority",
    "other",
)


def _load_jsonl(path: Path) -> List[Dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify_hard_window(time_window: str) -> str:
    label = str(time_window or "")
    if "surface_contradiction" in label:
        return "surface_contradiction"
    if "near_tie" in label:
        return "near_tie"
    if "counterfactual" in label:
        return "counterfactual"
    if "mixed_priority" in label:
        return "mixed_priority"
    return "other"


def _iter_dataset_dirs(root: Path, source_filename: str) -> Iterable[Path]:
    if (root / source_filename).is_file():
        yield root
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / source_filename).is_file():
            yield child


def split_dataset_dir(
    dataset_dir: Path,
    *,
    source_filename: str = "llm3_grpo_hard.jsonl",
    output_subdir: str = "hard_eval_subsets",
) -> Dict[str, object]:
    source_path = dataset_dir / source_filename
    rows = _load_jsonl(source_path)

    buckets: Dict[str, List[Dict]] = {subset: [] for subset in SUBSET_ORDER}
    for row in rows:
        subset = classify_hard_window(str(row.get("time_window", "")))
        buckets.setdefault(subset, []).append(row)

    subset_dir = dataset_dir / output_subdir
    files = {}
    counts = {}
    for subset in SUBSET_ORDER:
        output_path = subset_dir / f"{subset}.jsonl"
        _write_jsonl(output_path, buckets[subset])
        files[subset] = str(output_path)
        counts[subset] = len(buckets[subset])

    manifest = {
        "dataset_dir": str(dataset_dir),
        "source_file": str(source_path),
        "output_dir": str(subset_dir),
        "counts": counts,
        "files": files,
        "total_windows": len(rows),
    }
    manifest_path = subset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split llm3_grpo_hard.jsonl into subtype-specific hard-eval JSONL files.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="A shard directory containing llm3_grpo_hard.jsonl, or a root directory containing seed_* shard dirs.",
    )
    parser.add_argument(
        "--source-filename",
        default="llm3_grpo_hard.jsonl",
        help="Name of the hard-window source file inside each dataset dir.",
    )
    parser.add_argument(
        "--output-subdir",
        default="hard_eval_subsets",
        help="Subdirectory name used to store the split JSONL files.",
    )
    args = parser.parse_args()

    root = args.path.resolve()
    results = [
        split_dataset_dir(
            dataset_dir,
            source_filename=args.source_filename,
            output_subdir=args.output_subdir,
        )
        for dataset_dir in _iter_dataset_dirs(root, args.source_filename)
    ]

    if not results:
        raise SystemExit(f"No dataset directories with {args.source_filename} found under {root}")

    if len(results) > 1:
        aggregate = {
            "root": str(root),
            "source_filename": args.source_filename,
            "output_subdir": args.output_subdir,
            "datasets": results,
        }
        aggregate_path = root / args.output_subdir / "aggregate_manifest.json"
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
