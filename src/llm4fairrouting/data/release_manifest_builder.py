"""Aggregate per-shard release manifests into a dataset-level release plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


def _iter_shard_dirs(root_dir: Path) -> Iterable[Path]:
    for child in sorted(root_dir.iterdir()):
        if child.is_dir() and (child / "release_manifest.json").exists():
            yield child


def build_aggregate_release_manifest(root_dir: str | Path) -> Dict[str, object]:
    root_dir = Path(root_dir)
    status_counts: Counter = Counter()
    shards_by_status: dict[str, List[str]] = defaultdict(list)
    recommended_files: dict[str, List[str]] = defaultdict(list)
    aggregate_counts: Counter = Counter()

    for shard_dir in _iter_shard_dirs(root_dir):
        release_manifest = json.loads((shard_dir / "release_manifest.json").read_text(encoding="utf-8"))
        dataset_manifest = json.loads((shard_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
        status = str(release_manifest.get("release_status", "debug_only"))
        status_counts[status] += 1
        shards_by_status[status].append(shard_dir.name)
        for key in release_manifest.get("recommended_training_files", []) or []:
            relative_file = dataset_manifest.get("files", {}).get(key)
            if relative_file:
                recommended_files[key].append(str((shard_dir / relative_file).resolve()))
                aggregate_counts[key] += int(dataset_manifest.get("counts", {}).get(key, 0) or 0)

    return {
        "schema_version": "priority_aggregate_release_manifest_v1",
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "root_dir": str(root_dir.resolve()),
        "status_counts": dict(status_counts),
        "shards_by_status": dict(shards_by_status),
        "recommended_training_files": dict(recommended_files),
        "aggregate_counts": dict(aggregate_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate shard release manifests into a dataset-level release plan.",
    )
    parser.add_argument("root_dir", help="Directory containing per-shard output subdirectories")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path; defaults to <root_dir>/aggregate_release_manifest.json",
    )
    args = parser.parse_args()

    manifest = build_aggregate_release_manifest(args.root_dir)
    output_path = Path(args.output) if args.output else Path(args.root_dir) / "aggregate_release_manifest.json"
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote aggregate release manifest to {output_path}")


if __name__ == "__main__":
    main()
