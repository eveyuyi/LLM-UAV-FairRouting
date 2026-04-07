"""Repair generated training shards after priority-policy updates.

This utility avoids regenerating raw LLM dialogues. It rebuilds derived labels
and release artifacts from the existing shard contents:

- restore simulator latent priority on dialogues
- recompute dialogue audit outputs
- rebuild llm2/clean/pipeline/hard training files with current policy
- refresh quality_report.json, release_manifest.json, and dataset_manifest.json
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from llm4fairrouting.data.event_data import load_event_records
from llm4fairrouting.data.event_structuring import build_gold_structured_demand
from llm4fairrouting.data.priority_labels import attach_priority_labels, build_window_priority_targets
from llm4fairrouting.data.quality_gates import (
    DEFAULT_QUALITY_THRESHOLDS,
    build_quality_report,
    build_release_manifest,
    filter_accepted_dialogues,
)
from llm4fairrouting.data.training_dataset_builder import (
    TRAINING_OUTPUT_FILENAMES,
    _build_clean_structured_windows,
    _build_hard_contrastive_windows,
    _build_llm2_sft_records,
)
from llm4fairrouting.llm.dialogue_generation import audit_dialogue


def _load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_dataset_dirs(root: Path) -> Iterable[Path]:
    if (root / "dataset_manifest.json").is_file():
        yield root
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "dataset_manifest.json").is_file():
            yield child


def _repair_dialogues(dialogues: List[Dict], event_by_id: Dict[str, Dict]) -> List[Dict]:
    repaired: List[Dict] = []
    for dialogue in dialogues:
        working = deepcopy(dialogue)
        annotations = dict(working.get("annotations", {}) or {})
        metadata = dict(working.get("metadata", {}) or {})
        source_event_id = str(metadata.get("event_id", "")).strip()
        event_record = event_by_id.get(source_event_id)

        true_latent = annotations.get("seed_latent_priority")
        if true_latent is None and event_record is not None:
            true_latent = event_record.get("latent_priority")
        if true_latent is None:
            true_latent = annotations.get("latent_priority", 4)
        true_latent = int(true_latent or 4)

        annotations["latent_priority"] = true_latent
        annotations["seed_latent_priority"] = true_latent

        if event_record is not None:
            gold = build_gold_structured_demand(event_record)
        else:
            gold = deepcopy(annotations.get("gold_structured_demand", {}) or {})
            attach_priority_labels(
                gold,
                latent_priority=true_latent,
                dialogue_observable_priority=(working.get("audit", {}) or {}).get("dialogue_observable_priority"),
            )
            gold["latent_priority"] = true_latent

        annotations["gold_structured_demand"] = gold
        working["annotations"] = annotations
        working["audit"] = audit_dialogue(working)

        refreshed_gold = deepcopy(working["annotations"]["gold_structured_demand"])
        attach_priority_labels(
            refreshed_gold,
            latent_priority=true_latent,
            dialogue_observable_priority=working["audit"].get("dialogue_observable_priority"),
        )
        refreshed_gold["latent_priority"] = true_latent
        working["annotations"]["gold_structured_demand"] = refreshed_gold
        repaired.append(working)
    return repaired


def _repair_pipeline_windows(pipeline_windows: List[Dict], event_by_id: Dict[str, Dict]) -> List[Dict]:
    repaired_windows: List[Dict] = []
    for window in pipeline_windows:
        repaired_window = deepcopy(window)
        repaired_demands: List[Dict] = []
        for demand in repaired_window.get("demands", []) or []:
            repaired_demand = deepcopy(demand)
            source_event_id = str(repaired_demand.get("source_event_id", "")).strip()
            event_record = event_by_id.get(source_event_id)
            true_latent = None
            if event_record is not None:
                true_latent = event_record.get("latent_priority")
            if true_latent is None:
                true_latent = (
                    repaired_demand.get("labels", {}) or {}
                ).get("latent_priority", repaired_demand.get("latent_priority", 4))
            true_latent = int(true_latent or 4)
            attach_priority_labels(
                repaired_demand,
                latent_priority=true_latent,
                dialogue_observable_priority=(repaired_demand.get("labels", {}) or {}).get("dialogue_observable_priority"),
            )
            repaired_demand["latent_priority"] = true_latent
            repaired_demands.append(repaired_demand)

        repaired_window["demands"] = repaired_demands
        targets = build_window_priority_targets(repaired_demands, priority_field="extraction_observable_priority")
        repaired_window["priority_labels"] = targets["demand_configs"]
        repaired_window["pairwise_preferences"] = targets["pairwise_preferences"]
        repaired_window["critical_topk_targets"] = targets["critical_topk_targets"]
        repaired_windows.append(repaired_window)
    return repaired_windows


def repair_dataset_dir(dataset_dir: Path, *, dry_run: bool = False) -> Dict[str, object]:
    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    files = dict(dataset_manifest.get("files", {}) or {})

    event_records = load_event_records(dataset_dir / files["events_manifest"])
    event_by_id = {
        str(record.get("event_id", "")).strip(): record
        for record in event_records
        if str(record.get("event_id", "")).strip()
    }
    dialogues = _load_jsonl(dataset_dir / files["dialogues"])
    pipeline_windows = _load_jsonl(dataset_dir / files["llm3_sft_pipeline"])

    repaired_dialogues = _repair_dialogues(dialogues, event_by_id)
    accepted_dialogues = filter_accepted_dialogues(repaired_dialogues)
    llm2_sft = _build_llm2_sft_records(accepted_dialogues)
    clean_windows = _build_clean_structured_windows(
        event_records,
        base_date=str(dataset_manifest.get("base_date", "2024-03-15")),
        window_minutes=int(dataset_manifest.get("window_minutes", 5) or 5),
    )
    repaired_pipeline_windows = _repair_pipeline_windows(pipeline_windows, event_by_id)
    hard_windows = _build_hard_contrastive_windows(clean_windows)

    quality_thresholds = {
        **DEFAULT_QUALITY_THRESHOLDS,
        **dict(dataset_manifest.get("quality_thresholds", {}) or {}),
    }
    quality_report = build_quality_report(
        event_records=event_records,
        dialogues=repaired_dialogues,
        pipeline_windows=repaired_pipeline_windows,
        hard_windows=hard_windows,
        thresholds=quality_thresholds,
    )

    counts = {
        "events_manifest": len(event_records),
        "dialogues": len(repaired_dialogues),
        "llm2_sft": len(llm2_sft),
        "llm3_sft_clean": len(clean_windows),
        "llm3_sft_pipeline": len(repaired_pipeline_windows),
        "llm3_grpo_hard": len(hard_windows),
    }
    dataset_manifest["generated_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    dataset_manifest["counts"] = counts
    dataset_manifest["raw_counts"] = {
        "dialogues": len(repaired_dialogues),
        "accepted_dialogues": len(accepted_dialogues),
    }
    dataset_manifest["quality_thresholds"] = quality_thresholds

    release_manifest = build_release_manifest(
        dataset_manifest=dataset_manifest,
        quality_report=quality_report,
    )
    dataset_manifest["release_status"] = release_manifest["release_status"]

    if not dry_run:
        _write_jsonl(repaired_dialogues, dataset_dir / files["dialogues"])
        _write_jsonl(llm2_sft, dataset_dir / files["llm2_sft"])
        _write_jsonl(clean_windows, dataset_dir / files["llm3_sft_clean"])
        _write_jsonl(repaired_pipeline_windows, dataset_dir / files["llm3_sft_pipeline"])
        _write_jsonl(hard_windows, dataset_dir / files["llm3_grpo_hard"])
        (dataset_dir / files["quality_report"]).write_text(
            json.dumps(quality_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (dataset_dir / files["release_manifest"]).write_text(
            json.dumps(release_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        dataset_manifest_path.write_text(
            json.dumps(dataset_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "dataset_dir": str(dataset_dir),
        "release_status": release_manifest["release_status"],
        "failed_checks": list(release_manifest.get("failed_checks", [])),
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair training shards after priority-policy changes without regenerating dialogues.",
    )
    parser.add_argument(
        "path",
        help="A shard directory containing dataset_manifest.json, or a root directory containing seed_* shard dirs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.path)
    results = [repair_dataset_dir(dataset_dir, dry_run=args.dry_run) for dataset_dir in _iter_dataset_dirs(root)]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
