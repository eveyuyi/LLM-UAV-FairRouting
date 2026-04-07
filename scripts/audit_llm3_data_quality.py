"""Audit LLM3 pipeline training windows for nulls, schema drift, and label consistency."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm3_verl_utils import load_jsonl
from llm4fairrouting.data.priority_labels import build_window_priority_targets


def _resolve_dirs(patterns: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            resolved.extend(Path(match).expanduser().resolve() for match in matches)
        else:
            candidate = Path(pattern).expanduser().resolve()
            if candidate.exists():
                resolved.append(candidate)
    deduped: List[Path] = []
    seen = set()
    for path in resolved:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def _record_example(examples: Dict[str, List[Dict]], issue: str, payload: Dict, limit: int) -> None:
    bucket = examples[issue]
    if len(bucket) < limit:
        bucket.append(payload)


def _priority_map(priority_labels: Sequence[Dict]) -> Dict[str, Dict]:
    mapping: Dict[str, Dict] = {}
    for row in priority_labels:
        demand_id = str(row.get("demand_id", "")).strip()
        if demand_id:
            mapping[demand_id] = row
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit llm3_sft_pipeline.jsonl files for nulls and label consistency.",
    )
    parser.add_argument("--input-dir", action="append", default=[], help="Directory or glob for seed dirs.")
    parser.add_argument(
        "--filename",
        default="llm3_sft_pipeline.jsonl",
        help="Target filename inside each input dir.",
    )
    parser.add_argument("--urgent-threshold", type=int, default=2)
    parser.add_argument("--example-limit", type=int, default=5)
    parser.add_argument("--json-out", help="Optional path to write the full JSON report.")
    args = parser.parse_args()

    input_dirs = _resolve_dirs(args.input_dir)
    if not input_dirs:
        raise SystemExit("No input dirs found. Pass one or more --input-dir values or globs.")

    counts = Counter()
    null_counts = Counter()
    type_counts = Counter()
    issue_counts = Counter()
    examples: Dict[str, List[Dict]] = defaultdict(list)

    for input_dir in input_dirs:
        file_path = input_dir / args.filename
        if not file_path.exists():
            issue_counts["missing_pipeline_file"] += 1
            _record_example(
                examples,
                "missing_pipeline_file",
                {"input_dir": str(input_dir), "filename": args.filename},
                args.example_limit,
            )
            continue

        for line_no, sample in enumerate(load_jsonl(file_path), start=1):
            counts["windows"] += 1
            base_info = {
                "input_dir": str(input_dir),
                "line": line_no,
                "time_window": str(sample.get("time_window", "")),
            }

            demands = sample.get("demands", [])
            priority_labels = sample.get("priority_labels", [])
            pairwise_preferences = sample.get("pairwise_preferences", [])
            critical_topk_targets = sample.get("critical_topk_targets", [])

            if not isinstance(demands, list):
                issue_counts["demands_not_list"] += 1
                _record_example(examples, "demands_not_list", {**base_info, "value_type": type(demands).__name__}, args.example_limit)
                continue
            if not isinstance(priority_labels, list):
                issue_counts["priority_labels_not_list"] += 1
                _record_example(
                    examples,
                    "priority_labels_not_list",
                    {**base_info, "value_type": type(priority_labels).__name__},
                    args.example_limit,
                )
                continue
            if not isinstance(pairwise_preferences, list):
                issue_counts["pairwise_preferences_not_list"] += 1
            if not isinstance(critical_topk_targets, list):
                issue_counts["critical_topk_targets_not_list"] += 1

            counts["demands"] += len(demands)
            counts["priority_labels"] += len(priority_labels)
            counts["pairwise_preferences"] += len(pairwise_preferences) if isinstance(pairwise_preferences, list) else 0
            counts["critical_topk_targets"] += len(critical_topk_targets) if isinstance(critical_topk_targets, list) else 0

            demand_ids: List[str] = []
            for idx, demand in enumerate(demands):
                if not isinstance(demand, dict):
                    issue_counts["demand_not_dict"] += 1
                    _record_example(
                        examples,
                        "demand_not_dict",
                        {**base_info, "demand_index": idx, "value_type": type(demand).__name__},
                        args.example_limit,
                    )
                    continue
                demand_id = str(demand.get("demand_id", "")).strip()
                if not demand_id:
                    issue_counts["missing_demand_id"] += 1
                    _record_example(examples, "missing_demand_id", {**base_info, "demand_index": idx}, args.example_limit)
                else:
                    demand_ids.append(demand_id)

                context_signals = demand.get("context_signals")
                if context_signals is None:
                    null_counts["context_signals_none"] += 1
                elif not isinstance(context_signals, list):
                    type_counts["context_signals_not_list"] += 1

                signals = demand.get("priority_evaluation_signals")
                if signals is None:
                    null_counts["priority_evaluation_signals_none"] += 1
                    signals = {}
                elif not isinstance(signals, dict):
                    type_counts["priority_evaluation_signals_not_dict"] += 1
                    signals = {}

                special_handling = signals.get("special_handling")
                if special_handling is None:
                    null_counts["special_handling_none"] += 1
                elif not isinstance(special_handling, list):
                    type_counts["special_handling_not_list"] += 1

                vulnerability = signals.get("population_vulnerability")
                if vulnerability is None:
                    null_counts["population_vulnerability_none"] += 1
                elif not isinstance(vulnerability, dict):
                    type_counts["population_vulnerability_not_dict"] += 1

            if len(demand_ids) != len(set(demand_ids)):
                issue_counts["duplicate_demand_id"] += 1
                _record_example(
                    examples,
                    "duplicate_demand_id",
                    {**base_info, "demand_ids": demand_ids},
                    args.example_limit,
                )

            label_map = _priority_map(priority_labels)
            demand_id_set = set(demand_ids)
            label_id_set = set(label_map)

            if label_id_set != demand_id_set:
                issue_counts["priority_label_id_set_mismatch"] += 1
                _record_example(
                    examples,
                    "priority_label_id_set_mismatch",
                    {
                        **base_info,
                        "missing_from_labels": sorted(demand_id_set - label_id_set),
                        "extra_in_labels": sorted(label_id_set - demand_id_set),
                    },
                    args.example_limit,
                )

            ranks = []
            for row in priority_labels:
                demand_id = str(row.get("demand_id", "")).strip()
                priority = row.get("priority")
                rank = row.get("window_rank")
                if priority not in (1, 2, 3, 4):
                    issue_counts["invalid_priority_value"] += 1
                    _record_example(
                        examples,
                        "invalid_priority_value",
                        {**base_info, "demand_id": demand_id, "priority": priority},
                        args.example_limit,
                    )
                try:
                    ranks.append(int(rank))
                except (TypeError, ValueError):
                    issue_counts["invalid_window_rank"] += 1
                    _record_example(
                        examples,
                        "invalid_window_rank",
                        {**base_info, "demand_id": demand_id, "window_rank": rank},
                        args.example_limit,
                    )

            if ranks:
                expected_ranks = list(range(1, len(priority_labels) + 1))
                if sorted(ranks) != expected_ranks:
                    issue_counts["window_ranks_not_contiguous"] += 1
                    _record_example(
                        examples,
                        "window_ranks_not_contiguous",
                        {**base_info, "window_ranks": sorted(ranks), "expected": expected_ranks},
                        args.example_limit,
                    )

            if isinstance(pairwise_preferences, list):
                for pair in pairwise_preferences:
                    higher_id = str(pair.get("higher_priority_demand_id", "")).strip()
                    lower_id = str(pair.get("lower_priority_demand_id", "")).strip()
                    if higher_id not in demand_id_set or lower_id not in demand_id_set:
                        issue_counts["pairwise_unknown_demand_id"] += 1
                        _record_example(
                            examples,
                            "pairwise_unknown_demand_id",
                            {**base_info, "pair": pair},
                            args.example_limit,
                        )
                        continue
                    higher = label_map.get(higher_id)
                    lower = label_map.get(lower_id)
                    if not higher or not lower:
                        continue
                    higher_priority = int(higher.get("priority", 4))
                    lower_priority = int(lower.get("priority", 4))
                    if higher_priority >= lower_priority:
                        issue_counts["pairwise_priority_conflict"] += 1
                        _record_example(
                            examples,
                            "pairwise_priority_conflict",
                            {
                                **base_info,
                                "pair": pair,
                                "higher_priority": higher_priority,
                                "lower_priority": lower_priority,
                            },
                            args.example_limit,
                        )
                    gap = pair.get("priority_gap")
                    if gap is not None and gap != lower_priority - higher_priority:
                        issue_counts["pairwise_gap_mismatch"] += 1
                        _record_example(
                            examples,
                            "pairwise_gap_mismatch",
                            {
                                **base_info,
                                "pair": pair,
                                "expected_gap": lower_priority - higher_priority,
                            },
                            args.example_limit,
                        )

            if isinstance(critical_topk_targets, list):
                for target_id in critical_topk_targets:
                    if str(target_id) not in demand_id_set:
                        issue_counts["topk_unknown_demand_id"] += 1
                        _record_example(
                            examples,
                            "topk_unknown_demand_id",
                            {**base_info, "target_id": target_id},
                            args.example_limit,
                        )
                expected_topk = sorted(
                    demand_id
                    for demand_id, row in label_map.items()
                    if int(row.get("priority", 4)) <= args.urgent_threshold
                )
                actual_topk = sorted(str(item) for item in critical_topk_targets)
                if actual_topk != expected_topk:
                    issue_counts["topk_mismatch_vs_priority_labels"] += 1
                    _record_example(
                        examples,
                        "topk_mismatch_vs_priority_labels",
                        {**base_info, "actual_topk": actual_topk, "expected_topk": expected_topk},
                        args.example_limit,
                    )

            try:
                expected_targets = build_window_priority_targets(
                    demands,
                    priority_field="extraction_observable_priority",
                    urgent_threshold=args.urgent_threshold,
                )
                expected_priority_labels = expected_targets["demand_configs"]
                expected_pairwise = expected_targets["pairwise_preferences"]
                expected_topk = expected_targets["critical_topk_targets"]
                if priority_labels != expected_priority_labels:
                    issue_counts["priority_labels_mismatch_vs_demands"] += 1
                    _record_example(
                        examples,
                        "priority_labels_mismatch_vs_demands",
                        {
                            **base_info,
                            "expected_first": expected_priority_labels[:3],
                            "actual_first": priority_labels[:3],
                        },
                        args.example_limit,
                    )
                if isinstance(pairwise_preferences, list) and pairwise_preferences != expected_pairwise:
                    issue_counts["pairwise_mismatch_vs_demands"] += 1
                    _record_example(
                        examples,
                        "pairwise_mismatch_vs_demands",
                        {
                            **base_info,
                            "expected_count": len(expected_pairwise),
                            "actual_count": len(pairwise_preferences),
                        },
                        args.example_limit,
                    )
                if isinstance(critical_topk_targets, list) and critical_topk_targets != expected_topk:
                    issue_counts["topk_mismatch_vs_demands"] += 1
                    _record_example(
                        examples,
                        "topk_mismatch_vs_demands",
                        {
                            **base_info,
                            "expected_topk": expected_topk,
                            "actual_topk": critical_topk_targets,
                        },
                        args.example_limit,
                    )
            except Exception as exc:
                issue_counts["failed_to_recompute_targets"] += 1
                _record_example(
                    examples,
                    "failed_to_recompute_targets",
                    {**base_info, "error": str(exc)},
                    args.example_limit,
                )

    report = {
        "filename": args.filename,
        "input_dirs": [str(path) for path in input_dirs],
        "counts": dict(counts),
        "null_counts": dict(null_counts),
        "type_counts": dict(type_counts),
        "issue_counts": dict(issue_counts),
        "examples": dict(examples),
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out:
        output_path = Path(args.json_out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
