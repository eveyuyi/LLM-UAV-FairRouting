"""Clean, compact, and export llm_selection JSONL for SFT/GRPO training.

This exporter is CPU-only and defaults to JSONL outputs (no parquet dependency).
It applies the following steps:
1) strict per-line JSON parsing (skip malformed lines)
2) record-level validation and deduplication
3) objective-group candidate compression with gold-solution retention
4) train/val split with optional multi-group oversampling on train
5) export SFT + GRPO JSONL files and a summary JSON report
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SYSTEM_PROMPT = (
    "You are a policy-aware selector for UAV routing Pareto-frontier solutions. "
    "Select one candidate solution and return JSON only."
)

DEFAULT_KEEP_CANDIDATES_PER_GROUP = 12
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42
DEFAULT_MULTIGROUP_OVERSAMPLE = 3
DEFAULT_MAX_DEMAND_CARDS = 16

INPUT_TOP_KEYS = (
    "schema_version",
    "sample_id",
    "selection_profile",
    "problem_context",
    "grouping_policy",
    "scene_summary",
    "selection_rules",
    "demand_cards",
    "objective_groups",
)

CANDIDATE_OBJECTIVE_KEYS = (
    "final_total_distance_m",
    "average_delivery_time_h",
    "final_total_noise_impact",
    "service_rate",
    "n_used_drones",
)

CANDIDATE_SERVICE_KEYS = (
    "n_served_demands",
    "served_high_priority_count",
    "served_priority_medical_count",
    "high_priority_medical_service_rate",
    "served_quiet_sensitive_demand_count",
    "quiet_sensitive_service_rate",
)

CANDIDATE_FAIRNESS_KEYS = (
    "weighted_priority_coverage",
    "elderly_population_coverage",
)

CANDIDATE_LAND_USE_KEYS = (
    "served_land_use_counts",
    "served_medical_neighborhood_count",
    "quiet_sensitive_service_rate",
    "quiet_sensitive_noise_proxy",
)

CANDIDATE_TIE_BREAK_KEYS = (
    "served_population_250m",
    "served_elderly_population_65plus_250m",
    "served_medical_neighborhood_count",
    "served_quiet_sensitive_demand_count",
    "quiet_sensitive_service_rate",
    "quiet_sensitive_noise_proxy",
)


@dataclass
class ExportStats:
    total_lines: int = 0
    parsed_records: int = 0
    malformed_lines: int = 0
    duplicate_record_id: int = 0
    missing_fields: int = 0
    missing_target_solution: int = 0
    missing_target_group: int = 0
    invalid_messages: int = 0
    kept_records: int = 0
    train_records: int = 0
    val_records: int = 0
    train_records_after_oversample: int = 0
    multigroup_records_train_before_oversample: int = 0
    multigroup_records_train_after_oversample: int = 0


def _pick_fields(source: Dict, keys: Sequence[str]) -> Dict:
    return {key: deepcopy(source[key]) for key in keys if key in source}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _candidate_score(card: Dict) -> Tuple[float, float, float, float]:
    fairness = card.get("fairness_features") or {}
    objectives = card.get("objectives") or {}
    service = card.get("service_features") or {}
    tie_break = card.get("tie_break_features") or {}

    weighted_priority_coverage = _safe_float(fairness.get("weighted_priority_coverage"), 0.0)
    service_rate = _safe_float(objectives.get("service_rate"), 0.0)
    high_medical_rate = _safe_float(service.get("high_priority_medical_service_rate"), 0.0)
    quiet_sensitive_service_rate = _safe_float(
        tie_break.get("quiet_sensitive_service_rate"),
        _safe_float(service.get("quiet_sensitive_service_rate"), 0.0),
    )

    # Higher is better. Sorting is descending by this tuple.
    return (
        weighted_priority_coverage,
        service_rate,
        high_medical_rate,
        quiet_sensitive_service_rate,
    )


def _compress_candidate_card(card: Dict) -> Dict:
    return {
        "solution_id": card.get("solution_id"),
        "objectives": _pick_fields(card.get("objectives") or {}, CANDIDATE_OBJECTIVE_KEYS),
        "service_features": _pick_fields(card.get("service_features") or {}, CANDIDATE_SERVICE_KEYS),
        "fairness_features": _pick_fields(card.get("fairness_features") or {}, CANDIDATE_FAIRNESS_KEYS),
        "land_use_features": _pick_fields(card.get("land_use_features") or {}, CANDIDATE_LAND_USE_KEYS),
        "tie_break_features": _pick_fields(card.get("tie_break_features") or {}, CANDIDATE_TIE_BREAK_KEYS),
    }


def _compress_demand_cards(selection_input: Dict, max_demand_cards: int) -> List[Dict]:
    cards = list(selection_input.get("demand_cards", []) or [])
    compact_cards: List[Dict] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        compact_cards.append(
            {
                "demand_id": card.get("demand_id"),
                "priority": card.get("priority"),
                "cargo_type": card.get("cargo_type"),
                "supply_type": card.get("supply_type"),
                "destination_land_use": card.get("destination_land_use"),
                "dest_population_250m": card.get("dest_population_250m"),
                "elderly_population_65plus_250m": card.get("elderly_population_65plus_250m"),
                "is_priority_medical": card.get("is_priority_medical"),
                "is_quiet_sensitive_land_use": card.get("is_quiet_sensitive_land_use"),
            }
        )

    # Prioritize urgent/medical/quiet-sensitive demands first, then by id.
    compact_cards.sort(
        key=lambda item: (
            _safe_int(item.get("priority"), 9),
            0 if bool(item.get("is_priority_medical")) else 1,
            0 if bool(item.get("is_quiet_sensitive_land_use")) else 1,
            str(item.get("demand_id", "")),
        )
    )
    max_n = max(1, int(max_demand_cards))
    return compact_cards[:max_n]


def _compress_objective_groups(
    selection_input: Dict,
    selected_solution_id: str,
    keep_candidates_per_group: int,
) -> List[Dict]:
    compact_groups: List[Dict] = []
    for group in selection_input.get("objective_groups", []) or []:
        cards = list(group.get("candidate_cards", []) or [])
        sorted_cards = sorted(cards, key=_candidate_score, reverse=True)
        keep_n = max(1, int(keep_candidates_per_group))
        kept = sorted_cards[:keep_n]

        # Ensure gold solution is retained in its group.
        gold_card = next((card for card in cards if card.get("solution_id") == selected_solution_id), None)
        if gold_card is not None and all(card.get("solution_id") != selected_solution_id for card in kept):
            if kept:
                kept[-1] = gold_card
            else:
                kept = [gold_card]

        compact_groups.append(
            {
                "group_id": group.get("group_id"),
                "objective_signature": deepcopy(group.get("objective_signature")),
                "group_size": group.get("group_size", len(cards)),
                "candidate_solution_ids": [card.get("solution_id") for card in kept],
                "group_summary": deepcopy(group.get("group_summary")),
                "candidate_cards": [_compress_candidate_card(card) for card in kept],
            }
        )
    return compact_groups


def _compact_selection_input(record: Dict, keep_candidates_per_group: int, max_demand_cards: int) -> Dict:
    source_input = record.get("selection_input", {}) or {}
    target = record.get("selection_target", {}) or {}
    selected_solution_id = str(target.get("selected_solution_id", "")).strip()

    compact = _pick_fields(source_input, INPUT_TOP_KEYS)
    compact["demand_cards"] = _compress_demand_cards(source_input, max_demand_cards=max_demand_cards)
    compact["objective_groups"] = _compress_objective_groups(
        source_input,
        selected_solution_id=selected_solution_id,
        keep_candidates_per_group=keep_candidates_per_group,
    )
    return compact


def _assistant_payload(target: Dict) -> Dict:
    training_labels = target.get("training_labels") or {}
    return {
        "selected_group_id": target.get("selected_group_id"),
        "selected_solution_id": target.get("selected_solution_id"),
        "selection_mode": target.get("selection_mode"),
        "primary_reason_codes": list(target.get("primary_reason_codes") or []),
        "decision_confidence": target.get("decision_confidence"),
        "training_labels": {
            "scene_type": training_labels.get("scene_type"),
            "scene_tags": training_labels.get("scene_tags"),
            "recommended_profiles": training_labels.get("recommended_profiles"),
            "decision_difficulty": training_labels.get("decision_difficulty"),
            "label_quiet_sensitive_service_rate": training_labels.get("label_quiet_sensitive_service_rate"),
            "label_quiet_sensitive_noise_proxy": training_labels.get("label_quiet_sensitive_noise_proxy"),
        },
    }


def _render_user_prompt(compact_input: Dict) -> str:
    return (
        "Select one Pareto-frontier solution for the UAV routing problem.\n"
        "Return JSON with keys: selected_group_id, selected_solution_id, selection_mode, "
        "primary_reason_codes, decision_confidence, training_labels.\n\n"
        "Input JSON:\n"
        f"{json.dumps(compact_input, ensure_ascii=False, indent=2)}"
    )


def _validate_record(record: Dict, stats: ExportStats) -> bool:
    if not isinstance(record, dict):
        stats.missing_fields += 1
        return False

    required = ("record_id", "selection_input", "selection_target")
    if any(key not in record for key in required):
        stats.missing_fields += 1
        return False

    selection_input = record.get("selection_input") or {}
    selection_target = record.get("selection_target") or {}
    objective_groups = selection_input.get("objective_groups", []) or []

    selected_solution_id = str(selection_target.get("selected_solution_id", "")).strip()
    selected_group_id = str(selection_target.get("selected_group_id", "")).strip()

    if not selected_solution_id:
        stats.missing_target_solution += 1
        return False
    if not selected_group_id:
        stats.missing_target_group += 1
        return False

    group_ids = {str(group.get("group_id", "")).strip() for group in objective_groups}
    if selected_group_id not in group_ids:
        stats.missing_target_group += 1
        return False

    candidate_ids = set()
    for group in objective_groups:
        for card in group.get("candidate_cards", []) or []:
            sid = str(card.get("solution_id", "")).strip()
            if sid:
                candidate_ids.add(sid)
    if selected_solution_id not in candidate_ids:
        stats.missing_target_solution += 1
        return False

    messages = record.get("messages")
    if messages is not None and (not isinstance(messages, list) or len(messages) < 3):
        stats.invalid_messages += 1
        return False
    return True


def _load_and_clean_records(input_path: Path, stats: ExportStats) -> List[Dict]:
    seen_record_ids = set()
    records: List[Dict] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            stats.total_lines += 1
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.malformed_lines += 1
                continue
            stats.parsed_records += 1

            record_id = str(record.get("record_id", "")).strip()
            if not record_id or record_id in seen_record_ids:
                stats.duplicate_record_id += 1
                continue
            seen_record_ids.add(record_id)

            if not _validate_record(record, stats):
                continue
            records.append(record)

    stats.kept_records = len(records)
    return records


def _split_train_val(records: Sequence[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0).")
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    if len(items) <= 1 or val_ratio == 0.0:
        return items, []
    val_size = max(1, int(round(len(items) * val_ratio)))
    val_size = min(val_size, len(items) - 1)
    return items[val_size:], items[:val_size]


def _is_multigroup(record: Dict) -> bool:
    groups = record.get("selection_input", {}).get("objective_groups", []) or []
    return len(groups) > 1


def _oversample_multigroup(train_records: Sequence[Dict], factor: int) -> List[Dict]:
    factor = max(1, int(factor))
    if factor == 1:
        return list(train_records)
    augmented: List[Dict] = []
    for record in train_records:
        repeats = factor if _is_multigroup(record) else 1
        augmented.extend([record] * repeats)
    return augmented


def _build_sft_row(record: Dict, compact_input: Dict) -> Dict:
    target = record.get("selection_target") or {}
    response_payload = _assistant_payload(target)
    user_prompt = _render_user_prompt(compact_input)
    response_text = json.dumps(response_payload, ensure_ascii=False, indent=2)
    return {
        "record_id": record.get("record_id"),
        "task_type": record.get("task_type"),
        "prompt": user_prompt,
        "response": response_text,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response_text},
        ],
        "compact_selection_input": compact_input,
        "selection_target": response_payload,
        "metadata": record.get("metadata", {}),
    }


def _build_grpo_row(record: Dict, compact_input: Dict) -> Dict:
    target = record.get("selection_target") or {}
    user_prompt = _render_user_prompt(compact_input)

    candidate_solution_ids: List[str] = []
    for group in compact_input.get("objective_groups", []) or []:
        for sid in group.get("candidate_solution_ids", []) or []:
            sid_str = str(sid).strip()
            if sid_str:
                candidate_solution_ids.append(sid_str)

    reward_gt = {
        "selected_group_id": target.get("selected_group_id"),
        "selected_solution_id": target.get("selected_solution_id"),
        "selection_mode": target.get("selection_mode"),
        "primary_reason_codes": list(target.get("primary_reason_codes") or []),
        "decision_confidence": target.get("decision_confidence"),
        "candidate_solution_ids": sorted(set(candidate_solution_ids)),
        "group_ids": [
            str(group.get("group_id", "")).strip()
            for group in compact_input.get("objective_groups", [])
            if str(group.get("group_id", "")).strip()
        ],
        "training_labels": target.get("training_labels", {}),
    }
    return {
        "record_id": record.get("record_id"),
        "data_source": "llm_selection_pareto_window",
        "ability": "pareto_selection",
        "prompt": [{"role": "user", "content": user_prompt}],
        "reward_model": {"style": "rule", "ground_truth": reward_gt},
        "extra_info": {
            "task_type": record.get("task_type"),
            "metadata": record.get("metadata", {}),
        },
    }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(rows: Iterable[Dict], out_path: Path) -> int:
    _ensure_parent(out_path)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _export_split(
    records: Sequence[Dict],
    keep_candidates_per_group: int,
    max_demand_cards: int,
    sft_out: Path,
    grpo_out: Path,
) -> Tuple[int, int]:
    sft_rows: List[Dict] = []
    grpo_rows: List[Dict] = []
    for record in records:
        compact_input = _compact_selection_input(
            record,
            keep_candidates_per_group=keep_candidates_per_group,
            max_demand_cards=max_demand_cards,
        )
        sft_rows.append(_build_sft_row(record, compact_input))
        grpo_rows.append(_build_grpo_row(record, compact_input))
    sft_count = _write_jsonl(sft_rows, sft_out)
    grpo_count = _write_jsonl(grpo_rows, grpo_out)
    return sft_count, grpo_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean + compact + export llm_selection JSONL into SFT/GRPO JSONL datasets."
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to raw llm_selection JSONL.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write exported files.",
    )
    parser.add_argument(
        "--keep-candidates-per-group",
        type=int,
        default=DEFAULT_KEEP_CANDIDATES_PER_GROUP,
        help="Number of candidate cards kept in each objective_group after compression.",
    )
    parser.add_argument(
        "--max-demand-cards",
        type=int,
        default=DEFAULT_MAX_DEMAND_CARDS,
        help="Maximum demand cards kept in prompt input after compression.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Validation split ratio in [0.0, 1.0).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--multigroup-oversample",
        type=int,
        default=DEFAULT_MULTIGROUP_OVERSAMPLE,
        help="Repeat factor for multi-group samples in train split (1 disables oversample).",
    )
    parser.add_argument(
        "--summary-name",
        default="export_summary.json",
        help="Summary file name in output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = ExportStats()
    cleaned_records = _load_and_clean_records(input_path, stats)
    train_records, val_records = _split_train_val(cleaned_records, args.val_ratio, args.seed)

    stats.train_records = len(train_records)
    stats.val_records = len(val_records)
    stats.multigroup_records_train_before_oversample = sum(1 for row in train_records if _is_multigroup(row))

    train_aug = _oversample_multigroup(train_records, args.multigroup_oversample)
    stats.train_records_after_oversample = len(train_aug)
    stats.multigroup_records_train_after_oversample = sum(1 for row in train_aug if _is_multigroup(row))

    train_sft = output_dir / "train_sft_compact.jsonl"
    val_sft = output_dir / "val_sft_compact.jsonl"
    train_grpo = output_dir / "train_grpo_compact.jsonl"
    val_grpo = output_dir / "val_grpo_compact.jsonl"

    train_sft_count, train_grpo_count = _export_split(
        train_aug,
        keep_candidates_per_group=args.keep_candidates_per_group,
        max_demand_cards=args.max_demand_cards,
        sft_out=train_sft,
        grpo_out=train_grpo,
    )
    val_sft_count, val_grpo_count = _export_split(
        val_records,
        keep_candidates_per_group=args.keep_candidates_per_group,
        max_demand_cards=args.max_demand_cards,
        sft_out=val_sft,
        grpo_out=val_grpo,
    )

    summary = {
        "input_jsonl": str(input_path),
        "output_dir": str(output_dir),
        "config": {
            "keep_candidates_per_group": args.keep_candidates_per_group,
            "max_demand_cards": args.max_demand_cards,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "multigroup_oversample": args.multigroup_oversample,
            "format": "jsonl_only",
        },
        "stats": stats.__dict__,
        "outputs": {
            "train_sft_compact_jsonl": str(train_sft),
            "val_sft_compact_jsonl": str(val_sft),
            "train_grpo_compact_jsonl": str(train_grpo),
            "val_grpo_compact_jsonl": str(val_grpo),
            "train_sft_rows": train_sft_count,
            "val_sft_rows": val_sft_count,
            "train_grpo_rows": train_grpo_count,
            "val_grpo_rows": val_grpo_count,
        },
    }

    summary_path = output_dir / args.summary_name
    _ensure_parent(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
