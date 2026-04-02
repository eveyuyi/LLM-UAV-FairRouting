"""Quality gates for observable-priority training data shards.

This module keeps the quality policy explicit and reusable:
- sample-level acceptance: only audit-passed dialogues feed llm2/pipeline training
- shard-level release gates: decide whether a shard is accepted or needs regeneration
- hard-case quotas: ensure anti-shortcut coverage before GRPO release
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Mapping, MutableMapping

from llm4fairrouting.data.event_data import load_event_records

QUALITY_GATE_POLICY_VERSION = "quality_gate_v1"

DEFAULT_QUALITY_THRESHOLDS = {
    "min_audit_pass_rate": 0.90,
    "min_priority_pass_rate_p1": 0.80,
    "min_priority_pass_rate_p2": 0.80,
    "max_priority_gap_vs_p4": 0.15,
    "max_scenario_context_missing_rate": 0.15,
    "max_requester_role_missing_rate": 0.05,
    "min_requester_role_match_rate": 0.85,
    "min_surface_contradiction_ratio": 0.10,
    "min_near_tie_ratio": 0.10,
    "min_surface_contradiction_count": 3,
    "min_near_tie_count": 3,
}


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _round(value: float) -> float:
    return round(float(value), 4)


def _normalize_role(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _load_jsonl(path: str | Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if str(line).strip()
        ]


def filter_accepted_dialogues(dialogues: Iterable[Mapping[str, object]]) -> List[Dict]:
    """Return only dialogues whose audit passed."""
    accepted: List[Dict] = []
    for dialogue in dialogues:
        audit = dict(dialogue.get("audit", {}) or {})
        if bool(audit.get("passed", False)):
            accepted.append(dict(dialogue))
    return accepted


def _dialogue_pass_by_priority(dialogues: Iterable[Mapping[str, object]]) -> dict[int, Dict[str, float | int]]:
    counts: dict[int, MutableMapping[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    for dialogue in dialogues:
        latent_priority = int(dialogue.get("annotations", {}).get("latent_priority", 4) or 4)
        passed = bool(dialogue.get("audit", {}).get("passed", False))
        counts[latent_priority]["total"] += 1
        if passed:
            counts[latent_priority]["passed"] += 1
    return {
        priority: {
            "passed": payload["passed"],
            "total": payload["total"],
            "rate": _round(_safe_rate(payload["passed"], payload["total"])),
        }
        for priority, payload in sorted(counts.items())
    }


def _observability_summary(dialogues: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    scores = [
        float(dialogue.get("audit", {}).get("observability_score"))
        for dialogue in dialogues
        if dialogue.get("audit", {}).get("observability_score") is not None
    ]
    if not scores:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": _round(sum(scores) / len(scores)),
        "median": _round(median(scores)),
        "min": _round(min(scores)),
        "max": _round(max(scores)),
    }


def _missing_factor_counts(dialogues: Iterable[Mapping[str, object]]) -> tuple[Counter, Counter]:
    counts: Counter = Counter()
    totals: Counter = Counter()
    for dialogue in dialogues:
        annotations = dict(dialogue.get("annotations", {}) or {})
        for factor in annotations.get("must_mention_factors", []) or []:
            name = str(factor.get("name", "")).strip()
            if name:
                totals[name] += 1
        for name in dialogue.get("audit", {}).get("missing_must_mention_factors", []) or []:
            if str(name).strip():
                counts[str(name)] += 1
    return counts, totals


def _requester_role_metrics(
    event_records: Iterable[Mapping[str, object]],
    pipeline_windows: Iterable[Mapping[str, object]],
) -> Dict[str, object]:
    event_by_id = {
        str(record.get("event_id")): dict(record)
        for record in event_records
        if record.get("event_id") is not None
    }
    total = 0
    matched = 0
    missing = 0
    drift_examples: Counter = Counter()
    for window in pipeline_windows:
        for demand in window.get("demands", []) or []:
            source_event_id = str(demand.get("source_event_id", "")).strip()
            if not source_event_id or source_event_id not in event_by_id:
                continue
            total += 1
            expected_role = _normalize_role(event_by_id[source_event_id].get("requester_role", ""))
            observed_role = _normalize_role(
                demand.get("requester_role")
                or demand.get("priority_evaluation_signals", {}).get("requester_role")
                or ""
            )
            if not observed_role:
                missing += 1
                drift_examples[(expected_role, None)] += 1
                continue
            if observed_role == expected_role:
                matched += 1
            else:
                drift_examples[(expected_role, observed_role)] += 1
    return {
        "total": total,
        "matched": matched,
        "missing": missing,
        "match_rate": _round(_safe_rate(matched, total)),
        "missing_rate": _round(_safe_rate(missing, total)),
        "top_drifts": [
            {
                "expected": expected,
                "observed": observed,
                "count": count,
            }
            for (expected, observed), count in drift_examples.most_common(10)
        ],
    }


def _hard_case_metrics(hard_windows: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    counts: Counter = Counter()
    total = 0
    for window in hard_windows:
        total += 1
        label = str(window.get("time_window", ""))
        if "surface_contradiction" in label:
            counts["surface_contradiction"] += 1
        elif "near_tie" in label:
            counts["near_tie"] += 1
        elif "counterfactual" in label:
            counts["counterfactual"] += 1
        elif "mixed_priority" in label:
            counts["mixed_priority"] += 1
        else:
            counts["other"] += 1
    ratios = {
        key: _round(_safe_rate(value, total))
        for key, value in counts.items()
    }
    for key in ("surface_contradiction", "near_tie", "counterfactual", "mixed_priority", "other"):
        counts.setdefault(key, 0)
        ratios.setdefault(key, 0.0)
    return {
        "total": total,
        "counts": dict(counts),
        "ratios": ratios,
    }


def _check(value: float, *, min_value: float | None = None, max_value: float | None = None) -> Dict[str, object]:
    passed = True
    threshold: Dict[str, float] = {}
    if min_value is not None:
        threshold["min"] = min_value
        passed = passed and value >= min_value
    if max_value is not None:
        threshold["max"] = max_value
        passed = passed and value <= max_value
    return {
        "value": _round(value),
        "threshold": threshold,
        "passed": bool(passed),
    }


def _skipped_check(*, min_value: float | None = None, max_value: float | None = None, reason: str) -> Dict[str, object]:
    threshold: Dict[str, float] = {}
    if min_value is not None:
        threshold["min"] = min_value
    if max_value is not None:
        threshold["max"] = max_value
    return {
        "value": None,
        "threshold": threshold,
        "passed": True,
        "skipped": True,
        "reason": reason,
    }


def build_quality_report(
    *,
    event_records: List[Dict],
    dialogues: List[Dict],
    pipeline_windows: List[Dict],
    hard_windows: List[Dict],
    thresholds: Mapping[str, float] | None = None,
) -> Dict[str, object]:
    thresholds = {**DEFAULT_QUALITY_THRESHOLDS, **dict(thresholds or {})}
    accepted_dialogues = filter_accepted_dialogues(dialogues)
    pass_by_priority = _dialogue_pass_by_priority(dialogues)
    missing_counts, missing_totals = _missing_factor_counts(dialogues)
    scenario_total = int(missing_totals.get("scenario_context", 0))
    scenario_missing = int(missing_counts.get("scenario_context", 0))
    requester_role = _requester_role_metrics(event_records, pipeline_windows)
    hard_cases = _hard_case_metrics(hard_windows)

    overall_audit_pass = _safe_rate(len(accepted_dialogues), len(dialogues))
    p4_rate = float(pass_by_priority.get(4, {}).get("rate", 0.0))
    p1_rate = float(pass_by_priority.get(1, {}).get("rate", 0.0))
    p2_rate = float(pass_by_priority.get(2, {}).get("rate", 0.0))
    p1_total = int(pass_by_priority.get(1, {}).get("total", 0) or 0)
    p2_total = int(pass_by_priority.get(2, {}).get("total", 0) or 0)
    p4_total = int(pass_by_priority.get(4, {}).get("total", 0) or 0)
    gap_candidates: list[float] = []
    if p4_total and p1_total:
        gap_candidates.append(p4_rate - p1_rate)
    if p4_total and p2_total:
        gap_candidates.append(p4_rate - p2_rate)
    largest_gap = max(gap_candidates + [0.0])

    missing_rates = {
        name: _round(_safe_rate(count, int(missing_totals.get(name, 0))))
        for name, count in missing_counts.items()
    }

    checks = {
        "audit_pass_rate": _check(
            overall_audit_pass,
            min_value=float(thresholds["min_audit_pass_rate"]),
        ),
        "priority_1_audit_pass_rate": (
            _check(
                p1_rate,
                min_value=float(thresholds["min_priority_pass_rate_p1"]),
            )
            if p1_total
            else _skipped_check(
                min_value=float(thresholds["min_priority_pass_rate_p1"]),
                reason="No priority-1 dialogues in this shard",
            )
        ),
        "priority_2_audit_pass_rate": (
            _check(
                p2_rate,
                min_value=float(thresholds["min_priority_pass_rate_p2"]),
            )
            if p2_total
            else _skipped_check(
                min_value=float(thresholds["min_priority_pass_rate_p2"]),
                reason="No priority-2 dialogues in this shard",
            )
        ),
        "priority_gap_vs_p4": (
            _check(
                largest_gap,
                max_value=float(thresholds["max_priority_gap_vs_p4"]),
            )
            if gap_candidates
            else _skipped_check(
                max_value=float(thresholds["max_priority_gap_vs_p4"]),
                reason="No comparable priority-4 vs priority-1/2 slices in this shard",
            )
        ),
        "scenario_context_missing_rate": _check(
            _safe_rate(scenario_missing, scenario_total),
            max_value=float(thresholds["max_scenario_context_missing_rate"]),
        ),
        "requester_role_missing_rate": _check(
            float(requester_role["missing_rate"]),
            max_value=float(thresholds["max_requester_role_missing_rate"]),
        ),
        "requester_role_match_rate": _check(
            float(requester_role["match_rate"]),
            min_value=float(thresholds["min_requester_role_match_rate"]),
        ),
        "surface_contradiction_ratio": _check(
            float(hard_cases["ratios"]["surface_contradiction"]),
            min_value=float(thresholds["min_surface_contradiction_ratio"]),
        ),
        "near_tie_ratio": _check(
            float(hard_cases["ratios"]["near_tie"]),
            min_value=float(thresholds["min_near_tie_ratio"]),
        ),
        "surface_contradiction_count": _check(
            float(hard_cases["counts"]["surface_contradiction"]),
            min_value=float(thresholds["min_surface_contradiction_count"]),
        ),
        "near_tie_count": _check(
            float(hard_cases["counts"]["near_tie"]),
            min_value=float(thresholds["min_near_tie_count"]),
        ),
    }

    return {
        "schema_version": "priority_quality_report_v1",
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "quality_gate_policy_version": QUALITY_GATE_POLICY_VERSION,
        "thresholds": thresholds,
        "counts": {
            "events": len(event_records),
            "dialogues_raw": len(dialogues),
            "dialogues_accepted": len(accepted_dialogues),
            "pipeline_windows": len(pipeline_windows),
            "hard_windows": len(hard_windows),
        },
        "metrics": {
            "dialogue_audit_pass_rate": _round(overall_audit_pass),
            "dialogue_audit_pass_by_priority": pass_by_priority,
            "dialogue_observability": _observability_summary(dialogues),
            "missing_must_mention_counts": dict(missing_counts),
            "missing_must_mention_rates": missing_rates,
            "requester_role_pipeline": requester_role,
            "hard_case_coverage": hard_cases,
        },
        "checks": checks,
    }


def build_release_manifest(
    *,
    dataset_manifest: Mapping[str, object],
    quality_report: Mapping[str, object],
) -> Dict[str, object]:
    checks = dict(quality_report.get("checks", {}) or {})
    pipeline_ready = all(
        bool(checks.get(name, {}).get("passed", False))
        for name in (
            "audit_pass_rate",
            "priority_1_audit_pass_rate",
            "priority_2_audit_pass_rate",
            "priority_gap_vs_p4",
            "scenario_context_missing_rate",
            "requester_role_missing_rate",
            "requester_role_match_rate",
        )
    )
    grpo_ready = pipeline_ready and all(
        bool(checks.get(name, {}).get("passed", False))
        for name in (
            "surface_contradiction_ratio",
            "near_tie_ratio",
            "surface_contradiction_count",
            "near_tie_count",
        )
    )
    clean_ready = int(dataset_manifest.get("counts", {}).get("llm3_sft_clean", 0) or 0) > 0

    if not clean_ready:
        release_status = "debug_only"
    elif pipeline_ready and grpo_ready:
        release_status = "accepted"
    else:
        release_status = "needs_regen"

    recommended_files = ["llm3_sft_clean"] if clean_ready else []
    if pipeline_ready:
        recommended_files.extend(["llm2_sft", "llm3_sft_pipeline"])
    if grpo_ready:
        recommended_files.append("llm3_grpo_hard")

    failed_checks = [
        name
        for name, payload in checks.items()
        if not bool(payload.get("passed", False))
    ]

    return {
        "schema_version": "priority_release_manifest_v1",
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "quality_gate_policy_version": quality_report.get("quality_gate_policy_version", QUALITY_GATE_POLICY_VERSION),
        "release_status": release_status,
        "recommended_training_files": recommended_files,
        "failed_checks": failed_checks,
        "quality_report_file": "quality_report.json",
        "dataset_manifest_file": "dataset_manifest.json",
        "counts": dict(dataset_manifest.get("counts", {}) or {}),
    }


def build_quality_outputs_for_directory(
    dataset_dir: str | Path,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> tuple[Dict[str, object], Dict[str, object]]:
    dataset_dir = Path(dataset_dir)
    dataset_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    files = dict(dataset_manifest.get("files", {}) or {})
    event_records = load_event_records(dataset_dir / files["events_manifest"])
    dialogues = _load_jsonl(dataset_dir / files["dialogues"])
    pipeline_windows = _load_jsonl(dataset_dir / files["llm3_sft_pipeline"])
    hard_windows = _load_jsonl(dataset_dir / files["llm3_grpo_hard"])
    quality_report = build_quality_report(
        event_records=event_records,
        dialogues=dialogues,
        pipeline_windows=pipeline_windows,
        hard_windows=hard_windows,
        thresholds=thresholds,
    )
    release_manifest = build_release_manifest(
        dataset_manifest=dataset_manifest,
        quality_report=quality_report,
    )
    return quality_report, release_manifest


def write_quality_outputs(
    dataset_dir: str | Path,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> tuple[Path, Path]:
    dataset_dir = Path(dataset_dir)
    quality_report, release_manifest = build_quality_outputs_for_directory(
        dataset_dir,
        thresholds=thresholds,
    )
    quality_path = dataset_dir / "quality_report.json"
    release_path = dataset_dir / "release_manifest.json"
    quality_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")
    release_path.write_text(json.dumps(release_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return quality_path, release_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build quality and release manifests for a generated training-data shard.",
    )
    parser.add_argument("dataset_dir", help="Directory containing dataset_manifest.json and shard JSONL files")
    parser.add_argument("--min-audit-pass-rate", type=float, default=DEFAULT_QUALITY_THRESHOLDS["min_audit_pass_rate"])
    parser.add_argument("--min-priority-pass-rate-p1", type=float, default=DEFAULT_QUALITY_THRESHOLDS["min_priority_pass_rate_p1"])
    parser.add_argument("--min-priority-pass-rate-p2", type=float, default=DEFAULT_QUALITY_THRESHOLDS["min_priority_pass_rate_p2"])
    parser.add_argument("--max-priority-gap-vs-p4", type=float, default=DEFAULT_QUALITY_THRESHOLDS["max_priority_gap_vs_p4"])
    parser.add_argument("--max-scenario-context-missing-rate", type=float, default=DEFAULT_QUALITY_THRESHOLDS["max_scenario_context_missing_rate"])
    parser.add_argument("--max-requester-role-missing-rate", type=float, default=DEFAULT_QUALITY_THRESHOLDS["max_requester_role_missing_rate"])
    parser.add_argument("--min-requester-role-match-rate", type=float, default=DEFAULT_QUALITY_THRESHOLDS["min_requester_role_match_rate"])
    parser.add_argument("--min-surface-contradiction-ratio", type=float, default=DEFAULT_QUALITY_THRESHOLDS["min_surface_contradiction_ratio"])
    parser.add_argument("--min-near-tie-ratio", type=float, default=DEFAULT_QUALITY_THRESHOLDS["min_near_tie_ratio"])
    parser.add_argument("--min-surface-contradiction-count", type=int, default=DEFAULT_QUALITY_THRESHOLDS["min_surface_contradiction_count"])
    parser.add_argument("--min-near-tie-count", type=int, default=DEFAULT_QUALITY_THRESHOLDS["min_near_tie_count"])
    args = parser.parse_args()

    thresholds = {
        "min_audit_pass_rate": args.min_audit_pass_rate,
        "min_priority_pass_rate_p1": args.min_priority_pass_rate_p1,
        "min_priority_pass_rate_p2": args.min_priority_pass_rate_p2,
        "max_priority_gap_vs_p4": args.max_priority_gap_vs_p4,
        "max_scenario_context_missing_rate": args.max_scenario_context_missing_rate,
        "max_requester_role_missing_rate": args.max_requester_role_missing_rate,
        "min_requester_role_match_rate": args.min_requester_role_match_rate,
        "min_surface_contradiction_ratio": args.min_surface_contradiction_ratio,
        "min_near_tie_ratio": args.min_near_tie_ratio,
        "min_surface_contradiction_count": args.min_surface_contradiction_count,
        "min_near_tie_count": args.min_near_tie_count,
    }
    quality_path, release_path = write_quality_outputs(args.dataset_dir, thresholds=thresholds)
    print(f"Wrote quality report to {quality_path}")
    print(f"Wrote release manifest to {release_path}")
