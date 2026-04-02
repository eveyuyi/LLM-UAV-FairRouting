"""Build training corpora with observable priority labels for LLM2 and LLM3."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from llm4fairrouting.config.runtime_env import env_text, prepare_env_file
from llm4fairrouting.data.demand_event_generation import generate_daily_demand_dataset
from llm4fairrouting.data.event_semantics import DIALOGUE_STYLE_VARIANTS
from llm4fairrouting.data.event_structuring import build_gold_structured_demand
from llm4fairrouting.data.priority_policy import PRIORITY_POLICY_VERSION
from llm4fairrouting.data.quality_gates import (
    DEFAULT_QUALITY_THRESHOLDS,
    QUALITY_GATE_POLICY_VERSION,
    build_quality_report,
    build_release_manifest,
    filter_accepted_dialogues,
)
from llm4fairrouting.data.demand_event_generation import save_event_manifest
from llm4fairrouting.data.priority_labels import (
    attach_priority_labels,
    build_window_priority_targets,
)
from llm4fairrouting.data.seed_paths import (
    BUILDING_DATA_PATH,
    DEMAND_TRAINING_DATASET_DIR,
    STATION_DATA_PATH,
)
from llm4fairrouting.llm.client_utils import create_openai_client
from llm4fairrouting.llm.demand_extraction import extract_all_demands, extract_demands_offline
from llm4fairrouting.llm.dialogue_generation import (
    generate_dialogues_offline,
    generate_dialogues_online,
    load_stations,
    save_dialogues,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_OUTPUT_FILENAMES = {
    "events_manifest": "events_manifest.jsonl",
    "dialogues": "dialogues.jsonl",
    "llm2_sft": "llm2_sft.jsonl",
    "llm3_sft_clean": "llm3_sft_clean.jsonl",
    "llm3_sft_pipeline": "llm3_sft_pipeline.jsonl",
    "llm3_grpo_hard": "llm3_grpo_hard.jsonl",
    "quality_report": "quality_report.json",
    "release_manifest": "release_manifest.json",
    "dataset_manifest": "dataset_manifest.json",
}


def _window_label_for_slot(base_date: str, time_slot: int, window_minutes: int) -> str:
    timestamp = datetime.strptime(base_date, "%Y-%m-%d") + timedelta(minutes=time_slot * 5)
    abs_start = (timestamp.hour * 60 + timestamp.minute) // window_minutes * window_minutes
    abs_end = abs_start + window_minutes
    h_start, m_start = divmod(abs_start, 60)
    h_end, m_end = divmod(abs_end, 60)
    return (
        f"{timestamp.date().isoformat()}T{h_start:02d}:{m_start:02d}"
        f"-{h_end:02d}:{m_end:02d}"
    )


def _write_jsonl(records: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_output_paths(output_dir: str | Path) -> Dict[str, Path]:
    root = Path(output_dir)
    return {
        name: root / filename
        for name, filename in TRAINING_OUTPUT_FILENAMES.items()
    }


def _event_gold_demand(record: Dict) -> Dict:
    gold = deepcopy(record.get("gold_structured_demand") or build_gold_structured_demand(record))
    labels = gold.get("labels", {})
    attach_priority_labels(
        gold,
        latent_priority=labels.get("latent_priority", record.get("latent_priority")),
        dialogue_observable_priority=labels.get("dialogue_observable_priority"),
    )
    return gold


def _window_sample(time_window: str, demands: List[Dict], dataset_source: str) -> Dict:
    targets = build_window_priority_targets(demands, priority_field="extraction_observable_priority")
    return {
        "time_window": time_window,
        "dataset_source": dataset_source,
        "demands": demands,
        "priority_labels": targets["demand_configs"],
        "pairwise_preferences": targets["pairwise_preferences"],
        "critical_topk_targets": targets["critical_topk_targets"],
    }


def _build_clean_structured_windows(
    event_records: List[Dict],
    *,
    base_date: str,
    window_minutes: int,
) -> List[Dict]:
    by_window: Dict[str, List[Dict]] = {}
    for record in event_records:
        label = _window_label_for_slot(base_date, int(record.get("time_slot", 0)), window_minutes)
        by_window.setdefault(label, []).append(_event_gold_demand(record))
    return [
        _window_sample(time_window=label, demands=demands, dataset_source="clean_structured")
        for label, demands in sorted(by_window.items())
    ]


def _build_llm2_sft_records(dialogues: List[Dict]) -> List[Dict]:
    records = []
    for dialogue in dialogues:
        annotations = dialogue.get("annotations", {})
        records.append({
            "dialogue_id": dialogue.get("dialogue_id"),
            "event_id": dialogue.get("metadata", {}).get("event_id"),
            "dialogue_style": annotations.get("dialogue_style", "direct"),
            "conversation": dialogue.get("conversation", ""),
            "gold_extraction": annotations.get("gold_structured_demand"),
            "dialogue_audit": dialogue.get("audit", {}),
        })
    return records


def _append_context_signal(demand: Dict, note: str) -> None:
    demand["context_signals"] = list(
        dict.fromkeys(list(demand.get("context_signals", [])) + [note])
    )


def _refresh_synthetic_labels(demand: Dict) -> Dict:
    """Recompute labels for synthetic hard cases and align latent to the new variant."""
    labels = dict(demand.get("labels", {}) or {})
    attach_priority_labels(
        demand,
        latent_priority=labels.get("latent_priority"),
        dialogue_observable_priority=labels.get("dialogue_observable_priority"),
    )
    synthetic_latent = int(demand.get("labels", {}).get("extraction_observable_priority", 4))
    attach_priority_labels(demand, latent_priority=synthetic_latent)
    demand["latent_priority"] = synthetic_latent
    return demand


def _counterfactual_variants(gold: Dict) -> List[Dict]:
    variants = []

    deadline_variant = deepcopy(gold)
    if int(deadline_variant.get("time_constraint", {}).get("deadline_minutes", 120) or 120) > 15:
        deadline_variant["demand_id"] = f"{deadline_variant['demand_id']}_cf_deadline15"
        deadline_variant["time_constraint"]["deadline_minutes"] = 15
        deadline_variant["time_constraint"]["type"] = "hard"
        deadline_variant["time_constraint"]["description"] = "Counterfactual hard deadline within 15 minutes"
        deadline_variant["priority_evaluation_signals"]["time_sensitivity"] = "Immediate action required"
        _append_context_signal(deadline_variant, "Counterfactual: deadline tightened to 15 minutes")
        _refresh_synthetic_labels(deadline_variant)
        variants.append(deadline_variant)

    role_variant = deepcopy(gold)
    if role_variant.get("requester_role") != "emergency_doctor":
        role_variant["demand_id"] = f"{role_variant['demand_id']}_cf_emergency_role"
        role_variant["requester_role"] = "emergency_doctor"
        role_variant["priority_evaluation_signals"]["requester_role"] = "emergency_doctor"
        _append_context_signal(role_variant, "Counterfactual: requester role upgraded to emergency doctor")
        _refresh_synthetic_labels(role_variant)
        variants.append(role_variant)

    handling_variant = deepcopy(gold)
    special_handling = list(handling_variant.get("special_handling", []))
    if "cold_chain" not in special_handling:
        handling_variant["demand_id"] = f"{handling_variant['demand_id']}_cf_cold_chain"
        handling_variant["special_handling"] = special_handling + ["cold_chain"]
        handling_variant["priority_evaluation_signals"]["special_handling"] = list(handling_variant["special_handling"])
        _append_context_signal(handling_variant, "Counterfactual: cold-chain handling added")
        _refresh_synthetic_labels(handling_variant)
        variants.append(handling_variant)

    readiness_variant = deepcopy(gold)
    if readiness_variant.get("receiver_ready", False):
        readiness_variant["demand_id"] = f"{readiness_variant['demand_id']}_cf_not_ready"
        readiness_variant["receiver_ready"] = False
        readiness_variant["operational_readiness"] = "Standard handoff readiness"
        readiness_variant["priority_evaluation_signals"]["operational_readiness"] = "Standard handoff readiness"
        _append_context_signal(readiness_variant, "Counterfactual: receiver is not ready for immediate handoff")
        _refresh_synthetic_labels(readiness_variant)
        variants.append(readiness_variant)

    vulnerability_variant = deepcopy(gold)
    vulnerability = dict(vulnerability_variant.get("population_vulnerability", {}) or {})
    if not vulnerability.get("vulnerable_community"):
        vulnerability.update({
            "elderly_ratio": max(float(vulnerability.get("elderly_ratio", 0.0) or 0.0), 0.55),
            "population": max(int(vulnerability.get("population", 0) or 0), 2500),
            "elderly_involved": True,
            "children_involved": True,
            "vulnerable_community": True,
        })
        vulnerability_variant["demand_id"] = f"{vulnerability_variant['demand_id']}_cf_vulnerable"
        vulnerability_variant["population_vulnerability"] = vulnerability
        vulnerability_variant["priority_evaluation_signals"]["population_vulnerability"] = dict(vulnerability)
        _append_context_signal(vulnerability_variant, "Counterfactual: vulnerable community evidence strengthened")
        _refresh_synthetic_labels(vulnerability_variant)
        variants.append(vulnerability_variant)

    destination_variant = deepcopy(gold)
    destination = dict(destination_variant.get("destination", {}) or {})
    if str(destination.get("type", "")) not in {"hospital", "clinic", "community_health_center"}:
        destination_variant["demand_id"] = f"{destination_variant['demand_id']}_cf_clinic"
        destination["type"] = "clinic"
        destination_variant["destination"] = destination
        destination_variant["priority_evaluation_signals"]["nearby_critical_facility"] = "clinic"
        _append_context_signal(destination_variant, "Counterfactual: receiving point changed to a clinic handoff")
        _refresh_synthetic_labels(destination_variant)
        variants.append(destination_variant)

    return variants


def _surface_contradiction_windows(flattened: List[Dict], *, max_windows: int = 4) -> List[List[Dict]]:
    if len(flattened) < 2:
        return []

    ranked = sorted(
        (deepcopy(demand) for demand in flattened),
        key=lambda item: (
            int(item.get("labels", {}).get("extraction_observable_priority", 4)),
            -int(item.get("labels", {}).get("extraction_observable_score", 0)),
            str(item.get("demand_id", "")),
        ),
    )
    windows: List[List[Dict]] = []
    for index in range(min(max_windows, len(ranked) // 2)):
        high_need = deepcopy(ranked[index])
        low_need = deepcopy(ranked[-(index + 1)])
        if high_need.get("demand_id") == low_need.get("demand_id"):
            continue

        low_need["demand_id"] = f"{low_need['demand_id']}_surface_urgent_{index + 1}"
        low_need["priority_evaluation_signals"]["medical_urgency_self_report"] = (
            "The requester repeatedly says this feels urgent, but provides no stronger structural evidence."
        )
        low_need["priority_evaluation_signals"]["time_sensitivity"] = (
            "The caller sounds highly anxious and keeps using urgent wording."
        )
        _append_context_signal(low_need, "Hard case: urgent wording without strong structural need")
        _refresh_synthetic_labels(low_need)

        high_need["demand_id"] = f"{high_need['demand_id']}_surface_calm_{index + 1}"
        high_need["priority_evaluation_signals"]["medical_urgency_self_report"] = (
            "The requester speaks calmly and clinically while the structural need remains unchanged."
        )
        high_need["priority_evaluation_signals"]["time_sensitivity"] = (
            "The team communicates in a calm tone, but the operational deadline still stands."
        )
        _append_context_signal(high_need, "Hard case: calm wording despite stronger structural need")
        _refresh_synthetic_labels(high_need)
        windows.append([high_need, low_need])
    return windows


def _near_tie_windows(flattened: List[Dict], *, max_windows: int = 4) -> List[List[Dict]]:
    by_priority: Dict[int, List[Dict]] = {}
    for demand in flattened:
        priority = int(demand.get("labels", {}).get("extraction_observable_priority", 4))
        by_priority.setdefault(priority, []).append(deepcopy(demand))

    windows: List[List[Dict]] = []
    for priority in sorted(by_priority):
        candidates = by_priority[priority]
        if len(candidates) < 2:
            continue
        candidates.sort(
            key=lambda item: (
                int(item.get("labels", {}).get("extraction_observable_score", 0)),
                str(item.get("demand_id", "")),
            )
        )
        tightest_pair = min(
            zip(candidates, candidates[1:]),
            key=lambda pair: abs(
                int(pair[0].get("labels", {}).get("extraction_observable_score", 0))
                - int(pair[1].get("labels", {}).get("extraction_observable_score", 0))
            ),
        )
        neighbor = None
        for delta in (1, -1, 2):
            neighbor_priority = priority + delta
            if neighbor_priority in by_priority and by_priority[neighbor_priority]:
                neighbor = deepcopy(
                    sorted(
                        by_priority[neighbor_priority],
                        key=lambda item: (
                            -int(item.get("labels", {}).get("extraction_observable_score", 0)),
                            str(item.get("demand_id", "")),
                        ),
                    )[0]
                )
                break
        if neighbor is None:
            continue
        windows.append([deepcopy(tightest_pair[0]), deepcopy(tightest_pair[1]), neighbor])
        if len(windows) >= max_windows:
            break
    return windows


def _build_hard_contrastive_windows(clean_windows: List[Dict]) -> List[Dict]:
    hard_windows: List[Dict] = []
    for window in clean_windows[: min(12, len(clean_windows))]:
        demands = window.get("demands", [])
        if not demands:
            continue
        base = deepcopy(demands[0])
        variants = _counterfactual_variants(base)
        for chunk_index in range(0, len(variants), 2):
            chunk = variants[chunk_index: chunk_index + 2]
            if not chunk:
                continue
            hard_windows.append(
                _window_sample(
                    time_window=f"{window['time_window']}::counterfactual_{chunk_index // 2 + 1}",
                    demands=[deepcopy(base)] + chunk,
                    dataset_source="hard_contrastive",
                )
            )

    flattened = [deepcopy(demand) for window in clean_windows for demand in window.get("demands", [])]
    flattened.sort(
        key=lambda item: (
            int(item.get("labels", {}).get("extraction_observable_priority", 4)),
            str(item.get("demand_id", "")),
        )
    )
    if len(flattened) >= 4:
        hard_mix = [flattened[0], flattened[min(1, len(flattened) - 1)], flattened[-2], flattened[-1]]
        hard_windows.append(
            _window_sample(
                time_window="hard_window::mixed_priority",
                demands=hard_mix,
                dataset_source="hard_contrastive",
            )
        )

    for index, surface_contrast in enumerate(_surface_contradiction_windows(flattened), start=1):
        hard_windows.append(
            _window_sample(
                time_window=f"hard_window::surface_contradiction_{index}",
                demands=surface_contrast,
                dataset_source="hard_contrastive",
            )
        )

    for index, near_tie in enumerate(_near_tie_windows(flattened), start=1):
        hard_windows.append(
            _window_sample(
                time_window=f"hard_window::near_tie_{index}",
                demands=near_tie,
                dataset_source="hard_contrastive",
            )
        )
    return hard_windows


def _dialogues_by_style(dialogues: List[Dict], styles: List[str]) -> Dict[str, List[Dict]]:
    by_style = {style: [] for style in styles}
    for dialogue in dialogues:
        style = str(dialogue.get("annotations", {}).get("dialogue_style", "direct"))
        by_style.setdefault(style, []).append(dialogue)
    return by_style


def _build_pipeline_structured_windows(
    *,
    dialogues: List[Dict],
    styles: List[str],
    offline: bool,
    client,
    model: str,
    window_minutes: int,
    extraction_concurrency: int = 1,
) -> tuple[List[Dict], Dict[str, List[Dict]]]:
    pipeline_windows: List[Dict] = []
    extracted_payloads: Dict[str, List[Dict]] = {}
    for style, style_dialogues in _dialogues_by_style(dialogues, styles).items():
        if not style_dialogues:
            continue
        if offline:
            extracted = extract_demands_offline(style_dialogues, window_minutes=window_minutes)
        else:
            extracted = extract_all_demands(
                style_dialogues,
                client=client,
                model=model,
                window_minutes=window_minutes,
                temperature=0.0,
                max_concurrency=extraction_concurrency,
            )
        extracted_payloads[style] = extracted
        for window in extracted:
            demands = [deepcopy(demand) for demand in window.get("demands", [])]
            if not demands:
                continue
            pipeline_windows.append(
                _window_sample(
                    time_window=f"{window['time_window']}::{style}",
                    demands=demands,
                    dataset_source="pipeline_structured",
                )
            )
    return pipeline_windows, extracted_payloads


def build_priority_training_dataset(
    *,
    event_records: Optional[List[Dict]] = None,
    building_file: str = str(BUILDING_DATA_PATH),
    output_dir: Optional[str] = str(DEMAND_TRAINING_DATASET_DIR),
    stations_path: Optional[str] = str(STATION_DATA_PATH),
    base_date: str = "2024-03-15",
    styles: Optional[List[str]] = None,
    offline: bool = True,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    batch_size: int = 5,
    window_minutes: int = 5,
    dialogue_concurrency: int = 1,
    extraction_concurrency: int = 1,
    quality_thresholds: Optional[Dict[str, float]] = None,
    **event_generation_kwargs,
) -> Dict:
    styles = [
        style for style in [
            *(styles or []),
    ] if style
    ] or list(DIALOGUE_STYLE_VARIANTS)

    output_paths = _resolve_output_paths(output_dir) if output_dir else {}
    event_manifest_path = str(output_paths["events_manifest"]) if output_paths else None
    dialogue_output_path = str(output_paths["dialogues"]) if output_paths else None

    if event_records is None:
        event_records = generate_daily_demand_dataset(
            building_file=building_file,
            manifest_file=event_manifest_path,
            **event_generation_kwargs,
        )

    stations = load_stations(stations_path) if stations_path else []
    client = None if offline else create_openai_client(api_base, api_key)
    if offline:
        dialogues = generate_dialogues_offline(event_records, stations, base_date=base_date, styles=styles)
    else:
        dialogues = generate_dialogues_online(
            event_records,
            stations,
            client=client,
            model=model,
            base_date=base_date,
            temperature=temperature,
            batch_size=batch_size,
            styles=styles,
            max_concurrency=dialogue_concurrency,
        )
    accepted_dialogues = filter_accepted_dialogues(dialogues)

    clean_windows = _build_clean_structured_windows(
        event_records,
        base_date=base_date,
        window_minutes=window_minutes,
    )
    pipeline_windows, extracted_payloads = _build_pipeline_structured_windows(
        dialogues=accepted_dialogues,
        styles=styles,
        offline=offline,
        client=client,
        model=model,
        window_minutes=window_minutes,
        extraction_concurrency=extraction_concurrency,
    )
    hard_windows = _build_hard_contrastive_windows(clean_windows)
    llm2_sft = _build_llm2_sft_records(accepted_dialogues)

    quality_thresholds = {**DEFAULT_QUALITY_THRESHOLDS, **(quality_thresholds or {})}
    quality_report = build_quality_report(
        event_records=event_records,
        dialogues=dialogues,
        pipeline_windows=pipeline_windows,
        hard_windows=hard_windows,
        thresholds=quality_thresholds,
    )

    artifacts = {
        "events_manifest": event_records,
        "dialogues": dialogues,
        "llm2_sft": llm2_sft,
        "llm3_sft_clean": clean_windows,
        "llm3_sft_pipeline": pipeline_windows,
        "llm3_grpo_hard": hard_windows,
    }
    counts = {
        name: len(records)
        for name, records in artifacts.items()
    }
    dataset_manifest = {
        "schema_version": "priority_observability_v2",
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "base_date": base_date,
        "styles": styles,
        "offline": offline,
        "model": model,
        "window_minutes": window_minutes,
        "priority_policy_version": PRIORITY_POLICY_VERSION,
        "quality_gate_policy_version": QUALITY_GATE_POLICY_VERSION,
        "output_dir": str(Path(output_dir).resolve()) if output_dir else None,
        "files": {
            name: TRAINING_OUTPUT_FILENAMES[name]
            for name in (*artifacts.keys(), "quality_report", "release_manifest")
        },
        "counts": counts,
        "raw_counts": {
            "dialogues": len(dialogues),
            "accepted_dialogues": len(accepted_dialogues),
        },
        "extracted_demands_by_style": {
            style: len(windows)
            for style, windows in extracted_payloads.items()
        },
        "quality_thresholds": quality_thresholds,
    }
    release_manifest = build_release_manifest(
        dataset_manifest=dataset_manifest,
        quality_report=quality_report,
    )
    dataset_manifest["release_status"] = release_manifest["release_status"]

    if output_paths:
        save_event_manifest(event_records, str(output_paths["events_manifest"]))
        save_dialogues(dialogues, str(output_paths["dialogues"]))
        _write_jsonl(llm2_sft, output_paths["llm2_sft"])
        _write_jsonl(clean_windows, output_paths["llm3_sft_clean"])
        _write_jsonl(pipeline_windows, output_paths["llm3_sft_pipeline"])
        _write_jsonl(hard_windows, output_paths["llm3_grpo_hard"])
        output_paths["quality_report"].write_text(
            json.dumps(quality_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_paths["release_manifest"].write_text(
            json.dumps(release_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_path = output_paths["dataset_manifest"]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(dataset_manifest, handle, ensure_ascii=False, indent=2)

    return {
        **dataset_manifest,
        "artifacts": artifacts,
        "quality_report": quality_report,
        "release_manifest": release_manifest,
    }


def main() -> None:
    active_env_file = prepare_env_file(PROJECT_ROOT)
    parser = argparse.ArgumentParser(
        description="Build observable-priority training data for the LLM2/LLM3 pipeline.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(active_env_file) if active_env_file else None,
        help="Environment file path; defaults to the project .env when present",
    )
    parser.add_argument("--building-input", default=str(BUILDING_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEMAND_TRAINING_DATASET_DIR))
    parser.add_argument("--stations", default=str(STATION_DATA_PATH))
    parser.add_argument("--base-date", default="2024-03-15")
    parser.add_argument("--styles", nargs="+", default=list(DIALOGUE_STYLE_VARIANTS))
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--api-base", default=env_text("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=env_text("OPENAI_API_KEY"))
    parser.add_argument("--model", default=env_text("LLM4FAIRROUTING_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-minutes", type=int, default=5)
    parser.add_argument("--demands-per-window-min", type=int, default=4)
    parser.add_argument("--demands-per-window-max", type=int, default=10)
    parser.add_argument("--medical-ratio", type=float, default=0.2)
    parser.add_argument("--num-supply-medical", type=int, default=5)
    parser.add_argument("--num-supply-commercial", type=int, default=5)
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
    parser.add_argument("--dialogue-concurrency", type=int, default=1)
    parser.add_argument("--extraction-concurrency", type=int, default=1)
    args = parser.parse_args()

    dataset = build_priority_training_dataset(
        building_file=args.building_input,
        output_dir=args.output_dir,
        stations_path=args.stations,
        base_date=args.base_date,
        styles=args.styles,
        offline=args.offline,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        batch_size=args.batch_size,
        window_minutes=args.window,
        seed=args.seed,
        time_window_minutes=args.window_minutes,
        demands_per_window_min=args.demands_per_window_min,
        demands_per_window_max=args.demands_per_window_max,
        medical_ratio=args.medical_ratio,
        num_supply_medical=args.num_supply_medical,
        num_supply_commercial=args.num_supply_commercial,
        dialogue_concurrency=args.dialogue_concurrency,
        extraction_concurrency=args.extraction_concurrency,
        quality_thresholds={
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
        },
    )
    print(
        f"Built training dataset in {dataset['output_dir']} with "
        f"{dataset['counts']['events_manifest']} events, "
        f"{dataset['counts']['dialogues']} dialogues, "
        f"{dataset['counts']['llm2_sft']} accepted LLM2 samples. "
        f"Release status: {dataset['release_status']}"
    )


if __name__ == "__main__":
    main()
