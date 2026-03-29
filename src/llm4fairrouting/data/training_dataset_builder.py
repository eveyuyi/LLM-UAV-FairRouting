"""Build training corpora with observable priority labels for LLM2 and LLM3."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from llm4fairrouting.config.runtime_env import env_text, prepare_env_file
from llm4fairrouting.data.demand_event_generation import (
    DIALOGUE_STYLE_VARIANTS,
    build_gold_structured_demand,
    generate_daily_demand_dataset,
    generate_daily_demand_records,
    save_event_manifest,
)
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


def _counterfactual_variants(gold: Dict) -> List[Dict]:
    variants = []

    deadline_variant = deepcopy(gold)
    if int(deadline_variant.get("time_constraint", {}).get("deadline_minutes", 120) or 120) > 15:
        deadline_variant["demand_id"] = f"{deadline_variant['demand_id']}_cf_deadline15"
        deadline_variant["time_constraint"]["deadline_minutes"] = 15
        deadline_variant["time_constraint"]["type"] = "hard"
        deadline_variant["time_constraint"]["description"] = "Counterfactual hard deadline within 15 minutes"
        deadline_variant["priority_evaluation_signals"]["time_sensitivity"] = "Immediate action required"
        deadline_variant["context_signals"] = list(deadline_variant.get("context_signals", [])) + [
            "Counterfactual: deadline tightened to 15 minutes",
        ]
        attach_priority_labels(deadline_variant, latent_priority=gold.get("labels", {}).get("latent_priority"))
        variants.append(deadline_variant)

    role_variant = deepcopy(gold)
    if role_variant.get("requester_role") != "emergency_doctor":
        role_variant["demand_id"] = f"{role_variant['demand_id']}_cf_emergency_role"
        role_variant["requester_role"] = "emergency_doctor"
        role_variant["priority_evaluation_signals"]["requester_role"] = "emergency_doctor"
        role_variant["context_signals"] = list(role_variant.get("context_signals", [])) + [
            "Counterfactual: requester role upgraded to emergency doctor",
        ]
        attach_priority_labels(role_variant, latent_priority=gold.get("labels", {}).get("latent_priority"))
        variants.append(role_variant)

    handling_variant = deepcopy(gold)
    special_handling = list(handling_variant.get("special_handling", []))
    if "cold_chain" not in special_handling:
        handling_variant["demand_id"] = f"{handling_variant['demand_id']}_cf_cold_chain"
        handling_variant["special_handling"] = special_handling + ["cold_chain"]
        handling_variant["priority_evaluation_signals"]["special_handling"] = list(handling_variant["special_handling"])
        handling_variant["context_signals"] = list(handling_variant.get("context_signals", [])) + [
            "Counterfactual: cold-chain handling added",
        ]
        attach_priority_labels(handling_variant, latent_priority=gold.get("labels", {}).get("latent_priority"))
        variants.append(handling_variant)

    return variants


def _build_hard_contrastive_windows(clean_windows: List[Dict]) -> List[Dict]:
    hard_windows: List[Dict] = []
    for window in clean_windows[: min(12, len(clean_windows))]:
        demands = window.get("demands", [])
        if not demands:
            continue
        base = deepcopy(demands[0])
        variants = _counterfactual_variants(base)
        if variants:
            hard_windows.append(
                _window_sample(
                    time_window=f"{window['time_window']}::counterfactual",
                    demands=[base] + variants[:2],
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
    events_csv_path: Optional[str] = None,
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
        _, event_records = generate_daily_demand_dataset(
            building_file=building_file,
            output_file=events_csv_path,
            manifest_file=event_manifest_path,
            **event_generation_kwargs,
        )
    else:
        if event_manifest_path:
            save_event_manifest(event_records, event_manifest_path)

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
        )
    if dialogue_output_path:
        save_dialogues(dialogues, dialogue_output_path)

    clean_windows = _build_clean_structured_windows(
        event_records,
        base_date=base_date,
        window_minutes=window_minutes,
    )
    pipeline_windows, extracted_payloads = _build_pipeline_structured_windows(
        dialogues=dialogues,
        styles=styles,
        offline=offline,
        client=client,
        model=model,
        window_minutes=window_minutes,
    )
    hard_windows = _build_hard_contrastive_windows(clean_windows)
    llm2_sft = _build_llm2_sft_records(dialogues)

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
        "output_dir": str(Path(output_dir).resolve()) if output_dir else None,
        "files": {
            name: TRAINING_OUTPUT_FILENAMES[name]
            for name in artifacts
        },
        "counts": counts,
        "extracted_demands_by_style": {
            style: len(windows)
            for style, windows in extracted_payloads.items()
        },
    }

    if output_paths:
        save_event_manifest(event_records, str(output_paths["events_manifest"]))
        save_dialogues(dialogues, str(output_paths["dialogues"]))
        _write_jsonl(llm2_sft, output_paths["llm2_sft"])
        _write_jsonl(clean_windows, output_paths["llm3_sft_clean"])
        _write_jsonl(pipeline_windows, output_paths["llm3_sft_pipeline"])
        _write_jsonl(hard_windows, output_paths["llm3_grpo_hard"])
        manifest_path = output_paths["dataset_manifest"]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(dataset_manifest, handle, ensure_ascii=False, indent=2)

    return {
        **dataset_manifest,
        "artifacts": artifacts,
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
    parser.add_argument("--events-csv", default=None)
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
    args = parser.parse_args()

    dataset = build_priority_training_dataset(
        building_file=args.building_input,
        events_csv_path=args.events_csv,
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
    )
    print(
        f"Built training dataset in {dataset['output_dir']} with "
        f"{dataset['counts']['events_manifest']} events, "
        f"{dataset['counts']['dialogues']} dialogues, "
        f"{dataset['counts']['llm2_sft']} LLM2 samples."
    )


if __name__ == "__main__":
    main()
