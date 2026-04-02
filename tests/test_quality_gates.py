from __future__ import annotations

import json

from llm4fairrouting.data.quality_gates import build_quality_report, build_release_manifest
from llm4fairrouting.data.release_manifest_builder import build_aggregate_release_manifest


def _make_dialogue(dialogue_id: str, latent_priority: int, *, passed: bool, missing: list[str] | None = None) -> dict:
    return {
        "dialogue_id": dialogue_id,
        "annotations": {
            "latent_priority": latent_priority,
            "must_mention_factors": [
                {"name": "scenario_context"},
                {"name": "deadline_minutes"},
                {"name": "requester_role"},
            ],
        },
        "audit": {
            "passed": passed,
            "observability_score": 1.0 if passed else 0.6,
            "missing_must_mention_factors": list(missing or []),
        },
    }


def test_quality_report_detects_failed_priority_and_requester_role_gates():
    events = [
        {"event_id": "E1", "requester_role": "emergency_doctor"},
        {"event_id": "E2", "requester_role": "consumer"},
    ]
    dialogues = [
        _make_dialogue("D1", 1, passed=False, missing=["scenario_context"]),
        _make_dialogue("D2", 2, passed=False, missing=["scenario_context"]),
        _make_dialogue("D3", 4, passed=True),
        _make_dialogue("D4", 4, passed=True),
    ]
    pipeline_windows = [{
        "time_window": "2024-03-15T00:00-00:30::direct",
        "demands": [
            {"source_event_id": "E1", "requester_role": None},
            {"source_event_id": "E2", "requester_role": "consumer"},
        ],
    }]
    hard_windows = [
        {"time_window": "hard_window::counterfactual_1"},
        {"time_window": "hard_window::surface_contradiction_1"},
    ]

    report = build_quality_report(
        event_records=events,
        dialogues=dialogues,
        pipeline_windows=pipeline_windows,
        hard_windows=hard_windows,
    )
    dataset_manifest = {
        "counts": {
            "llm3_sft_clean": 2,
            "llm2_sft": 1,
            "llm3_sft_pipeline": 1,
            "llm3_grpo_hard": 2,
        },
    }
    release = build_release_manifest(dataset_manifest=dataset_manifest, quality_report=report)

    assert not report["checks"]["audit_pass_rate"]["passed"]
    assert not report["checks"]["priority_1_audit_pass_rate"]["passed"]
    assert not report["checks"]["requester_role_missing_rate"]["passed"]
    assert not report["checks"]["surface_contradiction_count"]["passed"]
    assert release["release_status"] == "needs_regen"
    assert release["recommended_training_files"] == ["llm3_sft_clean"]


def test_aggregate_release_manifest_collects_recommended_files(tmp_path):
    root = tmp_path / "batch"
    root.mkdir()
    shard = root / "seed_1"
    shard.mkdir()
    (shard / "dataset_manifest.json").write_text(json.dumps({
        "files": {
            "llm2_sft": "llm2_sft.jsonl",
            "llm3_sft_clean": "llm3_sft_clean.jsonl",
        },
        "counts": {
            "llm2_sft": 10,
            "llm3_sft_clean": 4,
        },
    }), encoding="utf-8")
    (shard / "release_manifest.json").write_text(json.dumps({
        "release_status": "accepted",
        "recommended_training_files": ["llm2_sft", "llm3_sft_clean"],
    }), encoding="utf-8")
    (shard / "llm2_sft.jsonl").write_text("", encoding="utf-8")
    (shard / "llm3_sft_clean.jsonl").write_text("", encoding="utf-8")

    manifest = build_aggregate_release_manifest(root)

    assert manifest["status_counts"] == {"accepted": 1}
    assert manifest["aggregate_counts"]["llm2_sft"] == 10
    assert manifest["aggregate_counts"]["llm3_sft_clean"] == 4
    assert len(manifest["recommended_training_files"]["llm2_sft"]) == 1
