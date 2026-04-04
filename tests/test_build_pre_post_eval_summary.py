from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from evals.build_pre_post_eval_summary import (
    build_pre_post_summary,
    render_summary_markdown,
)


def _make_case_dir(prefix: str) -> Path:
    path = Path.cwd() / ".test_artifacts" / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_build_pre_post_summary_for_rank_only_mode() -> None:
    base = _make_case_dir("pre_post_summary_rank")
    eval_dir = base / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)

    (base / "fixed_inputs").mkdir(parents=True, exist_ok=True)
    selection_manifest = base / "fixed_inputs" / "selection_manifest.json"
    selection_manifest.write_text(
        json.dumps(
            {
                "selection_mode": "window_indices",
                "n_windows_selected": 3,
                "selected_time_windows": ["W0", "W1", "W2"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pre_alignment = eval_dir / "pre_alignment.json"
    post_alignment = eval_dir / "post_alignment.json"
    delta_path = eval_dir / "post_vs_pre_alignment_delta.json"
    pre_alignment.write_text(
        json.dumps(
            {
                "n_aligned_demands": 12,
                "truth_source": "fixed_demands",
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "weighted_f1": 0.45,
                "spearman": 0.1,
                "kendall_tau": 0.05,
                "top_k_hit_rate": {"hit_rate": 0.4},
                "priority_1_metrics": {"recall": 0.2, "f1": 0.2},
                "urgent_metrics": {"recall": 0.3, "f1": 0.3},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    post_alignment.write_text(
        json.dumps(
            {
                "n_aligned_demands": 12,
                "truth_source": "fixed_demands",
                "accuracy": 0.75,
                "macro_f1": 0.7,
                "weighted_f1": 0.72,
                "spearman": 0.4,
                "kendall_tau": 0.25,
                "top_k_hit_rate": {"hit_rate": 0.8},
                "priority_1_metrics": {"recall": 0.9, "f1": 0.8},
                "urgent_metrics": {"recall": 0.85, "f1": 0.82},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    delta_path.write_text(
        json.dumps(
            {
                "delta_metrics_post_minus_pre": {
                    "accuracy": 0.25,
                    "macro_f1": 0.3,
                    "weighted_f1": 0.27,
                    "spearman": 0.3,
                    "kendall_tau": 0.2,
                    "top_k_hit_rate": 0.4,
                    "priority_1_recall": 0.7,
                    "priority_1_f1": 0.6,
                    "urgent_recall": 0.55,
                    "urgent_f1": 0.52,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = eval_dir / "eval_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "mode": "rank_only_alignment",
                "rank_only_mode": "llm3_only",
                "truth_source": "fixed_demands",
                "selection_manifest": str(selection_manifest),
                "slot_sampling": "",
                "pre_alignment": str(pre_alignment),
                "post_alignment": str(post_alignment),
                "post_vs_pre_alignment_delta": str(delta_path),
                "pre_run_dir": "/tmp/pre",
                "post_run_dir": "/tmp/post",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = build_pre_post_summary(manifest)

    assert summary["verdicts"]["primary"]["overall"] == "post_better"
    assert summary["alignment"]["delta_post_minus_pre"]["accuracy"] == 0.25
    assert summary["sample"]["n_selected_windows"] == 3

    markdown = render_summary_markdown(summary)
    assert "POST looks better than PRE" in markdown
    assert "Alignment Delta (Post - Pre)" in markdown


def test_build_pre_post_summary_for_operational_mode() -> None:
    base = _make_case_dir("pre_post_summary_operational")
    eval_dir = base / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)

    slot_sampling = eval_dir / "slot_sampling.json"
    slot_sampling.write_text(
        json.dumps(
            {
                "mode": "fixed_window_stratified_sample",
                "selected_time_windows": ["W0", "W1"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pre_alignment = eval_dir / "pre_alignment.json"
    post_alignment = eval_dir / "post_alignment.json"
    operational = eval_dir / "post_vs_pre_operational_impact.json"
    for path, accuracy in ((pre_alignment, 0.6), (post_alignment, 0.62)):
        path.write_text(
            json.dumps(
                {
                    "n_aligned_demands": 20,
                    "truth_source": "fixed_demands",
                    "accuracy": accuracy,
                    "macro_f1": accuracy,
                    "weighted_f1": accuracy,
                    "spearman": accuracy,
                    "kendall_tau": accuracy,
                    "top_k_hit_rate": {"hit_rate": accuracy},
                    "priority_1_metrics": {"recall": accuracy, "f1": accuracy},
                    "urgent_metrics": {"recall": accuracy, "f1": accuracy},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    operational.write_text(
        json.dumps(
            {
                "reference_method": "pre",
                "methods": {
                    "pre": {
                        "n_aligned_demands": 20,
                        "overall": {"service_rate": 0.6, "on_time_rate": 0.5},
                        "priority_1": {"service_rate": 0.5, "on_time_rate": 0.4, "avg_delivery_latency_min": 18.0},
                        "urgent": {"service_rate": 0.55, "on_time_rate": 0.45, "avg_delivery_latency_min": 19.0},
                        "priority_weighted_service_score": 0.6,
                        "priority_weighted_on_time_score": 0.5,
                    },
                    "post": {
                        "n_aligned_demands": 20,
                        "overall": {"service_rate": 0.7, "on_time_rate": 0.65},
                        "priority_1": {"service_rate": 0.7, "on_time_rate": 0.6, "avg_delivery_latency_min": 12.0},
                        "urgent": {"service_rate": 0.72, "on_time_rate": 0.61, "avg_delivery_latency_min": 13.0},
                        "priority_weighted_service_score": 0.72,
                        "priority_weighted_on_time_score": 0.68,
                    },
                },
                "comparisons": {
                    "post_vs_pre": {
                        "priority_1_latency_improvement": 0.333333,
                        "urgent_latency_improvement": 0.315789,
                        "priority_1_service_rate_gain": 0.2,
                        "urgent_service_rate_gain": 0.17,
                        "priority_1_on_time_rate_gain": 0.2,
                        "urgent_on_time_rate_gain": 0.16,
                        "priority_weighted_service_gain": 0.12,
                        "priority_weighted_on_time_gain": 0.18,
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = eval_dir / "eval_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "mode": "operational_impact_sampled",
                "operational_mode": "llm3_only",
                "truth_source": "fixed_demands",
                "selection_manifest": "",
                "slot_sampling": str(slot_sampling),
                "pre_alignment": str(pre_alignment),
                "post_alignment": str(post_alignment),
                "post_vs_pre_operational_impact": str(operational),
                "pre_run_dir": "/tmp/pre",
                "post_run_dir": "/tmp/post",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = build_pre_post_summary(manifest)

    assert summary["verdicts"]["primary"]["overall"] == "post_better"
    assert summary["operational"]["comparison"]["priority_1_on_time_rate_gain"] == 0.2
    assert summary["sample"]["n_selected_windows"] == 2

    markdown = render_summary_markdown(summary)
    assert "Operational Comparison" in markdown
    assert "priority_weighted_on_time_gain" in markdown
