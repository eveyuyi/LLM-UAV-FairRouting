"""Build a compact, human-readable summary for pre/post evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def _load_json(path: str | Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _optional_json(path_text: object) -> Optional[Dict[str, object]]:
    path = str(path_text or "").strip()
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return _load_json(candidate)


def _run_meta(run_dir_text: object) -> Optional[Dict[str, object]]:
    run_dir = str(run_dir_text or "").strip()
    if not run_dir:
        return None
    return _optional_json(Path(run_dir) / "run_meta.json")


def _nested(payload: Optional[Dict[str, object]], path: Sequence[str]) -> Optional[object]:
    current: object = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _float_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _diff(post_value: object, pre_value: object) -> Optional[float]:
    post_float = _float_or_none(post_value)
    pre_float = _float_or_none(pre_value)
    if post_float is None or pre_float is None:
        return None
    return round(post_float - pre_float, 6)


def _preview(items: Iterable[object], limit: int = 10) -> List[object]:
    values = list(items)
    if len(values) <= limit:
        return values
    return values[:limit]


ALIGNMENT_METRICS = {
    "accuracy": ("accuracy",),
    "macro_f1": ("macro_f1",),
    "weighted_f1": ("weighted_f1",),
    "spearman": ("spearman",),
    "kendall_tau": ("kendall_tau",),
    "top_k_hit_rate": ("top_k_hit_rate", "hit_rate"),
    "priority_1_recall": ("priority_1_metrics", "recall"),
    "priority_1_f1": ("priority_1_metrics", "f1"),
    "urgent_recall": ("urgent_metrics", "recall"),
    "urgent_f1": ("urgent_metrics", "f1"),
}

OPERATIONAL_COMPARISON_METRICS = {
    "priority_1_latency_improvement": ("priority_1_latency_improvement",),
    "urgent_latency_improvement": ("urgent_latency_improvement",),
    "priority_1_service_rate_gain": ("priority_1_service_rate_gain",),
    "urgent_service_rate_gain": ("urgent_service_rate_gain",),
    "priority_1_on_time_rate_gain": ("priority_1_on_time_rate_gain",),
    "urgent_on_time_rate_gain": ("urgent_on_time_rate_gain",),
    "priority_weighted_service_gain": ("priority_weighted_service_gain",),
    "priority_weighted_on_time_gain": ("priority_weighted_on_time_gain",),
}


def _alignment_snapshot(payload: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not payload:
        return {}
    metrics = {
        metric_name: _float_or_none(_nested(payload, path))
        for metric_name, path in ALIGNMENT_METRICS.items()
    }
    metrics["n_aligned_demands"] = payload.get("n_aligned_demands")
    metrics["truth_source"] = payload.get("truth_source")
    return metrics


def _alignment_delta(pre: Optional[Dict[str, object]], post: Optional[Dict[str, object]]) -> Dict[str, Optional[float]]:
    return {
        metric_name: _diff(_nested(post, path), _nested(pre, path))
        for metric_name, path in ALIGNMENT_METRICS.items()
    }


def _operational_snapshot(payload: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not payload:
        return {}

    methods = payload.get("methods") or {}
    if not isinstance(methods, dict):
        methods = {}
    comparisons = payload.get("comparisons") or {}
    if not isinstance(comparisons, dict):
        comparisons = {}

    method_snapshots: Dict[str, Dict[str, object]] = {}
    for method_name, method_payload in methods.items():
        if not isinstance(method_payload, dict):
            continue
        method_snapshots[str(method_name)] = {
            "n_aligned_demands": method_payload.get("n_aligned_demands"),
            "overall_service_rate": _float_or_none(_nested(method_payload, ("overall", "service_rate"))),
            "overall_on_time_rate": _float_or_none(_nested(method_payload, ("overall", "on_time_rate"))),
            "priority_1_service_rate": _float_or_none(_nested(method_payload, ("priority_1", "service_rate"))),
            "priority_1_on_time_rate": _float_or_none(_nested(method_payload, ("priority_1", "on_time_rate"))),
            "priority_1_avg_latency_min": _float_or_none(_nested(method_payload, ("priority_1", "avg_delivery_latency_min"))),
            "urgent_service_rate": _float_or_none(_nested(method_payload, ("urgent", "service_rate"))),
            "urgent_on_time_rate": _float_or_none(_nested(method_payload, ("urgent", "on_time_rate"))),
            "urgent_avg_latency_min": _float_or_none(_nested(method_payload, ("urgent", "avg_delivery_latency_min"))),
            "priority_weighted_service_score": _float_or_none(method_payload.get("priority_weighted_service_score")),
            "priority_weighted_on_time_score": _float_or_none(method_payload.get("priority_weighted_on_time_score")),
        }

    comparison_key = next(iter(comparisons.keys()), None)
    comparison_payload = comparisons.get(comparison_key, {}) if comparison_key else {}
    if not isinstance(comparison_payload, dict):
        comparison_payload = {}
    comparison_metrics = {
        metric_name: _float_or_none(_nested(comparison_payload, path))
        for metric_name, path in OPERATIONAL_COMPARISON_METRICS.items()
    }

    return {
        "reference_method": payload.get("reference_method"),
        "comparison_key": comparison_key,
        "methods": method_snapshots,
        "comparison": comparison_metrics,
    }


def _score_metrics(metrics: Dict[str, Optional[float]], preferred_positive: Iterable[str]) -> Dict[str, object]:
    positive = set(preferred_positive)
    better: List[str] = []
    worse: List[str] = []
    unchanged: List[str] = []
    available: Dict[str, float] = {}

    for metric_name, metric_value in metrics.items():
        if metric_value is None:
            continue
        available[metric_name] = metric_value
        if abs(metric_value) < 1e-9:
            unchanged.append(metric_name)
        elif metric_name in positive:
            if metric_value > 0:
                better.append(metric_name)
            else:
                worse.append(metric_name)
        else:
            if metric_value < 0:
                better.append(metric_name)
            else:
                worse.append(metric_name)

    if not available:
        overall = "inconclusive"
    elif better and not worse:
        overall = "post_better"
    elif worse and not better:
        overall = "pre_better"
    elif len(better) >= len(worse) + 2:
        overall = "post_better"
    elif len(worse) >= len(better) + 2:
        overall = "pre_better"
    else:
        overall = "mixed"

    return {
        "overall": overall,
        "better_metrics_for_post": sorted(better),
        "worse_metrics_for_post": sorted(worse),
        "unchanged_metrics": sorted(unchanged),
        "available_metrics": available,
    }


def _sample_snapshot(manifest: Dict[str, object]) -> Dict[str, object]:
    selection_manifest = _optional_json(manifest.get("selection_manifest"))
    slot_sampling = _optional_json(manifest.get("slot_sampling"))
    fixed_demands = str(manifest.get("fixed_extracted_demands") or "").strip()

    snapshot: Dict[str, object] = {
        "fixed_extracted_demands": fixed_demands or None,
        "selection_mode": None,
        "n_selected_windows": None,
        "n_selected_time_slots": None,
        "selected_time_windows_preview": [],
        "selected_time_slots_preview": [],
    }

    if selection_manifest:
        selected_windows = selection_manifest.get("selected_time_windows") or []
        if not isinstance(selected_windows, list):
            selected_windows = []
        snapshot.update(
            {
                "selection_mode": selection_manifest.get("selection_mode"),
                "n_selected_windows": selection_manifest.get("n_windows_selected"),
                "selected_time_windows_preview": _preview(selected_windows),
            }
        )

    if slot_sampling:
        selected_slots = slot_sampling.get("selected_time_slots") or []
        if not isinstance(selected_slots, list):
            selected_slots = []
        selected_windows = slot_sampling.get("selected_time_windows") or []
        if not isinstance(selected_windows, list):
            selected_windows = []
        snapshot.update(
            {
                "selection_mode": snapshot.get("selection_mode") or slot_sampling.get("mode"),
                "n_selected_time_slots": len(selected_slots) or snapshot.get("n_selected_time_slots"),
                "selected_time_slots_preview": _preview(selected_slots),
                "selected_time_windows_preview": snapshot.get("selected_time_windows_preview") or _preview(selected_windows),
                "n_selected_windows": snapshot.get("n_selected_windows") or (len(selected_windows) if selected_windows else None),
            }
        )

    time_slots_text = str(manifest.get("time_slots") or "").strip()
    if time_slots_text and snapshot.get("n_selected_time_slots") is None:
        requested_slots = [item for item in time_slots_text.split() if item]
        snapshot["n_selected_time_slots"] = len(requested_slots)
        snapshot["selected_time_slots_preview"] = _preview(requested_slots)
        snapshot["selection_mode"] = snapshot.get("selection_mode") or "time_slots"

    return snapshot


def _priority_mode_snapshot(manifest: Dict[str, object]) -> Dict[str, Optional[str]]:
    pre_mode = str(manifest.get("pre_priority_mode") or "").strip() or None
    post_mode = str(manifest.get("post_priority_mode") or "").strip() or None

    if pre_mode is None:
        pre_meta = _run_meta(manifest.get("pre_run_dir"))
        pre_mode = str((pre_meta or {}).get("priority_mode") or "").strip() or None
    if post_mode is None:
        post_meta = _run_meta(manifest.get("post_run_dir"))
        post_mode = str((post_meta or {}).get("priority_mode") or "").strip() or None

    common_mode = pre_mode if pre_mode and pre_mode == post_mode else None
    return {
        "common": common_mode,
        "pre": pre_mode,
        "post": post_mode,
    }


def _build_headline(manifest: Dict[str, object], verdict: Dict[str, object]) -> str:
    mode = str(manifest.get("mode") or "evaluation")
    scope = str(manifest.get("rank_only_mode") or manifest.get("operational_mode") or "unknown")
    overall = str(verdict.get("overall") or "inconclusive")
    if overall == "post_better":
        status = "POST looks better than PRE on the primary metrics."
    elif overall == "pre_better":
        status = "PRE looks better than POST on the primary metrics."
    elif overall == "mixed":
        status = "Results are mixed across the primary metrics."
    else:
        status = "Not enough signal to determine a winner."
    return f"{mode} ({scope}): {status}"


def build_pre_post_summary(manifest_path: str | Path) -> Dict[str, object]:
    manifest = _load_json(manifest_path)
    pre_alignment = _optional_json(manifest.get("pre_alignment"))
    post_alignment = _optional_json(manifest.get("post_alignment"))
    delta_alignment = _optional_json(manifest.get("post_vs_pre_alignment_delta"))
    operational_payload = _optional_json(manifest.get("post_vs_pre_operational_impact"))

    alignment = {
        "pre": _alignment_snapshot(pre_alignment),
        "post": _alignment_snapshot(post_alignment),
        "delta_post_minus_pre": (
            (delta_alignment or {}).get("delta_metrics_post_minus_pre")
            if isinstance(delta_alignment, dict)
            else None
        ) or _alignment_delta(pre_alignment, post_alignment),
    }
    alignment_verdict = _score_metrics(
        alignment["delta_post_minus_pre"],
        preferred_positive=("accuracy", "macro_f1", "weighted_f1", "spearman", "kendall_tau", "top_k_hit_rate", "priority_1_recall", "priority_1_f1", "urgent_recall", "urgent_f1"),
    )

    operational = _operational_snapshot(operational_payload)
    operational_verdict = _score_metrics(
        operational.get("comparison", {}) if operational else {},
        preferred_positive=(
            "priority_1_latency_improvement",
            "urgent_latency_improvement",
            "priority_1_service_rate_gain",
            "urgent_service_rate_gain",
            "priority_1_on_time_rate_gain",
            "urgent_on_time_rate_gain",
            "priority_weighted_service_gain",
            "priority_weighted_on_time_gain",
        ),
    )

    primary_verdict = operational_verdict if manifest.get("mode") == "operational_impact_sampled" else alignment_verdict

    return {
        "mode": manifest.get("mode"),
        "scope": manifest.get("rank_only_mode") or manifest.get("operational_mode"),
        "headline": _build_headline(manifest, primary_verdict),
        "truth_source": manifest.get("truth_source"),
        "priority_modes": _priority_mode_snapshot(manifest),
        "sample": _sample_snapshot(manifest),
        "verdicts": {
            "primary": primary_verdict,
            "alignment": alignment_verdict,
            "operational": operational_verdict if operational else None,
        },
        "alignment": alignment,
        "operational": operational if operational else None,
        "artifacts": {
            "manifest": str(manifest_path),
            "pre_run_dir": manifest.get("pre_run_dir"),
            "post_run_dir": manifest.get("post_run_dir"),
            "pre_alignment": manifest.get("pre_alignment"),
            "post_alignment": manifest.get("post_alignment"),
            "alignment_delta": manifest.get("post_vs_pre_alignment_delta"),
            "operational_impact": manifest.get("post_vs_pre_operational_impact"),
            "selection_manifest": manifest.get("selection_manifest"),
            "slot_sampling": manifest.get("slot_sampling"),
        },
    }


def _format_metric_block(title: str, metrics: Dict[str, object]) -> List[str]:
    lines = [f"## {title}"]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: {value}")
    return lines


def render_summary_markdown(summary: Dict[str, object]) -> str:
    lines: List[str] = [
        "# Evaluation Summary",
        "",
        f"- Headline: {summary.get('headline')}",
        f"- Mode: `{summary.get('mode')}`",
        f"- Scope: `{summary.get('scope')}`",
        f"- Truth source: `{summary.get('truth_source')}`",
    ]

    priority_modes = summary.get("priority_modes") or {}
    if isinstance(priority_modes, dict):
        if priority_modes.get("common"):
            lines.append(f"- Priority mode: `{priority_modes.get('common')}`")
        elif priority_modes.get("pre") or priority_modes.get("post"):
            lines.append(f"- Pre priority mode: `{priority_modes.get('pre')}`")
            lines.append(f"- Post priority mode: `{priority_modes.get('post')}`")

    sample = summary.get("sample") or {}
    if isinstance(sample, dict):
        lines.extend(
            [
                f"- Selection mode: `{sample.get('selection_mode')}`",
                f"- Selected windows: `{sample.get('n_selected_windows')}`",
                f"- Selected time slots: `{sample.get('n_selected_time_slots')}`",
            ]
        )
        preview_windows = sample.get("selected_time_windows_preview") or []
        preview_slots = sample.get("selected_time_slots_preview") or []
        if preview_windows:
            lines.append(f"- Selected window preview: `{preview_windows}`")
        if preview_slots:
            lines.append(f"- Selected time-slot preview: `{preview_slots}`")

    verdicts = summary.get("verdicts") or {}
    primary = verdicts.get("primary") or {}
    if isinstance(primary, dict):
        lines.extend(
            [
                "",
                "## Primary Verdict",
                f"- Overall: `{primary.get('overall')}`",
                f"- Metrics where POST improved: `{primary.get('better_metrics_for_post')}`",
                f"- Metrics where POST regressed: `{primary.get('worse_metrics_for_post')}`",
            ]
        )

    alignment = summary.get("alignment") or {}
    if isinstance(alignment, dict):
        lines.append("")
        lines.extend(_format_metric_block("Alignment Delta (Post - Pre)", alignment.get("delta_post_minus_pre") or {}))

    operational = summary.get("operational") or {}
    if isinstance(operational, dict) and operational:
        lines.append("")
        lines.extend(_format_metric_block("Operational Comparison", operational.get("comparison") or {}))

    artifacts = summary.get("artifacts") or {}
    if isinstance(artifacts, dict):
        lines.extend(
            [
                "",
                "## Key Artifacts",
                f"- Manifest: `{artifacts.get('manifest')}`",
                f"- Pre alignment: `{artifacts.get('pre_alignment')}`",
                f"- Post alignment: `{artifacts.get('post_alignment')}`",
                f"- Alignment delta: `{artifacts.get('alignment_delta')}`",
                f"- Operational impact: `{artifacts.get('operational_impact')}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact summary for pre/post evaluation runs.")
    parser.add_argument("--manifest", required=True, help="Path to eval_manifest.json")
    parser.add_argument("--output-json", required=True, help="Output path for summary JSON")
    parser.add_argument("--output-md", required=True, help="Output path for summary Markdown")
    args = parser.parse_args()

    summary = build_pre_post_summary(args.manifest)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_summary_markdown(summary), encoding="utf-8")

    print(f"Summary JSON saved to {output_json}")
    print(f"Summary Markdown saved to {output_md}")


if __name__ == "__main__":
    main()
