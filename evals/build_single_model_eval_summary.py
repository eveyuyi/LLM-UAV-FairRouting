"""Build a compact summary for single-model rank-only priority alignment eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


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


def _preview(items: Sequence[object], limit: int = 10) -> List[object]:
    values = list(items)
    if len(values) <= limit:
        return values
    return values[:limit]


def _sample_snapshot(manifest: Dict[str, object]) -> Dict[str, object]:
    selection_manifest = _optional_json(manifest.get("selection_manifest"))
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

    time_slots_text = str(manifest.get("time_slots") or "").strip()
    if time_slots_text and snapshot.get("n_selected_time_slots") is None:
        requested_slots = [item for item in time_slots_text.split() if item]
        snapshot["n_selected_time_slots"] = len(requested_slots)
        snapshot["selected_time_slots_preview"] = _preview(requested_slots)
        snapshot["selection_mode"] = snapshot.get("selection_mode") or "time_slots"

    return snapshot


def build_single_model_summary(manifest_path: str | Path) -> Dict[str, object]:
    manifest = _load_json(manifest_path)
    alignment_path = manifest.get("alignment")
    alignment = _optional_json(alignment_path) if alignment_path else None

    return {
        "mode": manifest.get("mode"),
        "scope": manifest.get("rank_only_mode"),
        "headline": (
            f"{manifest.get('mode')} ({manifest.get('rank_only_mode')}): "
            f"model={manifest.get('model_name')} @ {manifest.get('api_base')}"
        ),
        "truth_source": manifest.get("truth_source"),
        "priority_mode": manifest.get("priority_mode"),
        "model_name": manifest.get("model_name"),
        "api_base": manifest.get("api_base"),
        "sample": _sample_snapshot(manifest),
        "alignment": _alignment_snapshot(alignment),
        "artifacts": {
            "manifest": str(manifest_path),
            "run_dir": manifest.get("run_dir"),
            "alignment": manifest.get("alignment"),
            "selection_manifest": manifest.get("selection_manifest"),
        },
    }


def render_summary_markdown(summary: Dict[str, object]) -> str:
    lines: List[str] = [
        "# Single-model evaluation summary",
        "",
        f"- Headline: {summary.get('headline')}",
        f"- Mode: `{summary.get('mode')}`",
        f"- Scope: `{summary.get('scope')}`",
        f"- Truth source: `{summary.get('truth_source')}`",
        f"- Priority mode: `{summary.get('priority_mode')}`",
        f"- Model: `{summary.get('model_name')}`",
        f"- API base: `{summary.get('api_base')}`",
    ]

    sample = summary.get("sample") or {}
    if isinstance(sample, dict):
        lines.extend(
            [
                f"- Selection mode: `{sample.get('selection_mode')}`",
                f"- Selected windows: `{sample.get('n_selected_windows')}`",
                f"- Selected time slots: `{sample.get('n_selected_time_slots')}`",
            ]
        )

    alignment = summary.get("alignment") or {}
    if isinstance(alignment, dict) and alignment:
        lines.append("")
        lines.append("## Alignment metrics")
        for key, value in alignment.items():
            lines.append(f"- `{key}`: {value}")

    artifacts = summary.get("artifacts") or {}
    if isinstance(artifacts, dict):
        lines.extend(
            [
                "",
                "## Artifacts",
                f"- Manifest: `{artifacts.get('manifest')}`",
                f"- Run dir: `{artifacts.get('run_dir')}`",
                f"- Alignment JSON: `{artifacts.get('alignment')}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build summary for single-model rank-only eval.")
    parser.add_argument("--manifest", required=True, help="Path to eval_manifest.json")
    parser.add_argument("--output-json", required=True, help="Output path for summary JSON")
    parser.add_argument("--output-md", required=True, help="Output path for summary Markdown")
    args = parser.parse_args()

    summary = build_single_model_summary(args.manifest)

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
