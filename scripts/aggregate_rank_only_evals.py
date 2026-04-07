from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


SORT_METRICS = (
    "priority_1_recall",
    "top_k_hit_rate",
    "urgent_f1",
    "macro_f1",
    "accuracy",
)


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: Dict[str, object], section: str, key: str) -> Optional[float]:
    block = payload.get(section) or {}
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _primary_score(delta: Dict[str, Optional[float]]) -> float:
    return round(
        100.0 * float(delta.get("priority_1_recall") or 0.0)
        + 10.0 * float(delta.get("top_k_hit_rate") or 0.0)
        + 1.0 * float(delta.get("urgent_f1") or 0.0)
        + 0.1 * float(delta.get("macro_f1") or 0.0)
        + 0.01 * float(delta.get("accuracy") or 0.0),
        6,
    )


def build_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary_path in sorted(root.glob("*/evals/summary.json")):
        run_dir = summary_path.parent.parent
        payload = _load_json(summary_path)
        alignment = payload.get("alignment") or {}
        if not isinstance(alignment, dict):
            continue
        post = alignment.get("post") or {}
        pre = alignment.get("pre") or {}
        delta = alignment.get("delta_post_minus_pre") or {}
        if not isinstance(post, dict) or not isinstance(pre, dict) or not isinstance(delta, dict):
            continue
        row: Dict[str, object] = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "headline": payload.get("headline"),
            "overall": ((payload.get("verdicts") or {}).get("primary") or {}).get("overall"),
            "truth_source": payload.get("truth_source"),
            "n_selected_windows": ((payload.get("sample") or {}).get("n_selected_windows")),
            "n_aligned_demands": post.get("n_aligned_demands"),
        }
        for key in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "spearman",
            "kendall_tau",
            "top_k_hit_rate",
            "priority_1_recall",
            "priority_1_f1",
            "urgent_recall",
            "urgent_f1",
        ):
            row[f"pre_{key}"] = pre.get(key)
            row[f"post_{key}"] = post.get(key)
            row[f"delta_{key}"] = delta.get(key)
        row["primary_score"] = _primary_score(delta)
        rows.append(row)

    rows.sort(
        key=lambda item: tuple(float(item.get(f"delta_{metric}") or 0.0) for metric in SORT_METRICS),
        reverse=True,
    )
    for index, row in enumerate(rows, start=1):
        row["leaderboard_rank"] = index
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multiple rank-only pre/post eval runs.")
    parser.add_argument("root", type=Path, help="Parent directory that contains per-model eval run directories.")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    rows = build_rows(root)
    output_jsonl = args.output_jsonl or (root / "leaderboard.jsonl")
    output_csv = args.output_csv or (root / "leaderboard.csv")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rows:
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
