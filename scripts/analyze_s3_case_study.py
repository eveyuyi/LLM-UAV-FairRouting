"""S3: Case study selector.

Compares per-demand predictions from two eval runs (e.g., rule-only vs LLM)
and selects windows where LLM is correct while rule-based is wrong, with a
large priority gap. Outputs a Markdown case study report with dialogue text.

Usage:
    python scripts/analyze_s3_case_study.py \\
        --rule-weights  data/eval_runs/m0c_*/run_*/weight_configs \\
        --llm-weights   data/eval_runs/m1_*/run_*/weight_configs \\
        --demands       data/test/test_seeds/hard_eval/seed_5101/llm3_sft_pipeline.jsonl \\
        --ground-truth  data/test/test_seeds/hard_eval/seed_5101/events_manifest.jsonl \\
        --dialogues     data/test/test_seeds/hard_eval/seed_5101/dialogues.jsonl \\
        --output        data/evals/s3_case_study.md \\
        --top-n 3
"""
import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_weight_configs(weights_dir: Path) -> dict[str, dict]:
    configs = {}
    for f in sorted(weights_dir.glob("weight_config_window*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        tw = data.get("time_window", f.stem)
        configs[tw] = {cfg["demand_id"]: cfg["priority"] for cfg in data.get("demand_configs", [])}
    return configs


def _build_gt_map(events_manifest: Path) -> dict[str, int]:
    """Returns {event_id: latent_priority} from events_manifest.jsonl."""
    gt = {}
    for item in _load_jsonl(events_manifest):
        # top-level event record
        eid = item.get("event_id")
        pri = item.get("latent_priority") or item.get("priority")
        if eid and pri:
            gt[str(eid)] = int(pri)
    return gt


def _build_dialogue_map(dialogues_path: Path) -> dict[str, str]:
    """Returns {event_id: conversation_text} from dialogues.jsonl."""
    diag_map = {}
    for item in _load_jsonl(dialogues_path):
        # dialogues.jsonl: event_id lives in metadata.event_id
        eid = (item.get("metadata") or {}).get("event_id") or item.get("event_id")
        conv = item.get("conversation") or item.get("dialogue") or item.get("transcript") or ""
        if isinstance(conv, list):
            # list of {"role":..., "content":...} turns
            conv = "\n".join(
                f"{t.get('role','?')}: {t.get('content','')}" for t in conv
            )
        if eid:
            diag_map[str(eid)] = str(conv)[:800]
    return diag_map


def _build_event_map(demands_path: Path) -> dict[str, str]:
    """Returns {demand_id: source_event_id} from llm3_sft_pipeline.jsonl."""
    ev_map = {}
    for win in _load_jsonl(demands_path):
        for d in win.get("demands", []):
            did = d.get("demand_id")
            eid = d.get("source_event_id") or did
            if did:
                ev_map[str(did)] = str(eid)
    return ev_map


def main():
    parser = argparse.ArgumentParser(description="S3: Case study selection")
    parser.add_argument("--rule-weights", required=True, help="Path to rule-based weight_configs dir")
    parser.add_argument("--llm-weights", required=True, help="Path to LLM weight_configs dir")
    parser.add_argument("--demands", required=True, help="Path to llm3_sft_pipeline.jsonl (test demands)")
    parser.add_argument("--ground-truth", required=True, help="Path to events_manifest.jsonl")
    parser.add_argument("--dialogues", required=True, help="Path to dialogues.jsonl")
    parser.add_argument("--output", default="data/evals/s3_case_study.md")
    parser.add_argument("--priority-gap", type=int, default=2,
                        help="Minimum |llm_pred - gt| vs |rule_pred - gt| gap to select a case")
    parser.add_argument("--top-n", type=int, default=3, help="Max cases to report")
    parser.add_argument("--urgent-gt", type=int, default=None,
                        help="If set, only include cases with gt_priority <= this value (e.g. 2 for urgent)")
    args = parser.parse_args()

    rule_configs = _load_weight_configs(Path(args.rule_weights))
    llm_configs = _load_weight_configs(Path(args.llm_weights))
    gt_map = _build_gt_map(Path(args.ground_truth))
    diag_map = _build_dialogue_map(Path(args.dialogues))
    ev_map = _build_event_map(Path(args.demands))

    cases = []
    common_windows = set(rule_configs) & set(llm_configs)
    for tw in sorted(common_windows):
        rule_preds = rule_configs[tw]
        llm_preds = llm_configs[tw]
        for demand_id in set(rule_preds) & set(llm_preds):
            event_id = ev_map.get(demand_id, demand_id)
            gt_pri = gt_map.get(event_id) or gt_map.get(demand_id)
            if gt_pri is None:
                continue
            gt_pri = int(gt_pri)
            rule_p = int(rule_preds[demand_id])
            llm_p  = int(llm_preds[demand_id])
            rule_err = abs(rule_p - gt_pri)
            llm_err = abs(llm_p - gt_pri)
            gain = rule_err - llm_err
            if gain >= args.priority_gap and (args.urgent_gt is None or gt_pri <= args.urgent_gt):
                cases.append({
                    "time_window": tw,
                    "demand_id": demand_id,
                    "event_id": event_id,
                    "gt_priority": gt_pri,
                    "rule_pred": rule_p,
                    "llm_pred": llm_p,
                    "rule_error": rule_err,
                    "llm_error": llm_err,
                    "gain": gain,
                    "dialogue_snippet": diag_map.get(event_id, diag_map.get(demand_id, "(no dialogue found)")),
                })

    # Prefer cases where rule UNDER-estimated urgency (GT high, rule low, LLM correct)
    cases.sort(key=lambda c: (-c["gain"], c["gt_priority"]))
    selected = cases[: args.top_n]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# S3 Case Study: LLM vs Rule-Based Priority Inference\n"]
    lines.append(f"Showing top {len(selected)} cases where LLM outperformed rule-based "
                 f"by ≥ {args.priority_gap} priority levels.\n")
    for i, c in enumerate(selected, 1):
        lines.append(f"## Case {i}: `{c['demand_id']}` (window `{c['time_window']}`)\n")
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| Ground truth priority | P{c['gt_priority']} |")
        lines.append(f"| Rule-based prediction | P{c['rule_pred']} (error={c['rule_error']}) |")
        lines.append(f"| LLM prediction        | P{c['llm_pred']} (error={c['llm_error']}) |")
        lines.append(f"| Priority gain         | {c['gain']} levels |")
        lines.append(f"| Event ID              | `{c['event_id']}` |")
        lines.append("")
        lines.append("**Dialogue snippet:**")
        lines.append(f"```\n{c['dialogue_snippet']}\n```\n")

    if not selected:
        lines.append("*No cases found matching the criteria.*\n")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Case study report saved to {out} ({len(selected)} cases)")


if __name__ == "__main__":
    main()
