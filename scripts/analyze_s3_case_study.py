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
    gt = {}
    for item in _load_jsonl(events_manifest):
        for demand in item.get("demands", []) or [item]:
            did = demand.get("demand_id") or demand.get("event_id")
            pri = demand.get("priority") or demand.get("latent_priority")
            if did and pri:
                gt[str(did)] = int(pri)
    return gt


def _build_dialogue_map(dialogues_path: Path) -> dict[str, str]:
    diag_map = {}
    for item in _load_jsonl(dialogues_path):
        did = item.get("demand_id") or item.get("event_id")
        text = item.get("dialogue") or item.get("transcript") or str(item.get("messages", ""))
        if did:
            diag_map[str(did)] = str(text)[:500]
    return diag_map


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
    args = parser.parse_args()

    rule_configs = _load_weight_configs(Path(args.rule_weights))
    llm_configs = _load_weight_configs(Path(args.llm_weights))
    gt_map = _build_gt_map(Path(args.ground_truth))
    diag_map = _build_dialogue_map(Path(args.dialogues))

    cases = []
    common_windows = set(rule_configs) & set(llm_configs)
    for tw in sorted(common_windows):
        rule_preds = rule_configs[tw]
        llm_preds = llm_configs[tw]
        for demand_id in set(rule_preds) & set(llm_preds) & set(gt_map):
            gt_pri = gt_map[demand_id]
            rule_err = abs(rule_preds[demand_id] - gt_pri)
            llm_err = abs(llm_preds[demand_id] - gt_pri)
            gain = rule_err - llm_err
            if gain >= args.priority_gap:
                cases.append({
                    "time_window": tw,
                    "demand_id": demand_id,
                    "gt_priority": gt_pri,
                    "rule_pred": rule_preds[demand_id],
                    "llm_pred": llm_preds[demand_id],
                    "rule_error": rule_err,
                    "llm_error": llm_err,
                    "gain": gain,
                    "dialogue_snippet": diag_map.get(demand_id, "(no dialogue found)"),
                })

    cases.sort(key=lambda c: -c["gain"])
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
        lines.append("")
        lines.append("**Dialogue snippet:**")
        lines.append(f"```\n{c['dialogue_snippet']}\n```\n")

    if not selected:
        lines.append("*No cases found matching the criteria.*\n")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Case study report saved to {out} ({len(selected)} cases)")


if __name__ == "__main__":
    main()
