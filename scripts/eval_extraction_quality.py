"""Contribution 1: LLM2 extraction quality vs gold_extraction.

Reads llm3_sft_pipeline.jsonl (pre-extracted demands with gold_extraction),
computes per-field accuracy, critical signal recall, schema validity, and
priority chain consistency.

Usage:
  python scripts/eval_extraction_quality.py --seed 4111
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent

def _norm_type(s: Optional[str]) -> str:
    return (s or "").lower().replace("_", " ").strip()

def _get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict): return default
        d = d.get(k, default)
    return d

def evaluate_extraction(pipeline_path: Path) -> dict:
    with open(pipeline_path, encoding="utf-8") as f:
        if pipeline_path.suffix == ".jsonl":
            windows = [json.loads(l) for l in f if l.strip()]
        else:
            windows = json.load(f)

    counters = {k: 0 for k in [
        "total", "origin_fid", "dest_fid", "dest_type",
        "cargo_type", "weight_kg", "temp_sensitive",
        "deadline_minutes", "demand_tier",
        "requester_role", "elderly_involved", "children_involved",
        "vulnerable_community", "special_handling_empty_match",
        "all_critical_correct", "schema_valid",
        "priority_chain_full_consistent",
    ]}

    for window in windows:
        for dem in window.get("demands", []):
            gold = dem.get("gold_extraction")
            if not gold:
                continue
            counters["total"] += 1

            # structural fields
            if _get(dem, "origin", "fid") == _get(gold, "origin", "fid"):
                counters["origin_fid"] += 1
            if _get(dem, "destination", "fid") == _get(gold, "destination", "fid"):
                counters["dest_fid"] += 1
            dt_ex = _norm_type(_get(dem, "destination", "type"))
            dt_go = _norm_type(_get(gold, "destination", "type"))
            if dt_ex and dt_go and dt_ex == dt_go:
                counters["dest_type"] += 1

            # cargo
            ct_ex = _norm_type(_get(dem, "cargo", "type"))
            ct_go = _norm_type(_get(gold, "cargo", "type_cn") or _get(gold, "cargo", "type"))
            if ct_ex and ct_go and (ct_ex == ct_go or ct_ex in ct_go or ct_go in ct_ex):
                counters["cargo_type"] += 1
            if _get(dem, "cargo", "weight_kg") == _get(gold, "cargo", "weight_kg"):
                counters["weight_kg"] += 1
            if _get(dem, "cargo", "temperature_sensitive") == _get(gold, "cargo", "temperature_sensitive"):
                counters["temp_sensitive"] += 1

            # time constraint
            dl_ex = _get(dem, "time_constraint", "deadline_minutes")
            dl_go = _get(gold, "time_constraint", "deadline_minutes")
            if dl_ex == dl_go:
                counters["deadline_minutes"] += 1

            # tier
            if dem.get("demand_tier") == gold.get("demand_tier"):
                counters["demand_tier"] += 1

            # priority evaluation signals
            pes = dem.get("priority_evaluation_signals") or {}
            role_ex = (pes.get("requester_role") or "").strip().lower()
            role_go = (gold.get("requester_role") or "").strip().lower()
            if role_ex and role_go and role_ex == role_go:
                counters["requester_role"] += 1

            # vulnerability
            pv_ex_raw = pes.get("population_vulnerability")
            pv_ex = pv_ex_raw if isinstance(pv_ex_raw, dict) else {}
            pv_go_raw = gold.get("population_vulnerability")
            pv_go = pv_go_raw if isinstance(pv_go_raw, dict) else {}
            if pv_ex.get("elderly_involved") == pv_go.get("elderly_involved"):
                counters["elderly_involved"] += 1
            if pv_ex.get("children_involved") == pv_go.get("children_involved"):
                counters["children_involved"] += 1
            if pv_ex.get("vulnerable_community") == pv_go.get("vulnerable_community"):
                counters["vulnerable_community"] += 1

            # special_handling: both empty or both non-empty
            sh_ex = pes.get("special_handling") or []
            sh_go = gold.get("special_handling") or []
            if bool(sh_ex) == bool(sh_go):
                counters["special_handling_empty_match"] += 1

            # critical signal recall: ALL of deadline, requester_role, elderly, vulnerable must match
            crit_ok = (
                (dl_ex == dl_go) and
                (role_ex == role_go) and
                (pv_ex.get("elderly_involved") == pv_go.get("elderly_involved")) and
                (pv_ex.get("vulnerable_community") == pv_go.get("vulnerable_community"))
            )
            if crit_ok:
                counters["all_critical_correct"] += 1

            # schema validity: origin_fid, dest_fid non-null and deadline present
            if (_get(dem, "origin", "fid") and
                _get(dem, "destination", "fid") and
                dl_ex is not None):
                counters["schema_valid"] += 1

            # priority chain consistency
            labels = dem.get("labels") or {}
            lp = labels.get("latent_priority")
            dop = labels.get("dialogue_observable_priority")
            eop = labels.get("extraction_observable_priority")
            if lp is not None and dop is not None and eop is not None:
                if lp == dop == eop:
                    counters["priority_chain_full_consistent"] += 1

    n = counters["total"]
    def pct(k): return round(counters[k] / n * 100, 1) if n > 0 else None

    return {
        "n_demands": n,
        "per_field_accuracy": {
            "origin_fid":               pct("origin_fid"),
            "dest_fid":                 pct("dest_fid"),
            "dest_type":                pct("dest_type"),
            "cargo_type":               pct("cargo_type"),
            "weight_kg":                pct("weight_kg"),
            "temperature_sensitive":    pct("temp_sensitive"),
            "deadline_minutes":         pct("deadline_minutes"),
            "demand_tier":              pct("demand_tier"),
            "requester_role":           pct("requester_role"),
            "elderly_involved":         pct("elderly_involved"),
            "children_involved":        pct("children_involved"),
            "vulnerable_community":     pct("vulnerable_community"),
            "special_handling_match":   pct("special_handling_empty_match"),
        },
        "critical_signal_recall":        pct("all_critical_correct"),
        "schema_validity_rate":          pct("schema_valid"),
        "priority_chain_consistency":    pct("priority_chain_full_consistent"),
        "raw_counts": counters,
    }


def print_table(result: dict, seed: str) -> None:
    n = result["n_demands"]
    print(f"\n## Contribution 1 — LLM2 Extraction Quality  (seed={seed}, n={n})\n")
    print(f"| {'Field':<30} | Accuracy (%) |")
    print(f"| {'---':<30} | ---          |")
    for field, val in result["per_field_accuracy"].items():
        bar = "█" * int((val or 0) / 5)
        print(f"| {field:<30} | {val or '—':>6}%      {bar}")
    print()
    print(f"| **Critical signal recall (all 4)**  | **{result['critical_signal_recall']}%** |")
    print(f"| **Schema validity rate**             | **{result['schema_validity_rate']}%**  |")
    print(f"| Priority chain fully consistent      | {result['priority_chain_consistency']}%   |")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="4111")
    parser.add_argument("--split", default="norm_eval")
    parser.add_argument("--output", default="evals/results/extraction_quality_seed{seed}.json")
    args = parser.parse_args()

    pipeline_path = (ROOT / "data" / "test" / "test_seeds" / args.split
                     / f"seed_{args.seed}" / "llm3_sft_pipeline.jsonl")
    print(f"Loading: {pipeline_path}")
    result = evaluate_extraction(pipeline_path)
    print_table(result, args.seed)

    out = Path(args.output.format(seed=args.seed))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, **result}, f, ensure_ascii=False, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
