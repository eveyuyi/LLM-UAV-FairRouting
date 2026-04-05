"""Audit LLM3 prompt lengths for SFT, GRPO, and online ranking evaluation."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm3_verl_utils import SYSTEM_PROMPT, build_prompt_text, build_response_text, load_jsonl
from llm4fairrouting.llm.prompt_templates import DRONE_SYSTEM_PROMPT

SFT_FILE_MAP = {
    "clean": "llm3_sft_clean.jsonl",
    "pipeline": "llm3_sft_pipeline.jsonl",
}
GRPO_FILENAME = "llm3_grpo_hard.jsonl"


def _resolve_dirs(patterns: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            resolved.extend(Path(match).expanduser().resolve() for match in matches)
        else:
            candidate = Path(pattern).expanduser().resolve()
            if candidate.exists():
                resolved.append(candidate)
    deduped: List[Path] = []
    seen = set()
    for path in resolved:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def _load_tokenizer(model_path: Optional[str]):
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "transformers is required for token-based auditing. "
            "Install it in your training environment or omit --tokenizer-path."
        ) from exc
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _count_tokens(tokenizer, messages: Sequence[Dict[str, str]]) -> Optional[int]:
    if tokenizer is None:
        return None
    try:
        tokens = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
        )
        return len(tokens)
    except Exception:
        text = "\n".join(str(message.get("content", "")) for message in messages)
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def _safe_stats(values: Sequence[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(values)

    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        idx = (len(ordered) - 1) * p
        lower = int(idx)
        upper = min(lower + 1, len(ordered) - 1)
        weight = idx - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": int(ordered[0]),
        "p50": round(percentile(0.50), 2),
        "p95": round(percentile(0.95), 2),
        "max": int(ordered[-1]),
        "mean": round(statistics.fmean(ordered), 2),
    }


def _sample_record(
    *,
    split: str,
    source: str,
    input_dir: Path,
    sample: Dict,
    tokenizer,
    eval_max_context: int,
    eval_completion_budget: int,
) -> Dict:
    prompt_text = build_prompt_text(
        time_window=str(sample.get("time_window", "")),
        demands=sample.get("demands", []),
    )
    num_demands = len(sample.get("demands", []))

    sft_response_text = build_response_text(sample.get("priority_labels", []))
    sft_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": sft_response_text},
    ]
    grpo_messages = [{"role": "user", "content": prompt_text}]
    eval_messages = [
        {"role": "system", "content": DRONE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]

    sft_sequence_tokens = _count_tokens(tokenizer, sft_messages)
    grpo_prompt_tokens = _count_tokens(tokenizer, grpo_messages)
    eval_prompt_tokens = _count_tokens(tokenizer, eval_messages)

    return {
        "split": split,
        "source": source,
        "input_dir": str(input_dir),
        "time_window": str(sample.get("time_window", "")),
        "num_demands": num_demands,
        "prompt_chars": len(prompt_text),
        "sft_sequence_tokens": sft_sequence_tokens,
        "grpo_prompt_tokens": grpo_prompt_tokens,
        "eval_prompt_tokens": eval_prompt_tokens,
        "eval_one_shot_budget_tokens": (
            None if eval_prompt_tokens is None else eval_max_context - eval_completion_budget
        ),
    }


def _iter_records(
    split: str,
    input_dirs: Sequence[Path],
    sources: Sequence[str],
) -> Iterable[Dict]:
    for input_dir in input_dirs:
        for source in sources:
            if source in SFT_FILE_MAP:
                filename = SFT_FILE_MAP[source]
            elif source == "grpo":
                filename = GRPO_FILENAME
            else:
                continue
            path = input_dir / filename
            if not path.exists():
                continue
            for sample in load_jsonl(path):
                yield {
                    "split": split,
                    "source": source,
                    "input_dir": input_dir,
                    "sample": sample,
                }


def _summarize_group(
    rows: Sequence[Dict],
    *,
    sft_max_length: int,
    grpo_max_prompt_length: int,
    eval_max_context: int,
    eval_completion_budget: int,
) -> Dict:
    prompt_chars = [row["prompt_chars"] for row in rows]
    sft_tokens = [row["sft_sequence_tokens"] for row in rows if row["sft_sequence_tokens"] is not None]
    grpo_tokens = [row["grpo_prompt_tokens"] for row in rows if row["grpo_prompt_tokens"] is not None]
    eval_tokens = [row["eval_prompt_tokens"] for row in rows if row["eval_prompt_tokens"] is not None]

    summary = {
        "rows": len(rows),
        "num_demands": _safe_stats([row["num_demands"] for row in rows]),
        "prompt_chars": _safe_stats(prompt_chars),
        "sft_sequence_tokens": _safe_stats(sft_tokens),
        "grpo_prompt_tokens": _safe_stats(grpo_tokens),
        "eval_prompt_tokens": _safe_stats(eval_tokens),
    }

    if sft_tokens:
        summary["sft_over_limit"] = sum(tokens > sft_max_length for tokens in sft_tokens)
        summary["sft_over_limit_rate"] = round(summary["sft_over_limit"] / len(sft_tokens), 4)
    else:
        summary["sft_over_limit"] = None
        summary["sft_over_limit_rate"] = None

    if grpo_tokens:
        summary["grpo_over_limit"] = sum(tokens > grpo_max_prompt_length for tokens in grpo_tokens)
        summary["grpo_over_limit_rate"] = round(summary["grpo_over_limit"] / len(grpo_tokens), 4)
    else:
        summary["grpo_over_limit"] = None
        summary["grpo_over_limit_rate"] = None

    if eval_tokens:
        one_shot_limit = eval_max_context - eval_completion_budget
        summary["eval_over_context"] = sum(tokens > eval_max_context for tokens in eval_tokens)
        summary["eval_over_one_shot_budget"] = sum(tokens > one_shot_limit for tokens in eval_tokens)
        summary["eval_over_context_rate"] = round(summary["eval_over_context"] / len(eval_tokens), 4)
        summary["eval_over_one_shot_budget_rate"] = round(
            summary["eval_over_one_shot_budget"] / len(eval_tokens), 4
        )
    else:
        summary["eval_over_context"] = None
        summary["eval_over_one_shot_budget"] = None
        summary["eval_over_context_rate"] = None
        summary["eval_over_one_shot_budget_rate"] = None

    return summary


def _top_examples(rows: Sequence[Dict], key: str, limit: int) -> List[Dict]:
    sortable = [row for row in rows if row.get(key) is not None]
    sortable.sort(key=lambda row: int(row[key]), reverse=True)
    return [
        {
            "split": row["split"],
            "source": row["source"],
            "input_dir": row["input_dir"],
            "time_window": row["time_window"],
            "num_demands": row["num_demands"],
            key: row[key],
            "prompt_chars": row["prompt_chars"],
        }
        for row in sortable[:limit]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit LLM3 prompt lengths against SFT/GRPO/eval token budgets.",
    )
    parser.add_argument("--train-pattern", action="append", default=[], help="Glob or path for train seed dirs.")
    parser.add_argument("--val-pattern", action="append", default=[], help="Glob or path for val seed dirs.")
    parser.add_argument("--test-pattern", action="append", default=[], help="Glob or path for test seed dirs.")
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="Extra dirs treated as split=adhoc. Useful for quick single-shard audits.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["clean", "pipeline", "grpo"],
        default=["clean", "pipeline", "grpo"],
    )
    parser.add_argument("--tokenizer-path", help="HF model/tokenizer path used for exact token counting.")
    parser.add_argument("--sft-max-length", type=int, default=4096)
    parser.add_argument("--grpo-max-prompt-length", type=int, default=2048)
    parser.add_argument("--eval-max-context", type=int, default=16384)
    parser.add_argument("--eval-completion-budget", type=int, default=700)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--json-out", help="Optional path to write the full JSON report.")
    args = parser.parse_args()

    split_dirs = {
        "train": _resolve_dirs(args.train_pattern),
        "val": _resolve_dirs(args.val_pattern),
        "test": _resolve_dirs(args.test_pattern),
        "adhoc": _resolve_dirs(args.input_dir),
    }
    split_dirs = {split: dirs for split, dirs in split_dirs.items() if dirs}
    if not split_dirs:
        raise SystemExit("No input dirs found. Pass --train-pattern/--val-pattern/--test-pattern or --input-dir.")

    tokenizer = _load_tokenizer(args.tokenizer_path)
    rows: List[Dict] = []
    for split, input_dirs in split_dirs.items():
        for item in _iter_records(split, input_dirs, args.sources):
            rows.append(
                _sample_record(
                    split=item["split"],
                    source=item["source"],
                    input_dir=item["input_dir"],
                    sample=item["sample"],
                    tokenizer=tokenizer,
                    eval_max_context=args.eval_max_context,
                    eval_completion_budget=args.eval_completion_budget,
                )
            )

    if not rows:
        raise SystemExit("No matching JSONL samples found under the provided dirs.")

    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["source"])].append(row)

    report = {
        "tokenizer_path": args.tokenizer_path,
        "limits": {
            "sft_max_length": args.sft_max_length,
            "grpo_max_prompt_length": args.grpo_max_prompt_length,
            "eval_max_context": args.eval_max_context,
            "eval_completion_budget": args.eval_completion_budget,
            "eval_one_shot_input_budget": args.eval_max_context - args.eval_completion_budget,
        },
        "splits": {},
        "top_examples": {
            "largest_prompt_chars": _top_examples(rows, "prompt_chars", args.top_n),
            "largest_sft_sequences": _top_examples(rows, "sft_sequence_tokens", args.top_n),
            "largest_grpo_prompts": _top_examples(rows, "grpo_prompt_tokens", args.top_n),
            "largest_eval_prompts": _top_examples(rows, "eval_prompt_tokens", args.top_n),
        },
    }

    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        report["splits"][split] = {"all_sources": _summarize_group(
            split_rows,
            sft_max_length=args.sft_max_length,
            grpo_max_prompt_length=args.grpo_max_prompt_length,
            eval_max_context=args.eval_max_context,
            eval_completion_budget=args.eval_completion_budget,
        )}
        for source in args.sources:
            source_rows = grouped.get((split, source), [])
            if not source_rows:
                continue
            report["splits"][split][source] = _summarize_group(
                source_rows,
                sft_max_length=args.sft_max_length,
                grpo_max_prompt_length=args.grpo_max_prompt_length,
                eval_max_context=args.eval_max_context,
                eval_completion_budget=args.eval_completion_budget,
            )

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out:
        output_path = Path(args.json_out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
