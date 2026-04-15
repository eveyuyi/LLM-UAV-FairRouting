#!/usr/bin/env python
"""Evaluate base/SFT/GRPO llm_selection models on a fixed overlap test set."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = REPO_ROOT / "scripts" / "export_llm_selection_compact.py"
REWARD_PATH = REPO_ROOT / "scripts" / "verl_llm_selection_reward.py"

DEFAULT_BASE_MODEL_PATH = (
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/"
    "zskj-hub/model--Qwen-Qwen3-4B-Instruct-2507"
)
DEFAULT_TEST_JSONL = (
    REPO_ROOT
    / "data/train/llm_selection_training_jsonl_1000_qs_v1_20260413/compact_exports_4b_short/test_sft_overlap_100_seed42.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evals/llm_selection_model_compare"
DEFAULT_KEEP_CANDIDATES_PER_GROUP = 3
DEFAULT_MAX_DEMAND_CARDS = 6


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXPORTER = _load_module("llm_selection_exporter", EXPORTER_PATH)
REWARD = _load_module("llm_selection_reward", REWARD_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base / SFT / GRPO llm_selection models on a fixed test set."
    )
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--sft-model-path", default="")
    parser.add_argument("--grpo-model-path", default="")
    parser.add_argument("--keep-candidates-per-group", type=int, default=DEFAULT_KEEP_CANDIDATES_PER_GROUP)
    parser.add_argument("--max-demand-cards", type=int, default=DEFAULT_MAX_DEMAND_CARDS)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit evaluation samples; 0 means all.")
    parser.add_argument("--backend", choices=("auto", "vllm", "transformers"), default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-overflow-strategy",
        choices=("truncate", "skip", "error"),
        default="truncate",
        help="How to handle prompts longer than the effective input budget.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base model evaluation even if --base-model-path is set.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="vLLM prompt batch size.")
    return parser.parse_args()


def _latest_global_step_dir(root: Path) -> Path:
    candidates = []
    for path in root.glob("global_step_*"):
        suffix = path.name.replace("global_step_", "", 1)
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(f"No global_step_* directories found under {root}")
    return max(candidates, key=lambda item: item[0])[1]


def resolve_model_dir(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")
    if (path / "config.json").is_file():
        return path
    if path.name.startswith("global_step_"):
        merged = path / "huggingface_lora_merged"
        if (merged / "config.json").is_file():
            return merged
        raise FileNotFoundError(
            f"{path} is a checkpoint directory, but {merged} does not exist. "
            "Please merge it to HuggingFace format first."
        )
    latest = _latest_global_step_dir(path)
    merged = latest / "huggingface_lora_merged"
    if (merged / "config.json").is_file():
        return merged
    raise FileNotFoundError(
        f"{path} resolved to latest checkpoint {latest}, but merged model {merged} was not found."
    )


def load_test_records(path: Path, max_samples: int) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at line {line_no}: {exc}") from exc
    if max_samples > 0:
        records = records[:max_samples]
    return records


def _candidate_metadata_from_compact_input(compact_input: Dict) -> Tuple[List[str], List[str]]:
    candidate_solution_ids: List[str] = []
    group_ids: List[str] = []
    for group in compact_input.get("objective_groups", []) or []:
        group_id = str(group.get("group_id") or "").strip()
        if group_id:
            group_ids.append(group_id)
        for sid in group.get("candidate_solution_ids", []) or []:
            sid_str = str(sid or "").strip()
            if sid_str:
                candidate_solution_ids.append(sid_str)
    return sorted(set(candidate_solution_ids)), group_ids


def _build_item_from_raw_record(record: Dict, keep_candidates_per_group: int, max_demand_cards: int) -> Dict:
    compact_input = EXPORTER._compact_selection_input(record, keep_candidates_per_group, max_demand_cards)
    user_prompt = EXPORTER._render_user_prompt(compact_input)
    target = record.get("selection_target") or {}
    labels = target.get("training_labels") or {}
    candidate_solution_ids, group_ids = _candidate_metadata_from_compact_input(compact_input)
    return {
        "record_id": str(record.get("record_id") or ""),
        "messages": [
            {"role": "system", "content": EXPORTER.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "prompt": user_prompt,
        "ground_truth": {
            "selected_group_id": target.get("selected_group_id"),
            "selected_solution_id": target.get("selected_solution_id"),
            "selection_mode": target.get("selection_mode"),
            "primary_reason_codes": list(target.get("primary_reason_codes") or []),
            "decision_confidence": target.get("decision_confidence"),
            "candidate_solution_ids": candidate_solution_ids,
            "group_ids": group_ids,
            "training_labels": labels,
        },
        "scene_type": str(labels.get("scene_type") or ""),
        "selection_mode": str(target.get("selection_mode") or ""),
        "decision_difficulty": str(labels.get("decision_difficulty") or ""),
    }


def _build_item_from_sft_compact_record(record: Dict) -> Dict:
    target = record.get("selection_target") or {}
    labels = target.get("training_labels") or {}
    compact_input = record.get("compact_selection_input") or {}
    candidate_solution_ids, group_ids = _candidate_metadata_from_compact_input(compact_input)
    messages = list(record.get("messages") or [])
    if len(messages) < 2:
        raise ValueError(f"SFT compact record missing messages: {record.get('record_id')}")
    return {
        "record_id": str(record.get("record_id") or ""),
        "messages": messages[:2],
        "prompt": str(record.get("prompt") or ""),
        "ground_truth": {
            "selected_group_id": target.get("selected_group_id"),
            "selected_solution_id": target.get("selected_solution_id"),
            "selection_mode": target.get("selection_mode"),
            "primary_reason_codes": list(target.get("primary_reason_codes") or []),
            "decision_confidence": target.get("decision_confidence"),
            "candidate_solution_ids": candidate_solution_ids,
            "group_ids": group_ids,
            "training_labels": labels,
        },
        "scene_type": str(labels.get("scene_type") or ""),
        "selection_mode": str(target.get("selection_mode") or ""),
        "decision_difficulty": str(labels.get("decision_difficulty") or ""),
    }


def _build_item_from_grpo_compact_record(record: Dict) -> Dict:
    reward_model = record.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth") or {}
    labels = ground_truth.get("training_labels") or {}
    messages = list(record.get("prompt") or [])
    if not messages:
        raise ValueError(f"GRPO compact record missing prompt messages: {record.get('record_id')}")
    return {
        "record_id": str(record.get("record_id") or ""),
        "messages": messages,
        "prompt": str(messages[-1].get("content") or ""),
        "ground_truth": ground_truth,
        "scene_type": str(labels.get("scene_type") or ""),
        "selection_mode": str(ground_truth.get("selection_mode") or ""),
        "decision_difficulty": str(labels.get("decision_difficulty") or ""),
    }


def build_eval_items(records: Sequence[Dict], keep_candidates_per_group: int, max_demand_cards: int) -> List[Dict]:
    items: List[Dict] = []
    for record in records:
        if "reward_model" in record and "prompt" in record:
            items.append(_build_item_from_grpo_compact_record(record))
        elif "compact_selection_input" in record and "messages" in record:
            items.append(_build_item_from_sft_compact_record(record))
        else:
            items.append(_build_item_from_raw_record(record, keep_candidates_per_group, max_demand_cards))
    return items


def render_prompt(tokenizer, messages: Sequence[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        joined = []
        for message in messages:
            joined.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        joined.append("assistant:")
        return "\n\n".join(joined)


def encode_prompt(tokenizer, messages: Sequence[Dict[str, str]], prompt_text: str) -> List[int]:
    try:
        return list(
            tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    except Exception:
        return list(tokenizer(prompt_text, add_special_tokens=True)["input_ids"])


def decode_prompt(tokenizer, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def safe_json_load(text: str) -> Optional[Dict]:
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidates.append(stripped[start : end + 1])
    fence = stripped.replace("```json", "```")
    if "```" in fence:
        parts = fence.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                candidates.append(part)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [float(value) for value in values if value is not None]
    if not kept:
        return None
    return float(statistics.fmean(kept))


def extract_scene_type(value) -> str:
    if isinstance(value, dict):
        return REWARD._str(value.get("scene_type"))
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                scene_type = REWARD._str(item.get("scene_type"))
                if scene_type:
                    return scene_type
            else:
                item_text = REWARD._str(item)
                if item_text:
                    return item_text
    return ""


def normalize_reason_codes(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = safe_json_load(value)
        if isinstance(parsed, list):
            return [REWARD._str(item) for item in parsed if REWARD._str(item)]
        single = REWARD._str(value)
        return [single] if single else []
    if isinstance(value, list):
        return [REWARD._str(item) for item in value if REWARD._str(item)]
    if isinstance(value, tuple) or isinstance(value, set):
        return [REWARD._str(item) for item in value if REWARD._str(item)]
    single = REWARD._str(value)
    return [single] if single else []


def prepare_prompts(
    args: argparse.Namespace,
    tokenizer,
    items: Sequence[Dict],
) -> Tuple[List[str], List[Dict], Dict]:
    effective_max_prompt_tokens = max(1, int(args.max_model_len) - int(args.max_new_tokens))
    prepared_prompts: List[str] = []
    prepared_items: List[Dict] = []
    overflow_count = 0
    truncated_count = 0
    skipped_count = 0
    max_original_prompt_tokens = 0

    for item in items:
        prompt_text = render_prompt(tokenizer, item["messages"])
        token_ids = encode_prompt(tokenizer, item["messages"], prompt_text)
        original_tokens = len(token_ids)
        max_original_prompt_tokens = max(max_original_prompt_tokens, original_tokens)

        prepared_item = dict(item)
        prepared_item["prompt_token_length_original"] = original_tokens
        prepared_item["prompt_token_length_used"] = original_tokens
        prepared_item["prompt_truncated"] = False

        if original_tokens > effective_max_prompt_tokens:
            overflow_count += 1
            if args.prompt_overflow_strategy == "error":
                raise ValueError(
                    f"Prompt for record_id={item['record_id']} has {original_tokens} tokens, "
                    f"which exceeds effective limit {effective_max_prompt_tokens}."
                )
            if args.prompt_overflow_strategy == "skip":
                skipped_count += 1
                continue

            token_ids = token_ids[:effective_max_prompt_tokens]
            prompt_text = decode_prompt(tokenizer, token_ids)
            truncated_count += 1
            prepared_item["prompt_token_length_used"] = len(token_ids)
            prepared_item["prompt_truncated"] = True

        prepared_prompts.append(prompt_text)
        prepared_items.append(prepared_item)

    prompt_stats = {
        "effective_max_prompt_tokens": effective_max_prompt_tokens,
        "overflow_count": overflow_count,
        "truncated_count": truncated_count,
        "skipped_count": skipped_count,
        "max_original_prompt_tokens": max_original_prompt_tokens,
        "kept_count": len(prepared_items),
    }
    return prepared_prompts, prepared_items, prompt_stats


def compute_reason_f1(pred_codes: Sequence[str], gt_codes: Sequence[str]) -> Optional[float]:
    pred = REWARD._safe_set(pred_codes or [])
    gt = REWARD._safe_set(gt_codes or [])
    return REWARD._f1(pred, gt)


def score_prediction(raw_output: str, parsed_prediction: Optional[Dict], item: Dict) -> Dict:
    gt = item["ground_truth"]
    pred = parsed_prediction or {}
    pred_solution_id = REWARD._str(pred.get("selected_solution_id"))
    pred_group_id = REWARD._str(pred.get("selected_group_id"))
    pred_scene_type = extract_scene_type(pred.get("training_labels"))
    pred_reason_codes = normalize_reason_codes(pred.get("primary_reason_codes"))

    gt_solution_id = REWARD._str(gt.get("selected_solution_id"))
    gt_group_id = REWARD._str(gt.get("selected_group_id"))
    gt_scene_type = extract_scene_type(gt.get("training_labels"))
    gt_reason_codes = normalize_reason_codes(gt.get("primary_reason_codes"))
    candidate_solution_ids = REWARD._safe_set(gt.get("candidate_solution_ids") or [])

    reward = REWARD.compute_score(
        "llm_selection_pareto_window",
        raw_output,
        gt,
        extra_info=None,
    )
    reason_f1 = compute_reason_f1(pred_reason_codes, gt_reason_codes)
    return {
        "json_valid": 1.0 if parsed_prediction is not None else 0.0,
        "candidate_valid": (
            None
            if not candidate_solution_ids
            else (1.0 if pred_solution_id in candidate_solution_ids else 0.0)
        ),
        "group_match": 1.0 if pred_group_id and pred_group_id == gt_group_id else 0.0,
        "solution_match": 1.0 if pred_solution_id and pred_solution_id == gt_solution_id else 0.0,
        "scene_match": (
            None if not gt_scene_type else (1.0 if pred_scene_type == gt_scene_type else 0.0)
        ),
        "reason_f1": reason_f1,
        "reward": reward,
    }


def aggregate_metrics(rows: Sequence[Dict]) -> Dict:
    return {
        "count": len(rows),
        "truncated_prompt_rate": safe_mean(1.0 if row.get("prompt_truncated") else 0.0 for row in rows),
        "json_valid_rate": safe_mean(row["metrics"]["json_valid"] for row in rows),
        "candidate_valid_rate": safe_mean(row["metrics"]["candidate_valid"] for row in rows),
        "group_accuracy": safe_mean(row["metrics"]["group_match"] for row in rows),
        "solution_accuracy": safe_mean(row["metrics"]["solution_match"] for row in rows),
        "scene_accuracy": safe_mean(row["metrics"]["scene_match"] for row in rows),
        "reason_f1": safe_mean(row["metrics"]["reason_f1"] for row in rows),
        "avg_reward": safe_mean(row["metrics"]["reward"] for row in rows),
    }


def aggregate_breakdown(rows: Sequence[Dict], field: str) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        key = str(row.get(field) or "")
        grouped[key].append(row)
    return {key: aggregate_metrics(group_rows) for key, group_rows in sorted(grouped.items())}


def maybe_import_vllm():
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        return None, None
    return LLM, SamplingParams


def load_tokenizer(model_path: Path, trust_remote_code: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def generate_with_vllm(args: argparse.Namespace, model_path: Path, prompts: Sequence[str]) -> List[str]:
    LLM, SamplingParams = maybe_import_vllm()
    if LLM is None or SamplingParams is None:
        raise RuntimeError("vLLM is not installed; use --backend transformers or install vllm.")

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    outputs: List[str] = []
    for start in range(0, len(prompts), max(1, args.batch_size)):
        batch = list(prompts[start : start + max(1, args.batch_size)])
        batch_outputs = llm.generate(batch, sampling_params)
        for item in batch_outputs:
            outputs.append(item.outputs[0].text if item.outputs else "")

    del llm
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()
    return outputs


def generate_with_transformers(
    args: argparse.Namespace,
    model_path: Path,
    prompts: Sequence[str],
    tokenizer,
) -> List[str]:
    import torch
    from transformers import AutoModelForCausalLM

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "device_map": "auto",
    }
    dtype_name = str(args.dtype).lower()
    if dtype_name in {"bf16", "bfloat16"}:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif dtype_name in {"fp16", "float16", "half"}:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    outputs: List[str] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=args.temperature > 0.0,
                temperature=None if args.temperature <= 0.0 else args.temperature,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[0][encoded["input_ids"].shape[1] :]
        outputs.append(tokenizer.decode(continuation, skip_special_tokens=True))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return outputs


def evaluate_one_model(args: argparse.Namespace, stage_name: str, model_path: Path, items: Sequence[Dict]) -> Dict:
    tokenizer = load_tokenizer(model_path, trust_remote_code=args.trust_remote_code)
    prompts, prepared_items, prompt_stats = prepare_prompts(args, tokenizer, items)

    backend = args.backend
    if backend == "auto":
        LLM, SamplingParams = maybe_import_vllm()
        backend = "vllm" if LLM is not None and SamplingParams is not None else "transformers"

    if backend == "vllm":
        raw_outputs = generate_with_vllm(args, model_path, prompts)
    else:
        raw_outputs = generate_with_transformers(args, model_path, prompts, tokenizer)

    rows: List[Dict] = []
    for item, raw_output in zip(prepared_items, raw_outputs):
        parsed_prediction = safe_json_load(raw_output)
        metrics = score_prediction(raw_output, parsed_prediction, item)
        rows.append(
            {
                "record_id": item["record_id"],
                "scene_type": item["scene_type"],
                "selection_mode": item["selection_mode"],
                "decision_difficulty": item["decision_difficulty"],
                "ground_truth": item["ground_truth"],
                "prediction": parsed_prediction,
                "raw_output": raw_output,
                "prompt_token_length_original": item.get("prompt_token_length_original"),
                "prompt_token_length_used": item.get("prompt_token_length_used"),
                "prompt_truncated": bool(item.get("prompt_truncated")),
                "metrics": metrics,
            }
        )

    return {
        "stage": stage_name,
        "resolved_model_path": str(model_path),
        "backend": backend,
        "prompt_stats": prompt_stats,
        "metrics": aggregate_metrics(rows),
        "by_scene_type": aggregate_breakdown(rows, "scene_type"),
        "by_selection_mode": aggregate_breakdown(rows, "selection_mode"),
        "by_decision_difficulty": aggregate_breakdown(rows, "decision_difficulty"),
        "predictions": rows,
    }


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def dataset_profile(items: Sequence[Dict]) -> Dict:
    return {
        "count": len(items),
        "scene_type_counts": dict(Counter(item["scene_type"] for item in items)),
        "selection_mode_counts": dict(Counter(item["selection_mode"] for item in items)),
        "decision_difficulty_counts": dict(Counter(item["decision_difficulty"] for item in items)),
    }


def main() -> None:
    args = parse_args()
    test_jsonl = args.test_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = load_test_records(test_jsonl, max_samples=args.max_samples)
    items = build_eval_items(
        raw_records,
        keep_candidates_per_group=args.keep_candidates_per_group,
        max_demand_cards=args.max_demand_cards,
    )

    stage_specs: List[Tuple[str, str]] = []
    if not args.skip_base and args.base_model_path:
        stage_specs.append(("base", args.base_model_path))
    if args.sft_model_path:
        stage_specs.append(("sft", args.sft_model_path))
    if args.grpo_model_path:
        stage_specs.append(("grpo", args.grpo_model_path))
    if not stage_specs:
        raise SystemExit("Please provide at least one model path to evaluate.")

    results: Dict[str, Dict] = {}
    for stage_name, raw_model_path in stage_specs:
        resolved_model_path = resolve_model_dir(raw_model_path)
        result = evaluate_one_model(args, stage_name, resolved_model_path, items)
        results[stage_name] = {
            "resolved_model_path": result["resolved_model_path"],
            "backend": result["backend"],
            "prompt_stats": result["prompt_stats"],
            "metrics": result["metrics"],
            "by_scene_type": result["by_scene_type"],
            "by_selection_mode": result["by_selection_mode"],
            "by_decision_difficulty": result["by_decision_difficulty"],
        }
        write_jsonl(output_dir / f"{stage_name}_predictions.jsonl", result["predictions"])

    summary = {
        "test_jsonl": str(test_jsonl),
        "dataset_profile": dataset_profile(items),
        "settings": {
            "keep_candidates_per_group": args.keep_candidates_per_group,
            "max_demand_cards": args.max_demand_cards,
            "max_samples": args.max_samples,
            "backend": args.backend,
            "prompt_overflow_strategy": args.prompt_overflow_strategy,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "results": results,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
