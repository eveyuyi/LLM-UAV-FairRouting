"""Evaluate previously trained SFT models on seed-split validation data.

This is a compatibility bridge for SFT runs that finished before
`scripts/sweep_llm3_train.py` learned how to auto-evaluate completed trials.

Supported model inputs:
- a checkpoint root that contains `global_step_*` directories
- a direct `global_step_*` checkpoint directory
- an already merged HuggingFace model directory with `config.json`

For each supplied model, this script will:
1. resolve the concrete checkpoint/model directory
2. merge to HuggingFace format when needed
3. launch a temporary vLLM server
4. run fixed-demand rank-only validation on the requested val seeds
5. write one `trial_manifest.json`
6. refresh `leaderboard.jsonl` and `leaderboard.csv`
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT_PATH = REPO_ROOT / "scripts" / "sweep_llm3_train.py"


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location("sweep_llm3_train", SWEEP_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_name(text: str) -> str:
    lowered = text.strip().lower()
    sanitized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return sanitized or "model"


def _parse_model_specs(raw_specs: Sequence[str]) -> List[Tuple[str, Path]]:
    specs: List[Tuple[str, Path]] = []
    for raw in raw_specs:
        alias: Optional[str] = None
        path_str = raw
        if "=" in raw:
            left, right = raw.split("=", 1)
            if left and right:
                alias = left.strip() or None
                path_str = right.strip()
        path = Path(path_str).expanduser()
        specs.append((alias or path.name or "model", path))
    return specs


def _resolve_model_source(module, source_path: Path) -> Dict[str, object]:
    resolved = source_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Model source not found: {resolved}")

    if (resolved / "config.json").is_file():
        return {
            "kind": "merged_hf",
            "resolved_path": resolved,
            "latest_checkpoint": None,
            "merged_model_dir": resolved,
        }

    if resolved.is_dir() and resolved.name.startswith("global_step_"):
        return {
            "kind": "checkpoint",
            "resolved_path": resolved,
            "latest_checkpoint": resolved,
            "merged_model_dir": None,
        }

    latest = module.latest_global_step_dir(resolved)
    return {
        "kind": "checkpoint_root",
        "resolved_path": resolved,
        "latest_checkpoint": latest,
        "merged_model_dir": None,
    }


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(module) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing SFT checkpoints or merged HF models.")
    parser.add_argument("--dataset-root", type=Path, default=module.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-env", default="verl")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec. Supports /path/to/model or alias=/path/to/model. Repeat for multiple models.",
    )
    parser.add_argument("--val-seeds", nargs="+", default=["4109-4110"])
    parser.add_argument("--test-seeds", nargs="+", default=["4111-4112"])
    parser.add_argument("--priority-mode", default="llm-only", choices=("llm-only", "hybrid"))
    parser.add_argument("--baseline-mode", default="rule-only", choices=("rule-only",))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--served-model-name-prefix", default="llm3-eval")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--urgent-threshold", type=int, default=2)
    parser.add_argument("--port-base", type=int, default=19080)
    parser.add_argument("--startup-timeout-s", type=int, default=180)
    parser.add_argument("--startup-poll-interval-s", type=float, default=2.0)
    parser.add_argument("--stations-path", type=Path, default=module.DEFAULT_STATIONS_PATH)
    parser.add_argument("--building-data-path", type=Path, default=module.DEFAULT_BUILDING_DATA_PATH)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    module = _load_sweep_module()
    args = parse_args(module)

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    val_dirs = module.resolve_seed_dirs(dataset_root, module.parse_seed_specs(args.val_seeds))
    test_dirs = module.resolve_seed_dirs(dataset_root, module.parse_seed_specs(args.test_seeds))
    model_specs = _parse_model_specs(args.model)
    api_key = args.api_key or module.os.environ.get("OPENAI_API_KEY", "EMPTY")

    eval_settings = module.EvalSettings(
        enabled=True,
        priority_mode=args.priority_mode,
        baseline_mode=args.baseline_mode,
        api_key=api_key,
        served_model_name_prefix=args.served_model_name_prefix,
        host=args.host,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=not args.no_trust_remote_code,
        urgent_threshold=args.urgent_threshold,
        port_base=args.port_base,
        startup_timeout_s=args.startup_timeout_s,
        startup_poll_interval_s=args.startup_poll_interval_s,
        stations_path=str(args.stations_path.resolve()),
        building_data_path=str(args.building_data_path.resolve()),
    )

    plan = {
        "created_at": module._utc_now(),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "val_seeds": [int(path.name.split("_", 1)[1]) for path in val_dirs],
        "test_seeds": [int(path.name.split("_", 1)[1]) for path in test_dirs],
        "models": [str(path.resolve()) for _, path in model_specs],
        "eval": asdict(eval_settings),
        "dry_run": args.dry_run,
    }
    _write_json(output_root / "eval_plan.json", plan)

    trial_records_path = output_root / "trial_records.jsonl"

    for alias, raw_path in model_specs:
        source = _resolve_model_source(module, raw_path)
        source_basename = Path(str(source["resolved_path"])).name
        trial_name = f"sft_imported_{_safe_name(alias)}_{_safe_name(source_basename)}"
        trial_dir = output_root / "sft" / trial_name
        manifest_path = trial_dir / "trial_manifest.json"

        if args.skip_existing and manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("status") == "completed" and (payload.get("evaluation") or {}).get("enabled"):
                module._refresh_sft_leaderboard(output_root)
                continue

        latest_checkpoint = source["latest_checkpoint"]
        if latest_checkpoint is not None:
            latest_checkpoint = Path(str(latest_checkpoint))
            merged_model_dir = module.ensure_hf_checkpoint(
                sft_checkpoint_dir=latest_checkpoint,
                merged_model_dir=trial_dir / "merged_hf" / latest_checkpoint.name,
                conda_env=args.conda_env,
                dry_run=args.dry_run,
            )
        else:
            merged_model_dir = Path(str(source["merged_model_dir"]))

        evaluation = module.evaluate_served_model(
            trial_dir=trial_dir,
            merged_model_dir=merged_model_dir,
            val_dirs=val_dirs,
            conda_env=args.conda_env,
            eval_settings=eval_settings,
            dry_run=args.dry_run,
        )

        payload = {
            "stage": "sft",
            "status": "planned" if args.dry_run else "completed",
            "trial_name": trial_name,
            "created_at": module._utc_now(),
            "params": {
                "source_type": source["kind"],
                "source_alias": alias,
                "source_path": str(source["resolved_path"]),
                "imported": True,
            },
            "paths": {
                "trial_dir": str(trial_dir),
                "source_path": str(source["resolved_path"]),
                "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
                "merged_model_dir": str(merged_model_dir),
            },
            "splits": {
                "train_seeds": [],
                "val_seeds": [int(path.name.split("_", 1)[1]) for path in val_dirs],
                "test_seeds": [int(path.name.split("_", 1)[1]) for path in test_dirs],
            },
            "return_code": 0,
            "evaluation": evaluation,
        }
        _write_json(manifest_path, payload)
        module.append_jsonl(trial_records_path, module._build_trial_record_row(payload))
        module._refresh_sft_leaderboard(output_root)


if __name__ == "__main__":
    main()
