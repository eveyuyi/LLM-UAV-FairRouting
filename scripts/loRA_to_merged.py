#!/usr/bin/env python
import argparse
import os
import re
from pathlib import Path

import torch
import torch.distributed._tensor
from torch.distributed._tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a VERL SFT LoRA checkpoint into a HuggingFace model directory."
    )
    parser.add_argument(
        "--ckpt-dir",
        default=os.environ.get("CKPT_DIR"),
        help="SFT checkpoint step directory, e.g. data/checkpoints/.../global_step_123",
    )
    parser.add_argument(
        "--base-path",
        default=os.environ.get("BASE_PATH"),
        help="Base HuggingFace model path used for SFT training",
    )
    parser.add_argument(
        "--output-path",
        default=os.environ.get("OUTPUT_PATH"),
        help="Output HuggingFace directory; defaults to <ckpt-dir>/huggingface_lora_merged",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=int(os.environ.get("LORA_R", "32")),
        help="LoRA rank used during SFT",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=int(os.environ.get("LORA_ALPHA", "16")),
        help="LoRA alpha used during SFT",
    )
    args = parser.parse_args()
    if not args.ckpt_dir:
        parser.error("--ckpt-dir is required")
    if not args.base_path:
        parser.error("--base-path is required")
    if not args.output_path:
        args.output_path = str(Path(args.ckpt_dir) / "huggingface_lora_merged")
    return args


def resolve_local_dir(ckpt_dir: Path) -> Path:
    actor_dir = ckpt_dir / "actor"
    if actor_dir.is_dir():
        actor_shards = list(actor_dir.glob("model_world_size_*_rank_*.pt"))
        if actor_shards:
            return actor_dir
    return ckpt_dir


def sorted_shard_paths(local_dir: Path) -> list[Path]:
    pattern = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")
    matched = []
    for path in local_dir.glob("model_world_size_*_rank_*.pt"):
        m = pattern.fullmatch(path.name)
        if not m:
            continue
        world_size = int(m.group(1))
        rank = int(m.group(2))
        matched.append((world_size, rank, path))
    if not matched:
        raise FileNotFoundError(f"No shard files found under {local_dir}")
    world_sizes = {world_size for world_size, _, _ in matched}
    if len(world_sizes) != 1:
        raise RuntimeError(f"Inconsistent world_size shards under {local_dir}: {sorted(world_sizes)}")
    expected_world_size = next(iter(world_sizes))
    ranks = sorted(rank for _, rank, _ in matched)
    if ranks != list(range(expected_world_size)):
        raise RuntimeError(
            f"Shard ranks under {local_dir} are incomplete: expected 0..{expected_world_size - 1}, got {ranks}"
        )
    return [path for _, _, path in sorted(matched, key=lambda item: item[1])]


def unwrap_state_dict(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint


def local_tensor_and_shard_dim(value):
    if isinstance(value, DTensor):
        placements = getattr(value, "placements", ())
        shard_dims = [p.dim for p in placements if getattr(p, "__class__", type(p)).__name__ == "Shard"]
        shard_dim = shard_dims[0] if shard_dims else None
        return value.to_local().contiguous(), shard_dim
    if torch.is_tensor(value):
        return value.contiguous(), None
    return value, None


def reconstruct_tensor(values):
    local_tensors = []
    shard_dims = []
    for value in values:
        tensor, shard_dim = local_tensor_and_shard_dim(value)
        if not torch.is_tensor(tensor):
            return values[0]
        local_tensors.append(tensor)
        shard_dims.append(shard_dim)

    if all(torch.equal(local_tensors[0], tensor) for tensor in local_tensors[1:]):
        return local_tensors[0]

    shard_dims = [dim for dim in shard_dims if dim is not None]
    if shard_dims and len(set(shard_dims)) == 1:
        return torch.cat(local_tensors, dim=shard_dims[0])

    shapes = [tuple(tensor.shape) for tensor in local_tensors]
    ndim = local_tensors[0].ndim
    for dim in range(ndim):
        candidate = list(shapes[0])
        compatible = True
        for shape in shapes[1:]:
            if len(shape) != ndim:
                compatible = False
                break
            if any(shape[i] != candidate[i] for i in range(ndim) if i != dim):
                compatible = False
                break
        if compatible:
            return torch.cat(local_tensors, dim=dim)

    if ndim >= 1:
        return torch.cat(local_tensors, dim=0)
    raise RuntimeError(f"Cannot reconstruct scalar tensor from {len(local_tensors)} shards")


def load_base_model(base_path: str):
    kwargs = dict(trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    return model, tokenizer


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    base_path = args.base_path
    output_path = Path(args.output_path)
    scaling = args.lora_alpha / args.lora_rank

    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    if not Path(base_path, "config.json").is_file():
        raise FileNotFoundError(f"Base model must be a HuggingFace directory: {base_path}")

    local_dir = resolve_local_dir(ckpt_dir)
    shard_paths = sorted_shard_paths(local_dir)

    print("Step 1: Loading rank shards...")
    for shard_path in shard_paths:
        print(f"  - {shard_path}")
    shard_state_dicts = [unwrap_state_dict(path) for path in shard_paths]

    print("Step 2: Reconstructing full tensors from shards...")
    sd_full = {}
    for key in shard_state_dicts[0]:
        try:
            sd_full[key] = reconstruct_tensor([state_dict[key] for state_dict in shard_state_dicts])
        except Exception as exc:
            print(f"  WARNING {key}: {exc}, using rank0 local tensor")
            fallback_tensor, _ = local_tensor_and_shard_dim(shard_state_dicts[0][key])
            sd_full[key] = fallback_tensor

    sample_lora_A = next(key for key in sd_full if "lora_A.default.weight" in key)
    sample_lora_B = sample_lora_A.replace("lora_A", "lora_B")
    print(f"  Sample lora_A shape: {sd_full[sample_lora_A].shape}")
    print(f"  Sample lora_B shape: {sd_full[sample_lora_B].shape}")

    print("Step 3: Loading base model...")
    model, tokenizer = load_base_model(base_path)
    base_sd = model.state_dict()

    print("Step 4: Merging LoRA weights...")
    lora_modules = sorted(
        {
            key.replace(".lora_A.default.weight", "")
            for key in sd_full
            if "lora_A.default.weight" in key
        }
    )
    print(f"  Found {len(lora_modules)} LoRA modules")
    merged_count = 0
    skipped = []
    for module in lora_modules:
        lora_A = sd_full[f"{module}.lora_A.default.weight"].float()
        lora_B = sd_full[f"{module}.lora_B.default.weight"].float()
        base_key = module.replace("base_model.model.", "") + ".weight"
        if base_key not in base_sd:
            skipped.append(base_key)
            continue
        try:
            delta = lora_B @ lora_A * scaling
            base_sd[base_key] = (base_sd[base_key].float() + delta).to(torch.bfloat16)
            merged_count += 1
        except RuntimeError:
            print(
                f"  SHAPE ERROR {base_key}: lora_A={lora_A.shape}, "
                f"lora_B={lora_B.shape}, base={base_sd[base_key].shape}"
            )
            skipped.append(base_key)

    print(f"  Merged: {merged_count}, Skipped: {len(skipped)}")
    if skipped:
        print(f"  Skipped keys: {skipped[:5]}")

    print("Step 5: Saving merged model...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(base_sd)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print(f"\nDone! -> {output_path}")


if __name__ == "__main__":
    main()