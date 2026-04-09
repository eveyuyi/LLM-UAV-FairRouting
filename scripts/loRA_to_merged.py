import torch
import torch.distributed._tensor
from torch.distributed._tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
CKPT_DIR = "data/checkpoints/expA_sft_baseline/global_step_594"
BASE_PATH = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/model--Qwen-Qwen3-4B-Instruct-2507"
OUTPUT_PATH = f"{CKPT_DIR}/huggingface_lora_merged"
LORA_R = 32
LORA_ALPHA = 16
SCALING = LORA_ALPHA / LORA_R  # 0.5
print("Step 1: Loading both rank shards...")
sd0 = torch.load(f"{CKPT_DIR}/model_world_size_2_rank_0.pt", map_location="cpu", weights_only=False)
sd1 = torch.load(f"{CKPT_DIR}/model_world_size_2_rank_1.pt", map_location="cpu", weights_only=False)
if "state_dict" in sd0: sd0 = sd0["state_dict"]
if "state_dict" in sd1: sd1 = sd1["state_dict"]
def reconstruct_tensor(v0, v1):
    """从两个rank的DTensor/Tensor重建完整权重"""
    # 先拿local tensor
    t0 = v0.to_local().contiguous() if isinstance(v0, DTensor) else v0.contiguous()
    t1 = v1.to_local().contiguous() if isinstance(v1, DTensor) else v1.contiguous()
    if t0.shape == t1.shape:
        if torch.allclose(t0.float(), t1.float(), atol=1e-6):
            return t0  # replicated，取任一
        else:
            return torch.cat([t0, t1], dim=0)
    else:
        for dim in range(len(t0.shape)):
            if t0.shape[dim] != t1.shape[dim]:
                return torch.cat([t0, t1], dim=dim)
        raise ValueError(f"Cannot reconstruct: {t0.shape} vs {t1.shape}")
print("Step 2: Reconstructing full tensors from shards...")
sd_full = {}
for k in sd0:
    try:
        sd_full[k] = reconstruct_tensor(sd0[k], sd1[k])
    except Exception as e:
        print(f"  WARNING {k}: {e}, using rank0 local")
        v0 = sd0[k]
        sd_full[k] = v0.to_local().contiguous() if isinstance(v0, DTensor) else v0.contiguous()
sample_lora_A = next(k for k in sd_full if "lora_A.default.weight" in k)
sample_lora_B = sample_lora_A.replace("lora_A", "lora_B")
print(f"  Sample lora_A shape: {sd_full[sample_lora_A].shape}")  # 应该是 [r, in_features]
print(f"  Sample lora_B shape: {sd_full[sample_lora_B].shape}")  # 应该是 [out_features, r]
print("Step 3: Loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE_PATH, dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
base_sd = model.state_dict()
print("Step 4: Merging LoRA weights...")
lora_modules = sorted(set(
    k.replace(".lora_A.default.weight", "")
    for k in sd_full if "lora_A.default.weight" in k
))
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
        delta = lora_B @ lora_A * SCALING
        base_sd[base_key] = (base_sd[base_key].float() + delta).to(torch.bfloat16)
        merged_count += 1
    except RuntimeError as e:
        print(f"  SHAPE ERROR {base_key}: lora_A={lora_A.shape}, lora_B={lora_B.shape}, base={base_sd[base_key].shape}")
        skipped.append(base_key)
print(f"  Merged: {merged_count}, Skipped: {len(skipped)}")
if skipped:
    print(f"  Skipped keys: {skipped[:5]}")
print("Step 5: Saving merged model...")
os.makedirs(OUTPUT_PATH, exist_ok=True)
model.load_state_dict(base_sd)
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"\nDone! -> {OUTPUT_PATH}")