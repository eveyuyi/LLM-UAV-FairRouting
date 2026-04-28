"""Download specified folders from HuggingFace with rate-limit-safe single-worker mode."""
import os
import shutil
from huggingface_hub import snapshot_download

LOCAL_BASE = "/scratch/yy6120/code/LLM-UAV-FairRouting"
REPO_ID = "eveyuyi/LLM-UAV-FairRouting-data"

patterns = [
    "data/train/llm3_medium_5min_v1/**",
    "data/train/llm3_5min_large_v1/**",
    "data/test/test_seeds/**",
    "data/eval_runs/**",
]

print("Downloading from HuggingFace (max_workers=1)...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir=LOCAL_BASE + "/_hf_tmp",
    max_workers=1,
)
print("Download complete, moving files...")

src_base = os.path.join(LOCAL_BASE, "_hf_tmp", "data")
dst_base = os.path.join(LOCAL_BASE, "data")
count = 0
for root, dirs, files in os.walk(src_base):
    for fname in files:
        src_path = os.path.join(root, fname)
        rel = os.path.relpath(src_path, src_base)
        dst_path = os.path.join(dst_base, rel)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        count += 1
        if count % 100 == 0:
            print(f"  moved {count} files...")

shutil.rmtree(os.path.join(LOCAL_BASE, "_hf_tmp"), ignore_errors=True)
print(f"Done! Total {count} files moved.")
