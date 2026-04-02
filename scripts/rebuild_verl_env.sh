#!/usr/bin/env bash
# 重建稳定的 VERL GRPO 训练环境
# 用法: bash scripts/rebuild_verl_env.sh
# 可选: ENV_NAME=my_env bash scripts/rebuild_verl_env.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-verl310_grpo_stable}"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
PIP="${CONDA_PREFIX:-$HOME/miniconda3}/envs/${ENV_NAME}/bin/pip"

echo "==> 1/5 删除旧环境（如存在）"
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true

echo "==> 2/5 创建 Python 3.10 基础环境"
conda create -n "${ENV_NAME}" python=3.10 pip setuptools wheel -y

echo "==> 3/5 安装 torch/torchvision/torchaudio (CUDA 12.8 wheel)"
"${PIP}" install \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --extra-index-url "${TORCH_INDEX}"

echo "==> 4/5 安装锁定依赖（verl + vllm 0.12.0 等）"
"${PIP}" install \
  --extra-index-url "${TORCH_INDEX}" \
  -r env/verl-grpo-requirements-lock.txt

echo "==> 5/5 快速自检"
"${PIP%%/pip}"/python - <<'PY'
import sys
checks = {
    'numpy':        '1.26',
    'torch':        '2.9',
    'transformers': '4',
    'verl':         '0.7',
    'vllm':         '0.12',
    'ray':          '2',
    'peft':         '0',
    'pydantic':     '1',
}
import importlib.metadata as m
ok = True
for pkg, expected in checks.items():
    try:
        ver = m.version(pkg)
        status = '✓' if ver.startswith(expected) else '✗ (got ' + ver + ')'
        if not ver.startswith(expected):
            ok = False
    except m.PackageNotFoundError:
        ver = 'NOT FOUND'
        status = '✗'
        ok = False
    print(f'  {pkg:20s} {ver:20s} {status}')
sys.exit(0 if ok else 1)
PY

echo ""
echo "==> 环境重建完成！激活命令:"
echo "    conda activate ${ENV_NAME}"
echo "    bash scripts/training_grpo.sh"
