#!/bin/bash
# 多租户托管：一个 base model 挂载 data/adapters/ 下的所有 LoRA。
# 用法: scripts/serve.sh [base_model路径或名字] [端口]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${2:-8000}"
PY=.venv/bin/python

MODEL_PATH=$($PY -c "from engram.common import resolve_model; print(resolve_model('$MODEL'))")

# flashinfer JIT 采样内核需要 ninja，确保 venv 的 bin 在 PATH 里
export PATH="$PWD/.venv/bin:$PATH"

LORA_MODULES=()
for d in data/adapters/*/; do
  [ -f "$d/adapter_config.json" ] || continue
  name=$(basename "$d")
  LORA_MODULES+=("$name=$d")
  echo "[serve] 挂载租户 adapter: $name -> $d"
done

LORA_ARGS=()
if [ ${#LORA_MODULES[@]} -gt 0 ]; then
  LORA_ARGS=(--lora-modules "${LORA_MODULES[@]}")
else
  echo "[serve] 警告: data/adapters/ 下没有任何 adapter，仅启动 base model"
fi

exec .venv/bin/vllm serve "$MODEL_PATH" \
  --enable-lora --max-lora-rank 64 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --port "$PORT" \
  "${LORA_ARGS[@]}"
