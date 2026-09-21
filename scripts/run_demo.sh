#!/bin/bash
# 端到端 demo：虚构公司语料 → 合成 QA → 烤 LoRA → 闭卷/烤入对比评测
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
TENANT="${1:-xinglan}"

# GPU 守卫：SEAL(1B) 实验优先用卡，本 demo 等显存空闲再进 GPU 环节
wait_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && break
    echo "[gpu] 显存已占用 ${used}MiB，等待优先任务释放 GPU（每分钟检查）..."
    sleep 60
  done
}

echo "===== [1/5] 生成 demo 语料 ====="
$PY scripts/make_demo_corpus.py
echo "===== [2/5] ingest: 语料切块 ====="
$PY -m engram.ingest --tenant "$TENANT" --source data/corpus/"$TENANT"
echo "===== [3/5] synth: self-study 合成 QA ====="
wait_gpu
$PY -m engram.synth --tenant "$TENANT"
echo "===== [4/5] train: 烤 LoRA adapter ====="
wait_gpu
$PY -m engram.train_lora --tenant "$TENANT"
echo "===== [5/5] eval: 闭卷 vs 烤入后对比 ====="
wait_gpu
$PY -m engram.eval --tenant "$TENANT"
echo "===== demo 完成 ====="
