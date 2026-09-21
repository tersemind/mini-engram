#!/bin/bash
# 实验编排：① synth 增强重训 ② 遗忘检查 ③ 第二租户+串味检查+多 LoRA 托管
# 每个 GPU 环节前等 SEAL 释放显存（SEAL 优先）
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python

wait_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && break
    echo "[gpu] 等待 SEAL 释放 GPU（当前 ${used}MiB）..."; sleep 60
  done
}

echo "########## 实验①: synth 双通道增强 → 重训 xinglan → 复测"
wait_gpu; $PY -m engram.synth --tenant xinglan --company 星澜科技
wait_gpu; $PY -m engram.train_lora --tenant xinglan --company 星澜科技 --epochs 20 --lr 2e-4 --grad-accum 1
wait_gpu; $PY -m engram.eval --tenant xinglan --company 星澜科技

echo "########## 实验②: 遗忘检查 + 泄漏探针"
wait_gpu; $PY -m engram.forgetting_check --tenant xinglan

echo "########## 实验③: 第二租户 hanhai 全流程"
$PY scripts/make_demo_corpus2.py
$PY -m engram.ingest --tenant hanhai --source data/corpus/hanhai
wait_gpu; $PY -m engram.synth --tenant hanhai --company 瀚海机器人
wait_gpu; $PY -m engram.train_lora --tenant hanhai --company 瀚海机器人 --epochs 20 --lr 2e-4 --grad-accum 1
wait_gpu; $PY -m engram.eval --tenant hanhai --company 瀚海机器人

echo "########## 实验③b: 串味检查（交叉问答，应低分）"
wait_gpu; $PY -m engram.eval --tenant hanhai --company 瀚海机器人 --quiz xinglan || true
wait_gpu; $PY -m engram.eval --tenant xinglan --company 星澜科技 --quiz hanhai || true

echo "########## 实验④: 多租户托管演示（vLLM multi-LoRA）"
wait_gpu
bash scripts/serve.sh &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
  MODELS=$(curl -s --max-time 2 http://127.0.0.1:8000/v1/models || true)
  echo "$MODELS" | grep -q xinglan && echo "$MODELS" | grep -q hanhai && break
  sleep 10
done
echo "[serve] 服务已就绪: $(echo "$MODELS" | head -c 300)"
echo "----- 问 xinglan 租户 -----"
$PY scripts/chat.py --tenant xinglan "星澜科技的报销系统叫什么？" || true
echo "----- 问 hanhai 租户 -----"
$PY scripts/chat.py --tenant hanhai "瀚海机器人最大的客户是谁？" || true
kill $SERVER_PID 2>/dev/null || true
trap - EXIT

echo "########## 全部实验完成"
