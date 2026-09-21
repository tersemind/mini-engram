#!/bin/bash
# 多租户托管演示：起服务 → 等两个租户都就绪 → 各问一题 → 关服务
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] && break
  echo "[gpu] 等待 GPU 空闲（当前 ${used}MiB）..."; sleep 60
done

bash scripts/serve.sh &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

READY=0
for i in $(seq 1 60); do
  MODELS=$(curl -s --max-time 2 http://127.0.0.1:8000/v1/models || true)
  if echo "$MODELS" | grep -q '"xinglan"' && echo "$MODELS" | grep -q '"hanhai"'; then
    READY=1; break
  fi
  sleep 10
done
[ "$READY" = 1 ] || { echo "[serve] 超时未就绪"; exit 1; }
echo "[serve] 双租户已就绪"

echo "----- 问 base vs xinglan -----"
$PY scripts/chat.py --tenant xinglan "星澜科技的报销系统叫什么？"
echo "----- 问 base vs hanhai -----"
$PY scripts/chat.py --tenant hanhai "瀚海机器人最大的客户是谁？"
echo "----- 交叉：问 hanhai 租户一个星澜的问题（应答不上）-----"
$PY scripts/chat.py --tenant hanhai "星澜科技的报销系统叫什么？" || true

kill $SERVER_PID 2>/dev/null || true
trap - EXIT
echo "[serve] 演示完成，服务已关闭"
