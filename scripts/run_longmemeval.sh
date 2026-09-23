#!/usr/bin/env bash
# LongMemEval-S(50) 端到端（v3：新配方全量版）
# prep → 50 租户各烤 adapter（GPU1/2/3 三链并行，避让 GPU0 的 dr-4b）→ bench → judge
# v3 配方：synth 加跨session12/时间10/偏好8 专项 pass；train r=32 epochs=8
# v2 特性保留：跳过已有 adapter 的租户；等待 GPU 空闲显存；失败重试
# 用法: bash scripts/run_longmemeval.sh 2>&1 | tee /root/train-study/logs/engram_G_longmemeval.log
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

$PY -m engram.longmemeval_prep || exit 1

wait_gpu() {  # $1=gpu idx, $2=所需空闲 MiB
  local gpu=$1 need=${2:-20500} free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || echo 0)
    [ "${free:-0}" -ge "$need" ] && return 0
    sleep 60
  done
}

chain() {
  local gpu=$1; shift
  local t
  for t in "$@"; do
    if [ -f "data/adapters/$t/adapter_config.json" ]; then
      echo "[chain gpu$gpu] 跳过 $t（adapter 已存在）"
      continue
    fi
    wait_gpu "$gpu" 20500
    echo "[chain gpu$gpu] synth $t (v3: +cross12 +temporal10 +preference8)"
    if ! timeout 5400 env CUDA_VISIBLE_DEVICES=$gpu $PY -m engram.synth --tenant "$t" --chat \
         --qa-per-chunk 4 --cross-session 12 --temporal 10 --preference 8 --update 10; then
      local try
      for try in 1 2 3 4 5; do
        echo "[chain gpu$gpu] synth $t 失败（第 $try 次重试前等待 5 分钟）"
        sleep 300
        wait_gpu "$gpu" 20500
        timeout 5400 env CUDA_VISIBLE_DEVICES=$gpu $PY -m engram.synth --tenant "$t" --chat \
          --qa-per-chunk 4 --cross-session 12 --temporal 10 --preference 8 --update 10 && break
        [ "$try" = 5 ] && return 1
      done
    fi
    echo "[chain gpu$gpu] train $t (r=32, epochs=8)"
    wait_gpu "$gpu" 18500
    if ! timeout 5400 env CUDA_VISIBLE_DEVICES=$gpu $PY -m engram.train_lora --tenant "$t" \
         --r 32 --epochs 8; then
      local try
      for try in 1 2 3 4 5; do
        echo "[chain gpu$gpu] train $t 失败（第 $try 次重试前等待 5 分钟）"
        sleep 300
        wait_gpu "$gpu" 18500
        timeout 5400 env CUDA_VISIBLE_DEVICES=$gpu $PY -m engram.train_lora --tenant "$t" \
          --r 32 --epochs 8 && break
        [ "$try" = 5 ] && return 1
      done
    fi
  done
}

mapfile -t TS < <(ls -d data/dataset/lme_* | xargs -n1 basename)
N=${#TS[@]}
A=(); B=(); C=()
for i in "${!TS[@]}"; do
  case $((i % 3)) in
    0) A+=("${TS[$i]}");; 1) B+=("${TS[$i]}");; 2) C+=("${TS[$i]}");;
  esac
done

chain 1 "${A[@]}" & P1=$!
chain 2 "${B[@]}" & P2=$!
chain 3 "${C[@]}" & P3=$!
RC=0
wait $P1 || RC=1
wait $P2 || RC=1
wait $P3 || RC=1
[ $RC -ne 0 ] && exit 1

echo "[run_longmemeval] 全部 adapter 就绪，开始 bench"
wait_gpu 1 21500
CUDA_VISIBLE_DEVICES=1 $PY -m engram.longmemeval_bench --conditions base,full,rag5,lora || exit 1

echo "[run_longmemeval] 开始 judge（OpenRouter deepseek-v3.2）"
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "需要 OPENROUTER_API_KEY 环境变量（judge 用）" >&2
  exit 1
fi
$PY -m engram.lme_judge --votes 3
