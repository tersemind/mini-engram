#!/usr/bin/env bash
# LoCoMo 端到端：prep → 10 段对话各烤一个 adapter（GPU2/GPU3 双链并行）→ locomo_bench
# 用法: bash scripts/run_locomo.sh 2>&1 | tee /root/train-study/logs/engram_E_locomo.log
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

$PY -m engram.locomo_prep || exit 1

chain() {
  local gpu=$1; shift
  local t speakers
  for t in "$@"; do
    speakers=$(awk -F'\t' -v t="$t" '$1==t{print $2}' data/locomo/tenants.txt)
    echo "[chain gpu$gpu] synth $t ($speakers)"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m engram.synth --tenant "$t" --dialog \
      --speakers "$speakers" --qa-per-chunk 6 || return 1
    echo "[chain gpu$gpu] train $t"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m engram.train_lora --tenant "$t" \
      --epochs 6 || return 1
  done
}

mapfile -t TS < <(cut -f1 data/locomo/tenants.txt)
A=("${TS[@]:0:5}")
B=("${TS[@]:5}")
[ ${#B[@]} -eq 0 ] && B=("${A[@]:4}") && A=("${A[@]:0:4}")

chain 2 "${A[@]}" & P1=$!
chain 3 "${B[@]}" & P2=$!
RC=0
wait $P1 || RC=1
wait $P2 || RC=1
[ $RC -ne 0 ] && exit 1

echo "[run_locomo] 全部 adapter 就绪，开始 bench"
CUDA_VISIBLE_DEVICES=2 $PY -m engram.locomo_bench --conditions base,full,rag5,lora
