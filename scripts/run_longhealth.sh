#!/usr/bin/env bash
# LongHealth 端到端：prep → 20 个病人各烤一个 adapter（GPU0/GPU1 双链并行）→ longhealth_bench
# 用法: bash scripts/run_longhealth.sh 2>&1 | tee /root/train-study/logs/engram_F_longhealth.log
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

$PY -m engram.longhealth_prep || exit 1

chain() {
  local gpu=$1; shift
  local t
  for t in "$@"; do
    echo "[chain gpu$gpu] synth $t"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m engram.synth --tenant "$t" \
      --company "${t#lh_}" --qa-per-chunk 8 || return 1
    echo "[chain gpu$gpu] train $t"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m engram.train_lora --tenant "$t" \
      --company "${t#lh_}" --epochs 20 || return 1
  done
}

mapfile -t TS < <(ls -d data/dataset/lh_* | xargs -n1 basename)
chain 1 "${TS[@]}"
RC=$?
[ $RC -ne 0 ] && exit 1

echo "[run_longhealth] 全部 adapter 就绪，开始 bench"
CUDA_VISIBLE_DEVICES=0 $PY -m engram.longhealth_bench --conditions base,full,rag5,lora
