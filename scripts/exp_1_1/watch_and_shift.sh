#!/usr/bin/env bash
# 等 sft 两臂跑完 → 在 GPU1 上接力 base_oneid；GPU0 只保留 base_det。
# 不改变任何实验条件，只改负载分布。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TRACES="outputs/exp_1_1/run_full2/traces"
TARGET="$TRACES/sft_oneid_seed61.jsonl"
EXPECT=715
PY=/home/ubuntu/miniconda3/envs/OEA/bin/python
LOG="outputs/exp_1_1/watcher.log"

echo "[watch] $(date '+%H:%M:%S') 等待 sft 两臂完成（目标 $EXPECT 条）" >> "$LOG"
while true; do
  n=$(wc -l < "$TARGET" 2>/dev/null || echo 0)
  running=$(pgrep -fc "cells sft_det,sft_oneid" 2>/dev/null || echo 0)
  echo "[watch] $(date '+%H:%M:%S') sft_oneid=$n/$EXPECT running=$running" >> "$LOG"
  if [ "$n" -ge "$EXPECT" ] || [ "$running" -eq 0 ]; then
    break
  fi
  sleep 120
done

echo "[watch] $(date '+%H:%M:%S') sft 完成，等 30s 释放显存后启动 base_oneid@GPU1" >> "$LOG"
sleep 30
CUDA_VISIBLE_DEVICES=1 ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI \
  $PY -m scripts.exp_1_1.run --data /home/ubuntu/00_Data/RSTI \
      --cells base_oneid --seed 61 --tools prompt --out outputs/exp_1_1/run_full2 \
  >> "outputs/exp_1_1/full2_gpu1_shifted.log" 2>&1
echo "[watch] $(date '+%H:%M:%S') base_oneid 结束" >> "$LOG"
