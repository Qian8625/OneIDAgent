#!/usr/bin/env bash
# A/B：同批样本、同 seed，只改 --max-new-tokens，比较截断导致的失败率
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY=/home/ubuntu/miniconda3/envs/OEA/bin/python
DATA=/home/ubuntu/00_Data/RSTI
LOG=outputs/exp_1_1/ab_tokens.log

for tokens in 512 384; do
  echo "[ab] $(date '+%H:%M:%S') max_new_tokens=$tokens" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 ONEID_DATA_ROOT=$DATA $PY -m scripts.exp_1_1.run --data $DATA \
    --cells base_det --tasks task4,task5 --per-family 5 --seed 71 --tools prompt \
    --max-new-tokens "$tokens" --out "outputs/exp_1_1/ab_tokens_${tokens}" \
    >> "$LOG" 2>&1
done
echo "[ab] $(date '+%H:%M:%S') done" >> "$LOG"
