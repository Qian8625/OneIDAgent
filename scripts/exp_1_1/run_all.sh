#!/usr/bin/env bash
# exp_1_1 全流程：P0 体检 -> P1 pilot -> P2 闭卷 -> P3 主跑(双卡) -> P4/P5 探针 -> P6 汇总
# 用法：bash scripts/exp_1_1/run_all.sh [pilot|full]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export ONEID_DATA_ROOT="${ONEID_DATA_ROOT:-/home/ubuntu/00_Data/RSTI}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/ubuntu/00_CPK}"
PY="${PY:-/home/ubuntu/miniconda3/envs/OEA/bin/python}"
STAGE="${1:-full}"
OUT="${OUT:-outputs/exp_1_1}"

echo "== P0 体检 =="
"$PY" -m scripts.exp_1_1.run --data "$ONEID_DATA_ROOT" --check

echo "== P1 pilot（4 臂 x 每任务 2 条）=="
"$PY" -m scripts.exp_1_1.run --data "$ONEID_DATA_ROOT" --cells arms --per-family 2 \
      --seed 1 --out "$OUT/pilot" --no-resume
"$PY" -m scripts.exp_1_1.analyze --traces "$OUT/pilot/traces" --out "$OUT/pilot/report"

if [[ "$STAGE" == "pilot" ]]; then
  echo "pilot 完成：$OUT/pilot/REPORT.md"
  exit 0
fi

echo "== P2 闭卷基线（分离记忆）=="
"$PY" -m scripts.exp_1_1.run --data "$ONEID_DATA_ROOT" --cells base_none,sft_none \
      --seed 1 --out "$OUT/run_seed1"

echo "== P3 主跑 seed=1（双卡并行）=="
CUDA_VISIBLE_DEVICES=0 "$PY" -m scripts.exp_1_1.run --data "$ONEID_DATA_ROOT" \
    --cells base_det,base_oneid --seed 1 --out "$OUT/run_seed1" \
    > "$OUT/run_seed1_gpu0.log" 2>&1 &
GPU0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -m scripts.exp_1_1.run --data "$ONEID_DATA_ROOT" \
    --cells sft_det,sft_oneid --seed 1 --out "$OUT/run_seed1" \
    > "$OUT/run_seed1_gpu1.log" 2>&1 &
GPU1=$!
wait $GPU0 $GPU1

echo "== P4/P5 探针 =="
"$PY" -m scripts.exp_1_1.probes memory --data "$ONEID_DATA_ROOT" --pairs 100 \
      --out "$OUT/probes" || true
"$PY" -m scripts.exp_1_1.probes splice --data "$ONEID_DATA_ROOT" --pairs 100 \
      --cells base_oneid,sft_oneid --out "$OUT/probes" || true

echo "== P6 汇总 =="
"$PY" -m scripts.exp_1_1.analyze --traces "$OUT/run_seed1/traces" \
      --probes "$OUT/probes" --out "$OUT/report"

echo "完成：$OUT/report/REPORT.md"
echo "补重复实验：for s in 2 3; do $PY -m scripts.exp_1_1.run --data $ONEID_DATA_ROOT --cells all --seed \$s --out $OUT/run_seed\$s; done"
