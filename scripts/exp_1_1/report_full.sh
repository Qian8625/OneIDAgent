#!/usr/bin/env bash
# 全量四臂跑完后出表：主表 + 8 项工具指标，统一用当前 score.py 口径（含属性归一化修复）
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY="${PY:-/home/ubuntu/miniconda3/envs/OEA/bin/python}"
TRACES="${TRACES:-outputs/exp_1_1/run_full/traces}"
OUT="${OUT:-outputs/exp_1_1/report_full}"
CELLS="base_det,base_oneid,sft_det,sft_oneid"

echo "== 主表 / 增益（--rescore）=="
"$PY" -m scripts.exp_1_1.analyze --traces "$TRACES" --rescore --out "$OUT"

echo "== 工具指标（--rescore）=="
"$PY" -m scripts.exp_1_1.metrics --traces "$TRACES" --rescore \
      --cells "$CELLS" --out "$OUT"

echo "完成：$OUT/REPORT.md  $OUT/metrics.csv  $OUT/tool_metrics.csv"
