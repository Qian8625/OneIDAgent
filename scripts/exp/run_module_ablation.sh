#!/usr/bin/env bash
# Question-1 experiment: hold the SFT checkpoint fixed, swap only the estimator
# behind CrossModalInterpret, measure final-answer accuracy.
#
#   A0  legacy_hard      -> conventional hard geometry match (module off proxy)
#   A1  candidate_graph  -> physical candidate graph + prior, no inference
#   A3* shared_identity  -> the paper's module, in five config rungs
#
# Nothing is retrained.  Prompt / tool schema / decoder / checkpoint are
# identical across arms; only these two env vars differ:
#   ONEID_FORCE_MODE  (worker, wins over any `mode` the model passes)
#   ONEID_VI_CONFIG   (arm YAML from scripts/exp/arms/)
#
# Usage
#   export EVAL_MODEL_API_KEY=... JUDGE_API_KEY=...
#   bash scripts/exp/run_module_ablation.sh               # task4 subset, core arms
#   ARMS="A0_hard A3_F" DATASET=test.jsonl bash scripts/exp/run_module_ablation.sh
#   MODEL_PATH=$PWD/models/0_basemodel/Qwen3-4B-Instruct-2507 \
#       ARMS="A0_hard A3_F" RUN_TAG=base ...               # the SFT-off column
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

ONEID_DATA_ROOT="${ONEID_DATA_ROOT:-/home/ubuntu/00_Data/RSTI}"
export ONEID_DATA_ROOT

# same checkpoint as the paper's Table 1 unless overridden
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/0_sft_train/qwen3_sft_text_0614}"
DATASET="${DATASET:-test_task4.jsonl}"      # task4 subset; use test.jsonl for the full pass
MAX_ROUNDS="${MAX_ROUNDS:-15}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/module_ablation/${RUN_TAG}}"
ARMS="${ARMS:-A0_hard A1_prior A3_online A3_F A3_F_nobudget A3_F_nocap A3_F_nofeedback}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export MAX_ROUNDS
export MODEL_PATH

# ---- one-shot prep -------------------------------------------------------- #
python scripts/exp/make_arm_configs.py >/dev/null
ARMDIR="${REPO_ROOT}/scripts/exp/arms"
SUBSET_PATH="${ONEID_DATA_ROOT}/${DATASET}"
if [[ ! -f "${SUBSET_PATH}" ]]; then
  echo "building ${SUBSET_PATH}"
  python scripts/exp/make_task4_subset.py \
    --src "${ONEID_DATA_ROOT}/test.jsonl" --out "${SUBSET_PATH}"
fi
export ONEID_DATASET_PATH="${SUBSET_PATH}"
EXPECTED=$(wc -l < "${SUBSET_PATH}")

arm_mode()   { case "$1" in A0_*) echo legacy_hard;; A1_*) echo candidate_graph;; *) echo shared_identity;; esac; }
arm_config() { case "$1" in A0_*|A1_*) echo "";; A3_online) echo f_online;; A3_F) echo f_ctx;;
                 A3_F_nobudget) echo f_ctx_nobudget;; A3_F_nocap) echo f_ctx_nocap;;
                 A3_F_nofeedback) echo f_ctx_nofeedback;; *) echo f_ctx;; esac; }

mkdir -p "${OUT_ROOT}"
MANIFEST="${OUT_ROOT}/manifest.jsonl"
if ! grep -qs "\"run_tag\":\"${RUN_TAG}\"" "${MANIFEST}" 2>/dev/null; then
  printf '{"run_tag":"%s","model":"%s","dataset":"%s","expected":%s}\n' \
    "${RUN_TAG}" "${MODEL_PATH}" "${DATASET}" "${EXPECTED}" >> "${MANIFEST}"
fi

echo "=============================================================="
echo "run_tag : ${RUN_TAG}"
echo "model   : ${MODEL_PATH}"
echo "dataset : ${SUBSET_PATH}  (${EXPECTED} samples)"
echo "out     : ${OUT_ROOT}"
echo "arms    : ${ARMS}"
echo "=============================================================="

for ARM in ${ARMS}; do
  MODE=$(arm_mode "${ARM}")
  CFG=$(arm_config "${ARM}")

  export OUTPUT_ROOT="${OUT_ROOT}"
  export RESULT_NAME="${RUN_TAG}_${ARM}"
  export OUTPUT_PATH="${OUT_ROOT}/${ARM}/eval"
  export SAVE_DIR="${OUT_ROOT}/${ARM}/tools_output"
  export CKPT_PATH="${OUT_ROOT}/${ARM}/ckpt/${RESULT_NAME}_rsagent_e2e_ckpt.jsonl"
  export ONEID_ARM_LOG="${OUT_ROOT}/${ARM}/armlog.jsonl"
  export ONEID_FORCE_MODE="${MODE}"

  if [[ -n "${CFG}" ]]; then
    export ONEID_VI_CONFIG="${ARMDIR}/${CFG}.yaml"
  else
    unset ONEID_VI_CONFIG || true
  fi

  mkdir -p "${OUTPUT_PATH}" "${SAVE_DIR}" "$(dirname "${CKPT_PATH}")"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${CKPT_PATH}" ]]; then
    DONE=$(wc -l < "${CKPT_PATH}")
    if (( DONE >= EXPECTED )); then
      echo "[skip] ${ARM}: ckpt already has ${DONE}/${EXPECTED}"
      continue
    fi
    echo "[resume] ${ARM}: ${DONE}/${EXPECTED}"
  fi

  printf '{"arm":"%s","mode":"%s","config":"%s","destructive":false}\n' \
    "${ARM}" "${MODE}" "${CFG:-none}" >> "${MANIFEST}"

  echo
  echo "---- ${ARM}  mode=${MODE}  config=${CFG:-none} ----"
  bash scripts/eval/eval_e2e.sh
done

echo
echo "all arms finished.  results under ${OUT_ROOT}"
echo "next: python scripts/exp/analyze_module_ablation.py --root ${OUT_ROOT}"
