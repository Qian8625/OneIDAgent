#!/usr/bin/env bash

ONEID_HOME="${ONEID_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ONEID_DATA_ROOT="${ONEID_DATA_ROOT:-${ONEID_HOME}/data}"
set -euo pipefail



: "${EVAL_MODEL_API_KEY:?EVAL_MODEL_API_KEY must be exported}"

export EVAL_MODEL_API_KEY

: "${JUDGE_API_KEY:?JUDGE_API_KEY must be exported}"

export JUDGE_API_KEY



SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

cd "${REPO_ROOT}"



export accelerate_config="${ACCELERATE_CONFIG:-tool_server/tf_eval/scripts/configs/accelerate_config/deepspeed.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5}"

export JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"

export JUDGE_API_URL="${JUDGE_API_URL:-https://api.deepseek.com/chat/completions}"

DEFAULT_PROJ_DATA="$(python -c 'import os, rasterio; print(os.path.join(os.path.dirname(rasterio.__file__), "proj_data"))' 2>/dev/null || true)"

if [[ -d "${DEFAULT_PROJ_DATA}" ]]; then

  export PROJ_DATA="${DEFAULT_PROJ_DATA}"

  export PROJ_LIB="${DEFAULT_PROJ_DATA}"

else

  printf 'ERROR: expected PROJ_DATA directory not found: %s\n' "${DEFAULT_PROJ_DATA}" >&2

  exit 1

fi



MODEL_NAME="${MODEL_NAME:-qwen}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/0_sft_train/qwen3_sft_text_0614}"

BATCH_SIZE="${BATCH_SIZE:-1}"

MAX_ROUNDS="${MAX_ROUNDS:-15}"

SKILL_MODE="${SKILL_MODE:-no_skill}"

MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29401}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"

CONTROLLER_ADDR="${CONTROLLER_ADDR:-http://0.0.0.0:20001}"

CACHE_DIR="${CACHE_DIR:-${ONEID_DATA_ROOT}/gpkgs}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/e2e_output}"

RESULT_NAME="${RESULT_NAME:-${MODEL_PATH##*/}}"

RESULT_NAME="${RESULT_NAME//\//_}"

RESULT_NAME="${RESULT_NAME// /_}"

OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/${RESULT_NAME}/eval}"

SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/${RESULT_NAME}/tools_output}"

CKPT_PATH="${CKPT_PATH:-${OUTPUT_ROOT}/${RESULT_NAME}/ckpt/${RESULT_NAME}_rsagent_e2e_ckpt.jsonl}"

DATASET_PATH="${DATASET_PATH:-${ONEID_DATA_ROOT}/test.jsonl}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

RUN_LOG="${RUN_LOG:-${OUTPUT_ROOT}/${RESULT_NAME}/logs/eval_e2e_${RUN_ID}.log}"



mkdir -p logs

mkdir -p "$(dirname "${RUN_LOG}")"

mkdir -p "$(dirname "${CKPT_PATH}")"

mkdir -p "${OUTPUT_PATH}"

mkdir -p "${SAVE_DIR}"

mkdir -p "${CACHE_DIR}"



exec > >(tee -a "${RUN_LOG}") 2>&1



printf 'RUN_ID=%s\n' "${RUN_ID}"

printf 'MODEL_NAME=%s\n' "${MODEL_NAME}"

printf 'MODEL_PATH=%s\n' "${MODEL_PATH}"

printf 'RESULT_NAME=%s\n' "${RESULT_NAME}"

printf 'OUTPUT_ROOT=%s\n' "${OUTPUT_ROOT}"

printf 'CKPT_PATH=%s\n' "${CKPT_PATH}"

printf 'OUTPUT_PATH=%s\n' "${OUTPUT_PATH}"

printf 'SAVE_DIR=%s\n' "${SAVE_DIR}"

printf 'MAX_ROUNDS=%s\n' "${MAX_ROUNDS}"

printf 'SKILL_MODE=%s\n' "${SKILL_MODE}"

printf 'NUM_PROCESSES=%s\n' "${NUM_PROCESSES}"

printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"

printf 'PROJ_DATA=%s\n' "${PROJ_DATA:-}"

printf 'PROJ_LIB=%s\n' "${PROJ_LIB:-}"

printf 'RUN_LOG=%s\n' "${RUN_LOG}"



accelerate launch \

  --num_processes "${NUM_PROCESSES}" \

  --main_process_port "${MAIN_PROCESS_PORT}" \

  --config_file "${accelerate_config}" \

  -m tool_server.tf_eval \

  --model "${MODEL_NAME}" \

  --model_args "pretrained=${MODEL_PATH}" \

  --task_name rsagent \

  --task_type e2e \

  --skill_mode "${SKILL_MODE}" \

  --verbosity INFO \

  --output_path "${OUTPUT_PATH}" \

  --batch_size "${BATCH_SIZE}" \

  --max_rounds "${MAX_ROUNDS}" \

  --controller_addr "${CONTROLLER_ADDR}" \

  --save_to_ckpt "rsagent=${CKPT_PATH}" \

  --resume_from_ckpt "rsagent=${CKPT_PATH}" \

  --cache_dir "${CACHE_DIR}" \

  --save_dir "${SAVE_DIR}"
