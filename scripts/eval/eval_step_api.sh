#!/bin/bash

ONEID_HOME="${ONEID_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ONEID_DATA_ROOT="${ONEID_DATA_ROOT:-${ONEID_HOME}/data}"
set -euo pipefail



: "${EVAL_MODEL_API_KEY:?EVAL_MODEL_API_KEY must be exported}"

export EVAL_MODEL_API_KEY

: "${JUDGE_API_KEY:?JUDGE_API_KEY must be exported}"

export JUDGE_API_KEY



EVAL_MODEL_NAME=mimo-v2-omni

EVAL_MODEL_API_KEY_ENV=${EVAL_MODEL_API_KEY_ENV:-EVAL_MODEL_API_KEY}

EVAL_MODEL_API_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1



JUDGE_MODEL=deepseek-v4-flash

JUDGE_API_KEY_ENV=${JUDGE_API_KEY_ENV:-JUDGE_API_KEY}

JUDGE_API_URL=https://api.deepseek.com/chat/completions



MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

TEMPERATURE=${TEMPERATURE:-0}

RETRIES=${RETRIES:-6}

ENABLE_THINKING=${ENABLE_THINKING:-false}

BATCH_SIZE=${BATCH_SIZE:-1}

MAX_ROUNDS=${MAX_ROUNDS:-15}

CONTROLLER_ADDR=${CONTROLLER_ADDR:-http://0.0.0.0:20001}

RESULT_NAME=${RESULT_NAME:-${EVAL_MODEL_NAME//\//_}}

CACHE_DIR=${CACHE_DIR:-${ONEID_DATA_ROOT}/gpkgs}

SAVE_DIR=${SAVE_DIR:-tool_server/tool_workers/tools_output}

OUTPUT_PATH=${OUTPUT_PATH:-tool_server/tf_eval/results/eval_step_api/${RESULT_NAME}}

CKPT_PATH=${CKPT_PATH:-tool_server/tf_eval/results/infer_step_api/${RESULT_NAME}_rsagent_step_ckpt.jsonl}



if [[ -z "${!EVAL_MODEL_API_KEY_ENV:-}" ]]; then

  echo "Missing eval model API key: export ${EVAL_MODEL_API_KEY_ENV}=..." >&2

  exit 1

fi



if [[ -z "${!JUDGE_API_KEY_ENV:-}" ]]; then

  echo "Missing judge API key: export ${JUDGE_API_KEY_ENV}=..." >&2

  exit 1

fi



mkdir -p logs

mkdir -p "$(dirname "${CKPT_PATH}")"

mkdir -p "${OUTPUT_PATH}"

mkdir -p "${SAVE_DIR}"

mkdir -p "${CACHE_DIR}"



MODEL_ARGS="pretrained=${EVAL_MODEL_NAME},api_key_env=${EVAL_MODEL_API_KEY_ENV},max_model_len=${MAX_MODEL_LEN},temperature=${TEMPERATURE},retries=${RETRIES},enable_thinking=${ENABLE_THINKING}"

if [[ -n "${EVAL_MODEL_API_BASE_URL}" ]]; then

  MODEL_ARGS="${MODEL_ARGS},base_url=${EVAL_MODEL_API_BASE_URL}"

fi



export JUDGE_API_KEY="${!JUDGE_API_KEY_ENV}"

export JUDGE_API_URL

export JUDGE_MODEL



python -m tool_server.tf_eval \

  --model openai \

  --model_args "${MODEL_ARGS}" \

  --task_name rsagent \

  --task_type step \

  --verbosity INFO \

  --output_path "${OUTPUT_PATH}" \

  --batch_size "${BATCH_SIZE}" \

  --max_rounds "${MAX_ROUNDS}" \

  --controller_addr "${CONTROLLER_ADDR}" \

  --save_to_ckpt "rsagent=${CKPT_PATH}" \

  --resume_from_ckpt "rsagent=${CKPT_PATH}" \

  --cache_dir "${CACHE_DIR}" \

  --save_dir "${SAVE_DIR}"
