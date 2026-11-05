ONEID_HOME="${ONEID_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ONEID_DATA_ROOT="${ONEID_DATA_ROOT:-${ONEID_HOME}/data}"
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY must be exported (no hardcoded keys in eval scripts)}"

export DEEPSEEK_API_KEY

export accelerate_config="tool_server/tf_eval/scripts/configs/accelerate_config/deepspeed.yaml"

export CUDA_VISIBLE_DEVICES=0,1

mkdir -p logs

mkdir -p tool_server/tf_eval/results/infer_step_cross_time

mkdir -p tool_server/tf_eval/results/eval_step_cross_time

mkdir -p tool_server/tool_workers/tools_output

mkdir -p "${ONEID_DATA_ROOT}/gpkgs"

accelerate launch \

  --main_process_port 29500 \

  --config_file "${accelerate_config}" \

  -m tool_server.tf_eval \

  --model glm \

  --model_args pretrained=${ONEID_HOME}/models/0_basemodel/Glm-4-9b-chat \

  --task_name rsagent \

  --task_type step \

  --verbosity INFO \

  --output_path tool_server/tf_eval/results/eval_step_Glm-4-9b-chat \

  --batch_size 4 \

  --max_rounds 15 \

  --controller_addr http://0.0.0.0:20001 \

  --save_to_ckpt rsagent=tool_server/tf_eval/results/infer_step_cross_time/Glm-4-9b-chat_rsagent_step_ckpt.jsonl \

  --resume_from_ckpt rsagent=tool_server/tf_eval/results/infer_step_cross_time/Glm-4-9b-chat_rsagent_step_ckpt.jsonl \

  --cache_dir "${ONEID_DATA_ROOT}/gpkgs" \

  --save_dir tool_server/tool_workers/tools_output
