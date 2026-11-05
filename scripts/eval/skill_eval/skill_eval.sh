export ONEID_HOME="${ONEID_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export PYTHONPATH=${ONEID_HOME}
export CUDA_VISIBLE_DEVICES=1
export PRETRAINED_PATH=${ONEID_HOME}/models/0_sft_train/qwen3_sft_text_skill_0611
export ONEID_DATA_ROOT=${ONEID_DATA_ROOT:-${ONEID_HOME}/data}
export SKILL_EVAL_DATA_ROOT=${SKILL_EVAL_DATA_ROOT:-${ONEID_DATA_ROOT}}
export ONEID_IMAGE_ROOT=${ONEID_IMAGE_ROOT:-${SKILL_EVAL_DATA_ROOT}}

# Do not force Rasterio/Fiona/pyproj to use a PROJ/GDAL
# database from another binary installation.
unset PROJ_LIB
unset PROJ_DATA
unset GDAL_DATA

# Use the official EPSG definition when GeoTIFF keys differ slightly.
export GTIFF_SRS_SOURCE=EPSG

python -m tool_server.tf_eval.tasks.rsagent_skill_ablation.task \
  --dataset-path ${SKILL_EVAL_DATA_ROOT}/test_skill.jsonl \
  --output-dir ${ONEID_HOME}/outputs/skill_eval/qwen3_sft_text_skill_0611 \
  --max-rounds 16 \
  --image-root "${ONEID_IMAGE_ROOT}" \
  --dataset-root "${SKILL_EVAL_DATA_ROOT}" \
  --modes no_skill,auto_skill,oracle_skill
