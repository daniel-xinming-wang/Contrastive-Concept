#!/bin/bash
set -euo pipefail

#MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
CATEGORIES="${CATEGORIES:-}"
MAX_PAIRS_PER_CATEGORY="${MAX_PAIRS_PER_CATEGORY:-}"
MAX_STATEMENTS="${MAX_STATEMENTS:-100}"
BATCH_SIZE="${BATCH_SIZE:-100}"
SAVE_FORMAT="${SAVE_FORMAT:-npy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_v2/hidden_states_system_user_conflict_v2_${MODEL##*/}}"
VARIANTS="${VARIANTS:-system_positive_only system_negative_only syspos_userneg_conflict sysneg_userpos_conflict}"

CONCEPTS_FILE="${CONCEPTS_FILE:-contrastive_concepts.txt}"
STATEMENT_FILE="${STATEMENT_FILE:-contrastive_hidden_states/data/statements_300/class_1.txt}"
DTYPE="${DTYPE:-}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ADD_GENERATION_PROMPT="${ADD_GENERATION_PROMPT:-1}"
MANUAL_QWEN_CHAT_TEMPLATE="${MANUAL_QWEN_CHAT_TEMPLATE:-0}"

ARGS=(
  --model "${MODEL}"
  --concepts-file "${CONCEPTS_FILE}"
  --statement-file "${STATEMENT_FILE}"
  --max-statements "${MAX_STATEMENTS}"
  --batch-size "${BATCH_SIZE}"
  --save-format "${SAVE_FORMAT}"
  --output-dir "${OUTPUT_DIR}"
  --device-map "${DEVICE_MAP}"
)

if [[ -n "${DTYPE}" ]]; then
  ARGS+=(--torch-dtype "${DTYPE}")
fi

if [[ -n "${CATEGORIES}" ]]; then
  read -r -a CATEGORY_ARRAY <<< "${CATEGORIES}"
  ARGS+=(--categories "${CATEGORY_ARRAY[@]}")
fi

if [[ -n "${MAX_PAIRS_PER_CATEGORY}" ]]; then
  ARGS+=(--max-pairs-per-category "${MAX_PAIRS_PER_CATEGORY}")
fi

if [[ -n "${VARIANTS}" ]]; then
  read -r -a VARIANT_ARRAY <<< "${VARIANTS}"
  ARGS+=(--variants "${VARIANT_ARRAY[@]}")
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" || "${TRUST_REMOTE_CODE}" == "true" || "${TRUST_REMOTE_CODE}" == "True" ]]; then
  ARGS+=(--trust-remote-code)
fi

if [[ "${ADD_GENERATION_PROMPT}" == "0" || "${ADD_GENERATION_PROMPT}" == "false" || "${ADD_GENERATION_PROMPT}" == "False" ]]; then
  ARGS+=(--no-add-generation-prompt)
fi

if [[ "${MANUAL_QWEN_CHAT_TEMPLATE}" == "1" || "${MANUAL_QWEN_CHAT_TEMPLATE}" == "true" || "${MANUAL_QWEN_CHAT_TEMPLATE}" == "True" ]]; then
  ARGS+=(--manual-qwen-chat-template)
fi

echo "Running extract_hidden_states_v2.py"
echo "Model: ${MODEL}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Concepts file: ${CONCEPTS_FILE}"
echo "Statement file: ${STATEMENT_FILE}"
echo "Categories: ${CATEGORIES:-all}"
echo "Variants: ${VARIANTS:-default}"
echo "Max statements: ${MAX_STATEMENTS}"
echo "Max pairs per category: ${MAX_PAIRS_PER_CATEGORY:-all}"
echo "Batch size: ${BATCH_SIZE}"
echo "Save format: ${SAVE_FORMAT}"
echo "Dtype: ${DTYPE:-default}"
echo "Device map: ${DEVICE_MAP}"
echo "Trust remote code: ${TRUST_REMOTE_CODE}"
echo "Add generation prompt: ${ADD_GENERATION_PROMPT}"
echo "Manual Qwen chat template: ${MANUAL_QWEN_CHAT_TEMPLATE}"

python3 extract_hidden_states_v2.py "${ARGS[@]}"
