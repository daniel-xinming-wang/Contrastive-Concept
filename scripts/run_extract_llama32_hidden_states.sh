#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
CATEGORIES="${CATEGORIES:-}"
MAX_PAIRS_PER_CATEGORY="${MAX_PAIRS_PER_CATEGORY:-}"
MAX_STATEMENTS="${MAX_STATEMENTS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAVE_FORMAT="${SAVE_FORMAT:-npy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hidden_states_llama32}"
VARIANTS="${VARIANTS:-}"

ARGS=(
  --model "${MODEL}"
  --max-statements "${MAX_STATEMENTS}"
  --batch-size "${BATCH_SIZE}"
  --save-format "${SAVE_FORMAT}"
  --output-dir "${OUTPUT_DIR}"
)

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

echo "Running extract_hidden_states.py for Llama 3.2"
echo "Model: ${MODEL}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Categories: ${CATEGORIES:-all}"
echo "Variants: ${VARIANTS:-default}"
echo "Max statements: ${MAX_STATEMENTS}"
echo "Max pairs per category: ${MAX_PAIRS_PER_CATEGORY:-all}"
echo "Batch size: ${BATCH_SIZE}"

python3 extract_hidden_states.py "${ARGS[@]}"
