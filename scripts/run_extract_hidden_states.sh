#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-qwen2_5_3b_it}"
CATEGORIES="${CATEGORIES:-}"
MAX_PAIRS_PER_CATEGORY="${MAX_PAIRS_PER_CATEGORY:-}"
MAX_STATEMENTS="${MAX_STATEMENTS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAVE_FORMAT="${SAVE_FORMAT:-npy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hidden_states}"

ARGS=(
  --model "${MODEL}"
  --max-statements "${MAX_STATEMENTS}"
  --batch-size "${BATCH_SIZE}"
  --save-format "${SAVE_FORMAT}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${CATEGORIES}" ]]; then
  # Space-separated category keys, e.g. "linguistic_style ideology"
  read -r -a CATEGORY_ARRAY <<< "${CATEGORIES}"
  ARGS+=(--categories "${CATEGORY_ARRAY[@]}")
fi

if [[ -n "${MAX_PAIRS_PER_CATEGORY}" ]]; then
  ARGS+=(--max-pairs-per-category "${MAX_PAIRS_PER_CATEGORY}")
fi

python extract_hidden_states.py "${ARGS[@]}"
