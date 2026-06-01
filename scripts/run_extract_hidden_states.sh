#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
CATEGORIES="${CATEGORIES:-}"
MAX_PAIRS_PER_CATEGORY="${MAX_PAIRS_PER_CATEGORY:-}"
MAX_STATEMENTS="${MAX_STATEMENTS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAVE_FORMAT="${SAVE_FORMAT:-npy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hidden_states}"
STRIP_DEFAULT_SYSTEM_PROMPT="${STRIP_DEFAULT_SYSTEM_PROMPT:-0}"
VARIANTS="${VARIANTS:-}"

ARGS=(
  --model "${MODEL}"
  --max-statements "${MAX_STATEMENTS}"
  --batch-size "${BATCH_SIZE}"
  --save-format "${SAVE_FORMAT}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${STRIP_DEFAULT_SYSTEM_PROMPT}" == "1" || "${STRIP_DEFAULT_SYSTEM_PROMPT}" == "true" || "${STRIP_DEFAULT_SYSTEM_PROMPT}" == "True" ]]; then
  ARGS+=(--strip-default-system-prompt)
fi

if [[ -n "${CATEGORIES}" ]]; then
  # Space-separated category keys, e.g. "linguistic_style ideology"
  read -r -a CATEGORY_ARRAY <<< "${CATEGORIES}"
  ARGS+=(--categories "${CATEGORY_ARRAY[@]}")
fi

if [[ -n "${MAX_PAIRS_PER_CATEGORY}" ]]; then
  ARGS+=(--max-pairs-per-category "${MAX_PAIRS_PER_CATEGORY}")
fi

if [[ -n "${VARIANTS}" ]]; then
  # Space-separated prompt variants, e.g. "negpos posneg"
  read -r -a VARIANT_ARRAY <<< "${VARIANTS}"
  ARGS+=(--variants "${VARIANT_ARRAY[@]}")
fi

python extract_hidden_states.py "${ARGS[@]}"
