#!/bin/bash
set -euo pipefail

judge_type=${1:-${JUDGE_TYPE:-"gemma3"}}

if [[ $# -ge 2 ]]; then
  judge_model="$2"
elif [[ -n "${JUDGE_MODEL:-}" ]]; then
  judge_model="$JUDGE_MODEL"
elif [[ "$judge_type" == "openai" ]]; then
  judge_model="gpt-4o-2024-11-20"
elif [[ "$judge_type" == "gemma3" ]]; then
  judge_model="google/gemma-3-4b-it"
elif [[ "$judge_type" == "llama" ]]; then
  judge_model="meta-llama/Llama-3.1-8B-Instruct"
else
  judge_model="Qwen/Qwen2.5-7B-Instruct"
fi

judge_slug=$(echo "${judge_type}_${judge_model}" | tr '/:' '__')
output_dir=${OUTPUT_DIR:-"outputs/${judge_slug}_judgements_vllm_0.3_dosample0"}
prompt_dir=${PROMPT_DIR:-"contrastive_hidden_states/data/evaluation_prompts"}
input_dirs=${INPUT_DIRS:-"outputs/generations_vllm_0.3_base_to_neg_test_dosample0 outputs/generations_vllm_0.3_base_to_pos_test_dosample0"}

limit=${LIMIT-4}
category_key=${CATEGORY_KEY:-}
pair_slug=${PAIR_SLUG:-}
dry_run=${DRY_RUN:-0}
no_resume=${NO_RESUME:-0}
full_run=${FULL_RUN:-0}

if [[ "$full_run" == "1" || "$full_run" == "true" || "$full_run" == "True" ]]; then
  limit=""
fi

read -r -a input_dir_array <<< "$input_dirs"

args=(
  --judge_type "$judge_type"
  --judge_model "$judge_model"
  --prompt_dir "$prompt_dir"
  --output_dir "$output_dir"
  --input_dirs "${input_dir_array[@]}"
)

if [[ -n "$limit" ]]; then
  args+=(--limit "$limit")
fi

if [[ -n "$category_key" ]]; then
  args+=(--category_key "$category_key")
fi

if [[ -n "$pair_slug" ]]; then
  args+=(--pair_slug "$pair_slug")
fi

if [[ "$dry_run" == "1" || "$dry_run" == "true" || "$dry_run" == "True" ]]; then
  args+=(--dry_run)
fi

if [[ "$no_resume" == "1" || "$no_resume" == "true" || "$no_resume" == "True" ]]; then
  args+=(--no_resume)
fi

echo "Running evaluate_steered_generations.py"
echo "Judge type: $judge_type"
echo "Judge model: $judge_model"
echo "Input dirs: $input_dirs"
echo "Prompt dir: $prompt_dir"
echo "Output dir: $output_dir"
echo "Limit: ${limit:-all}"
echo "Category key: ${category_key:-all}"
echo "Pair slug: ${pair_slug:-all}"

python3 evaluate_steered_generations.py "${args[@]}"
