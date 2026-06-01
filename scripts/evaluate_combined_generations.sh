#!/bin/bash
set -euo pipefail

judge_type=${1:-${JUDGE_TYPE:-"vllm"}}

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
elif [[ "$judge_type" == "vllm" ]]; then
  judge_model="google/gemma-3-4b-it"
else
  judge_model="Qwen/Qwen2.5-7B-Instruct"
fi

judge_slug=$(echo "${judge_type}_${judge_model}" | tr '/:' '__')
generation_model=${GENERATION_MODEL:-Qwen2.5-1.5B-Instruct}
output_dir=${OUTPUT_DIR:-"outputs/${judge_slug}_judgements_combined_full_${generation_model}"}
prompt_file=${PROMPT_FILE:-"contrastive_hidden_states/data/evaluation_prompts/combined_concepts_success_eval_v1.txt"}
input_dirs=${INPUT_DIRS:-"outputs/generations_negpos_full outputs/generations_posneg_full"}

limit=${LIMIT-4}
category_key=${CATEGORY_KEY:-}
pair_slug=${PAIR_SLUG:-}
combined_order=${COMBINED_ORDER:-}
dry_run=${DRY_RUN:-0}
no_resume=${NO_RESUME:-0}
full_run=${FULL_RUN:-1}
judge_max_new_tokens=${JUDGE_MAX_NEW_TOKENS:-512}
judge_batch_size=${JUDGE_BATCH_SIZE:-100}
tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-1}
dtype=${DTYPE:-bfloat16}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.9}
max_model_len=${MAX_MODEL_LEN:-}
trust_remote_code=${TRUST_REMOTE_CODE:-0}

if [[ "$full_run" == "1" || "$full_run" == "true" || "$full_run" == "True" ]]; then
  limit=""
fi

read -r -a input_dir_array <<< "$input_dirs"

args=(
  --judge_type "$judge_type"
  --judge_model "$judge_model"
  --prompt_file "$prompt_file"
  --output_dir "$output_dir"
  --input_dirs "${input_dir_array[@]}"
  --judge_max_new_tokens "$judge_max_new_tokens"
  --judge_batch_size "$judge_batch_size"
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

if [[ -n "$generation_model" ]]; then
  args+=(--generation_model "$generation_model")
fi

if [[ -n "$combined_order" ]]; then
  args+=(--combined_order "$combined_order")
fi

if [[ "$dry_run" == "1" || "$dry_run" == "true" || "$dry_run" == "True" ]]; then
  args+=(--dry_run)
fi

if [[ "$no_resume" == "1" || "$no_resume" == "true" || "$no_resume" == "True" ]]; then
  args+=(--no_resume)
fi

if [[ "$judge_type" == "vllm" ]]; then
  args+=(
    --tensor_parallel_size "$tensor_parallel_size"
    --dtype "$dtype"
    --gpu_memory_utilization "$gpu_memory_utilization"
  )

  if [[ -n "$max_model_len" ]]; then
    args+=(--max_model_len "$max_model_len")
  fi

  if [[ "$trust_remote_code" == "1" || "$trust_remote_code" == "true" || "$trust_remote_code" == "True" ]]; then
    args+=(--trust_remote_code)
  fi
fi

echo "Running evaluate_combined_generations.py"
echo "Judge type: $judge_type"
echo "Judge model: $judge_model"
echo "Input dirs: $input_dirs"
echo "Prompt file: $prompt_file"
echo "Output dir: $output_dir"
echo "Limit: ${limit:-all}"
echo "Category key: ${category_key:-all}"
echo "Pair slug: ${pair_slug:-all}"
echo "Generation model: ${generation_model:-all}"
echo "Combined order: ${combined_order:-all}"
echo "Judge max new tokens: $judge_max_new_tokens"
echo "Judge batch size: $judge_batch_size"
if [[ "$judge_type" == "vllm" ]]; then
  echo "Tensor parallel size: $tensor_parallel_size"
  echo "Dtype: $dtype"
  echo "GPU memory utilization: $gpu_memory_utilization"
  echo "Max model len: ${max_model_len:-default}"
  echo "Trust remote code: $trust_remote_code"
fi

python3 evaluate_combined_generations.py "${args[@]}"
