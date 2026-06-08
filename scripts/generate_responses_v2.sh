#!/bin/bash
set -euo pipefail

#model=${1:-${MODEL:-"meta-llama/Llama-3.2-3B-Instruct"}}
model=${1:-${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}}
tensor_parallel_size=${2:-${TENSOR_PARALLEL_SIZE:-1}}
variants=${3:-${VARIANTS:-"system_positive_only system_negative_only syspos_userneg_conflict sysneg_userpos_conflict"}}
categories=${4:-${CATEGORIES:-""}}

concepts_file=${CONCEPTS_FILE:-"contrastive_concepts.txt"}
statement_file=${STATEMENT_FILE:-"contrastive_hidden_states/data/statements_300/class_1.txt"}
max_pairs_per_category=${MAX_PAIRS_PER_CATEGORY:-}
max_statements=${MAX_STATEMENTS:-100}
batch_size=${BATCH_SIZE:-100}
output_dir=${OUTPUT_DIR:-"outputs_v2_new/generations_system_user_conflict_v2_${model##*/}"}

torch_dtype=${DTYPE:-${TORCH_DTYPE:-bfloat16}}
max_model_len=${MAX_MODEL_LEN:-}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.9}
trust_remote_code=${TRUST_REMOTE_CODE:-0}

do_sample=${DO_SAMPLE:-0}
temperature=${TEMPERATURE:-0.7}
top_p=${TOP_P:-0.8}
top_k=${TOP_K:-20}
repetition_penalty=${REPETITION_PENALTY:-1.05}
max_new_tokens=${MAX_NEW_TOKENS:-1024}
add_generation_prompt=${ADD_GENERATION_PROMPT:-1}
manual_qwen_chat_template=${MANUAL_QWEN_CHAT_TEMPLATE:-0}

args=(
  --model "$model"
  --concepts-file "$concepts_file"
  --statement-file "$statement_file"
  --output-dir "$output_dir"
  --batch-size "$batch_size"
  --max-statements "$max_statements"
  --tensor-parallel-size "$tensor_parallel_size"
  --torch-dtype "$torch_dtype"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --max-new-tokens "$max_new_tokens"
  --temperature "$temperature"
  --top-p "$top_p"
  --top-k "$top_k"
  --repetition-penalty "$repetition_penalty"
)

if [[ -n "$max_pairs_per_category" ]]; then
  args+=(--max-pairs-per-category "$max_pairs_per_category")
fi

if [[ -n "$max_model_len" ]]; then
  args+=(--max-model-len "$max_model_len")
fi

if [[ -n "$categories" ]]; then
  read -r -a category_array <<< "$categories"
  args+=(--categories "${category_array[@]}")
fi

if [[ -n "$variants" ]]; then
  read -r -a variant_array <<< "$variants"
  args+=(--variants "${variant_array[@]}")
fi

if [[ "$do_sample" == "1" || "$do_sample" == "true" || "$do_sample" == "True" ]]; then
  args+=(--do-sample)
fi

if [[ "$trust_remote_code" == "1" || "$trust_remote_code" == "true" || "$trust_remote_code" == "True" ]]; then
  args+=(--trust-remote-code)
fi

if [[ "$add_generation_prompt" == "0" || "$add_generation_prompt" == "false" || "$add_generation_prompt" == "False" ]]; then
  args+=(--no-add-generation-prompt)
fi

if [[ "$manual_qwen_chat_template" == "1" || "$manual_qwen_chat_template" == "true" || "$manual_qwen_chat_template" == "True" ]]; then
  args+=(--manual-qwen-chat-template)
fi

echo "Running generate_responses_v2.py with following parameters:"
echo "Model: $model"
echo "Output dir: $output_dir"
echo "Concepts file: $concepts_file"
echo "Statement file: $statement_file"
echo "Tensor parallel size: $tensor_parallel_size"
echo "Dtype: $torch_dtype"
echo "GPU memory utilization: $gpu_memory_utilization"
echo "Max model len: ${max_model_len:-default}"
echo "Batch size: $batch_size"
echo "Max statements: $max_statements"
echo "Max pairs per category: ${max_pairs_per_category:-all}"
echo "Variants: $variants"
echo "Categories: ${categories:-all}"
echo "Do sample: $do_sample"
echo "Max new tokens: $max_new_tokens"
echo "Trust remote code: $trust_remote_code"
echo "Manual Qwen chat template: $manual_qwen_chat_template"

python generate_responses_v2.py "${args[@]}"
