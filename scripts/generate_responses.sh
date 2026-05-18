#!/bin/bash
set -euo pipefail

model=${1:-"Qwen/Qwen2.5-3B-Instruct"}
tensor_parallel_size=${2:-1}
steering_strength=${3:-0.5}
#variants=${4:-"negative base positive"}
variants=${4:-"base"}
categories=${5:-"ideology"}
steer=${STEER:-1}

steering_direction=${STEERING_DIRECTION:-"base_to_pos"}
steering_vector_path=${STEERING_VECTOR_PATH:-"contrastive_hidden_states/data/steering_vectors/{model_name}/{category_key}/{pair_slug}/${steering_direction}.npy"}
steering_layers=${STEERING_LAYERS:-"all"}

max_pairs_per_category=${MAX_PAIRS_PER_CATEGORY:-20}
max_statements=${MAX_STATEMENTS:-100}
batch_size=${BATCH_SIZE:-8}
output_dir=${OUTPUT_DIR:-"outputs/generations_vllm_${steering_strength}_${steering_direction}_layers_${steering_layers}_test_dosample0_ideology"}

gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.9}

if [[ "$steer" == "0" || "$steer" == "false" || "$steer" == "False" || "$steer" == "no" || "$steer" == "No" ]]; then
  steering_vector_path="Empty"
fi

do_sample=${DO_SAMPLE:-0}
strip_default_system_prompt=${STRIP_DEFAULT_SYSTEM_PROMPT:-0}
temperature=${TEMPERATURE:-0.7}
top_p=${TOP_P:-0.8}
top_k=${TOP_K:-20}
repetition_penalty=${REPETITION_PENALTY:-1.05}
max_new_tokens=${MAX_NEW_TOKENS:-1024}

args=(
  --model "$model"
  --output-dir "$output_dir"
  --batch-size "$batch_size"
  --max-statements "$max_statements"
  --tensor-parallel-size "$tensor_parallel_size"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --steering-vector-path "$steering_vector_path"
  --steering-strength "$steering_strength"
  --steering-layers "$steering_layers"
  --max-new-tokens "$max_new_tokens"
  --temperature "$temperature"
  --top-p "$top_p"
  --top-k "$top_k"
  --repetition-penalty "$repetition_penalty"
)

if [[ -n "$max_pairs_per_category" ]]; then
  args+=(--max-pairs-per-category "$max_pairs_per_category")
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

if [[ "$strip_default_system_prompt" == "1" || "$strip_default_system_prompt" == "true" || "$strip_default_system_prompt" == "True" ]]; then
  args+=(--strip-default-system-prompt)
fi

echo "Running generate_responses.py with following parameters:"
echo "Model: $model"
echo "Output dir: $output_dir"
echo "Tensor parallel size: $tensor_parallel_size"
echo "Steer: $steer"
echo "Steering vector path/template: $steering_vector_path"
echo "Steering strength: $steering_strength"
echo "Steering layers: $steering_layers"
echo "Strip default system prompt: $strip_default_system_prompt"
echo "Variants: $variants"
echo "Categories: ${categories:-all}"

python generate_responses.py "${args[@]}"
