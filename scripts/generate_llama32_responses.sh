#!/bin/bash
set -euo pipefail

model=${1:-${MODEL:-"meta-llama/Llama-3.2-3B-Instruct"}}
tensor_parallel_size=${2:-${TENSOR_PARALLEL_SIZE:-1}}

variants=${VARIANTS:-"negpos"}
#variants=${VARIANTS:-"posneg"}
categories=${CATEGORIES:-"ideology linguistic_style semantic_framing"}
max_pairs_per_category=${MAX_PAIRS_PER_CATEGORY:-}
max_statements=${MAX_STATEMENTS:-100}
batch_size=${BATCH_SIZE:-32}
output_dir=${OUTPUT_DIR:-"outputs/generations_llama32_negpos_full"}
#output_dir=${OUTPUT_DIR:-"outputs/generations_llama32_posneg_full"}

torch_dtype=${TORCH_DTYPE:-bfloat16}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.9}
max_model_len=${MAX_MODEL_LEN:-}
trust_remote_code=${TRUST_REMOTE_CODE:-0}

do_sample=${DO_SAMPLE:-1}
temperature=${TEMPERATURE:-0.6}
top_p=${TOP_P:-0.9}
top_k=${TOP_K:-20}
repetition_penalty=${REPETITION_PENALTY:-1.05}
max_new_tokens=${MAX_NEW_TOKENS:-1024}

read -r -a variant_array <<< "$variants"

args=(
  --model "$model"
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
  --variants "${variant_array[@]}"
)

if [[ -n "$categories" ]]; then
  read -r -a category_array <<< "$categories"
  args+=(--categories "${category_array[@]}")
fi

if [[ -n "$max_pairs_per_category" ]]; then
  args+=(--max-pairs-per-category "$max_pairs_per_category")
fi

if [[ -n "$max_model_len" ]]; then
  args+=(--max-model-len "$max_model_len")
fi

if [[ "$trust_remote_code" == "1" || "$trust_remote_code" == "true" || "$trust_remote_code" == "True" ]]; then
  args+=(--trust-remote-code)
fi

if [[ "$do_sample" == "1" || "$do_sample" == "true" || "$do_sample" == "True" ]]; then
  args+=(--do-sample)
fi

echo "Running generate_llama32_responses.py"
echo "Model: $model"
echo "Output dir: $output_dir"
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

python3 generate_llama32_responses.py "${args[@]}"
