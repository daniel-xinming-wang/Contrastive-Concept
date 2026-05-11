from __future__ import annotations

import argparse
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, ModelRegistry

from contrastive_hidden_states.concepts import parse_contrastive_concepts
from contrastive_hidden_states.generation import generate_responses, save_generation_bundle
from contrastive_hidden_states.models import resolve_model_name
from contrastive_hidden_states.prompts import (
    VARIANT_ORDER,
    build_pair_examples,
    load_statement_groups,
)
from steer_qwen2_vllm_0111 import SteerQwen2ForCausalLM


DEFAULT_STATEMENT_FILES = [
    "contrastive_hidden_states/data/statements_300/class_0.txt",
    "contrastive_hidden_states/data/statements_300/class_1.txt",
    "contrastive_hidden_states/data/statements_300/class_2.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build contrastive prompts and save generated responses for each concept pair."
    )
    parser.add_argument(
        "--concepts-file",
        default="contrastive_concepts.txt",
        help="Path to the contrastive concept list.",
    )
    parser.add_argument(
        "--statement-files",
        nargs="+",
        default=DEFAULT_STATEMENT_FILES,
        help="Three statement files mapped to negative/base/positive in that order.",
    )
    parser.add_argument(
        "--model",
        default="qwen2_5_7b_it",
        help="Model alias or Hugging Face model id.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/generations",
        help="Directory where generated responses will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Generation batch size.",
    )
    parser.add_argument(
        "--max-statements",
        type=int,
        default=10,
        help="Optional limit on the number of statements loaded from each class file.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional subset of category keys to run, e.g. linguistic_style ideology.",
    )
    parser.add_argument(
        "--max-pairs-per-category",
        type=int,
        default=5,
        help="Optional limit on how many pairs to process per category.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_ORDER,
        default=list(VARIANT_ORDER),
        help="Prompt variants to generate: negative, base, positive.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        help="vLLM dtype: auto, float16, bfloat16, float32, half, or float.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Deprecated; ignored because generation now uses vLLM.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional vLLM max_model_len override.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization.",
    )
    parser.add_argument(
        "--steering-vector-path",
        default="Empty",
        help="Path to steering vector .npy. Use Empty to disable steering.",
    )
    parser.add_argument(
        "--steering-strength",
        type=float,
        default=0.0,
        help="Steering strength applied to every layer when steering is enabled.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when loading the model/tokenizer.",
    )
    parser.add_argument(
        "--add-generation-prompt",
        action="store_true",
        default=True,
        help="Whether to include the assistant generation tag in chat formatting.",
    )
    parser.add_argument(
        "--no-add-generation-prompt",
        action="store_false",
        dest="add_generation_prompt",
        help="Disable the assistant generation tag in chat formatting.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling instead of greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Used only when --do-sample is set.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Sampling top-p. Used only when --do-sample is set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Sampling top-k. Used only when --do-sample is set.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="vLLM repetition penalty.",
    )
    return parser.parse_args()


def configure_steering(
    resolved_name: str,
    steering_vector_path: str,
    steering_strength: float,
    trust_remote_code: bool,
) -> bool:
    if steering_vector_path == "Empty":
        os.environ.pop("steering_vector_path", None)
        os.environ.pop("steering_strength_list", None)
        return False

    ModelRegistry.register_model("Qwen2ForCausalLM", SteerQwen2ForCausalLM)

    config = AutoConfig.from_pretrained(
        resolved_name,
        trust_remote_code=trust_remote_code,
    )
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    steering_strength_list = [steering_strength] * text_config.num_hidden_layers

    os.environ["steering_vector_path"] = steering_vector_path
    os.environ["steering_strength_list"] = ",".join(map(str, steering_strength_list))
    print("Finish registering model SteerQwen2ForCausalLM")
    print(f"Set steering_vector_path to: {steering_vector_path}")
    print(f"Set steering_strength_list to: {steering_strength_list}")
    return True


def load_vllm_and_tokenizer(args: argparse.Namespace) -> tuple[LLM, object, str, bool]:
    resolved_name = resolve_model_name(args.model)
    steering_enabled = configure_steering(
        resolved_name=resolved_name,
        steering_vector_path=args.steering_vector_path,
        steering_strength=args.steering_strength,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_name,
        padding_side="left",
        legacy=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    llm_kwargs = {
        "model": resolved_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.torch_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    return llm, tokenizer, resolved_name, steering_enabled


def main() -> None:
    args = parse_args()

    concept_groups = parse_contrastive_concepts(args.concepts_file)
    if args.categories:
        requested = set(args.categories)
        concept_groups = {
            key: pairs for key, pairs in concept_groups.items() if key in requested
        }

    statement_groups = load_statement_groups(
        args.statement_files,
        max_statements_per_group=args.max_statements,
    )

    llm, tokenizer, resolved_name, steering_enabled = load_vllm_and_tokenizer(args)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "concepts_file": str(Path(args.concepts_file).resolve()),
        "statement_files": [str(Path(path).resolve()) for path in args.statement_files],
        "statement_mapping": {
            "negative": Path(args.statement_files[0]).name,
            "base": Path(args.statement_files[1]).name,
            "positive": Path(args.statement_files[2]).name,
        },
        "num_statements_per_variant": {
            variant: len(statements) for variant, statements in statement_groups.items()
        },
        "model_arg": args.model,
        "resolved_model_name": resolved_name,
        "backend": "vllm",
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.torch_dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "steering_enabled": steering_enabled,
        "steering_vector_path": args.steering_vector_path if steering_enabled else None,
        "steering_strength": args.steering_strength if steering_enabled else None,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "variants": args.variants,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "top_k": args.top_k if args.do_sample else None,
        "repetition_penalty": args.repetition_penalty,
        "add_generation_prompt": args.add_generation_prompt,
    }

    for category_key, pairs in concept_groups.items():
        if args.max_pairs_per_category is not None:
            pairs = pairs[: args.max_pairs_per_category]

        for pair in pairs:
            print(f"Generating {pair.slug}")
            examples = build_pair_examples(
                pair=pair,
                statement_groups=statement_groups,
                tokenizer=tokenizer,
                add_generation_prompt=args.add_generation_prompt,
                variants=args.variants,
            )
            records = generate_responses(
                examples=examples,
                llm=llm,
                forward_batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )

            pair_output_dir = output_root / Path(resolved_name).name / category_key / pair.slug
            save_generation_bundle(
                output_dir=pair_output_dir,
                records=records,
                run_config=run_config,
            )


if __name__ == "__main__":
    main()
