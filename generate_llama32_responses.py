from __future__ import annotations

import argparse
import gc
from pathlib import Path

from contrastive_hidden_states.concepts import parse_contrastive_concepts
from contrastive_hidden_states.prompts import (
    VALID_VARIANTS,
    VARIANT_ORDER,
    build_pair_examples,
    load_statement_groups,
)


DEFAULT_STATEMENT_FILES = [
    "contrastive_hidden_states/data/statements_300/class_0.txt",
    "contrastive_hidden_states/data/statements_300/class_1.txt",
    "contrastive_hidden_states/data/statements_300/class_2.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate no-steering contrastive responses with Llama 3.2 via vLLM."
    )
    parser.add_argument("--concepts-file", default="contrastive_concepts.txt")
    parser.add_argument("--statement-files", nargs="+", default=DEFAULT_STATEMENT_FILES)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--output-dir", default="outputs/llama32_generations")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-statements", type=int, default=100)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--max-pairs-per-category", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VALID_VARIANTS,
        default=list(VARIANT_ORDER),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--add-generation-prompt", action="store_true", default=True)
    parser.add_argument(
        "--no-add-generation-prompt",
        action="store_false",
        dest="add_generation_prompt",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    return parser.parse_args()


def create_vllm(args: argparse.Namespace):
    from vllm import LLM

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.torch_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    return LLM(**llm_kwargs)


def cleanup_vllm_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer

    from contrastive_hidden_states.generation import generate_responses, save_generation_bundle

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
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        legacy=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    llm = create_vllm(args)
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
        "model_arg": args.model,
        "resolved_model_name": args.model,
        "backend": "vllm",
        "steering_enabled": False,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.torch_dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
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
            pair_output_dir = output_root / Path(args.model).name / category_key / pair.slug
            save_generation_bundle(
                output_dir=pair_output_dir,
                records=records,
                run_config=run_config,
            )

    llm = None
    cleanup_vllm_memory()


if __name__ == "__main__":
    main()
