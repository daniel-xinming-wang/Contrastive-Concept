from __future__ import annotations

import argparse
from pathlib import Path

from contrastive_hidden_states.concepts import parse_contrastive_concepts
from contrastive_hidden_states.generation import generate_responses, save_generation_bundle
from contrastive_hidden_states.models import load_model_and_tokenizer
from contrastive_hidden_states.prompts import build_pair_examples, load_statement_groups


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
        "--torch-dtype",
        default=None,
        help="Optional dtype override: float16, bfloat16, or float32.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Passed through to from_pretrained.",
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
        default=128,
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
        default=1.0,
        help="Sampling temperature. Used only when --do-sample is set.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Sampling top-p. Used only when --do-sample is set.",
    )
    return parser.parse_args()


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

    model, tokenizer, resolved_name = load_model_and_tokenizer(
        model_name_or_path=args.model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

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
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
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
            )
            records = generate_responses(
                examples=examples,
                model=model,
                tokenizer=tokenizer,
                forward_batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            pair_output_dir = output_root / Path(resolved_name).name / category_key / pair.slug
            save_generation_bundle(
                output_dir=pair_output_dir,
                records=records,
                run_config=run_config,
            )


if __name__ == "__main__":
    main()
