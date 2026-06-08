from __future__ import annotations

import argparse
from pathlib import Path

from contrastive_hidden_states.concepts import parse_contrastive_concepts
from contrastive_hidden_states.hidden_states import get_hidden_states, save_pair_bundle
from contrastive_hidden_states.models import default_hidden_layers, load_model_and_tokenizer
from generate_responses_v2 import (
    CONFLICT_VARIANTS,
    build_conflict_examples,
    load_statements,
)


DEFAULT_STATEMENT_FILE = "contrastive_hidden_states/data/statements_300/class_1.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract hidden states for system-vs-user conflict prompts. "
            "This matches the prompt construction in generate_responses_v2.py."
        )
    )
    parser.add_argument("--concepts-file", default="contrastive_concepts.txt")
    parser.add_argument("--statement-file", default=DEFAULT_STATEMENT_FILE)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--output-dir", default="outputs/hidden_states_system_user_conflict_v2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-statements", type=int, default=100)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--max-pairs-per-category", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=CONFLICT_VARIANTS,
        default=list(CONFLICT_VARIANTS),
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Optional dtype override: float16, bfloat16, or float32.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--add-generation-prompt", action="store_true", default=True)
    parser.add_argument(
        "--no-add-generation-prompt",
        action="store_false",
        dest="add_generation_prompt",
    )
    parser.add_argument(
        "--manual-qwen-chat-template",
        action="store_true",
        help=(
            "Format prompts with explicit Qwen <|im_start|> system/user tags instead "
            "of tokenizer.apply_chat_template. This avoids Qwen's default system prompt."
        ),
    )
    parser.add_argument(
        "--save-format",
        choices=("npy", "pt"),
        default="npy",
        help="How to save each variant's hidden states.",
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

    statements = load_statements(args.statement_file, max_statements=args.max_statements)

    model, tokenizer, resolved_name = load_model_and_tokenizer(
        model_name_or_path=args.model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    hidden_layers = default_hidden_layers(model)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "script": "extract_hidden_states_v2.py",
        "prompt_design": "system_user_conflict_v2",
        "concepts_file": str(Path(args.concepts_file).resolve()),
        "statement_file": str(Path(args.statement_file).resolve()),
        "num_statements": len(statements),
        "model_arg": args.model,
        "resolved_model_name": resolved_name,
        "batch_size": args.batch_size,
        "hidden_layers": hidden_layers,
        "add_generation_prompt": args.add_generation_prompt,
        "manual_qwen_chat_template": args.manual_qwen_chat_template,
        "save_format": args.save_format,
        "variants": args.variants,
        "variant_definitions": {
            "system_positive_only": "system=positive, user has no concept instruction",
            "system_negative_only": "system=negative, user has no concept instruction",
            "syspos_userneg_conflict": "system=positive, user says not positive and requests negative",
            "sysneg_userpos_conflict": "system=negative, user says not negative and requests positive",
        },
    }

    for category_key, pairs in concept_groups.items():
        if args.max_pairs_per_category is not None:
            pairs = pairs[: args.max_pairs_per_category]

        for pair in pairs:
            print(f"Extracting system/user conflict hidden states for {pair.slug}")
            examples = build_conflict_examples(
                pair=pair,
                statements=statements,
                tokenizer=tokenizer,
                variants=args.variants,
                add_generation_prompt=args.add_generation_prompt,
                manual_qwen_chat_template=args.manual_qwen_chat_template,
            )
            prompts = [example.prompt for example in examples]
            hidden_states = get_hidden_states(
                prompts=prompts,
                model=model,
                tokenizer=tokenizer,
                hidden_layers=hidden_layers,
                forward_batch_size=args.batch_size,
            )

            pair_output_dir = output_root / Path(resolved_name).name / category_key / pair.slug
            save_pair_bundle(
                output_dir=pair_output_dir,
                examples=examples,
                hidden_states=hidden_states,
                run_config=run_config,
                save_format=args.save_format,
            )


if __name__ == "__main__":
    main()
