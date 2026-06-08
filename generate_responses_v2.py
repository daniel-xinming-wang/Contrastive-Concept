from __future__ import annotations

import argparse
import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from contrastive_hidden_states.concepts import ContrastivePair, parse_contrastive_concepts
from contrastive_hidden_states.prompts import CATEGORY_INSTRUCTIONS


DEFAULT_STATEMENT_FILE = "contrastive_hidden_states/data/statements_300/class_1.txt"

CONFLICT_VARIANTS = (
    "system_positive_only",
    "system_negative_only",
    "syspos_userneg_conflict",
    "sysneg_userpos_conflict",
)


@dataclass(frozen=True)
class ConflictPromptExample:
    category_name: str
    category_key: str
    pair_slug: str
    positive_concept: str
    negative_concept: str
    statement_index: int
    statement_source: str
    statement: str
    variant: str
    system_concept: str
    user_concept: str
    conflict: bool
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate system-vs-user conflict responses for contrastive concept pairs. "
            "Unlike generate_responses.py, this script puts one concept in the system "
            "message and one concept in the user message."
        )
    )
    parser.add_argument("--concepts-file", default="contrastive_concepts.txt")
    parser.add_argument("--statement-file", default=DEFAULT_STATEMENT_FILE)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--output-dir", default="outputs/generations_system_user_conflict_v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-statements", type=int, default=100)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--max-pairs-per-category", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=CONFLICT_VARIANTS,
        default=list(CONFLICT_VARIANTS),
        help=(
            "system_positive_only/system_negative_only are system-only baselines. "
            "syspos_userneg and sysneg_userpos are true system-vs-user conflict conditions."
        ),
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
    parser.add_argument(
        "--manual-qwen-chat-template",
        action="store_true",
        help=(
            "Format prompts with explicit Qwen <|im_start|> system/user tags instead "
            "of tokenizer.apply_chat_template. This avoids Qwen's default system prompt."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    return parser.parse_args()


def load_statements(path: str | Path, max_statements: int | None) -> list[tuple[str, str]]:
    statement_path = Path(path)
    statements: list[tuple[str, str]] = []
    for line in statement_path.read_text(encoding="utf-8").splitlines():
        statement = line.strip()
        if statement:
            statements.append((statement_path.name, statement))
    if max_statements is not None:
        statements = statements[:max_statements]
    return statements


def concept_instruction(category_key: str, concept: str) -> str:
    template = CATEGORY_INSTRUCTIONS.get(
        category_key,
        "Adopt the following concept: {concept}.",
    )
    return template.format(concept=concept)


def conflict_instruction(
    category_key: str,
    target_concept: str,
    opposing_concept: str | None = None,
) -> str:
    instruction = concept_instruction(category_key, target_concept)
    if opposing_concept:
        instruction += f" Do not use or rely on the contrasting concept: {opposing_concept}."
    return instruction


def build_system_message(
    category_key: str,
    concept: str,
    forbidden_concept: str | None = None,
) -> str:
    return conflict_instruction(
        category_key=category_key,
        target_concept=concept,
        opposing_concept=forbidden_concept,
    )


def build_user_message(
    category_key: str,
    statement: str,
    system_concept: str,
    user_concept: str | None,
    conflict: bool,
) -> str:
    if conflict:
        if user_concept is None:
            raise ValueError("Conflict prompts require a user_concept.")
        instruction = conflict_instruction(
            category_key=category_key,
            target_concept=user_concept,
            opposing_concept=system_concept,
        )
    else:
        instruction = "Answer the question directly."

    return (
        f"{instruction}\n\n"
        "What are your thoughts on the following statement?\n"
        f"Statement: {statement}\n"
        "Response:"
    )


def variant_concepts(pair: ContrastivePair, variant: str) -> tuple[str, str | None, bool]:
    if variant == "system_positive_only":
        return pair.positive, None, False
    if variant == "system_negative_only":
        return pair.negative, None, False
    if variant == "syspos_userneg_conflict":
        return pair.positive, pair.negative, True
    if variant == "sysneg_userpos_conflict":
        return pair.negative, pair.positive, True
    raise ValueError(f"Unknown conflict variant: {variant}")


def format_chat_prompt(
    tokenizer: Any,
    system_message: str,
    user_message: str,
    add_generation_prompt: bool,
    manual_qwen_chat_template: bool,
) -> str:
    if manual_qwen_chat_template:
        prompt = (
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>"
        )
        if add_generation_prompt:
            prompt += "\n<|im_start|>assistant\n"
        return prompt.strip()

    chat = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        ).strip()
    return f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"


def build_conflict_examples(
    pair: ContrastivePair,
    statements: list[tuple[str, str]],
    tokenizer: Any,
    variants: list[str],
    add_generation_prompt: bool,
    manual_qwen_chat_template: bool,
) -> list[ConflictPromptExample]:
    examples: list[ConflictPromptExample] = []
    for variant in variants:
        system_concept, user_concept, conflict = variant_concepts(pair, variant)
        for statement_index, (statement_source, statement) in enumerate(statements):
            system_message = build_system_message(
                category_key=pair.category_key,
                concept=system_concept,
                forbidden_concept=user_concept if conflict else None,
            )
            user_message = build_user_message(
                category_key=pair.category_key,
                statement=statement,
                system_concept=system_concept,
                user_concept=user_concept,
                conflict=conflict,
            )
            prompt = format_chat_prompt(
                tokenizer=tokenizer,
                system_message=system_message,
                user_message=user_message,
                add_generation_prompt=add_generation_prompt,
                manual_qwen_chat_template=manual_qwen_chat_template,
            )
            examples.append(
                ConflictPromptExample(
                    category_name=pair.category_name,
                    category_key=pair.category_key,
                    pair_slug=pair.slug,
                    positive_concept=pair.positive,
                    negative_concept=pair.negative,
                    statement_index=statement_index,
                    statement_source=statement_source,
                    statement=statement,
                    variant=variant,
                    system_concept=system_concept,
                    user_concept=user_concept or "",
                    conflict=conflict,
                    prompt=prompt,
                )
            )
    return examples


def create_vllm(args: argparse.Namespace) -> Any:
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


def generate_conflict_responses(
    examples: list[ConflictPromptExample],
    llm: Any,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> list[dict[str, Any]]:
    from tqdm import tqdm
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if do_sample else 0,
        repetition_penalty=repetition_penalty,
    )

    records: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(examples), batch_size), desc="Generating conflict responses"):
        batch_examples = examples[start : start + batch_size]
        prompts = [example.prompt for example in batch_examples]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for example, prompt, output in zip(batch_examples, prompts, outputs):
            generated_text = output.outputs[0].text.strip()
            records.append(
                {
                    **asdict(example),
                    "full_text": f"{prompt}{generated_text}".strip(),
                    "generated_text": generated_text,
                }
            )
    return records


def cleanup_vllm_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()

    concept_groups = parse_contrastive_concepts(args.concepts_file)
    if args.categories:
        requested = set(args.categories)
        concept_groups = {
            key: pairs for key, pairs in concept_groups.items() if key in requested
        }

    statements = load_statements(args.statement_file, max_statements=args.max_statements)
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
        "script": "generate_responses_v2.py",
        "prompt_design": "system_user_conflict_v2",
        "concepts_file": str(Path(args.concepts_file).resolve()),
        "statement_file": str(Path(args.statement_file).resolve()),
        "num_statements": len(statements),
        "model_arg": args.model,
        "resolved_model_name": args.model,
        "backend": "vllm",
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.torch_dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "variants": args.variants,
        "variant_definitions": {
            "system_positive_only": "system=positive, user has no concept instruction",
            "system_negative_only": "system=negative, user has no concept instruction",
            "syspos_userneg_conflict": "system=positive and rejects negative; user=negative and rejects positive",
            "sysneg_userpos_conflict": "system=negative and rejects positive; user=positive and rejects negative",
        },
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "top_k": args.top_k if args.do_sample else None,
        "repetition_penalty": args.repetition_penalty,
        "add_generation_prompt": args.add_generation_prompt,
        "manual_qwen_chat_template": args.manual_qwen_chat_template,
    }

    for category_key, pairs in concept_groups.items():
        if args.max_pairs_per_category is not None:
            pairs = pairs[: args.max_pairs_per_category]

        for pair in pairs:
            print(f"Generating system/user conflict responses for {pair.slug}")
            examples = build_conflict_examples(
                pair=pair,
                statements=statements,
                tokenizer=tokenizer,
                variants=args.variants,
                add_generation_prompt=args.add_generation_prompt,
                manual_qwen_chat_template=args.manual_qwen_chat_template,
            )
            records = generate_conflict_responses(
                examples=examples,
                llm=llm,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
            pair_output_dir = output_root / Path(args.model).name / category_key / pair.slug
            from contrastive_hidden_states.generation import save_generation_bundle

            save_generation_bundle(
                output_dir=pair_output_dir,
                records=records,
                run_config=run_config,
            )

    llm = None
    cleanup_vllm_memory()


if __name__ == "__main__":
    main()
