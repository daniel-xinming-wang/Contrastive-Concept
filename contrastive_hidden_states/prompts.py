from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .concepts import ContrastivePair


DEFAULT_BASE_TEMPLATE = (
    "What are your thoughts on the following statement?\n"
    "Statement: {statement}\n"
    "Response:"
)

CATEGORY_INSTRUCTIONS = {
    "linguistic_style": "Adopt the following linguistic style: {concept}.",
    "ideology": "Adopt the following ideological stance: {concept}.",
    "semantic_framing": "Adopt the following semantic framing: {concept}.",
}


@dataclass(frozen=True)
class PromptExample:
    category_name: str
    category_key: str
    pair_slug: str
    positive_concept: str
    negative_concept: str
    statement_index: int
    statement_source: str
    statement: str
    variant: str
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_neutral_statements(statement_paths: list[str | Path]) -> list[tuple[str, str]]:
    statements: list[tuple[str, str]] = []
    for statement_path in statement_paths:
        path = Path(statement_path)
        for line in path.read_text(encoding="utf-8").splitlines():
            statement = line.strip()
            if statement:
                statements.append((path.name, statement))
    return statements


def format_prompt(tokenizer: Any, prompt_text: str, add_generation_prompt: bool = False) -> str:
    if tokenizer is None:
        return prompt_text

    if hasattr(tokenizer, "apply_chat_template"):
        chat = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return formatted.strip()

    return prompt_text


def build_prompt_triplet(
    pair: ContrastivePair,
    statement: str,
    tokenizer: Any,
    add_generation_prompt: bool = False,
    base_template: str = DEFAULT_BASE_TEMPLATE,
) -> dict[str, str]:
    base_prompt = base_template.format(statement=statement)
    instruction_template = CATEGORY_INSTRUCTIONS.get(
        pair.category_key,
        "Adopt the following concept: {concept}.",
    )

    prompts = {
        "base": base_prompt,
        "positive": f"{instruction_template.format(concept=pair.positive)}\n\n{base_prompt}",
        "negative": f"{instruction_template.format(concept=pair.negative)}\n\n{base_prompt}",
    }

    return {
        variant: format_prompt(
            tokenizer=tokenizer,
            prompt_text=prompt,
            add_generation_prompt=add_generation_prompt,
        )
        for variant, prompt in prompts.items()
    }


def build_pair_examples(
    pair: ContrastivePair,
    statements: list[tuple[str, str]],
    tokenizer: Any,
    add_generation_prompt: bool = False,
    base_template: str = DEFAULT_BASE_TEMPLATE,
) -> list[PromptExample]:
    examples: list[PromptExample] = []
    for idx, (source_name, statement) in enumerate(statements):
        triplet = build_prompt_triplet(
            pair=pair,
            statement=statement,
            tokenizer=tokenizer,
            add_generation_prompt=add_generation_prompt,
            base_template=base_template,
        )
        for variant in ("negative", "base", "positive"):
            examples.append(
                PromptExample(
                    category_name=pair.category_name,
                    category_key=pair.category_key,
                    pair_slug=pair.slug,
                    positive_concept=pair.positive,
                    negative_concept=pair.negative,
                    statement_index=idx,
                    statement_source=source_name,
                    statement=statement,
                    variant=variant,
                    prompt=triplet[variant],
                )
            )
    return examples

