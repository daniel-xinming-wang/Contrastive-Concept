from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .prompts import PromptExample


def generate_responses(
    examples: list[PromptExample],
    model: Any,
    tokenizer: Any,
    forward_batch_size: int,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[dict[str, Any]]:
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    records: list[dict[str, Any]] = []
    for start_idx in tqdm(range(0, len(examples), forward_batch_size), desc="Generating responses"):
        batch_examples = examples[start_idx : start_idx + forward_batch_size]
        batch_prompts = [example.prompt for example in batch_examples]
        encoded_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=encoded_inputs["input_ids"],
                attention_mask=encoded_inputs["attention_mask"],
                **generation_kwargs,
            )

        prompt_width = encoded_inputs["input_ids"].shape[1]
        for example, generated_ids in zip(batch_examples, output_ids):
            generated_token_ids = generated_ids[prompt_width:]
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
            full_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            records.append(
                {
                    **asdict(example),
                    "full_text": full_text,
                    "generated_text": generated_text,
                }
            )

    return records


def save_generation_bundle(
    output_dir: str | Path,
    records: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_to_records = {"negative": [], "base": [], "positive": []}
    for record in records:
        variant_to_records[record["variant"]].append(record)

    saved_files: dict[str, list[str]] = {}
    for variant, variant_records in variant_to_records.items():
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = variant_dir / "generations.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in variant_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        saved_files[variant] = [jsonl_path.name]

    metadata_payload = {
        "run_config": run_config,
        "num_examples": len(records),
        "variant_counts": {variant: len(variant_records) for variant, variant_records in variant_to_records.items()},
        "saved_files": saved_files,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")
