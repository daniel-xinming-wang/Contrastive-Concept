from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm
from vllm import SamplingParams

from .prompts import PromptExample


def generate_responses(
    examples: list[PromptExample],
    llm: Any,
    forward_batch_size: int,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
) -> list[dict[str, Any]]:
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if do_sample else 0,
        repetition_penalty=repetition_penalty,
    )

    records: list[dict[str, Any]] = []
    for start_idx in tqdm(range(0, len(examples), forward_batch_size), desc="Generating responses"):
        batch_examples = examples[start_idx : start_idx + forward_batch_size]
        batch_prompts = [example.prompt for example in batch_examples]
        outputs = llm.generate(batch_prompts, sampling_params)

        for example, prompt, output in zip(batch_examples, batch_prompts, outputs):
            generated_text = output.outputs[0].text.strip()
            full_text = f"{prompt}{generated_text}".strip()
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

    variant_to_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        variant_to_records.setdefault(record["variant"], []).append(record)

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
