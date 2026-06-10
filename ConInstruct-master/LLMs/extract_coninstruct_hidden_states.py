from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from huggingface_hub.errors import GatedRepoError
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

from gpt4 import (  # noqa: E402
    idx_2_conflict_type,
    inject_conflict_constraints,
    list_of_ints,
    read_input,
    str_2_bool,
)


def decoder_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        "Could not find decoder layers on model. Expected one of "
        "model.model.layers, model.language_model.layers, or model.transformer.h."
    )


def default_hidden_layers(model: Any) -> list[int]:
    if hasattr(model.config, "num_hidden_layers"):
        return list(range(model.config.num_hidden_layers))
    return list(range(len(decoder_layers(model))))


def get_hidden_states(
    prompts: list[str],
    model: Any,
    tokenizer: Any,
    hidden_layers: list[int],
    forward_batch_size: int,
    rep_token: int = -1,
    all_positions: bool = False,
) -> dict[int | str, torch.Tensor]:
    encoded_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)
    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"].half()

    dataset = TensorDataset(encoded_inputs["input_ids"], encoded_inputs["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=forward_batch_size)

    all_hidden_states: dict[int | str, list[torch.Tensor]] = {
        layer: [] for layer in hidden_layers
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting hidden states"):
            input_ids, attention_mask = batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            out_hidden_states = outputs.hidden_states
            num_layers = len(decoder_layers(model))

            for layer_idx, hidden_state in enumerate(out_hidden_states[1 : num_layers + 1]):
                if layer_idx not in all_hidden_states:
                    continue
                if all_positions:
                    all_hidden_states[layer_idx].append(hidden_state.detach().cpu())
                else:
                    all_hidden_states[layer_idx].append(
                        hidden_state[:, rep_token, :].detach().cpu()
                    )

    return {
        layer_idx: torch.cat(layer_hidden_states, dim=0)
        for layer_idx, layer_hidden_states in all_hidden_states.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract hidden states for the 900 ConInstruct single-conflict "
            "resolution prompts without generating model responses."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--input_filename", default="./datasets/conflict_instruction.jsonl")
    parser.add_argument(
        "--output-dir",
        default="./outputs/coninstruct_hidden_states",
        help="Root directory for saved hidden states.",
    )
    parser.add_argument(
        "--conflict_type_idx",
        default="1-2-3-4-5-6-7-8-9",
        help="Conflict type ids to extract, joined by '-'. Default extracts all 900 prompts.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shuffle_new_constraint", default="true", type=str_2_bool)
    parser.add_argument("--new_constraint_position", default="after")
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--add-generation-prompt", action="store_true", default=True)
    parser.add_argument(
        "--no-add-generation-prompt",
        action="store_false",
        dest="add_generation_prompt",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        dest="enable_thinking",
        help="Enable Qwen3.5 thinking mode in the chat template. Default is non-thinking mode.",
    )
    parser.add_argument(
        "--save-format",
        choices=("npy", "pt"),
        default="npy",
        help="How to save hidden states for each conflict type.",
    )
    parser.add_argument(
        "--all-positions",
        action="store_true",
        help="Save hidden states for all token positions instead of only the final token.",
    )
    parser.add_argument(
        "--rewrite-filename",
        default=None,
        help=(
            "JSONL file from rewrite_instructions_vllm.py. When provided, "
            "extract org/new hidden states into conflict_resolution_org_new."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("conflict", "org", "new"),
        default=["conflict"],
        help="Hidden-state variants to extract. Use org new with --rewrite-filename.",
    )
    args = parser.parse_args()
    args.conflict_type_idx_list = list_of_ints(args.conflict_type_idx)
    return args


def resolve_model_path(args: argparse.Namespace) -> str:
    return os.path.join(args.cache_dir, args.model) if args.cache_dir else args.model


def load_model_and_tokenizer(
    model_name_or_path: str,
    device_map: str,
    torch_dtype: str | None,
    trust_remote_code: bool,
) -> tuple[Any, Any, str]:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    except (OSError, ValueError, GatedRepoError) as exc:
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            raise exc
        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        legacy=False,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    return model.eval(), tokenizer, model_name_or_path


def format_resolution_prompt(
    tokenizer: Any,
    instruction: str,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def build_examples_for_conflict_type(
    input_filename: str,
    tokenizer: Any,
    conflict_type_idx: int,
    shuffle_new_constraint: bool,
    new_constraint_position: str,
    add_generation_prompt: bool,
    enable_thinking: bool,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for sample_id, instruction_dict in enumerate(read_input(input_filename), start=1):
        if max_samples is not None and len(examples) >= max_samples:
            break

        instruction, conflict_dict = inject_conflict_constraints(
            instruction_dict,
            [conflict_type_idx],
            idx_2_conflict_type,
            conflict_type_list=None,
            shuffle_new_constraint=shuffle_new_constraint,
            new_constraint_position=new_constraint_position,
        )
        if instruction is None:
            continue

        conflict_type = idx_2_conflict_type[conflict_type_idx]
        conflict_info = conflict_dict[conflict_type]
        prompt = format_resolution_prompt(
            tokenizer=tokenizer,
            instruction=instruction,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        examples.append(
            {
                "sample_id": sample_id,
                "conflict_type_idx": conflict_type_idx,
                "conflict_type": conflict_type,
                "instruction": instruction,
                "prompt": prompt,
                "org_constraint": conflict_info["org_constraint"],
                "new_constraint": conflict_info["new_constraint"],
            }
        )
    return examples


def model_output_name(model_name: str) -> str:
    return model_name.strip("/").replace("/", "__")


def save_hidden_state_bundle(
    output_dir: Path,
    hidden_states: dict[int | str, torch.Tensor],
    save_format: str,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    if save_format == "pt":
        output_path = output_dir / "hidden_states.pt"
        torch.save(hidden_states, output_path)
        return [output_path.name]

    for layer_idx, layer_hidden_states in hidden_states.items():
        output_path = output_dir / f"layer_{layer_idx}.npy"
        np.save(output_path, layer_hidden_states.to(torch.float32).numpy())
        saved_files.append(output_path.name)
    return saved_files


def load_rewrite_records(
    rewrite_filename: str,
    conflict_type_idx_list: list[int],
    max_samples: int | None,
) -> dict[int, list[dict[str, Any]]]:
    selected_types = set(conflict_type_idx_list)
    records_by_type: dict[int, list[dict[str, Any]]] = {
        conflict_type_idx: [] for conflict_type_idx in conflict_type_idx_list
    }
    with open(rewrite_filename, "r", encoding="utf-8") as fr:
        for line in fr:
            if not line.strip():
                continue
            record = json.loads(line)
            conflict_type_idx = record["conflict_type_idx"]
            if conflict_type_idx not in selected_types:
                continue
            if (
                max_samples is not None
                and len(records_by_type[conflict_type_idx]) >= max_samples
            ):
                continue
            records_by_type[conflict_type_idx].append(record)
    return records_by_type


def build_org_new_examples(
    records: list[dict[str, Any]],
    tokenizer: Any,
    variant: str,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        if variant == "org":
            instruction = record["original_instruction"]
        elif variant == "new":
            instruction = record["new_instruction"]
        else:
            raise ValueError(f"Unsupported org/new variant: {variant}")

        prompt = format_resolution_prompt(
            tokenizer=tokenizer,
            instruction=instruction,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        examples.append(
            {
                "sample_id": record["sample_id"],
                "conflict_type_idx": record["conflict_type_idx"],
                "conflict_type": record["conflict_type"],
                "variant": variant,
                "instruction": instruction,
                "prompt": prompt,
                "original_instruction": record["original_instruction"],
                "new_instruction": record["new_instruction"],
                "org_constraint": record["org_constraint"],
                "new_constraint": record["new_constraint"],
                "rewrite_model": record.get("rewrite_model"),
            }
        )
    return examples


def run_conflict_mode(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    resolved_name: str,
    hidden_layers: list[int],
) -> None:
    if any(variant != "conflict" for variant in args.variants):
        raise ValueError("Variants org/new require --rewrite-filename.")

    output_root = (
        Path(args.output_dir)
        / model_output_name(args.model)
        / "conflict_resolution"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "run_config": {
            "script": "extract_coninstruct_hidden_states.py",
            "mode": "conflict",
            "prompt_source": "ConInstruct conflict_resolution single-conflict prompts",
            "input_filename": str(Path(args.input_filename).resolve()),
            "model_arg": args.model,
            "resolved_model_name": resolved_name,
            "hidden_layers": hidden_layers,
            "batch_size": args.batch_size,
            "conflict_type_idx": args.conflict_type_idx,
            "shuffle_new_constraint": args.shuffle_new_constraint,
            "new_constraint_position": args.new_constraint_position,
            "add_generation_prompt": args.add_generation_prompt,
            "enable_thinking": args.enable_thinking,
            "save_format": args.save_format,
            "all_positions": args.all_positions,
            "variants": args.variants,
        },
        "num_examples": 0,
        "conflict_type_counts": {},
        "saved_files": {},
        "examples": [],
    }

    for conflict_type_idx in args.conflict_type_idx_list:
        conflict_type = idx_2_conflict_type[conflict_type_idx]
        print(f"Extracting conflict_type_{conflict_type_idx}: {conflict_type}")
        examples = build_examples_for_conflict_type(
            input_filename=args.input_filename,
            tokenizer=tokenizer,
            conflict_type_idx=conflict_type_idx,
            shuffle_new_constraint=args.shuffle_new_constraint,
            new_constraint_position=args.new_constraint_position,
            add_generation_prompt=args.add_generation_prompt,
            enable_thinking=args.enable_thinking,
            max_samples=args.max_samples,
        )
        prompts = [example["prompt"] for example in examples]
        hidden_states = get_hidden_states(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            hidden_layers=hidden_layers,
            forward_batch_size=args.batch_size,
            all_positions=args.all_positions,
        )

        conflict_dir_name = f"conflict_type_{conflict_type_idx}"
        saved_files = save_hidden_state_bundle(
            output_dir=output_root / conflict_dir_name,
            hidden_states=hidden_states,
            save_format=args.save_format,
        )

        metadata["num_examples"] += len(examples)
        metadata["conflict_type_counts"][conflict_dir_name] = len(examples)
        metadata["saved_files"][conflict_dir_name] = saved_files
        metadata["examples"].extend(examples)

    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def run_org_new_mode(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    resolved_name: str,
    hidden_layers: list[int],
) -> None:
    selected_variants = [variant for variant in args.variants if variant in ("org", "new")]
    if not selected_variants:
        raise ValueError("Use --variants org new with --rewrite-filename.")

    output_root = (
        Path(args.output_dir)
        / model_output_name(args.model)
        / "conflict_resolution_org_new"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    records_by_type = load_rewrite_records(
        rewrite_filename=args.rewrite_filename,
        conflict_type_idx_list=args.conflict_type_idx_list,
        max_samples=args.max_samples,
    )

    metadata: dict[str, Any] = {
        "run_config": {
            "script": "extract_coninstruct_hidden_states.py",
            "mode": "org_new",
            "prompt_source": "ConInstruct rewritten org/new single-conflict prompts",
            "rewrite_filename": str(Path(args.rewrite_filename).resolve()),
            "model_arg": args.model,
            "resolved_model_name": resolved_name,
            "hidden_layers": hidden_layers,
            "batch_size": args.batch_size,
            "conflict_type_idx": args.conflict_type_idx,
            "add_generation_prompt": args.add_generation_prompt,
            "enable_thinking": args.enable_thinking,
            "save_format": args.save_format,
            "all_positions": args.all_positions,
            "variants": selected_variants,
        },
        "num_examples": 0,
        "conflict_type_counts": {},
        "saved_files": {},
        "examples": [],
    }

    for conflict_type_idx in args.conflict_type_idx_list:
        conflict_type = idx_2_conflict_type[conflict_type_idx]
        conflict_dir_name = f"conflict_type_{conflict_type_idx}"
        records = records_by_type.get(conflict_type_idx, [])
        metadata["conflict_type_counts"][conflict_dir_name] = {}
        metadata["saved_files"][conflict_dir_name] = {}
        print(
            f"Extracting {conflict_dir_name}: {conflict_type} "
            f"from {len(records)} rewrite records"
        )

        for variant in selected_variants:
            examples = build_org_new_examples(
                records=records,
                tokenizer=tokenizer,
                variant=variant,
                add_generation_prompt=args.add_generation_prompt,
                enable_thinking=args.enable_thinking,
            )
            prompts = [example["prompt"] for example in examples]
            hidden_states = get_hidden_states(
                prompts=prompts,
                model=model,
                tokenizer=tokenizer,
                hidden_layers=hidden_layers,
                forward_batch_size=args.batch_size,
                all_positions=args.all_positions,
            )

            saved_files = save_hidden_state_bundle(
                output_dir=output_root / conflict_dir_name / variant,
                hidden_states=hidden_states,
                save_format=args.save_format,
            )
            metadata["num_examples"] += len(examples)
            metadata["conflict_type_counts"][conflict_dir_name][variant] = len(examples)
            metadata["saved_files"][conflict_dir_name][variant] = saved_files
            metadata["examples"].extend(examples)

    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args)

    model, tokenizer, resolved_name = load_model_and_tokenizer(
        model_name_or_path=model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    hidden_layers = default_hidden_layers(model)

    if args.rewrite_filename:
        run_org_new_mode(args, model, tokenizer, resolved_name, hidden_layers)
    else:
        run_conflict_mode(args, model, tokenizer, resolved_name, hidden_layers)


if __name__ == "__main__":
    main()
