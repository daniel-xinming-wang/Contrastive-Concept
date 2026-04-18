from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .prompts import PromptExample


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

    all_hidden_states: dict[int | str, list[torch.Tensor]] = {layer: [] for layer in hidden_layers}
    use_concat = list(hidden_layers) == ["concat"]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting hidden states"):
            input_ids, attention_mask = batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            out_hidden_states = outputs.hidden_states
            num_layers = len(model.model.layers)

            hidden_states_all_layers: list[torch.Tensor] = []
            for layer_idx, hidden_state in zip(
                range(-1, -num_layers, -1),
                reversed(out_hidden_states),
            ):
                if not use_concat and layer_idx not in all_hidden_states:
                    continue
                if use_concat:
                    hidden_states_all_layers.append(hidden_state[:, rep_token, :].detach().cpu())
                elif all_positions:
                    all_hidden_states[layer_idx].append(hidden_state.detach().cpu())
                else:
                    all_hidden_states[layer_idx].append(
                        hidden_state[:, rep_token, :].detach().cpu()
                    )

            if use_concat:
                all_hidden_states["concat"].append(torch.cat(hidden_states_all_layers, dim=1))

    return {
        layer_idx: torch.cat(layer_hidden_states, dim=0)
        for layer_idx, layer_hidden_states in all_hidden_states.items()
    }


def save_pair_bundle(
    output_dir: str | Path,
    examples: list[PromptExample],
    hidden_states: dict[int | str, torch.Tensor],
    run_config: dict[str, Any],
    save_format: str = "npy",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_to_indices = {"negative": [], "base": [], "positive": []}
    for idx, example in enumerate(examples):
        variant_to_indices[example.variant].append(idx)

    saved_files: dict[str, list[str]] = {}
    for variant, indices in variant_to_indices.items():
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        saved_files[variant] = []

        variant_hidden_states = {
            layer_idx: layer_hidden_states[indices]
            for layer_idx, layer_hidden_states in hidden_states.items()
        }

        if save_format == "pt":
            variant_path = variant_dir / f"{variant}_hidden_states.pt"
            torch.save(variant_hidden_states, variant_path)
            saved_files[variant].append(str(variant_path.name))
            continue

        if save_format != "npy":
            raise ValueError(f"Unsupported save format: {save_format}")

        for layer_idx, layer_hidden_states in variant_hidden_states.items():
            layer_path = variant_dir / f"layer_{layer_idx}.npy"
            np.save(layer_path, layer_hidden_states.numpy())
            saved_files[variant].append(str(layer_path.name))

    metadata_path = output_dir / "metadata.json"
    metadata_payload = {
        "run_config": run_config,
        "num_examples": len(examples),
        "save_format": save_format,
        "variant_counts": {variant: len(indices) for variant, indices in variant_to_indices.items()},
        "saved_files": saved_files,
        "examples": [asdict(example) for example in examples],
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
