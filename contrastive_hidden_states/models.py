from __future__ import annotations

from typing import Any

import torch
from huggingface_hub.errors import GatedRepoError
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ALIASES = {
    "llama_3_8b_it": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama_3.3_70b_4bit_it": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    "llama_3.1_70b_4bit_it": "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
    "llama_3.3_70b_it": "meta-llama/Llama-3.3-70B-Instruct",
    "gemma_2_9b_it": "google/gemma-2-9b-it",
    "qwq_32b": "Qwen/QwQ-32B",
    "qwen2_0_5b_it": "Qwen/Qwen2-0.5B-Instruct",
    "qwen2_5_3b_it": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2_5_7b_it": "Qwen/Qwen2.5-7B-Instruct",
}

DTYPE_ALIASES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_model_name(model_name_or_path: str) -> str:
    return MODEL_ALIASES.get(model_name_or_path, model_name_or_path)


def load_model_and_tokenizer(
    model_name_or_path: str,
    device_map: str = "auto",
    torch_dtype: str | None = None,
    trust_remote_code: bool = False,
) -> tuple[Any, Any, str]:
    resolved_name = resolve_model_name(model_name_or_path)
    dtype = DTYPE_ALIASES.get(torch_dtype.lower()) if torch_dtype else None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except (OSError, GatedRepoError) as exc:
        message = str(exc)
        if "gated repo" in message.lower() or isinstance(exc, GatedRepoError):
            raise RuntimeError(
                "Model access failed because the requested Hugging Face repo is gated.\n"
                f"Requested model: {resolved_name}\n\n"
                "Options:\n"
                "1. Authenticate with Hugging Face and request access to the gated model.\n"
                "2. Pass a local model path with --model /path/to/model.\n"
                "3. Use an open model alias such as qwen2_5_3b_it or qwen2_5_7b_it."
            ) from exc
        raise

    use_fast = "LlamaForCausalLM" not in getattr(model.config, "architectures", [])
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_name,
        use_fast=use_fast,
        padding_side="left",
        legacy=False,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    return model.eval(), tokenizer, resolved_name


def default_hidden_layers(model: Any) -> list[int]:
    num_layers = model.config.num_hidden_layers
    return list(range(-1, -num_layers, -1))
