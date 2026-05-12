from __future__ import annotations

from typing import Any

from huggingface_hub.errors import GatedRepoError
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name_or_path: str,
    device_map: str = "auto",
    torch_dtype: str | None = None,
    trust_remote_code: bool = False,
) -> tuple[Any, Any, str]:
    resolved_name = model_name_or_path

    try:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
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
                "3. Use an open Hugging Face model id such as Qwen/Qwen2.5-3B-Instruct."
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
    return list(range(num_layers))
