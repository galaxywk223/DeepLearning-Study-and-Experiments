from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .utils import clear_torch_cache


def resolve_compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _minor = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def resolve_device_map(device_map: str) -> str | dict[str, int | str] | None:
    normalized = device_map.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"cuda", "cuda:0"} and torch.cuda.is_available():
        return {"": 0}
    if normalized == "cpu":
        return {"": "cpu"}
    return None


def build_quantization_config(quantization_mode: str):
    normalized = quantization_mode.strip().lower()
    if normalized == "none":
        return None
    if normalized != "4bit":
        raise ValueError(f"Unsupported quantization mode: {quantization_mode}")

    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=resolve_compute_dtype(),
    )


def load_tokenizer(model_id_or_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_for_inference(
    *,
    model_id: str,
    adapter_dir: Path | None,
    quantization_mode: str,
    device_map: str,
) -> Any:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    resolved_device_map = resolve_device_map(device_map)
    quantization_config = build_quantization_config(quantization_mode)
    load_kwargs: dict[str, object] = {"trust_remote_code": False}
    if resolved_device_map is not None:
        load_kwargs["device_map"] = resolved_device_map
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["dtype"] = resolve_compute_dtype()

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except TypeError:
        if "dtype" not in load_kwargs:
            raise
        fallback_kwargs = dict(load_kwargs)
        fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_id, **fallback_kwargs)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(
            model,
            str(adapter_dir),
            autocast_adapter_dtype=False,
        )
    if quantization_config is None and resolved_device_map is None and torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return model


def unload_model(model: Any | None) -> None:
    if model is None:
        return
    del model
    clear_torch_cache()


def generate_answer(
    model,
    tokenizer,
    *,
    system_prompt: str,
    user_message: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    from .data import render_chat

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    prompt_text = render_chat(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    target_device = next(model.parameters()).device
    inputs = {name: tensor.to(target_device) for name, tensor in inputs.items()}

    generation_kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = generated[0][prompt_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer.strip()
