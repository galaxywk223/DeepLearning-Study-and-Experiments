from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset

from .config import ExperimentConfig
from .dataset_builder import ensure_dataset
from .utils import read_jsonl


@dataclass(slots=True)
class SFTSplitBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_records: list[dict[str, object]]
    raw_counts: dict[str, int]


def load_sft_splits(
    config: ExperimentConfig,
    tokenizer,
    *,
    overwrite_dataset: bool = False,
) -> SFTSplitBundle:
    ensure_dataset(config, overwrite=overwrite_dataset)
    records = read_jsonl(config.dataset_path)

    train_records = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "val"]
    test_records = [record for record in records if record["split"] == "test"]

    if config.max_train_samples > 0:
        train_records = train_records[: config.max_train_samples]
    if config.max_eval_samples > 0:
        val_records = val_records[: config.max_eval_samples]
        test_records = test_records[: config.max_eval_samples]

    processed_train = build_processed_records(train_records, tokenizer, config)
    processed_val = build_processed_records(val_records, tokenizer, config)
    if not processed_train:
        raise ValueError("No train examples were available after tokenization.")
    if not processed_val:
        raise ValueError("No validation examples were available after tokenization.")

    return SFTSplitBundle(
        train_dataset=Dataset.from_list(processed_train),
        val_dataset=Dataset.from_list(processed_val),
        test_records=test_records,
        raw_counts={
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
    )


def build_processed_records(
    records: list[dict[str, object]],
    tokenizer,
    config: ExperimentConfig,
) -> list[dict[str, list[int]]]:
    processed: list[dict[str, list[int]]] = []
    for record in records:
        tokenized = tokenize_record(record, tokenizer, config)
        if tokenized is not None:
            processed.append(tokenized)
    return processed


def compose_user_message(record: dict[str, object]) -> str:
    instruction = str(record["instruction"]).strip()
    model_input = str(record.get("input", "")).strip()
    if not model_input:
        return instruction
    return f"{instruction}\n\n补充要求：\n{model_input}"


def build_messages(
    record: dict[str, object],
    *,
    system_prompt: str,
    include_answer: bool,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": compose_user_message(record)},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": str(record["output"]).strip()})
    return messages


def render_chat(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    rendered = []
    for message in messages:
        role = message["role"].upper()
        rendered.append(f"{role}: {message['content'].strip()}")
    if add_generation_prompt:
        rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)


def tokenize_record(
    record: dict[str, object],
    tokenizer,
    config: ExperimentConfig,
) -> dict[str, list[int]] | None:
    prompt_messages = build_messages(
        record,
        system_prompt=config.system_prompt,
        include_answer=False,
    )
    full_messages = build_messages(
        record,
        system_prompt=config.system_prompt,
        include_answer=True,
    )

    prompt_text = render_chat(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    full_text = render_chat(
        tokenizer,
        full_messages,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and (not full_ids or full_ids[-1] != eos_token_id):
        full_ids.append(eos_token_id)

    labels = list(full_ids)
    prompt_token_count = min(len(prompt_ids), len(labels))
    for index in range(prompt_token_count):
        labels[index] = -100

    if len(full_ids) > config.max_seq_length:
        full_ids = full_ids[: config.max_seq_length]
        labels = labels[: config.max_seq_length]

    if all(label == -100 for label in labels):
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class SupervisedDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
