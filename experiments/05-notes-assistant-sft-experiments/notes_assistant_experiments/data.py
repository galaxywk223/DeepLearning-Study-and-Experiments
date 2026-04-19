from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset

from .config import ExperimentConfig
from .dataset_builder import ensure_dataset
from .utils import read_jsonl


@dataclass(slots=True)
class SFTSplitBundle:
    # 训练和验证阶段使用 tokenized Dataset，测试阶段保留原始记录供对照评测和样例导出复用。
    train_dataset: Dataset
    val_dataset: Dataset
    test_records: list[dict[str, object]]
    raw_counts: dict[str, int]
    template_counts: dict[str, dict[str, int]]


def load_sft_splits(
    config: ExperimentConfig,
    tokenizer,
    *,
    overwrite_dataset: bool = False,
) -> SFTSplitBundle:
    # 训练前先确保本地 JSONL 数据存在，必要时会从课程笔记重新生成。
    ensure_dataset(config, overwrite=overwrite_dataset)
    records = read_jsonl(config.dataset_path)

    train_records = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "val"]
    test_records = [record for record in records if record["split"] == "test"]

    # 训练/验证集可按模板组裁剪，只保留当前实验希望学习的问题类型。
    selected_template_ids = set(config.selected_template_ids)
    train_records = [
        record
        for record in train_records
        if str(record["template_id"]) in selected_template_ids
    ]
    val_records = [
        record
        for record in val_records
        if str(record["template_id"]) in selected_template_ids
    ]

    if config.max_train_samples > 0:
        train_records = train_records[: config.max_train_samples]
    if config.max_eval_samples > 0:
        val_records = val_records[: config.max_eval_samples]
        test_records = test_records[: config.max_eval_samples]

    # 原始记录在这里被转换为 Trainer 期望的 input_ids / attention_mask / labels。
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
        template_counts={
            "train": count_template_ids(train_records),
            "val": count_template_ids(val_records),
            "test": count_template_ids(test_records),
        },
    )


def build_processed_records(
    records: list[dict[str, object]],
    tokenizer,
    config: ExperimentConfig,
) -> list[dict[str, list[int]]]:
    # tokenization 可能因为超长截断或标签全部被 mask 而丢弃单条记录。
    processed: list[dict[str, list[int]]] = []
    for record in records:
        tokenized = tokenize_record(record, tokenizer, config)
        if tokenized is not None:
            processed.append(tokenized)
    return processed


def count_template_ids(records: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        template_id = str(record.get("template_id", "unknown"))
        counts[template_id] = counts.get(template_id, 0) + 1
    return dict(sorted(counts.items()))


def compose_user_message(record: dict[str, object]) -> str:
    # instruction 是主问题，input 作为补充约束并显式拼接到 user 段中。
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
    # 消息列表保持标准 chat 结构，是否附带 assistant 取决于当前是在构造 prompt 还是完整样本。
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
    # 优先复用底座 tokenizer 自带的 chat template，保持训练和推理格式一致。
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # 没有 chat template 时退化为显式角色前缀，保证最小链路仍可运行。
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
    # prompt_messages 只保留 system/user，full_messages 额外包含 assistant 参考答案。
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
    # full_text 是真正参与 tokenization 的完整监督样本。
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

    # labels 初始时等于完整序列，随后把 prompt 部分改成 ignore_index。
    labels = list(full_ids)
    prompt_token_count = min(len(prompt_ids), len(labels))
    for index in range(prompt_token_count):
        # -100 是 CrossEntropyLoss 的 ignore_index，表示 system/user 段不参与损失。
        labels[index] = -100

    # 超长样本直接截断，保持 input_ids 和 labels 长度一致。
    if len(full_ids) > config.max_seq_length:
        full_ids = full_ids[: config.max_seq_length]
        labels = labels[: config.max_seq_length]

    # 如果 assistant 段被完全截掉，则该样本对监督训练没有价值，直接跳过。
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
        # 批内对齐时需要同时补齐 input_ids、attention_mask 和 labels。
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            # padding 位置同样不应参与损失，因此标签也补为 -100。
            labels.append(feature["labels"] + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
