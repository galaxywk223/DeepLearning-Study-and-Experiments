from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import torch

from .config import ExperimentConfig
from .tokenizer import BytePairTokenizer
from .utils import ensure_dir


@dataclass(slots=True)
class TokenizedDatasetBundle:
    train_documents: list[torch.Tensor]
    val_documents: list[torch.Tensor]
    tokenizer: BytePairTokenizer
    source_path: Path
    tokenizer_path: Path
    document_count: int
    train_token_count: int
    val_token_count: int
    observed_token_ids: list[int]


def prepare_dataset(config: ExperimentConfig) -> TokenizedDatasetBundle:
    source_path = ensure_corpus(config)
    text = source_path.read_text(encoding="utf-8")
    tokenizer = prepare_tokenizer(text, config)
    documents = split_documents(text)

    encoded_documents = [
        torch.tensor(
            tokenizer.encode(document, add_bos=True, add_eos=True),
            dtype=torch.long,
        )
        for document in documents
        if document.strip()
    ]
    encoded_documents = [document for document in encoded_documents if len(document) >= 3]
    if not encoded_documents:
        raise ValueError("No usable documents were found after tokenization.")

    train_documents, val_documents = split_train_val_documents(
        encoded_documents,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    train_token_count = sum(len(document) for document in train_documents)
    val_token_count = sum(len(document) for document in val_documents)
    observed_token_ids = sorted(
        {
            int(token_id)
            for document in encoded_documents
            for token_id in document.tolist()
        }
    )
    return TokenizedDatasetBundle(
        train_documents=train_documents,
        val_documents=val_documents,
        tokenizer=tokenizer,
        source_path=source_path,
        tokenizer_path=config.tokenizer_path,
        document_count=len(encoded_documents),
        train_token_count=train_token_count,
        val_token_count=val_token_count,
        observed_token_ids=observed_token_ids,
    )


def ensure_corpus(config: ExperimentConfig) -> Path:
    ensure_dir(config.data_dir)
    destination = config.corpus_path
    if destination.exists():
        return destination

    with urlopen(config.source_url, timeout=30) as response:
        text = response.read().decode("utf-8")
    destination.write_text(text, encoding="utf-8")
    return destination


def prepare_tokenizer(text: str, config: ExperimentConfig) -> BytePairTokenizer:
    ensure_dir(config.data_dir)
    if config.tokenizer_path.exists():
        return BytePairTokenizer.load(config.tokenizer_path)

    tokenizer = BytePairTokenizer.train_from_text(
        text,
        vocab_size=config.tokenizer_vocab_size,
        min_pair_frequency=config.min_pair_frequency,
        special_tokens=[config.pad_token, config.bos_token, config.eos_token],
    )
    tokenizer.save(config.tokenizer_path)
    return tokenizer


def split_documents(text: str) -> list[str]:
    parts = re.split(r"(\n\s*\n)", text)
    documents: list[str] = []
    for index in range(0, len(parts), 2):
        content = parts[index]
        delimiter = parts[index + 1] if index + 1 < len(parts) else ""
        document = f"{content}{delimiter}"
        if document.strip():
            documents.append(document)

    if documents:
        return documents
    return [text]


def split_train_val_documents(
    documents: list[torch.Tensor],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if len(documents) == 1:
        single_document = documents[0]
        split_index = int(len(single_document) * (1.0 - val_ratio))
        split_index = min(max(split_index, 2), len(single_document) - 1)
        train_document = single_document[:split_index]
        val_document = single_document[split_index - 1 :]
        return [train_document], [val_document]

    indices = list(range(len(documents)))
    random.Random(seed).shuffle(indices)
    shuffled = [documents[index] for index in indices]
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_documents = shuffled[:val_count]
    train_documents = shuffled[val_count:]
    if not train_documents:
        train_documents, val_documents = shuffled[:-1], shuffled[-1:]
    return train_documents, val_documents


def sample_batch(
    documents: list[torch.Tensor],
    *,
    batch_size: int,
    block_size: int,
    min_sequence_length: int,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.full((batch_size, block_size), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, block_size), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, block_size), dtype=torch.bool)

    for row in range(batch_size):
        document = documents[random.randrange(len(documents))]
        max_window_length = min(block_size + 1, len(document))
        min_window_length = min(max_window_length, max(2, min_sequence_length + 1))
        if max_window_length > min_window_length:
            window_length = random.randint(min_window_length, max_window_length)
        else:
            window_length = max_window_length

        start_max = len(document) - window_length
        start = random.randint(0, start_max) if start_max > 0 else 0
        window = document[start : start + window_length]
        source = window[:-1]
        target = window[1:]
        valid_length = len(source)

        inputs[row, :valid_length] = source
        targets[row, :valid_length] = target
        attention_mask[row, :valid_length] = True

    return inputs.to(device), targets.to(device), attention_mask.to(device)
