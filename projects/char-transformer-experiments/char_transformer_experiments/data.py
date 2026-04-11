from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import torch

from .config import ExperimentConfig
from .utils import ensure_dir


@dataclass(slots=True)
class CharDatasetBundle:
    train_data: torch.Tensor
    val_data: torch.Tensor
    stoi: dict[str, int]
    itos: list[str]
    vocab_size: int
    total_characters: int
    source_path: Path


def prepare_dataset(config: ExperimentConfig) -> CharDatasetBundle:
    source_path = ensure_corpus(config)
    text = source_path.read_text(encoding="utf-8")
    if len(text) <= config.block_size + 1:
        raise ValueError("Corpus is too short for the configured block_size.")

    chars = sorted(set(text))
    stoi = {character: index for index, character in enumerate(chars)}
    encoded = torch.tensor([stoi[character] for character in text], dtype=torch.long)

    split_index = int(len(encoded) * 0.9)
    train_data = encoded[:split_index]
    val_data = encoded[split_index:]

    if len(train_data) <= config.block_size or len(val_data) <= config.block_size:
        raise ValueError(
            "Train/validation split is too short for the configured block_size."
        )

    return CharDatasetBundle(
        train_data=train_data,
        val_data=val_data,
        stoi=stoi,
        itos=chars,
        vocab_size=len(chars),
        total_characters=len(text),
        source_path=source_path,
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


def sample_batch(
    data: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_offset = len(data) - block_size - 1
    if max_offset <= 0:
        raise ValueError("Data tensor is too short for batch sampling.")

    offsets = torch.randint(0, max_offset + 1, (batch_size,))
    inputs = torch.stack([data[offset : offset + block_size] for offset in offsets])
    targets = torch.stack(
        [data[offset + 1 : offset + block_size + 1] for offset in offsets]
    )
    return inputs.to(device), targets.to(device)


def encode_text(text: str, stoi: dict[str, int]) -> torch.Tensor:
    missing = sorted({character for character in text if character not in stoi})
    if missing:
        joined = ", ".join(repr(character) for character in missing)
        raise ValueError(f"Text contains characters outside the vocabulary: {joined}")
    return torch.tensor([stoi[character] for character in text], dtype=torch.long)


def decode_tokens(tokens: torch.Tensor | list[int], itos: list[str]) -> str:
    if isinstance(tokens, torch.Tensor):
        token_ids = tokens.tolist()
    else:
        token_ids = tokens
    return "".join(itos[index] for index in token_ids)
