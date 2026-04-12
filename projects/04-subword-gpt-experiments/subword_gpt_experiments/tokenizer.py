from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

BYTE_VOCAB_SIZE = 256


def count_pairs(sequence: list[int]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for left, right in zip(sequence, sequence[1:]):
        counts[(left, right)] += 1
    return counts


def replace_pair(
    sequence: list[int],
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(sequence):
        if (
            index < len(sequence) - 1
            and sequence[index] == pair[0]
            and sequence[index + 1] == pair[1]
        ):
            merged.append(new_id)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged


@dataclass(slots=True)
class BytePairTokenizer:
    token_bytes: list[bytes]
    merges: list[tuple[int, int]]
    special_tokens: list[str]
    merge_to_id: dict[tuple[int, int], int] = field(init=False)
    special_token_to_id: dict[str, int] = field(init=False)
    id_to_special_token: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.merge_to_id = {
            pair: BYTE_VOCAB_SIZE + index for index, pair in enumerate(self.merges)
        }
        self.special_token_to_id = {
            token: len(self.token_bytes) + index
            for index, token in enumerate(self.special_tokens)
        }
        self.id_to_special_token = {
            token_id: token for token, token_id in self.special_token_to_id.items()
        }

    @property
    def vocab_size(self) -> int:
        return len(self.token_bytes) + len(self.special_tokens)

    @property
    def learned_merges(self) -> int:
        return len(self.merges)

    @property
    def pad_token_id(self) -> int:
        return self.special_token_to_id[self.special_tokens[0]]

    @property
    def bos_token_id(self) -> int:
        return self.special_token_to_id[self.special_tokens[1]]

    @property
    def eos_token_id(self) -> int:
        return self.special_token_to_id[self.special_tokens[2]]

    @classmethod
    def train_from_text(
        cls,
        text: str,
        *,
        vocab_size: int,
        min_pair_frequency: int,
        special_tokens: list[str],
    ) -> "BytePairTokenizer":
        target_base_vocab_size = max(BYTE_VOCAB_SIZE, vocab_size - len(special_tokens))
        token_bytes = [bytes([index]) for index in range(BYTE_VOCAB_SIZE)]
        merges: list[tuple[int, int]] = []
        sequence = list(text.encode("utf-8"))

        if len(sequence) < 2:
            return cls(token_bytes=token_bytes, merges=merges, special_tokens=special_tokens)

        while len(token_bytes) < target_base_vocab_size:
            pair_counts = count_pairs(sequence)
            if not pair_counts:
                break

            best_pair, best_frequency = max(
                pair_counts.items(),
                key=lambda item: (item[1], item[0]),
            )
            if best_frequency < min_pair_frequency:
                break

            new_id = len(token_bytes)
            token_bytes.append(token_bytes[best_pair[0]] + token_bytes[best_pair[1]])
            merges.append(best_pair)
            sequence = replace_pair(sequence, best_pair, new_id)

        return cls(token_bytes=token_bytes, merges=merges, special_tokens=special_tokens)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        sequence = list(text.encode("utf-8"))
        for new_id, pair in enumerate(self.merges, start=BYTE_VOCAB_SIZE):
            sequence = replace_pair(sequence, pair, new_id)

        if add_bos:
            sequence.insert(0, self.bos_token_id)
        if add_eos:
            sequence.append(self.eos_token_id)
        return sequence

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        pieces: list[bytes] = []
        text_parts: list[str] = []

        for token_id in token_ids:
            if token_id in self.id_to_special_token:
                if skip_special_tokens:
                    continue
                if pieces:
                    text_parts.append(b"".join(pieces).decode("utf-8", errors="replace"))
                    pieces = []
                text_parts.append(self.id_to_special_token[token_id])
                continue
            pieces.append(self.token_bytes[token_id])

        if pieces:
            text_parts.append(b"".join(pieces).decode("utf-8", errors="replace"))
        return "".join(text_parts)

    def save(self, path: Path) -> None:
        payload = {
            "token_bytes": [piece.hex() for piece in self.token_bytes],
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BytePairTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        token_bytes = [bytes.fromhex(piece) for piece in payload["token_bytes"]]
        merges = [tuple(pair) for pair in payload["merges"]]
        special_tokens = [str(token) for token in payload["special_tokens"]]
        return cls(
            token_bytes=token_bytes,
            merges=merges,
            special_tokens=special_tokens,
        )
