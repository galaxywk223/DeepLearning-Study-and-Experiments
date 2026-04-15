from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, object] | list[object]) -> None:
    ensure_parent_dir(path)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    ensure_parent_dir(path)
    path.write_text(content, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def write_csv(
    path: Path,
    rows: Iterable[dict[str, object]],
    fieldnames: list[str],
) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(text: str, max_length: int = 48) -> str:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text.strip().lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    if not normalized:
        normalized = "item"
    return normalized[:max_length].rstrip("-")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def char_level_f1(prediction: str, reference: str) -> float:
    normalized_prediction = normalize_text(prediction)
    normalized_reference = normalize_text(reference)
    if not normalized_prediction and not normalized_reference:
        return 1.0
    if not normalized_prediction or not normalized_reference:
        return 0.0

    prediction_counter = Counter(normalized_prediction)
    reference_counter = Counter(normalized_reference)
    overlap = sum((prediction_counter & reference_counter).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(normalized_prediction)
    recall = overlap / len(normalized_reference)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def summarize_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return trainable, total


def clear_torch_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
