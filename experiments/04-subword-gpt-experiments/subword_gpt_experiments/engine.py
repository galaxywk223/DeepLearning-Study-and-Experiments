from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .config import ExperimentConfig
from .data import sample_batch


def train_one_epoch(
    model: nn.Module,
    train_documents: list[torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    *,
    pad_token_id: int,
    device: torch.device,
    amp_enabled: bool,
    scaler: Any | None,
    global_step: int,
    total_training_steps: int,
) -> tuple[float, int, float]:
    model.train()
    total_loss = 0.0
    current_lr = optimizer.param_groups[0]["lr"]

    for _ in range(config.steps_per_epoch):
        current_lr = compute_learning_rate(
            config=config,
            step=global_step,
            total_training_steps=total_training_steps,
        )
        set_learning_rate(optimizer, current_lr)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(config.grad_accum_steps):
            inputs, targets, attention_mask = sample_batch(
                train_documents,
                batch_size=config.batch_size,
                block_size=config.block_size,
                min_sequence_length=config.min_sequence_length,
                pad_token_id=pad_token_id,
                device=device,
            )
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(inputs, attention_mask=attention_mask)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                step_loss += loss.item()
                loss = loss / config.grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler is not None:
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        total_loss += step_loss / config.grad_accum_steps
        global_step += 1

    return total_loss / config.steps_per_epoch, global_step, current_lr


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_documents: list[torch.Tensor],
    val_documents: list[torch.Tensor],
    criterion: nn.Module,
    config: ExperimentConfig,
    *,
    pad_token_id: int,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    results: dict[str, float] = {}

    for split_name, split_documents in (
        ("train", train_documents),
        ("val", val_documents),
    ):
        split_losses: list[float] = []
        for _ in range(config.eval_steps):
            inputs, targets, attention_mask = sample_batch(
                split_documents,
                batch_size=config.batch_size,
                block_size=config.block_size,
                min_sequence_length=config.min_sequence_length,
                pad_token_id=pad_token_id,
                device=device,
            )
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(inputs, attention_mask=attention_mask)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
            split_losses.append(loss.item())
        results[split_name] = sum(split_losses) / len(split_losses)

    return results


def compute_learning_rate(
    *,
    config: ExperimentConfig,
    step: int,
    total_training_steps: int,
) -> float:
    if config.warmup_steps <= 0:
        return cosine_decay(
            step=step,
            start_lr=config.learning_rate,
            end_lr=config.min_learning_rate,
            total_training_steps=total_training_steps,
        )

    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    post_warmup_steps = max(1, total_training_steps - config.warmup_steps)
    adjusted_step = min(step - config.warmup_steps, post_warmup_steps)
    return cosine_decay(
        step=adjusted_step,
        start_lr=config.learning_rate,
        end_lr=config.min_learning_rate,
        total_training_steps=post_warmup_steps,
    )


def cosine_decay(
    *,
    step: int,
    start_lr: float,
    end_lr: float,
    total_training_steps: int,
) -> float:
    if total_training_steps <= 0 or math.isclose(start_lr, end_lr):
        return start_lr

    progress = min(max(step, 0), total_training_steps) / total_training_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end_lr + cosine * (start_lr - end_lr)


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

