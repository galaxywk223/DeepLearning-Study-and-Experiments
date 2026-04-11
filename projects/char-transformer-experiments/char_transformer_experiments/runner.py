from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn, optim

from .config import ExperimentConfig
from .data import CharDatasetBundle, decode_tokens, encode_text, prepare_dataset
from .engine import estimate_loss, train_one_epoch
from .models import build_model
from .utils import (
    count_parameters,
    ensure_dir,
    resolve_device,
    set_seed,
    write_json,
    write_text,
)
from .visualize import save_loss_curve_in_subprocess


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    set_seed(config.seed)

    device = resolve_device(config.device)
    run_dir = ensure_dir(config.run_dir)
    dataset = prepare_dataset(config)
    model = build_model(config.variant, vocab_size=dataset.vocab_size, config=config).to(device)

    amp_enabled = config.use_amp and device.type == "cuda"
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None
    total_training_steps = config.epochs * config.steps_per_epoch
    parameter_count = count_parameters(model)

    write_json(
        config.config_path,
        {
            **config.to_dict(),
            "vocab_size": dataset.vocab_size,
            "total_characters": dataset.total_characters,
            "parameter_count": parameter_count,
        },
    )

    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0
    history: list[dict[str, float | int]] = []

    print(
        f"Running {config.experiment_name} "
        f"({config.variant.upper()}) on {device.type}"
    )
    print(f"Artifacts: {run_dir}")
    print(
        f"Corpus: {dataset.source_path.name} | "
        f"chars={dataset.total_characters} | vocab={dataset.vocab_size}"
    )
    print(f"Trainable parameters: {parameter_count:,}")

    for epoch in range(1, config.epochs + 1):
        train_loss, global_step, current_lr = train_one_epoch(
            model,
            dataset.train_data,
            criterion,
            optimizer,
            config,
            device,
            amp_enabled=amp_enabled,
            scaler=scaler if amp_enabled else None,
            global_step=global_step,
            total_training_steps=total_training_steps,
        )
        loss_estimates = estimate_loss(
            model,
            dataset.train_data,
            dataset.val_data,
            criterion,
            config,
            device,
            amp_enabled=amp_enabled,
        )
        train_eval_loss = loss_estimates["train"]
        val_loss = loss_estimates["val"]
        val_perplexity = math.exp(min(val_loss, 20.0))

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_eval_loss": train_eval_loss,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
                "learning_rate": current_lr,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), config.checkpoint_path)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_eval={train_eval_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_ppl={val_perplexity:.2f} | "
            f"lr={current_lr:.6f}"
        )

    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    final_losses = estimate_loss(
        model,
        dataset.train_data,
        dataset.val_data,
        criterion,
        config,
        device,
        amp_enabled=amp_enabled,
    )
    sample_text = generate_sample_text(model, dataset, config, device)
    write_text(config.sample_path, sample_text)
    loss_curve_saved, loss_curve_message = save_loss_curve_in_subprocess(
        history,
        config.loss_curve_path,
        show_plot=config.show_plot,
    )

    metrics: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "variant": config.variant,
        "device": device.type,
        "parameter_count": parameter_count,
        "vocab_size": dataset.vocab_size,
        "total_characters": dataset.total_characters,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": math.exp(min(best_val_loss, 20.0)),
        "final_train_loss": final_losses["train"],
        "final_val_loss": final_losses["val"],
        "final_val_perplexity": math.exp(min(final_losses["val"], 20.0)),
        "amp_enabled": amp_enabled,
        "loss_curve_saved": loss_curve_saved,
        "loss_curve_message": loss_curve_message,
        "history": history,
        "artifacts": {
            "config": str(config.config_path),
            "best_model": str(config.checkpoint_path),
            "samples": str(config.sample_path),
            "loss_curve": str(config.loss_curve_path),
        },
    }
    write_json(config.metrics_path, metrics)

    if not loss_curve_saved:
        print("Loss curve export skipped:", loss_curve_message)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final val loss: {final_losses['val']:.4f}")
    return metrics


def build_optimizer(
    model: nn.Module,
    config: ExperimentConfig,
) -> optim.Optimizer:
    if config.variant == "bigram":
        return optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.variant == "transformer":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
    raise ValueError(f"Unsupported variant: {config.variant}")


def generate_sample_text(
    model: nn.Module,
    dataset: CharDatasetBundle,
    config: ExperimentConfig,
    device: torch.device,
) -> str:
    model.eval()
    prompt = sanitize_prompt(config.sample_prompt, dataset.stoi)
    prompt_tokens = encode_text(prompt, dataset.stoi).unsqueeze(0).to(device)
    generated_tokens = model.generate(
        prompt_tokens,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
    )
    generated_text = decode_tokens(generated_tokens[0].cpu(), dataset.itos)
    return (
        f"Prompt:\n{prompt}\n\n"
        f"Temperature: {config.temperature}\n"
        f"Generated:\n{generated_text}\n"
    )


def sanitize_prompt(prompt: str, stoi: dict[str, int]) -> str:
    filtered = "".join(character for character in prompt if character in stoi)
    if filtered:
        return filtered
    if "\n" in stoi:
        return "\n"
    return next(iter(stoi))
