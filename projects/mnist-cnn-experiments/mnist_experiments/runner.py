from __future__ import annotations

from typing import Any

import torch
from torch import nn, optim

from .config import ExperimentConfig
from .data import build_mnist_dataloaders
from .engine import evaluate, train_one_epoch
from .models import build_model
from .utils import ensure_dir, resolve_device, set_seed, write_json
from .visualize import save_prediction_grid


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    set_seed(config.seed)

    device = resolve_device(config.device)
    run_dir = ensure_dir(config.run_dir)
    pin_memory = device.type == "cuda"

    train_loader, test_loader = build_mnist_dataloaders(
        config,
        pin_memory=pin_memory,
    )
    model = build_model(
        config.model_name,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    write_json(config.config_path, config.to_dict())
    best_accuracy = 0.0
    history: list[dict[str, float | int]] = []

    print(
        f"Running {config.experiment_name} "
        f"({config.model_name.upper()}) on {device.type}"
    )
    print(f"Artifacts: {run_dir}")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

        if scheduler is not None:
            scheduler.step(test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), config.checkpoint_path)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_accuracy * 100:.2f}%"
        )

    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    final_loss, final_accuracy = evaluate(model, test_loader, criterion, device)
    save_prediction_grid(
        model,
        test_loader,
        device,
        config.preview_path,
        num_images=config.preview_count,
        show_plot=config.show_plot,
    )

    metrics: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "device": device.type,
        "best_accuracy": best_accuracy,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "history": history,
        "artifacts": {
            "config": str(config.config_path),
            "best_model": str(config.checkpoint_path),
            "preview": str(config.preview_path),
        },
    }
    write_json(config.metrics_path, metrics)

    print(f"Best accuracy: {best_accuracy * 100:.2f}%")
    print(f"Final accuracy: {final_accuracy * 100:.2f}%")
    return metrics


def build_optimizer(
    model: nn.Module,
    config: ExperimentConfig,
) -> optim.Optimizer:
    if config.model_name == "mlp":
        return optim.SGD(model.parameters(), lr=config.learning_rate)
    if config.model_name == "cnn":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported model_name: {config.model_name}")


def build_scheduler(
    optimizer: optim.Optimizer,
    config: ExperimentConfig,
) -> optim.lr_scheduler.ReduceLROnPlateau | None:
    if config.model_name == "cnn":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=2,
        )
    return None
