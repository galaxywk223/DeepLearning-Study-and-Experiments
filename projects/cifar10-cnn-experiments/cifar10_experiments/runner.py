from __future__ import annotations

from typing import Any

import torch
from torch import nn, optim

from .config import ExperimentConfig
from .data import build_cifar10_dataloaders
from .engine import evaluate, train_one_epoch
from .models import build_model
from .utils import ensure_dir, resolve_device, set_seed, write_json
from .visualize import save_prediction_grid


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    set_seed(config.seed)

    device = resolve_device(config.device)
    run_dir = ensure_dir(config.run_dir)
    pin_memory = device.type == "cuda"
    amp_enabled = config.use_amp and device.type == "cuda"

    train_loader, test_loader, loader_info = build_cifar10_dataloaders(
        config,
        pin_memory=pin_memory,
    )
    model = build_model(config.variant, dropout=config.dropout).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None

    write_json(config.config_path, config.to_dict())
    best_accuracy = 0.0
    best_epoch = 0
    history: list[dict[str, float | int]] = []

    print(
        f"Running {config.experiment_name} "
        f"({config.variant.upper()}) on {device.type}"
    )
    print(f"Artifacts: {run_dir}")
    print(
        "DataLoader workers: "
        f"{loader_info['effective_num_workers']} "
        f"(requested {loader_info['requested_num_workers']})"
    )
    if loader_info["requested_num_workers"] != loader_info["effective_num_workers"]:
        print("Windows worker cap applied to avoid slow startup and worker stalls.")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            amp_enabled=amp_enabled,
            scaler=scaler if amp_enabled else None,
        )
        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
            amp_enabled=amp_enabled,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "learning_rate": current_lr,
            }
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), config.checkpoint_path)

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_accuracy * 100:.2f}% | "
            f"lr={current_lr:.6f}"
        )

    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    final_loss, final_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
        amp_enabled=amp_enabled,
    )
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
        "variant": config.variant,
        "device": device.type,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "amp_enabled": amp_enabled,
        "dataloader": loader_info,
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
    if config.optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer_name: {config.optimizer_name}")


def build_scheduler(
    optimizer: optim.Optimizer,
    config: ExperimentConfig,
) -> optim.lr_scheduler._LRScheduler | None:
    if config.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
        )
    if config.scheduler == "multistep":
        milestones = [
            max(1, int(config.epochs * 0.5)),
            max(2, int(config.epochs * 0.75)),
        ]
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1,
        )
    if config.scheduler == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")
