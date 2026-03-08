from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD


def save_prediction_grid(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    destination: Path,
    *,
    num_images: int,
    show_plot: bool,
) -> None:
    model.eval()
    images, labels = next(iter(loader))
    images = images[:num_images]
    labels = labels[:num_images]
    batch = images.to(device, non_blocking=True)

    with torch.no_grad():
        predictions = model(batch).argmax(dim=1).cpu()

    images = denormalize(images).permute(0, 2, 3, 1).clamp(0.0, 1.0)

    cols = min(num_images, 4)
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = flatten_axes(axes)

    for index in range(num_images):
        axes[index].imshow(images[index])
        axes[index].set_title(
            f"T: {CIFAR10_CLASSES[labels[index].item()]}\n"
            f"P: {CIFAR10_CLASSES[predictions[index].item()]}",
            fontsize=9,
        )
        axes[index].axis("off")

    for index in range(num_images, len(axes)):
        axes[index].axis("off")

    fig.tight_layout()
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def flatten_axes(axes):
    if isinstance(axes, plt.Axes):
        return [axes]
    if hasattr(axes, "ravel"):
        return list(axes.ravel())
    return list(axes)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return images * std + mean
