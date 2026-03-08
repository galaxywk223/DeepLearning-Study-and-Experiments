from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import MNIST_MEAN, MNIST_STD


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

    images = denormalize(images).squeeze(1)

    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2.2, 3))
    if num_images == 1:
        axes = [axes]

    for index, axis in enumerate(axes):
        axis.imshow(images[index], cmap="gray")
        axis.set_title(
            f"True: {labels[index].item()}\nPred: {predictions[index].item()}",
            fontsize=9,
        )
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MNIST_MEAN).view(1, 1, 1, 1)
    std = torch.tensor(MNIST_STD).view(1, 1, 1, 1)
    return images * std + mean
